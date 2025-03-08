import click
import librosa
import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from rift_svc import DiT, RF
from rift_svc.feature_extractors import HubertModelWithFinalProj, RMSExtractor, get_mel_spectrogram
from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rmvpe import RMVPE
from rift_svc.utils import linear_interpolate_tensor, post_process_f0, f0_ensemble, f0_ensemble_light, get_f0_pw, get_f0_pm
from slicer import Slicer


torch.set_grad_enabled(False)


def extract_state_dict(ckpt):
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '')
            new_state_dict[new_k] = v
    spk2idx = ckpt['hyper_parameters']['cfg']['spk2idx']
    model_cfg = ckpt['hyper_parameters']['cfg']['model']
    dataset_cfg = ckpt['hyper_parameters']['cfg']['dataset']
    return new_state_dict, spk2idx, model_cfg, dataset_cfg


def load_models(model_path, device):
    """Load all required models and return them"""
    click.echo("Loading models...")
    
    # Load the conversion model
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict, spk2idx, dit_cfg, dataset_cfg = extract_state_dict(ckpt)

    transformer = DiT(num_speaker=len(spk2idx), **dit_cfg)
    svc_model = RF(transformer=transformer)
    svc_model.load_state_dict(state_dict)
    svc_model = svc_model.to(device)
    svc_model.eval()
    
    # Load additional models
    vocoder = NsfHifiGAN('pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt').to(device)
    rmvpe = RMVPE(model_path="pretrained/rmvpe/model.pt", hop_length=160, device=device)
    hubert = HubertModelWithFinalProj.from_pretrained("pretrained/content-vec-best").to(device)
    rms_extractor = RMSExtractor().to(device)
    
    return svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg


def load_audio(file_path, target_sr):
    """Load and preprocess audio file"""
    click.echo("Loading audio...")
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    if len(audio.shape) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return audio.numpy().squeeze()


def apply_fade(audio, fade_samples, fade_in=True):
    """Apply fade in/out using half of a Hanning window"""
    fade_window = np.hanning(fade_samples * 2)
    if fade_in:
        fade_curve = fade_window[:fade_samples]
    else:
        fade_curve = fade_window[fade_samples:]
    audio[:fade_samples] *= fade_curve
    return audio


def extract_features(audio_segment, sample_rate, hop_length, rmvpe, hubert, rms_extractor, 
                     device, key_shift=0, ds_cfg_strength=0.0, cvec_downsample_rate=2, target_loudness=-18.0,
                     robust_f0=0):
    """Extract all required features from an audio segment"""
    # Normalize input segment
    meter = pyln.Meter(sample_rate, block_size=0.1)
    original_loudness = meter.integrated_loudness(audio_segment)
    normalized_audio = pyln.normalize.loudness(audio_segment, original_loudness, target_loudness)

    # Handle potential clipping
    max_amp = np.max(np.abs(normalized_audio))
    if max_amp > 1.0:
        normalized_audio = normalized_audio * (0.99 / max_amp)

    audio_tensor = torch.from_numpy(normalized_audio).float().unsqueeze(0).to(device)
    audio_16khz = torch.from_numpy(librosa.resample(normalized_audio, orig_sr=sample_rate, target_sr=16000)).float().unsqueeze(0).to(device)

    # Extract mel spectrogram
    mel = get_mel_spectrogram(
        audio_tensor,
        sampling_rate=sample_rate,
        n_fft=2048,
        num_mels=128,
        hop_size=512,
        win_size=2048,
        fmin=40,
        fmax=16000
    ).transpose(1, 2)

    # Extract content vector
    cvec = hubert(audio_16khz)["last_hidden_state"].squeeze(0)
    cvec = linear_interpolate_tensor(cvec, mel.shape[1])[None, :]

    # Create bad_cvec (downsampled) for classifier-free guidance
    if ds_cfg_strength > 0:
        cvec_ds = cvec.clone()
        # Downsample and then interpolate back, similar to dataset.py
        cvec_ds = cvec_ds[0, ::2, :]  # Take every other frame
        cvec_ds = linear_interpolate_tensor(cvec_ds, cvec_ds.shape[0]//cvec_downsample_rate)
        cvec_ds = linear_interpolate_tensor(cvec_ds, mel.shape[1])[None, :]
    else:
        cvec_ds = None

    # Extract f0
    if robust_f0 > 0:
        # Parameters for F0 extraction
        time_step = hop_length / sample_rate
        f0_min = 40
        f0_max = 1100
        
        # Extract F0 using multiple methods
        rmvpe_f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
        rmvpe_f0 = post_process_f0(rmvpe_f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
        pw_f0 = get_f0_pw(normalized_audio, sample_rate, time_step, f0_min, f0_max)
        pmac_f0 = get_f0_pm(normalized_audio, sample_rate, time_step, f0_min, f0_max)
        
        if robust_f0 == 1:
            # Level 1: Light ensemble that preserves expressiveness
            rms_np = rms_extractor(audio_tensor).squeeze().cpu().numpy()
            f0 = f0_ensemble_light(rmvpe_f0, pw_f0, pmac_f0, rms=rms_np)
        else:
            # Level 2: Strong ensemble with more filtering
            f0 = f0_ensemble(rmvpe_f0, pw_f0, pmac_f0)
    else:
        # Level 0: Use only RMVPE for F0 extraction (original method)
        f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
        f0 = post_process_f0(f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
    
    if key_shift != 0:
        f0 = f0 * 2 ** (key_shift / 12)
    f0 = torch.from_numpy(f0).float().to(device)[None, :]
    
    # Extract RMS
    rms = rms_extractor(audio_tensor)
    
    return mel, cvec, cvec_ds, f0, rms, original_loudness


def run_inference(
    model, mel, cvec, f0, rms, cvec_ds, spk_id, 
    infer_steps, ds_cfg_strength, spk_cfg_strength, 
    skip_cfg_strength, cfg_skip_layers, cfg_rescale,
    sliced_inference=False
):
    """Run the actual inference through the model"""
    if sliced_inference:
        # Use sliced inference for long segments
        sliced_len = 256
        mel_crossfade_len = 8  # Number of frames to crossfade in mel domain
        
        # If the segment is shorter than one slice, just process it directly
        if mel.shape[1] <= sliced_len:
            mel_out, _ = model.sample(
                src_mel=mel,
                spk_id=spk_id,
                f0=f0,
                rms=rms,
                cvec=cvec,
                steps=infer_steps,
                bad_cvec=cvec_ds,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                skip_cfg_strength=skip_cfg_strength,
                cfg_skip_layers=cfg_skip_layers,
                cfg_rescale=cfg_rescale,
            )
            return mel_out
        
        # Create a tensor to hold the full output with crossfading
        full_mel_out = torch.zeros_like(mel)
        
        # Process each slice
        for i in range(0, mel.shape[1], sliced_len - mel_crossfade_len):
            # Determine slice boundaries
            start_idx = i
            end_idx = min(i + sliced_len, mel.shape[1])
            
            # Skip if we're at the end
            if start_idx >= mel.shape[1]:
                break
            
            # Extract slices for this window
            mel_slice = mel[:, start_idx:end_idx, :]
            cvec_slice = cvec[:, start_idx:end_idx, :]
            f0_slice = f0[:, start_idx:end_idx]
            rms_slice = rms[:, start_idx:end_idx]
            
            # Slice the bad_cvec if it exists
            cvec_ds_slice = None
            if cvec_ds is not None:
                cvec_ds_slice = cvec_ds[:, start_idx:end_idx, :]
            
            # Process with model
            mel_out_slice, _ = model.sample(
                src_mel=mel_slice,
                spk_id=spk_id,
                f0=f0_slice,
                rms=rms_slice,
                cvec=cvec_slice,
                steps=infer_steps,
                bad_cvec=cvec_ds_slice,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                skip_cfg_strength=skip_cfg_strength,
                cfg_skip_layers=cfg_skip_layers,
                cfg_rescale=cfg_rescale,
            )
            
            # Create crossfade weights
            slice_len = end_idx - start_idx
            
            # Apply different strategies depending on position
            if i == 0:  # First slice
                # No crossfade at the beginning
                weights = torch.ones((1, slice_len, 1), device=mel.device)
                if i + sliced_len < mel.shape[1]:  # If not the last slice too
                    # Fade out at the end - use the minimum of slice_len and mel_crossfade_len
                    actual_crossfade_len = min(mel_crossfade_len, slice_len)
                    if actual_crossfade_len > 0:  # Only apply if we have space
                        fade_out = torch.linspace(1, 0, actual_crossfade_len, device=mel.device)
                        weights[:, -actual_crossfade_len:, :] = fade_out.view(1, -1, 1)
            elif end_idx >= mel.shape[1]:  # Last slice
                # Fade in at the beginning - use the minimum of slice_len and mel_crossfade_len
                weights = torch.ones((1, slice_len, 1), device=mel.device)
                actual_crossfade_len = min(mel_crossfade_len, slice_len)
                if actual_crossfade_len > 0:  # Only apply if we have space
                    fade_in = torch.linspace(0, 1, actual_crossfade_len, device=mel.device)
                    weights[:, :actual_crossfade_len, :] = fade_in.view(1, -1, 1)
            else:  # Middle slices
                # Crossfade both sides, handling the case where slice_len < 2*mel_crossfade_len
                weights = torch.ones((1, slice_len, 1), device=mel.device)
                
                # Determine the actual crossfade length (might be shorter for small slices)
                actual_crossfade_len = min(mel_crossfade_len, slice_len // 2)
                if actual_crossfade_len > 0:
                    fade_in = torch.linspace(0, 1, actual_crossfade_len, device=mel.device)
                    fade_out = torch.linspace(1, 0, actual_crossfade_len, device=mel.device)
                    weights[:, :actual_crossfade_len, :] = fade_in.view(1, -1, 1)
                    weights[:, -actual_crossfade_len:, :] = fade_out.view(1, -1, 1)
            
            # Apply weights to current slice output
            mel_out_slice = mel_out_slice * weights
            
            # Add to the appropriate region of the output
            full_mel_out[:, start_idx:end_idx, :] += mel_out_slice
            
        # Return the full crossfaded output
        mel_out = full_mel_out
    else:
        # Process the entire segment at once
        mel_out, _ = model.sample(
            src_mel=mel,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            steps=infer_steps,
            bad_cvec=cvec_ds,
            ds_cfg_strength=ds_cfg_strength,
            spk_cfg_strength=spk_cfg_strength,
            skip_cfg_strength=skip_cfg_strength,
            cfg_skip_layers=cfg_skip_layers,
            cfg_rescale=cfg_rescale,
        )
    
    return mel_out


def generate_audio(vocoder, mel_out, f0, original_loudness=None, restore_loudness=True):
    """Generate audio from mel spectrogram using vocoder"""
    audio_out = vocoder(mel_out.transpose(1, 2), f0)
    audio_out = audio_out.squeeze().cpu().numpy()

    if restore_loudness and original_loudness is not None:
        # Restore original loudness
        meter = pyln.Meter(44100, block_size=0.1)  # Using default sample rate for vocoder
        audio_out_loudness = meter.integrated_loudness(audio_out)
        audio_out = pyln.normalize.loudness(audio_out, audio_out_loudness, original_loudness)

        # Handle clipping
        max_amp = np.max(np.abs(audio_out))
        if max_amp > 1.0:
            audio_out = audio_out * (0.99 / max_amp)
            
    return audio_out


def process_segment(
    audio_segment, 
    svc_model, vocoder, rmvpe, hubert, rms_extractor, 
    speaker_id, sample_rate, hop_length, device,
    key_shift=0, 
    infer_steps=32,
    ds_cfg_strength=0.0, 
    spk_cfg_strength=0.0, 
    skip_cfg_strength=0.0, 
    cfg_skip_layers=None, 
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    restore_loudness=True,
    sliced_inference=False,
    robust_f0=0
):
    """Process a single audio segment and return the converted audio"""
    # Extract features
    mel, cvec, cvec_ds, f0, rms, original_loudness = extract_features(
        audio_segment, sample_rate, hop_length, rmvpe, hubert, rms_extractor, 
        device, key_shift, ds_cfg_strength, cvec_downsample_rate, target_loudness,
        robust_f0
    )
    
    # Prepare speaker ID
    spk_id = torch.LongTensor([speaker_id]).to(device)
    
    # Run inference
    mel_out = run_inference(
        svc_model, mel, cvec, f0, rms, cvec_ds, spk_id,
        infer_steps, ds_cfg_strength, spk_cfg_strength,
        skip_cfg_strength, cfg_skip_layers, cfg_rescale,
        sliced_inference
    )
    
    # Generate audio
    audio_out = generate_audio(
        vocoder, mel_out, f0, 
        original_loudness if restore_loudness else None, 
        restore_loudness
    )
    
    return audio_out


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--input', type=click.Path(exists=True), required=True, help='Input audio file')
@click.option('--output', type=click.Path(), required=True, help='Output audio file')
@click.option('--speaker', type=str, required=True, help='Target speaker')
@click.option('--key-shift', type=int, default=0, help='Pitch shift in semitones')
@click.option('--device', type=str, default=None, help='Device to use (cuda/cpu)')
@click.option('--infer-steps', type=int, default=32, help='Number of inference steps')
@click.option('--ds-cfg-strength', type=float, default=0.0, help='Downsampled content vector guidance strength')
@click.option('--spk-cfg-strength', type=float, default=0.0, help='Speaker guidance strength')
@click.option('--skip-cfg-strength', type=float, default=0.0, help='Skip layer guidance strength')
@click.option('--cfg-skip-layers', type=int, default=None, help='Layer to skip for classifier-free guidance')
@click.option('--cfg-rescale', type=float, default=0.7, help='Classifier-free guidance rescale factor')
@click.option('--cvec-downsample-rate', type=int, default=2, help='Downsampling rate for bad_cvec creation')
@click.option('--target-loudness', type=float, default=-18.0, help='Target loudness in LUFS for normalization')
@click.option('--restore-loudness', default=True, help='Restore loudness to original')
@click.option('--fade-duration', type=float, default=20.0, help='Fade duration in milliseconds')
@click.option('--sliced-inference', is_flag=True, default=False, help='Use sliced inference for processing long segments')
@click.option('--robust-f0', type=int, default=0, help='Level of robust f0 filtering (0=none, 1=light, 2=aggressive)')
def main(
    model,
    input,
    output,
    speaker,
    key_shift,
    device,
    infer_steps,
    ds_cfg_strength,
    spk_cfg_strength,
    skip_cfg_strength,
    cfg_skip_layers,
    cfg_rescale,
    cvec_downsample_rate,
    target_loudness,
    restore_loudness,
    fade_duration,
    sliced_inference,
    robust_f0
):
    """Convert the voice in an audio file to a target speaker."""

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load models
    svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg = load_models(model, device)

    try:
        speaker_id = spk2idx[speaker]
    except KeyError:
        raise ValueError(f"Speaker {speaker} not found in the model's speaker list, valid speakers are {spk2idx.keys()}")
    
    # Get config from loaded model
    hop_length = 512
    sample_rate = 44100

    # Load audio
    audio = load_audio(input, sample_rate)

    # Initialize Slicer
    slicer = Slicer(
        sr=sample_rate,
        threshold=-35.0,
        min_length=3000,
        min_interval=100,
        hop_size=10,
        max_sil_kept=300,
        look_ahead_frames=4,
        min_slice_length=2000
    )

    # Step (1): Use slicer to segment the input audio and get positions
    click.echo("Slicing audio...")
    segments_with_pos = slicer.slice(audio)  # Now returns list of (start_pos, chunk)

    if restore_loudness:
        click.echo(f"Will restore loudness to original")

    # Calculate fade size in samples
    fade_samples = int(fade_duration * sample_rate / 1000)

    # Process segments
    click.echo(f"Processing {len(segments_with_pos)} segments...")
    result_audio = np.zeros(len(audio) + fade_samples)  # Extra space for potential overlap

    with torch.no_grad():
        for idx, (start_sample, chunk) in enumerate(tqdm(segments_with_pos)):

            # Process the segment
            audio_out = process_segment(
                chunk, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                speaker_id, sample_rate, hop_length, device,
                key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                skip_cfg_strength, cfg_skip_layers, cfg_rescale,
                cvec_downsample_rate, target_loudness, restore_loudness, sliced_inference,
                robust_f0
            )
            
            # Ensure consistent length
            expected_length = len(chunk)
            if len(audio_out) > expected_length:
                audio_out = audio_out[:expected_length]
            elif len(audio_out) < expected_length:
                audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')
            
            # Apply fades
            if idx > 0:  # Not first segment
                audio_out = apply_fade(audio_out.copy(), fade_samples, fade_in=True)
                result_audio[start_sample:start_sample + fade_samples] *= \
                    np.linspace(1, 0, fade_samples)  # Fade out previous
            
            if idx < len(segments_with_pos) - 1:  # Not last segment
                audio_out[-fade_samples:] *= np.linspace(1, 0, fade_samples)  # Fade out
            
            # Add to result
            result_audio[start_sample:start_sample + len(audio_out)] += audio_out

    # Trim any extra padding
    result_audio = result_audio[:len(audio)]

    # Save output
    click.echo("Saving output...")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, torch.from_numpy(result_audio).unsqueeze(0), sample_rate)
    click.echo("Done!")


if __name__ == '__main__':
    main()