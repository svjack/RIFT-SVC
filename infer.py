import torch
import torchaudio
import click
import numpy as np
import math
from pathlib import Path
import pyloudnorm as pyln
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rmvpe import RMVPE
from rift_svc.modules import get_mel_spectrogram, RMSExtractor
from rift_svc.encoders import WhisperEncoder, HubertModelWithFinalProj
from rift_svc.utils import post_process_f0, interpolate_tensor
from rift_svc import RF, DiT

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
    model_cfg = ckpt['hyper_parameters']['cfg']['model']['cfg']
    dataset_cfg = ckpt['hyper_parameters']['cfg']['dataset']
    return new_state_dict, spk2idx, model_cfg, dataset_cfg


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--input', type=click.Path(exists=True), required=True, help='Input audio file')
@click.option('--output', type=click.Path(), required=True, help='Output audio file')
@click.option('--speaker', type=str, required=True, help='Target speaker')
@click.option('--key-shift', type=int, default=0, help='Pitch shift in semitones')
@click.option('--device', type=str, default=None, help='Device to use (cuda/cpu)')
@click.option('--infer-steps', type=int, default=32, help='Number of inference steps')
@click.option('--cfg-strength', type=float, default=0.0, help='Classifier-free guidance strength')
@click.option('--target-loudness', type=float, default=-18.0, help='Target loudness in LUFS for normalization')
@click.option('--restore-loudness', is_flag=True, default=False, help='Restore loudness to original')
@click.option('--interpolate-src', type=float, default=0.0, help='Interpolate source audio')
@click.option('--fade-duration', type=float, default=20.0, help='Fade duration in milliseconds')
def main(
    model,
    input,
    output,
    speaker,
    key_shift,
    device,
    infer_steps,
    cfg_strength,
    target_loudness,
    restore_loudness,
    interpolate_src,
    fade_duration
):
    """Convert the voice in an audio file to a target speaker."""

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load models
    click.echo("Loading models...")

    # Load the conversion model
    ckpt = torch.load(model, map_location='cpu')
    state_dict, spk2idx, dit_cfg, dataset_cfg = extract_state_dict(ckpt)

    transformer = DiT(num_speaker=len(spk2idx), **dit_cfg)
    model = RF(transformer=transformer)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    try:
        speaker_id = spk2idx[speaker]
    except KeyError:
        raise ValueError(f"Speaker {speaker} not found in the model's speaker list, valid speakers are {spk2idx.keys()}")
    hop_length = dataset_cfg['hop_length']
    sample_rate = dataset_cfg['sample_rate']

    vocoder = NsfHifiGAN('pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt').to(device)
    rmvpe = RMVPE(model_path="pretrained/rmvpe/model.pt", hop_length=160, device=device)
    hubert = HubertModelWithFinalProj.from_pretrained("pretrained/content-vec-best").to(device)
    rms_extractor = RMSExtractor(hop_length=hop_length).to(device)
    whisper_encoder = WhisperEncoder.from_pretrained("pretrained/whisper-large-v3").to(device)
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("pretrained/whisper-large-v3")

    # Load and preprocess input audio
    click.echo("Loading audio...")
    audio, sr = torchaudio.load(input)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    if len(audio.shape) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    audio = audio.numpy().squeeze()

    # Initialize Slicer
    slicer = Slicer(
        sr=sample_rate,
        threshold=-30.0,
        min_length=3000,
        min_interval=100,
        hop_size=10,
        max_sil_kept=300
    )

    # Initialize Loudness Meter
    meter = pyln.Meter(sample_rate, block_size=0.1)  # Create BS.1770 meter

    crossfade_ms = 40  # crossfade length in milliseconds
    crossfade_size = int(crossfade_ms * sample_rate / 1000)  # convert to samples

    # Create empty audio array with extra space for crossfade
    result_audio = np.zeros(len(audio) + crossfade_size)

    # Step (1): Use slicer to segment the input audio and get positions
    click.echo("Slicing audio...")
    segments_with_pos = slicer.slice(audio)  # Now returns list of (start_pos, chunk)

    if restore_loudness:
        click.echo(f"Will restore loudness to original")

    # Add these utility functions
    def apply_fade(audio, fade_samples, fade_in=True):
        """Apply fade in/out using half of a Hanning window"""
        fade_window = np.hanning(fade_samples * 2)
        if fade_in:
            fade_curve = fade_window[:fade_samples]
        else:
            fade_curve = fade_window[fade_samples:]
        audio[:fade_samples] *= fade_curve
        return audio

    def process_segment(audio_segment, mel=None, cvec=None, f0=None, rms=None):
        """Process a single audio segment with consistent handling"""
        # Normalize input segment
        original_loudness = meter.integrated_loudness(audio_segment)
        normalized_audio = pyln.normalize.loudness(audio_segment, original_loudness, target_loudness)
        
        # Handle potential clipping
        max_amp = np.max(np.abs(normalized_audio))
        if max_amp > 1.0:
            normalized_audio = normalized_audio * (0.99 / max_amp)
        
        # Convert to tensor if not already provided
        if mel is None or cvec is None or f0 is None or rms is None:
            audio_tensor = torch.from_numpy(normalized_audio).float().unsqueeze(0).to(device)
            audio_16khz = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
            
            mel = get_mel_spectrogram(
                audio_tensor,
                sampling_rate=sample_rate,
                n_fft=2048,
                num_mels=128,
                hop_size=hop_length,
                win_size=2048,
                fmin=40,
                fmax=16000
            ).transpose(1, 2)
            
            cvec = hubert(audio_16khz)["last_hidden_state"].squeeze(0)
            cvec = interpolate_tensor(cvec, mel.shape[1])[None, :]

            input_features = whisper_feature_extractor(audio_16khz.cpu().numpy(), sampling_rate=16000, return_tensors="pt", device=device, do_normalize=True)
            input_features = {k: v.to(device) for k, v in input_features.items()}
            whisper_outputs = whisper_encoder(**input_features, output_hidden_states=True)
            trunc_len = math.floor((audio_16khz.shape[1] / 16000)*50)
            whisper = whisper_outputs.hidden_states[-2][0, :trunc_len]
            whisper = interpolate_tensor(whisper, mel.shape[1])[None, :]


            f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
            f0 = post_process_f0(f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
            if key_shift != 0:
                f0 = f0 * 2 ** (key_shift / 12)
            f0 = torch.from_numpy(f0).float().to(device)[None, :]
            
            rms = rms_extractor(audio_tensor)
            spk_id = torch.LongTensor([speaker_id]).to(device)
        
        # Process with model
        mel_out, _ = model.sample(
            src_mel=mel,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            whisper=whisper,
            steps=infer_steps,
            cfg_strength=cfg_strength,
            interpolate_condition=True if interpolate_src > 0 else False,
            t_inter=interpolate_src
        )
        
        # Generate audio
        audio_out = vocoder(mel_out.transpose(1, 2), f0)
        audio_out = audio_out.squeeze().cpu().numpy()

        if restore_loudness:
            # Restore original loudness
            audio_out_loudness = meter.integrated_loudness(audio_out)
            audio_out = pyln.normalize.loudness(audio_out, audio_out_loudness, original_loudness)

            # Handle clipping
            max_amp = np.max(np.abs(audio_out))
            if max_amp > 1.0:
                audio_out = audio_out * (0.99 / max_amp)
            
        return audio_out

    # Calculate fade size in samples
    fade_samples = int(fade_duration * sample_rate / 1000)

    # Process segments
    click.echo(f"Processing {len(segments_with_pos)} segments...")
    result_audio = np.zeros(len(audio) + fade_samples)  # Extra space for potential overlap

    with torch.no_grad():
        for idx, (start_sample, chunk) in enumerate(tqdm(segments_with_pos)):
            # Process the segment
            audio_out = process_segment(chunk)
            
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