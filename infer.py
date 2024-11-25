import os
import torch
import torchaudio
import click
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rmvpe import RMVPE
from rift_svc.modules import get_mel_spectrogram, RMSExtractor, HubertModelWithFinalProj
from rift_svc.utils import post_process_f0, interpolate_tensor
from rift_svc import CFM, DiT

from slicer import Slicer  # Importing the updated Slicer

# Import for loudness normalization
import pyloudnorm as pyln


def extract_state_dict(ckpt):
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '')
            new_state_dict[new_k] = v
    num_speakers = new_state_dict["transformer.spk_embed.weight"].shape[0]
    return new_state_dict, num_speakers


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--input', type=click.Path(exists=True), required=True, help='Input audio file')
@click.option('--output', type=click.Path(), required=True, help='Output audio file')
@click.option('--speaker-id', type=int, default=0, help='Target speaker ID')
@click.option('--key', type=int, default=0, help='Pitch shift in semitones')
@click.option('--device', type=str, default=None, help='Device to use (cuda/cpu)')
@click.option('--hop-length', type=int, default=512, help='Hop length')
@click.option('--window-size', type=int, default=256, help='Should align with the max len of model')
@click.option('--overlap-size', type=int, default=32, help='Overlap size')
@click.option('--sample-rate', type=int, default=44100, help='Sample rate')
@click.option('--infer-steps', type=int, default=32, help='Number of inference steps')
@click.option('--cfg-strength', type=float, default=2.0, help='Classifier-free guidance strength')
@click.option('--target_loudness', type=float, default=-18.0, help='Target loudness in LUFS for normalization')
def main(model, input, output, speaker_id, key, device, hop_length, sample_rate, infer_steps, cfg_strength, target_loudness):
    """Convert the voice in an audio file to a target speaker."""

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load models
    click.echo("Loading models...")
    vocoder = NsfHifiGAN('pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt').to(device)
    rmvpe = RMVPE(model_path="pretrained/rmvpe/model.pt", hop_length=160, device=device)
    hubert = HubertModelWithFinalProj.from_pretrained("pretrained/content-vec-best").to(device)
    rms_extractor = RMSExtractor(hop_length=hop_length).to(device)

    # Load the conversion model
    ckpt = torch.load(model, map_location='cpu')
    state_dict, num_speakers = extract_state_dict(ckpt)

    transformer = DiT(dim=768, depth=12, num_speaker=num_speakers)
    model = CFM(transformer=transformer)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

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
        threshold=-40.0,
        min_length=5000,
        min_interval=300,
        hop_size=20,
        max_sil_kept=5000
    )

    # Initialize Loudness Meter
    meter = pyln.Meter(sample_rate)  # Create BS.1770 meter

    # Create empty audio array
    result_audio = np.zeros_like(audio)

    # Step (1): Use slicer to segment the input audio and get positions
    click.echo("Slicing audio...")
    segments_with_pos = slicer.slice(audio)  # Now returns list of (start_pos, chunk)

    # Step (6): Repeat for all segments
    click.echo(f"Processing {len(segments_with_pos)} segments...")
    with torch.no_grad():
        for idx, (start_sample, chunk) in enumerate(tqdm(segments_with_pos)):
            end_sample = start_sample + len(chunk)

            # Handle potential overflow
            if end_sample > len(result_audio):
                end_sample = len(result_audio)
                chunk = chunk[:end_sample - start_sample]

            # --- Loudness Normalization Start ---
            # Measure the loudness of the segment
            loudness = meter.integrated_loudness(chunk)

            # Normalize the segment to the target loudness
            loudness_normalized_audio = pyln.normalize.loudness(chunk, loudness, target_loudness)

            # Handle clipping by scaling audio if necessary
            max_amp = np.max(np.abs(loudness_normalized_audio))
            if max_amp > 1.0:
                loudness_normalized_audio = loudness_normalized_audio * (0.99 / max_amp)
            segment = loudness_normalized_audio
            # --- Loudness Normalization End ---

            # Step (2): Obtain mel, cvec, f0, and rms
            audio_segment = torch.from_numpy(segment).float().unsqueeze(0).to(device)
            audio_segment_16khz = torchaudio.functional.resample(audio_segment, sample_rate, 16000)

            # Generate mel spectrogram
            mel = get_mel_spectrogram(
                audio_segment,
                sampling_rate=sample_rate,
                n_fft=2048,
                num_mels=128,
                hop_size=hop_length,
                win_size=2048,
                fmin=40,
                fmax=16000
            ).transpose(1, 2)

            # Extract content vectors
            cvec = hubert(audio_segment_16khz)["last_hidden_state"].squeeze(0)
            cvec = interpolate_tensor(cvec, mel.shape[1])[None, :]

            # Extract F0
            f0 = rmvpe.infer_from_audio(audio_segment, sample_rate=sample_rate, device=device)
            f0 = post_process_f0(f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
            if key != 0:
                f0 = f0 * 2 ** (key / 12)

            # Extract RMS
            rms = rms_extractor(audio_segment)

            # Prepare inputs
            spk_id = torch.LongTensor([speaker_id]).to(device)
            f0 = torch.from_numpy(f0).float().to(device)[None, :]

            # Step (2 continued): Infer mel using overlapping sliding window
            # Define sliding window parameters
            window_size = 256
            overlap_size = 32
            step_size = window_size - overlap_size
            total_frames = mel.shape[1]
            inferred_mel = torch.zeros_like(mel)

            hann_window = torch.hann_window(window_size).to(device)

            for frame in range(0, total_frames, step_size):
                end = frame + window_size
                if end > total_frames:
                    end = total_frames
                    frame = max(0, end - window_size)
                mel_window = mel[:, frame:end, :]
                cvec_window = cvec[:, frame:end, :]
                f0_window = f0[:, frame:end]
                rms_window = rms[:, frame:end]

                mel_out, _ = model.sample(
                    src_mel=mel_window,
                    spk_id=spk_id,
                    f0=f0_window,
                    rms=rms_window,
                    cvec=cvec_window,
                    steps=infer_steps,
                    cfg_strength=cfg_strength
                )
                inferred_mel[:, frame:end, :] += mel_out * hann_window[:end - frame].unsqueeze(0).unsqueeze(-1)
            
            # Normalize the inferred mel to account for overlapping windows
            window_sum = torch.zeros_like(inferred_mel)
            for frame in range(0, total_frames, step_size):
                end = frame + window_size
                if end > total_frames:
                    end = total_frames
                    frame = max(0, end - window_size)
                window_sum[:, frame:end, :] += hann_window[:end - frame].unsqueeze(0).unsqueeze(-1)
            inferred_mel /= window_sum + 1e-8  # Avoid division by zero

            # Step (4): Input mel and f0 into vocoder
            audio_pred = vocoder(inferred_mel.transpose(1, 2), f0)
            audio_pred = audio_pred.squeeze().cpu().numpy()

            # Ensure audio_pred matches the segment length
            expected_length = end_sample - start_sample
            actual_length = len(audio_pred)

            if actual_length > expected_length:
                audio_pred = audio_pred[:expected_length]
            elif actual_length < expected_length:
                # Pad with zeros if shorter
                audio_pred = np.pad(audio_pred, (0, expected_length - actual_length), 'constant')

            # Step (5): Fill the predicted audio segment into the empty audio
            result_audio[start_sample:end_sample] = audio_pred

    # Step (6): Save the filled audio
    click.echo("Saving output...")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, torch.from_numpy(result_audio).unsqueeze(0), sample_rate)
    click.echo("Done!")


if __name__ == '__main__':
    main()