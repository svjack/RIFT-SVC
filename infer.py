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


def split(audio, sample_rate, hop_size, max_mel_frames=256, overlap=32):
    """
    Split audio into overlapping chunks based on maximum mel frames.

    Args:
        audio (np.ndarray): The input audio array.
        sample_rate (int): The sample rate of the audio.
        hop_size (int): The hop size used in mel spectrogram.
        max_mel_frames (int): Maximum number of mel frames per chunk.
        overlap (int): Number of overlapping frames between chunks.

    Returns:
        List of tuples containing (start_frame, audio_chunk).
    """
    frame_length = max_mel_frames * hop_size
    step = (max_mel_frames - overlap) * hop_size
    total_length = len(audio)
    result = []
    current_start = 0

    while current_start < total_length:
        current_end = current_start + frame_length
        audio_chunk = audio[current_start:current_end]
        result.append((current_start // hop_size, audio_chunk))
        if current_end >= total_length:
            break
        current_start += step

    return result

def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    """Cross fade two audio segments."""
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx:a.shape[0]] = (1 - k) * a[idx:] + k * b[:fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


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
@click.option('--sample-rate', type=int, default=44100, help='Sample rate')
@click.option('--infer-steps', type=int, default=32, help='Number of inference steps')
@click.option('--cfg-strength', type=float, default=2.0, help='Classifier-free guidance strength')
def main(model, input, output, speaker_id, key, device, hop_length, sample_rate, infer_steps, cfg_strength):
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

    # Process audio using sliding window
    click.echo("Processing audio with sliding window...")
    overlap = 32
    segments = split(audio, sample_rate, hop_length, max_mel_frames=256, overlap=overlap)
    click.echo(f"Split into {len(segments)} segments")
    
    result = np.zeros(0)
    current_length = 0

    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            audio_segment = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
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

            # Model inference
            mel_out, _ = model.sample(
                src_mel=mel,
                spk_id=spk_id,
                f0=f0,
                rms=rms,
                cvec=cvec,
                steps=infer_steps,
                cfg_strength=cfg_strength
            )

            # Generate audio
            audio_out = vocoder(mel_out.transpose(1, 2), f0)
            audio_out = audio_out.squeeze().cpu().numpy()

            # Cross fade and concatenate
            segment_length = len(audio_out)
            fade_length = overlap * hop_length
            if current_length > 0 and fade_length > 0:
                # Apply crossfade
                result = cross_fade(result, audio_out, current_length - fade_length)
            else:
                result = np.append(result, audio_out)
            current_length = len(result)
    
    # Save output
    click.echo("Saving output...")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, torch.from_numpy(result).unsqueeze(0), sample_rate)
    click.echo("Done!")

if __name__ == '__main__':
    main()