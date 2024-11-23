import os
import torch
import torchaudio
import click
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rmvpe import RMVPE
from rift_svc.modules import get_mel_spectrogram
from rift_svc import CFM, DiT

def split(audio, sample_rate, hop_size, db_thresh=-40, min_len=5000):
    """Split audio into chunks at silences."""
    from slicer import Slicer
    slicer = Slicer(
        sr=sample_rate,
        threshold=db_thresh,
        min_length=min_len
    )
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                    start_frame,
                    audio[int(start_frame * hop_size):int(end_frame * hop_size)]
                ))
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
    
    # Load the conversion model
    ckpt = torch.load(model, map_location='cpu')
    model_args = ckpt['hyper_parameters']['model_config']
    
    transformer = DiT(**model_args)
    model = CFM(transformer=transformer)
    model.load_state_dict(ckpt['state_dict'])
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

        # Process audio in chunks
    click.echo("Processing audio in chunks...")
    segments = split(audio, sample_rate, hop_length)
    click.echo(f"Split into {len(segments)} segments")
    
    result = np.zeros(0)
    current_length = 0
    
    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            audio_segment = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            
            # Extract F0
            f0 = rmvpe.infer_from_audio(audio_segment, sample_rate=sample_rate, device=device)
            if key != 0:
                f0 = f0 * 2 ** (key / 12)
            
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
            
            # Prepare inputs
            spk_id = torch.LongTensor([speaker_id]).to(device)
            f0 = torch.from_numpy(f0).float().to(device)[None, :]
            
            # Model inference
            mel_out, _ = model.sample(
                src_mel=mel,
                spk_id=spk_id,
                f0=f0,
                steps=infer_steps,
                cfg_strength=cfg_strength
            )
            
            # Generate audio
            audio_out = vocoder(mel_out.transpose(1, 2), f0)
            audio_out = audio_out.squeeze().cpu().numpy()
            
            # Cross fade and concatenate
            silent_length = round(start_frame * hop_length) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, audio_out)
            else:
                result = cross_fade(result, audio_out, current_length + silent_length)
            current_length = current_length + silent_length + len(audio_out)
    
    # Save output
    click.echo("Saving output...")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, torch.from_numpy(result).unsqueeze(0), sample_rate)
    click.echo("Done!")

if __name__ == '__main__':
    main()