"""
generate_mel_specs.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
generates Mel spectrograms for each audio file using the provided MelSpec class, and saves the spectrograms
as .mel.pt files in the same directory as the original audio files.

Usage:
    python generate_mel_specs.py --meta-info META_INFO_JSON --data-dir DATA_DIR [OPTIONS]

Options:
    --meta-info FILE_PATH        Path to the meta_info.json file. (Required)
    --data-dir DIRECTORY         Path to the root of the preprocessed dataset directory. (Required)
    --hop-length INTEGER         Hop length for Mel spectrogram. (Default: 256)
    --n-mel-channels INTEGER     Number of Mel channels. (Default: 128)
    --sample-rate INTEGER        Target sample rate in Hz. (Default: 22050)
    --verbose                    Enable verbose output.
"""

import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
import click
from tqdm import tqdm

from model.modules import get_mel_spectrogram


@click.command()
@click.option(
    '--meta-info',
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help='Path to the meta_info.json file.'
)
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the root of the preprocessed dataset directory.'
)
@click.option(
    '--hop-length',
    type=int,
    default=256,
    show_default=True,
    help='Hop length for Mel spectrogram.'
)
@click.option(
    '--n-mel-channels',
    type=int,
    default=128,
    show_default=True,
    help='Number of Mel channels.'
)
@click.option(
    '--sample-rate',
    type=int,
    default=44100,
    show_default=True,
    help='Target sample rate in Hz.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_mel_specs(meta_info, data_dir, hop_length, n_mel_channels, sample_rate, verbose):
    """
    Generate Mel spectrograms for each audio file specified in the meta_info.json and save them as .mel.pt files.
    """
    # Load meta_info.json
    try:
        with open(meta_info, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    speakers = meta.get('speakers', [])
    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])

    # Combine train and test audios
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        click.echo(f"Using device: {device}")

    # Initialize the global MelSpec object
    # mel_spec = MelSpec(
    #     hop_length=hop_length,
    #     n_mel_channels=n_mel_channels,
    #     target_sample_rate=sample_rate,
    # ).to(device)
    # mel_spec.eval()  # Set to evaluation mode

    # Disable gradient computation
    torch.set_grad_enabled(False)

    i = 0
    num_train = len(train_audios)
    # Process each audio file
    for audio in tqdm(all_audios, desc="Generating Mel Spectrograms", unit="file"):
        speaker = audio.get('speaker')
        file_name = audio.get('file_name')

        if not speaker or not file_name:
            if verbose:
                click.echo(f"Skipping invalid entry: {audio}", err=True)
            continue

        # Construct paths
        wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
        mel_path = Path(data_dir) / speaker / f"{file_name}.mel.pt"

        if not wav_path.is_file():
            if verbose:
                click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
            continue

        try:
            # Load audio
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.to(device)

            # Ensure waveform has proper shape
            # MelSpec expects (batch, channels, samples) or (batch, samples)
            # Since we're processing one file at a time, add batch dimension if necessary
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
                # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if necessary (though preprocessing assumes consistent sample rate)
            #if sr != sample_rate:
            #    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

            # Generate Mel spectrogram
            mel = get_mel_spectrogram(
                waveform,
                hop_size=hop_length,
                num_mels=n_mel_channels,
                sampling_rate=sample_rate,
            )
            mel = mel.cpu()  # Move to CPU for saving

            # Save the Mel spectrogram
            torch.save(mel, mel_path)

            if i < num_train:
                meta["train_audios"][i]["frame_len"] = mel.shape[-1]
            else:
                meta["test_audios"][i-num_train]["frame_len"] = mel.shape[-1]

            if verbose:
                click.echo(f"Saved Mel spectrogram: {mel_path}")

        except Exception as e:
            click.echo(f"Error processing {wav_path}: {e}", err=True)
            continue

        i+=1

    # Update meta_info.json with frame lengths
    meta_info_path = Path(meta_info)
    with open(meta_info_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    click.echo("Mel spectrogram generation complete.")
    

if __name__ == "__main__":
    generate_mel_specs()