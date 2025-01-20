"""
generate_mel_specs.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
generates Mel spectrograms for each audio file using multiprocessing, and saves the spectrograms
as .mel.pt files in the same directory as the original audio files.

Usage:
    python prepare_mel.py --meta-info META_INFO_JSON --data-dir DATA_DIR [OPTIONS]

Options:
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
import torchaudio
import click
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

from rift_svc.modules import get_mel_spectrogram


def process_audio(audio, data_dir, hop_length, n_mel_channels, sample_rate, verbose, overwrite):
    """
    Process a single audio file: load, generate Mel spectrogram, save, and return frame length.

    Parameters:
        audio (dict): Dictionary containing audio metadata.
        data_dir (str): Root directory of the preprocessed dataset.
        hop_length (int): Hop length for Mel spectrogram.
        n_mel_channels (int): Number of Mel channels.
        sample_rate (int): Target sample rate in Hz.
        verbose (bool): Flag to enable verbose output.
        overwrite (bool): Flag to overwrite existing Mel spectrograms.
    Returns:
        tuple or None: Returns a tuple (audio_type, index, frame_len) if successful, else None.
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    audio_type = audio.get('type')
    index = audio.get('index')

    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}", err=True)
        return None

    # Construct paths
    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    mel_path = Path(data_dir) / speaker / f"{file_name}.mel.pt"

    if mel_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing Mel spectrogram: {mel_path}", err=True)
        return None

    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
        return None

    try:
        # Load audio
        waveform, sr = torchaudio.load(str(wav_path))

        # Ensure waveform has proper shape
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Generate Mel spectrogram
        mel = get_mel_spectrogram(
            waveform.cuda(),
            hop_size=hop_length,
            num_mels=n_mel_channels,
            sampling_rate=sample_rate,
            n_fft=2048,
            win_size=2048,
            fmin=40,
            fmax=16000,
        )
        mel = mel.cpu()  # Move to CPU for saving

        # Save the Mel spectrogram directly to disk
        torch.save(mel, mel_path)

        if verbose:
            click.echo(f"Saved Mel spectrogram: {mel_path}")

    except Exception as e:
        click.echo(f"Error processing {wav_path}: {e}", err=True)

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the root of the preprocessed dataset directory.'
)
@click.option(
    '--hop-length',
    type=int,
    default=512,
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
    '--num-workers',
    type=int,
    default=cpu_count(),
    show_default=True,
    help='Number of workers.'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='Overwrite existing Mel spectrograms.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_mel_specs(data_dir, hop_length, n_mel_channels, sample_rate, num_workers, verbose, overwrite):
    """
    Generate Mel spectrograms for each audio file specified in the meta_info.json and save them as .mel.pt files.
    This version uses multiprocessing for enhanced efficiency.
    """
    meta_info = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    speakers = meta.get('speakers', [])
    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])

    # Combine train and test audios with labels
    all_audios = [{'type': 'train', 'index': i, **audio} for i, audio in enumerate(train_audios)] + \
                 [{'type': 'test', 'index': i, **audio} for i, audio in enumerate(test_audios)]

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        click.echo(f"Using device: {device}")

    # Disable gradient computation
    torch.set_grad_enabled(False)

    # Initialize multiprocessing Pool
    if verbose:
        click.echo(f"Starting multiprocessing with {num_workers} workers...")

    # Create a partial function with fixed parameters
    process_func = partial(
        process_audio,
        data_dir=data_dir,
        hop_length=hop_length,
        n_mel_channels=n_mel_channels,
        sample_rate=sample_rate,
        verbose=verbose,
        overwrite=overwrite
    )

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance
        tqdm(
            pool.imap_unordered(process_func, all_audios),
            total=len(all_audios),
            desc="Generating Mel Spectrograms",
            unit="file"
        )


    click.echo("Mel spectrogram generation complete.")


if __name__ == "__main__":
    generate_mel_specs()