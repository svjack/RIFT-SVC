"""
generate_rms.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts the RMS energy for each audio file using a PyTorch-based approach, and saves the RMS energy
as a .rms.pt file in the same directory as the original audio file.

Usage:
    python generate_rms.py --meta-info META_INFO_JSON --data-dir DATA_DIR [OPTIONS]

Options:
    --data-dir DIRECTORY         Path to the root of the preprocessed dataset directory. (Required)
    --hop-length INTEGER         Hop length for RMS extraction. (Default: 256)
    --verbose                    Enable verbose output.
"""

import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import torch
import torchaudio
import click
from tqdm import tqdm
from rift_svc.modules import RMSExtractor


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
    help='Hop length for RMS extraction.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_rms(data_dir, hop_length, verbose):
    """
    Generate RMS energy for each audio file specified in the meta_info.json and save them as .rms.pt files.
    """
    # Load meta_info.json
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

    # Combine train and test audios
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        click.echo(f"Using device: {device}")

    # Initialize the global RMSExtractor object
    rms_extractor = RMSExtractor(hop_length=hop_length).to(device)
    rms_extractor.eval()  # Set to evaluation mode

    # Disable gradient computation
    torch.set_grad_enabled(False)

    # Process each audio file
    for audio in tqdm(all_audios, desc="Extracting RMS Energy", unit="file"):
        speaker = audio.get('speaker')
        file_name = audio.get('file_name')

        if not speaker or not file_name:
            if verbose:
                click.echo(f"Skipping invalid entry: {audio}", err=True)
            continue

        # Construct paths
        wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
        rms_path = Path(data_dir) / speaker / f"{file_name}.rms.pt"

        if not wav_path.is_file():
            if verbose:
                click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
            continue

        try:
            # Load audio - Convert Path to string
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.to(device)

            # Ensure waveform has proper shape
            # RMSExtractor expects (batch, samples)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Shape: (1, samples)
            elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
                # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)  # Shape: (1, samples)

            # Resample if necessary (though preprocessing assumes consistent sample rate)
            # If your preprocessing ensured all audios have the target sample rate, this can be skipped
            # Otherwise, implement resampling here if needed

            # Extract RMS energy
            rms = rms_extractor(waveform)  # Shape: (1, frames)

            # Move RMS to CPU for saving
            rms = rms.cpu()

            # Save the RMS energy
            torch.save(rms, rms_path)

            if verbose:
                click.echo(f"Saved RMS energy: {rms_path}")

        except Exception as e:
            click.echo(f"Error processing {wav_path}: {e}", err=True)
            continue

    click.echo("RMS energy extraction complete.")


if __name__ == "__main__":
    generate_rms()