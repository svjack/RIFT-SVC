"""
generate_rms.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts the RMS energy for each audio file using a PyTorch-based approach with multiprocessing,
and saves the RMS energy as a .rms.pt file in the same directory as the original audio file.

Usage:
    python generate_rms.py --meta-info META_INFO_JSON --data-dir DATA_DIR [OPTIONS]

Options:
    --data-dir DIRECTORY         Path to the root of the preprocessed dataset directory. (Required)
    --hop-length INTEGER         Hop length for RMS extraction. (Default: 256)
    --num-workers INTEGER        Number of parallel workers. (Default: number of CPU cores)
    --verbose                    Enable verbose output.
"""

import json
import sys, os
from pathlib import Path
import torch
import torchaudio
import click
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rift_svc.modules import RMSExtractor


def process_rms(audio, data_dir, hop_length, device, verbose):
    """
    Process a single audio file: load, extract RMS energy, and save.

    Parameters:
        audio (dict): Dictionary containing audio metadata.
        data_dir (str): Root directory of the preprocessed dataset.
        hop_length (int): Hop length for RMS extraction.
        device (torch.device): Device to perform computations on.
        verbose (bool): Flag to enable verbose output.

    Returns:
        bool: True if processing is successful, else False.
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')

    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}", err=True)
        return False

    # Construct paths
    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    rms_path = Path(data_dir) / speaker / f"{file_name}.rms.pt"

    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
        return False

    try:
        # Load audio
        waveform, sr = torchaudio.load(str(wav_path))
        waveform = waveform.to(device)

        # Ensure waveform has proper shape
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # Shape: (1, samples)
        elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Shape: (1, samples)

        # Initialize RMSExtractor if not already done
        if not hasattr(process_rms, "rms_extractor"):
            process_rms.rms_extractor = RMSExtractor(hop_length=hop_length).to(device)
            process_rms.rms_extractor.eval()
            torch.set_grad_enabled(False)

        # Extract RMS energy
        rms = process_rms.rms_extractor(waveform)  # Shape: (1, frames)

        # Move RMS to CPU for saving
        rms = rms.cpu()

        # Save the RMS energy
        torch.save(rms, rms_path)

        if verbose:
            click.echo(f"Saved RMS energy: {rms_path}")

        return True

    except Exception as e:
        click.echo(f"Error processing {wav_path}: {e}", err=True)
        return False


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
    '--num-workers',
    type=int,
    default=cpu_count(),
    show_default=True,
    help='Number of parallel workers.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_rms(data_dir, hop_length, num_workers, verbose):
    """
    Generate RMS energy for each audio file specified in the meta_info.json and save them as .rms.pt files.
    This version uses multiprocessing for enhanced efficiency.
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
        click.echo(f"Starting multiprocessing with {num_workers} workers...")

    # Partial function with fixed parameters
    process_func = partial(
        process_rms,
        data_dir=data_dir,
        hop_length=hop_length,
        device=device,
        verbose=verbose
    )

    successful = 0
    total = len(all_audios)

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_func, all_audios),
            total=total,
            desc="Extracting RMS Energy",
            unit="file"
        ):
            if result:
                successful += 1

    click.echo(f"RMS energy extraction complete. Successfully processed {successful}/{total} files.")


if __name__ == "__main__":
    generate_rms()