import os
import json
import sys
import click
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the root of the preprocessed dataset directory.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def combine_features(data_dir, verbose):
    """
    Combine precomputed features (mel_spec, rms, f0, cvec) into a single file for each audio.
    This enhances loading speed by reducing the number of file I/O operations during data loading.
    """
    meta_info = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])
    all_audios = train_audios + test_audios

    if verbose:
        click.echo(f"Total audios to process: {len(all_audios)}")

    # Set up multiprocessing
    num_processes = cpu_count()
    process_fn = partial(process_single_audio, data_dir=data_dir, verbose=verbose)
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_fn, all_audios),
            total=len(all_audios),
            desc="Combining Features",
            unit="audio"
        ))

    successful = sum(1 for r in results if r)
    if verbose:
        click.echo(f"Feature combination complete. Successfully processed {successful}/{len(all_audios)} files.")

def process_single_audio(audio, data_dir, verbose):
    """Helper function to process a single audio file"""
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')

    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}")
        return False

    # Construct paths for individual feature files
    base_path = os.path.join(data_dir, speaker, file_name)
    mel_path = f"{base_path}.mel.pt"
    rms_path = f"{base_path}.rms.pt"
    f0_path = f"{base_path}.f0.pt"
    cvec_path = f"{base_path}.cvec.pt"

    # Check if all feature files exist
    missing_files = []
    for path in [mel_path, rms_path, f0_path, cvec_path]:
        if not os.path.isfile(path):
            missing_files.append(path)

    if missing_files:
        if verbose:
            click.echo(f"Missing files for {file_name}: {missing_files}")
        return False

    try:
        # Load all features
        mel = torch.load(mel_path, weights_only=True)
        rms = torch.load(rms_path, weights_only=True)
        f0 = torch.load(f0_path, weights_only=True)
        cvec = torch.load(cvec_path, weights_only=True)

        # Combine features into a single dictionary
        combined_features = {
            'mel': mel,
            'rms': rms,
            'f0': f0,
            'cvec': cvec,
        }

        # Save the combined features
        combined_path = f"{base_path}.combined.pt"
        torch.save(combined_features, combined_path)

        if verbose:
            click.echo(f"Combined features saved: {combined_path}")

        # Remove individual feature files
        os.remove(mel_path)
        os.remove(rms_path)
        os.remove(f0_path)
        os.remove(cvec_path)
        return True

    except Exception as e:
        click.echo(f"Error combining features for {file_name}: {e}", err=True)
        return False

if __name__ == "__main__":
    combine_features()