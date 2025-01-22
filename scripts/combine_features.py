import os
import json
import sys
import click
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import concurrent.futures

def process_single_audio(audio, data_dir, verbose):
    """Helper function to process a single audio file."""
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')

    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}")
        return False

    base_path = os.path.join(data_dir, speaker, file_name)
    mel_path = f"{base_path}.mel.pt"
    rms_path = f"{base_path}.rms.pt"
    f0_path = f"{base_path}.f0.pt"
    cvec_path = f"{base_path}.cvec.pt"
    whisper_path = f"{base_path}.whisper.pt"
    spk_path = f"{base_path}.spk.pt"

    missing_files = [p for p in [mel_path, rms_path, f0_path, cvec_path, whisper_path, spk_path] 
                     if not os.path.isfile(p)]

    if missing_files:
        if verbose:
            click.echo(f"Missing files for {file_name}: {missing_files}")
        return False

    try:
        mel = torch.load(mel_path, map_location='cpu', weights_only=True)
        rms = torch.load(rms_path, map_location='cpu', weights_only=True)
        f0 = torch.load(f0_path, map_location='cpu', weights_only=True)
        cvec = torch.load(cvec_path, map_location='cpu', weights_only=True)
        whisper = torch.load(whisper_path, map_location='cpu', weights_only=True)
        spk = torch.load(spk_path, map_location='cpu', weights_only=True)

        combined_features = {
            'mel': mel,
            'rms': rms,
            'f0': f0,
            'cvec': cvec,
            'whisper': whisper,
            'spk': spk,
        }

        combined_path = f"{base_path}.combined.pt"
        torch.save(combined_features, combined_path)

        if verbose:
            click.echo(f"Combined features saved: {combined_path}")

        # Remove original feature files
        for path in [mel_path, rms_path, f0_path, cvec_path, whisper_path, spk_path]:
            os.remove(path)

        return True

    except Exception as e:
        if verbose:
            click.echo(f"Error combining features for {file_name}: {e}", err=True)
        return False

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
    """Combine precomputed features into a single file for each audio."""
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

    MAX_WORKERS = min(os.cpu_count(), 8)
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_func = partial(process_single_audio, data_dir=data_dir, verbose=verbose)
        for result in tqdm(
            executor.map(process_func, all_audios),
            total=len(all_audios),
            desc="Combining Features",
            unit="audio"
        ):
            results.append(result)

    successful = sum(r for r in results if r)
    if verbose:
        click.echo(f"Feature combination complete. Successfully processed {successful}/{len(all_audios)} files.")

if __name__ == "__main__":
    combine_features()