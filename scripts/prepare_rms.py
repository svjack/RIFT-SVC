"""
generate_rms.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts the RMS energy for each audio file using a PyTorch-based approach, and saves the RMS energy
as a .rms.pt file in the same directory as the original audio file.

Usage:
    python generate_rms.py --data-dir DATA_DIR [OPTIONS]

Options:
    --data-dir DIRECTORY         Path to the root of the preprocessed dataset directory. (Required)
    --hop-length INTEGER         Hop length for RMS extraction. (Default: 512)
    --overwrite                  Overwrite existing RMS files.
    --verbose                    Enable verbose output.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
import torchaudio
import click
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count

from rift_svc.modules import RMSExtractor

def worker_process(audio_subset, data_dir, hop_length, queue, verbose, overwrite):
    """
    Worker function to extract RMS energy from a subset of audio files.

    Args:
        audio_subset (list): List of audio entries to process.
        data_dir (Path): Root directory of the preprocessed dataset.
        hop_length (int): Hop length for RMS extraction.
        queue (Queue): Multiprocessing queue to communicate progress.
        verbose (bool): If True, enable verbose output.
        overwrite (bool): If True, overwrite existing RMS files.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        rms_extractor = RMSExtractor(hop_length=hop_length).to(device)
        rms_extractor.eval()
    except Exception as e:
        queue.put(f"Error initializing RMSExtractor: {e}")
        return

    torch.set_grad_enabled(False)

    for audio in audio_subset:
        speaker = audio.get('speaker')
        file_name = audio.get('file_name')

        if not speaker or not file_name:
            if verbose:
                queue.put(f"Skipping invalid entry: {audio}")
            continue

        wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
        rms_path = Path(data_dir) / speaker / f"{file_name}.rms.pt"

        if rms_path.is_file() and not overwrite:
            if verbose:
                queue.put(f"Skipping existing RMS file: {rms_path}")
            queue.put("PROGRESS")
            continue

        if not wav_path.is_file():
            if verbose:
                queue.put(f"Warning: WAV file not found: {wav_path}")
            continue

        try:
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.to(device)

            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            rms = rms_extractor(waveform)
            rms = rms.cpu()
            torch.save(rms, rms_path)

            if verbose:
                queue.put(f"Saved RMS energy: {rms_path}")

            queue.put("PROGRESS")

        except Exception as e:
            queue.put(f"Error processing {wav_path}: {e}")
            continue

    queue.put("PROCESS_COMPLETE")

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
    '--overwrite',
    is_flag=True,
    default=False,
    help='Overwrite existing RMS files.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_rms(data_dir, hop_length, verbose, overwrite):
    """
    Generate RMS energy for each audio file specified in the meta_info.json and save them as .rms.pt files.
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

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    num_workers = min(cpu_count(), len(all_audios))
    split_audios = [[] for _ in range(num_workers)]
    for i, audio in enumerate(all_audios):
        split_audios[i % num_workers].append(audio)

    queue = Queue()
    processes = []
    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(
                split_audios[i],
                Path(data_dir),
                hop_length,
                queue,
                verbose,
                overwrite
            ),
            name=f"Process-{i+1}"
        )
        p.start()
        processes.append(p)

    with tqdm(total=len(all_audios), desc="Extracting RMS Energy", unit="file") as pbar:
        completed_processes = 0
        while completed_processes < num_workers:
            message = queue.get()
            if message == "PROGRESS":
                pbar.update(1)
            elif message.startswith("Saved RMS energy") and verbose:
                pbar.set_postfix({"Last Saved": message})
            elif message.startswith("Warning") and verbose:
                pbar.write(message)
            elif message.startswith("Error"):
                pbar.write(message)
            elif message.startswith("Error initializing RMSExtractor"):
                pbar.write(message)
            elif message == "PROCESS_COMPLETE":
                completed_processes += 1
            else:
                if verbose:
                    pbar.write(message)

    for p in processes:
        p.join()

    click.echo("RMS energy extraction complete.")

if __name__ == "__main__":
    generate_rms()