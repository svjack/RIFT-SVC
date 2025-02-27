#!/usr/bin/env python3
"""
prepare_cvec.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts content vectors for each audio file using a HuBERT-like model, and saves the content vectors
as .cvec.pt files in the same directory as the original audio files.

Usage:
    python prepare_cvec.py --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --data-dir DIRECTORY            Path to the root of the preprocessed dataset directory. (Required)
    --model-path FILE_PATH          Path to the pre-trained HuBERT-like model file. (Required)
    --num-workers-per-device INTEGER  Number of workers per device for multiprocessing. (Default: 2)
    --verbose                       Enable verbose output.
    --overwrite                     Overwrite existing content vectors.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import multiprocessing
import click
import torch
import torchaudio
from multiprocessing_utils import BaseWorker, run_multiprocessing
from rift_svc.feature_extractors import HubertModelWithFinalProj


def roll_pad(wav, shift):
    wav = torch.roll(wav, shift, dims=1)
    if shift > 0:
        wav[:, :shift] = 0
    else:
        wav[:, shift:] = 0
    return wav


class ContentVectorWorker(BaseWorker):
    CVEC_SAMPLE_RATE = 16000

    def load_model(self):
        """
        Load and return the HuBERT-like model.
        """
        model = HubertModelWithFinalProj.from_pretrained(self.model_path)
        model = model.to(self.device)
        model.eval()
        return model

    def process_audio(self, waveform, sr, overwrite=False):
        """
        Extract content vectors from the waveform.
        """
        if sr != self.CVEC_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.CVEC_SAMPLE_RATE).to(self.device)
            waveform = resampler(waveform)

        with torch.no_grad():
            cvec = self.model(waveform)  # Assuming the model returns a dict with 'last_hidden_state'
            cvec = cvec["last_hidden_state"].squeeze(0).cpu()

            waveform = roll_pad(waveform, -160)
            cvec_shifted = self.model(waveform)["last_hidden_state"].squeeze(0).cpu()

            n, d = cvec.shape
            cvec = torch.stack([cvec, cvec_shifted], dim=1).view(n*2, d)

        return cvec

    def save_output(self, output, output_path):
        """
        Save the content vector to the specified path.
        """
        torch.save(output, output_path)

    def get_output_path(self, speaker, file_name):
        """
        Determine the output path for the content vector.
        """
        return Path(self.data_dir) / speaker / f"{file_name}.cvec.pt"


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the root of the preprocessed dataset directory.'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=False,
    default="pretrained/content-vec-best",
    help='Path to the pre-trained HuBERT-like model file.'
)
@click.option(
    '--num-workers-per-device',
    type=int,
    default=2,
    show_default=True,
    help='Number of workers per device for multiprocessing.'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='Overwrite existing content vectors.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def prepare_contentvec(data_dir, model_path, num_workers_per_device, overwrite, verbose):
    """
    Prepare content vectors for each audio file specified in the meta_info.json and save them as .cvec.pt files.
    """
    meta_info_path = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])

    # Combine train and test audios
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    run_multiprocessing(
        worker_cls=ContentVectorWorker,
        all_audios=all_audios,
        data_dir=data_dir,
        model_path=model_path,
        num_workers_per_device=num_workers_per_device,
        verbose=verbose,
        overwrite=overwrite
    )

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    prepare_contentvec()
