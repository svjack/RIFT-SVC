#!/usr/bin/env python3
"""
prepare_f0.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts the fundamental frequency (f0) for each audio file using a pre-trained RMVPE model,
applies post-processing to the extracted f0, and saves the f0 as a .f0.pt file in the same directory
as the original audio file.

Usage:
    python prepare_f0.py --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --data-dir DIRECTORY            Path to the root of the preprocessed dataset directory. (Required)
    --model-path FILE_PATH          Path to the pre-trained RMVPE model file. (Required)
    --hop-length INTEGER            Hop length for f0 extraction. (Default: 256)
    --sample-rate INTEGER           Target sample rate in Hz. (Default: 22050)
    --num-workers-per-device INTEGER  Number of workers per device for multiprocessing. (Default: 2)
    --verbose                       Enable verbose output.
    --overwrite                     Overwrite existing f0 files.
"""

import json
import sys
from pathlib import Path

import click
import torch
import torchaudio
from multiprocessing_utils import BaseWorker, run_multiprocessing
from rift_svc.rmvpe.inference import RMVPE
from rift_svc.utils import post_process_f0


class F0Worker(BaseWorker):
    def load_model(self):
        """
        Initialize and return the RMVPE model.
        """
        rmvpe = RMVPE(model_path=self.model_path, hop_length=self.kwop_length, device=self.device)
        return rmvpe

    def process_audio(self, waveform, sr, hop_length=256, sample_rate=22050, **kwargs):
        """
        Extract and post-process f0 from the waveform.
        """
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate).to(self.device)
            waveform = resampler(waveform)

        # Extract f0 using RMVPE
        f0 = self.model.infer_from_audio(
            waveform,
            sample_rate=sample_rate,
            device=self.device,
            thred=0.03,
            use_viterbi=False
        )

        # Compute number of frames based on audio length and hop_length
        n_frames = int(waveform.shape[-1] // hop_length) + 1

        # Post-process f0 to align with Mel spectrogram frames
        f0_processed = post_process_f0(
            f0=f0,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_frames=n_frames,
            silence_front=0.0  # Adjust if you have leading silence
        )

        return torch.from_numpy(f0_processed).float().cpu()

    def save_output(self, output, output_path):
        """
        Save the f0 tensor to the specified path.
        """
        torch.save(output, output_path)

    def get_output_path(self, speaker, file_name):
        """
        Determine the output path for the f0 tensor.
        """
        return Path(self.data_dir) / speaker / f"{file_name}.f0.pt"


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
    required=True,
    help='Path to the pre-trained RMVPE model file.'
)
@click.option(
    '--hop-length',
    type=int,
    default=256,
    show_default=True,
    help='Hop length for f0 extraction.'
)
@click.option(
    '--sample-rate',
    type=int,
    default=22050,
    show_default=True,
    help='Sample rate of the audio files in Hz.'
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
    help='Overwrite existing f0 files.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def prepare_f0(data_dir, model_path, hop_length, sample_rate, num_workers_per_device, overwrite, verbose):
    """
    Prepare f0 for each audio file specified in the meta_info.json and save them as .f0.pt files.
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
        worker_cls=F0Worker,
        all_audios=all_audios,
        data_dir=data_dir,
        model_path=model_path,
        num_workers_per_device=num_workers_per_device,
        verbose=verbose,
        overwrite=overwrite,
        hop_length=hop_length,
        sample_rate=sample_rate
    )

if __name__ == "__main__":
    prepare_f0()