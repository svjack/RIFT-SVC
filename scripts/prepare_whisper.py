#!/usr/bin/env python3
"""
prepare_whisper.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts content vectors for each audio file using a Whisper-like model, and saves the content vectors
as .whisper.pt files in the same directory as the original audio files.

Usage:
    python prepare_whisper.py --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --data-dir DIRECTORY             Path to the root of the preprocessed dataset directory. (Required)
    --model-path FILE_PATH           Path to the pre-trained Whisper model file. (Required)
    --layer-index INTEGER            Layer index to extract embeddings from. (Default: -2)
    --num-workers-per-device INTEGER  Number of workers per device for multiprocessing. (Default: 2)
    --verbose                        Enable verbose output.
    --overwrite                      Overwrite existing Whisper embeddings.
"""

import json
import math
import sys
from pathlib import Path

import click
import torch
import torchaudio
from transformers import AutoFeatureExtractor

from multiprocessing_utils import BaseWorker, run_multiprocessing
from rift_svc.encoders import WhisperEncoder


class WhisperWorker(BaseWorker):
    def load_model(self):
        """
        Load and return the Whisper model along with the feature extractor.
        """
        model = WhisperEncoder.from_pretrained(self.model_path)
        model = model.to(self.device)
        model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        return model

    def process_audio(self, waveform, sr, layer_index=-2, **kwargs):
        """
        Extract Whisper embeddings from the waveform.
        """
        # Resample if necessary
        if sr != self.feature_extractor.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.feature_extractor.sampling_rate)
            sr = self.feature_extractor.sampling_rate

        waveform = waveform.numpy()

        input_features = self.feature_extractor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            device=self.device,
            do_normalize=True
        )
        input_features = {k: v.to(self.device) for k, v in input_features.items()}

        with torch.no_grad():
            outputs = self.model(**input_features, output_hidden_states=True)
            whisper = outputs.hidden_states[layer_index].squeeze(0).cpu()

        # Compute truncation length based on duration and model specifics
        duration = waveform.shape[-1] / sr
        trunc_len = math.floor(duration * 50)  # Example computation
        whisper = whisper[:trunc_len].contiguous()

        return whisper

    def save_output(self, output, output_path):
        """
        Save the Whisper embedding to the specified path.
        """
        torch.save(output, output_path)

    def get_output_path(self, speaker, file_name):
        """
        Determine the output path for the Whisper embedding.
        """
        return Path(self.data_dir) / speaker / f"{file_name}.whisper.pt"

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
    help='Path to the pre-trained Whisper model file.'
)
@click.option(
    '--layer-index',
    type=int,
    default=-2,
    show_default=True,
    help='Layer index to extract embeddings from. -2 for the second last layer.'
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
    help='Overwrite existing Whisper embeddings.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def prepare_whisper(data_dir, model_path, layer_index, num_workers_per_device, overwrite, verbose):
    """
    Prepare Whisper embeddings for each audio file specified in the meta_info.json and save them as .whisper.pt files.
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
        worker_cls=WhisperWorker,
        all_audios=all_audios,
        data_dir=data_dir,
        model_path=model_path,
        num_workers_per_device=num_workers_per_device,
        verbose=verbose,
        overwrite=overwrite,
        layer_index=layer_index
    )

if __name__ == "__main__":
    prepare_whisper()
