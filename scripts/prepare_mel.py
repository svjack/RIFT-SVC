#!/usr/bin/env python3
"""
generate_mel_specs.py

该脚本从meta_info.json中读取音频信息，对每个音频生成Mel spectrogram，
并以 .mel.pt 格式保存在相应目录下。

Usage:
    python prepare_mel.py --data-dir DATA_DIR [OPTIONS]

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
from functools import partial

from rift_svc.feature_extractors import get_mel_spectrogram
from multiprocessing_utils import run_parallel, get_device

def process_audio(audio, data_dir, hop_length, n_mel_channels, sample_rate, verbose, overwrite, device):
    """
    处理单个音频：读取WAV文件，生成Mel spectrogram，并保存为 .mel.pt 文件。
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    # 如果信息不全则跳过
    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}", err=True)
        return

    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    mel_path = Path(data_dir) / speaker / f"{file_name}.mel.pt"

    if mel_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing Mel spectrogram: {mel_path}", err=True)
        return

    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
        return

    try:
        waveform, sr = torchaudio.load(str(wav_path))
        # 确保形状正确
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(device)

        mel = get_mel_spectrogram(
            waveform,
            hop_size=hop_length,
            num_mels=n_mel_channels,
            sampling_rate=sample_rate,
            n_fft=2048,
            win_size=2048,
            fmin=40,
            fmax=16000,
        )
        mel = mel.cpu()  # 转回CPU保存

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
    help='预处理数据集的根目录。'
)
@click.option(
    '--hop-length',
    type=int,
    default=512,
    show_default=True,
    help='Mel spectrogram的hop length。'
)
@click.option(
    '--n-mel-channels',
    type=int,
    default=128,
    show_default=True,
    help='Mel通道数。'
)
@click.option(
    '--sample-rate',
    type=int,
    default=44100,
    show_default=True,
    help='目标采样率（Hz）。'
)
@click.option(
    '--num-workers',
    type=int,
    default=4,
    show_default=True,
    help='并行进程数。'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='是否覆盖已存在的Mel spectrogram。'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='是否打印详细日志。'
)
def generate_mel_specs(data_dir, hop_length, n_mel_channels, sample_rate, num_workers, verbose, overwrite):
    """
    对meta_info.json中的音频生成Mel spectrogram，并保存为 .mel.pt 文件。
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
    all_audios = (
        [{'type': 'train', 'index': i, **audio} for i, audio in enumerate(train_audios)] +
        [{'type': 'test', 'index': i, **audio} for i, audio in enumerate(test_audios)]
    )

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    device = get_device()
    if verbose:
        click.echo(f"Using device: {device}")

    torch.set_grad_enabled(False)

    process_func = partial(
        process_audio,
        data_dir=data_dir,
        hop_length=hop_length,
        n_mel_channels=n_mel_channels,
        sample_rate=sample_rate,
        verbose=verbose,
        overwrite=overwrite,
        device=device
    )

    run_parallel(all_audios, process_func, num_workers=num_workers, desc="Generating Mel Spectrograms")
    click.echo("Mel spectrogram generation complete.")

if __name__ == "__main__":
    generate_mel_specs()