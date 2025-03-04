#!/usr/bin/env python3
"""
generate_rms.py

该脚本从meta_info.json中读取音频信息，对每个音频提取RMS energy，
并保存为 .rms.pt 文件。

Usage:
    python generate_rms.py --data-dir DATA_DIR [OPTIONS]
"""

import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
import torchaudio
import click
from functools import partial

from multiprocessing_utils import run_parallel, get_device
from rift_svc.feature_extractors import RMSExtractor


def get_rms_extractor(hop_length):
    """
    懒加载RMSExtractor，每个进程首次调用时加载并缓存。
    """
    if not hasattr(get_rms_extractor, "extractor"):
        
        device = get_device()
        extractor = RMSExtractor(hop_length=hop_length).to(device)
        extractor.eval()
        get_rms_extractor.extractor = extractor
        get_rms_extractor.device = device
    return get_rms_extractor.extractor, get_rms_extractor.device

def process_rms(audio, data_dir, hop_length, verbose):
    """
    对单个音频提取RMS energy，并保存为 .rms.pt 文件。
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')

    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}", err=True)
        return

    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    rms_path = Path(data_dir) / speaker / f"{file_name}.rms.pt"

    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}", err=True)
        return

    if rms_path.is_file():
        if verbose:
            click.echo(f"Skipping existing file: {rms_path}", err=True)
        return

    try:
        waveform, sr = torchaudio.load(str(wav_path))
        extractor, device = get_rms_extractor(hop_length)
        waveform = waveform.to(device)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        rms = extractor(waveform)  # 输出形状: (1, frames)
        rms = rms.cpu()
        torch.save(rms, rms_path)
        if verbose:
            click.echo(f"Saved RMS energy: {rms_path}")
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
    help='RMS提取的hop length。'
)
@click.option(
    '--num-workers',
    type=int,
    default=4,
    show_default=True,
    help='并行进程数。'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='是否打印详细日志。'
)
def generate_rms(data_dir, hop_length, num_workers, verbose):
    """
    对meta_info.json中的音频提取RMS energy，并保存为 .rms.pt 文件。
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

    process_func = partial(
        process_rms,
        data_dir=data_dir,
        hop_length=hop_length,
        verbose=verbose
    )

    run_parallel(
        all_audios,
        process_func,
        num_workers=num_workers,
        desc="Extracting RMS Energy"
    )

    click.echo("RMS energy extraction complete.")

if __name__ == "__main__":
    generate_rms()