#!/usr/bin/env python3
"""
prepare_f0.py

该脚本从meta_info.json中读取音频信息，对每个音频提取f0，
并保存为 .f0.pt 文件。

Usage:
    python prepare_f0.py --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --data-dir DIRECTORY            预处理数据集根目录 (必选)。
    --model-path FILE_PATH          预训练RMVPE模型路径 (必选)。
    --hop-length INTEGER            f0提取的hop length (默认: 256)。
    --sample-rate INTEGER           音频采样率 (默认: 22050)。
    --num-workers INTEGER           并行进程数 (默认: 2)。
    --overwrite                     是否覆盖已存在的f0文件。
    --verbose                       是否打印详细日志。
"""

import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import click
import torch
import torchaudio
from functools import partial
import multiprocessing

from multiprocessing_utils import run_parallel, get_device
from rift_svc.rmvpe.inference import RMVPE

RMVPE_HOP_LENGTH = 160

def get_f0_model(model_path):
    """
    懒加载RMVPE模型，每个进程首次调用时加载并缓存。
    """
    if not hasattr(get_f0_model, "model"):
        device = get_device()
        model = RMVPE(model_path=model_path, hop_length=RMVPE_HOP_LENGTH, device=device)
        get_f0_model.model = model
        get_f0_model.device = device
    return get_f0_model.model, get_f0_model.device

def process_f0(audio, data_dir, model_path, hop_length, sample_rate, overwrite, verbose):
    """
    对单个音频提取f0，并保存为 .f0.pt 文件。
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}")
        return
    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    f0_path = Path(data_dir) / speaker / f"{file_name}.f0.pt"
    
    if f0_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing f0 file: {f0_path}")
        return
    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}")
        return
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        model, device = get_f0_model(model_path)
        waveform = waveform.to(device)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        with torch.no_grad():
            f0 = model.infer_from_audio(
                waveform,
                sample_rate=sample_rate,
                device=device,
                thred=0.03,
                use_viterbi=False
            )
        n_frames = int(waveform.shape[-1] // hop_length) + 1

        from rift_svc.utils import post_process_f0
        f0_processed = post_process_f0(
            f0=f0,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_frames=n_frames,
            silence_front=0.0
        )
        f0_tensor = torch.from_numpy(f0_processed).float().cpu()
        torch.save(f0_tensor, f0_path)
        if verbose:
            click.echo(f"Saved f0: {f0_path}")
    except Exception as e:
        click.echo(f"Error processing {wav_path}: {e}")

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='预处理数据集的根目录。'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=False,
    default="pretrained/rmvpe/model.pt",
    help='预训练RMVPE模型路径。'
)
@click.option(
    '--hop-length',
    type=int,
    default=512,
    show_default=True,
    help='f0提取的hop length。'
)
@click.option(
    '--sample-rate',
    type=int,
    default=44100,
    show_default=True,
    help='音频采样率（Hz）。'
)
@click.option(
    '--num-workers',
    type=int,
    default=2,
    show_default=True,
    help='并行进程数。'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='是否覆盖已存在的f0文件。'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='是否打印详细日志。'
)
def prepare_f0(data_dir, model_path, hop_length, sample_rate, num_workers, overwrite, verbose):
    """
    对meta_info.json中的音频提取f0，并保存为 .f0.pt 文件。
    """
    meta_info_path = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}")
        sys.exit(1)

    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.")
        sys.exit(1)

    process_func = partial(
        process_f0,
        data_dir=data_dir,
        model_path=model_path,
        hop_length=hop_length,
        sample_rate=sample_rate,
        overwrite=overwrite,
        verbose=verbose
    )

    run_parallel(
        all_audios,
        process_func,
        num_workers=num_workers,
        desc="Extracting f0"
    )

    click.echo("f0 extraction complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    prepare_f0()