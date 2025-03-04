#!/usr/bin/env python3
"""
prepare_cvec.py

该脚本从meta_info.json中读取音频信息，对每个音频提取content vector，
并保存为 .cvec.pt 文件。

Usage:
    python prepare_cvec.py --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --data-dir DIRECTORY            预处理数据集根目录 (必选)。
    --model-path FILE_PATH          预训练HuBERT-like模型路径 (必选)。
    --num-workers INTEGER           并行进程数 (默认: 2)。
    --overwrite                     是否覆盖已存在的content vectors。
    --verbose                       是否打印详细日志。
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
from functools import partial

from multiprocessing_utils import run_parallel, get_device
from rift_svc.feature_extractors import HubertModelWithFinalProj


def roll_pad(wav, shift):
    wav = torch.roll(wav, shift, dims=1)
    if shift > 0:
        wav[:, :shift] = 0
    else:
        wav[:, shift:] = 0
    return wav

CVEC_SAMPLE_RATE = 16000

def get_cvec_model(model_path):
    """
    懒加载HuBERT模型，每个进程首次调用时加载并缓存。
    """
    if not hasattr(get_cvec_model, "model"):
        device = get_device()
        model = HubertModelWithFinalProj.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        get_cvec_model.model = model
        get_cvec_model.device = device
    return get_cvec_model.model, get_cvec_model.device

def process_cvec(audio, data_dir, model_path, overwrite, verbose):
    """
    对单个音频提取content vector并保存为 .cvec.pt 文件。
    """
    speaker = audio.get('speaker')
    file_name = audio.get('file_name')
    if not speaker or not file_name:
        if verbose:
            click.echo(f"Skipping invalid entry: {audio}")
        return
    
    wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
    cvec_path = Path(data_dir) / speaker / f"{file_name}.cvec.pt"
    
    if cvec_path.is_file() and not overwrite:
        if verbose:
            click.echo(f"Skipping existing content vector: {cvec_path}")
        return
    
    if not wav_path.is_file():
        if verbose:
            click.echo(f"Warning: WAV file not found: {wav_path}")
        return
    
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        model, device = get_cvec_model(model_path)
        waveform = waveform.to(device)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != CVEC_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, CVEC_SAMPLE_RATE)

        with torch.no_grad():
            output = model(waveform)  # 返回字典，包含 "last_hidden_state"
            cvec = output["last_hidden_state"].squeeze(0).cpu()

            # 处理移位后的情况
            waveform_shifted = roll_pad(waveform, -160)
            output_shifted = model(waveform_shifted)
            cvec_shifted = output_shifted["last_hidden_state"].squeeze(0).cpu()

            n, d = cvec.shape
            cvec = torch.stack([cvec, cvec_shifted], dim=1).view(n * 2, d)
        
        torch.save(cvec, cvec_path)
        if verbose:
            click.echo(f"Saved content vector: {cvec_path}")
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
    default="pretrained/content-vec-best",
    help='预训练HuBERT-like模型路径。'
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
    help='是否覆盖已存在的content vectors。'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='是否打印详细日志。'
)
def prepare_contentvec(data_dir, model_path, num_workers, overwrite, verbose):
    """
    对meta_info.json中的音频提取content vectors，并保存为 .cvec.pt 文件。
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
    
    # 将 model_path 作为参数传入处理函数
    process_func = partial(
        process_cvec,
        data_dir=data_dir,
        model_path=model_path,
        overwrite=overwrite,
        verbose=verbose
    )
    
    run_parallel(
        all_audios,
        process_func,
        num_workers=num_workers,
        desc="Extracting Content Vectors"
    )
    
    click.echo("Content vector extraction complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    prepare_contentvec()
