import io
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Int
from PIL import Image


def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v: Any) -> bool:
    return v is not None

def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d


def draw_mel_specs(gt: np.ndarray, gen: np.ndarray, diff: np.ndarray, cache_path: str):
    vmin = min(gt.min(), gen.min())
    vmax = max(gt.max(), gen.max())
    
    # Create figure with space for colorbar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'hspace': 0})
    
    # Plot all spectrograms with the same scale
    im1 = ax1.imshow(gt, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_ylabel('GT', fontsize=14)
    ax1.set_xticks([])
    
    im2 = ax2.imshow(gen, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Gen', fontsize=14)
    ax2.set_xticks([])
    
    # Find symmetric limits for difference plot
    diff_abs_max = max(abs(diff.min()), abs(diff.max()))
    
    im3 = ax3.imshow(diff, origin='lower', aspect='auto',
                     cmap='RdBu_r',  # Red-White-Blue colormap (reversed)
                     vmin=-diff_abs_max, vmax=diff_abs_max)
    ax3.set_ylabel('Diff', fontsize=14)
    
    fig.colorbar(im1, ax=[ax1, ax2], location='right', label='Magnitude')
    fig.colorbar(im3, ax=[ax3], location='right', label='Difference')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Open with PIL and save as compressed JPEG
    img = Image.open(buf)
    img = img.convert('RGB')
    img.save(cache_path, 'JPEG', quality=85, optimize=True)
    buf.close()



# tensor helpers

def lens_to_mask(
    t: Int[torch.Tensor, "b"],
    length: int | None = None
) -> Bool[torch.Tensor, "b n"]: 

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return seq < t[..., None]


def l2_grad_norm(model: torch.nn.Module):
    return torch.cat([p.grad.data.flatten() for p in model.parameters() if p.grad is not None]).norm(2)


def nearest_interpolate_tensor(tensor, new_size):
    # Add two dummy dimensions to make it [1, 1, n, d]
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    # Interpolate
    interpolated = F.interpolate(tensor, size=(new_size, tensor.shape[-1]), mode='nearest')
    
    # Remove the dummy dimensions
    interpolated = interpolated.squeeze(0).squeeze(0)
    
    return interpolated


def linear_interpolate_tensor(tensor, new_size):
    # Assumes input tensor shape is [n, d]
    # Rearrange tensor to shape [1, d, n] to prepare for linear interpolation
    tensor = tensor.transpose(0, 1).unsqueeze(0)
    
    # Interpolate along the length dimension (last dimension) using linear interpolation.
    # align_corners=True preserves the boundary values; adjust this flag if needed.
    interpolated = F.interpolate(tensor, size=new_size, mode='linear', align_corners=True)
    
    # Restore the tensor to shape [new_size, d]
    return interpolated.squeeze(0).transpose(0, 1)


# f0 helpers

def post_process_f0(f0, sample_rate, hop_length, n_frames, silence_front=0.0, cut_last=True):
    """
    Post-process the extracted f0 to align with Mel spectrogram frames.

    Args:
        f0 (numpy.ndarray): Extracted f0 array.
        sample_rate (int): Sample rate of the audio.
        hop_length (int): Hop length used during processing.
        n_frames (int): Total number of frames (for alignment).
        silence_front (float): Seconds of silence to remove from the front.

    Returns:
        numpy.ndarray: Processed f0 array aligned with Mel spectrogram frames.
    """
    # Calculate number of frames to skip based on silence_front
    start_frame = int(silence_front * sample_rate / hop_length)
    real_silence_front = start_frame * hop_length / sample_rate
    # Assuming silence_front has been handled during RMVPE inference if needed

    # Handle unvoiced frames by interpolation
    uv = f0 == 0
    if np.any(~uv):
        f0_interp = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0[uv] = f0_interp
    else:
        # If no voiced frames, set all to zero
        f0 = np.zeros_like(f0)

    # Align with hop_length frames
    origin_time = 0.01 * np.arange(len(f0))  # Placeholder: Adjust based on RMVPE's timing
    target_time = hop_length / sample_rate * np.arange(n_frames - start_frame)
    f0 = np.interp(target_time, origin_time, f0)
    uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
    f0[uv] = 0

    # Pad the silence_front if needed
    f0 = np.pad(f0, (start_frame, 0), mode='constant')

    if cut_last:
        return f0[:-1]
    else:
        return f0


# LR scheduler helpers

from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayWithWarmup(_LRScheduler):
    def __init__(self, optimizer, num_training_steps, warmup_steps=0, min_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_training_steps (int): Total number of training steps.
            warmup_steps (int): Number of steps to linearly warm-up the learning rate.
            min_lr (float): The minimum learning rate value after decay.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(LinearDecayWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using a linear schedule with warmup."""
        # current step is stored in self.last_epoch (which is incremented every call to step())
        step = self.last_epoch

        # If within warmup phase, increase linearly from 0 to base_lr
        if step < self.warmup_steps:
            return [base_lr * float(step) / float(max(1, self.warmup_steps))
                    for base_lr in self.base_lrs]
        # If within the decay phase
        elif step < self.num_training_steps:
            # progress is a number between 0 and 1 indicating how far along we are in the decay phase
            progress = float(step - self.warmup_steps) / float(max(1, self.num_training_steps - self.warmup_steps))
            # linearly interpolate between base_lr and min_lr
            return [((1.0 - progress) * (base_lr - self.min_lr)) + self.min_lr
                    for base_lr in self.base_lrs]
        # If passed all training steps, set lr to min_lr
        else:
            return [self.min_lr for _ in self.base_lrs]