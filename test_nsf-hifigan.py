# %%

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.nsf_hifigan import Vocoder, STFT

# %%

vocoder = Vocoder('nsf-hifigan', 'pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt')

# %%
mel = torch.randn(1, 256, 128).to(vocoder.device)
f0 = torch.randn(1, 256, 16).to(vocoder.device)
audio = vocoder.infer(mel, f0)


# %%

from model.rmvpe import RMVPE

rmvpe = RMVPE(model_path="pretrained/rmvpe/model.pt", hop_length=160, device=vocoder.device)

def post_process_f0(f0, sample_rate, hop_length, n_frames, silence_front=0.0):
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

    return f0

# %%

audio_path = "data/pretrain/kiritan/01_0.wav"

import torchaudio


waveform, sr = torchaudio.load(str(audio_path))  # Convert Path to string
waveform = waveform.to(vocoder.device)

# Ensure waveform has proper shape for RMVPE (batch, samples)
if len(waveform.shape) == 1:
    waveform = waveform.unsqueeze(0)  # Shape: (1, samples)
elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
    # Convert to mono by averaging channels
    waveform = waveform.mean(dim=0, keepdim=True)  # Shape: (1, samples)

# Extract f0 using RMVPE
f0 = rmvpe.infer_from_audio(waveform, sample_rate=sr, device=vocoder.device, thred=0.03, use_viterbi=False)


# %%

stft = STFT(
    sr=44100,
    n_fft=2048,
    n_mels=128,
    hop_length=512,
    win_size=2048,
    fmin=40,
    fmax=16000,
)
# %%

mel = stft.get_mel(waveform)
mel = mel.transpose(1, 2)
# %%

f0 = post_process_f0(f0, sr, 512, mel.size(1)).astype(np.float32)
# %%

mel.shape, f0.shape
# %%

audio = vocoder.infer(mel, torch.tensor(f0)[None, :, None].cuda())

from IPython.display import Audio
Audio(audio.squeeze().cpu().numpy(), rate=44100)
# %%

audio.shape 
# %%

# Squeeze to remove batch dimension and convert to numpy
mel_np = mel.squeeze().cpu().numpy()
f0_np = f0

# Create a figure with two subplots sharing the same x-axis
fig, ax1 = plt.subplots(figsize=(12, 8), sharex=True)

sns.heatmap(mel_np.T[::-1], ax=ax1, cmap='viridis', cbar=True)
ax1.set_title('Mel Spectrogram and F0 Overlay')
ax1.set_ylabel('Mel Bands')

# Create a secondary y-axis for F0
ax2 = ax1.twinx()
ax2.plot(f0_np, color='r', label='F0')
ax2.set_ylabel('Frequency (Hz)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend for F0
ax2.legend(loc='upper right')

# Set x-axis label
ax1.set_xlabel('Time Frames')

plt.tight_layout()
plt.savefig('mel_f0.png')

# %%

from model.modules import get_mel_spectrogram

mel2 = get_mel_spectrogram(
    waveform,
    sampling_rate=44100,
    n_fft=2048,
    num_mels=128,
    hop_size=512,
    win_size=2048,
    fmin=40,
    fmax=16000,
)

# %%

mel2.shape, mel.shape
# %%

mel2_np = mel2.squeeze().cpu().numpy()

fig, ax1 = plt.subplots(figsize=(12, 8), sharex=True)

sns.heatmap(mel2_np, ax=ax1, cmap='viridis', cbar=True)
# Reverse the y-axis for the heatmap
ax1.invert_yaxis()

ax1.set_title('Mel Spectrogram and F0 Overlay')
ax1.set_ylabel('Mel Bands')

# Create a secondary y-axis for F0
ax2 = ax1.twinx()
ax2.plot(f0_np, color='r', label='F0')
ax2.set_ylabel('Frequency (Hz)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend for F0
ax2.legend(loc='upper right')

# Set x-axis label
ax1.set_xlabel('Time Frames')

plt.tight_layout()
plt.savefig('mel2_f0.png')
# %%

audio = vocoder.infer(mel2.transpose(1, 2), torch.tensor(f0)[None, :, None].cuda())

from IPython.display import Audio
Audio(audio.squeeze().cpu().numpy(), rate=44100)
# %%

Audio(waveform.squeeze().cpu().numpy(), rate=44100)
# %%
import torch.nn as nn

class RMSExtractor(nn.Module):
    def __init__(self, hop_length=512, window_length=2048):
        """
        Initializes the RMSExtractor with the specified hop_length.

        Args:
            hop_length (int): Number of samples between successive frames.
        """
        super(RMSExtractor, self).__init__()
        self.hop_length = hop_length
        self.window_length = window_length

    def forward(self, inp):
        """
        Extracts RMS energy from the input audio tensor.

        Args:
            inp (Tensor): Audio tensor of shape (batch, samples).

        Returns:
            Tensor: RMS energy tensor of shape (batch, frames).
        """
        # Square the audio signal
        audio_squared = inp ** 2

        # Use the same padding as mel spectrogram
        padding = (self.window_length - self.hop_length) // 2
        audio_padded = torch.nn.functional.pad(
            audio_squared, (padding, padding), mode='reflect'
        )

        # Unfold to create frames with window_length instead of hop_length
        frames = audio_padded.unfold(1, self.window_length, self.hop_length)  # Shape: (batch, frames, window_length)

        # Compute mean energy per frame
        mean_energy = frames.mean(dim=-1)  # Shape: (batch, frames)

        # Compute RMS by taking square root
        rms = torch.sqrt(mean_energy)  # Shape: (batch, frames)

        return rms


# %%



rms = RMSExtractor()
rms_np = rms(waveform).squeeze().cpu().numpy()
# %%

rms_np.shape
# %%
