import math

from einops import rearrange
from jaxtyping import Float, Bool
from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import apply_rotary_pos_emb


def dynamic_range_compression_torch(
        x: Float[torch.Tensor, "n_mels mel_len"],
        C: float = 1,
        clip_val: float = 1e-5
) -> Float[torch.Tensor, "n_mels mel_len"]:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(
        magnitudes: Float[torch.Tensor, "n_mels mel_len"]
) -> Float[torch.Tensor, "n_mels mel_len"]:
    return dynamic_range_compression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def get_mel_spectrogram(
    y: Float[torch.Tensor, "n"],
    n_fft: int = 2048,
    num_mels: int = 128,
    sampling_rate: int = 44100,
    hop_size: int = 512,
    win_size: int = 2048,
    fmin: int = 40,
    fmax: int | None = 16000,
    center: bool = False,
) -> Float[torch.Tensor, "n_mels mel_len"]:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal with shape (n,).
        n_fft (int, optional): FFT size. Defaults to 1024.
        num_mels (int, optional): Number of mel bins. Defaults to 128.
        sampling_rate (int, optional): Sampling rate of the input signal. Defaults to 44100.
        hop_size (int, optional): Hop size for STFT. Defaults to 256.
        win_size (int, optional): Window size for STFT. Defaults to 1024.
        fmin (int, optional): Minimum frequency for mel filterbank. Defaults to 0.
        fmax (int | None, optional): Maximum frequency for mel filterbank. If None, defaults to sr/2.0. Defaults to None.
        center (bool, optional): Whether to pad the input to center the frames. Defaults to False.

    Returns:
        torch.Tensor: Mel spectrogram with shape (n_mels, mel_len).
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


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


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb = None):
        emb = self.proj(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.proj = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.proj(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

# ReLU^2
class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x, inplace=True).square()

# FeedForward
class MLP(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        #self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.activation = ReLU2()
        self.dropout = nn.Dropout(dropout)
        self.mlp_proj = nn.Linear(dim, inner_dim)
        self.mlp_out = nn.Linear(inner_dim, dim_out)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.mlp_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.mlp_out(x)
        return x


class ConvLinear(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        #self.dwconv = nn.Conv1d(dim, dim_out, kernel_size=7, padding=3, groups=dim)
        self.dwconv = nn.Conv1d(dim, dim_out, kernel_size=31, padding=15, groups=dim)
        self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.dim = dim
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = int(dim // head_dim)
        self.inner_dim = dim
        self.dropout = dropout
        self.scale = 1 / dim

        # self.q_proj = nn.Linear(dim, self.inner_dim)
        # self.k_proj = nn.Linear(dim, self.inner_dim)
        # self.v_proj = nn.Linear(dim, self.inner_dim)
        self.q_proj = nn.Linear(dim, self.inner_dim)
        self.k_proj = ConvLinear(dim, self.inner_dim)
        self.v_proj = ConvLinear(dim, self.inner_dim)

        self.norm_q = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)
        self.norm_k = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)

        self.attn_out = nn.Linear(self.inner_dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[torch.Tensor, "b n d"],  # noised input x
        mask: Bool[torch.Tensor, "b n"] | None = None,
        rope = None,  # rotary position embedding for x
    ) -> Float[torch.Tensor, "b n d"]:
        batch_size = x.shape[0]

        # projections
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if xpos_scale is not None else (1., 1.)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.num_heads
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # mask
        if mask is not None:
            attn_mask = mask
            attn_mask = rearrange(attn_mask, 'b n -> b 1 1 n')
            attn_mask = attn_mask.expand(batch_size, self.num_heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scale)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        x = x.to(query.dtype)

        # linear proj and dropout
        x = self.attn_out(x)
        x = self.attn_dropout(x)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            x = x.masked_fill(~mask, 0.)

        return x


# DiT Block
class DiTBlock(nn.Module):

    def __init__(self, dim: int, head_dim: int, ff_mult: float = 4, dropout: float = 0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim = dim,
            head_dim = head_dim,
            dropout = dropout,
        )
        
        self.mlp_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim = dim, mult = ff_mult, dropout = dropout)

    def forward(
        self,
        x: Float[torch.Tensor, "b n d"],  # noised input
        t: Float[torch.Tensor, "b d"],  # time embedding
        mask: Bool[torch.Tensor, "b n"] | None = None,
        rope: Float[torch.Tensor, "b d"] | None = None,
    ) -> Float[torch.Tensor, "b n d"]:
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        norm = self.mlp_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        mlp_output = self.mlp(norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


# sinusoidal position embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[torch.Tensor, "b"], scale: float = 1000) -> Float[torch.Tensor, "b d"]:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# time step conditioning embedding
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time2emb = SinusPositionEmbedding(freq_embed_dim)
        self.time_emb = nn.Linear(freq_embed_dim, dim)
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim, dim)

    def forward(self, timestep: Float[torch.Tensor, "b"]) -> Float[torch.Tensor, "b d"]:
        time = self.time2emb(timestep)
        time = self.time_emb(time)
        time = self.act(time)
        time = self.proj(time)
        return time
