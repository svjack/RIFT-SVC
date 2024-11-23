import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from einops import rearrange

from rift_svc.utils import (
    exists, 
    lens_to_mask,
) 


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma: float = 0.,
        odeint_kwargs: dict = dict(
            method='euler'  # 'midpoint'
        ),
        spk_drop_prob: float = 0.2,
        num_mel_channels: int | None = 128,
        lognorm: bool = False,
    ):
        super().__init__()

        self.num_mel_channels = num_mel_channels

        # Unconditional guiding
        self.spk_drop_prob = spk_drop_prob

        # Transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # Condition flow related parameters
        self.sigma = sigma

        # Sampling related parameters
        self.odeint_kwargs = odeint_kwargs

        self.mel_min = -12
        self.mel_max = 2

        self.lognorm = lognorm

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        src_mel: torch.Tensor,           # [b n d]
        spk_id: torch.Tensor,        # [b]
        f0: torch.Tensor,            # [b n]
        rms: torch.Tensor,           # [b n]
        cvec: torch.Tensor,          # [b n d]
        *,
        frame_lens: torch.Tensor,    # Merged from duration and lens
        steps: int = 32,
        cfg_strength: float = 2.,
        sway_sampling_coef: float | None = None,
        seed: int | None = None,
        interpolate_condition: bool = False,
        t_inter: float = 0.1,
    ):
        self.eval()

        batch, mel_seq_len, device = *src_mel.shape[:2], src_mel.device

        mask = lens_to_mask(frame_lens)

        # Define the ODE function
        def fn(t, x):
            # Predict flow
            pred = self.transformer(
                x=x, 
                spk=spk_id, 
                f0=f0, 
                rms=rms, 
                cvec=cvec, 
                time=t, 
                drop_spk=False,  # Update as needed
                mask=mask
            )
            if cfg_strength < 1e-5:
                return pred
            
            null_pred = self.transformer(
                x=x, 
                spk=spk_id, 
                f0=f0, 
                rms=rms, 
                cvec=cvec, 
                time=t, 
                drop_spk=True, 
                mask=mask
            )
            return pred + (pred - null_pred) * cfg_strength

        # Noise input
        y0 = []
        for _ in range(batch):
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(cvec.shape[1], self.num_mel_channels, device=self.device))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # Handle duplicate test case
        if interpolate_condition:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * self.norm_mel(src_mel)
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps, device=self.device)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        
        sampled = trajectory[-1]
        out = self.denorm_mel(sampled)
        out = torch.where(mask.unsqueeze(-1), out, src_mel)

        return out, trajectory

    def forward(
        self,
        inp: torch.Tensor,        # mel
        spk_id: torch.Tensor,     # [b]
        f0: torch.Tensor,         # [b n]
        rms: torch.Tensor,        # [b n]
        cvec: torch.Tensor,       # [b n d]
        *,
        lens: torch.Tensor | None = None,
    ):
        batch, seq_len, dtype, device, sigma = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # Handle lengths and masks
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # Typically padded to max length in batch

        x1 = self.norm_mel(inp)
        x0 = torch.randn_like(x1)

        if self.lognorm:
            quantiles = torch.linspace(0, 1, batch + 1).to(x1.device)
            z = quantiles[:-1] + torch.rand((batch,)).to(x1.device) / batch
            # now transform to normal
            z = torch.erfinv(2 * z - 1) * math.sqrt(2)
            time = torch.sigmoid(z)
        else:
            time = torch.rand((batch,), dtype=dtype, device=self.device)

        t = rearrange(time, 'b -> b 1 1')
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # Transformer and unconditional guiding dropout rates
        drop_spk = random.random() < self.spk_drop_prob  # Adjusted for DiT's drop_spk

        # Call Transformer
        pred = self.transformer(
            x=φ, 
            spk=spk_id, 
            f0=f0, 
            rms=rms, 
            cvec=cvec, 
            time=time, 
            drop_spk=drop_spk, 
            mask=mask
        )

        # Flow matching loss
        loss = F.mse_loss(pred, flow, reduction='none')
        loss = loss[mask]

        return loss.mean(), pred

    def norm_mel(self, mel: torch.Tensor):
        return (mel - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1
    
    def denorm_mel(self, mel: torch.Tensor):
        return (mel + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min
