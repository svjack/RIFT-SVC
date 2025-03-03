from typing import Union, List, Literal
from jaxtyping import Bool
import torch
from torch import nn
import torch.nn.functional as F
import math
from torchdiffeq import odeint

from einops import rearrange

from rift_svc.utils import (
    exists, 
    lens_to_mask,
) 


def sample_time(time_schedule: Literal['uniform', 'lognorm'], size: int, device: torch.device):
    if time_schedule == 'uniform':
        t = torch.rand((size,), device=device)
    elif time_schedule == 'lognorm':
        # stratified sampling of normals
        # first stratified sample from uniform
        quantiles = torch.linspace(0, 1, size + 1).to(device)
        z = quantiles[:-1] + torch.rand((size,)).to(device) / size
        # now transform to normal
        z = torch.erfinv(2 * z - 1) * math.sqrt(2)
        t = torch.sigmoid(z)
    return t


class RF(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        time_schedule: Literal['uniform', 'lognorm'] = 'lognorm',
        odeint_kwargs: dict = dict(
            method='euler'
        ),
    ):
        super().__init__()

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # Sampling related parameters
        self.odeint_kwargs = odeint_kwargs
        self.time_schedule = time_schedule

        self.mel_min = -12
        self.mel_max = 2


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
        frame_len: torch.Tensor | None = None, # [b]
        steps: int = 32,
        bad_cvec: torch.Tensor | None = None,
        ds_cfg_strength: float = 0.0,
        spk_cfg_strength: float = 0.0,
        skip_cfg_strength: float = 0.0,
        cfg_skip_layers: Union[int, List[int], None] = None,
        cfg_rescale: float = 0.7,
    ):
        self.eval()

        batch, mel_seq_len, num_mel_channels = src_mel.shape
        device = src_mel.device

        if not exists(frame_len):
            frame_len = torch.full((batch,), mel_seq_len, device=device)

        mask = lens_to_mask(frame_len)

        # Define the ODE function
        def fn(t, x):
            pred = self.transformer(
                x=x, 
                spk=spk_id, 
                f0=f0, 
                rms=rms, 
                cvec=cvec, 
                time=t, 
                mask=mask
            )
            cfg_flag = (ds_cfg_strength > 1e-5) or (skip_cfg_strength > 1e-5) or (spk_cfg_strength > 1e-5)
            if cfg_rescale > 1e-5 and cfg_flag:
                std_pred = pred.std()

            if ds_cfg_strength > 1e-5:
                assert exists(bad_cvec), "bad_cvec is required when cfg_strength is greater than 0"
                bad_cvec_pred = self.transformer(
                    x=x, 
                    spk=spk_id, 
                    f0=f0, 
                    rms=rms, 
                    cvec=bad_cvec, 
                    time=t, 
                    mask=mask,
                    skip_layers=cfg_skip_layers
                )

                pred = pred + (pred - bad_cvec_pred) * ds_cfg_strength
            
            if skip_cfg_strength > 1e-5:
                skip_pred = self.transformer(
                    x=x, 
                    spk=spk_id, 
                    f0=f0, 
                    rms=rms, 
                    cvec=cvec,
                    time=t, 
                    mask=mask,
                    skip_layers=cfg_skip_layers
                )

                pred = pred + (pred - skip_pred) * skip_cfg_strength
            
            if spk_cfg_strength > 1e-5:
                null_spk_pred = self.transformer(
                    x=x, 
                    spk=spk_id, 
                    f0=f0, 
                    rms=rms, 
                    cvec=cvec, 
                    time=t, 
                    mask=mask,
                    drop_speaker=True
                )

                pred = pred + (pred - null_spk_pred) * spk_cfg_strength

            if cfg_rescale > 1e-5 and cfg_flag:
                std_cfg = pred.std()
                pred_rescaled = pred * (std_pred / std_cfg)
                pred = cfg_rescale * pred_rescaled + (1 - cfg_rescale) * pred

            return pred

        # Noise input
        y0 = torch.randn(batch, mel_seq_len, num_mel_channels, device=self.device)
        # mask out the padded tokens
        y0 = y0.masked_fill(~mask.unsqueeze(-1), 0)

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = self.denorm_mel(sampled)
        out = torch.where(mask.unsqueeze(-1), out, src_mel)

        return out, trajectory

    def forward(
        self,
        mel: torch.Tensor,        # mel
        spk_id: torch.Tensor,     # [b]
        f0: torch.Tensor,         # [b n]
        rms: torch.Tensor,        # [b n]
        cvec: torch.Tensor,       # [b n d]
        frame_len: torch.Tensor | None = None,
        drop_speaker: Union[bool, Bool[torch.Tensor, "b"]] = False,
    ):
        batch, seq_len, dtype, device = *mel.shape[:2], mel.dtype, self.device

        # Handle lengths and masks
        if not exists(frame_len):
            frame_len = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(frame_len, length=seq_len)  # Typically padded to max length in batch

        x1 = self.norm_mel(mel)
        x0 = torch.randn_like(x1)

        # uniform time steps sampling
        time = sample_time(self.time_schedule, batch, self.device)

        t = rearrange(time, 'b -> b 1 1')
        xt = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.transformer(
            x=xt, 
            spk=spk_id, 
            f0=f0, 
            rms=rms, 
            cvec=cvec, 
            time=time, 
            drop_speaker=drop_speaker,
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
