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
            # Check if we need to do batched processing
            need_batched = False
            num_cond = 1  # Regular prediction
            
            if ds_cfg_strength > 1e-5:
                assert exists(bad_cvec), "bad_cvec is required when cfg_strength is greater than 0"
                need_batched = True
                num_cond += 1
            
            if spk_cfg_strength > 1e-5:
                need_batched = True
                num_cond += 1
            
            if not need_batched:
                # Standard case - just do the regular prediction
                pred = self.transformer(
                    x=x, 
                    spk=spk_id, 
                    f0=f0, 
                    rms=rms, 
                    cvec=cvec, 
                    time=t, 
                    mask=mask
                )
            else:
                # Get original batch size
                orig_batch = x.shape[0]
                total_batch = orig_batch * num_cond
                
                # Batched processing - prepare inputs by repeating interleaved
                # For each input sample, we'll create num_cond versions in sequence
                
                # Handle x: reshape as [total_batch, seq_len, feat_dim]
                x_batched = x.repeat_interleave(num_cond, dim=0)
                
                # Handle speaker ID: reshape as [total_batch]
                spk_batched = spk_id.repeat_interleave(num_cond, dim=0)
                
                # Handle f0 and rms: reshape as [total_batch, seq_len]
                f0_batched = f0.repeat_interleave(num_cond, dim=0)
                rms_batched = rms.repeat_interleave(num_cond, dim=0)
                
                # Create batched cvec, handling bad_cvec if needed
                if ds_cfg_strength > 1e-5 and spk_cfg_strength > 1e-5:
                    # Need to create interleaved: [cvec, bad_cvec, cvec] for each original batch item
                    cvec_expanded = []
                    for i in range(orig_batch):
                        cvec_expanded.append(cvec[i:i+1])  # Regular
                        cvec_expanded.append(bad_cvec[i:i+1])  # Bad cvec
                        cvec_expanded.append(cvec[i:i+1])  # Regular (for null spk)
                    cvec_batched = torch.cat(cvec_expanded, dim=0)
                elif ds_cfg_strength > 1e-5:
                    # Interleave: [cvec, bad_cvec] for each original batch item
                    cvec_list = []
                    for i in range(orig_batch):
                        cvec_list.append(cvec[i:i+1])
                        cvec_list.append(bad_cvec[i:i+1])
                    cvec_batched = torch.cat(cvec_list, dim=0)
                elif spk_cfg_strength > 1e-5:
                    # Interleave: [cvec, cvec] for each original batch item
                    cvec_batched = cvec.repeat_interleave(num_cond, dim=0)
                
                if isinstance(t, torch.Tensor) and t.ndim > 0:
                    t_batched = t.repeat_interleave(num_cond, dim=0)
                else:
                    t_batched = t  # It's a scalar, handled by the transformer
                
                # Handle mask if exists
                mask_batched = mask.repeat_interleave(num_cond, dim=0) if exists(mask) else None
                
                # Create drop_speaker flag tensor - only activate for the appropriate indices
                drop_speaker_batched = torch.zeros(total_batch, dtype=torch.bool, device=x.device)
                
                if spk_cfg_strength > 1e-5:
                    # Set drop_speaker=True for the third condition of each original batch item
                    if ds_cfg_strength > 1e-5:
                        # Pattern is [False, False, True] repeated
                        for i in range(orig_batch):
                            drop_speaker_batched[i*num_cond + 2] = True
                    else:
                        # Pattern is [False, True] repeated
                        for i in range(orig_batch):
                            drop_speaker_batched[i*num_cond + 1] = True
                
                # Single batched forward pass
                preds_batched = self.transformer(
                    x=x_batched,
                    spk=spk_batched,
                    f0=f0_batched,
                    rms=rms_batched,
                    cvec=cvec_batched,
                    time=t_batched,
                    mask=mask_batched,
                    drop_speaker=drop_speaker_batched
                )
                
                # Reshape and extract the predictions for each condition
                # First, reshape the predictions to [orig_batch, num_cond, seq_len, feat_dim]
                predictions = []
                
                # Extract predictions for each original batch item
                for b in range(orig_batch):
                    batch_predictions = []
                    for c in range(num_cond):
                        idx = b * num_cond + c
                        batch_predictions.append(preds_batched[idx:idx+1])
                    predictions.append(batch_predictions)
                
                # Apply classifier-free guidance per original batch item
                pred_results = []
                for b in range(orig_batch):
                    pred = predictions[b][0]  # Regular prediction
                    
                    cond_idx = 1
                    if ds_cfg_strength > 1e-5:
                        bad_cvec_pred = predictions[b][cond_idx]
                        pred = pred + (pred - bad_cvec_pred) * ds_cfg_strength
                        cond_idx += 1
                    
                    if spk_cfg_strength > 1e-5:
                        null_spk_pred = predictions[b][cond_idx]
                        pred = pred + (pred - null_spk_pred) * spk_cfg_strength
                    
                    pred_results.append(pred)
                
                # Combine back to original batch dimension
                pred = torch.cat(pred_results, dim=0)

            cfg_flag = (ds_cfg_strength > 1e-5) or (skip_cfg_strength > 1e-5) or (spk_cfg_strength > 1e-5)
            if cfg_rescale > 1e-5 and cfg_flag:
                std_pred = pred.std()

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
