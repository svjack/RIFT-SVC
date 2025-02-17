import math
from typing import Union

from einops import repeat
from jaxtyping import Bool, Float, Int
import torch
from torch import nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding

from rift_svc.modules import (
    AdaLayerNormZero_Final,
    DiTBlock,
    MLP,
    TimestepEmbedding,
    ConvLinear,
)


# Conditional embedding for f0, rms, cvec
class CondEmbedding(nn.Module):
    def __init__(self, cvec_dim: int, cvec2_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.cvec2_dim = cvec2_dim
        self.cond_dim = cond_dim

        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        self.cvec2_embed = nn.Linear(cvec2_dim, cond_dim)
        self.cvec_conv = ConvLinear(cond_dim, cond_dim)
        self.cvec2_conv = ConvLinear(cond_dim, cond_dim)
        self.null_cvec2_embed = nn.Embedding(1, cond_dim)

        self.mlp = MLP(cond_dim, cond_dim, mult = 2)

        self.ln_cvec = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        self.ln_cvec2 = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6)
        self.ln = nn.LayerNorm(cond_dim, elementwise_affine=True, eps=1e-6)


    def forward(
            self,
            f0: Float[torch.Tensor, "b n"],
            rms: Float[torch.Tensor, "b n"],
            cvec: Float[torch.Tensor, "b n d"],
            cvec2: Float[torch.Tensor, "b n d2"],
            drop_cvec2: Bool[torch.Tensor, "b"],
        ):
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)

        f0_embed = self.f0_embed(f0 / 1200)
        rms_embed = self.rms_embed(rms)
        cvec_embed = self.ln_cvec(self.cvec_embed(cvec))
        cvec2_embed = self.ln_cvec2(self.cvec2_embed(cvec2))

        drop_cvec2 = drop_cvec2[:, None, None]
        b, n, d = cvec2_embed.shape
        cvec2_embed = torch.where(
            drop_cvec2,
            self.null_cvec2_embed.weight.expand(b, n, d),
            cvec2_embed
        )

        cond = f0_embed + rms_embed + self.cvec_conv(cvec_embed) + self.cvec2_conv(cvec2_embed)
        cond = self.mlp(cond) + cond
        return self.ln(cond)


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.mlp = MLP(out_dim, out_dim, mult = 2)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, x: Float[torch.Tensor, "b n d1"], cond_embed: Float[torch.Tensor, "b n d2"]):
        x = self.mel_embed(x)
        x = torch.cat((x, cond_embed), dim = -1)
        x = self.proj(x)
        x = self.mlp(x) + x
        x = self.ln(x)
        return x


# backbone using DiT blocks
class DiT(nn.Module):
    def __init__(self,
                 dim: int, depth: int, head_dim: int = 64, dropout: float = 0.0, ff_mult: int = 4,
                 mel_dim: int = 128, num_speaker: int = 1, cvec_dim: int = 768, cvec2_dim: int = 768,
                 init_std: float = 1):
        super().__init__()
    
        self.num_speaker = num_speaker
        self.spk_embed = nn.Embedding(num_speaker, dim)
        self.time_embed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, cvec2_dim, dim)   
        self.input_embed = InputEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(head_dim)

        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    head_dim = head_dim,
                    ff_mult = ff_mult,
                    dropout = dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.final_dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.output = nn.Linear(dim, mel_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.attn_norm.proj.weight, 0)
            torch.nn.init.constant_(block.attn_norm.proj.bias, 0)

        torch.nn.init.constant_(self.cond_embed.mlp.mlp_out.weight, 0)
        torch.nn.init.constant_(self.input_embed.mlp.mlp_out.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.bias, 0)
        torch.nn.init.constant_(self.output.weight, 0)
        torch.nn.init.constant_(self.output.bias, 0)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            # Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813).
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # weight shape: (out_channels, in_channels/groups, kernel_size)
            fan_out = module.weight.shape[0]  # out_channels
            fan_in = module.weight.shape[1] * module.weight.shape[2]  # (in_channels/groups) * kernel_size
            init_std = (self.init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std/math.sqrt(self.dim))


    def forward(
        self,
        x: Float[torch.Tensor, "b n d1`"],  # nosied input mel
        spk: Int[torch.Tensor, "b"],  # speaker
        f0: Float[torch.Tensor, "b n"],
        rms: Float[torch.Tensor, "b n"],
        cvec: Float[torch.Tensor, "b n d2"],
        cvec2: Float[torch.Tensor, "b n d3"],
        drop_cvec2: bool | Bool[torch.Tensor, "b"],
        time: Float[torch.Tensor, "b"] | Float[torch.Tensor, "b n"],  # time step
        mask: Bool[torch.Tensor, "b n"] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)

        t = self.time_embed(time)
        spk_embeds = self.spk_embed(spk)
        t = t + spk_embeds

        if isinstance(drop_cvec2, bool):
            drop_cvec2 = torch.full((batch,), drop_cvec2, device=x.device)

        cond_embed = self.cond_embed(f0, rms, cvec, cvec2, drop_cvec2)
        x = self.input_embed(x, cond_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask = mask, rope = rope)

        x = self.norm_out(x, t)
        x = x.permute(0, 2, 1)
        x = self.final_dwconv(x)
        x = x.permute(0, 2, 1)
        output = self.output(x)

        return output
