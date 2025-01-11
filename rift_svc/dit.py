import math
from typing import Union

from einops import repeat
from jaxtyping import Bool, Float, Int
import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from rift_svc.modules import (
    AdaLayerNormZero_Final,
    DiTBlock,
    MLP,
    TimestepEmbedding,
)


# Conditional embedding for f0, rms, cvec
class CondEmbedding(nn.Module):
    def __init__(self, cvec_dim: int, whisper_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.whisper_dim = whisper_dim
        self.cond_dim = cond_dim
        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)
        self.whisper_embed = nn.Linear(whisper_dim, cond_dim)
        self.mlp = MLP(cond_dim, cond_dim, mult = 2)

    def forward(
            self,
            f0: Float[torch.Tensor, "b n"],
            rms: Float[torch.Tensor, "b n"],
            cvec: Float[torch.Tensor, "b n d"],
            whisper: Float[torch.Tensor, "b n d2"]
        ):
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)

        cond = self.f0_embed(f0 / 1200) + self.rms_embed(rms) + self.cvec_embed(cvec) + self.whisper_embed(whisper)
        cond = self.mlp(cond) + cond
        return cond


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)
        self.mlp = MLP(out_dim, out_dim, mult = 2)

    def forward(self, x: Float[torch.Tensor, "b n d1"], cond_embed: Float[torch.Tensor, "b n d2"]):
        x = self.mel_embed(x)
        x = torch.cat((x, cond_embed), dim = -1)
        x = self.proj(x)
        x = self.mlp(x) + x
        return x


# backbone using DiT blocks
class DiT(nn.Module):
    def __init__(self,
                 dim: int, depth: int, head_dim: int = 64, dropout: float = 0.1, ff_mult: int = 4,
                 mel_dim: int = 128, num_speaker: int = 1, cvec_dim: int = 768, whisper_dim: int = 1280,
                 init_std: float = 1):
        super().__init__()

        self.num_speaker = num_speaker
        self.spk_embed = nn.Embedding(num_speaker, dim)
        #self.null_spk_embed = nn.Embedding(1, dim)
        self.null_whisper_embed = nn.Embedding(1, whisper_dim)
        self.time_embed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, whisper_dim, dim)
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
        whisper: Float[torch.Tensor, "b n d3"],
        time: Float[torch.Tensor, "b"] | Float[torch.Tensor, "b n"],  # time step
        #drop_spk: Bool[torch.Tensor, "b"],  # cfg for speaker
        drop_whisper: Union[bool, Bool[torch.Tensor, "b"]],  # cfg for whisper
        mask: Bool[torch.Tensor, "b n"] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)

        t = self.time_embed(time)

        # if drop_spk.ndim == 0:
        #     drop_spk = repeat(drop_spk, '-> b', b=batch)
        
        # # Expand null embedding to match batch size
        # null_spk = repeat(self.null_spk_embed.weight, '1 d -> b d', b=batch)
        # Get speaker embeddings for the batch
        spk_embeds = self.spk_embed(spk)
        # Select null or speaker embedding based on drop_spk
        # spk_embed = torch.where(
        #     drop_spk.unsqueeze(-1),  # [b, 1]
        #     null_spk,                # [b, d]
        #     spk_embeds               # [b, d]
        # )
        # the spk embed is added to the time embed
        t = t + spk_embeds

        if isinstance(drop_whisper, bool):
            drop_whisper = torch.full((batch,), drop_whisper, device=x.device)

        # Expand null whisper embedding to match batch size and sequence length
        null_whisper = repeat(self.null_whisper_embed.weight, '1 d -> b n d', b=batch, n=seq_len)
        # Select null or whisper embedding based on drop_whisper
        whisper = torch.where(
            drop_whisper.unsqueeze(-1).unsqueeze(-1),  # [b, 1, 1]
            null_whisper,                              # [b, n, d]
            whisper                            # [b, n, d]
        )

        cond_embed = self.cond_embed(f0, rms, cvec, whisper)
        x = self.input_embed(x, cond_embed)
        
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask = mask, rope = rope)

        x = self.norm_out(x, t)
        output = self.output(x)

        return output
