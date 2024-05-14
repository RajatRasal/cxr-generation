import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from ddpm.models.utils import default, exists


class Upsample(nn.Module):

    def __init__(self, dim: int, dim_out: Optional[int] = None, scale_factor: int = 2, kernel_size: int = 3, padding: int = 1):
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv1d(dim, default(dim_out, dim), kernel_size=kernel_size, padding=padding)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    
    def __init__(self, dim: int, dim_out: Optional[int] = None, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        self.conv = nn.Conv1d(dim, default(dim, dim_out), kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.conv(x)


class SinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    
    def __init__(self, dim: int, dim_out: int, groups: int = 8, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.FloatTensor,
        scale_shift: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    ) -> torch.FloatTensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.FloatTensor, time_emb: torch.FloatTensor = None) -> torch.FloatTensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim = 1)
            
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        h = h + self.res_conv(x)
        return h


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, context_dim = 512):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.to_kv = nn.Conv1d(context_dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, n = x.shape
        if context is None:
            q, k, v = self.to_qkv(x).chunk(3, dim=1)
        else:
            q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), (q, k, v))

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)
