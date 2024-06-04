from functools import partial
from typing import Any, Dict, List

import torch
from torch import nn

from ddpm.models.two_dimensions.components import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb, ResnetBlock, CrossAttention, Downsample, Upsample
from ddpm.models.utils import cast_tuple, divisible_by


class Unet(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_mults: List[int] = [1, 2, 4, 8],
        channels: int = 3,
        resnet_block_groups: int = 8,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        context_dim: int = 1,
        # full_attn: bool = True,
        flash_attn: bool = False,
    ):
        super().__init__()

        # dimensions
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        out_dim = channels
        mid_dim = dims[-1]
        num_resolutions = len(in_out)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention
        # if not full_attn:
        #     full_attn = (*((False,) * (len(dim_mults) - 1)), True)
        num_stages = len(dim_mults)
        # full_attn  = cast_tuple(full_attn, num_stages)
        full_attn  = cast_tuple(True, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        FullAttention = partial(CrossAttention, flash=flash_attn, context_dim=context_dim)
        assert len(full_attn) == len(dim_mults)

        # layers
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.init_conv = nn.Conv2d(channels, dim, kernel_size=7, padding=3)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = FullAttention  # if layer_full_attn else LinearAttention
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention  # if layer_full_attn else LinearAttention
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, out_dim, kernel_size=1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    # TODO: Change cross_attn_kwargs to a ConfigDict
    # TODO: in diffusion.py time is a LongTensor
    def forward(
        self,
        x: torch.FloatTensor,
        time: int,
        cond: torch.FloatTensor,
        cross_attn_kwargs: Dict[str, Any],
    ) -> torch.FloatTensor:
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x, cond, **cross_attn_kwargs) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, cond, **cross_attn_kwargs) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x, cond, **cross_attn_kwargs) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        return x
