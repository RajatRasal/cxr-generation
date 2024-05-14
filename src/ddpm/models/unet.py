from functools import partial
from typing import List

import torch
from torch import nn

from ddpm.models.components import SinusoidalPosEmb, ResnetBlock, Residual, PreNorm, CrossAttention, Downsample, Upsample


class Unet(nn.Module):
    
    def __init__(
        self,
        dim: int,
        dim_mults: List[int] = [1, 2, 4, 8],
        channels: int = 1,
        resnet_block_groups: int = 8,
        sinusoidal_pos_emb_theta: int = 10000,
        context_dim: int = 2,
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
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        self.init_conv = nn.Conv1d(channels, dim, 7, padding=3)
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, dim_head=32, heads=4, context_dim=context_dim))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, dim_head=32, heads=4, context_dim=context_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, dim_head=32, heads=4, context_dim=context_dim))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, out_dim, 1)

    def forward(self, x: torch.FloatTensor, time: int, cond: torch.FloatTensor) -> torch.FloatTensor:
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        for block, attn, downsample in self.downs:
            x = block(x, t)
            x = attn(x, cond)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, cond)
        x = self.mid_block2(x, t)
        
        for block, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t)
            x = attn(x, cond)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        return x
