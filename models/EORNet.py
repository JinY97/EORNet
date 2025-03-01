'''
Author: Yin Jin
Date: 2024-05-06 14:45:04
LastEditors: Yin Jin
LastEditTime: 2025-03-01 13:17:40
'''
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from mamba_ssm import Mamba
from timm.models.layers import DropPath

from einops import rearrange
from einops.layers.torch import Rearrange

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

class PatchEmbedding(nn.Module):
    def __init__(self, chan=1, emb_size=32):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(chan, emb_size, 7, stride=1, padding=3), 
        )

    def forward(self, x):
        x = self.projection(x)
        return x
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))
        
class MambaBlock(nn.Sequential):
    def __init__(self, emb_size, t_size, e_size):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(t_size),
                Mamba(d_model=t_size, # Model dimension d_model
                  d_state=32,  # SSM state expansion factor
                  d_conv=3,    # Local convolution width，
                  expand=e_size,
                  device="cuda"),    # Block expansion factor
                nn.Linear(t_size, t_size),
                DropPath(0.1))
            ),
        )

class MambaBlockEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, t_size, e_size):
        super().__init__(*[MambaBlock(emb_size, t_size, e_size) for _ in range(depth)])
    
class OutputHead(nn.Sequential):
    def __init__(self, emb_size, chan):
        super().__init__()
        self.outputhead = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, 9, stride=1, padding="same"),
            nn.Conv1d(emb_size, chan, 9, stride=1, padding="same")
        )
    
    def forward(self, x):
        out = self.outputhead(x)
        return out

class FeatureExtractor(nn.Sequential):
    def __init__(self, data_num=512, emb_size=32, depth=4, chan=19, e_size=2, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.embedding = PatchEmbedding(chan, emb_size)
        self.MEncoder = MambaBlockEncoder(depth, emb_size, data_num, e_size)
        self.out = OutputHead(emb_size, chan)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.MEncoder(x)
        out = self.out(x)

        return out
        
if __name__ == '__main__':
    x = torch.rand((8, 1, 500)).to("cuda")
    model = FeatureExtractor(emb_size=64, data_num=500, chan=1).to("cuda")
    y = model(x)
    print(y.shape)