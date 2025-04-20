from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor

from ..attention import SparseMultiHeadAttention, SerializeMode


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            fvnn.Linear(channels, int(channels * mlp_ratio)),
            fvnn.GELU(approximate="tanh"),
            fvnn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: VDBTensor) -> VDBTensor:
        return self.mlp(x)

class LayerNorm(nn.LayerNorm):
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.data.jdata.size(1)
        num_batches = input.grid.grid_count

        flat_data, flat_offsets = input.data.jdata, input.data.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[0 + b]:flat_offsets[1 + b]]
            if feat.size(0) != 0:
                feat = feat.reshape(1, -1, num_channels)
                feat = super().forward(feat)
                feat = feat.reshape(-1, num_channels)

                result_data[flat_offsets[0 + b]:flat_offsets[1 + b]] = feat

        return VDBTensor(input.grid, input.data.jagged_like(result_data), kmap=input.kmap)

class SparseTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
    
    def _forward(self, x: VDBTensor) -> VDBTensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = h + x
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: VDBTensor) -> VDBTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
        



