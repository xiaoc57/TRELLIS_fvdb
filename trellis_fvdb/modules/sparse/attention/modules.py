from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import fvdb
import fvdb.nn as fvnn

from .serialized_attn import SerializeMode
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[fvnn.VDBTensor, torch.Tensor]) -> Union[fvnn.VDBTensor, torch.Tensor]:
        x_type = x.dtype
        if isinstance(x, fvnn.VDBTensor):
            # x = x.replace(F.normalize(x.feats, dim=-1))
            x_data = x.data.jdata.float()
            x = fvnn.VDBTensor(x.grid, x.data.jagged_like(F.normalize(x_data, dim=-1)), x.kmap)
        else:
            x = x.float()
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)



class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()        
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        
        if self._type == "self":
            self.to_qkv = fvnn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = fvnn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)

        self.to_out = fvnn.Linear(channels, channels)

        if use_rope:
            raise NotImplementedError("must be implement")

    def _fuse_pre(self, x: Union[fvnn.VDBTensor, torch.Tensor], num_fused: int) -> Union[fvnn.VDBTensor, torch.Tensor]:
        if isinstance(x, fvnn.VDBTensor):
            x_feats = x.data.jdata
            x_feats = x_feats.reshape(*x_feats.shape[:1], num_fused, self.num_heads, -1)
            return fvnn.VDBTensor(x.grid, x.data.jagged_like(x_feats))
        else:
            x_feats = x
            x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
            # torch.Size([1, 1374, 2, 16, 64])
            return x_feats

    def forward(self, x: fvnn.VDBTensor, context: torch.Tensor = None) -> fvnn.VDBTensor:
        if self._type == "self":
            qkv = self.to_qkv(x)
            n = qkv.data.jdata.shape[0]
            qkv = self._fuse_pre(qkv, num_fused=3)
            if self.use_rope:
                raise NotImplementedError("")
            if self.qk_rms_norm:
                q, k, v = qkv.data.jdata.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = fvnn.VDBTensor(qkv.grid, qkv.data.jagged_like(torch.stack([q, k, v], dim=1)))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(qkv, self.window_size, shift_window=self.shift_window)
            else:
                raise NotImplementedError("")
            h = fvnn.VDBTensor(h.grid, h.data.jagged_like(h.data.jdata.reshape(n, -1)))
            h = self.to_out(h)
            return h
        else:
            q = self.to_q(x)
            n = q.data.jdata.shape[0]
            q = fvnn.VDBTensor(q.grid, q.data.jagged_like(q.data.jdata.reshape(n, self.num_heads, -1)))
            kv = self.to_kv(context)
            kv = self._fuse_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = torch.stack([k, v], dim=1)
            h = sparse_scaled_dot_product_attention(q, kv)
            h = fvnn.VDBTensor(h.grid, h.data.jagged_like(h.data.jdata.reshape(n, -1)))
            h = self.to_out(h)
            return h