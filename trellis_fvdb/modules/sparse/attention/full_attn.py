from typing import *
import torch

import fvdb
import fvdb.nn as fvnn

# use xformers
import xformers.ops as xops

@overload
def sparse_scaled_dot_product_attention(qkv: fvnn.VDBTensor) -> fvnn.VDBTensor:
    ...

@overload
def sparse_scaled_dot_product_attention(q: fvnn.VDBTensor, kv: torch.Tensor) -> fvnn.VDBTensor:
    ...

def sparse_scaled_dot_product_attention(*args):

    num_all_args = len(args)

    if num_all_args == 1:
        qkv = args[0]
        device = qkv.device
        q_seqlen = [qkv.data.joffsets[1 + b] for b in range(qkv.grid.grid_count)]

        kv_seqlen = q_seqlen
        qkv_data = qkv.data.jdata

        q, k, v = qkv_data.unbind(dim=1)

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]

        return fvnn.VDBTensor(qkv.grid, qkv.data.jagged_like(out))
    
    elif num_all_args == 2:
        q = args[0]
        kv = args[1]

        device = q.device
        q_seqlen = [q.data.joffsets[1 + b] for b in range(q.grid.grid_count)]

        N, L, _, H, C = kv.shape
        kv_seqlen = [L] * N
        kv = kv.reshape(N*L, 2, H, C)
        k, v = kv.unbind(dim=1)

        q_data = q.data.jdata

        q_data = q_data.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q_data, k, v, mask)[0]

        return fvnn.VDBTensor(q.grid, q.data.jagged_like(out))