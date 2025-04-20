from typing import *
import math
import torch

import fvdb
import fvdb.nn as fvnn

# use xformers
import xformers.ops as xops

def calc_window_partition(
    tensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """
    DIM = tensor.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32) // OFFSET[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices

def sparse_windowed_scaled_dot_product_self_attention(
    qkv: fvnn.VDBTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> fvnn.VDBTensor:
    fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition(
        torch.cat([qkv.grid.jidx.unsqueeze(1), qkv.grid.ijk.jdata], dim=-1), window_size, shift_window=shift_window
    )

    qkv_feats = qkv.data.jdata[fwd_indices]
    q, k, v = qkv_feats.unbind(dim=1)                       # [M, H, C]
    q = q.unsqueeze(0)                                      # [1, M, H, C]
    k = k.unsqueeze(0)                                      # [1, M, H, C]
    v = v.unsqueeze(0)                                      # [1, M, H, C]
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
    out = xops.memory_efficient_attention(q, k, v, mask)[0] # [M, H, C]
    out = out[bwd_indices]
    out = out.reshape(qkv_feats.shape[0], -1)
    return fvnn.VDBTensor(qkv.grid, qkv.data.jagged_like(out), qkv.kmap)
 


