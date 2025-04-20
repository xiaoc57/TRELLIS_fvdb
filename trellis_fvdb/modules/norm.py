import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x


import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor

class LayerNorm(nn.LayerNorm):
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.data.jdata.size(1)
        num_batches = input.grid.grid_count

        flat_data, flat_offsets = input.data.jdata, input.data.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[0 + b]:flat_offsets[1 + b]]
            if feat.size(0) != 0:
                feat_dtype = feat.dtype
                feat = feat.reshape(1, -1, num_channels)
                feat = super().forward(feat.float())
                feat = feat.reshape(-1, num_channels)

                result_data[flat_offsets[0 + b]:flat_offsets[1 + b]] = feat.to(feat_dtype)
        return VDBTensor(input.grid, input.data.jagged_like(result_data), kmap=input.kmap)