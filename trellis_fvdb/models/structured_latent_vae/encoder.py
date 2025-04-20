from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ...modules import sparse as sp
from .base import SparseTransformerBase


import fvdb
import fvdb.nn as fvnn


class SLatEncoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode = "full", 
        window_size = None, 
        pe_mode = "ape", 
        use_fp16 = False, 
        use_checkpoint = False, 
        qk_rms_norm = False
    ):
        super().__init__(
            in_channels, 
            model_channels, 
            num_blocks, 
            num_heads, 
            num_head_channels, 
            mlp_ratio, 
            attn_mode, 
            window_size, 
            pe_mode, 
            use_fp16, 
            use_checkpoint, 
            qk_rms_norm
        )
        self.resolution = resolution
        self.out_layer = fvnn.Linear(model_channels, 2 * latent_channels)
        
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: fvnn.VDBTensor, sample_posterior=True, return_raw=False):
        h = super().forward(x)
        h = fvnn.VDBTensor(h.grid, h.data.jagged_like(F.layer_norm(h.data.jdata.to(x.data.jdata.dtype), h.data.jdata.shape[-1:])), h.kmap)
        h = self.out_layer(h)
        return self.vdbchunk(h, sample_posterior, return_raw)


    def vdbchunk(self, input: fvnn.VDBTensor, sample_posterior, return_raw):
        num_channels = input.data.jdata.size(1)
        num_batches = input.grid.grid_count
        flat_data, flat_offsets = input.data.jdata, input.data.joffsets
        
        result_data = torch.empty_like(flat_data.chunk(2, dim=-1)[0])
        mean = torch.empty_like(result_data)
        logvar = torch.empty_like(result_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[0 + b]:flat_offsets[1 + b]]
            if feat.size(0) != 0:
                feat = feat.reshape(1, -1, num_channels)

                mt, lt = feat.chunk(2, dim=-1)
                if sample_posterior:
                    std = torch.exp(0.5 * lt)
                    z = mt + std * torch.randn_like(std)
                else:
                    z = mt

                mean[flat_offsets[0 + b]:flat_offsets[1 + b]] = mt
                logvar[flat_offsets[0 + b]:flat_offsets[1 + b]] = lt
                result_data[flat_offsets[0 + b]:flat_offsets[1 + b]] = z

        if return_raw:
            f = fvnn.VDBTensor(input.grid, input.data.jagged_like(result_data), input.kmap)
            m = fvnn.VDBTensor(input.grid, input.data.jagged_like(mean), input.kmap)
            l = fvnn.VDBTensor(input.grid, input.data.jagged_like(logvar), input.kmap)
            return f, m, l
        else:
            return fvnn.VDBTensor(input.grid, input.data.jagged_like(result_data), input.kmap)
