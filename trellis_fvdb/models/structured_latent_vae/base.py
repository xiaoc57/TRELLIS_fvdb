from typing import *
import torch
import torch.nn as nn
from ...modules.utils import convert_module_to_f16
# from ...modules import sparse as sp
# from ...modules.transformer import AbsolutePositionEmbedder
from ...modules.sparse.transformer import SparseTransformerBlock
from ...modules.transformer import AbsolutePositionEmbedder

import fvdb
import fvdb.nn as fvnn

def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        # if self.attn_mode == "shift_window":
        #     yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        # elif self.attn_mode == "shift_sequence":
        #     yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        # elif self.attn_mode == "shift_order":
        #     yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        if self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        # self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.input_layer = fvnn.Linear(in_channels, model_channels, bias=True)
        # self.input_layer = nn.Linear(in_channels, model_channels, bias=True)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    # def convert_to_fp32(self) -> None:
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: fvnn.VDBTensor) -> fvnn.VDBTensor:
        
        # w = torch.load("w.pt", weights_only=True)
        # b = torch.load("b.pt", weights_only=True)

        # tmp = x.data.jdata.cpu() @ w.T + b
        # atol=1e-4, rtol=1e-3
        h = self.input_layer(x)

        # true
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.grid.ijk.jdata.detach().clone())
        
        h = h.type(self.dtype)
        

        # import numpy as np

        # feature = np.load("feature_0a6e1a80d2e34d5981d6b2b440bbc8cd.npz")
        # latent_t = np.load("latent_0a6e1a80d2e34d5981d6b2b440bbc8cd.npz")
        # latent_t = torch.from_numpy(latent_t['feats']).float().cuda()
        
        # ijks = torch.from_numpy(feature['indices']).int().cuda()

        # i1 = self.pos_embedder(ijks)
        # i2 = self.pos_embedder(h.grid.ijk.jdata[h.grid.ijk_to_index(ijks).jdata])

        # t2 = i1 + h.data.jdata[h.grid.ijk_to_index(ijks).jdata]
        # t2 = t2.to(self.dtype)
        # h.data = h.grid.jagged_like(t2)

        # if self.pe_mode == "ape":
        #     inp = h.grid.jdata
        #     k = self.pos_embedder(fvnn.VDBTensor())
        #     # k = self.pos_embedder(h)
        #     h = h + self.pos_embedder(h)
        # h = h.type(self.dtype)


        for block in self.blocks:
            h = block(h)
        return h
