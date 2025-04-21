import os
import json
import torch
import imageio
import numpy as np

from easydict import EasyDict as edict
from safetensors.torch import load_file

from trellis_fvdb.models.structured_latent_vae import SLatEncoder, SLatGaussianDecoder
from trellis_fvdb.utils import render_utils

import fvdb
import fvdb.nn as fvnn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.allow_tf32 = False
# # in PyTorch 1.12 and later.
# torch.backends.cuda.matmul.allow_tf32 = False

if __name__ == "__main__":

    
    cfg_path = "ckpts/slat_enc_swin8_B_64l8_fp16.json"
    ckpt_path = "ckpts/slat_enc_swin8_B_64l8_fp16.safetensors"
    cfg = edict(json.load(open(cfg_path, 'r')))
    ckpt = load_file(ckpt_path)

    cfg_path_gs = "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json"
    ckpt_path_gs = "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
    cfg_gs = edict(json.load(open(cfg_path_gs, 'r')))
    ckpt_gs = load_file(ckpt_path_gs)

    model = SLatEncoder(**cfg["args"])
    model.load_state_dict(ckpt)
    model = model.to("cuda")

    model_gs = SLatGaussianDecoder(**cfg_gs["args"])
    model_gs.load_state_dict(ckpt_gs)
    model_gs = model_gs.to("cuda")

    # feature = np.load("assets/059a7936ed89419ba9eae3153753ae86.npz")
    # latent_t = np.load("example_data/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10.npz")
    # latent_t = torch.from_numpy(latent_t['feats']).float().cuda()
    
    feature = torch.load("assets/0a0af057d1b8488c80b311cb196659d0.pkl")

    ijks = feature['indices'].int().cuda()
    assert torch.all(ijks >= 0) and torch.all(ijks < 64), "Some vertices are out of bounds"
    
    vox_size = 1 / 64
    vox_origin = (-0.5 + (vox_size / 2), -0.5+ (vox_size / 2), -0.5+ (vox_size / 2))

    # features = torch.from_numpy(feature['patchtokens']).float().cuda()
    features = feature["feature"].float().cuda()
    grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijks), voxel_sizes=vox_size)
    features = features[grid[0].ijk_to_inv_index(ijks).jdata]

    f = fvdb.JaggedTensor(features)

    # f = grid.jagged_like(features)
    x = fvnn.VDBTensor(grid, f)
    # x2 = fvnn.VDBTensor(grid, f)
    # x = x + x2

    # fix bug
    # feature and grid 对应
    # grid.ijk_to_index(ijks)

    z = model(x, sample_posterior=False)

    assert torch.isfinite(z.data.jdata).all(), "Non-finite latent"

    pack = {
        "feats": z.data.jdata.detach().clone().cpu().numpy().astype(np.float32),
        "coords": z.grid.ijk.jdata.detach().clone().cpu().numpy().astype(np.uint8),
    }

    outputs = model_gs(z)

    video = render_utils.render_video(outputs[0], bg_color=(1, 1, 1))['color']
    imageio.mimsave("sample_gs.mp4", video, fps=30)