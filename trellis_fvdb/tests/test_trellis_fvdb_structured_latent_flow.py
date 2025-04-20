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

    feature = np.load("feature_0a6e1a80d2e34d5981d6b2b440bbc8cd.npz")
    latent_t = np.load("latent_0a6e1a80d2e34d5981d6b2b440bbc8cd.npz")
    latent_t = torch.from_numpy(latent_t['feats']).float().cuda()
    
    ijks = torch.from_numpy(feature['indices']).int().cuda()

    assert torch.all(ijks >= 0) and torch.all(ijks < 64), "Some vertices are out of bounds"
    
    vox_size = 1
    vox_origin = (0, 0, 0)
    # grid = fvdb.GridBatch(device="cuda")
    # grid.set_from_ijk(
    #     fvdb.JaggedTensor([ijks]), [0] * 3, [0] * 3, voxel_sizes=vox_size, origins=vox_origin)

    features = torch.from_numpy(feature['patchtokens']).float().cuda()
    grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor([ijks, ijks]), voxel_sizes=vox_size)


    features = features[grid[0].ijk_to_inv_index(ijks).jdata]
    f = fvdb.JaggedTensor([features, features])

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

    video = render_utils.render_video(outputs[1], bg_color=(1, 1, 1))['color']
    imageio.mimsave("sample_gs.mp4", video, fps=30)

    # save_path = os.path.join("latent.npz")
    # np.savez_compressed(save_path, **pack)

    print()