import argparse
import torch
import utils3d
import numpy as np
import sys
import os

from einops import repeat
from easydict import EasyDict as edict
from fvdb import GridBatch, JaggedTensor
from torch_scatter import scatter_mean

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from visual_feature_aggregation.visualization_pc._shapenet_feature import ShapenetLoader


def load_voxel(path):
    data = torch.load(path)
    # fixed resolution 64
    indices = data["indices"].long()
    positions = data["positions"]
    assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
    return {
        "positions": positions,
        "indices": indices
    }

def get_rays(K, height, width, T_world_camera):
    """获取相机原点和射线方向
    Returns:
        Tuple[np.ndarray]: (ray_origins, ray_directions)
        ray_origins: (H*W, 3) 相机原点
        ray_directions: (H*W, 3) 射线方向（已归一化）
    """
    
    # 创建像素坐标网格
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3)
    
    # 计算相机坐标系下的射线方向
    pixels = pixels.reshape(-1, 3).T  # (3, H*W)
    rays_d = np.linalg.inv(K) @ pixels  # (3, H*W)
    rays_d = rays_d.T  # (H*W, 3)
    
    # 将射线方向转换到世界坐标系
    R = T_world_camera[0:3, 0:3]  # (3, 3)
    rays_d = (R @ rays_d.T).T  # (H*W, 3)
    
    # 归一化射线方向
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    
    # 获取相机原点（在世界坐标系中）
    rays_o = T_world_camera[0:3, 3]  # (3,)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)  # (H*W, 3)
    
    return rays_o, rays_d


def create_rays_from_intrinsic_torch_batch(pose_matric, intrinsic):
    """
    Args:
        pose_matric: (B, 4, 4)
        intrinsic: (B, 6), [fx, fy, cx, cy, w, h]
    Returns:
        camera_origin: (B, 3)
        d: (B, H, W, 3)
    """
    camera_origin = pose_matric[:, :3, 3] # (B, 3)
    fx, fy, cx, cy, w, h = intrinsic.unbind(1) # [B,]
    w, h = int(w[0]), int(h[0])
    # attention, indexing is 'xy'
    ii, jj = torch.meshgrid(torch.arange(w).to(intrinsic.device), torch.arange(h).to(intrinsic.device), indexing='xy') 

    ii = ii[None].repeat(pose_matric.shape[0], 1, 1) # (B, H, W)
    jj = jj[None].repeat(pose_matric.shape[0], 1, 1) # (B, H, W)

    uu, vv = (ii - cx[:, None, None]) / fx[:, None, None], (jj - cy[:, None, None]) / fy[:, None, None]
    local_xyz = torch.stack([uu, vv, torch.ones_like(uu, device=uu.device)], dim=-1) # (B, H, W, 3)
    local_xyz = torch.cat([local_xyz, torch.ones((local_xyz.shape[0], int(h), int(w), 1)).to(local_xyz)], axis=-1)
    pixel_xyz = torch.einsum('bij, bhwj->bhwi', pose_matric, local_xyz)[:, :, :, :3] # (B, H, W, 3) # ! fix error

    d = (pixel_xyz - camera_origin[:, None, None, :])  # (B, H, W, 3)
    # normalize the direction
    d = d / torch.norm(d, dim=-1, keepdim=True) # (B, H, W, 3)

    return camera_origin, d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/home/jiangyun/documents/trellis_fvdb/assets/example_data/02691156-1a04e3eab45ca15dd86060f189eb133")
    parser.add_argument('--voxel_path', type=str, default="/home/jiangyun/documents/trellis_fvdb/assets/example_data/02691156-1a04e3eab45ca15dd86060f189eb133/02691156-1a04e3eab45ca15dd86060f189eb133.pkl")
    parser.add_argument('--fvdb_save_path', type=str, default="./result.pkl")
    opt = parser.parse_args()
    opt = edict(vars(opt))

    voxel_data = load_voxel(opt.voxel_path)
    
    positions = voxel_data["positions"].float().cuda()
    indices = voxel_data["indices"]

    batch_size = 36
    loader = ShapenetLoader(opt.image_path)
    data = loader.get_batch()

    n_views = data["rgb"].shape[0]

    H = loader.H
    W = loader.W

    uv_lst = []
    for i in range(0, n_views, batch_size):
        batch_extrinsics = data['extrinsics'][i:i+batch_size].cuda()
        batch_intrinsics = data['intrinsics'][i:i+batch_size].cuda()
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        uv_lst.append(uv)

    patchtokens = data["rgb"].float()
    uv = torch.cat(uv_lst, dim=0)
    
    device = "cuda"

    # fixed resolution 64
    vox_size = 1 / 64
    vox_origin = (-0.5 + (vox_size / 2), -0.5 + (vox_size / 2), -0.5 + (vox_size / 2))
    grid = GridBatch(device=device)
    grid.set_from_ijk(indices.cuda(), voxel_sizes=vox_size, origins=vox_origin)

    K = data["intrinsics"][0]
    K[0, :] = K[0, :] * loader.W
    K[1, :] = K[1, :] * loader.H

    nimg_origins, nimg_directions = create_rays_from_intrinsic_torch_batch(
        torch.linalg.inv(data["extrinsics"]).cuda(),
        repeat(torch.tensor([K[0][0], K[1][1], K[0][2], K[1][2], loader.W, loader.H], device="cuda"), "c -> b c", b=data["rgb"].shape[0])
    )
    
    n_imgs = data["rgb"].shape[0]

    nimg_origins = nimg_origins.view(n_imgs, 1, 1, 3).expand(-1, H, W, -1).reshape(-1, 3)
    nimg_directions = nimg_directions.reshape(-1, 3)

    out_voxel_ids, ray_start_end = grid.voxels_along_rays(JaggedTensor([nimg_origins]), 
                                                                JaggedTensor([nimg_directions]), 
                                                                max_voxels=1, 
                                                                return_ijk=False)

    nimg_features = patchtokens.permute(0, 2, 3, 1).contiguous().view(n_imgs * H * W, -1) # N, C, H, W -> N, H, W, C -> N * H * W, C
    
    effective_feature_mask = data["mask"].view(n_imgs * H * W) # N, H, W, 1 -> N * H * W

    mask = (ray_start_end.joffsets[1:] - ray_start_end.joffsets[:-1]).bool().cpu() # [N_ray]
    pixel_feature = nimg_features[mask, :] # [N_ray_hit, C]
    out_voxel_ids = out_voxel_ids.jdata.to(torch.int64).cpu()
    effective_feature_mask = effective_feature_mask[mask]

    qshape = nimg_features.shape[1]

    # if any effective_feature_mask has 0 value
    if (effective_feature_mask == 0).any():
        pixel_feature = pixel_feature[effective_feature_mask > 0]
        out_voxel_ids = out_voxel_ids[effective_feature_mask > 0]
        
    out_voxel_features = torch.zeros((grid.total_voxels, qshape), device="cpu")
    out_voxel_features = scatter_mean(pixel_feature, out_voxel_ids, out=out_voxel_features, dim=0)

    # 保存前先转到CPU
    save_dict = {
        "indices": indices.cpu(),
        "feature": out_voxel_features.cpu()
    }
    
    torch.save(save_dict, opt.fvdb_save_path)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
