import os
import dataclasses
import json
from pathlib import Path
from typing import Tuple, cast

import torch
import cv2
import numpy as np
import numpy.typing as npt
from einops import rearrange

class ObjaverseLoader:
    def __init__(self,
                 base_path):

        self.fps = 1
        self.H = 252
        self.W = 252

        self.base_path = base_path
        with open(os.path.join(base_path, "transforms.json"), 'r') as f:
            self.metadata = json.load(f)
        self.K = torch.tensor([
            [self.metadata["fl_x"], 0, self.metadata["cx"]],
            [0, self.metadata["fl_y"], self.metadata["cx"]],
            [0, 0, 1]
        ])
        self._num_frames = len(self.metadata["frames"])
    
    def num_frames(self) -> int:
        return self._num_frames

    def get_batch(self):

        rgb_ = []
        depth_ = []
        depth_mask_ = []
        c2w_ = []

        for index in range(self.num_frames()):
            file_name = self.metadata["frames"][index]["file_path"][:-4]
            depth_path = os.path.join(self.base_path, file_name + "_depth.png")
            rgb_path = os.path.join(self.base_path, file_name + ".png")
            max_depth = 10 
            # w2c colmap
            c2w_blender = np.array(self.metadata["frames"][index]["transform_matrix"])
            c2w_colmap = c2w_blender
            c2w_colmap[:, 1:3] = c2w_colmap[:, 1:3] * -1

            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            # rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[:,:,::-1]  # BGR转RGB
            
            # # 将RGB图像调整为252x252，使用双线性插值
            # rgb = cv2.resize(rgb, (self.H, self.H), interpolation=cv2.INTER_LINEAR)
            # /* 1. 带 alpha 读入：shape = (H, W, 4) */
            rgba = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)        # B G R A

            b, g, r, a = cv2.split(rgba)                         # 各为 uint8

            # /* 2. 归一化 α，准备做加权 */
            alpha = a.astype(np.float32) / 255.0                 # 0~1
            alpha = cv2.merge([alpha, alpha, alpha])             # (H,W,3)

            # /* 3. 把前景叠到白底上：white*(1-alpha) + rgb*alpha */
            rgb = cv2.merge([r, g, b]).astype(np.float32)        # 变成 RGB
            white_bg = np.ones_like(rgb) * 255                   # (H,W,3)

            rgb_w = rgb * alpha + white_bg * (1 - alpha)
            rgb_w = rgb_w.astype(np.uint8)                       # 回到 uint8

            # /* 4. 想要 252×252 就再 resize */
            # target = 252
            rgb_w = cv2.resize(rgb_w, (self.H, self.W),
                            interpolation=cv2.INTER_AREA)     # (width, height)

            # /* 5. 后续转 tensor / transform 时再做 BGR→RGB */
            rgb = cv2.cvtColor(rgb_w, cv2.COLOR_BGR2RGB)            
            # 将深度图调整为252x252，使用最近邻插值
            depth_img = cv2.resize(depth_img, (self.H, self.H), interpolation=cv2.INTER_NEAREST)

            depth = depth_img.astype(float) / 65535.0 * max_depth

            # 从深度图生成mask，使用相对阈值
            depth_mask = (depth > 0.01) & (depth < max_depth)  # 添加一个小的最小阈值避免噪声

            rgb_.append(torch.from_numpy(rgb))
            depth_.append(torch.from_numpy(depth))
            depth_mask_.append(torch.from_numpy(depth_mask))
            c2w_.append(torch.from_numpy(c2w_colmap))
        
        return {
            "rgb": torch.stack(rgb_).float(),
            "depth": torch.stack(depth_),
            "depth_mask": torch.stack(depth_mask_),
            "c2w": torch.stack(c2w_)
        }

    def get_point_cloud(self, depth, mask, c2w, rgb):
        """
        参数:
        depth: [B, H, W] 深度图
        mask: [B, H, W] 深度mask
        c2w: [B, 4, 4] 相机到世界坐标系的变换矩阵
        rgb参数: [B, H, W, 3]
        
        返回:
        points: [N, 3] 世界坐标系中的点云
        colors: [N, 3] 对应的颜色
        """
        B = depth.shape[0]
        device = depth.device

        # 创建像素坐标网格 [H, W, 2]
        y, x = torch.meshgrid(
            torch.arange(self.H, device=device),
            torch.arange(self.W, device=device),
            indexing='ij'
        )
        # [H, W, 3]
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
        
        # [B, H, W, 3]
        pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1)
        
        # [B, 3, H*W]
        pixels = pixels.reshape(B, self.H * self.W, 3).transpose(1, 2)
        
        # [B, 3, 3] @ [B, 3, H*W] = [B, 3, H*W]
        K_inv = torch.inverse(self.K.unsqueeze(0).expand(B, -1, -1).to(device))
        rays = K_inv @ pixels
        
        # [B, H*W] -> [B, 1, H*W]
        depth_flat = depth.reshape(B, -1).unsqueeze(1)
        
        # [B, 3, H*W]
        points_cam = rays * depth_flat
        
        # 转换到世界坐标系 [B, 4, 4] @ [B, 4, H*W]
        points_homo = torch.cat([points_cam, 
                               torch.ones_like(points_cam[:, :1, :])], dim=1)
        points_world = c2w @ points_homo
        points_world = points_world[:, :3, :].transpose(1, 2)  # [B, H*W, 3]
        
        # 使用mask过滤点
        mask_flat = mask.reshape(B, -1)  # [B, H*W]
        
        # 收集所有有效点和颜色
        valid_points = []
        valid_colors = []
        rgb_flat = rgb.reshape(B, -1, 3)  # [B, H*W, 3]
        
        for b in range(B):
            valid_points.append(points_world[b, mask_flat[b]])
            valid_colors.append(rgb_flat[b, mask_flat[b]])
        
        points_all = torch.cat(valid_points, dim=0)
        colors_all = torch.cat(valid_colors, dim=0)
        
        return points_all.cpu().numpy(), colors_all.cpu().long().numpy()

    def get_feature_point_cloud(self, depth, mask, c2w, feature):
        """
        参数:
        depth: [B, H, W] 深度图
        mask: [B, H, W] 深度mask
        c2w: [B, 4, 4] 相机到世界坐标系的变换矩阵
        rgb参数: [B, H, W, C]
        
        返回:
        points: [N, 3] 世界坐标系中的点云
        colors: [N, 3] 对应的颜色
        """
        B = depth.shape[0]
        device = depth.device

        # 创建像素坐标网格 [H, W, 2]
        y, x = torch.meshgrid(
            torch.arange(self.H, device=device),
            torch.arange(self.W, device=device),
            indexing='ij'
        )
        # [H, W, 3]
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
        
        # [B, H, W, 3]
        pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1)
        
        # [B, 3, H*W]
        pixels = pixels.reshape(B, self.H * self.W, 3).transpose(1, 2)
        
        # [B, 3, 3] @ [B, 3, H*W] = [B, 3, H*W]
        K_inv = torch.inverse(self.K.unsqueeze(0).expand(B, -1, -1).to(device))
        rays = K_inv @ pixels
        
        # [B, H*W] -> [B, 1, H*W]
        depth_flat = depth.reshape(B, -1).unsqueeze(1)
        
        # [B, 3, H*W]
        points_cam = rays * depth_flat
        
        # 转换到世界坐标系 [B, 4, 4] @ [B, 4, H*W]
        points_homo = torch.cat([points_cam, 
                               torch.ones_like(points_cam[:, :1, :])], dim=1)
        points_world = c2w @ points_homo
        points_world = points_world[:, :3, :].transpose(1, 2)  # [B, H*W, 3]
        
        # 使用mask过滤点
        mask_flat = mask.reshape(B, -1)  # [B, H*W]
        
        # 收集所有有效点和颜色
        valid_points = []
        valid_colors = []
        rgb_flat = rearrange(feature, "b h w c -> b (h w) c")
        
        for b in range(B):
            valid_points.append(points_world[b, mask_flat[b]])
            valid_colors.append(rgb_flat[b, mask_flat[b]])
        
        points_all = torch.cat(valid_points, dim=0)
        colors_all = torch.cat(valid_colors, dim=0)
        
        return points_all.cpu(), colors_all.cpu()


if __name__ == "__main__":

    loader = ObjaverseLoader("/home/jiangyun/documents/Sp2Sl/data/render_outputs/objaverse/renders/0a0a8274693445a6b533dce7f97f747c")

    data = loader.get_batch()

    point, color = loader.get_point_cloud(data["depth"], data["depth_mask"], data["c2w"], data["rgb"])

    import time
    import viser

    server = viser.ViserServer()

    server.scene.add_point_cloud(
                name=f"point_cloud",
                points=point,
                colors=color,
                point_shape="rounded",
            )

    while True:
        time.sleep(1.0)

    print()
