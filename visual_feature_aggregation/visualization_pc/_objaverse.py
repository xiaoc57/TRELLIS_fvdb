import os
import dataclasses
import json
from pathlib import Path
from typing import Tuple, cast

import cv2
import numpy as np
import numpy.typing as npt
from einops import rearrange


class ObjaverseLoader:
    def __init__(self,
                 base_path,
                 split: str="train",
                 fps: int=1):
        self.base_path = base_path
        self.fps = fps
        if split == "train":
            with open(os.path.join(base_path, "transforms.json"), 'r') as f:
                self.metadata = json.load(f)
        else:
            raise NotImplementedError()
        self._num_frames = len(self.metadata["frames"])
    
    def num_frames(self) -> int:
        return self._num_frames

    def get_frame(self, index: int):
        file_name = self.metadata["frames"][index]["file_path"][:-4]
        depth_path = os.path.join(self.base_path, file_name + "_depth.png")
        rgb_path = os.path.join(self.base_path, file_name + ".png")

        K = np.array([
            [self.metadata["fl_x"], 0, self.metadata["cx"]],
            [0, self.metadata["fl_y"], self.metadata["cx"]],
            [0, 0, 1]
        ])
        # fixed, https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/blender_script/render.py
        max_depth = 10 
        # w2c colmap
        c2w_blender = np.array(self.metadata["frames"][index]["transform_matrix"])
        c2w_colmap = c2w_blender
        c2w_colmap[:, 1:3] = c2w_colmap[:, 1:3] * -1
        # must 3 \times 3

        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[:,:,::-1]  # BGR转RGB

        depth = depth_img.astype(float) / 65535.0 * max_depth

        # 从深度图生成mask，使用相对阈值
        depth_mask = (depth > 0.01) & (depth < max_depth)  # 添加一个小的最小阈值避免噪声

        return ObjaverseFrame(
            K=K,
            rgb=rgb,
            depth=depth,
            mask=depth_mask,
            T_world_camera=c2w_colmap # c2w
        )

@dataclasses.dataclass
class ObjaverseFrame:
    K: npt.NDArray[np.float32]
    rgb: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    T_world_camera: npt.NDArray[np.float32]

    def get_point_cloud(
        self, downsample_factor: int = 1
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        # 创建像素坐标网格
        height, width = self.depth.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
        
        # 计算相机坐标系下的3D点
        pixels = pixels.reshape(-1, 3).T
        rays = np.linalg.inv(self.K) @ pixels
        points = rays * self.depth.reshape(-1)[None,:]
        
        # print(self.T_world_camera.shape)
        R = self.T_world_camera[0:3, 0:3]
        t = self.T_world_camera[0:3, 3]
        
        # 转换到世界坐标系
        points = (R @ points) + t[:,None]
        points = points.T
        
        # 使用深度mask过滤点
        valid_mask = self.mask.reshape(-1)
        points = points[valid_mask]
        colors = self.rgb.reshape(-1, 3)[valid_mask]
        
        return np.asarray(points), np.asarray(colors)