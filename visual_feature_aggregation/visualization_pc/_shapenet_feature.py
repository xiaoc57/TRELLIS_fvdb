import os
import dataclasses
import json
from pathlib import Path
from typing import Tuple, cast

import torch
import cv2
import numpy as np
import numpy.typing as npt
from einops import rearrange, repeat

class ShapenetLoader:
    def __init__(
        self,
        base_path
    ):

        self.fps = 1
        self.H = 392
        self.W = 392

        self.base_path = base_path
        with open(os.path.join(base_path, "transforms_train.json"), 'r') as f:
            self.metadata = json.load(f)

        w = h = int(392)
        fx = fy = 0.5/np.tan(0.5*self.metadata['camera_angle_x'])

        self.K = np.float32([[fx, 0, 1/2],
                        [0, fy, 1/2],
                        [0,  0,   1]])
        # self.K = torch.tensor([
        #     [self.metadata["fl_x"], 0, self.metadata["cx"]],
        #     [0, self.metadata["fl_y"], self.metadata["cx"]],
        #     [0, 0, 1]
        # ])
        self._num_frames = len(self.metadata["frames"])
    
    def num_frames(self) -> int:
        return self._num_frames

    def get_batch(self):

        rgb_ = []
        mask_ = []
        c2w_ = []

        for index in range(self.num_frames()):
            file_name = self.metadata["frames"][index]["file_path"]
            # depth_path = os.path.join(self.base_path, file_name + "_depth.png")
            rgb_path = os.path.join(self.base_path, file_name + ".png")

            # w2c colmap
            c2w_blender = np.array(self.metadata["frames"][index]["transform_matrix"])
            c2w_colmap = c2w_blender
            c2w_colmap[:, 1:3] = c2w_colmap[:, 1:3] * -1

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

            rgb_w = cv2.resize(rgb_w, (self.H, self.W),
                            interpolation=cv2.INTER_AREA)     # (width, height)


            rgb = cv2.cvtColor(rgb_w, cv2.COLOR_BGR2RGB)    
            mask = (alpha > 0.1)
            mask = cv2.resize(mask.astype(np.uint8), (self.H, self.W), interpolation=cv2.INTER_NEAREST)

            rgb_.append(torch.from_numpy(rearrange(rgb, "h w c -> c h w"))) # (3 h w)
            mask_.append(torch.from_numpy(rearrange(mask, "h w c -> c h w")[0:1])) # (h w)
            c2w_.append(torch.from_numpy(c2w_colmap))
        
        return {
            "rgb": torch.stack(rgb_).float(),
            "mask": torch.stack(mask_).int(),
            "extrinsics": torch.linalg.inv(torch.stack(c2w_)).float(),
            "intrinsics": torch.from_numpy(repeat(self.K, "h w -> n h w", n=len(rgb_))).float(),
        }