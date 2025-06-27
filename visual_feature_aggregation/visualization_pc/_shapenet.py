import dataclasses
import json
from pathlib import Path
from typing import Tuple, cast

import cv2
import numpy as np
import numpy.typing as npt
import skimage.transform

def load_camera_params(json_path):
    """加载相机参数，转换为COLMAP/OpenCV坐标系"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取相机参数
    x_vector = data['x']
    y_vector = data['y']
    z_vector = data['z']
    origin = data['origin']

    # 定义Blender世界坐标系到COLMAP世界坐标系的转换矩阵
    # world_blender_to_colmap = np.array([
    #     [1, 0, 0],    # x保持不变
    #     [0, 0, 1],    # blender的z轴映射到colmap的y轴
    #     [0, -1, 0]    # blender的-y轴映射到colmap的z轴
    # ])

    # 转换旋转矩阵
    rotation_matrix = np.array([x_vector, y_vector, z_vector]).T
    # rotation_matrix = world_blender_to_colmap @ rotation_matrix

    # 转换平移向量
    translation_vector = np.array(origin)
    # translation_vector = world_blender_to_colmap @ translation_vector

    # 构建变换矩阵
    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector

    # pose_radius_scale = 1.0
    # rt_matrix[:, 3] /= np.linalg.norm(rt_matrix[:, 3])/pose_radius_scale
    # 计算世界坐标到相机坐标的变换矩阵
    w2c = np.linalg.inv(rt_matrix)
    R = w2c[:3,:3]
    t = w2c[:3, 3]

    # rotation_matrix = np.array([x_vector, y_vector, z_vector]).T

    # translation_vector = np.array(origin)

    # rt_matrix = np.eye(4)
    # rt_matrix[:3, :3] = rotation_matrix
    # rt_matrix[:3, 3] = translation_vector

    # print("RT Matrix:")
    # print(rt_matrix)

    # # Since `rt_matrix` transforms coordinates from a reference frame to the new frame,
    # # its inverse as shown the below, `w2c`, will transform coordinates from the new frame back to the reference frame.
    # w2c = np.linalg.inv(rt_matrix)
    # R = w2c[:3,:3]
    # t = w2c[:3, 3]

    max_depth = data['max_depth']
    # max_depth = 3.3
    # 构建相机内参矩阵
    height = width = 512.0  # 根据实际情况调整
    x_fov = data['x_fov']
    y_fov = data['y_fov']
    fx = width / (2 * np.tan(x_fov / 2.0))
    fy = height / (2 * np.tan(y_fov / 2.0))
    K = np.array([
        [fx, 0, width/2],
        [0, fy, height/2],
        [0, 0, 1]
    ])
    
    return K, R, t, max_depth

class ShapenetLoader:
    def __init__(self, 
                 data_path: Path,
                 num_frames: int = 8):
        self.data_path = data_path
        self._num_frames = num_frames
        self.fps = 1
    
    def num_frames(self) -> int:
        return self._num_frames
    
    def get_frame(self, index: int):
        
        depth_path = self.data_path / f'{index:05d}_depth.png'
        rgb_path = self.data_path / f'{index:05d}.png'  # 注意这里是 '00000.png'
        json_path = self.data_path / f'{index:05d}.json'

        # 加载相机参数（先获取max_depth）
        K, R, t, max_depth = load_camera_params(json_path)
        T_world_cameras = np.eye(4)
        T_world_cameras[0:3, 0:3] = R
        T_world_cameras[0:3, 3] = t
        # 读取图像
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[:,:,::-1]  # BGR转RGB
        print(rgb.shape)
        # 将深度图归一化值转换为实际深度值
        # 假设深度图是16位图像，值范围是0-65535
        depth = depth_img.astype(float) / 65535.0 * max_depth
        
        # 从深度图生成mask，使用相对阈值
        depth_mask = (depth > 0.01) & (depth < max_depth)  # 添加一个小的最小阈值避免噪声

        return ShapenetFrame(
            K=K,
            rgb=rgb,
            depth=depth,
            mask=depth_mask,
            T_world_camera=np.linalg.inv(T_world_cameras)
        )
    
@dataclasses.dataclass
class ShapenetFrame:
    """A single frame from a Record3D capture."""

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
        
        # rgb = self.rgb[::downsample_factor, ::downsample_factor]
        # depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        # mask = cast(
        #     npt.NDArray[np.bool_],
        #     skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        # )
        # assert depth.shape == rgb.shape[:2]

        # K = self.K
        # T_world_camera = self.T_world_camera

        # img_wh = rgb.shape[:2][::-1]

        # grid = (
        #     np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        # )
        # grid = grid * downsample_factor

        # homo_grid = np.pad(grid[mask], np.array([[0, 0], [0, 1]]), constant_values=1)
        # local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        # dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
        # points = (T_world_camera[:, -1] + dirs * depth[mask, None]).astype(np.float32)
        # point_colors = rgb[mask]

        # return points, point_colors


