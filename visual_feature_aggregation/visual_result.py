import logging
import torch
import fvdb
import numpy as np

# use to visualize
import polyscope as ps

from fvdb import GridBatch
from einops import rearrange, repeat

def create_cube_mesh(center, size=1.0):
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    # 创建一个立方体的顶点
    vertices = vertices * size + rearrange(center, "c -> 1 c")

    # 定义立方体的面（每个面由两个三角形组成）
    faces = torch.tensor([
        [0,1,2], [0,2,3],  # 前
        [1,5,6], [1,6,2],  # 右
        [5,4,7], [5,7,6],  # 后
        [4,0,3], [4,3,7],  # 左
        [3,2,6], [3,6,7],  # 上
        [4,5,1], [4,1,0]   # 下
    ])
    return vertices, faces

def create_voxel_mesh(positions, size):
    all_vertices = []
    all_faces = []
    for i, pos in enumerate(positions):
        vertices, faces = create_cube_mesh(pos, size)
        all_vertices.append(vertices)
        all_faces.append(faces + i * 8)  # 8是每个立方体的顶点数
    
    return np.vstack(all_vertices), np.vstack(all_faces)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    
    device = torch.device('cuda')
    dtype = torch.float32

    # read file, you should replace this path.
    input_data = torch.load("result.pkl")
    
    indices = input_data['indices'].int()
    color = input_data['feature'].float().cuda()

    vox_size = 1 / 64
    vox_origin = (-0.5 + (vox_size / 2), -0.5 + (vox_size / 2), -0.5 + (vox_size / 2))

    # maybe here need new fvdb api, but this is a small issue.
    # grid = fvdb.gridbatch_from_ijk(tmp, voxel_sizes=vox_size, origins=vox_origin)
    indices = fvdb.JaggedTensor(indices.cuda())
    index = GridBatch(device=device)
    index.set_from_ijk(indices, voxel_sizes=vox_size,  origins=vox_origin)

    gp = index.ijk
    gp = index.grid_to_world(gp.type(dtype))

    color = color.cpu()
    gp = gp.cpu()

    # must not do on a headless device.
    ps.init()

    # 创建体素的立方体表示
    voxel_size = vox_size
    vertices, faces = create_voxel_mesh(gp.jdata, voxel_size)
    voxel_mesh = ps.register_surface_mesh("voxels", vertices, faces)
    
    repeated_colors = repeat(color / 255., "n c -> (n v) c", v=8)  # 每个体素有8个顶点
    voxel_mesh.add_color_quantity("normal colors", repeated_colors, defined_on='vertices', enabled=True)

    ps.show()

if __name__ == "__main__":
    main()
