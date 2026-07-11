"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import open3d as o3d
import cv2
import time
import os
import json
from utils.color_processing import ColorProcessing as CP
from utils.intrinsic_decomposition import IntrinsicDecomposition as ID


if __name__ == '__main__':
    # Load options from JSON file
    with open('options.json', 'r') as f:
        options = json.load(f)

    file_path = options['in_folder']
    out_path = options['out_folder']
    clusters = options['clusters']
    blending = options['blending']
    color_transfer_method = options['color_transfer_method']

    os.makedirs(out_path, exist_ok=True)

    # 1. Load mesh
    print("-------------------------\n- (1/13) Loading mesh...", end=' ')
    total_start_time = time.time()
    start_time = time.time()
    mesh = o3d.io.read_triangle_mesh(os.path.join(file_path, "mesh.obj"))
    end_time = time.time()
    duration = end_time - start_time
    print(f"({duration:.2f} seconds)")

    # 2. Load textures
    print("-------------------------\n- (2/13) Loading textures...", end=' ')
    start_time = time.time()
    img_texture = CP.read_image(os.path.join(file_path, "mesh.png"), scale=1.0)
    img_normal = CP.read_image(os.path.join(file_path, "mesh_normal.png"), scale=1.0)
    img_shading = CP.read_image(os.path.join(file_path, "mesh_shading.png"), scale=1.0, color_space=cv2.COLOR_BGR2GRAY)
    img_height = CP.read_image(os.path.join(file_path, "mesh_height.png"), scale=1.0, color_space=cv2.COLOR_BGR2GRAY)
    img_mask = CP.read_image(os.path.join(file_path, "mesh_mask.png"), scale=1.0, interpolation=cv2.INTER_NEAREST, color_space=cv2.COLOR_BGR2GRAY)
    img_mask[img_mask != 0] = 255
    end_time = time.time()
    duration = end_time - start_time
    print(f"({duration:.2f} seconds)")

    # 3. Scale textures
    print("-------------------------\n- (3/13) Scale textures...", end=' ')
    start_time = time.time()
    size=(1024, 1024)
    interpolation=cv2.INTER_LINEAR
    img_texture = cv2.resize(img_texture, size, interpolation=interpolation)
    img_normal = cv2.resize(img_normal, size, interpolation=interpolation)
    img_shading = cv2.resize(img_shading, size, interpolation=interpolation)
    img_height = cv2.resize(img_height, size, interpolation=interpolation)
    img_mask = cv2.resize(img_mask, size, interpolation=interpolation)
    end_time = time.time()
    duration = end_time - start_time
    print(f"({duration:.2f} seconds)")

    # Intrinsic Decomposition
    img_reflectance = ID.decompose(img_texture, 
                                img_shading, 
                                img_normal, 
                                img_mask, 
                                img_height, 
                                mesh, 
                                out_path,
                                num_clusters=clusters,
                                ct_method=color_transfer_method,
                                blending=blending,
                                save_meshes=True)

    end_time = time.time()
    duration = end_time - total_start_time
    print(f"(TOTAL: {duration:.2f} seconds)")