"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import os
import numpy as np
import open3d as o3d
from PIL import Image

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class ColorProcessing:
    # --------------------------------------------------------------------------------------------------------------
    # Loads image with PIL
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_image(file_path, scale=1.0, interpolation=cv2.INTER_LINEAR, color_space=cv2.COLOR_BGR2RGB):
        pil_img = Image.open(file_path)

        if color_space in (cv2.COLOR_BGR2GRAY, cv2.COLOR_RGB2GRAY):
            img = np.array(pil_img.convert("L"))
        elif color_space == cv2.COLOR_RGB2BGR:
            img = np.array(pil_img.convert("RGB"))
            img = cv2.cvtColor(img, color_space)
        else:
            img = np.array(pil_img.convert("RGB"))

        if scale != 1.0:
            # Compute the new size
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            # Resize the image
            img = cv2.resize(img, new_size, interpolation=interpolation)

        return img
        
    # --------------------------------------------------------------------------------------------------------------
    # Saves RGB image to PNG
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod     
    def save_image(img, savepath, mesh=None, save_meshes=False):
        # Create the directory structure if it does not exist
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        # Convert RGB to BGR for cv2.imwrite (which stores it as-is)
        # This way PNG contains BGR data that PIL reads correctly
        if len(img.shape) == 3:
            bgr_img = img# cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(savepath, bgr_img)
        else:
            cv2.imwrite(savepath, img)
        
        # Keep RGB for mesh texture
        save_img = img

        # If a mesh is provided, save it with the texture
        if mesh is not None and save_meshes:
            # Set the mesh texture (RGB format)
            mesh.textures = [o3d.geometry.Image(save_img)]

            # Save the mesh
            mesh_savepath = os.path.splitext(savepath)[0] + ".obj"
            o3d.io.write_triangle_mesh(mesh_savepath, mesh)

            # Delete the file with the "_0" suffix
            texture_0_path = os.path.splitext(savepath)[0] + "_0.png"
            if os.path.exists(texture_0_path):
                os.remove(texture_0_path)

            mtl_path = os.path.splitext(savepath)[0] + ".mtl"
            wrong_tex_name = os.path.splitext(savepath.split('/')[-1])[0] + "_0.png"
            correct_tex_name = os.path.splitext(savepath.split('/')[-1])[0] + ".png"

            # Read the MTL file and replace the line
            if os.path.exists(mtl_path):
                with open(mtl_path, 'r') as file:
                    lines = file.readlines()

                with open(mtl_path, 'w') as file:
                    for line in lines:
                        if line.strip().startswith('map_Kd') and wrong_tex_name in line:
                            line = line.replace(wrong_tex_name, correct_tex_name)
                        file.write(line)

    # --------------------------------------------------------------------------------------------------------------
    # Creates linear RGB color gradient for cluster visualization
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_color_gradient(num_colors=10, start=[0, 255, 0], end=[255, 0, 0]):
        # Define the start (green) and end (red) colors
        start_color = np.array(start)
        end_color = np.array(end)

        # Create a linear space between the start and end colors
        color_gradient = np.linspace(start_color, end_color, num_colors)

        # Convert the colors to integers and create a dictionary
        color_dict = {i: color.astype(int) for i, color in enumerate(color_gradient)}

        return color_dict