"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import numpy as np
import time
import os
import warnings
import multiprocessing as mp
from sklearn.cluster import KMeans
import open3d as o3d
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point, LineString
from scipy.ndimage import uniform_filter
from utils.color_processing import ColorProcessing as CP
from utils.color_transfer import ColorTransfer as CT

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class IntrinsicDecomposition:
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # Segments shading image using K-means clustering on white pixels
    # --------------------------------------------------------------------------------------------------------------
    def __hard_clustering(self, shading, mask, num_clusters):
        # Assume mask is a grayscale image with white pixels having value 255
        white_pixels = mask == 255

        # Transform image to 2D array and select only white pixels
        pixels = shading[white_pixels].reshape(-1, 1)


        # Apply KMeans Clustering
        kmeans = KMeans(n_clusters=num_clusters, max_iter=1000, n_init="auto")
        kmeans.fit(pixels)

        # Get labels for each pixel
        labels = kmeans.labels_

        # Define a mapping from cluster labels to colors
        colors = CP.create_color_gradient(num_colors=num_clusters)

        # Replace each pixel with its cluster color
        segmented_img = np.array([colors[label] for label in labels])

        # Create an empty image with all pixels set to black
        final_img = np.zeros((*shading.shape, 3), dtype=np.uint8)

        # Place the segmented image on the white pixels of the final image
        final_img[white_pixels] = segmented_img

        # Reshape the final image back to the original shape
        final_img = final_img.reshape((*shading.shape, 3))

        # Create empty masks for each cluster
        masks = [np.zeros_like(mask) for _ in range(num_clusters)]

        # Set corresponding pixels in each mask
        for i in range(num_clusters):
            masks[i][white_pixels] = (labels == i) * 255

        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Create a list of tuples (mask, cluster_center)
        masks_with_centers = [(masks[i], cluster_centers[i]) for i in range(num_clusters)]

        # Sort the list by cluster center
        masks_with_centers.sort(key=lambda x: x[1])

        # Get the sorted masks
        sorted_masks = [mask for mask, center in masks_with_centers]

        # get shading segments
        shading_segments = self.__cluster_extraction(shading, sorted_masks)

        # Return a visualization of the clustering and the masks for each cluster
        return final_img, shading_segments, sorted_masks
    
    # --------------------------------------------------------------------------------------------------------------
    # Extracts image regions corresponding to cluster masks
    # --------------------------------------------------------------------------------------------------------------
    def __cluster_extraction(self, img, cluster_masks):
        # Initialize a list to hold the images for each cluster
        cluster_images = []

        # For each cluster mask, create a new image
        for i, mask in enumerate(cluster_masks):
            # Create a new image filled with zeros (black)
            cluster_image = np.zeros_like(img)

            # Use the mask to copy the pixels from the original image
            cluster_image[mask == 255] = img[mask == 255]

            # Add the cluster image to the list
            cluster_images.append(cluster_image)

        # Return the list of cluster images
        return cluster_images

    # --------------------------------------------------------------------------------------------------------------
    # Blends cluster images using weighted masks for seamless integration
    # --------------------------------------------------------------------------------------------------------------
    def __recombine_clusters(self, cluster_imgs, cluster_masks):
        # Initialize an empty image with the same shape as the cluster images
        combined_img = np.zeros_like(cluster_imgs[0], dtype=np.float32)
        total_weight = np.zeros_like(cluster_imgs[0], dtype=np.float32)

        # Iterate over the cluster images and masks
        for img, mask in zip(cluster_imgs, cluster_masks):
            # Convert the mask to float and normalize it to [0, 1]
            mask = mask.astype(float) / 255

            # Multiply the image by the mask to apply the grayscale weighting
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mask_3d = mask_3d.astype(np.float32)
            img = img.astype(np.float32)

            img = cv2.multiply(img, mask_3d)

            # Add the resulting image to the combined image
            combined_img = cv2.add(combined_img, img)
            
            # Add the mask to the total weight
            total_weight = cv2.add(total_weight, mask_3d)

        # Replace zeros in total_weight with a small number to avoid division by zero
        total_weight[total_weight == 0] = 1e-6

        # Divide the combined image by the total weight to get the blended image
        combined_img = cv2.divide(combined_img, total_weight)

        return combined_img

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # Checks if line intersects 2D triangle using Shapely
    # --------------------------------------------------------------------------------------------------------------  
    @staticmethod
    def does_line_intersect_triangle(uv0, uv1, uv2, line_start, line_end):
        # Define the triangle using shapely
        triangle = Polygon([uv0, uv1, uv2])
        
        # Define the line using shapely
        line = LineString([line_start, line_end])
        
        # Check if the line intersects the triangle
        return triangle.intersects(line)   
    
    # --------------------------------------------------------------------------------------------------------------
    # Rasterizes triangle to texture coordinates for pixel mapping
    # --------------------------------------------------------------------------------------------------------------  
    @staticmethod
    def get_pixels_within_triangle(uv0, uv1, uv2, v0, v1, v2, size):
        # Scale the UV coordinates to the range [0, size-1]x[0, size-1]
        uv0_scaled = np.array(uv0) * size
        uv1_scaled = np.array(uv1) * size
        uv2_scaled = np.array(uv2) * size

        # Define the triangle in the pixel space using shapely
        triangle = Polygon([uv0_scaled, uv1_scaled, uv2_scaled])

        # Get the bounding box of the triangle
        min_x = int(min(uv0_scaled[0], uv1_scaled[0], uv2_scaled[0]))
        max_x = int(max(uv0_scaled[0], uv1_scaled[0], uv2_scaled[0]))
        min_y = int(min(uv0_scaled[1], uv1_scaled[1], uv2_scaled[1]))
        max_y = int(max(uv0_scaled[1], uv1_scaled[1], uv2_scaled[1]))

        # Initialize an empty list to store the pixel positions and 3D coordinates within the triangle
        pixels_and_coords_within_triangle = []

        # Iterate over all pixels within the bounding box of the triangle
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # determine if the pixel is inside the triangle
                inside = False
                connected = False

                # Create a shapely point for the center of the pixel
                pixel_center = Point(x + 0.5, y + 0.5)
                pixel_bl = Point(x + 0.0, y + 0.0)
                pixel_br = Point(x + 1.0, y + 0.0)
                pixel_tl = Point(x + 0.0, y + 1.0)
                pixel_tr = Point(x + 1.0, y + 1.0)

                if triangle.intersects(pixel_center):
                    inside = True
                    connected = True
                else:
                    line_bottom = IntrinsicDecomposition.does_line_intersect_triangle(uv0_scaled, uv1_scaled, uv2_scaled, pixel_tl, pixel_tr)
                    line_top = IntrinsicDecomposition.does_line_intersect_triangle(uv0_scaled, uv1_scaled, uv2_scaled, pixel_bl, pixel_br)
                    line_left = IntrinsicDecomposition.does_line_intersect_triangle(uv0_scaled, uv1_scaled, uv2_scaled, pixel_tl, pixel_bl)
                    line_right = IntrinsicDecomposition.does_line_intersect_triangle(uv0_scaled, uv1_scaled, uv2_scaled, pixel_tr, pixel_br)
                    if line_bottom or line_top or line_left or line_right:
                        connected = True

                if connected:
                    # Compute vectors
                    v0_2d = triangle.exterior.coords[2] - np.array(triangle.exterior.coords[0])
                    v1_2d = triangle.exterior.coords[1] - np.array(triangle.exterior.coords[0])
                    v2_2d = np.array([x, y]) - np.array(triangle.exterior.coords[0])

                    # Compute dot products
                    dot00 = np.dot(v0_2d, v0_2d)
                    dot01 = np.dot(v0_2d, v1_2d)
                    dot02 = np.dot(v0_2d, v2_2d)
                    dot11 = np.dot(v1_2d, v1_2d)
                    dot12 = np.dot(v1_2d, v2_2d)

                    # Compute barycentric coordinates
                    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
                    u = (dot11 * dot02 - dot01 * dot12) * invDenom
                    v = (dot00 * dot12 - dot01 * dot02) * invDenom

                    # Interpolate the 3D coordinates
                    w = 1 - u - v

                    coord_3d = w * v0 + u * v1 + v * v2

                    pixels_and_coords_within_triangle.append(((x, y), coord_3d, inside))

        return pixels_and_coords_within_triangle

    # --------------------------------------------------------------------------------------------------------------
    # Maps single triangle to texture space for 3D-to-2D projection
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_triangle(i, triangles, vertices, triangle_uvs, size):
        print(f"- Processing triangle {i+1}/{len(triangles)}", end='\r')

        # Get the vertices of the triangle
        v0, v1, v2 = vertices[triangles[i]]

        # Get the UV coordinates of the triangle
        uv0 = triangle_uvs[i*3]
        uv1 = triangle_uvs[i*3+1]
        uv2 = triangle_uvs[i*3+2]

        # Get the pixels within the triangle
        pixels_and_coords_3d = IntrinsicDecomposition.get_pixels_within_triangle(uv0, uv1, uv2, v0, v2, v1, size)
        current_pixel_2d_coords = [item[0] for item in pixels_and_coords_3d]

        # Return the results
        return (current_pixel_2d_coords, 
                [item[1] for item in pixels_and_coords_3d], 
                [item[2] for item in pixels_and_coords_3d])

    # --------------------------------------------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_mapping_test(mesh, size):
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        triangle_uvs = np.asarray(mesh.triangle_uvs)
        

        # Prepare the input data
        args = [(i, triangles, vertices, triangle_uvs, size) for i in range(len(triangles))]

        # Create a pool of workers
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(IntrinsicDecomposition.process_triangle, args)

        # Combine the results
        pixel_2d_coords = []
        pixel_3d_coords = []
        inout = []

        for result in results:
            pixel_2d_coords += result[0]
            pixel_3d_coords += result[1]
            inout += result[2]

        print("- All triangles have been processed")  

        # Initialize the mapping array with None
        mapping = np.full((size, size, 3), None)
        mapping_inside = np.full((size, size, 1), None)

        # Fill the mapping array with the corresponding 3D coordinates
        for (x, y), coords_3d, inside in zip(pixel_2d_coords, pixel_3d_coords, inout):
            mapping[size - 1 - y, x] = coords_3d
            mapping_inside[size - 1 - y, x] = inside

        return mapping
    
    # --------------------------------------------------------------------------------------------------------------
    # Creates 3D-to-2D texture mapping via sequential triangle rasterization
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_mapping(mesh, size):
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        triangle_uvs = np.asarray(mesh.triangle_uvs)
        
        # Initialize an empty list to store the 3D coordinates of the pixels
        pixel_3d_coords = []
        pixel_2d_coords = []
        inout = []

        # Iterate over each triangle
        for i in range(len(triangles)):
            print(f"- Processing triangle {i+1}/{len(triangles)}", end='\r')

            # Get the vertices of the triangle
            v0, v1, v2 = vertices[triangles[i]]

            # Get the UV coordinates of the triangle
            uv0 = triangle_uvs[i*3]
            uv1 = triangle_uvs[i*3+1]
            uv2 = triangle_uvs[i*3+2]

            # Get the pixels within the triangle
            pixels_and_coords_3d = IntrinsicDecomposition.get_pixels_within_triangle(uv0, uv1, uv2, v0, v2, v1, size)
            current_pixel_2d_coords = [item[0] for item in pixels_and_coords_3d]

            # Fill the specified positions with white values
            pixel_2d_coords += current_pixel_2d_coords
            pixel_3d_coords += [item[1] for item in pixels_and_coords_3d]
            inout += [item[2] for item in pixels_and_coords_3d]

        print("- All triangles have been processed")        

        # Initialize the mapping array with None
        mapping = np.full((size, size, 3), None)
        mapping_inside = np.full((size, size, 1), None)

        # Fill the mapping array with the corresponding 3D coordinates
        for (x, y), coords_3d, inside in zip(pixel_2d_coords, pixel_3d_coords, inout):
            mapping[size - 1 - y, x] = coords_3d
            mapping_inside[size - 1 - y, x] = inside

        return mapping
    
    # --------------------------------------------------------------------------------------------------------------
    # Projects 2D texture to 3D point cloud using coordinate mapping
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def project_texture_to_3d(texture_img, mapping):
        # Create a point cloud
        pcd = o3d.geometry.PointCloud()

        if len(texture_img.shape) == 2:
            texture_img = cv2.cvtColor(texture_img, cv2.COLOR_GRAY2RGB)
        else:
            texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)


        points = mapping.reshape(mapping.shape[0] * mapping.shape[1], mapping.shape[2])
        colors = texture_img.reshape(texture_img.shape[0] * texture_img.shape[1], texture_img.shape[2])

        valid_indices = np.where(points != None)[0]
        filtered_points = points[valid_indices]
        filtered_colors = colors[valid_indices]

        # Assign the points to the point cloud
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255)
        return pcd

    # --------------------------------------------------------------------------------------------------------------
    # Reprojects point cloud back to texture coordinates
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def project_3d_to_texture(PC, mapping, size):
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(PC)
        colors = np.asarray(pcd.colors)

        # Initialize an empty image with the same size as the texture image
        texture_img = np.zeros((size * size, 3), dtype=np.uint8)

        points = mapping.reshape(mapping.shape[0] * mapping.shape[1], mapping.shape[2])
        valid_indices = np.where(points != None)[0]

        texture_img[valid_indices] = colors * 255
        texture_img_reshape = texture_img.reshape(size, size, 3)
        texture_img_rgb = cv2.cvtColor(texture_img_reshape, cv2.COLOR_BGR2RGB)

        return texture_img_rgb
 
    # --------------------------------------------------------------------------------------------------------------
    # Blurs the colors of a point cloud in batches
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def blur_pointcloud_color_batch(PC, radius, extend_type="blurring"):
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(PC)
        print("- Point cloud has " + str(len(pcd.points)) + " points")
        # Convert the point cloud to a numpy array
        points = np.asarray(pcd.points)#[:20000]
        colors = np.asarray(pcd.colors)#[:20000]

        # Build a KDTree for nearest neighbor search
        tree = KDTree(points)
        print("- KDTree has been built")

        step = 1000000
        end = len(points)
        values = list(range(0, end, step))
        if values[-1] != end:
            values.append(end)

        print("- with " + str(len(values)) + " chunks")

        # List for storing the average_neighbor_colors arrays
        all_average_neighbor_colors = []

        k = 100
        for a, b in zip(values, values[1:]):
            _, indices = tree.query(points[a:b], k=k+1)
            neighbor_colors = colors[indices]
            print("- Nearest neighbors have been found for chunk " + str(a) + " to " + str(b))

            if extend_type == "blurring":
                average_neighbor_colors = np.mean(neighbor_colors, axis=1)
                all_average_neighbor_colors.append(average_neighbor_colors)
            else:
                # Step 1: Create a mask for non-black pixels
                non_black_mask = np.any(neighbor_colors != [0, 0, 0], axis=2)
                # Step 2: Replace black pixels with NaN in a copy of neighbor_colors
                neighbor_colors_copy = np.where(non_black_mask[..., None], neighbor_colors, np.nan)
                # Step 3: Compute the mean while excluding NaN values
                # Note the mean of only nan values will lead to a RuntimeWarning: Mean of empty slice, which can be ignored
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    average_neighbor_colors = np.nanmean(neighbor_colors_copy, axis=1)

                # Add the result to the list
                all_average_neighbor_colors.append(average_neighbor_colors)
        
            # Concatenate all average_neighbor_colors arrays along the first dimension
        combined_average_neighbor_colors = np.concatenate(all_average_neighbor_colors, axis=0)

        if extend_type == "blurring":
            pass
        else:
            # Step 4: Identify the non-black pixels in colors
            non_black_pixels_mask = np.any(colors != [0, 0, 0], axis=1)
            # Step 5: Copy the non-black pixels from colors to average_neighbor_colors
            combined_average_neighbor_colors[non_black_pixels_mask] = colors[non_black_pixels_mask]

        # Assign the blurred colors to the point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(combined_average_neighbor_colors)
        return pcd

    # --------------------------------------------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def replace_black_pixels_with_mean(recombined_img):
        # 1. apply sum filter in a 3x3 winwow for each pixel
        sum_image = uniform_filter(recombined_img, size=(3, 3, 1), mode='constant', cval=0) * 9

        # 2. Create 3D mask
        # Create a mask for black pixels (all channels are 0)
        black_mask = np.all(recombined_img == [0, 0, 0], axis=-1)
        # Initialize the mask with ones
        mask = np.ones(recombined_img.shape[:2], dtype=np.float32)
        # Set the black pixels to 0
        mask[black_mask] = 0
        three_channel_mask = np.stack([mask] * 3, axis=-1)

        # 3. sum all pixels in the 3x3 window within the mask to count all non-black pixels per pixel
        count_image = uniform_filter(three_channel_mask, size=(3, 3, 1), mode='constant', cval=0) * 9
        # Replace zeros in count_image with a small number to avoid division by zero
        count_image[count_image == 0] = 1e-6

        # 4. divide the sum image by the count image to get the mean of all non-black pixels per pixel
        mean_image = sum_image / count_image

        # 5. copy non-black pixels from the recombined image to the mean image to remove the blurring effect
        non_black_mask = ~black_mask
        mean_image[non_black_mask] = recombined_img[non_black_mask]
        
        return mean_image
    
    # --------------------------------------------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def decompose(img, 
                  shading, 
                  normal, 
                  mask, 
                  height, 
                  mesh, 
                  out_path, 
                  num_clusters=3, 
                  ct_method="fuzzy", 
                  blending=True, 
                  save_meshes=False):
        
        h, _, _ = img.shape

        # 4. Clustering of the shading image (with consideration of the mask) into {num_clusters} clusters
        print("-------------------------\n- (4/13) Clustering shading image...", end=' ')
        start_time = time.time()
        vis_segments, shading_segments, mask_segments = IntrinsicDecomposition().__hard_clustering(shading, mask, num_clusters)
        # save the visualization of the clustering
        CP.save_image(vis_segments, os.path.join(out_path, "000_segments/00_vis_segment.png"), mesh, save_meshes)


        # save the mask and shading segments
        for i in range(num_clusters):
            CP.save_image(mask_segments[i], os.path.join(out_path, f"000_segments/00_mask_segment_{i}.png"), mesh, save_meshes)
            CP.save_image(shading_segments[i], os.path.join(out_path, f'000_segments/00_shading_segment_{i}.png'), mesh, save_meshes)
        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")

        # 5. Extraction of the individual clusters from the color, normal and height images
        print("-------------------------\n- (5/13) Extracting clusters...", end=' ')
        start_time = time.time()

        scaled_height = height

        texture_segments = IntrinsicDecomposition().__cluster_extraction(img, mask_segments)
        normal_segments = IntrinsicDecomposition().__cluster_extraction(normal, mask_segments)
        height_segments = IntrinsicDecomposition().__cluster_extraction(scaled_height, mask_segments)
        for i in range(num_clusters):
            CP.save_image(texture_segments[i], os.path.join(out_path, f'000_segments/01_texture_segment_{i}.png'), mesh, save_meshes)
            CP.save_image(normal_segments[i], os.path.join(out_path, f'000_segments/01_normal_segment_{i}.png'), mesh, save_meshes)
            CP.save_image(height_segments[i], os.path.join(out_path, f'000_segments/01_height_segment_{i}.png'), mesh, save_meshes)
        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")


        # 6. Color transfer of the dark texture segments
        print("-------------------------\n- (6/13) Color transfer...", end=' ')
        start_time = time.time()
        ct_segemented_imgs = CT().apply(img, 
                                        texture_segments, 
                                        mask_segments, 
                                        normal_segments, 
                                        height_segments,
                                        out_path, 
                                        mesh,
                                        method=ct_method)
        
        for i, simg in enumerate(ct_segemented_imgs):
            CP.save_image(simg, os.path.join(out_path, f'002_colorTransfer/03_texture_ct_segment_{i}.png'), mesh, save_meshes)

        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")

        # 7. Create mapping between 2D and 3D coordinates and save as numpy array
        print(f'-------------------------\n- (7/13) Create mapping of size {h}x{h}...')
        start_time = time.time()

        os.makedirs(os.path.join(out_path, "003_mapping"), exist_ok=True)
        if os.path.exists(os.path.join(out_path, f'003_mapping/mapping_{h}.npy')):
            mapping = np.load(os.path.join(out_path, f'003_mapping/mapping_{h}.npy'), allow_pickle=True)
        else:
            mapping = IntrinsicDecomposition.create_mapping_test(mesh, h)
            np.save(os.path.join(out_path, f'003_mapping/mapping_{h}.npy'), mapping)
   
        end_time = time.time()
        duration = end_time - start_time
        print(f"- ({duration:.2f} seconds)")

        # 8. Project the texture to 3D using mapping
        print("-------------------------\n- (8/13) Project Texture to 3D...", end=' ')
        start_time = time.time()
        os.makedirs(os.path.join(out_path, "004_pointclouds"), exist_ok=True)
        for i, simg in enumerate(ct_segemented_imgs):
            texture_pcd = IntrinsicDecomposition.project_texture_to_3d(simg, mapping)
            o3d.io.write_point_cloud(os.path.join(out_path, f'004_pointclouds/04_shading_texture_pcd_{i}.ply'), texture_pcd, write_ascii=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")


        # 9. Extending of the texture point clouds and projection
        print("-------------------------\n- (9/13) Extending of the texture point clouds and projection...")
        start_time = time.time()
        os.makedirs(os.path.join(out_path, "005_projected"), exist_ok=True)
        for i in range(num_clusters):
            print(f'- Process cluster {i}...')
            # Extension of texture point cloud
            blended_texture_pcd = IntrinsicDecomposition.blur_pointcloud_color_batch(os.path.join(out_path, f'004_pointclouds/04_shading_texture_pcd_{i}.ply'), 0.1, extend_type="extend")
            o3d.io.write_point_cloud(os.path.join(out_path, f"004_pointclouds/06_extended_texture_pcd_{i}.ply"), blended_texture_pcd)
            # Create weight maps for texture blending
            extended_texture = IntrinsicDecomposition.project_3d_to_texture(os.path.join(out_path, f"004_pointclouds/06_extended_texture_pcd_{i}.ply"), 
                                                                            mapping,
                                                                            h)
            
            # swap R and B channel
            #extended_texture = extended_texture[:, :, [2, 1, 0]]
            CP.save_image(extended_texture, os.path.join(out_path, f"005_projected/07_extended_texture_{i}.png"), mesh, save_meshes)

        end_time = time.time()
        duration = end_time - start_time
        print(f"- ({duration:.2f} seconds)")

        # 10. Project the mask to 3D using mapping
        print("-------------------------\n- (10/13) Project mask to 3D...", end=' ')
        start_time = time.time()
        for i, mm in enumerate(mask_segments):
            mask_pcd = IntrinsicDecomposition.project_texture_to_3d(mm, mapping)
            o3d.io.write_point_cloud(os.path.join(out_path, f'004_pointclouds/08_shading_mask_pcd_{i}.ply'), mask_pcd, write_ascii=True)

        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")

        # 11. Blurring of the mask point clouds and projection
        print("-------------------------\n- (11/13) Blurring of the mask point clouds and projection...")
        start_time = time.time()
        for i in range(num_clusters):
            print(f'- Process cluster {i}...')
            # 5. Blending of shading mask point clouds
            blended_shading_mask_pcd = IntrinsicDecomposition.blur_pointcloud_color_batch(os.path.join(out_path, f'004_pointclouds/08_shading_mask_pcd_{i}.ply'), 0.1)
            o3d.io.write_point_cloud(os.path.join(out_path, f"004_pointclouds/09_blurred_shading_mask_pcd_{i}.ply"), blended_shading_mask_pcd)

            # 6. Create weight maps for texture blending
            blurred_mask = IntrinsicDecomposition.project_3d_to_texture(os.path.join(out_path, f"004_pointclouds/09_blurred_shading_mask_pcd_{i}.ply"), 
                                                                        mapping,
                                                                        h)
            
            CP.save_image(blurred_mask, os.path.join(out_path, f"005_projected/10_blurred_mask_{i}.png"), mesh, save_meshes)

        end_time = time.time()
        duration = end_time - start_time
        print(f"- ({duration:.2f} seconds)")

        # 12. Merging of the color images
        print("-------------------------\n- (12/13) Merging of the color images...", end=' ')
        start_time = time.time()
        if blending:
            colortransferred_imgs = [CP.read_image(os.path.join(out_path, f'005_projected/07_extended_texture_{i}.png'), scale=1.0) for i in range(num_clusters)]
            colortransferred_imgs = np.array(colortransferred_imgs)

            blurred_cluster_masks = [CP.read_image(os.path.join(out_path, f'005_projected/10_blurred_mask_{i}.png'), scale=1.0, color_space=cv2.COLOR_BGR2GRAY) for i in range(num_clusters)]
            blurred_cluster_masks = np.array(blurred_cluster_masks)

        else:
            colortransferred_imgs = [CP.read_image(os.path.join(out_path, f'002_colorTransfer/03_texture_ct_segment_{i}.png'), scale=1.0) for i in range(num_clusters)]
            colortransferred_imgs = np.array(colortransferred_imgs)

            blurred_cluster_masks = [CP.read_image(os.path.join(out_path, f'000_segments/00_mask_segment_{i}.png'), scale=1.0, color_space=cv2.COLOR_BGR2GRAY) for i in range(num_clusters)]
            blurred_cluster_masks = np.array(blurred_cluster_masks)

        recombined_img = IntrinsicDecomposition().__recombine_clusters(colortransferred_imgs, blurred_cluster_masks)
        CP.save_image(recombined_img, os.path.join(out_path, '006_result/11_recombined.png'), mesh, save_meshes)

        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")

        # 13. Extension of borders to prevent artifacts
        # Note: recombined_img fits perfectly into the uvs of the mesh, but tools like MeshLab
        # and Blender do not properly visualize the border. Therefore, we extend the borders.
        print("-------------------------\n- (13/13) Extension of borders to prevent artifacts...", end=' ')
        recombined_ext_img = IntrinsicDecomposition().replace_black_pixels_with_mean(recombined_img)
        CP.save_image(recombined_ext_img, os.path.join(out_path, '006_result/12_recombined_ext.png'), mesh)

        end_time = time.time()
        duration = end_time - start_time
        print(f"({duration:.2f} seconds)")

        return recombined_ext_img