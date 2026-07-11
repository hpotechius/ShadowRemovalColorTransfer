"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import cv2
from utils.color_processing import ColorProcessing as CP
import os


class ColorTransfer:
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorithm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(texture, segmented_imgs, mask_segments, normal_segments, height_segments, out_path, mesh, method="fuzzy"):
        outputs = []

        # Transform image to 2D array and select only white pixels
        # last entry is the reference image
        white_pixels_ref = mask_segments[-1] == 255
        reference = segmented_imgs[-1][white_pixels_ref].reshape(-1, 3)
        reference_normal = normal_segments[-1][white_pixels_ref].reshape(-1, 3)
        reference_height = height_segments[-1][white_pixels_ref].reshape(-1, 1)

        for i, (zip_img, zip_normal, zip_mask, zip_height) in enumerate(zip(segmented_imgs, normal_segments, mask_segments, height_segments)):
            white_pixels = zip_mask == 255

            # Transform image to 2D array and select only white pixels
            source = zip_img[white_pixels].reshape(-1, 3)
            normal = zip_normal[white_pixels].reshape(-1, 3)
            height = zip_height[white_pixels].reshape(-1, 1)

            # Apply color transfer
            if i == len(segmented_imgs) - 1:
                final_img = segmented_imgs[-1]
            else:
                if method == "reinhard":
                    output = ColorTransfer().reinhard(texture, source, reference)
                else:
                    normal_src_2d, normal_ref_2d, output  = ColorTransfer().fuzzy(texture, source, reference, normal, reference_normal, height, reference_height)
                
                    ColorTransfer.__save_normal_maps(normal_ref_2d, white_pixels_ref, zip_img, os.path.join(out_path, f'001_fuzzy/02_vis_ref_fuzzy_segment_0.png'), mesh)
                    ColorTransfer.__save_normal_maps(normal_src_2d, white_pixels, zip_img, os.path.join(out_path, f'001_fuzzy/02_vis_src_fuzzy_segment_{i+1}.png'), mesh)

                output = np.squeeze(output, axis=1)

                # Create an empty image with all pixels set to black
                final_img = np.zeros(zip_img.shape, dtype=np.uint8)
                # Place the segmented image on the white pixels of the final image
                final_img[white_pixels] = output
                # Reshape the final image back to the original shape
                final_img = final_img.reshape(zip_img.shape)

            outputs.append(final_img)

        return outputs
    # ------------------------------------------------------------------------------------------------------------------
    # Saves 2D normal map visualization as RGB image
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __save_normal_maps(normal_src_2d, white_pixels, zip_img, out_path, mesh):
        # safe normal clusters
        normal_src_3d = np.zeros((normal_src_2d.shape[0], 3))
        # Assign the green and red channels based on the 2D points
        # The green channel is the first value, the red channel is the second value
        normal_src_3d[:, 1] = normal_src_2d[:, 0]  # Green
        normal_src_3d[:, 0] = normal_src_2d[:, 1]  # Red

        normal_fuzzy_src = np.zeros(zip_img.shape, dtype=np.uint8)
        normal_fuzzy_src[white_pixels] = normal_src_3d * 255
        CP.save_image(normal_fuzzy_src, out_path, mesh)

    # ------------------------------------------------------------------------------------------------------------------
    # Normalizes 3D vectors and integrates height information into 2D representation
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __scale_normal(normal, height):
        # Color coding for the normal map
        #                     [[255, 128, 128],  # +x-Achse
        #                      [  0, 128, 128],  # -x-Achse
        #                      [128, 255, 128],  # +y-Achse
        #                      [128,   0, 128],  # -y-Achse
        #                      [128, 128, 255],  # +z-Achse
        #                      [128, 128,   0]]) # -z-Achse

        # they have to be scaled in order to get values in the range [-1, 1]
        normal_scaled = (normal - 128.0) / 128.0

        # sign of the normal vector is not important
        normal_3d = np.asarray([[abs(a),abs(b),abs(c)] for a, b, c in normal_scaled])

        # replace -0.0 with 0.0
        normal_3d = np.where(normal_3d == -0.0, 0.0, normal_3d)

        # merge directions (x,y) by summing up the values
        sum_first_two = np.sum(normal_3d[:, :2], axis=1)

        z_val = np.sum(normal_3d[:, 2:], axis=1)
        
        # combine to a new array
        normal_2d = np.vstack((sum_first_two, z_val)).T

        # Calculate the sum of the absolute values along axis 1
        sum_abs = np.sum(np.abs(normal_2d), axis=1)

        # Replace zeros in the sum with a small value in order to prevent division by zero
        sum_abs[sum_abs == 0] = 1e-10

        # the sum of the absolute values of each pixel has to be 1
        normal_2d = normal_2d / sum_abs[:, None]

        # integrate height information into the normal
        diff = normal_2d[:,1]
        height = height.reshape(-1)
        normal_2d[:,0] += height/255 * diff
        normal_2d[:,1] -= height/255 * diff

        # Example:
        # Adapted Normal:   [0.3, 0.7]
        # Height:           0.1
        # Scaled Normal:    [0.3 + 0.1*0.7, 0.7 - 0.1*0.7] = [0.37, 0.63]
        # -----------------------------------------------------
        # Adapted Normal:   [0.0, 1.0]
        # Height:           0.9
        # Scaled Normal:    [0.0 + 0.9*1.0, 1.0 - 0.9*1.0] = [0.9, 0.1] -> Roof Area
        # -----------------------------------------------------
        # Adapted Normal:   [0.0, 1.0]
        # Height:           0.1
        # Scaled Normal:    [0.0 + 0.1*1.0, 1.0 - 0.1*1.0] = [0.1, 0.9] -> Floor Area
        
        return normal_2d

    # ------------------------------------------------------------------------------------------------------------------
    # Computes weighted color statistics based on normal vector weights
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __fuzzy_statistics(src, normal_src_2d, cluster_index):
        h, c = src.shape
        weights = normal_src_2d[:, cluster_index].reshape(h, 1)  # Use only the selected column of normal_src_2d
        weights_expanded = np.repeat(weights, 3, axis=1)
        # Expand the weights to match the shape of src
        src_weighted = src * weights_expanded
        # Compute the weighted mean per channel
        mean_src = np.sum(src_weighted, axis=0) / np.sum(weights)
        # Compute the difference between src and the weighted mean
        diff = src - mean_src
        # Square the differences and multiply by the weights
        squared_diff_weighted = (diff ** 2) * weights_expanded
        # Compute the weighted variance
        variance_weighted = np.sum(squared_diff_weighted, axis=0) / np.sum(weights)
        # Compute the weighted standard deviation
        std_dev_weighted = np.sqrt(variance_weighted)

        return mean_src, std_dev_weighted
    
    # ------------------------------------------------------------------------------------------------------------------
    # Fuzzy color transfer using normal and height weights in LAB space
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def fuzzy(texture, src, ref, normal_src, normal_ref, height_src, height_ref):
        # create partition matrix based on normal information
        normal_src_2d = ColorTransfer.__scale_normal(normal_src, height_src)
        normal_ref_2d = ColorTransfer.__scale_normal(normal_ref, height_ref)

        l_fuzzy = np.zeros((src.shape[0],1))
        a_fuzzy = np.zeros((src.shape[0],1))
        b_fuzzy = np.zeros((src.shape[0],1))

        for i in range(2):
            # Convert the images from the BGR to the L*a*b* color space
            src_cluster = np.expand_dims(src, axis=1)
            ref_cluster = np.expand_dims(ref, axis=1)
            src_cluster = src_cluster.astype("float32") / 255.0
            ref_cluster = ref_cluster.astype("float32") / 255.0

            src_cluster = cv2.cvtColor(src_cluster, cv2.COLOR_RGB2LAB)
            ref_cluster = cv2.cvtColor(ref_cluster, cv2.COLOR_RGB2LAB)

            mean_src, std_dev_weighted_src = ColorTransfer.__fuzzy_statistics(np.squeeze(src_cluster), normal_src_2d, i)
            mean_ref, std_dev_weighted_ref = ColorTransfer.__fuzzy_statistics(np.squeeze(ref_cluster), normal_ref_2d, i)

            # # Compute color statistics for the source and reference images
            lMeanSrc, aMeanSrc, bMeanSrc = mean_src
            lMeanRef, aMeanRef, bMeanRef = mean_ref
            lStdSrc, aStdSrc, bStdSrc = std_dev_weighted_src
            lStdRef, aStdRef, bStdRef = std_dev_weighted_ref

            # Subtract the means from the source image
            (l, a, b) = cv2.split(src_cluster)
            l -= lMeanSrc
            a -= aMeanSrc
            b -= bMeanSrc

            # Scale by the standard deviations
            l = (lStdRef / lStdSrc) * l
            a = (aStdRef / aStdSrc) * a
            b = (bStdRef / bStdSrc) * b

            # Add in the means from the reference image
            l += lMeanRef
            a += aMeanRef
            b += bMeanRef

            l_fuzzy += l * normal_src_2d[:, i].reshape(-1,1)
            a_fuzzy += a * normal_src_2d[:, i].reshape(-1,1)
            b_fuzzy += b * normal_src_2d[:, i].reshape(-1,1)

        # Merge the channels together and convert back to the RGB color space
        transfer = cv2.merge([l_fuzzy, a_fuzzy, b_fuzzy])
        transfer = cv2.cvtColor(transfer.astype("float32"), cv2.COLOR_LAB2RGB) * 255
        transfer = np.clip(transfer, 0, 255).astype("uint8")

        # Return the color transferred image
        return normal_src_2d, normal_ref_2d, transfer           

    # ------------------------------------------------------------------------------------------------------------------
    # Reinhard color transfer using statistical color matching in LAB space
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def reinhard(texture, src, ref):
        # Convert the images from the RGB to the L*a*b* color space
        texture = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB).astype("float32")
        src = np.expand_dims(src, axis=1)
        ref = np.expand_dims(ref, axis=1)

        src = src.astype("float32") / 255
        ref = ref.astype("float32") / 255
        
        src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        ref = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)

        # Compute color statistics for the source and reference images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = ColorTransfer.image_stats(src)
        (lMeanRef, lStdRef, aMeanRef, aStdRef, bMeanRef, bStdRef) = ColorTransfer.image_stats(ref)

        # Subtract the means from the source image
        (l, a, b) = cv2.split(src)
        l -= lMeanSrc
        a -= aMeanSrc
        b -= bMeanSrc
        # Scale by the standard deviations
        l = (lStdRef / lStdSrc) * l
        a = (aStdRef / aStdSrc) * a
        b = (bStdRef / bStdSrc) * b

        # Add in the means from the reference image
        l += lMeanRef
        a += aMeanRef
        b += bMeanRef

        # Merge the channels together and convert back to the RGB color space
        transfer = cv2.merge([l, a, b])
        transfer = cv2.cvtColor(transfer.astype("float32"), cv2.COLOR_LAB2RGB) * 255.0
        transfer = np.clip(transfer, 0, 255).astype("uint8")

        # Return the color transferred image
        return transfer

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the mean and standard deviation of each channel in the L*a*b* color space
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Computes mean and standard deviation for each LAB channel
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def image_stats(image):
        # Compute the mean and standard deviation of each channel
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())

        # Return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)