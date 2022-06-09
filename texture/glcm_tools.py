import math
import numpy as np

from skimage.feature import greycomatrix, greycoprops


def region_glcm(image, distances, angles, levels, standardize=False):
    # With an array of shape (#_offsets, 2), return an array (#_offsets, levels, levels) corresponding to #_offsets
    # gray level co-occurrence matrices.
    # To calculate the intra-mask co-occurrence matrix, the input image is converted to uint16. The image pixels outside
    # the mask are assigned 'intensities' as intensity and the co-occurrence matrix is calculated with intensities in
    # the range [0, intensities]. Then, the last row and column from the matrix are suppressed.

    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    mask = np.where(image == 0)
    if standardize:
        true_pixels_mask = np.where(image > 0)
        texture_mean = np.mean(image[true_pixels_mask])
        texture_std = np.std(image[true_pixels_mask])

        standardized_image = np.copy(image)
        limits = [round(texture_mean - 3.1*texture_std), round(texture_mean + 3.1*texture_std)]
        standardized_image[np.where(standardized_image < limits[0])] = limits[0]
        standardized_image[np.where(standardized_image > limits[1])] = limits[1]

        bin_image = np.floor(levels * (standardized_image.astype(np.float) - limits[0]) / (limits[1] - limits[0] + 1)).astype(np.uint16)
    else:
        bin_image = np.floor(levels * image.astype(np.float) / 256.0).astype(np.uint16)

    processed_image = np.copy(bin_image)
    bin_image[mask] = levels

    glcms = greycomatrix(bin_image, distances, angles, levels + 1, symmetric=True)

    glcms = glcms[:-1, :-1, ...].astype(np.float)  # Crop the row and column corresponding to intensity levels + 1
    for i in range(len(distances)):
        for j in range(len(angles)):
            n_pairs = np.sum(glcms[..., i, j])
            if n_pairs > 0:
                p = glcms[..., i, j]/np.sum(glcms[..., i, j])
            else:
                p = glcms[..., i, j]
            glcms[..., i, j] = p

    return glcms, processed_image


def get_glcm_features(glcm, props, avg_and_range=False):

    if avg_and_range:
        features = np.zeros(glcm.shape[2] * 2 * len(props))

        for idx, prop in enumerate(props):
            f = greycoprops(glcm, prop)
            a = np.mean(f, -1)
            r = np.ptp(f, -1)
            features[idx*glcm.shape[2]*2:idx*glcm.shape[2]*2+glcm.shape[2]*2] = np.concatenate((a, r))

    else:
        features = np.zeros(glcm.shape[2] * glcm.shape[3] * len(props))

        for idx, prop in enumerate(props):
            f = greycoprops(glcm, prop)
            features[idx * glcm.shape[2] * glcm.shape[3]:
                     idx * glcm.shape[2] * glcm.shape[3] + glcm.shape[2] * glcm.shape[3]] = f.flatten('F')

    return features[None, ...]


def get_glcm_feature_names(distances, angles, props):
    glcm_feature_names = []
    for prop in props:
        for angle in angles:
            for distance in distances:
                glcm_feature_names.append("glcm_{:03d}px_{:03d}deg_{}".format(distance, int(angle * 180 / math.pi), prop))
    return glcm_feature_names