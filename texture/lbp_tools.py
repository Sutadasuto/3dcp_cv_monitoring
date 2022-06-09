import cv2
import numpy as np

from skimage.feature.texture import local_binary_pattern


def region_lbp(image, radii, ps, levels=256, standardize=False):

    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    mask = np.where(image == 0)
    if standardize:
        true_pixels_mask = np.where(image > 0)
        texture_mean = np.mean(image[true_pixels_mask])
        texture_std = np.std(image[true_pixels_mask])

        standardized_image = np.copy(image)
        limits = [round(texture_mean - 3.1 * texture_std), round(texture_mean + 3.1 * texture_std)]
        standardized_image[np.where(standardized_image < limits[0])] = limits[0]
        standardized_image[np.where(standardized_image > limits[1])] = limits[1]

        bin_image = np.floor(
            levels * (standardized_image.astype(np.float) - limits[0]) / (limits[1] - limits[0] + 1)).astype(np.uint16)
    else:
        bin_image = np.floor(levels * image.astype(np.float) / 256.0).astype(np.uint16)

    processed_image = np.copy(bin_image)
    bin_image[mask] = levels

    lbps = np.zeros((image.shape[0], image.shape[1], len(radii), len(ps)))
    for i, radius in enumerate(radii):
        for j, p in enumerate(ps):
            lbp = local_binary_pattern(bin_image, p, radius, method='default')
            lbp[mask] = 2**p
            lbps[..., i, j] = lbp

    return lbps


def get_lbp_histograms(lbp, bins):

    hs = []
    for i in range(lbp.shape[3]):
        for j in range(lbp.shape[2]):
            current_image = lbp[..., j, i]
            mask_value = np.max(current_image)
            bin_image = np.floor(bins * current_image / mask_value)

            values, counts = np.unique(bin_image, return_counts=True)

            h = np.zeros((1, bins))
            for k in range(len(values)):
                if not values[k] == bins:
                    h[0, int(values[k])] = counts[k]
            h = h/np.sum(h)
            hs.append(h)
    features = np.ravel(np.concatenate(hs))[None, ...]
    return features


def get_lbp_feature_names(radii, ps, bins):
    lbp_feature_names = []
    for p in ps:
        for radius in radii:
            for bin in range(bins):
                lbp_feature_names.append("lbp_radius{:02d}_p{:02d}_bin{:03d}".format(radius, p, bin))
    return lbp_feature_names