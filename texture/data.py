import os

from texture.lbp_tools import *
from texture.glcm_tools import *


def calculate_features(photo_dir, **kwargs):
    img_names = sorted([f for f in os.listdir(photo_dir)
                        if not f.startswith(".") and os.path.isfile(os.path.join(photo_dir, f))
                        and not f == "matlab_flag" and not f == "python_flag" and not f.endswith(".csv")],
                       key=lambda f: f.lower())

    # GLCM parameters
    distances = kwargs["distances"]
    angles = kwargs["angles"]
    standardize_glcm_image = kwargs["standardize_glcm_image"]
    glcm_levels = kwargs["glcm_levels"]
    props = kwargs["props"]

    # LBP parameters
    ps = kwargs["ps"]
    radii = kwargs["radii"]
    standardize_lbp_image = kwargs["standardize_lbp_image"]
    lbp_levels = kwargs["lbp_levels"]
    bins = kwargs["bins"]

    names = []
    X = []
    for name in img_names:
        names.append([name])
        img = cv2.imread(os.path.join(photo_dir, name), cv2.IMREAD_GRAYSCALE)
        img = delete_black_regions(img)
        glcms, bin_image = region_glcm(img, distances, angles, glcm_levels, standardize=standardize_glcm_image)
        glcm_features = get_glcm_features(glcms, props)
        lbps = region_lbp(img, radii, ps, lbp_levels, standardize=standardize_lbp_image)
        lbps_features = get_lbp_histograms(lbps, bins)
        X.append(np.concatenate((glcm_features[0, :], lbps_features[0, :])))

    glcm_feature_names = get_glcm_feature_names(distances, angles, props)
    lbps_feature_names = get_lbp_feature_names(radii, ps, bins)
    feature_names = glcm_feature_names + lbps_feature_names
    names = np.array(names)
    X = np.array(X)

    return X, names, np.array(feature_names)[None, ...]


def delete_black_regions(img):
    binary_image = img > 0
    h, w = binary_image.shape

    for row in range(h):
        if np.sum(binary_image[row, :]) == w:
            first_row = row
            break
    for row in reversed(range(h)):
        if np.sum(binary_image[row, :]) == w:
            last_row = row + 1
            break

    try:
        binary_image = binary_image[first_row:last_row, :]
    except UnboundLocalError:
        print("Failed to calculate maximum container box. Returning half size centered box instead")
        return img[int(h/4):int(3*h/4), int(w/4):int(3*w/4)]
    h, w = binary_image.shape

    for col in range(w):
        if np.sum(binary_image[:, col]) == h:
            first_col = col
            break
    for col in reversed(range(w)):
        if np.sum(binary_image[:, col]) == h:
            last_col = col + 1
            break

    return img[first_row:last_row, first_col:last_col]
