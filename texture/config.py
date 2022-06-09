from math import pi

# GLCM parameters (number_of_features = len(distances) * len(angles) * len(props))
distances = list(range(1, 50+1, 1))  # Offset distance
angles = [0, pi/2]  # Offset direction (in radians)
standardize_glcm_image = True  # Standardize the image so that the GLCM levels correspond to the range [μ-3.1σ, μ+3.1σ] in the input image
glcm_levels = 12  # Number of intensity bins to calculate the GLCM (the resulting matrix size is glcm_levels×glcm_levels)
props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']  # Properties to calculate from a GLCM according to scikit-image documentation

# LBP parameters (number_of_features = len(ps) * len(radii) * bins)
ps = [8]  # Number of neighbors to calculate the binary pattern. Transformed pixels have a value [0, 2^neighbors]
radii = list(range(1, 40+1, 10))  # Offset radius from the pixel to its neighbors
standardize_lbp_image = True  # The input image is standardized from [μ-3.1σ, μ+3.1σ] to [0, lbp_levels-1] (see line below)
lbp_levels = 64  # Number of intensity bins in the image used to calculate the LBP
bins = 8  # The histogram of intensities from the transformed image is calculated using this number of bins
