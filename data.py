import cv2
import importlib
import numpy as np
import random
import scipy.io
import os

from math import ceil
from tensorflow.keras.preprocessing import image


### Loading images for Keras

# Utilities
def manual_padding(image, n_pooling_layers):
    # Assuming N pooling layers of size 2x2 with pool size stride (like in U-net and multiscale U-net), we add the
    # necessary number of rows and columns to have an image fully compatible with up sampling layers.
    divisor = 2 ** n_pooling_layers
    try:
        h, w = image.shape
    except ValueError:
        h, w, c = image.shape
    new_h = divisor * ceil(h / divisor)
    new_w = divisor * ceil(w / divisor)
    if new_h == h and new_w == w:
        return image

    if new_h != h:
        new_rows = np.flip(image[h - new_h:, :, ...], axis=0)
        image = np.concatenate([image, new_rows], axis=0)
    if new_w != w:
        new_cols = np.flip(image[:, w - new_w:, ...], axis=1)
        image = np.concatenate([image, new_cols], axis=1)
    return image


# Test model on images
def get_preprocessor(model):
    """
    :param model: A Tensorflow model
    :return: A preprocessor corresponding to the model name
    Model name should match with the name of a model from
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/
    This assumes you used a model with RGB inputs as the first part of your model,
    therefore your input data should be preprocessed with the corresponding
    'preprocess_input' function.
    If the model model is not part of the keras applications models, None is returned
    """
    try:
        m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
        return getattr(m, "preprocess_input")
    except ModuleNotFoundError:
        return None


# Image generators
def get_image(im_path):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

    im = manual_padding(im, n_pooling_layers=4)
    if len(im.shape) == 2:
        im = im[..., None]  # Channels last
    return im


def test_image_from_path(model, input_path, rgb_preprocessor=None, verbose=0):
    if rgb_preprocessor is None:
        rgb_preprocessor = get_preprocessor(model)
    rgb = True if rgb_preprocessor else False
    if rgb:
        prediction = model.predict(
            rgb_preprocessor(get_image(input_path))[None, ...], verbose=verbose)[0, ...]
    else:
        prediction = model.predict(get_image(input_path))

    input_image = cv2.cvtColor(get_image(input_path), cv2.COLOR_BGR2GRAY)[..., None] / 255.0
    return [input_image, prediction]