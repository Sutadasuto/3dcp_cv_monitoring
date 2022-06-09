import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # Disable eager execution to avoid conflicts with the encoder-decoder architecture

import argparse
import cv2
import numpy as np
import os
import subprocess

from distutils.util import strtobool

import data

from models.available_models import get_models_dict
from texture.texture_thread import TextureClassifier

# Used to create neural network architecture
models_dict = get_models_dict()


def main(args):
    script_directory = os.path.abspath(os.path.dirname(__file__))

    # The outputs of the script will be saved here
    results_path = os.path.join(script_directory, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Get the pre-trained weights of U-VGG19
    with open(os.path.join(script_directory, "config_files", "config"), 'r') as parameters_file:
        lines = parameters_file.read().strip().split('\n')
    for line in lines:
        key, value = line.strip().split(", ")
        if key == "weights_path":
            weights_path = value

    # Deal with GPU usage
    print("Preparing neural network for interlayer segmentation")
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Used for memory error in RTX2070
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Initialize U-VGG19 foor interlayer segmentation
    input_size = (None, None)
    model = models_dict["uvgg19"]((input_size[0], input_size[1], 1))
    rgb_preprocessor = data.get_preprocessor(model)
    model.load_weights(weights_path)

    # Prepare the texture CNN in a parallel thread
    texture_classifier = TextureClassifier().start()

    # Detect the interlayer lines
    [im, pred] = data.test_image_from_path(model, args.image, rgb_preprocessor=rgb_preprocessor)

    x_color = cv2.imread(args.image)
    or_shape = x_color.shape
    pred = pred[:or_shape[0], :or_shape[1], 0]

    # Save images to the results folder
    segmentation_path = os.path.join(results_path, "interlayer_lines.png")
    cv2.imwrite(segmentation_path, np.where(pred >= 0.5, 255, 0))
    image_path = os.path.join(results_path, "input_image.png")
    cv2.imwrite(image_path, x_color)

    # Perform the characterization in Matlab
    plots_path = os.path.join(results_path, "plots.png")
    histograms_path = os.path.join(results_path, "histograms.png")
    matlabCommands = "cd %s;figures=[figure();figure()];analyze('%s','%s',figures);figures(1).WindowState='maximized';saveas(figures(1),'%s');figures(2).WindowState='maximized';saveas(figures(2),'%s');exit" % (
        script_directory, image_path, segmentation_path, plots_path, histograms_path)
    command = 'matlab -r "%s"' % matlabCommands
    subprocess.run(command, shell=True)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to the image to analyze")
    parser.add_argument("--gpu", type=str, default=True)

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


# Run the script
if __name__ == "__main__":
    args = parse_args()  # Parse user arguments
    main(args)  # Run main function with user arguments
