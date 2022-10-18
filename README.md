# 3D Concrete Printing monitoring
Tool for monitoring and anomaly detection of 3D Concrete Printing using Computer Vision 

This code can be used to produce the results presented in

```
@article{RILLGARCIA2022103175,
title = {Inline monitoring of 3D concrete printing using computer vision},
journal = {Additive Manufacturing},
volume = {60},
pages = {103175},
year = {2022},
issn = {2214-8604},
doi = {https://doi.org/10.1016/j.addma.2022.103175},
url = {https://www.sciencedirect.com/science/article/pii/S2214860422005644},
author = {Rodrigo Rill-García and Eva Dokladalova and Petr Dokládal and Jean-François Caron and Romain Mesnil and Pierre Margerit and Malo Charrier},
}
```

If using this tool, kindly cite the reference above.

## Inputs and outputs
Given an input image of a 3D printed piece of concrete (with a lateral view), the present tool will produce local measurements about the geometry and texture of the observed piece. The geometry measures correspond to the orientation and curvature of the interlayer lines, the width of the printed layers, and the distance of the printing nozzle with respect to the last printed layer (for furhter information about these measurements and the properties of the expected input image, please refer to the article). The texture measurements are used to provide a region-wise classification of the observed material, either good or one of three anomalous classes: fluid, dry or tearing (for further information about these classes, please refer to the article). Once the measures are obtained, one histogram per each one of them is calculated.

Both the measured values and the produced histograms are shown as below:

![alt text](https://github.com/Sutadasuto/3dcp_cv_monitoring/blob/main/results/plots.png?raw=true)
![alt text](https://github.com/Sutadasuto/3dcp_cv_monitoring/blob/main/results/histograms.png?raw=true)

The green dotted lines show the user-defined ranges of admissible values.
This ranges can be modified in the 'ranges.txt' file.
Values outside these ranges are detected as anomalies and shown in blue if below the range or red if above it.
Input-image specific parameters (such as the px/mm ratio) are specified in 'image_properties.txt'.
Specific variable values used to obtain the measures, as explained in the article, are accessible at 'measurement_parameters.txt'.
The parameters to generate the histograms (namely bin sizes and min-max values) are available at 'histogram_parameters.txt'.
All the mentioned text files are contained in the 'config_files' directory.

## Software requirements
The current repository uses Python and Matlab code.
Regarding the Python setup, an 'environment.yml' file is provided to replicate the conda environment used for testing.
The Python environment is used to build the interlayer segmentation and the texture classification networks on Tensorflow 2.
Regarding the Matlab code, it was tested on the 2020b version; it uses the image and the curve fitting toolboxes. The Matlab script performs the geometrical characterization.

## How to run
The analysis of an image can be performed simply by running:

```
python main.py "path/to/image"
```

A GPU option is provided.
The script tries to use a GPU by default; the user can explicitly ask for no GPU usage by providing the "--gpu False" argument.
After running the script, the input image (in PNG format), the interlayer segmentation map, the plots of local measures, and the histograms of the obtained measures will be saved in the 'results' directory.

Notice, however, that you need to unzip the U-VGG19 weights ('weights.zip') and provide the path to the extracted file in 'config_files/config'.

The 2 study cases presented in the article are available at the 'sample_images' directory.
For more images, take a look at our I3DCP dataset: https://github.com/Sutadasuto/I3DCP
