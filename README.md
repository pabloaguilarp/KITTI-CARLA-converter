# KITTI-CARLA to SemanticKITTI converter

This project implements the conversion from the [KITTI-CARLA](https://npm3d.fr/kitti-carla) dataset
to [SemanticKITTI](https://www.semantic-kitti.org/dataset.html#format) format. KITTI-CARLA does not
contain intensity information, so two possibilities are implemented: using an intensities distribution
file or using a fixed value.

## Installation
### Requirements
- numpy (tested with 1.21.6)
- [PyYAML](https://pypi.org/project/PyYAML/) >= 6.0.1

### Data preparation
KITTI-CARLA dataset must be previously downloaded and extracted. Also, SemanticKITTI dataset
is recommended but not required. If SemanticKITTI dataset is downloaded, it is recommended to
convert the KITTI-CARLA towns into the sequences' folder, to keep the order.

## Conversion
In order to execute the conversion process, simply run:
```bash
python main.py -t /path/to/KITTI-CARLA/dataset/town -s /path/to/SemanticKitti/dataset/sequences/seq
```
An additional `convert.sh` file is provided for simplicity. Substitute the paths in the file to match your own and run
```bash
sh convert.sh Town01 42
```
to convert the KITTI-CARLA's Town01 to a new SemanticKITTI sequence with index 42.

Using the `convert_batch.sh` script simplifies the whole process by converting all the towns in a single run.

## Intensity
Since KITTI-CARLA dataset has no intensity information in it, a value has to be assigned to each point.
If the command line argument `--intensities-file` is not set, then the fixed value set by `--intensity-value` is 
assigned to all the points. If `--intensity-value` is not set, a value of 0.5 is used.

Alternatively, a statistics distribution of intensities for each class can be used. If `--intensities-file` points to 
a `.yaml` file that contains the mean and standard deviation for each class, a random normal distribution is computed
and assigned to each class in the sequence. This method helps mimic the real distribution of intensities in the
converted sequence. An intensities' distribution file is provided for simplicity. This file has been calculated on all 
the labeled sequences in SemanticKITTI.

In case you want to recalculate the intensities' distribution parameters you can use the `intensity_distribution.py` 
script. This script computes the intensities' distribution on the provided split and generates a text file that contains 
the distribution parameters (mean, standard deviation and covariance).

To compute the intensities' parameters, simply run:
```bash
python intensity_distribution.py \
-d /mnt/netapp2/Store_uni/home/uvi/et/pap/datasets/SemanticKitti/dataset \
-s labeled \
-c experiments/semantic-kitti/semantic-kitti-real-only.yaml \
-o experiments/intensity_distribution/intensity_distribution.txt
```

The parameters of the output file can be directly copied and pasted into the intensities' distribution `.yaml` file. 