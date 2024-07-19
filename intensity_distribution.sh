#!/bin/bash

/mnt/netapp2/Store_uni/home/uvi/et/pap/.virtualenv/bin/python \
/mnt/netapp2/Store_uni/home/uvi/et/pap/KITTI-CARLA-converter/intensity_distribution.py \
-d /path/to/SemanticKITTI/dataset \
-s labeled \
-c path/to/config/semantic-kitti.yaml \
-o path/to/output_folder/intensity_distribution.txt