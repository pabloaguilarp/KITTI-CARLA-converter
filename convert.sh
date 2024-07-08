#!/bin/bash

town=$1
seq=$2

/mnt/netapp2/Store_uni/home/uvi/et/pap/.virtualenv/bin/python /mnt/netapp2/Store_uni/home/uvi/et/pap/KITTI-CARLA-converter/main.py \
-t /path/to/KITTI-CARLA/dataset/"$town" \
-s /path/to/SemanticKitti/dataset/sequences/"$seq"