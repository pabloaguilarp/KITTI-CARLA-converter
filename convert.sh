#!/bin/bash

town=$1
seq=$2

/mnt/netapp2/Store_uni/home/uvi/et/pap/.virtualenv/bin/python /mnt/netapp2/Store_uni/home/uvi/et/pap/KITTI-CARLA-converter/main.py \
-t /mnt/lustre/scratch/nlsas/home/uvi/et/pap/data/KITTI-CARLA/dataset/"$town" \
-s /mnt/netapp2/Store_uni/home/uvi/et/pap/datasets/SemanticKitti/dataset/sequences/"$seq"