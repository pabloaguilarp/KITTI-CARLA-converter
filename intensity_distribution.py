#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import math
import os
import yaml
import numpy as np
import collections
from tqdm import tqdm

from auxiliary.laserscan import SemLaserScan

splits = ["train",   # Train set
          "valid",   # Valid set
          "test",    # Test set
          "all",     # All sets
          "labeled"  # Train + valid
          ]


def parse_args():
    parser = argparse.ArgumentParser("./intensity_distribution.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default=None,
        help='Dataset dir'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default="valid",
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Dataset config file',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=False,
        default="",
        help='Output file',
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def compute_sequences(FLAGS):
    sequences = []
    if FLAGS.split == 'all':
        sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        print("Error. Cannot compute intensity distributions on non-labeled splits")
        exit(1)
    elif FLAGS.split == 'train':
        sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    elif FLAGS.split == 'test':
        sequences = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        print("Error. Cannot compute intensity distributions on non-labeled splits")
        exit(1)
    elif FLAGS.split == 'valid':
        sequences = [8]
    elif FLAGS.split == 'labeled':
        sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        print(f"Error. Provided split '{FLAGS.split}' not supported. Provide one of the list '{splits}'")
        exit(1)

    return sequences


def populate_names(seq, foldername):
    paths = os.path.join(FLAGS.dataset, 'sequences', str(seq).zfill(2), foldername)
    if not os.path.isdir(paths):
        print(f"{foldername.capitalize()} folder '{paths}' doesn't exist! Exiting...")
        quit()
    names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(paths)) for f in fn]
    names.sort()
    return names


def populate_sequence_names(sequence):
    scan_names = populate_names(sequence, "velodyne")
    label_names = populate_names(sequence, "labels")
    assert (len(label_names) == len(scan_names))
    return scan_names, label_names


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    seqs = compute_sequences(FLAGS)
    print(f"Calculating intensity distribution on split '{FLAGS.split}'. Sequences: {seqs}")

    learning_map_inv = CFG["learning_map_inv"]
    learning_map = CFG["learning_map"]
    labels = CFG["labels"]
    seq_accum = {}
    for key, _ in learning_map_inv.items():
        seq_accum[key] = []

    seq_accum_n = {}
    seq_accum_mean = {}
    seq_accum_m2 = {}
    for key, _ in learning_map_inv.items():
        seq_accum_n[key] = 0
        seq_accum_mean[key] = 0
        seq_accum_m2[key] = 0

    for seq in seqs:
        scan_names, label_names = populate_sequence_names(seq)
        print(f"Loaded {len(scan_names)} files in sequence {seq}")

        scan = SemLaserScan(CFG["color_map"], project=False)

        for idx in tqdm(range(len(scan_names))):
            scan.open_scan(scan_names[idx])
            scan.open_label(label_names[idx])

            for pt in range(len(scan.points)):
                key = learning_map[scan.sem_label[pt]]
                seq_accum_n[key] = seq_accum_n[key] + 1
                delta = scan.remissions[pt] - seq_accum_mean[key]
                seq_accum_mean[key] = seq_accum_mean[key] + delta / seq_accum_n[key]
                seq_accum_m2[key] = seq_accum_m2[key] + delta * (scan.remissions[pt] - seq_accum_mean[key])

    seq_accum_variance = {}
    seq_accum_stddev = {}
    for key, _ in learning_map_inv.items():
        seq_accum_variance[key] = seq_accum_m2[key] / (seq_accum_n[key] - 1)
        seq_accum_stddev[key] = math.sqrt(seq_accum_variance[key])

    for key, _ in learning_map_inv.items():
        if abs(seq_accum_mean[key]) < 0.0001 or abs(seq_accum_stddev[key]) < 0.0001:
            continue
        print(
            f"Class {key} ({labels[learning_map_inv[key]]}). \u03BC: {seq_accum_mean[key]}, \u03C3: {seq_accum_stddev[key]}")

    if FLAGS.output != "":
        with open(FLAGS.output, "w") as file:
            print(f"Writing data to file {FLAGS.output}")
            file.write("class id,class name,mean,stddev,var\n")
            for key, _ in learning_map_inv.items():
                file.write(
                    f"{key},{labels[learning_map_inv[key]]},{seq_accum_mean[key]},{seq_accum_stddev[key]},{seq_accum_variance[key]}\n")
