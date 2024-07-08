import argparse
import os
import sys
import numpy as np
import yaml

import ply
import io_utils as io


def iterate_frames(path):
    town_frames_path = os.path.join(path, "generated", "frames")
    for index, item in enumerate(os.listdir(town_frames_path)):
        filename_abs = os.path.join(town_frames_path, item)
        if os.path.isfile(filename_abs) and item.endswith(".ply"):
            yield index, filename_abs, item


def load_lidar(scan_path, label_path):
    scan = io.read_points(scan_path)
    labels = io.read_labels(label_path)
    print(np.array(scan.points).shape)
    print(np.array(scan.remissions).shape)

    print(f"Scan dimensions: {np.min(scan.points, axis=0)}, {np.max(scan.points, axis=0)}")
    print(f"Scan remissions: {np.min(scan.remissions, axis=0)}, {np.max(scan.remissions, axis=0)}")
    print(np.unique(np.array(labels, dtype=np.uint16))) # as uint16 to get only the semantic part and ignore the instances

    return scan.points, scan.remissions, labels

def load_frame(seq_path, file_idx):
    scan_path = os.path.join(seq_path, "velodyne", str(file_idx).zfill(6) + ".bin")
    label_path = os.path.join(seq_path, "labels", str(file_idx).zfill(6) + ".label")
    return load_lidar(scan_path, label_path)

def save_lidar(points, remissions, labels, pc_filepath, labels_filepath):
    # concatenate x,y,z and intensity
    point_cloud = np.column_stack((points, remissions))

    # save
    point_cloud.astype(np.float32).tofile(pc_filepath)  # note: must save as float32, otherwise loading errors
    io.write_labels(labels_filepath, labels)

def populate_remissions(labels, means, stddevs, remap_dict):
    labels_remap = np.vectorize(remap_dict.get)(labels).astype(np.uint16)
    labels_unique = np.unique(labels_remap)
    remissions = np.zeros(labels_remap.shape, dtype=np.float32)
    for label in labels_unique:
        mean = means[label]
        stddev = stddevs[label]
        dis_array = np.random.normal(mean, stddev, np.sum(labels_remap == label)).astype(np.float32)
        remissions[labels_remap == label] = dis_array
    return remissions, np.unique(labels)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert from KITTI-CARLA to SemanticKITTI")
    parser.add_argument('-t', "--town")
    parser.add_argument('-s', "--sequence-path")
    parser.add_argument('-i', "--intensities-file", default="intensities_dist.yaml", required=False)
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)

    with open('labels.yaml', "r") as file:
        yaml_data = yaml.safe_load(file)

    with open('mapping.yaml', "r") as file:
        mapping_data = yaml.safe_load(file)

    with open("semantic-kitti.yaml", "r") as file:
        config = yaml.safe_load(file)

    with open(args.intensities_file, "r") as file:
        stats = yaml.safe_load(file)

    kc_to_sk = mapping_data["mapping"]

    seq_path = args.sequence_path
    velo_path = os.path.join(args.sequence_path, "velodyne")
    label_path = os.path.join(args.sequence_path, "labels")

    os.makedirs(seq_path)
    os.mkdir(velo_path)
    os.mkdir(label_path)

    # Set paths
    for index, filename_abs, _ in iterate_frames(args.town):
        # load scan
        pc = ply.read_ply(filename_abs)
        points = np.vstack((pc['x'], pc['y'], pc['z'])).T

        semantic = pc['semantic'].astype(np.uint16)
        # remap labels
        labels_remap = np.vectorize(kc_to_sk.get)(semantic).astype(np.uint16)

        if args.intensities_file == "":
            remissions = np.full(len(points), 0.5, dtype=np.float32)
        else:
            remissions, labels_unique = populate_remissions(labels_remap, stats["mean"], stats["stddev"], config["learning_map"])

        # save files
        bin_file = os.path.join(velo_path, str(index).zfill(6) + ".bin")
        label_file = os.path.join(label_path, str(index).zfill(6) + ".label")
        save_lidar(points, remissions, labels_remap, bin_file, label_file)
