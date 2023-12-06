import sys
import subprocess

import gcsfs

from tqdm import tqdm

fs = gcsfs.GCSFileSystem()


def main():
    start, end = int(sys.argv[1]), int(sys.argv[2])
    for i in tqdm(range(start, end)):
        fs.get(
            f"waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training/training.tfrecord-{i:05d}-of-01000",
            "/home/steve/mtr/data/waymo/scenario/training/training.tfrecord-{i:05d}-of-01000",
        )


if __name__ == "__main__":
    main()
