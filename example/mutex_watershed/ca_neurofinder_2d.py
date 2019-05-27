import os
import sys
import time
import argparse

import numpy as np
import h5py

from scipy.ndimage import convolve
from affogato.segmentation import compute_mws_segmentation

try:
    # import mutex_watershed as mws
    import constrained_mst as cmst
    WITH_MWS = True
except ImportError:
    WITH_MWS = False

# affinity offsets
OFFSETS = [[-1, 0], [0, -1],
           [-9, 0], [0, -9], [-9, -9], [9, -9], [-9, -4], [-4, -9], [4, -9], [9, -4],
           [-27, 0], [0, -27]]


def mws_segmentation(affs, algo, strides=None):
    t0 = time.time()
    seperating_channel = 2
    seg = compute_mws_segmentation(affs, OFFSETS, seperating_channel,
                                   algorithm=algo, strides=strides)
    return seg, time.time() - t0


if __name__ == '__main__':
    # TODO add more options to parser to allow for different data
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help='path to neuro data')
    parser.add_argument(
        'key', type=str, help='path to neuro data')
    args = parser.parse_args()
    path = args.path
    assert os.path.exists(path), path

    # load affinities and invert the repulsive channels
    with h5py.File(path, "r") as f:
        affs = f[args.key][:]
        affs[2:] *= -1
        affs[2:] += 1

    print("Computing MWS segmentation ...")

    affs += 0.001 * np.random.rand(*affs.shape)
    affs /= 1.001

    strides = [2, 2]
    seg0, t0 = mws_segmentation(affs, 'kruskal', strides=strides)

    with h5py.File(path+"result.h5", "w") as f:
        f.create_dataset("segmentation", data=seg0, compression="gzip")
