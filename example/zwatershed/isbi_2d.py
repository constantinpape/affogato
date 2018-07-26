import os
import sys
import time
import argparse

import numpy as np
import h5py

from scipy.ndimage import convolve
from affogato.segmentation import compute_zws_segmentation


def seg2edges(segmentation):
    gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
    gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
    return ((gx ** 2 + gy ** 2) > 0)


# TODO use matplotlib instead
def view_res(data, labels):
    sys.path.append('/home/cpape/Work/my_projects/cremi_tools')
    from cremi_tools.viewer.volumina import view
    view(data, labels)


if __name__ == '__main__':
    # TODO add more options to parser to allow for different data
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='path to example data (download from https://hcicloud.iwr.uni-heidelberg.de/index.php/s/6LuE7nxBN3EFRtL)')
    args = parser.parse_args()
    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, 'isbi_test_volume.h5')
    assert os.path.exists(path), path

    slice_ = np.s_[0, :512, :512]
    aff_slice = (slice(1,3),) + slice_
    # load affinities and invert the repulsive channels
    with h5py.File(path) as f:
        raw = f['raw'][slice_]
        affs = f['affinities'][aff_slice]
        affs = 1. - affs

    print("Computing ZWS segmentation ...")
    t0 = time.time()
    seg0, _ = compute_zws_segmentation(affs, 0.2, 0.98, 0.5, 25)
    # seg0, _ = compute_zws_segmentation(affs, 0.2, 0.98, 0, 0)
    t0 = time.time() - t0
    print("... in %f s" % t0)
    # edges0 = seg2edges(seg0)
    data = [raw, seg0]  #, edges0]
    labels = ['raw', 'seg_zws'] # , 'edges_zws']
    view_res(data, labels)
