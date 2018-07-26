import os
import sys
import time
import h5py
import numpy as np
from scipy.ndimage import convolve
from affogato.segmentation import compute_mws_segmentation

try:
    import mutex_watershed as mws
    WITH_MWS = True
except ImportError:
    WITH_MWS = False

# affinity offsets
OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
           # direct 3d nhood for attractive edges
           [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
           # indirect 3d nhood for dam edges
           [0, -9, 0], [0, 0, -9],
           # long range direct hood
           [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
           # inplane diagonal dam edges
           [0, -27, 0], [0, 0, -27]]


def seg2edges(segmentation):
    gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
    gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
    return ((gx ** 2 + gy ** 2) > 0)


def mws_segmentation_old(affs, stride=np.array([1, 1]),
                         randomize_bounds=False):
    t0 = time.time()
    vol_shape = affs.shape[1:]
    sorted_edges = np.argsort(affs.ravel())
    # run the mst watershed
    seperating_channel = 2
    mst = mws.MutexWatershed(np.array(vol_shape), OFFSETS,
                             seperating_channel, stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_mst_cut(sorted_edges)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation, time.time() - t0


def mws_segmentation(affs, algo):
    t0 = time.time()
    seperating_channel = 2
    seg = compute_mws_segmentation(affs, OFFSETS, seperating_channel,
                                    algorithm=algo)
    return seg, time.time() - t0


def get_2d_from_3d_offsets(offsets):
    # only keep in-plane channels
    keep_channels = [ii for ii, off in enumerate(offsets) if off[0] == 0]
    offsets = [off[1:] for ii, off in enumerate(offsets) if ii in keep_channels]
    return offsets, keep_channels


# TODO use matplotlib instead
def view_res(data, labels):
    sys.path.append('/home/cpape/Work/my_projects/cremi_tools')
    from cremi_tools.viewer.volumina import view
    view(data, labels)


if __name__ == '__main__':

    top_dir = '/home/cpape/Work/data/isbi2012'
    raw_path = os.path.join(top_dir, 'isbi2012_test_volume.h5')
    aff_path = os.path.join(top_dir,
                            'isbi_test_offsetsV4_3d_meantda_damws2deval_final.h5')

    slice_ = np.s_[0, :256, :256]
    aff_slice = (slice(None),) + slice_
    # load affinities and invert the repulsive channels
    with h5py.File(aff_path) as f:
        affs = f['data'][aff_slice]
        affs[:3] *= -1
        affs[:3] += 1
    OFFSETS, keep_channels = get_2d_from_3d_offsets(OFFSETS)
    affs = affs[keep_channels]

    print("Computing MWS segmentation ...")
    seg0, t0 = mws_segmentation(affs, 'kruskal')
    edges0 = seg2edges(seg0)
    print("... in %f s" % t0)
    segs = [seg0]
    labels = ['raw', 'seg_mws']

    print("Computing Prim MWS segmentation ...")
    seg1, t1 = mws_segmentation(affs, 'prim')
    edges1 = seg2edges(seg1)
    print("... in %f s" % t1)
    segs.append(seg1)
    labels.append('seg_prim')

    if np.allclose(edges0, edges1):
        print("Prim and Kruskal segmentation agree")
    else:
        disagree = np.logical_not(np.isclose(edges0, edges1)).sum()
        print("Prim and Kruskal segmentation dis-agree in %i / % i pixels" % (disagree, edges1.size()))

    # if WITH_MWS:
    #     print("Computing old MWS segmentation ...")
    #     affs[:2] *= -1
    #     affs[:2] += 1
    #     affs[2:] *= -1
    #     affs[2:] += 1
    #     seg2, t2 = mws_segmentation_old(affs)
    #     print("... in %f s" % t2)
    #     segs.append(seg2)
    #     labels.append('seg_mws_old')

    with h5py.File(raw_path) as f:
        raw = f['volumes/raw'][slice_]

    view_res([raw] + segs, labels)
