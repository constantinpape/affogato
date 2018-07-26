# import argparse
import time
import sys
import h5py
import numpy as np
from affogato.segmentation import compute_mws_segmentation, compute_mws_clustering

sys.path.append('/home/cpape/Work/my_projects/cremi_tools')

try:
    import mutex_watershed as mws
    WITH_MWS = True
except ImportError:
    WITH_MWS = False

try:
    import nifty
    WITH_NIFTY = True
except ImportError:
    WITH_NIFTY = False

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


def old_mws_segmentation(affs, stride=np.array([1, 1, 1]),
                         randomize_bounds=True):
    t0 = time.time()
    vol_shape = affs.shape[1:]
    sorted_edges = np.argsort(affs.ravel())
    # run the mst watershed
    seperating_channel = 3
    mst = mws.MutexWatershed(np.array(vol_shape), OFFSETS,
                             seperating_channel, stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation, time.time() - t0


def mws_segmentation(affs):
    t0 = time.time()
    seperating_channel = 3
    seg = compute_mws_segmentation(seperating_channel,
                                   OFFSETS, affs)
    return seg, time.time() - t0


def nifty_mws_segmentation(affs):
    from nifty.mws import computeMwsClustering
    t0 = time.time()
    shape = affs.shape[1:]
    graph = nifty.graph.undirectedGridGraph(shape)

    local_uvs = graph.uvIds()

    n_lifted, local_weights, lifted_uvs, lifted_weights = graph.liftedProblemFromLongRangeAffinities(affs, OFFSETS)
    lifted_uvs = lifted_uvs[:n_lifted]
    lifted_weights = lifted_weights[:n_lifted]

    # seg = compute_mws_clustering(graph.numberOfNodes,
    #                              local_uvs, lifted_uvs,
    #                              local_weights, lifted_weights).reshape(shape)
    seg = computeMwsClustering(graph.numberOfNodes,
                               local_uvs, lifted_uvs,
                               local_weights, lifted_weights).reshape(shape)
    return seg, time.time() - t0


if __name__ == '__main__':
    from cremi_tools.viewer.volumina import view

    raw_path = '/home/cpape/Work/data/isbi2012/isbi2012_test_volume.h5'
    aff_path = '/home/cpape/Work/data/isbi2012/isbi_test_offsetsV4_3d_meantda_damws2deval_final.h5'

    slice_ = np.s_[:2, :256, :256]
    aff_slice = (slice(None),) + slice_
    # load affinities and invert the repulsive channels
    with h5py.File(aff_path) as f:
        affs = f['data'][aff_slice]
        affs[3:] *= -1
        affs[3:] += 1

    # print("Computing MWS segmentation ...")
    # seg0, t0 = mws_segmentation(affs)
    # segs = [seg0]
    # labels = ['raw', 'seg_mws']
    # print("... in %f s" % t0)
    if WITH_MWS:
        print("Computing old MWS segmentation ...")
        seg1, t1 = old_mws_segmentation(affs)
        print("... in %f s" % t1)
        segs.append(seg1)
        labels.append('seg_mws_old')
    # if WITH_NIFTY:
    #     print("Computing clustering MWS segmentation ...")
    #     seg2, t2 = nifty_mws_segmentation(affs)
    #     print("... in %f s" % t2)
    #     segs.append(seg2)
    #     labels.append('seg_mws_nifty')

    with h5py.File(raw_path) as f:
        raw = f['volumes/raw'][slice_]

    view([raw] + segs, labels)
