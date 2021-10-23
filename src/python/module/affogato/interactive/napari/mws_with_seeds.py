import numpy as np
from vigra.analysis import relabelConsecutive
from affogato.segmentation import MWSGridGraph, compute_mws_clustering


def compute_grid_graph(shape, mask=None, seeds=None):
    """ Compute MWS grid graph.
    """
    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask)
    if seeds is not None:
        grid_graph.update_seeds(seeds)
    return grid_graph


# TODO refactor these changes into elf.segmentation.mutex_watershed...
# to keep up-to-date with the affogato changes
def mws_with_seeds(affs, offsets, seeds, strides,
                   randomize_strides=False, mask=None):
    ndim = len(offsets[0])

    # compute grid graph with seeds and optional mask
    shape = affs.shape[1:]
    grid_graph = compute_grid_graph(shape, mask, seeds)

    # compute nn and mutex nh
    grid_graph.add_attractive_seed_edges = True
    uvs, weights = grid_graph.compute_nh_and_weights(1. - np.require(affs[:ndim], requirements='C'),
                                                     offsets[:ndim])

    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[ndim:],
                                                                            requirements='C'),
                                                                 offsets[ndim:], strides,
                                                                 randomize_strides)

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    seg = compute_mws_clustering(n_nodes, uvs, mutex_uvs, weights, mutex_weights)
    relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    grid_graph.relabel_to_seeds(seg)
    seg = seg.reshape(shape)
    if mask is not None:
        seg[np.logical_not(mask)] = 0

    return seg
