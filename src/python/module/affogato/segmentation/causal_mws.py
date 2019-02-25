import numpy as np

from .mws import compute_mws_segmentation
from ._segmentation import compute_mws_clustering, MWSGridGraph


def cartesian_product(ids):
    return np.array([[x, y] for x in ids for y in ids if x < y], dtype='uint64')


# TODO support larger time window than t - 1
# TODO support 3d + t as well
# TODO implement size-filter for individual time steps ????
def compute_causal_mws(weights, offsets, mask,
                       strides=None, randomize_strides=False):
    shape = mask.shape
    assert mask.shape == weights.shape[1:]
    ndim_spatial = mask.ndim - 1
    assert ndim_spatial == 2, "We only support 2d + t for now"
    assert len(offsets) == weights.shape[0]
    # TODO support causal strides
    if strides is None:
        strides = [1] * ndim_spatial
    assert len(strides) == ndim_spatial

    n_time_steps = shape[0]
    segmentation = np.zeros(shape, dtype='uint64')

    # extract the causal and purely spatial offset channels
    causal_channels = [i for i, off in enumerate(offsets) if off[0] != 0]
    spatial_channels = [i for i, off in enumerate(offsets) if off[0] == 0]

    causal_offsets = [off for i, off in enumerate(offsets) if i in causal_channels]
    spatial_offsets = [off[1:] for i, off in enumerate(offsets) if i in spatial_channels]

    # we only support negative causal offsets (would be straightforward to also support positive,
    # but want to keep the code simple for now)
    # also, hard-coded to one time step, need to generalize later
    assert all(off[0] == -1 for off in causal_offsets)

    # segment the first time step un-constrained
    weights0 = np.require(weights[spatial_channels, 0], requirements='C')
    mask0 = np.require(mask[0], requirements='C')

    # TODO: generalize this to more than one attractive offset per dim
    #  implicit assumption is that we have a direct grid graph
    nattractive_spatial = ndim_spatial

    print("Run first unconstrained mws")
    seg0 = compute_mws_segmentation(weights0, spatial_offsets, nattractive_spatial, strides=strides,
                                    randomize_strides=randomize_strides, mask=mask0)
    segmentation[0] = seg0

    # segment all other time steps constrained on the previous time step
    for t in range(1, n_time_steps):
        print("Run constrained mws for t", t)

        # compute the region graph of the last time step, connect all regions by mutex edges
        seg_prev = segmentation[t - 1]
        seg_ids = np.unique(seg_prev)

        if seg_ids[0] == 0:
            seg_ids = seg_ids[1:]
        mutex_uvs_prev = cartesian_product(seg_ids)
        n_nodes_prev = int(mutex_uvs_prev.max()) + 1
        mutex_costs_prev = 2 * np.ones(len(mutex_uvs_prev), dtype='float32')

        # compute the spatial grid-graph of the current time-step
        weights_t = np.require(weights[spatial_channels, t], requirements='C')
        mask_t = np.require(mask[t], requirements='C')
        graph = MWSGridGraph(mask_t.shape)
        graph.set_mask(mask_t)
        graph.compute_weights_and_nh_from_affs(weights_t, spatial_offsets, strides, randomize_strides)

        # uv ids and costs (= weights for mutex watershed) from current graph
        uvs, mutex_uvs = graph.uv_ids(), graph.lr_uv_ids()
        # offset the node ids with the number of previous nodes
        uvs += n_nodes_prev
        mutex_uvs += n_nodes_prev
        costs, mutex_costs = graph.weights(), graph.lr_weights()

        # TODO support causal mutex edges
        # TODO support causal strides
        # connect the grid graph to region graph of the last time step
        causal_weights = np.require(weights[causal_channels, t], requirements='C')
        causal_uvs, causal_costs = graph.get_causal_edges(causal_weights, seg_prev,
                                                          causal_offsets, n_nodes_prev)

        # concat all edges
        uvs = np.concatenate([uvs, causal_uvs], axis=0)
        costs = np.concatenate([costs, causal_costs], axis=0)
        mutex_uvs = np.concatenate([mutex_uvs, mutex_uvs_prev], axis=0)
        mutex_costs = np.concatenate([mutex_costs, mutex_costs_prev], axis=0)

        n_nodes = int(uvs.max()) + 1
        seg_t = compute_mws_clustering(n_nodes, uvs, mutex_uvs,
                                       costs, mutex_costs)

        # prev_nodes, seg_t = seg_t[:n_nodes_prev], seg_t[n_nodes_prev:]
        # TODO : relabels seg_t using seg_ids and prev_nodes
        # segmentation[t] = seg_t.reshape(shape[1:])

    return segmentation
