import numpy as np
import vigra

from .mws import compute_mws_segmentation
from ._segmentation import compute_mws_clustering, MWSGridGraph


def cartesian_product(ids):
    return np.array([[x, y] for x in ids for y in ids if x < y], dtype='uint64')


# TODO more efficient implementation
def relabel_from_assignments(node_labels, src_ids, trgt_ids):
    for sid, tid in zip(src_ids, trgt_ids):
        node_labels[node_labels == sid] = tid
    return node_labels


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
    vigra.analysis.relabelConsecutive(seg0, out=seg0, start_label=1, keep_zeros=True)
    id_offset = int(seg0.max()) + 1
    segmentation[0] = seg0

    # segment all other time steps constrained on the previous time step
    for t in range(1, n_time_steps):
        print("Run constrained mws for t", t)
        print("current id-offset:", id_offset)

        # compute the spatial grid-graph of the current time-step
        weights_t = np.require(weights[spatial_channels, t], requirements='C')
        mask_t = np.require(mask[t], requirements='C')
        graph = MWSGridGraph(mask_t.shape)
        graph.set_mask(mask_t)
        graph.compute_weights_and_nh_from_affs(weights_t, spatial_offsets, strides, randomize_strides)
        print("here")

        # uv ids and costs (= weights for mutex watershed) from current graph
        uvs, mutex_uvs = graph.uv_ids(), graph.lr_uv_ids()
        costs, mutex_costs = graph.weights(), graph.lr_weights()
        n_nodes_t = graph.n_nodes
        print("there")

        # compute the region graph of the last time step, connect all regions by mutex edges
        seg_prev = segmentation[t - 1].copy()
        seg_ids = np.unique(seg_prev)
        # if seg_ids[0] == 0:
        #     seg_ids = seg_ids[1:]
        # print("Previous segmentation ids", seg_ids)
        # compute consecutive seg_ids
        seg_ids_flat, _, _ = vigra.analysis.relabelConsecutive(seg_ids)
        # offset by number of nodes in current timestep
        seg_ids_flat += n_nodes_t
        print("even here")

        mutex_uvs_prev = cartesian_product(seg_ids_flat)
        mutex_costs_prev = 2 * np.ones(len(mutex_uvs_prev), dtype='float32')

        # TODO support causal mutex edges
        # TODO support causal strides
        # connect the grid graph to region graph of the last time step
        causal_weights = np.require(weights[causal_channels, t], requirements='C')
        # apply proper offset to seg_prev
        seg_prev[seg_prev != 0] += n_nodes_t
        causal_uvs, causal_costs = graph.get_causal_edges(causal_weights, seg_prev, causal_offsets)
        print("further")

        # concat all edges
        uvs = np.concatenate([uvs, causal_uvs], axis=0)
        costs = np.concatenate([costs, causal_costs], axis=0)
        mutex_uvs = np.concatenate([mutex_uvs, mutex_uvs_prev], axis=0)
        mutex_costs = np.concatenate([mutex_costs, mutex_costs_prev], axis=0)

        # run mws clustering
        n_nodes = max([int(uvs.max()) + 1,
                       int(mutex_uvs.max()) + 1])
        print("going to cluster")
        node_labels = compute_mws_clustering(n_nodes, uvs, mutex_uvs,
                                             costs, mutex_costs)
        print("clustereddd")

        # relabel the node labels consecutively, starting from id-offset
        node_labels, id_offset, _ = vigra.analysis.relabelConsecutive(node_labels,
                                                                      start_label=id_offset, keep_zeros=False)
        id_offset += 1

        # separate into current and previous node labels
        node_labels, prev_node_labels = node_labels[:n_nodes_t], node_labels[n_nodes_t:]

        # label all nodes which are connected to a previous node
        # with the correct `seg_id`
        assert len(prev_node_labels) == len(seg_ids), "%i, %i" % (len(prev_node_labels), len(seg_ids))
        relabel_from_assignments(node_labels, prev_node_labels, seg_ids)

        # reshape node_labels and set all nodes outside mask to 0
        node_labels = node_labels.reshape(mask_t.shape)
        node_labels[np.logical_not(mask_t)] = 0
        segmentation[t] = node_labels

    return segmentation
