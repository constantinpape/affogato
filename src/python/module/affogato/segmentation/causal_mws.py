import numpy as np
import nifty

from .mws import get_valid_edges, compute_mws_segmentation
from ._segmentation import compute_mws_clustering


# TODO support 3d + t as well
# TODO implement size-filter for individual time steps ????
def compute_causal_mws(weights, offsets, mask,
                       strides=None, randomize_strides=False):
    shape = mask.shape
    assert mask.shape == weights.shape[1:]
    ndim_spatial = mask.ndim - 1
    assert ndim_spatial == 2, "We only support 2d + t for now"
    assert len(offsets) == weigths.shape[0]
    assert len(strides) == ndim_spatial + 1

    n_time_steps = shape[0]
    segmentation = np.zeros(shape)

    # extract the causal and purely spatial offset channels
    causal_channels = [i for i, off in enumerate(offsets) if off[0] != 0]
    spatial_channels = [i for i, off in enumerate(offsets) if off[0] == 0]

    causal_offsets = [off for i, off in enumerate(offsets) if i in causal_channels]
    spatial_offsets = [off for i, off in enumerate(offsets) if i in spatial_channels]

    # segment the first time step un-constrained
    weights0 = np.require(weights[spatial_channels, 0], requirements='C')
    mask0 = np.require(mask[0], requirements='C')
    seg0 = compute_mws_segmentation(weights0, spatial_offsets, ndim_spatial, strides=strides[1:],
                                    randomize_strides=randomize_strides, mask=mask0)
    segmentation[0] = seg0

    # segment all other time steps constrained on the previous time step
    for t in range(1, n_time_steps):

        # compute the grid-graph of the current time-step

        # compute the region graph of the last time step; connecting all regions by mutex edges

        # connect the grid graph to region graph of the last time step

        seg_t = compute_mws_clustering(graph).reshape(shape[1:])
        segmentation[t] = seg_t

    return segmentation
