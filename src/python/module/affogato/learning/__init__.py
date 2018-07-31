import numpy as np
from ._learning import *
from ..segmentation import get_valid_edges


# TODO expose pass parameter
def mutex_malis(weights, gt_labels, offsets,
                number_of_attractive_channels,
                strides=None, randomize_strides=False):

    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    valid_edges = get_valid_edges(weights.shape, offsets, number_of_attractive_channels,
                                  strides, randomize_strides)
    masked_weights = np.ma.masked_array(weights, mask=np.logical_not(valid_edges))
    sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]

    loss, grad = mutex_malis_impl(weights.ravel(), sorted_flat_indices, valid_edges.ravel(),
                                  gt_labels.ravel(), offsets, number_of_attractive_channels,
                                  image_shape)
    return loss, grad.reshape(weights.shape)
