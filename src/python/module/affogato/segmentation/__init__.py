from ._segmentation import *
import numpy as np


def compute_mws_segmentation(number_of_attractive_channels, offsets, weights):

    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    # compute valid edges
    valid_edges = np.ones(weights.shape, dtype=bool)
    image_shape = weights.shape[1:]

    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            # invalid_slice = (i, ) + i * (slice(None), ) + slice(o)
            inv_slice = slice(0, -o) if o < 0 else slice(image_shape[j] - o, image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))

            valid_edges[invalid_slice] = 0

    # sort and flatten weights
    # TODO: add masking and ignore masked weights during sorting
    sorted_flat_indices = np.argsort(weights, axis=None)[::-1]

    return compute_mws_segmentation_impl(number_of_attractive_channels,
                                         offsets,
                                         image_shape,
                                         sorted_flat_indices,
                                         valid_edges.ravel()).reshape(image_shape)

def compute_mws_prim_segmentation(number_of_attractive_channels, offsets, weights):

    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    # compute valid edges
    valid_edges = np.ones(weights.shape, dtype=bool)
    image_shape = weights.shape[1:]

    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            # invalid_slice = (i, ) + i * (slice(None), ) + slice(o)
            inv_slice = slice(0, -o) if o < 0 else slice(image_shape[j] - o, image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))

            valid_edges[invalid_slice] = 0

    return compute_mws_prim_segmentation_impl(number_of_attractive_channels,
                                         offsets,
                                         image_shape,
                                         weights.ravel(),
                                         valid_edges.ravel()).reshape(image_shape)
