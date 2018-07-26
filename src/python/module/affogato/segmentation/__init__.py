from ._segmentation import *
import numpy as np


# TODO: add strides and randomize bounds arguments
def compute_mws_segmentation(weights, offsets, number_of_attractive_channels,
                             algorithm='kruskal'):
    assert algorithm in ('kruskal', 'prim'), "Unsupported algorithm, %s" % algorithm
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    # compute valid edges
    valid_edges = np.ones(weights.shape, dtype=bool)
    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            # invalid_slice = (i, ) + i * (slice(None), ) + slice(o)
            inv_slice = slice(0, -o) if o < 0 else slice(image_shape[j] - o, image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))
            valid_edges[invalid_slice] = 0

    # TODO: mask additional edges once we have strides

    if algorithm == 'kruskal':
        # sort and flatten weights
        # TODO: ignore masked weights during sorting
        # we can maybe use np.ma for this
        sorted_flat_indices = np.argsort(weights, axis=None)[::-1]
        labels = compute_mws_segmentation_impl(sorted_flat_indices,
                                               valid_edges.ravel(),
                                               offsets,
                                               number_of_attractive_channels,
                                               image_shape)
    else:
        labels = compute_mws_prim_segmentation_impl(weights.ravel(),
                                                    valid_edges.ravel(),
                                                    offsets,
                                                    number_of_attractive_channels,
                                                    image_shape)

    return labels.reshape(image_shape)
