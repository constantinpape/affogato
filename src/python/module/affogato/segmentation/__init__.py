import numpy as np
from ._segmentation import *
import time


# TODO duble check with steffen
def get_valid_edges(shape, offsets, number_of_attractive_channels,
                    strides, randomize_strides):
    # compute valid edges, i.e. the ones not going out of the image boundaries
    ndim = len(offsets[0])
    image_shape = shape[1:]
    valid_edges = np.ones(shape, dtype=bool)
    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            # invalid_slice = (i, ) + i * (slice(None), ) + slice(o)
            inv_slice = slice(0, -o) if o < 0 else slice(image_shape[j] - o, image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))
            valid_edges[invalid_slice] = 0

    # mask additional edges if we have strides
    if strides is not None:
        assert len(strides) == ndim
        # TODO implement
        if randomize_strides:
            stride_factor = np.prod(strides)
            raise NotImplementedError("Randomized strides not implemented yet!")
        else:
            stride_edges = np.zeros_like(valid_edges, dtype='bool')
            stride_edges[:number_of_attractive_channels] = 1
            valid_slice = (slice(number_of_attractive_channels, None),) +\
                tuple(slice(None, None, stride) for stride in strides)
            stride_edges[valid_slice] = 1
            valid_edges = np.logical_and(valid_edges, stride_edges)
    return valid_edges

def get_sorted_flat_indices_and_valid_edges(weights, offsets, number_of_attractive_channels,
                                            strides=None, randomize_strides=False, invert_repulsive_weights=True,
                                            bias_cut=0.):
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    valid_edges = get_valid_edges(weights.shape, offsets, number_of_attractive_channels,
                                  strides, randomize_strides)
    if invert_repulsive_weights:
        weights[number_of_attractive_channels:] *= -1
        weights[number_of_attractive_channels:] += 1
    weights[:number_of_attractive_channels] += bias_cut

    masked_weights = np.ma.masked_array(weights, mask=np.logical_not(valid_edges))

    tick = time.time()
    sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]
    tock = time.time()
    print("Sorted edges in {}s".format(tock-tick))

    return valid_edges.ravel().astype('bool'), sorted_flat_indices.astype('uint64')

def run_mws(sorted_flat_indices,
                valid_edges,
                        offsets,
                        number_of_attractive_channels,
                        image_shape,
                        algorithm='kruskal'):
    assert algorithm in ('kruskal', 'divisive'), "Unsupported algorithm, %s" % algorithm
    if algorithm == 'kruskal':
        labels = compute_mws_segmentation_impl(sorted_flat_indices,
                                               valid_edges.ravel(),
                                               offsets,
                                               number_of_attractive_channels,
                                               image_shape)
    else:
        labels = compute_divisive_mws_segmentation_impl(sorted_flat_indices,
                                                    valid_edges.ravel(),
                                                    offsets,
                                                    number_of_attractive_channels,
                                                    image_shape)




def compute_mws_segmentation(weights, offsets, number_of_attractive_channels,
                             strides=None, randomize_strides=False, invert_repulsive_weights=True,
                             bias_cut=0.,
                             algorithm='kruskal'):
    """
    :param weights: Expected real affinities between 0. and 1. (0 means 'merge', 1 means 'split')
    """
    assert algorithm in ('kruskal', 'prim', 'divisive'), "Unsupported algorithm, %s" % algorithm
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    valid_edges = get_valid_edges(weights.shape, offsets, number_of_attractive_channels,
                                  strides, randomize_strides)
    weights = np.copy(weights)
    if invert_repulsive_weights:
        weights[number_of_attractive_channels:] *= -1
        weights[number_of_attractive_channels:] += 1
    weights[:number_of_attractive_channels] += bias_cut

    if algorithm == 'kruskal' or algorithm == 'divisive':
        # sort and flatten weights
        # ignore masked weights during sorting
        masked_weights = np.ma.masked_array(weights, mask=np.logical_not(valid_edges))

        tick = time.time()
        sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]
        tock = time.time()
        print("Sorted edges in {}s".format(tock-tick))

        # sorted_flat_indices = np.argsort(weights, axis=None)[::-1]
        if algorithm == 'kruskal':
            labels = compute_mws_segmentation_impl(sorted_flat_indices,
                                               valid_edges.ravel(),
                                               offsets,
                                               number_of_attractive_channels,
                                               image_shape)
        elif algorithm == 'divisive':
            labels = compute_divisive_mws_segmentation_impl(sorted_flat_indices,
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
