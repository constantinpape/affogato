import numpy as np

from ._segmentation import compute_mws_prim_segmentation_impl, compute_mws_segmentation_impl, compute_mws_segmentation_v2_impl
from ..affinities import compute_affinities


def get_valid_edges(shape, offsets, number_of_attractive_channels=None,
                    strides=None, randomize_strides=False, node_mask=None, edge_mask=None):
    # compute valid edges
    ndim = len(offsets[0])
    image_shape = shape[1:]
    valid_edges = np.ones(shape, dtype=bool)
    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            inv_slice = slice(0, -o) if o < 0 else slice(image_shape[j] - o, image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))
            valid_edges[invalid_slice] = 0

    # mask additional edges if we have strides
    if strides is not None:
        assert len(strides) == ndim
        assert number_of_attractive_channels is not None
        if randomize_strides:
            stride_factor = 1 / np.prod(strides)
            stride_edges = np.random.rand(*valid_edges.shape) < stride_factor
            stride_edges[:number_of_attractive_channels] = 1
            valid_edges = np.logical_and(valid_edges, stride_edges)
        else:
            stride_edges = np.zeros_like(valid_edges, dtype='bool')
            stride_edges[:number_of_attractive_channels] = 1
            valid_slice = (slice(number_of_attractive_channels, None),) +\
                tuple(slice(None, None, stride) for stride in strides)
            stride_edges[valid_slice] = 1
            valid_edges = np.logical_and(valid_edges, stride_edges)

    # if we have an external mask, mask all transitions to and within that mask
    if node_mask is not None:
        assert node_mask.shape == image_shape, "%s, %s" % (str(node_mask.shape), str(image_shape))
        assert node_mask.dtype == np.dtype('bool'), str(node_mask.dtype)
        # mask transitions to mask
        transition_to_mask, _ = compute_affinities(node_mask, offsets)
        transition_to_mask = transition_to_mask == 0
        valid_edges[transition_to_mask] = False
        # mask within mask
        valid_edges[:, node_mask] = False

    # if we have an external edge mask, mask all those edges
    if edge_mask is not None:
        assert edge_mask.shape == valid_edges.shape, "%s, %s" % (str(edge_mask.shape), str(valid_edges.shape))
        assert edge_mask.dtype == np.dtype('bool'), str(edge_mask.dtype)
        # mask edges
        valid_edges = np.logical_and(edge_mask, valid_edges)

    return valid_edges


def compute_mws_segmentation(weights, offsets, number_of_attractive_channels,
                             strides=None, randomize_strides=False,
                             algorithm='kruskal', mask=None):
    assert algorithm in ('kruskal', 'prim'), "Unsupported algorithm, %s" % algorithm
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    # we assume that we get a 'valid mask', i.e. a mask where valid regions are set true
    # and invalid regions are set to false.
    # for computation, we need the opposite though
    inv_mask = None if mask is None else np.logical_not(mask)
    valid_edges = get_valid_edges(weights.shape, offsets, number_of_attractive_channels,
                                  strides, randomize_strides, node_mask=inv_mask)

    if algorithm == 'kruskal':
        # sort and flatten weights
        # ignore masked weights during sorting
        masked_weights = np.ma.masked_array(weights, mask=np.logical_not(valid_edges))
        sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]

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

    labels = labels.reshape(image_shape)
    # if we had an external mask, make sure it is mapped to zero
    if mask is not None:
        # increase labels by 1, so we don't merge anything with the mask
        labels += 1
        labels[inv_mask] = 0
    return labels



def compute_mws_segmentation_from_signed_affinities(signed_affinities, offsets,
                                                    foreground_mask=None, edge_mask=None,
                                                    return_valid_edge_mask=False):
    """
    :param signed_affinities: If the image is N-dimensional, this is a N+1 dimensional array of positive and
            negative values (where the first dimension is the "channel" dimension)
    """
    # TODO: implement long-range edge-probability
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    assert signed_affinities.shape[0] == len(offsets)
    image_shape = signed_affinities.shape[1:]

    mutex_edges = signed_affinities < 0
    abs_affinities = np.abs(signed_affinities)

    # we assume that we get a 'valid mask', i.e. a mask where valid regions are set true
    # and invalid regions are set to false.
    # for computation, we need the opposite though
    inv_mask = None if foreground_mask is None else np.logical_not(foreground_mask)
    valid_edges = get_valid_edges(signed_affinities.shape, offsets, number_of_attractive_channels=None,
                                  strides=None, randomize_strides=None, node_mask=inv_mask,
                                  edge_mask=edge_mask)


    # sort and flatten weights
    # ignore masked weights during sorting
    masked_weights = np.ma.masked_array(abs_affinities, mask=np.logical_not(valid_edges))
    sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]

    labels = compute_mws_segmentation_v2_impl(sorted_flat_indices,
                                           valid_edges.ravel(),
                                           mutex_edges.ravel(),
                                           offsets,
                                           image_shape)

    labels = labels.reshape(image_shape)
    # if we had an external mask, make sure it is mapped to zero
    if foreground_mask is not None:
        # increase labels by 1, so we don't merge anything with the mask
        labels += 1
        labels[inv_mask] = 0
    if return_valid_edge_mask:
        return labels, valid_edges
    else:
        return labels


def compute_mws_segmentation_from_affinities(affinities, offsets,
                                             beta_parameter=0.5,
                                             foreground_mask=None, edge_mask=None,
                                             return_valid_edge_mask=False):
    """
    :param beta_parameter: Increase the parameter up to 1.0 to obtain a result biased towards over-segmentation.
                Decrease it down to 0. to obtain a result biased towards under-segmentation

    """
    return compute_mws_segmentation_from_signed_affinities(affinities-beta_parameter, offsets,
                                                           foreground_mask=foreground_mask, edge_mask=edge_mask,
                                                           return_valid_edge_mask=return_valid_edge_mask)
