import numpy as np
from ._affinities import *

try:
    from scipy import ndimage
    WITH_SCIPY = True
except ImportError:
    WITH_SCIPY = False


def compute_embedding_distances(values, offsets, norm='l2'):
    if norm == 'l2':
        return compute_embedding_distances_l2(values, offsets)
    elif norm == 'cosine':
        return compute_embedding_distances_cos(values, offsets)
    else:
        raise ValueError("Invalid norm %s" % norm)


def compute_affinities_with_glia(labels, offset,
                                 ignore_label=None,
                                 boundary_label=None,
                                 glia_label=None):

    have_ignore_label = ignore_label is not None
    have_boundary_label = boundary_label is not None
    have_glia_label = glia_label is not None
    ignore_label = ignore_label if ignore_label is not None else 0
    boundary_label = boundary_label if boundary_label is not None else 0
    glia_label = glia_label if glia_label is not None else 0

    return compute_affinities_with_glia_impl_(labels, offset,
                                         have_ignore_label, have_boundary_label, have_glia_label,
                                         ignore_label, boundary_label, glia_label)


if WITH_SCIPY:

    def affinity_distance_transform(affinities, clip_limit_short=100., clip_limit_long=30):
        aff_distances = affinities.copy()

        ndims = affinities.ndim - 1
        nchannels = affinities.shape[0]

        # invert the affinities
        aff_distances = 1. - aff_distances

        # invert the short range channels (why?)
        aff_distances[0:ndims] = 1 - aff_distances[0:ndims]
        # compute the distance transforms
        for i in range(nchannels):
            aff_distances[i] = ndimage.distance_transform_edt(aff_distances[i])

        # TODO I don't understand what is going on here ...
        aff_distances = -aff_distances
        aff_distances[0:ndims] = 1 + np.clip(aff_distances[0:ndims], -clip_limit_short, None) / clip_limit_short
        aff_distances[ndims:] = 1 + np.clip(aff_distances[ndims:], -clip_limit_long, None) / clip_limit_long

        # I guess we need this only for mws ?!
        aff_distances[ndims:] = 1 - aff_distances[ndims:]
        return aff_distances
