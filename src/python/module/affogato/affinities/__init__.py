import numpy as np
from ._affinities import (compute_affinities, compute_multiscale_affinities,
                          compute_embedding_distances_l2, compute_embedding_distances_cos)

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


def affinities_with_masked_ignore_transition(segmentation, offsets, ignore_label=0):
    # compute the regular affinities with mask for the ignore label
    affs, mask = compute_affinities(segmentation, offsets,
                                    have_ignore_label=True, ignore_label=ignore_label)
    # get a mask for the transition to ignore label, by computing the affinities of the ignore mask
    ignore_mask = segmentation == ignore_label
    ignore_transitions, _ = compute_affinities(ignore_mask, offsets)
    ignore_transitions = ignore_transitions == 0
    # set the affinities in the ignore transition to repulsive (=0)
    # and set the mask to be valid
    affs[ignore_transitions] = 0
    mask[ignore_transitions] = 1
    return affs, mask


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
