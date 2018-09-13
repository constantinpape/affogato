import numpy as np
from ._affinities import *

try:
    from scipy import ndimage
    WITH_SCIPY = True
except ImportError:
    WITH_SCIPY = False


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
