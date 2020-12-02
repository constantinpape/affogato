import numpy as np
from ._segmentation import compute_mws_clustering, MWSGridGraph
from ..affinities import compute_affinities_with_lut


# TODO support a data backend like zarr etc.
class InteractiveMWS():
    def __init__(self, affinities, offsets, n_attractive_channels=None,
                 strides=None, randomize_strides=False):
        if len(offsets) != affinities.shape[0]:
            raise ValueError("Number offsets and affinity channels do not match")
        self._shape = affinities.shape[1:]
        # set the state (grid graph, affinities, seeds)
        self._grid_graph = MWSGridGraph(self.shape)
        self._affinities = affinities
        self._offsets = offsets
        self._seeds = np.zeros(self.shape, dtype='uint64')
        self._n_attractive = self.ndim if n_attractive_channels is None else n_attractive_channels
        # strides and randomization
        self.strides = [1] * self.ndim if strides is None else strides
        self.randomize_strides = randomize_strides
        # comppute the initial graph shape (= uv-ids, mutex-uv-ids, ...)
        self._update_graph()

        self._locked_seeds = set()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def max_seed_id(self):
        return self._seeds.max()

    @property
    def offsets(self):
        return self._offsets

    #
    # update the graph
    #

    def _update_graph(self, mask=None):
        if mask is not None:
            self._grid_graph.clear_mask()
            self._grid_graph.set_mask(mask)

        # compute the attractive edges
        # we set to > 1 to make sure these are the very first in priority
        self._grid_graph.add_attractive_seed_edges = True
        self._uvs, self._weights = self._grid_graph.compute_nh_and_weights(1. - self._affinities[:self._n_attractive],
                                                                           self._offsets[:self._n_attractive])

        # compute the repulsive edges
        self._grid_graph.add_attractive_seed_edges = False
        (self._mutex_uvs,
         self._mutex_weights) = self._grid_graph.compute_nh_and_weights(self._affinities[self._n_attractive:],
                                                                        self._offsets[self._n_attractive:],
                                                                        strides=self.strides,
                                                                        randomize_strides=self.randomize_strides)
    #
    # seed functionality
    #

    def _update_seeds_dense(self, new_seeds, seed_offset):
        if new_seeds.shape != self.shape:
            raise ValueError("Dense seeds have incorrect shape")
        seed_mask = new_seeds != 0
        self._seeds[seed_mask] = (new_seeds[seed_mask] + seed_offset)

    def _update_seeds_sparse(self, new_seeds, seed_offset):
        new_seeds_array = np.zeros_like(self._seeds)
        for seed_id, coords in new_seeds.items():
            new_id = seed_id + seed_offset
            self._seeds[coords] = new_id
            new_seeds_array[coords] = new_id
        return new_seeds_array

    def update_seeds(self, new_seeds, seed_offset=0):
        if isinstance(new_seeds, np.ndarray):
            self._update_seeds_dense(new_seeds, seed_offset)
        elif isinstance(new_seeds, dict):
            new_seeds = self._update_seeds_sparse(new_seeds, seed_offset)
        else:
            raise ValueError("new_seeds must be np.ndarray or dict, got %s" % type(new_seeds))
        self._grid_graph.update_seeds(new_seeds)

    def clear_seeds(self):
        self._grid_graph.clear_seeds()
        self._seeds = np.zeros(self.shape, dtype='uint64')

    def merge(self, seg, ida, idb):

        seg_mask = np.isin(seg, [ida, idb])
        bb = np.where(seg_mask)
        bb = tuple(slice(b.min(), b.max() + 1) for b in bb)
        seg_sub = seg[bb]

        # computing the affmask for the bounding box of the two segment ids
        keys = np.array([[min(ida, idb), max(ida, idb)]], dtype='uint64')
        vals = np.array([1.], dtype='float32')
        aff_mask, _ = compute_affinities_with_lut(seg_sub, self._offsets, keys, vals,
                                                  default_val=0)
        aff_mask = aff_mask.astype('bool')
        n_mask = aff_mask.sum()

        # we only need to change the affinities if there is something in the mask
        if n_mask > 0:
            bb = (slice(None),) + bb
            self._affinities[bb][aff_mask] = 0

        return n_mask

    #
    # segmentation functionality
    #

    def __call__(self, prev_seg=None):
        # if we are passed a previous segmentation, we use it
        # to mask with the locked_seeds
        if prev_seg is not None and self._locked_seeds:
            mask = ~np.isin(prev_seg, list(self._locked_seeds))
        else:
            mask = None

        self._update_graph(mask=mask)
        n_nodes = self._grid_graph.n_nodes
        seg = compute_mws_clustering(n_nodes, self._uvs, self._mutex_uvs,
                                     self._weights, self._mutex_weights)

        # retrieve the old segmentation
        if mask is not None:
            mask = ~mask
            seg[mask.ravel()] = (prev_seg[mask] + seg.max())

        seg = self._grid_graph.relabel_to_seeds(seg)
        return seg.reshape(self.shape)

    #
    # locked segment functionality
    #

    @property
    def locked_seeds(self):
        return self._locked_seeds

    def lock_seeds(self, locked_seeds):
        self._locked_seeds.update(locked_seeds)

    def unlock_seeds(self, unlock_seeds):
        self._locked_seeds.difference_update(unlock_seeds)

    @property
    def affinities(self):
        return self._affinities

    @affinities.setter
    def affinities(self, affs):
        self._affinities = affs

    #
    # tiktorch functionality
    #

    # TODO support a ROI
    def update_affinities(self, affinities):
        if affinities.shape[1:] != self.shape:
            raise ValueError("Invalid Shape")
        if affinities.shape[0] != len(self._offsets):
            raise ValueError("Invalid number of channels")
        self._affinities = affinities
