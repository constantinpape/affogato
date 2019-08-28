import numpy as np
from ._segmentation import compute_mws_clustering, MWSGridGraph


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

        # TODO
        self._locked_seeds = {}

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    #
    # update the graph
    #

    def _update_graph(self):
        # compute the attractive edges
        # we set to > 1 to make sure these are the very first in priority
        self._grid_graph.add_attractive_seed_edges = True
        self._uvs, self._weights = self._grid_graph.compute_nh_and_weights(1. - self._affinities[:self._n_attractive],
                                                                           self._offsets[:self._n_attractive])

        # compute the repulsive edges
        self._grid_graph.add_attractive_seed_edges = False
        self._mutex_uvs, self._mutex_weights = self._grid_graph.compute_nh_and_weights(self._affinities[self._n_attractive:],
                                                                                       self._offsets[self._n_attractive:],
                                                                                       strides=self.strides,
                                                                                       randomize_strides=self.randomize_strides)
    #
    # seed functionality
    #

    # TODO we could also support a ROI
    def _updated_seeds_dense(self, new_seeds):
        if new_seeds.shape != self.shape:
            raise ValueError("Dense seeds have incorrect shape")
        seed_mask = new_seeds != 0
        self._seeds[seed_mask] = new_seeds[seed_mask]

    def _update_seeds_sparse(self, new_seeds):
        for seed_id, coords in new_seeds.items():
            self._seeds[coords] = seed_id

    def update_seeds(self, new_seeds):
        if isinstance(new_seeds, np.ndarray):
            self._updated_seeds_dense(new_seeds)
        elif isinstance(new_seeds, dict):
            self._update_seeds_sparse(new_seeds)
        else:
            raise ValueError("new_seeds must be np.ndarray or dict, got %s" % type(new_seeds))
        self._grid_graph.update_seeds(self._seeds)
        self._update_graph()

    def clear_seeds(self):
        self._grid_graph.clear_seeds()
        self._seeds = np.zeros(self.shape, dtype='uint64')

    def get_seeds(self):
        return self._seeds

    def get_segmentation_with_seeds(self, segmentation):
        seed_mask = self._seeds > 0
        seg_ids = np.unique(segmentation[seed_mask])
        unseeded = np.logical_not(np.isin(segmentation, seg_ids))
        segmentation[unseeded] = 0
        return segmentation

    #
    # segmentation functionality
    #

    def __call__(self):
        n_nodes = self._grid_graph.n_nodes
        # TODO if we have locked seeds / segments, we need to mask them here
        seg = compute_mws_clustering(n_nodes, self._uvs, self._mutex_uvs,
                                     self._weights, self._mutex_weights)
        seg = self._grid_graph.relabel_to_seeds(seg)
        return seg.reshape(self.shape)

    #
    # locked segment functionality
    #

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
        self._update_graph()

    # TODO return the locked segments
    def get_locked_segments(self):
        pass
