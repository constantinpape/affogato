import os
import unittest
from functools import partial

import numpy as np

try:
    import h5py
except Exception:
    h5py = None
try:
    from scipy.ndimage import convolve
except Exception:
    convolve = None


class TestMutexWatershed(unittest.TestCase):

    # test mutex watershed clustering on a graph
    # with random edges and random mutex edges
    def test_mws_clustering_random_graph(self):
        from affogato.segmentation import compute_mws_clustering
        number_of_labels = 500
        number_of_edges = 1000
        number_of_mutex_edges = 2000

        # random edges
        edges = np.random.randint(0, number_of_labels,
                                  size=(number_of_edges, 2),
                                  dtype='uint64')
        # filter for redundant entries
        edge_mask = edges[:, 0] != edges[:, 1]
        edges = edges[edge_mask]

        # random mutex edges
        mutex_edges = np.random.randint(0, number_of_labels,
                                        size=(number_of_mutex_edges, 2),
                                        dtype='uint64')
        # filter for redundant entries
        edge_mask = mutex_edges[:, 0] != mutex_edges[:, 1]
        mutex_edges = mutex_edges[edge_mask]

        # random weights
        edge_weights = np.random.rand(edges.shape[0])
        mutex_weights = np.random.rand(mutex_edges.shape[0])

        # compute mutex labeling
        node_labels = compute_mws_clustering(number_of_labels,
                                             edges, mutex_edges,
                                             edge_weights, mutex_weights)
        self.assertEqual(len(node_labels), number_of_labels)

    def random_weights_test(self, mws_impl):
        number_of_attractive_channels = 2
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, 3], [5, 5]]
        weights = np.random.rand(len(offsets), 100, 100)
        node_labels = mws_impl(weights, offsets,
                               number_of_attractive_channels)
        self.assertEqual(weights.shape[1:], node_labels.shape)
        self.assertFalse((node_labels == 0).all())

    # test mutex watershed segmentation
    # with random edges and random mutex edges
    def test_mws_segmentation_kruskal(self):
        from affogato.segmentation import compute_mws_segmentation
        self.random_weights_test(partial(compute_mws_segmentation, algorithm='kruskal'))

    def test_mws_segmentation_kruskal_strides(self):
        from affogato.segmentation import compute_mws_segmentation
        self.random_weights_test(partial(compute_mws_segmentation, algorithm='kruskal', strides=[2, 2]))

    # test mutex watershed segmentation
    # with random edges and random mutex edges
    def test_mws_segmentation_prim(self):
        from affogato.segmentation import compute_mws_segmentation
        self.random_weights_test(partial(compute_mws_segmentation, algorithm='prim'))

    def seg2edges_2d(self, segmentation):
        gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
        gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
        return ((gx ** 2 + gy ** 2) > 0)

    def seg2edges_3d(self, segmentation):
        gz = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
        gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
        gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
        return ((gx ** 2 + gy ** 2 + gz ** 2) > 0)

    def _check_segmentation(self, exp, seg):
        self.assertEqual(exp.shape, seg.shape)
        # we compare the segmentations by checking that their aedges agree
        if exp.ndim == 2:
            edges1 = self.seg2edges_2d(exp)
            edges2 = self.seg2edges_2d(seg)
        else:
            edges1 = self.seg2edges_3d(exp)
            edges2 = self.seg2edges_3d(seg)
        # print(np.isclose(edges1, edges2).sum(), '/', edges1.size)
        self.assertTrue(np.allclose(edges1, edges2))

    @unittest.skipIf(convolve is None, "Need scipy to compare segmentations")
    def test_mws_consistency(self):
        from affogato.segmentation import compute_mws_segmentation
        number_of_attractive_channels = 2
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, 3], [5, 5]]
        weights = np.random.rand(len(offsets), 100, 100)
        labels1 = compute_mws_segmentation(weights, offsets,
                                           number_of_attractive_channels,
                                           algorithm='kruskal')
        labels2 = compute_mws_segmentation(weights, offsets,
                                           number_of_attractive_channels,
                                           algorithm='prim')
        self._check_segmentation(labels1, labels2)

    def test_mws_masked(self):
        from affogato.segmentation import compute_mws_segmentation
        number_of_attractive_channels = 2
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, 3], [5, 5]]

        weights = np.random.rand(len(offsets), 100, 100)
        mask = np.ones((100, 100), dtype='bool')
        # exclude 10 % of pixel from foreground mask
        coords = np.where(mask)
        n_out = int(len(coords[0]) * .1)
        indices = np.random.permutation(len(coords[0]))[:n_out]
        coords = (coords[0][indices], coords[1][indices])
        mask[coords] = False

        node_labels = compute_mws_segmentation(weights, offsets,
                                               number_of_attractive_channels,
                                               mask=mask)
        self.assertEqual(weights.shape[1:], node_labels.shape)
        # make sure mask is all non-zero
        self.assertTrue((node_labels[mask] != 0).all())
        # make sure inv mask is all zeros
        self.assertTrue((node_labels[np.logical_not(mask)] == 0).all())

    # compare the mutex watershed segmentation results with a pre-computed reference solution (2d)
    @unittest.skipIf(convolve is None or h5py is None, "Need scipy to compare segmentations")
    def test_mws_reference_2d(self):
        from affogato.segmentation import compute_mws_segmentation
        test_path = os.path.join(os.path.split(__file__)[0], "../../../../data/test_data_2d.h5")
        with h5py.File(test_path, "r") as f:
            affs = f["affinities"][:]
            ref = f["segmentation"][:]
            offsets = f.attrs["offsets"]
        seg = compute_mws_segmentation(affs, offsets, 2, strides=None)
        self._check_segmentation(ref, seg)

    # compare the mutex watershed segmentation results with a pre-computed reference solution (3d)
    @unittest.skipIf(convolve is None or h5py is None, "Need scipy to compare segmentations")
    def test_mws_reference_3d(self):
        from affogato.segmentation import compute_mws_segmentation
        test_path = os.path.join(os.path.split(__file__)[0], "../../../../data/test_data_3d.h5")
        with h5py.File(test_path, "r") as f:
            affs = f["affinities"][:]
            ref = f["segmentation"][:]
            offsets = f.attrs["offsets"]
        seg = compute_mws_segmentation(affs, offsets, 3, strides=None)
        self._check_segmentation(ref, seg)


if __name__ == '__main__':
    unittest.main()
