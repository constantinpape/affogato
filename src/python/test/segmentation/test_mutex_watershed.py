import unittest
from functools import partial
import numpy as np

try:
    from scipy.ndimage import convolve
    WITH_SCIPY = True
except ImportError:
    WITH_SCIPY = False


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

    def seg2edges(self, segmentation):
        gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
        gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
        return ((gx ** 2 + gy ** 2) > 0)

    @unittest.skipUnless(WITH_SCIPY, "Need scipy to compare segmentations")
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
        self.assertEqual(labels1.shape, labels2.shape)
        # we compare the segmentations by checking that their aedges agree
        edges1 = self.seg2edges(labels1)
        edges2 = self.seg2edges(labels2)
        print(np.isclose(edges1, edges2).sum(), '/', edges1.size)
        self.assertTrue(np.allclose(edges1, edges2))


if __name__ == '__main__':
    unittest.main()
