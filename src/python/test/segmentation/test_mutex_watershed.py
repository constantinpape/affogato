import unittest
import numpy as np


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

    # test mutex watershed segmentation
    # with random edges and random mutex edges
    def test_mws_segmentation_random_weights(self):
        from affogato.segmentation import compute_mws_segmentation

        number_of_attractive_channels = 2
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, 3], [5, 5]]
        weights = np.random.rand(len(offsets), 100, 100)

        # compute mutex labeling
        node_labels = compute_mws_segmentation(number_of_attractive_channels,
                                               offsets,
                                               weights)
        self.assertEqual(weights.shape[1:], node_labels.shape)


if __name__ == '__main__':
    unittest.main()
