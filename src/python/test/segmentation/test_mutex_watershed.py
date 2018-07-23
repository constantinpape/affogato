import unittest
import numpy as np
from skimage.measure import label

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
        from affogato.segmentation import compute_mws_segmentation, compute_mws_prim_segmentation
        import h5py
        import scipy

        np.random.seed(100)
        number_of_attractive_channels = 2
        offsets = [[-1, 0], [0, -1], [-9, 0], [0, -9], [-9, -9], [9, -9],\
                [-9, -4], [-4, -9], [4, -9], [9, -4], [-27, 0], [0, -27]]
        # offsets = [[-1, 0], [0, -1], [-2, 0], [0, -2]]
        weights = np.random.rand(len(offsets), 10, 10)
        print (weights.shape)
        with h5py.File("im_111.h5", "r") as hpy:
            weights = np.array(hpy["data"].value.astype("float64"))
            weights[number_of_attractive_channels:] *= -1
            weights[number_of_attractive_channels:] += 1


        print (weights.shape)
        # compute mutex labeling
        node_labels = compute_mws_segmentation(number_of_attractive_channels,
                                               offsets,
                                               weights)
        number_of_colors = node_labels.size
        cmap = np.random.randint(255, size=(3, number_of_colors))
        scipy.misc.imsave(f"mws.png", np.moveaxis(cmap[:, node_labels], 0, -1))

        # for i in range(1, weights.size, 10):
        node_labels_prim = compute_mws_prim_segmentation(number_of_attractive_channels,
                                               offsets,
                                               weights)

        # seg_image = cmap[:,  np.repeat(np.repeat(node_labels_prim, 20, axis=0), 20, axis=1)]
        seg_image = cmap[:,  node_labels_prim]
        scipy.misc.imsave(f"it_{i:010}.png", np.moveaxis(seg_image, 0, -1))

        # with h5py.File("debug.h5", "w") as h5file:
        #     h5file.create_dataset("data", data=label(node_labels)) 
        #     h5file.create_dataset("data_prim", data=label(node_labels_prim)) 

        self.assertEqual(weights.shape[1:], node_labels.shape)
        np.testing.assert_array_equal(label(node_labels), label(node_labels_prim))


if __name__ == '__main__':
    unittest.main()
