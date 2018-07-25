
import unittest
import numpy as np


class TestZwatershed(unittest.TestCase):

    # test mutex watershed segmentation
    # with random edges and random mutex edges
    def test_mws_segmentation_random_weights(self):
        from affogato.segmentation import compute_zws_segmentation

        weights = np.random.rand(2, 100, 100)

        # compute mutex labeling
        labels, n_labels = compute_zws_segmentation(weights, 0.2, 0.98, 0.1, 2)
        self.assertEqual(weights.shape[1:], labels.shape)
        self.assertFalse((labels == 0).all())
        self.assertEqual(labels.max() + 1, n_labels)


if __name__ == '__main__':
    unittest.main()

