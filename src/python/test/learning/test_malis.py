import unittest
import numpy as np


# TODO check correctness for toy data
# TODO check ignore data
class TestMalis(unittest.TestCase):

    def test_malis_2d(self):
        from affogato.affinities import compute_affinities
        from affogato.learning import compute_malis_2d
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0], [0, -1]]
        affs, _ = compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        loss, grads = compute_malis_2d(affs, labels, offsets)
        self.assertEqual(grads.shape, affs.shape)
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))

    def test_malis_3d(self):
        from affogato.affinities import compute_affinities
        from affogato.learning import compute_malis_3d
        shape = (32, 64, 64)
        labels = np.random.randint(0, 1000, size=shape)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        affs, _ = compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        loss, grads = compute_malis_3d(affs, labels, offsets)
        self.assertEqual(grads.shape, affs.shape)
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))

    def seg2edges(self, segmentation):
        from scipy.ndimage import convolve
        gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
        gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
        return ((gx ** 2 + gy ** 2) > 0)

    def test_malis_2d_gradient_descent(self):
        from affogato.learning import compute_malis_2d
        from affogato.segmentation import connected_components
        shape = (100, 100)
        labels = np.zeros(shape)
        for i in range(10):
            for j in range(10):
                labels[10 * i:10 * (i + 1), 10 * j:10 * (j + 1)] = 10 * i + j + 1

        affs = 0.5 * np.ones((2, 100, 100))

        offsets = [[-1, 0], [0, -1]]

        for epoch in range(40):
            loss, grads = compute_malis_2d(affs, labels, offsets)
            affs -= 10000 * grads
            affs = np.clip(affs, 0, 1)

        labels1, _ = connected_components(affs, 0.5)

        self.assertEqual(grads.shape, affs.shape)
        self.assertTrue(np.allclose(grads, 0))
        self.assertEqual(loss, 0)

        edges1 = self.seg2edges(labels1)
        edges2 = self.seg2edges(labels)
        self.assertTrue(np.allclose(edges1, edges2))


if __name__ == '__main__':
    unittest.main()
