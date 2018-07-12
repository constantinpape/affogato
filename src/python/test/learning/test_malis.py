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
        # FIXME this fails
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))

    def test_malis_3d(self):
        from affogato.affinities import compute_affinities
        from affogato.learning import compute_malis_3d
        shape = (100, 100, 100)
        labels = np.random.randint(0, 1000, size=shape)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        affs, _ = compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        loss, grads = compute_malis_3d(affs, labels, offsets)
        self.assertEqual(grads.shape, affs.shape)
        # FIXME this fails
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))


if __name__ == '__main__':
    unittest.main()
