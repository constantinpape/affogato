import unittest
import numpy as np


class TestMultiscaleAffinities(unittest.TestCase):

    def test_ms_affs_2d(self):
        from affogato.affinities import compute_multiscale_affinities
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2], [10, 10], [5, 5]]
        for block_shape in block_shapes:
            affs, mask = compute_multiscale_affinities(labels, block_shape)
            expected_shape = (2,) + tuple(sh // bs + 1 if sh % bs else sh // bs
                                          for sh, bs in zip(shape, block_shape))
            self.assertEqual(affs.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)
            self.assertNotEqual(np.sum(affs == 0), 0)
            self.assertNotEqual(np.sum(mask == 0), 0)

    def test_ms_affs_ignore_2d(self):
        from affogato.affinities import compute_multiscale_affinities
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2], [10, 10], [5, 5]]
        for block_shape in block_shapes:
            affs, mask = compute_multiscale_affinities(labels, block_shape, True, 0)
            expected_shape = (2,) + tuple(sh // bs + 1 if sh % bs else sh // bs
                                          for sh, bs in zip(shape, block_shape))
            self.assertEqual(affs.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)
            self.assertNotEqual(np.sum(affs == 0), 0)
            self.assertNotEqual(np.sum(mask == 0), 0)

    def test_ms_affs_3d(self):
        from affogato.affinities import compute_multiscale_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2, 2], [10, 10, 10], [5, 5, 1]]
        for block_shape in block_shapes:
            affs, mask = compute_multiscale_affinities(labels, block_shape)
            expected_shape = (3,) + tuple(sh // bs + 1 if sh % bs else sh // bs
                                          for sh, bs in zip(shape, block_shape))
            self.assertEqual(affs.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)
            self.assertNotEqual(np.sum(affs == 0), 0)
            self.assertNotEqual(np.sum(mask == 0), 0)

    def test_ms_affs_ignore_3d(self):
        from affogato.affinities import compute_multiscale_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2, 2], [10, 10, 10], [5, 5, 1]]
        for block_shape in block_shapes:
            affs, mask = compute_multiscale_affinities(labels, block_shape, True, 0)
            expected_shape = (3,) + tuple(sh // bs + 1 if sh % bs else sh // bs
                                          for sh, bs in zip(shape, block_shape))
            self.assertEqual(affs.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)
            self.assertNotEqual(np.sum(affs == 0), 0)
            self.assertNotEqual(np.sum(mask == 0), 0)


if __name__ == '__main__':
    unittest.main()
