import unittest
import numpy as np


class TestAffinities(unittest.TestCase):

    def test_affs_2d(self):
        from affogato.affinities import compute_affinities
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0], [0, -1],
                   [-5, 0], [0, -5],
                   [10, 10], [3, 9]]

        affs, mask = compute_affinities(labels, offsets)
        expected_shape = (len(offsets),) + labels.shape
        self.assertEqual(affs.shape, expected_shape)
        self.assertEqual(mask.shape, expected_shape)
        self.assertNotEqual(np.sum(affs == 0), 0)
        self.assertNotEqual(np.sum(mask == 0), 0)

    def test_affs_ignore_2d(self):
        from affogato.affinities import compute_affinities
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0], [0, -1],
                   [-5, 0], [0, -5],
                   [10, 10], [3, 9]]

        affs, mask = compute_affinities(labels, offsets, ignore_label=0)
        expected_shape = (len(offsets),) + labels.shape
        self.assertEqual(affs.shape, expected_shape)
        self.assertEqual(mask.shape, expected_shape)
        self.assertNotEqual(np.sum(affs == 0), 0)
        self.assertNotEqual(np.sum(mask == 0), 0)

    def test_affs_3d(self):
        from affogato.affinities import compute_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-5, 0, 0], [0, -5, 0], [0, 0, -5],
                   [10, 10, 10], [3, 9, 27], [0, 9, 8]]

        affs, mask = compute_affinities(labels, offsets)
        expected_shape = (len(offsets),) + labels.shape
        self.assertEqual(affs.shape, expected_shape)
        self.assertEqual(mask.shape, expected_shape)
        self.assertNotEqual(np.sum(affs == 0), 0)
        self.assertNotEqual(np.sum(mask == 0), 0)

    def test_affs_ignore_3d(self):
        from affogato.affinities import compute_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-5, 0, 0], [0, -5, 0], [0, 0, -5],
                   [10, 10, 10], [3, 9, 27], [0, 9, 8]]

        affs, mask = compute_affinities(labels, offsets, ignore_label=0)
        expected_shape = (len(offsets),) + labels.shape
        self.assertEqual(affs.shape, expected_shape)
        self.assertEqual(mask.shape, expected_shape)
        self.assertNotEqual(np.sum(affs == 0), 0)
        self.assertNotEqual(np.sum(mask == 0), 0)

    def test_affs_with_extra_masks_3d(self):
        from affogato.affinities import compute_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        boundary_mask = np.random.randint(0, 1, size=shape, dtype='bool')
        glia_mask = np.random.randint(0, 1, size=shape, dtype='bool')
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-5, 0, 0], [0, -5, 0], [0, 0, -5],
                   [10, 10, 10], [3, 9, 27], [0, 9, 8]]

        affs, mask = compute_affinities(labels, offsets, ignore_label=0,
                                        boundary_mask=boundary_mask,
                                        glia_mask=glia_mask)
        expected_shape = (len(offsets),) + labels.shape
        self.assertEqual(affs.shape, expected_shape)
        self.assertEqual(mask.shape, expected_shape)
        self.assertNotEqual(np.sum(affs == 0), 0)
        self.assertNotEqual(np.sum(mask == 0), 0)


if __name__ == '__main__':
    unittest.main()
