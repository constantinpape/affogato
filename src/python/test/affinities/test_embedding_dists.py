import unittest
import numpy as np


class TestEmbeddingDstances(unittest.TestCase):

    def test_embed_dist(self):
        from affogato.affinities import compute_embedding_distances
        shape = (12, 100, 100)
        values = np.random.rand(*shape).astype('float32')
        offsets = [[-1, 0], [0, -1],
                   [-6, 0], [0, -6],
                   [-12, 0], [0, -12]]

        for norm in ('l2', 'cosine'):
            dist = compute_embedding_distances(values, offsets, norm=norm)
            expected_shape = (len(offsets),) + values.shape[1:]
            self.assertEqual(dist.shape, expected_shape)
            self.assertNotEqual(np.sum(dist == 0), 0)


if __name__ == '__main__':
    unittest.main()
