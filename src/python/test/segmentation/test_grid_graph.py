import unittest
import numpy as np


class TestGridGraph(unittest.TestCase):
    def test_grid_graph(self):
        from affogato.segmentation.causal_mws import MWSGridGraph

        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, -9]]

        shape = (len(offsets), 100, 100)
        affs = np.random.rand(*shape).astype('float32')
        g = MWSGridGraph(affs.shape[1:])
        g.compute_weights_and_nh_from_affs(affs, offsets)

        uv_ids = g.uv_ids()
        self.assertGreater(uv_ids.shape[0], 1)
        self.assertEqual(uv_ids.shape[1], 2)


if __name__ == '__main__':
    unittest.main()