import unittest
import numpy as np


class TestGridGraph(unittest.TestCase):
    def test_grid_graph(self):
        from affogato.segmentation.causal_mws import MWSGridGraph

        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, -9]]

        shape = (len(offsets), 100, 100)
        affs = np.random.rand(*shape).astype('float32')
        g = MWSGridGraph(affs.shape[1:])
        uv_ids, weights = g.compute_nh_and_weights(affs, offsets)

        self.assertGreater(uv_ids.shape[0], 1)
        self.assertEqual(uv_ids.shape[1], 2)

    def test_grid_graph_3d(self):
        from affogato.segmentation.causal_mws import MWSGridGraph

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3],
                   [-9, -9, 9]]

        shape = (len(offsets), 100, 100, 100)
        affs = np.random.rand(*shape).astype('float32')
        g = MWSGridGraph(affs.shape[1:])
        uv_ids, weights = g.compute_nh_and_weights(affs, offsets)

        self.assertGreater(uv_ids.shape[0], 1)
        self.assertEqual(uv_ids.shape[1], 2)

    def test_nodes_and_coords(self):
        from affogato.segmentation.causal_mws import MWSGridGraph
        shape = (32, 100, 100)
        g = MWSGridGraph(shape)

        # check node -> coord -> node
        n_nodes = g.n_nodes
        for node in range(n_nodes):
            coord = g.get_coordinate(node)
            node_out = g.get_node(coord)
            self.assertEqual(node, node_out)

        # check coord -> node -> coord
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    coord = [z, y, x]
                    node = g.get_node(coord)
                    coord_out = g.get_coordinate(node)
                    self.assertEqual(coord, coord_out)

    def test_nodes_and_coords_vectorized(self):
        from affogato.segmentation.causal_mws import MWSGridGraph
        shape = (32, 100, 100)
        g = MWSGridGraph(shape)

        n_nodes = g.n_nodes
        nodes = np.arange(n_nodes, dtype='uint64')
        # coords =

        # check nodes -> coords -> nodes
        coords_out = g.get_coordinates(nodes)
        # TODO  check the actual values
        # self.assertTrue(np.allclose(coords, coords_out))
        self.assertEqual(coords_out.shape, (len(nodes), 3))

        # check coords -> nodes -> coords
        # TODO




if __name__ == '__main__':
    unittest.main()
