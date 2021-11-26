import unittest
import numpy as np


# TODO check correctness for toy data
class TestConnectedComponents(unittest.TestCase):

    # checks are not correct
    @unittest.expectedFailure
    def test_cc_2d(self):
        from affogato.segmentation import connected_components
        shape = (2, 100, 100)
        affs = np.random.rand(*shape)
        ccs, max_label = connected_components(affs, 0.5)
        self.assertEqual(ccs.shape, shape[1:])
        self.assertGreater(max_label, 10)
        self.assertEqual(max_label, ccs.max())

    # checks are not correct
    # @unittest.expectedFailure
    @unittest.skip("Segfaults on windows")
    def test_cc_3d(self):
        from affogato.segmentation import connected_components
        shape = (2, 100, 100, 100)
        affs = np.random.rand(*shape)
        ccs, max_label = connected_components(affs, 0.5)
        self.assertEqual(ccs.shape, shape[1:])
        self.assertGreater(max_label, 10)
        self.assertEqual(max_label, ccs.max())


if __name__ == '__main__':
    unittest.main()
