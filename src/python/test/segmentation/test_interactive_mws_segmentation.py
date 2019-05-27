import unittest
import numpy as np


# TODO check correctness for toy data
class TestInteractiveMws(unittest.TestCase):
    shape = (128, 128)
    offsets = [[-1, 0], [0, -1],
               [-3, 0], [0, -3],
               [-9, 0], [0, -9]]

    def _make_imws(self):
        from affogato.segmentation import InteractiveMWS

        aff_shape = (len(self.offsets),) + self.shape
        affs = np.random.rand(*aff_shape).astype('float32')
        imws = InteractiveMWS(affs, self.offsets)
        return imws

    def test_interactive_mws(self):
        imws = self._make_imws()
        seg = imws()
        self.assertEqual(seg.shape, self.shape)


if __name__ == '__main__':
    unittest.main()
