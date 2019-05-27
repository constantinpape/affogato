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

    def test_interactive_mws_with_seeds(self):
        imws = self._make_imws()
        # update seeds from dict
        seeds = {1: (np.array([0, 0, 0], dtype='int'), np.array([1, 2, 3], dtype='int')),
                 2: (np.array([4, 5, 6], dtype='int'), np.array([1, 2, 3], dtype='int'))}
        imws.update_seeds(seeds)
        seg = imws()
        self.assertEqual(seg.shape, self.shape)
        for seed_id, coords in seeds.items():
            out = seg[coords]
            self.assertTrue(np.allclose(out, seed_id))

    def test_geojson_parser(self):
        from affogato.segmentation.interactive_mws import parse_geojson
        imws = self._make_imws()
        geojson = {'type': 'FeatureCollection',
                   'features': [
                                {'type': 'Feature',
                                 'geometry': 'Point',
                                 'coordinates': [0.1, 1.0],
                                 'properties': {'name': '1'}
                                },
                                {'type': 'Feature',
                                 'geometry': 'LineString',
                                 'coordinates': [[4., 1.], [5., 2.]],
                                 'properties': {'name': '2'}}
                               ]
                  }
        geo_seeds = parse_geojson(geojson)
        # update seeds from dict
        imws.update_seeds(geo_seeds)
        seg = imws()
        self.assertEqual(seg.shape, self.shape)

        seeds = {1: (np.array([0], dtype='int'), np.array([1], dtype='int')),
                 2: (np.array([4, 5], dtype='int'), np.array([1, 2], dtype='int'))}
        for seed_id, coords in seeds.items():
            out = seg[coords]
            self.assertTrue(np.allclose(out, seed_id))


if __name__ == '__main__':
    unittest.main()
