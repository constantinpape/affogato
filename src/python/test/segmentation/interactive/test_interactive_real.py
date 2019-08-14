import unittest
from itertools import product
import numpy as np
import h5py


class TestInteractiveMwsReal(unittest.TestCase):
    path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
    path_seeds = '/home/pape/Work/data/ilastik/mulastik/data/seeds.h5'
    strides = [6, 6]
    offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3],
               [-9, 0], [0, -9], [-27, 0], [0, -27]]

    # bounding box to speed up test
    bb = np.s_[:, :]
    # bb = np.s_[:512, :512]

    def test_imws_seeds(self):
        from affogato.segmentation import InteractiveMWS

        z = 0
        with h5py.File(self.path, 'r') as f:
            bb_raw = (z,) + self.bb
            raw = f['raw'][bb_raw]
            bb_affs = (slice(None), z) + self.bb
            affs = f['prediction'][bb_affs]
        with h5py.File(self.path_seeds, 'r') as f:
            seeds = f['data'][self.bb]

        self.assertEqual(raw.shape, seeds.shape)
        self.assertEqual(raw.shape, affs.shape[1:])
        self.assertEqual(len(self.offsets), affs.shape[0])

        imws = InteractiveMWS(affs, self.offsets, strides=self.strides,
                              randomize_strides=True)
        print("Add seeds ....")
        imws.update_seeds(seeds)
        print("Run mws ...")
        seg = imws()

        self.assertEqual(seg.shape, seeds.shape)
        seed_ids = np.unique(seeds)[1:]
        print("Found seeds", seed_ids)

        print("Check results ...")
        # make sure individual seeds are mapped to the same segment id
        for seed_id in seed_ids:
            seed_mask = seeds == seed_id
            seg_ids = np.unique(seg[seed_mask])
            self.assertEqual(len(seg_ids), 1)

        # make sure different seeds are mapped to different segment ids
        for seed_a, seed_b in product(seed_ids, seed_ids):
            if seed_a >= seed_b:
                continue
            print("Checking seed pair", seed_a, seed_b)
            mask_a = seeds == seed_a
            mask_b = seeds == seed_b

            id_a = np.unique(seg[mask_a])[0]
            id_b = np.unique(seg[mask_b])[0]
            self.assertNotEqual(id_a, id_b)


if __name__ == '__main__':
    unittest.main()
