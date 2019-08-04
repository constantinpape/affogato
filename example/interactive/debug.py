import time
import h5py
from affogato.segmentation import InteractiveMWS


def debug():
    z = 0
    path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
    with h5py.File(path, 'r') as f:
        # raw = f['raw'][z]
        affs = f['prediction'][:, z]

    strides = [4, 4]
    offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3],
               [-9, 0], [0, -9], [-27, 0], [0, -27]]

    with h5py.File('./seeds.h5') as f:
        seeds = f['data'][:]
    assert seeds.shape == affs.shape[1:]

    imws = InteractiveMWS(affs, offsets, n_attractive_channels=2,
                          strides=strides, randomize_strides=True)

    print("Compute segmentation without seeds ...")
    t0 = time.time()
    seg1 = imws()
    t0 = time.time() - t0
    print("... done in %f s" % t0)

    print("Add seeds ...")
    t0 = time.time()
    imws.update_seeds(seeds)
    t0 = time.time() - t0
    print("... done in %f s" % t0)

    print("Compute segmentation with seeds ...")
    t0 = time.time()
    seg2 = imws()
    t0 = time.time() - t0
    print("... done in %f s" % t0)

    assert seg1.shape == seg2.shape == seeds.shape


if __name__ == '__main__':
    debug()
