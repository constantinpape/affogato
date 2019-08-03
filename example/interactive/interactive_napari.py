import h5py
from affogato.interactive.napari import interactive_napari_mws


def interactive_napari():
    z = 0
    path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
    with h5py.File(path, 'r') as f:
        raw = f['raw'][z]
        affs = f['prediction'][:, z]

    strides = [4, 4]
    offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3],
               [-9, 0], [0, -9], [-27, 0], [0, -27]]
    interactive_napari_mws(raw, affs, offsets,
                           strides=strides, randomize_strides=True)


if __name__ == '__main__':
    interactive_napari()
