import h5py
from affogato.interactive.napari import TrainableNapariMWS


def trainable_napari():
    z = 0
    path = '/home/swolf/local/data/mulastik/data.h5'
    with h5py.File(path, 'r') as f:
        raw = f['raw'][z]

    checkpoint = '/home/swolf/local/src/affogato/example/data/Affinity_Unet'

    strides = [4, 4]
    offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3],
               [-9, 0], [0, -9], [-27, 0], [0, -27]]

    TrainableNapariMWS(raw, checkpoint, offsets,
                       strides=strides, randomize_strides=True)


if __name__ == '__main__':
    trainable_napari()