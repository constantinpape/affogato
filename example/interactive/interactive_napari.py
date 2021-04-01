import argparse
import json
import h5py
from affogato.interactive.napari import InteractiveNapariMWS

DEFAULT_STRIDES = [4, 4]
DEFAULT_OFFSETS = [[-1, 0], [0, -1],
                   [-3, 0], [0, -3],
                   [-9, 0], [0, -9],
                   [-27, 0], [0, -27]]


# path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
def interactive_napari(path, raw='raw', prediction='prediction',
                       strides=DEFAULT_STRIDES, offsets=DEFAULT_OFFSETS,
                       z=0):
    with h5py.File(path, 'r') as f:
        raw = f['raw'][z]
        affs = f['prediction'][:, z]

    imws = InteractiveNapariMWS(raw, affs, offsets,
                                strides=strides, randomize_strides=True)
    imws.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="Path to the input data (hdf5 file).")
    parser.add_argument('-r', '--raw', default='raw', help="Raw data key.")
    parser.add_argument('-p', '--prediction', default='prediction', help="Affinity prediction data key.")
    parser.add_argument('-o', '--offsets', default=None, help="Affinity offsets (json encoded)")
    parser.add_argument('-s', '--strides', default=None, help="Strides (json encoded)")
    parser.add_argument('-z', '--slice', default=0, type=int, help="Slice for 3d dataset")
    args = parser.parse_args()

    strides = args.strides
    if strides is None:
        strides = DEFAULT_STRIDES
    else:
        strides = json.loads(strides)

    offsets = args.offsets
    if offsets is None:
        offsets = DEFAULT_OFFSETS
    else:
        offsets = json.loads(offsets)

    interactive_napari(args.input, args.raw, args.prediction,
                       offsets=offsets, strides=strides,
                       z=args.slice)
