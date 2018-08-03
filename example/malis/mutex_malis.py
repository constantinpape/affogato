import os
import numpy as np
import h5py
import affogato.learning as affl
import sys


OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
           # direct 3d nhood for attractive edges
           [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
           # indirect 3d nhood for dam edges
           [0, -9, 0], [0, 0, -9],
           # long range direct hood
           [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
           # inplane diagonal dam edges
           [0, -27, 0], [0, 0, -27]]


# TODO replace this with matplotlib functionality
def view_res(data, labels):
    sys.path.append('/home/cpape/Work/my_projects/cremi_tools')
    from cremi_tools.viewer.volumina import view
    view(data, labels)


def get_2d_from_3d_offsets(offsets):
    # only keep in-plane channels
    keep_channels = [ii for ii, off in enumerate(offsets) if off[0] == 0]
    offsets = [off[1:] for ii, off in enumerate(offsets) if ii in keep_channels]
    return offsets, keep_channels


def test_malis(path, aff_path):
    slice_ = np.s_[0, :512, :512]
    aff_slcie = (slice(None),) + slice_
    with h5py.File(aff_path) as f:
        affs = f['data'][aff_slcie]
    with h5py.File(path) as f:
        raw = f['volumes/raw'][slice_]
        gt = f['volumes/labels/neuron_ids_3d'][slice_]

    offsets, keep_channels = get_2d_from_3d_offsets(OFFSETS)
    affs = affs[keep_channels]
    affs[:2] *= -1
    affs[:2] += 1
    affs = 0.2 * affs + 0.8 * np.random.rand(*affs.shape)
    # affs = 0.5 * affs + 0.5 * np.random.rand(*affs.shape)
    _, grads, labels_pos, labels_neg = affl.mutex_malis(affs, gt, offsets, 2)

    print(grads.shape)
    data = [raw, gt,
            grads.transpose((1, 2, 0)),
            labels_pos, labels_neg]
    labels = ['raw', 'gt', 'gradients',
              'labels_pos', 'labels_neg']
    view_res(data, labels)


if __name__ == '__main__':
    top_dir = '/home/cpape/Work/data/isbi2012'
    path = os.path.join(top_dir, 'isbi2012_train_volume.h5')
    aff_path = os.path.join(top_dir,
                            'isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5')
    test_malis(path, aff_path)
