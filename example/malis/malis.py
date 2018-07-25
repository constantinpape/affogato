import os
import numpy as np
import h5py
import affogato.learning as affl
import sys


# TODO replace this with matplotlib functionality
def view_res(data, labels):
    sys.path.append('/home/cpape/Work/my_projects/cremi_tools')
    from cremi_tools.viewer.volumina import view
    view(data, labels)


def test_malis(path, aff_path):
    slice_ = np.s_[0, :512, :512]
    aff_slcie = (slice(1, 3),) + slice_
    with h5py.File(aff_path) as f:
        affs = f['data'][aff_slcie]
    with h5py.File(path) as f:
        raw = f['volumes/raw'][slice_]
        gt = f['volumes/labels/neuron_ids_3d'][slice_]

    offsets = [[-1, 0], [0, -1]]
    _, grads = affl.compute_malis_2d(affs, gt, offsets, 0)
    bina = (grads != 0).astype('uint8')

    data = [raw, gt,
            affs.transpose((1, 2, 0)),
            grads.transpose((1, 2, 0)),
            bina.transpose((1, 2, 0))]
    labels = ['raw', 'gt', 'affinities',
              'gradients', 'binarized-gradients']
    view_res(data, labels)


if __name__ == '__main__':
    top_dir = '/home/cpape/Work/data/isbi2012'
    path = os.path.join(top_dir, 'isbi2012_train_volume.h5')
    aff_path = os.path.join(top_dir,
                            'isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5')
    test_malis(path, aff_path)
