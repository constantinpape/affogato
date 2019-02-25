import numpy as np
import h5py
from cremi_tools.viewer.volumina import view
from affogato.segmentation import compute_causal_mws, compute_mws_segmentation


# TODO
def run_causal_mws(affs, fg, offsets):
    mask = fg > .5
    segmentation = compute_causal_mws(affs, offsets, mask)
    return segmentation


def merge_causal(seg, seg_prev, affs):
    labeled = seg > 0
    labeled_prev = seg_prev > 0
    mask = np.logical_and(labeled, labeled_prev)
    uvs = np.concatenate([seg[mask], seg_prev[mask]])
    uvs = np.unique(uvs, axis=1)

    merge_scores = np.zeros(len(uvs))
    for ii in range(len(uvs)):
        u, v = uvs[ii]
        # NOTE asume only straight spatial offset
        ovlp = np.logical_and(seg == u, seg_prev == v)
        scores[ii] = np.mean(affs[0][ovlp])

    # TODO
    # merge strongest mean aff if > .5
    return seg


def run_default_mws_2d(affs, fg, offsets):
    shape = fg.shape
    mask = fg > .5
    segmentation = np.zeros(shape, dtype='uint32')

    spatial_channels = [i for i, off in enumerate(offsets) if off[0] == 0]
    causal_channels = [i for i, off in enumerate(offsets) if off[0] != 0]

    spatial_offsets = [off for i, off in enumerate(offsets) if i in spatial_channels]
    causal_offsets = [off for i, off in enumerate(offsets) if i in causal_channels]

    for t in range(shape[0]):
        affs_t = affs[:, t]
        mask_t = mask[t]

        affs_spatial = affs_t[spatial_channels]

        seg = compute_mws_segmentation(2, affs_spatial, spatial_offsets,
                                       strides=strides, mask=mask_t)
        if t > 0:
            affs_causal = affs_t[causal_channels]
            seg_prev = segmentation[t - 1]
            max_id = int(seg_prev.max()) + 1
            seg[seg != 0] += max_id
            seg = merge_causal(seg, seg_prev, affs_causal, causal_offsets)

        segmentation[t] = seg
    return segmentation


if __name__ == '__main__':
    bb = np.s_[:5]

    path = '/home/pape/Work/data/CTC/DIC-C2DH-HeLa/val_data.h5'
    with h5py.File(path) as f:
        raw = f['raw'][bb]
        fg = f['fg'][bb]

        ds_affs = f['affs']
        ds_affs.n_threads = 8
        affs = ds_affs[(slice(None),) + (bb,)]

        offsets = ds_affs.attrs['offsets'].tolist()

    seg = run_causal_mws(affs, fg, offsets)
    # seg = run_default_mws_2d(affs, seg, offsets)
    view([raw, fg, affs.transpose((1, 2, 3, 0)), seg],
         ['raw', 'fg', 'affs', 'seg'])
