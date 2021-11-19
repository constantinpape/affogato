import h5py
from affogato.segmentation import compute_mws_segmentation


OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
           [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
           [0, -9, 0], [0, 0, -9],
           [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
           [0, -27, 0], [0, 0, -27]]


def get_2d_from_3d_offsets(offsets):
    # only keep in-plane channels
    keep_channels = [ii for ii, off in enumerate(offsets) if off[0] == 0]
    offsets = [off[1:] for ii, off in enumerate(offsets) if ii in keep_channels]
    return offsets, keep_channels


def create_test_data_2d(path):
    with h5py.File(path, "r") as f:
        affs = f["affinities"][:, 0]
    assert affs.shape[0] == len(OFFSETS)
    offsets, keep_channels = get_2d_from_3d_offsets(OFFSETS)
    affs = affs[keep_channels]
    seperating_channel = 2
    affs[:seperating_channel] *= -1
    affs[:seperating_channel] += 1
    seg = compute_mws_segmentation(affs, offsets,
                                   number_of_attractive_channels=seperating_channel, strides=None)
    assert affs.shape[0] == len(offsets)

    # check the results
    import napari
    v = napari.Viewer()
    v.add_image(affs)
    v.add_labels(seg)
    napari.run()

    with h5py.File("../data/test_data_2d.h5", "w") as f:
        f.create_dataset("affinities", data=affs, compression="gzip")
        f.create_dataset("segmentation", data=seg, compression="gzip")
        f.attrs["offsets"] = offsets


def create_test_data_3d(path):
    with h5py.File(path, "r") as f:
        affs = f["affinities"][:, :4, :256, :256]
    assert affs.shape[0] == len(OFFSETS)
    seperating_channel = 3
    affs[:seperating_channel] *= -1
    affs[:seperating_channel] += 1
    offsets = OFFSETS
    seg = compute_mws_segmentation(affs, offsets,
                                   number_of_attractive_channels=seperating_channel, strides=None)

    assert affs.shape[0] == len(offsets)

    # check the results
    import napari
    v = napari.Viewer()
    v.add_image(affs)
    v.add_labels(seg)
    napari.run()

    with h5py.File("../data/test_data_3d.h5", "w") as f:
        f.create_dataset("affinities", data=affs, compression="gzip")
        f.create_dataset("segmentation", data=seg, compression="gzip")
        f.attrs["offsets"] = offsets


if __name__ == "__main__":
    # The example data from https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz
    path = "/home/pape/Work/data/isbi/isbi_test_volume.h5"
    create_test_data_2d(path)
    create_test_data_3d(path)
