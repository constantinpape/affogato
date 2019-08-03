import napari
# from ...segmentation import InteractiveMWS


def napari_mws_2d(raw, imws):
    # get the initial mws segmentation
    # seg = imws()

    # add initial layers to the viewer
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_layer(raw)
        # viewer.add_layer(seg)
        # viewer = napari.view(raw)


def interactive_napari_mws(raw, affs, offsets,
                           strides=None, randomize_strides=False):
    ndim = len(offsets[0])
    assert raw.ndim == ndim
    assert affs.ndim == ndim + 1
    assert ndim in (2, 3)

    # imws = InteractiveMWS(affs, offsets, n_attractive_channels=ndim,
    #                       strides=strides, randomize_strides=randomize_strides)
    imws = None

    if ndim == 2:
        napari_mws_2d(raw, imws)
    else:
        assert False
        # TODO implement 3d
        # napari_mws_3d()
