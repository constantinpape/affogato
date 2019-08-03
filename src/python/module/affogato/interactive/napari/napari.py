import numpy as np
import napari
from ...segmentation import InteractiveMWS


def napari_mws_2d(raw, imws):
    # get the initial mws segmentation
    seg = imws()

    # add initial layers to the viewer
    with napari.gui_qt():
        viewer = napari.Viewer()

        # add image layers and point layer for seeds
        viewer.add_image(raw, name='raw')
        viewer.add_labels(seg, name='segmentation')
        viewer.add_labels(np.zeros_like(seg), name='seeds')

        # add key-bindings

        # update segmentation by re-running mws
        @viewer.bind_key('u')
        def update_mws(viewer):
            print("Update mws triggered")
            layers = viewer.layers
            seeds = layers['seeds'].data
            seg_layer = layers['segmentation']
            print("Clearing seeds ...")
            imws.clear_seeds()
            # FIXME this takes much to long, something is wrong here
            print("Updating seeds ...")
            imws.update_seeds(seeds)
            print("Recomputing segmentation from seeds ...")
            seg = imws()
            print("... done")
            seg_layer.data = seg
            seg_layer.refresh()

        # update random colors of segmentation layer
        @viewer.bind_key('c')
        def update_random_colors(viewer):
            print("Update random colors")
            seg_layer = viewer.layers['segmentation']
            # TODO

        # save the current segmentation
        @viewer.bind_key('s')
        def save_segmentation(viewer):
            pass

        # save the current seeds
        @viewer.bind_key('v')
        def save_segmentation(viewer):
            pass

        # display help
        @viewer.bind_key('h')
        def print_help(viewer):
            pass


# TODO enable with seeds
def interactive_napari_mws(raw, affs, offsets,
                           strides=None, randomize_strides=False):
    ndim = len(offsets[0])
    assert raw.ndim == ndim
    assert affs.ndim == ndim + 1
    assert ndim in (2, 3)

    imws = InteractiveMWS(affs, offsets, n_attractive_channels=ndim,
                          strides=strides, randomize_strides=randomize_strides)

    if ndim == 2:
        napari_mws_2d(raw, imws)
    else:
        assert False
        # TODO implement 3d
        # napari_mws_3d()
