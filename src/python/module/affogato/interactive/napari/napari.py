# import os
import numpy as np
import h5py
import napari
from ...segmentation import InteractiveMWS


def _print_help():
    print("Interactive Mutex Watershed Application")
    print("Keybindigns:")
    print("[u] update segmentation")
    print("[s] save current segmentation to h5")
    print("[v] save current seeds to h5")
    print("[h] show help")


# TODO with auto completion
# https://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input
# https://gist.github.com/iamatypeofwalrus/5637895
def _read_file_path(path):
    if path is not None:
        inp = input("Do you want to keep the save path and override the result? [y] / n: ")
        if inp != 'n':
            return path
    path = input("Enter save path: ")
    # TODO check for valid save folder
    # save_folder = os.path.split(path)[0]
    # if not os.path.exists(save_folder):
    #     raise RuntimeError("Invalid folder %s" % save_folder)
    return path


def _save(path, data):
    # TODO don't use 'w', but check if data exists instead
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip')


class InteractiveNapariMWS:

    def __init__(self,
                 raw,
                 affs,
                 offsets,
                 strides=None,
                 randomize_strides=True):

        ndim = len(offsets[0])
        assert raw.ndim == ndim
        assert affs.ndim == ndim + 1
        assert ndim in (2, 3)

        self.raw = raw
        self.imws = InteractiveMWS(affs, offsets, n_attractive_channels=ndim,
                                   strides=strides, randomize_strides=randomize_strides)

        self.run()

    def run(self):
        # get the initial mws segmentation
        seg = self.imws()

        # initialize save paths for segmentation and seeds
        seg_path = None
        seed_path = None
        _print_help()

        # add initial layers to the viewer
        with napari.gui_qt():
            viewer = napari.Viewer()

            # add image layers and point layer for seeds
            viewer.add_image(self.raw, name='raw')
            viewer.add_labels(seg, name='segmentation')
            viewer.add_labels(np.zeros_like(seg), name='seeds')

            # add key-bindings

            # update segmentation by re-running mws
            @viewer.bind_key('u')
            def update_mws(viewer):
                self.update_mws_impl(viewer)

            # save the current segmentation
            @viewer.bind_key('s')
            def save_segmentation(viewer):
                nonlocal seg_path
                seg_path = _read_file_path(seg_path)
                seg = viewer.layers['segmentation'].data
                _save(seg_path, seg)

            # save the current seeds
            @viewer.bind_key('v')
            def save_seeds(viewer):
                nonlocal seed_path
                seed_path = _read_file_path(seed_path)
                seeds = viewer.layers['seeds'].data
                _save(seed_path, seeds)

            # save the current seeds
            @viewer.bind_key('t')
            def training_step(viewer):
                self.training_step_impl(viewer)

            # display help
            @viewer.bind_key('h')
            def print_help(viewer):
                _print_help()


    def training_step_impl(self, viewer):
        pass

    def update_mws_impl(self, viewer):
        print("Update mws triggered")
        layers = viewer.layers
        seeds = layers['seeds'].data

        seg_layer = layers['segmentation']
        print("Clearing seeds ...")
        self.imws.clear_seeds()
        # FIXME this takes much to long, something is wrong here
        print("Updating seeds ...")
        self.imws.update_seeds(seeds)
        print("Recomputing segmentation from seeds ...")
        seg = self.imws()
        print("... done")
        seg_layer.data = seg
        seg_layer.refresh()