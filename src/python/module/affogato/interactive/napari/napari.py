# import os
from itertools import product
import numpy as np
import h5py

import napari
from vispy.color import Colormap

from ...segmentation import InteractiveMWS

# TODO don't use elf functionality
from elf.segmentation.utils import seg_to_edges


def _print_help():
    print("Interactive Mutex Watershed Application")
    print("Keybindigns:")
    print("[u] update segmentation")
    print("[s] split mode for selected segment (split by painting seeds)")
    print("[a] attach segment under cursor to selected segment")
    # print("[s] save current segmentation to h5")
    # print("[v] save current seeds to h5")
    print("[y] test consistency if seeds and segmentation")
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
                 randomize_strides=True,
                 show_edges=True):

        ndim = len(offsets[0])
        assert raw.ndim == ndim
        assert affs.ndim == ndim + 1
        assert ndim in (2, 3)

        self.raw = raw
        self.imws = InteractiveMWS(affs, offsets, n_attractive_channels=ndim,
                                   strides=strides, randomize_strides=randomize_strides)
        self.show_edges = show_edges

        self.run()

    def get_cursor_position(self, viewer, layer_name):
        position = None
        scale = None
        layer_scale = None

        for layer in viewer.layers:
            if layer.selected:
                position = layer.coordinates
                scale = layer.scale
            if layer.name == layer_name:
                layer_scale = layer.scale

        assert position is not None
        scale = (1, 1, 1) if scale is None else scale
        layer_scale = (1, 1, 1) if layer_scale is None else layer_scale

        rel_scale = [sc / lsc for lsc, sc in zip(layer_scale, scale)]
        position = tuple(int(pos * sc) for pos, sc in zip(position, rel_scale))
        return position

    def get_id_under_cursor(self, viewer):
        pos = self.get_cursor_position(viewer, layer_name='segmentation')
        val = viewer.layers['segmentation'].data[pos]
        return val

    def run(self):
        # get the initial mws segmentation
        seg = self.imws()

        # initialize save paths for segmentation and seeds
        # seg_path = None
        # seed_path = None
        _print_help()

        # add initial layers to the viewer
        with napari.gui_qt():
            viewer = napari.Viewer()

            # add image layers and point layer for seeds
            viewer.add_image(self.raw, name='raw')
            viewer.add_image(self.imws.affinities, name='affinities', visible=False)
            viewer.add_labels(np.zeros_like(seg), name='seeds')

            if self.show_edges:
                # TODO don't use elf functionality
                edges = seg_to_edges(seg)
                cmap = Colormap([
                    [0., 0., 0., 0.],  # label 0 is transparent
                    [1., 1., 1., 1.]  # label 1 is white (better color?)
                ])
                viewer.add_image(edges, name='edges', colormap=cmap, contrast_limits=[0, 1])

            viewer.add_labels(seg, name='segmentation')

            # add key-bindings

            # update affinities and segmentation
            @viewer.bind_key('u')
            def update(viewer):
                self.update_impl(viewer)

            # TODO
            @viewer.bind_key('s')
            def toggle_split_mode(viewer):
                pass

            @viewer.bind_key('a')
            def attach(viewer):
                self.attach_impl(viewer)

            @viewer.bind_key('y')
            def test_consistency(viewer):
                seeds = viewer.layers['seeds'].data
                print("Test consistency of layers")
                self._test_consistency(viewer.layers['segmentation'].data, seeds)
                print("Test consistency of segmentation")
                self._test_consistency(self.imws(), seeds)

            # display help
            @viewer.bind_key('h')
            def print_help(viewer):
                _print_help()

            # # save the current segmentation
            # @viewer.bind_key('s')
            # def save_segmentation(viewer):
            #     nonlocal seg_path
            #     seg_path = _read_file_path(seg_path)
            #     seg = viewer.layers['segmentation'].data
            #     _save(seg_path, seg)

            # # save the current seeds
            # @viewer.bind_key('v')
            # def save_seeds(viewer):
            #     nonlocal seed_path
            #     seed_path = _read_file_path(seed_path)
            #     seeds = viewer.layers['seeds'].data
            #     _save(seed_path, seeds)

            # @viewer.bind_key('t')
            # def training_step(viewer):
            #     self.training_step_impl(viewer)

    def _test_consistency(self, seg, seeds):
        # check shapes
        if seg.shape != seeds.shape:
            print("Shapes do not agree %s, %s" % (str(seg.shape), str(seeds.shape)))
            return False

        # check seed ids
        seed_ids = np.unique(seeds)[1:]
        print("Found seeds:", seed_ids)
        for seed_id in seed_ids:
            seed_mask = seeds == seed_id
            seg_ids = np.unique(seg[seed_mask])
            if len(seg_ids) != 1:
                print("Expected a single segmentation id for seed %i, got %s" % (seed_id, str(seg_ids)))
                return False

        # check pairs of seed ids
        for seed_a, seed_b in product(seed_ids, seed_ids):
            if seed_a >= seed_b:
                continue
            print("Checking seed pair", seed_a, seed_b)
            mask_a = seeds == seed_a
            mask_b = seeds == seed_b
            ids_a = np.unique(seg[mask_a])
            ids_b = np.unique(seg[mask_b])
            if len(ids_a) != len(ids_b) != 1:
                print("Expected id arrays of len 1, got %s, %s" % (str(ids_a), str(ids_b)))
                return False
            if ids_a[0] == ids_b[0]:
                print("Seeds %i and %i were mapped to the same segment id %i" % (seed_a, seed_b, ids_a[0]))
                return False

        print("Passed")
        return True

    def training_step_impl(self, viewer):
        pass

    def attach_impl(self, viewer):
        seg_layer = viewer.layers['segmentation']
        selected_id = seg_layer.selected_label
        attach_id = self.get_id_under_cursor(viewer)
        print(f"Attaching {attach_id} to {selected_id}...")
        n_merge = self.imws.merge(seg_layer.data, selected_id, attach_id)
        if n_merge > 0:
            print("Press [u] to update the segmentation accordingly.")
        else:
            print("Could not attach, because the two ids are not touching.")

    def update_impl(self, viewer):
        print("Update mws triggered")
        layers = viewer.layers
        seeds = layers['seeds'].data

        seg_layer = layers['segmentation']
        print("Clearing seeds ...")
        self.imws.clear_seeds()

        print("Updating seeds ...")
        self.imws.update_seeds(seeds)

        print("Recomputing segmentation from seeds ...")
        seg = self.imws()
        print("... done")

        seg_layer.data = seg
        seg_layer.refresh()

        if self.show_edges:
            edges = seg_to_edges(seg)
            edge_layer = layers['edges']
            edge_layer.data = edges
            edge_layer.refresh()

        aff_layer = layers['affinities']
        aff_layer.data = self.imws.affinities
        aff_layer.refresh()
