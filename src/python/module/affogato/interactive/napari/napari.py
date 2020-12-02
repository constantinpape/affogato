import os
from itertools import product
import numpy as np
import h5py

import napari
from vispy.color import Colormap

from ...segmentation import InteractiveMWS
from .mws_with_seeds import mws_with_seeds

# TODO don't use elf functionality
from elf.segmentation.utils import seg_to_edges


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
        self._split_mode_id = None
        self._split_mask = None
        self._load_from = None

    def load_from_state(self, state_path):
        if not os.path.exists(state_path):
            raise ValueError(f"Cannot find state path {state_path}")
        self._load_from = state_path

    def get_initial_viewer_data(self):
        if self._load_from is None:
            seg = self.imws()
            seeds, mask = np.zeros_like(seg), np.zeros_like(seg)
        else:
            print("Initialize imws with state from", self._load_from)
            with h5py.File(self._load_from, 'r') as f:
                seg = f['segmentation'][:]
                seeds = f['seeds'][:]
                mask_ids = f['mask_ids'][:]
            mask = np.isin(seg, mask_ids)
            self.imws.lock_seeds(set(mask_ids))
        return seg, seeds, mask

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

    @property
    def split_mode_active(self):
        return self._split_mode_id is not None

    def add_keybindings(self, viewer):
        # update affinities and segmentation
        @viewer.bind_key('u')
        def update(viewer):
            self.update_impl(viewer)

        @viewer.bind_key('s')
        def toggle_split_mode(viewer):
            self.toggle_split_mode_impl(viewer)

        @viewer.bind_key('a')
        def attach(viewer):
            self.attach_impl(viewer)

        @viewer.bind_key('t')
        def toggle_lock(viewer):
            self.toggle_lock_impl(viewer)

        # display help
        @viewer.bind_key('h')
        def print_help(viewer):
            self.print_help_impl()

        # TODO add a key binding that gets the seed for the currently selected segment
        # (either next if there is no seed for it or the segment id if there is a seed for it)
        # next seed id
        @viewer.bind_key('n')
        def next_seed(viewer):
            self.select_next_seed(viewer)

        @viewer.bind_key('Shift-S')
        def save_state(viewer):
            save_path = './imws_saved_state.h5'
            print("Saving current viewer state to", save_path)
            self.save_state_impl(viewer, save_path)

        # @viewer.bind_key('y')
        # def test_consistency(viewer):
        #     seeds = viewer.layers['seeds'].data
        #     print("Test consistency of layers")
        #     self._test_consistency(viewer.layers['segmentation'].data, seeds)
        #     print("Test consistency of segmentation")
        #     self._test_consistency(self.imws(), seeds)

    def save_state_impl(self, viewer, save_path):
        seg = viewer.layers['segmentation'].data
        seeds = viewer.layers['seeds'].data
        mask_ids = list(self.imws.locked_seeds)

        # TODO don't open in w and allow for multiple checkpoints with time stamp
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('segmentation', data=seg, compression='gzip')
            f.create_dataset('seeds', data=seeds, compression='gzip')
            f.create_dataset('mask_ids', data=mask_ids)

    def print_help_impl(self):
        print("Interactive Mutex Watershed Keybindings")
        print("[U] update segmentation")
        print("[S] split mode for selected segment (split by painting seeds)")
        print("[A] attach segment under cursor to selected segment")
        print("[H] print help")
        print("[T] toggle lock mode for segment under cursor")
        print("[Shift-S] save state")
        # print("[y] test consistency if seeds and segmentation")

    def run(self):
        seg, seeds, mask = self.get_initial_viewer_data()

        self.print_help_impl()

        # add initial layers to the viewer
        with napari.gui_qt():
            viewer = napari.Viewer()

            # add image layers and point layer for seeds
            viewer.add_image(self.raw, name='raw')
            viewer.add_image(self.imws.affinities, name='affinities', visible=False)

            if self.show_edges:
                # TODO don't use elf functionality
                edges = seg_to_edges(seg)
                cmap = Colormap([
                    [0., 0., 0., 0.],  # label 0 is transparent
                    [1., 1., 1., 1.]  # label 1 is white (better color?)
                ])
                viewer.add_image(edges, name='edges', colormap=cmap, contrast_limits=[0, 1])

            viewer.add_labels(mask, name='locked-segment-mask', visible=False)
            viewer.add_labels(seeds, name='seeds')
            viewer.add_labels(seg, name='segmentation')

            self.add_keybindings(viewer)

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

    def select_next_seed(self, viewer):
        layer = viewer.layers['seeds']
        next_label = layer.data.max() + 1

        viewer.layers.unselect_all()
        layer.selected = True
        layer.mode = 'paint'
        layer.selected_label = next_label

    def toggle_lock_impl(self, viewer):
        if self.split_mode_active:
            print("Cannot toggle lock while the split mode is active")
            return

        lock_id = self.get_id_under_cursor(viewer)
        if lock_id in self.imws.locked_seeds:
            print("Removing", lock_id, "from locked segments")
            self.imws.unlock_seeds({lock_id})
        else:
            # make sure we have a seed for this segment, otherwise locking doesn't work
            lock_id = self.ensure_seed(viewer, lock_id)
            print("Adding", lock_id, "to locked segments")
            self.imws.lock_seeds({lock_id})

        self.update_mask(viewer)

    def ensure_seed(self, viewer, seg_id):
        seed_layer = viewer.layers['seeds']
        seeds = seed_layer.data
        last_seed = seeds.max()
        # if seg-id is smaller equal than our last seed id
        # it already has a seed and we don't need to do anything
        if seg_id <= last_seed:
            return seg_id

        # otherwise, we need to place a seed in the object
        # and update the segmentation
        else:
            next_seed = last_seed + 1
            seg = viewer.layers['segmentation'].data

            # TODO make a brush (skimage.draw.circle) instead of a single pixel
            coords = np.where(seg == seg_id)
            seed_coord_id = np.random.choice(len(coords[0]))
            seed_coord = tuple(coord[seed_coord_id] for coord in coords)

            seeds[seed_coord] = next_seed
            seed_layer.refresh()

            self._update_normal(viewer)
            return next_seed

    def update_mask(self, viewer):
        layers = viewer.layers
        seg = layers['segmentation'].data
        mask_layer = layers['locked-segment-mask']
        mask_layer.data = np.isin(seg, list(self.imws.locked_seeds))
        mask_layer.refresh()

    def attach_impl(self, viewer):
        layers = viewer.layers
        seg_layer = layers['segmentation']
        selected_id = seg_layer.selected_label
        attach_id = self.get_id_under_cursor(viewer)
        print(f"Attaching {attach_id} to {selected_id}...")
        n_merge = self.imws.merge(seg_layer.data, selected_id, attach_id)

        if n_merge > 0:
            aff_layer = layers['affinities']
            aff_layer.data = self.imws.affinities
            aff_layer.refresh()
            print("Press [u] to see the changes in the segmentation.")
        else:
            print("Could not attach, because the two ids are not touching.")

    def toggle_split_mode_impl(self, viewer):
        layers = viewer.layers
        seg_layer = layers['segmentation']

        # the split mode is active -> turn it of and update the segmentation
        if self.split_mode_active:
            self._split_mode_id = None
            self.update_impl(viewer)
            self._split_mask = None

            viewer.layers.unselect_all()
            seg_layer.selected = True
            seg_layer.mode = 'pick'

        # the split mode is inactive -> turn it on
        else:
            selected_id = seg_layer.selected_label
            self._split_mode_id = selected_id
            print("Activate split mode for segment", selected_id)

            seg = seg_layer.data
            self._split_mask = (seg == selected_id)
            seg[~self._split_mask] = 0
            seg_layer.data = seg
            seg_layer.refresh()

            self.select_next_seed(viewer)

    def _update_normal(self, viewer):
        layers = viewer.layers
        seg_layer = layers['segmentation']

        seeds = layers['seeds'].data
        # if we are just coming from split mode we set
        # the seeds outside the split mask to zero
        if self._split_mask is not None:
            seeds = seeds.copy()
            seeds[~self._split_mask] = 0
        self.imws.update_seeds(seeds)

        print("Recomputing segmentation")
        # TODO keep same id as seeds!
        seg = self.imws()

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

    def _update_split_mode(self, viewer):
        layers = viewer.layers
        seeds = layers['seeds'].data.copy()
        mask = self._split_mask
        seeds[~mask] = 0

        # TODO keep same id as seeds!
        seg = mws_with_seeds(self.imws.affinities, self.imws.offsets, seeds,
                             strides=self.imws.strides,
                             randomize_strides=self.imws.randomize_strides,
                             mask=mask)

        seg_layer = layers['segmentation']
        seg_layer.data = seg
        seg_layer.refresh()

    def update_impl(self, viewer):
        if self.split_mode_active:
            print("Update triggered in split mode ...")
            self._update_split_mode(viewer)
        else:
            print("Update triggered in normal mode ...")
            self._update_normal(viewer)
