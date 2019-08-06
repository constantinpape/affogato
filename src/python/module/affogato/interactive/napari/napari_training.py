import numpy as np
from .napari import InteractiveNapariMWS
from ...segmentation import InteractiveMWS
from tiktorch.types import Model
from os import path
import yaml

class TrainableNapariMWS(InteractiveNapariMWS):

    def __init__(self,
                 raw,
                 checkpoint,
                 offsets,
                 strides=None,
                 randomize_strides=True):

        self.seg_ids = None
        self.offsets = offsets

        # initialize network
        self.model = self.initialize_model(checkpoint)
        affs = self.compute_affinities(raw)

        super().__init__(raw,
                         affs,
                         offsets,
                         strides=strides,
                         randomize_strides=randomize_strides)

    def initialize_model(self, checkpoint):
        code = open(path.join(checkpoint, "model.py"), 'rb').read()
        conf_file = path.join(checkpoint, "tiktorch_config.yml")
        with open(conf_file) as stream:
            conf = yaml.safe_load(stream)

        self.training_shape = conf['training']["training_shape"]

        conf["model_init_kwargs"]["out_channels"] = len(self.offsets)

        return Model(code=code, config=conf)

    def compute_affinities(self, raw):
        affs = None
        # TODO: add network computation here
        return affs

    def training_step_impl(self, viewer):
        # TODO
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

        print("Extract regions with seeds")
        # extract segment ids with seed
        mask = seeds > 0
        self.seg_ids = np.unique(seg[mask])

        print("... done")
        seg_layer.data = seg
        seg_layer.refresh()
