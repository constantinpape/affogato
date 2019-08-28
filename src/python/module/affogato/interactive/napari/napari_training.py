import numpy as np
from .napari import InteractiveNapariMWS
from ...segmentation import InteractiveMWS
from tiktorch.types import Model, ModelState
from os import path
import yaml
from tiktorch.launcher import LocalServerLauncher, RemoteSSHServerLauncher, SSHCred, wait
from tiktorch.rpc import Client, TCPConnConf
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.types import NDArray
from tiktorch.tiktypes import TikTensorBatch
from tiktorch import tiktypes as types
from tiktorch.server.handler import datasets


def read_bytes(filename):
    with open(filename, "rb") as file:
        return file.read()


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
        self.neuralnetwork_client = self.initialize_model(checkpoint)
        print("Computing Affinities")
        affs = self.compute_affinities(raw)

        print("Starting InteractiveNapariMWS")

        super().__init__(raw,
                         affs,
                         offsets,
                         strides=strides,
                         randomize_strides=randomize_strides)

    def initialize_model(self, checkpoint):
        code = open(path.join(checkpoint, "model.py"), 'rb').read()
        conf_file = path.join(checkpoint, "tiktorch_config.yml")
        state_file = path.join(checkpoint, "state_dict.nn")
        srv_file = path.join(checkpoint, "server.yml")
        conf = yaml.safe_load(read_bytes(conf_file))

        self.training_shape = conf['training']["training_shape"]

        conf["model_init_kwargs"]["out_channels"] = len(self.offsets)

        server_conf = yaml.safe_load(read_bytes(srv_file))

        conn_conf = TCPConnConf(server_conf['ip'],
                                server_conf['port0'],
                                server_conf['port1'],
                                timeout=200)

        cred = SSHCred(server_conf['user'],
                       key_path=server_conf['key_file'])
        launcher = RemoteSSHServerLauncher(conn_conf, cred=cred)
        launcher.start()

        neuralnetwork_client = Client(INeuralNetworkAPI(), conn_conf)
        state = read_bytes(state_file)

        neuralnetwork_client.load_model(Model(code=code, config=conf),
                                        state=ModelState(model_state=state),
                                        devices=[server_conf['device']])

        return neuralnetwork_client

    def compute_affinities(self, raw):
        affs = self.neuralnetwork_client.forward(NDArray(raw[None]))
        return affs.result().as_numpy()

    def training_step_impl(self, viewer):
        self.neuralnetwork_client.resume_training()

    def update_affinities(self, raw):
        return self.compute_affinities(raw)

    def update_dataset(self, raw, seeds, segmentation):

        print("updating dataset")
        data = types.TikTensorBatch([types.TikTensor(raw[None], id_=(0, 0)), ])
        labels = types.TikTensorBatch([types.TikTensor(segmentation[None], id_=(0, 0)), ])

        self.neuralnetwork_client.update_training_data(data, labels)

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
