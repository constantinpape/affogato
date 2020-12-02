import os
import torch
import torch.multiprocessing as mp

# TODO don't use elf functionality
from elf.segmentation.utils import normalize_input

from .napari import InteractiveNapariMWS
from .train_utils import ConcatDataset, DefaultDataset, default_training


# TODO support passing initial pool of training data
# TODO add a reset button for affinities and the network
class TrainableInteractiveNapariMWS(InteractiveNapariMWS):

    def run_prediction(self, data, net, normalizer):
        # TODO this should be wrapped in a process lock
        net.eval()
        with torch.no_grad():
            inp = normalizer(data)
            # we assume we need to add channel and batch axis
            inp = torch.from_numpy(inp[None, None])
            affs = net(inp.to(self.device))
            affs = affs.cpu().numpy().squeeze()
        net.train()
        return affs

    def initialize_training_data(self,
                                 initial_training_data,
                                 offsets,
                                 normalizer,
                                 transforms):
        if not isinstance(initial_training_data, (list, tuple)):
            raise ValueError

        datasets = []
        for (raw, seg, mask_ids) in initial_training_data:
            datasets.append(DefaultDataset(normalizer(raw), seg, mask_ids,
                                           offsets=offsets, transforms=transforms))
        return datasets

    def __init__(self,
                 raw,
                 net,
                 offsets,
                 device,
                 strides=None,
                 randomize_strides=True,
                 show_edges=True,
                 normalizer=normalize_input,
                 training_function=default_training,
                 transforms=None,
                 initial_training_data=None):
        self._net = net
        # self._net.shared_memory()  # this might be necessary for gpu training
        self.device = device

        self._normalizer = normalizer
        affs = self.run_prediction(raw, self._net, self._normalizer)

        # # FIXME dirty hack
        # bias = 0.6
        # affs[2:] += bias

        # variables for training
        self.training_function = training_function
        self.transforms = transforms
        self.training_process = None
        self.keep_training = mp.Value('i', 1)

        if initial_training_data is not None:
            self._training_pool = self.initialize_training_data(initial_training_data,
                                                                offsets,
                                                                normalizer,
                                                                transforms)
        else:
            self._training_pool = None

        self.p_out, self.p_in = mp.Pipe()
        self.training_steps = 0

        super().__init__(raw, affs, offsets,
                         strides=strides, randomize_strides=randomize_strides,
                         show_edges=show_edges)

    def add_keybindings(self, viewer):
        super().add_keybindings(viewer)

        @viewer.bind_key('Shift-P')
        def predict(viewer):
            print("Rerun prediction")
            layers = viewer.layers
            raw = layers['raw'].data
            affs = self.run_prediction(raw, self._net, self._normalizer)
            self.imws.affinities = affs

            aff_layer = layers['affinities']
            aff_layer.data = affs
            aff_layer.refresh()
            print("""Affinities were updated from the prediction.
                     Press [u] to see the changes in the segmentation.""")

        @viewer.bind_key('Shift-T')
        def toggle_training(viewer):
            self.toggle_training_impl(viewer)

    def get_training_dataset(self, raw, seg):
        ds = DefaultDataset(raw, seg, list(self.imws.locked_seeds),
                            offsets=self.imws.offsets, transforms=self.transforms)
        if self._training_pool is not None:
            ds = ConcatDataset(*([ds] + self._training_pool))
        return ds

    def toggle_training_impl(self, viewer):
        layers = viewer.layers

        # if we have a training process running, just stop it
        # stop the training process if it is running
        if self.training_process is not None:
            print("Stop training")
            self.p_in.send(0)
            self.training_process.join()
            self.training_steps = self.p_out.recv()
            self.training_process = None
            return

        # otherwise, start a new training process with the currently
        # locked segments and optional additional training data
        print("Start training from step", self.training_steps)
        # check if we have any training data
        mask = layers['locked-segment-mask'].data
        if mask.sum() == 0:
            print("No training data available doing nothing")
            return

        seg = layers['segmentation'].data.copy()
        raw = self._normalizer(layers['raw'].data)
        dataset = self.get_training_dataset(raw, seg)

        self.p_in.send(1)
        self.training_process = mp.spawn(
            self.training_function,
            args=(
                self._net,
                dataset,
                (self.p_out, self.p_in),
                self.device,
                self.training_steps
            ),
            nprocs=1,
            join=False
        )

    def run(self):
        super().run()
        if self.training_process is not None:
            self.p_in.send(0)
            self.training_process.join()

    def save_state_impl(self, viewer, save_path):
        super().save_state_impl(viewer, save_path)
        # TODO it would be nicer to save to a buffer and then
        # save (and load) this buffer from hdf5 to have everything in one file
        model_save_path = os.path.splitext(save_path)[0] + '.torch'
        torch.save(self._net.state_dict(), model_save_path)

    def get_initial_viewer_data(self):
        save_path = self._load_from
        if save_path is not None:
            model_save_path = os.path.splitext(save_path)[0] + '.torch'
            if os.path.exists(model_save_path):
                state_dict = torch.load(model_save_path)
                self._net.load_state_dict(state_dict)
        return super().get_initial_viewer_data()

    def print_help_impl(self):
        super().print_help_impl()
        print("[Shift-T] toggle training using currently locked segments")
        print("[Shift-P] repredict affinities with current weights")
