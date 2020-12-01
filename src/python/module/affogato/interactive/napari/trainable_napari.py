import torch
import torch.multiprocessing as mp

from inferno.extensions.criteria import SorensenDiceLoss

# TODO don't use elf functionality
from elf.segmentation.utils import normalize_input

from ..affinities import compute_affinities
from .napari import InteractiveNapariMWS


class DefaultDataset(torch.utils.data.Dataset):
    """ Simple default dataset for generating affinities
    from segmentation and mask.
    """
    def __init__(self, raw, seg, mask, offsets):
        self.raw = raw

        # TODO ignore label etc.
        affs, aff_mask = compute_affinities(seg, offsets)

        self.affs = affs
        self.aff_mask = aff_mask

    def __getitem__(self, index):
        return self.raw, self.affs, self.aff_mask

    def __len__(self):
        return 1


class MaskedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = SorensenDiceLoss()

    # TODO do the masking !
    def foward(self, pred, y, mask):
        loss = self.criterion(pred, y)
        return loss


# TODO start a tensorboard
def default_training(net, raw, seg, mask, offsets, keep_training):
    ds = DefaultDataset(raw, seg, mask, offsets)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = MaskedLoss()
    loss = loss.to(net.device)

    while keep_training:
        for x, y, mask in loader:
            x = x.to(net.device)
            y, mask = y.to(net.device), mask.to(net.device)

            optimizer.zero_grad()

            pred = net(x)
            loss_val = loss(pred, y, mask)
            loss_val.backward()

            optimizer.step()


# TODO support passing initial pool of training data
class TrainableInteractiveNapariMWS(InteractiveNapariMWS):
    @staticmethod
    def run_prediction(self, data, net, normalizer):
        net.eval()
        with torch.no_grad():
            inp = normalizer(data)
            # we assume we need to add channel and batch axis
            inp = torch.from_numpy(data[None, None])
            affs = net(inp.to(net.device))
            affs = affs.cpu().numpy().squeeze()
        return affs

    def __init__(self,
                 raw,
                 net,
                 offsets,
                 strides=None,
                 randomize_strides=True,
                 show_edges=True,
                 normalizer=normalize_input,
                 training_function=default_training):
        self._net = net
        # self._net.shared_memory()

        self._normalzer = normalizer
        affs = self.predict(raw, self._net, self._normalizer)
        super().__init__(raw, affs, offsets,
                         strides=strides, randomize_strides=randomize_strides,
                         show_edges=show_edges)

        # variables for training
        self.training_function = training_function
        self.training_process = None
        self.keep_training = mp.Value('i', 1)

        # TODO allow setting this with initial data
        self._training_pool = []

    def add_keybindings(self, viewer):
        super().add_keybindings(viewer)

        @viewer.bind_key('p')
        def predict(viewer):
            print("Rerun prediction")
            layers = viewer.layers
            raw = layers['raw'].data
            affs = self.run_prediction(raw, self._net, self.normalizer)
            self.imws.affinities = affs

            aff_layer = layers['affinities']
            aff_layer.data = affs
            aff_layer.refresh()
            print("""Affinities were updated from the prediction.
                     Press [u] to see the changes in the segmentation.""")

        @viewer.bind_key('SHIFT + T')
        def update_training(viewer):
            self.update_training_impl(viewer)

    def update_training_impl(self, viewer):
        print("Update training called")
        layers = viewer.layers

        # check if we have any training data
        mask = layers['locked-segment-mask'].data
        if mask.sum() == 0:
            print("No training data available doing nothing")
            return

        seg = layers['segmentation'].data.copy()
        raw = self.normalizer(layers['raw'].data)

        # stop the training process if it is running
        if self.training_process is not None:
            self.keep_training.value = 0
            self.training_process.join()

        self.keep_training.value = 1
        self.training_process = mp.spawn(
            self.training_function,
            args=(
                self._net,
                raw,
                seg,
                mask,
                self.offsets,
                self.keep_training
            ),
            nprocs=1,
            join=False
        )
