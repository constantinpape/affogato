import numpy as np
import torch
import torch.multiprocessing as mp
from torchvision.utils import make_grid

from inferno.extensions.criteria import SorensenDiceLoss

# TODO don't use elf functionality
from elf.segmentation.utils import normalize_input

from affogato.affinities import compute_affinities
from .napari import InteractiveNapariMWS


class DefaultDataset(torch.utils.data.Dataset):
    """ Simple default dataset for generating affinities
    from segmentation and mask.
    """
    patch_shape = [512, 512]  # TODO expose this and other parameters

    def to_affinities(self, seg, mask):
        seg[~mask] = 0
        affs, aff_mask = compute_affinities(seg, self.offsets, have_ignore_label=True)
        aff_mask = aff_mask.astype('bool')
        affs = 1. - affs

        mask_transition, aff_mask2 = compute_affinities(mask, self.offsets)
        mask_transition[~aff_mask2.astype('bool')] = 1
        aff_mask[~mask_transition.astype('bool')] = True
        return affs, aff_mask

    @staticmethod
    def estimate_n_samples(shape, patch_shape):
        # we estimate the number of samples by tiling shape with patch_shape
        crops_per_dim = [sh / float(cs) for sh, cs in zip(shape, patch_shape)]
        return int(np.prod(crops_per_dim))

    def __init__(self, raw, seg, mask_ids, offsets, transforms=None):
        self.raw = raw
        self.seg = seg
        self.mask_ids = mask_ids
        self.offsets = offsets
        self.transforms = transforms
        self.n_samples = self.estimate_n_samples(self.raw.shape, self.patch_shape)

    def __getitem__(self, index):

        # TODO sample so that we are biased towards the mask
        def sample_raw_seg_mask():
            offset = [np.random.randint(0, sh - csh) if sh > csh else 0
                      for sh, csh in zip(self.raw.shape, self.patch_shape)]
            bb = tuple(slice(off, off + csh) for off, csh in zip(offset, self.patch_shape))

            raw = self.raw[bb]
            seg = self.seg[bb]

            if self.transforms is not None:
                raw, seg = self.transforms(raw, seg)

            raw, seg = raw.copy(), seg.copy()
            mask = np.isin(seg, self.mask_ids)

            return raw, seg, mask

        raw, seg, mask = sample_raw_seg_mask()

        # TODO ensure that we have some in-mask area
        # # some arbitrary but very small pixel threshold
        # while mask.sum() < 25:
        #     raw, seg, mask = sample_raw_seg_mask()

        # add channel dim
        raw = raw[None]

        # make affs and aff_mask
        affs, aff_mask = self.to_affinities(seg, mask)

        return raw, affs, aff_mask

    def __len__(self):
        return self.n_samples


class MaskedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = SorensenDiceLoss()

    def forward(self, pred, y, mask):
        mask.requires_grad = False
        masked_prediction = pred * mask
        loss = self.criterion(masked_prediction, y)
        return loss


def default_training(proc_id, net,
                     raw, seg, mask_ids,
                     offsets, transforms,
                     pipe, device, step):
    ds = DefaultDataset(raw, seg, mask_ids, offsets, transforms)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2)

    p_out, p_in = pipe

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = MaskedLoss()
    loss = loss.to(device)

    logger = torch.utils.tensorboard.SummaryWriter('./runs/imws')

    add_gradients = True
    log_frequency = 10

    print("Start training")
    while True:
        if p_out.poll():
            if not p_out.recv():
                p_in.send(step)
                break

        net.train()
        for x, y, mask in loader:
            x = x.to(device)
            y, mask = y.to(device), mask.to(device)

            optimizer.zero_grad()

            pred = net(x)
            pred.retain_grad()

            loss_val = loss(pred, y, mask)
            loss_val.backward()
            optimizer.step()

            logger.add_scalar("loss", loss_val.item(), step)

            step += 1
            if step % log_frequency == 0:
                print("Background training process iteration", step)
                x = x[0].detach().cpu()
                logger.add_image('input', x, step)
                y = y[0].detach().cpu()

                if add_gradients:
                    grads = pred.grad[0].detach().cpu()
                    grads -= grads.min()
                    grads /= grads.max()

                pred = torch.clamp(pred[0].detach().cpu(), 0.001, 0.999)
                tandp = [target.unsqueeze(0) for target in y]
                nrow = len(tandp)
                tandp.extend([p.unsqueeze(0) for p in pred])

                if add_gradients:
                    tandp.extend([grad.unsqueeze(0) for grad in grads])
                tandp = make_grid(tandp, nrow=nrow)
                logger.add_image('target_and_prediction', tandp, step)

                # for debugging
                # return x, y, pred, grads


# TODO support passing initial pool of training data
# TODO add a reset button for affinities and the network
class TrainableInteractiveNapariMWS(InteractiveNapariMWS):
    def run_prediction(self, data, net, normalizer):
        net.eval()
        with torch.no_grad():
            inp = normalizer(data)
            # we assume we need to add channel and batch axis
            inp = torch.from_numpy(inp[None, None])
            affs = net(inp.to(self.device))
            affs = affs.cpu().numpy().squeeze()
        return affs

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
                 transforms=None):
        self._net = net
        # self._net.shared_memory()  # this might be necessary for gpu training
        self.device = device

        self._normalizer = normalizer
        affs = self.run_prediction(raw, self._net, self._normalizer)

        # FIXME dirty hack
        bias = 0.6
        affs[2:] += bias

        # variables for training
        self.training_function = training_function
        self.transforms = transforms
        self.training_process = None
        self.keep_training = mp.Value('i', 1)

        # TODO allow setting this with initial data
        self._training_pool = []

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
        raw = self._normalizer(layers['raw'].data)

        # stop the training process if it is running
        if self.training_process is not None:
            self.p_in.send(0)
            self.training_process.join()
            self.training_steps = self.p_out.recv()

        self.p_in.send(1)
        self.training_process = mp.spawn(
            self.training_function,
            args=(
                self._net,
                raw,
                seg,
                list(self.imws.locked_seeds),
                self.imws.offsets,
                self.transforms,
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
