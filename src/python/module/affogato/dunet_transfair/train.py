from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin
from argparse import Namespace
from torch.utils.data import Dataset
from inferno.io.core.zip import Zip

from shutil import copyfile
from torch import load
import h5py
from inferno.io.volumetric import VolumeLoader
import torch


class AffinityDataset(Zip):

    def __init__(self, data_file, raw_key='raw', aff_key='prediction',
        window_size=(1, 128, 128), stride=(1, 16, 16)):

        with h5py.File(data_file, 'r') as f:
            raw = f[raw_key][:]
            affs = f[aff_key][:]

        rawds = VolumeLoader(raw, window_size, stride)
        segds = VolumeLoader(affs, window_size, stride, is_multichannel=True)

        super().__init__(rawds, segds)


    def __getitem__(self, idx):

        img, aff = super().__getitem__(idx)

        img = torch.from_numpy(img)
        aff = torch.from_numpy(aff)    
        aff = aff.squeeze(1)

        return img, aff



class Maiden(BaseExperiment, InfernoMixin, TensorboardMixin):

    def __init__(self):
        super(Maiden, self).__init__()
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = [
            'data_loader', '_device']
        self.auto_setup(update_git_revision=False,
                        dump_configuration=True,
                        construct_objects=True)

    # def save(self):
    #     torch.save(self.model,
    #                os.path.join(self.experiment_directory, 'model.pytorch'),
    #                pickle_module=self.trainer.pickle_module)


if __name__ == '__main__':
    exp = Maiden()
    exp.train()
