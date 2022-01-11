import argparse
import numpy as np

from affogato.interactive.napari import InteractiveNapariMWS

import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from xarray import DataArray


def interactive_napari(model):
    """Run interactive napari mws with affinities predicted with model in the
    bioimage.io model format. Requires bioimageio.core library available via
        pip install bioimageio.core
        conda install -c conda-forge bioimageio.core
    """
    # load the model representation
    model = bioimageio.core.load_resource_description(model)
    # load the example data for this model and use it as data for the IMWS
    raw = np.load(model.test_inputs[0])

    # get the offsets from the model metadata and determine the data ndim from it
    offsets = model.config["mws"]["offsets"]
    ndim = len(offsets[0])

    # predict the affinities
    # NOTE: this will also apply the correct normalization to the data
    with create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = DataArray(raw, dims=tuple(pp.input_specs[0].axes))
        affs = pp(input_)[0]

    # strip singelton dimensions and validate
    raw = raw.squeeze()
    affs = affs.squeeze()
    assert raw.ndim == ndim
    assert affs.ndim == ndim + 1
    assert affs.shape[0] == len(offsets)

    # start the interactive mws
    strides = [4] * ndim
    imws = InteractiveNapariMWS(raw, affs, offsets,
                                strides=strides, randomize_strides=True)
    imws.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to the bioimageio model (zip file).")
    args = parser.parse_args()
    interactive_napari(args.model)
