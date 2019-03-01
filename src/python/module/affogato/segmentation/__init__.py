import numpy as np
from ._segmentation import *
from ..affinities import compute_affinities

from ._segmentation import connected_components, compute_mws_clustering
from .mws import compute_mws_segmentation
try:
    from .causal_mws import compute_causal_mws, MWSGridGraph
except ImportError:
    print("Cannot import causal-mws due to missing nifty")

from .semantic_mws import compute_semantic_mws_segmentation
