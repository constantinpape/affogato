from ._segmentation import connected_components, compure_zws_segmentation, compute_mws_clustering
from .mws import compute_mws_segmentation
try:
    from .causal_mws import compute_causal_mws
except ImportError:
    print("Cannot import causal-mws due to missing nifty")
