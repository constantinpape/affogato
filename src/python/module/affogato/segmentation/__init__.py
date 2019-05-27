from ._segmentation import connected_components, compute_zws_segmentation, compute_mws_clustering, MWSGridGraph
from .mws import compute_mws_segmentation
from .semantic_mws import compute_semantic_mws_segmentation, compute_semantic_mws_clustering
from .interactive_mws import InteractiveMWS
# NOTE we don't include the causal mws here, because it still relies on vigra
# from .causal_mws import compute_causal_mws
