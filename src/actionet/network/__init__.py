"""Cell-cell interaction network — construction, diffusion, centrality,
clustering, and diffusion-based imputation.

Mirrors ``libactionet``'s C++ ``network/`` module and absorbs the two
former top-level Python modules ``clustering.py`` and ``imputation.py``
that were network-first in their behavior.
"""

from .build import build_network
from .centrality import compute_network_centrality
from .cluster import cluster_network
from .diffusion import compute_network_diffusion
from .imputation import impute_features

__all__ = [
    "build_network",
    "cluster_network",
    "compute_network_centrality",
    "compute_network_diffusion",
    "impute_features",
]
