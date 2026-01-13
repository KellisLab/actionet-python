"""ACTIONet: Single-cell multi-resolution data analysis toolkit."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("actionet")
except PackageNotFoundError:
    __version__ = "unknown"

# Import will happen after building the C++ extension
# from . import _core
from .core import (
    reduce_kernel,
    run_action,
    build_network,
    compute_network_diffusion,
    compute_feature_specificity,
    layout_network,
    run_svd,
)
from .anndata_utils import (
    anndata_to_matrix,
    matrix_to_anndata,
    add_action_results,
    add_network_to_anndata,
)

__all__ = [
    "__version__",
    "reduce_kernel",
    "run_action",
    "build_network",
    "compute_network_diffusion",
    "compute_feature_specificity",
    "layout_network",
    "run_svd",
    "anndata_to_matrix",
    "matrix_to_anndata",
    "add_action_results",
    "add_network_to_anndata",
]
