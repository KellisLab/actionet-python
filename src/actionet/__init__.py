"""ACTIONet: Single-cell multi-resolution data analysis toolkit.

Python bindings for the ACTIONet C++ backend (libactionet) via pybind11.
Uses AnnData as the core data container, integrates with the scanpy ecosystem.

System build requirements: CMake >= 3.19, C++17 compiler, BLAS/LAPACK,
HDF5 (C library), and OpenMP.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("actionet")
except PackageNotFoundError:
    __version__ = "unknown"

# Import will happen after building the C++ extension
from .core import (
    run_action,
    build_network,
    compute_network_diffusion,
    layout_network,
)
from .lazy_transform import LazyTransform, create_lazy_transform
from .reduction import (
    reduce_kernel,
    reduce_kernel_from_svd,
    run_svd,
    smooth_kernel,
)
from .specificity import (
    compute_feature_specificity,
    compute_archetype_feature_specificity,
)
from .clustering import (
    cluster_network,
)
from .anndata_utils import (
    anndata_to_matrix,
    matrix_to_anndata,
    add_action_results,
    add_network_to_anndata,
    aggregate_anndata,
)
from .batch_correction import (
    correct_batch_effect,
    correct_basal_expression,
)
from .imputation import (
    impute_features,
    impute_features_from_archetypes,
)
from .advanced import (
    run_archetypal_analysis,
    decompose_action,
    collect_archetypes,
    merge_archetypes,
    run_simplex_regression,
    run_spa,
    run_label_propagation,
    compute_coreness,
    compute_archetype_centrality,
)
from .annotation import (
    find_markers,
    annotate_cells,
    annotate_clusters,
)
from .visualization import (
    compute_node_colors,
)
from .plotting import (
    get_feature_abundance,
    get_mito_feats,
    plot_feature_expression,
    plot_feature_expression_raster,
    plot_mito_violin,
    plot_qc_violin,
    plot_umap,
    plot_umap_interactive,
    plot_umap_raster,
)
from .pipeline import (
    run_actionet,
)

from .preprocessing import (
    import_anndata_generic,
    filter_anndata,
    compute_filter_masks,
    apply_filter,
    subset_anndata,
    normalize_anndata,
    decompress_backed_storage,
)

from ._backed_persist import checkpoint_backed, subset_backed_inplace

from .tools import (
    scale,
    aggregate_matrix,
)

__all__ = [
    "__version__",
    # Core functions
    "LazyTransform",
    "create_lazy_transform",
    "reduce_kernel",
    "reduce_kernel_from_svd",
    "run_action",
    "build_network",
    "cluster_network",
    "compute_network_diffusion",
    "compute_feature_specificity",
    "compute_archetype_feature_specificity",
    "layout_network",
    "run_svd",
    # Batch correction
    "correct_batch_effect",
    "correct_basal_expression",
    # Imputation
    "impute_features",
    "impute_features_from_archetypes",
    "smooth_kernel",
    # Advanced functions
    "run_archetypal_analysis",
    "decompose_action",
    "collect_archetypes",
    "merge_archetypes",
    "run_simplex_regression",
    "run_spa",
    "run_label_propagation",
    "compute_coreness",
    "compute_archetype_centrality",
    # Annotation
    "find_markers",
    "annotate_cells",
    "annotate_clusters",
    # Visualization
    "compute_node_colors",
    "get_feature_abundance",
    "get_mito_feats",
    "plot_feature_expression",
    "plot_feature_expression_raster",
    "plot_mito_violin",
    "plot_qc_violin",
    "plot_umap",
    "plot_umap_interactive",
    "plot_umap_raster",
    # Pipeline
    "run_actionet",
    # Utilities
    "anndata_to_matrix",
    "matrix_to_anndata",
    "add_action_results",
    "add_network_to_anndata",
    "aggregate_anndata",
    "aggregate_matrix",
    "import_anndata_generic",
    "filter_anndata",
    "compute_filter_masks",
    "apply_filter",
    "subset_anndata",
    "normalize_anndata",
    "decompress_backed_storage",
    "checkpoint_backed",
    "subset_backed_inplace",
]
