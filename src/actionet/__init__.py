"""ACTIONet: Single-cell multi-resolution data analysis toolkit.

Python bindings for the ACTIONet C++ backend (libactionet) via pybind11.
Uses AnnData as the core data container.

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
    layout_network,
)
from .network import (
    build_network,
    cluster_network,
    compute_network_centrality,
    compute_network_diffusion,
    impute_features,
)
from .io.lazy_transform import LazyTransform, create_lazy_transform
from .decomposition import (
    reduce_kernel,
    reduce_kernel_from_svd,
    run_svd,
    smooth_kernel,
)
from .specificity import (
    compute_feature_specificity,
    compute_archetype_feature_specificity,
)
from .tools import (
    aggregate_anndata,
    anndata_to_matrix,
    correct_basal_expression,
    correct_batch_effect,
    derive_guide_thresholds,
    fit_guides_gmm,
    guide_call_gmm,
    sweep_guide_thresholds,
)
from .advanced import (
    run_archetypal_analysis,
    decompose_action,
    collect_archetypes,
    merge_archetypes,
    run_simplex_regression,
    run_spa,
    run_label_propagation,
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
    plot_mito_violin_raster,
    plot_qc_violin,
    plot_qc_violin_raster,
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

from .io.persist import (
    get_auto_persist,
    set_auto_persist,
)
from .io.checkpoint import checkpoint_backed
from .io.subset import materialize_backed, subset_backed_inplace

from .tools import (
    scale,
    aggregate_matrix,
    matrix_sums,
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
    "compute_network_centrality",
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
    "smooth_kernel",
    # Advanced functions
    "run_archetypal_analysis",
    "decompose_action",
    "collect_archetypes",
    "merge_archetypes",
    "run_simplex_regression",
    "run_spa",
    "run_label_propagation",
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
    "plot_mito_violin_raster",
    "plot_qc_violin",
    "plot_qc_violin_raster",
    "plot_umap",
    "plot_umap_interactive",
    "plot_umap_raster",
    # Pipeline
    "run_actionet",
    # Utilities
    "anndata_to_matrix",
    "aggregate_anndata",
    "aggregate_matrix",
    "matrix_sums",
    "fit_guides_gmm",
    "derive_guide_thresholds",
    "sweep_guide_thresholds",
    "guide_call_gmm",
    "import_anndata_generic",
    "filter_anndata",
    "compute_filter_masks",
    "apply_filter",
    "subset_anndata",
    "normalize_anndata",
    "decompress_backed_storage",
    "checkpoint_backed",
    "get_auto_persist",
    "materialize_backed",
    "set_auto_persist",
    "subset_backed_inplace",
]
