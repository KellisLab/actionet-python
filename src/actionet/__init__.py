"""ACTIONet: Single-cell multi-resolution data analysis toolkit.

Python front-end for the ACTIONet C++ backend (libactionet) via pybind11.
Uses AnnData as the primary data container.

System build requirements: CMake >= 3.19, C++17 compiler, BLAS/LAPACK,
HDF5 (C library), and OpenMP.

The public API is defined by ``__all__`` below.  Everything else is
implementation detail and may change without notice.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("actionet")
except PackageNotFoundError:
    __version__ = "unknown"

from .action import (
    collect_archetypes,
    compute_archetype_centrality,
    decompose_action,
    merge_archetypes,
    run_action,
    run_archetypal_analysis,
    run_label_propagation,
    run_simplex_regression,
    run_spa,
)
from .annotation import (
    annotate_cells,
    annotate_clusters,
    compute_archetype_feature_specificity,
    compute_feature_specificity,
    find_markers,
)
from .decomposition import (
    reduce_kernel,
    reduce_kernel_from_svd,
    run_svd,
    smooth_kernel,
)
from .io.checkpoint import checkpoint_backed
from .io.lazy_transform import LazyTransform, create_lazy_transform
from .io.persist import get_auto_persist, set_auto_persist
from .io.subset import materialize_backed, subset_backed_inplace
from .network import (
    build_network,
    cluster_network,
    compute_network_centrality,
    compute_network_diffusion,
    impute_features,
)
from .pipeline import run_actionet
from .preprocessing import (
    apply_filter,
    compute_filter_masks,
    decompress_backed_storage,
    filter_anndata,
    import_anndata_generic,
    normalize_anndata,
    subset_anndata,
)
from .tools import (
    aggregate_anndata,
    aggregate_matrix,
    anndata_to_matrix,
    correct_basal_expression,
    correct_batch_effect,
    derive_guide_thresholds,
    fit_guides_gmm,
    guide_call_gmm,
    matrix_sums,
    scale,
    sweep_guide_thresholds,
)
from .visualization import (
    compute_node_colors,
    get_feature_abundance,
    get_mito_feats,
    layout_network,
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

__all__ = [
    "__version__",
    # Pipeline
    "run_actionet",
    # Action (decomposition + archetypes)
    "collect_archetypes",
    "compute_archetype_centrality",
    "decompose_action",
    "merge_archetypes",
    "run_action",
    "run_archetypal_analysis",
    "run_label_propagation",
    "run_simplex_regression",
    "run_spa",
    # Annotation (markers + specificity)
    "annotate_cells",
    "annotate_clusters",
    "compute_archetype_feature_specificity",
    "compute_feature_specificity",
    "find_markers",
    # Decomposition (kernel + SVD)
    "reduce_kernel",
    "reduce_kernel_from_svd",
    "run_svd",
    "smooth_kernel",
    # Network
    "build_network",
    "cluster_network",
    "compute_network_centrality",
    "compute_network_diffusion",
    "impute_features",
    # Preprocessing
    "apply_filter",
    "compute_filter_masks",
    "decompress_backed_storage",
    "filter_anndata",
    "import_anndata_generic",
    "normalize_anndata",
    "subset_anndata",
    # Tools (matrix utilities, batch correction, guide calling)
    "aggregate_anndata",
    "aggregate_matrix",
    "anndata_to_matrix",
    "correct_basal_expression",
    "correct_batch_effect",
    "derive_guide_thresholds",
    "fit_guides_gmm",
    "guide_call_gmm",
    "matrix_sums",
    "scale",
    "sweep_guide_thresholds",
    # Visualization (layout + plotting)
    "compute_node_colors",
    "get_feature_abundance",
    "get_mito_feats",
    "layout_network",
    "plot_feature_expression",
    "plot_feature_expression_raster",
    "plot_mito_violin",
    "plot_mito_violin_raster",
    "plot_qc_violin",
    "plot_qc_violin_raster",
    "plot_umap",
    "plot_umap_interactive",
    "plot_umap_raster",
    # I/O (lazy transform + backed persistence)
    "LazyTransform",
    "checkpoint_backed",
    "create_lazy_transform",
    "get_auto_persist",
    "materialize_backed",
    "set_auto_persist",
    "subset_backed_inplace",
]
