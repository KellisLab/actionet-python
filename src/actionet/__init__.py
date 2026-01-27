"""ACTIONet: Single-cell multi-resolution data analysis toolkit."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("actionet")
except PackageNotFoundError:
    __version__ = "unknown"

# Import will happen after building the C++ extension
from .core import (
    reduce_kernel,
    run_action,
    build_network,
    compute_network_diffusion,
    compute_feature_specificity,
    compute_archetype_feature_specificity,
    layout_network,
    run_svd,
)
from .anndata_utils import (
    anndata_to_matrix,
    matrix_to_anndata,
    add_action_results,
    add_network_to_anndata,
)
from .batch_correction import (
    correct_batch_effect,
    correct_basal_expression,
)
from .imputation import (
    impute_features,
    impute_from_archetypes,
    smooth_kernel,
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
)
from .visualization import (
    compute_node_colors,
)
from .pipeline import (
    run_actionet,
)

from .preprocessing import (
    import_anndata_generic,
    filter_anndata,
)

from .tools import (
    scale,
)

__all__ = [
    "__version__",
    # Core functions
    "reduce_kernel",
    "run_action",
    "build_network",
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
    "impute_from_archetypes",
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
    # Visualization
    "compute_node_colors",
    # Pipeline
    "run_actionet",
    # Utilities
    "anndata_to_matrix",
    "matrix_to_anndata",
    "add_action_results",
    "add_network_to_anndata",
]
