"""General-purpose data manipulation and grouped-aggregation tools.

This subpackage collects the "everyday" helpers used across the rest
of ACTIONet:

- :mod:`scale` — :func:`scale`, :func:`l1_norm_scale`.
- :mod:`aggregate` — :func:`aggregate_matrix`, :func:`aggregate_anndata`,
  :func:`matrix_sums`.
- :mod:`anndata` — :func:`anndata_to_matrix`, :func:`resolve_network`,
  :func:`norm_method_to_int`, :func:`as_plain_labels`.
- :mod:`guide_calling` — fit-first per-guide 2-component GMM guide-RNA
  calling for perturb-seq (cells x guides).
- :mod:`batch_correction` — orthogonalization-based batch-effect and
  basal-expression correction on ACTIONet reductions, including
  within-group batch designs.
"""

from .aggregate import aggregate_anndata, aggregate_matrix, matrix_sums
from .anndata import (
    anndata_to_matrix,
    as_plain_labels,
    norm_method_to_int,
    resolve_network,
)
from .batch_correction import (
    build_batch_design,
    correct_basal_expression,
    correct_batch_effect,
    estimate_batch_groups,
)
from .guide_calling import (
    derive_guide_thresholds,
    fit_guides_gmm,
    guide_call_gmm,
    sweep_guide_thresholds,
)
from .scale import l1_norm_scale, scale

__all__ = [
    "aggregate_anndata",
    "aggregate_matrix",
    "anndata_to_matrix",
    "as_plain_labels",
    "build_batch_design",
    "correct_basal_expression",
    "correct_batch_effect",
    "derive_guide_thresholds",
    "estimate_batch_groups",
    "fit_guides_gmm",
    "guide_call_gmm",
    "l1_norm_scale",
    "matrix_sums",
    "norm_method_to_int",
    "resolve_network",
    "scale",
    "sweep_guide_thresholds",
]
