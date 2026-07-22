"""Annotation subpackage: marker detection, cluster/cell annotation, and feature specificity."""

from .annotation import (
    annotate_archetypes,
    annotate_cells,
    annotate_clusters,
    find_markers,
)
from .specificity import (
    compute_archetype_feature_specificity,
    compute_feature_specificity,
)

__all__ = [
    "annotate_archetypes",
    "annotate_cells",
    "annotate_clusters",
    "compute_archetype_feature_specificity",
    "compute_feature_specificity",
    "find_markers",
]
