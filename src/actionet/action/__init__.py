"""ACTION archetypal analysis subpackage."""

from .archetypes import (
    collect_archetypes,
    compute_archetype_centrality,
    decompose_action,
    merge_archetypes,
    run_archetypal_analysis,
    run_label_propagation,
    run_simplex_regression,
    run_spa,
)
from .run_action import run_action

__all__ = [
    "collect_archetypes",
    "compute_archetype_centrality",
    "decompose_action",
    "merge_archetypes",
    "run_action",
    "run_archetypal_analysis",
    "run_label_propagation",
    "run_simplex_regression",
    "run_spa",
]
