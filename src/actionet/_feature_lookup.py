"""First-match feature lookup utilities.

Centralises feature-name resolution for all public ACTIONet functions that
accept user-supplied gene/feature names.  Mirrors R's ``unique()`` +
``match()`` first-occurrence semantics so that Python and R front-ends
produce identical index mappings.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from anndata import AnnData


@dataclass(frozen=True)
class FeatureSpace:
    """Resolved feature-label vector with first-occurrence lookup."""

    labels: np.ndarray
    """Length-``n_vars`` array of raw feature labels."""

    lookup: dict
    """``{label: first_index}`` mapping (first occurrence wins)."""

    has_duplicates: bool
    """Whether *labels* contains any duplicated entries."""


def resolve_feature_space(
    adata: AnnData,
    features_use: Optional[str] = None,
    *,
    context: str = "",
) -> FeatureSpace:
    """Build a first-occurrence lookup map over the feature-label vector.

    Parameters
    ----------
    adata
        Annotated data object.
    features_use
        Column in ``adata.var`` to use.  ``None`` uses ``adata.var_names``.
    context
        Human-readable caller name included in duplicate warnings.

    Returns
    -------
    FeatureSpace
    """
    if features_use is None:
        labels = adata.var_names.to_numpy()
    else:
        if features_use not in adata.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        labels = adata.var[features_use].to_numpy()

    lookup: dict = {}
    has_dup = False
    for idx, lab in enumerate(labels):
        if lab not in lookup:
            lookup[lab] = idx
        else:
            has_dup = True

    if has_dup:
        ctx = f" ({context})" if context else ""
        warnings.warn(
            f"Feature labels contain duplicates{ctx}; "
            "only the first occurrence of each label will be used.",
            UserWarning,
            stacklevel=3,
        )

    return FeatureSpace(labels=labels, lookup=lookup, has_duplicates=has_dup)


@dataclass(frozen=True)
class ResolvedFeatures:
    """Result of matching requested feature names against a FeatureSpace."""

    matched_names: List[str]
    """Feature names that were found, in first-request order (deduplicated)."""

    matched_indices: np.ndarray
    """Corresponding column indices (int64), same length as *matched_names*."""

    missing_names: List[str]
    """Requested names that were not found."""


def resolve_requested_features(
    requested: Sequence[str],
    space: FeatureSpace,
    *,
    context: str = "",
) -> ResolvedFeatures:
    """Match requested feature names against a :class:`FeatureSpace`.

    Duplicate requested names are collapsed to first occurrence (matching R).

    Parameters
    ----------
    requested
        User-supplied feature names.
    space
        Feature space from :func:`resolve_feature_space`.
    context
        Human-readable caller name for warning/error messages.

    Returns
    -------
    ResolvedFeatures
    """
    seen: set = set()
    matched_names: list = []
    matched_indices: list = []
    missing_names: list = []

    for name in requested:
        name_str = str(name)
        if name_str in seen:
            continue
        seen.add(name_str)
        idx = space.lookup.get(name_str)
        if idx is not None:
            matched_names.append(name_str)
            matched_indices.append(idx)
        else:
            missing_names.append(name_str)

    if missing_names:
        ctx = f" ({context})" if context else ""
        print(f"Features missing{ctx}: {', '.join(missing_names)}")

    return ResolvedFeatures(
        matched_names=matched_names,
        matched_indices=np.array(matched_indices, dtype=np.int64) if matched_indices else np.array([], dtype=np.int64),
        missing_names=missing_names,
    )
