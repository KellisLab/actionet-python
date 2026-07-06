"""AnnData ↔ C++ core plumbing and misc data-shape helpers.

Collects the small utilities that translate between AnnData structures
and the C++ bindings' expected inputs. These used to live in
``anndata_utils.py``.
"""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def anndata_to_matrix(
    adata: AnnData,
    layer: Optional[str] = None,
    transpose: bool = False,
) -> Union[np.ndarray, sp.spmatrix]:
    """Extract the expression matrix from an AnnData for C++ input.

    Returns cells × genes (obs × var) orientation.  ``transpose`` is
    deprecated (the C++ core now expects the native orientation) and
    only left for compatibility.
    """
    X = adata.X if layer is None else adata.layers[layer]

    if transpose:
        warnings.warn(
            "transpose=True is deprecated; the C++ core now accepts native "
            "cells × genes orientation and no transpose is needed.",
            DeprecationWarning,
            stacklevel=2,
        )
        X = X.T.tocsr() if sp.issparse(X) else X.T

    return X


def resolve_network(adata: AnnData, network_key: str):
    """Return ``adata.obsp[network_key]`` after validating presence.

    Raises a ``ValueError`` with a consistent message when the key is
    missing from ``adata.obsp``. Used by all functions that consume a
    precomputed cell-cell network.
    """
    if network_key not in adata.obsp:
        raise ValueError(
            f"Network '{network_key}' not found. Run build_network first."
        )
    return adata.obsp[network_key]


def norm_method_to_int(norm_method) -> int:
    """Translate the ``norm_method`` argument into the C++ integer code.

    Accepts ``"pagerank"`` (=> ``0``), ``"pagerank_sym"`` (=> ``2``), or any
    value that can be cast to :class:`int`. Kept in one place so all diffusion
    call sites agree on the encoding.
    """
    if isinstance(norm_method, str):
        return 2 if norm_method == "pagerank_sym" else 0
    return int(norm_method)


def as_plain_labels(labels):
    """Coerce a pandas Categorical / Series to a plain ndarray of its values.

    A pandas ``Categorical`` (or any array-like exposing ``.categories``)
    carries a category order that can differ from lexicographic. Downstream
    code that re-builds a Categorical from the raw values needs the plain
    values so the rebuilt Categorical uses a deterministic (lexicographic)
    order. This helper is a no-op for objects that don't expose
    ``.categories``.
    """
    if hasattr(labels, "categories"):
        return np.asarray(labels)
    return labels
