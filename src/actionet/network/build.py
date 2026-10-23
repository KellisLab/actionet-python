"""Cell-cell interaction network construction (HNSW-based).

Provides :func:`build_network`, which constructs a cell-cell graph from
archetype footprints (or any low-rank embedding in ``adata.obsm``).
"""

from typing import Literal, Optional

import numpy as np
from anndata import AnnData

from .. import _core
from ..io.persist import persist_updates


def _check_hnsw_param(name: str, value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
        value = int(value)
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    return int(value)


def build_network(
    adata: AnnData,
    algorithm: Literal["knn", "k*nn"] = "k*nn",
    distance_metric: Literal["jsd", "l2", "ip"] = "jsd",
    density: float = 1.0,
    n_threads: int = 0,
    mutual_edges_only: bool = True,
    M: Optional[int] = None,
    ef_construction: Optional[int] = None,
    ef: Optional[int] = None,
    k: int = 10,
    obsm_key: str = "H_stacked",
    key_added: str = "actionet",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Build cell-cell interaction network from archetype footprints.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with ACTION results.
    obsm_key : str
        Key in adata.obsm containing archetype matrix.
    algorithm : str
        Network construction algorithm.
    distance_metric : str
        Distance metric for similarity.
    density : float
        Graph density factor (``algorithm="k*nn"`` only).
    M : int or None
        HNSW graph connectivity. ``None`` resolves to 16. Explicit values
        must be at least 2.
    ef_construction : int or None
        HNSW index construction search breadth. ``None`` resolves to
        ``max(200, 4 * k)`` for ``algorithm="knn"`` and ``max(200, kNN)`` for
        ``algorithm="k*nn"``, where ``kNN = min(n - 1, 5 * round(sqrt(n)))``
        is the adaptive query size. For ``"k*nn"``, explicit values are
        floored at ``kNN``.
    ef : int or None
        HNSW query search breadth. ``None`` resolves to ``max(200, 5 * k)``
        for ``algorithm="knn"`` and ``max(200, kNN)`` for ``algorithm="k*nn"``.
        For ``"k*nn"``, explicit values are floored at ``kNN``.
    k : int
        Number of nearest neighbors (``algorithm="knn"`` only).
    mutual_edges_only : bool
        Only keep mutual nearest neighbors.
    n_threads : int
        Number of threads (0=auto).
    key_added : str
        Key to store network in ``adata.obsp``.
    inplace : bool
        Modify in place or return a copy.

    Returns
    -------
    None or AnnData
        ``None`` if ``inplace=True``; modified copy otherwise.
    """
    M = _check_hnsw_param("M", M)
    ef_construction = _check_hnsw_param("ef_construction", ef_construction)
    ef = _check_hnsw_param("ef", ef)

    if not inplace:
        adata = adata.copy()
    if obsm_key not in adata.obsm:
        raise ValueError(f"Archetype matrix '{obsm_key}' not found. Run run_action first.")

    H = adata.obsm[obsm_key]

    H = np.ascontiguousarray(H, dtype=np.float32)

    if not np.isfinite(H).all():
        raise ValueError(
            f"obsm['{obsm_key}'] contains NaN or Inf values. "
            "Clean the input matrix before building a network."
        )

    G = _core.build_network(
        H, algorithm, distance_metric, density, n_threads,
        M, ef_construction, ef, mutual_edges_only, k
    )

    persist_updates(adata, obsp={key_added: G})
    if not inplace:
        return adata
    return None
