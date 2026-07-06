"""Cell-cell interaction network construction (HNSW-based).

Provides :func:`build_network`, which constructs a cell-cell graph from
archetype footprints (or any low-rank embedding in ``adata.obsm``).
"""

from typing import Literal, Optional

import numpy as np
from anndata import AnnData

from .. import _core
from ..io.persist import persist_updates


def build_network(
    adata: AnnData,
    algorithm: Literal["knn", "k*nn"] = "k*nn",
    distance_metric: Literal["jsd", "l2", "ip"] = "jsd",
    density: float = 1.0,
    n_threads: int = 0,
    mutual_edges_only: bool = True,
    M: float = 16,
    ef_construction: float = 200,
    ef: float = 200,
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
        Graph density factor.
    M : float
        HNSW graph connectivity parameter.
    ef_construction : float
        HNSW construction search breadth. For ``algorithm="k*nn"`` the
        effective value is ``max(ef_construction, kNN)``.
    ef : float
        HNSW query search breadth. For ``algorithm="k*nn"`` the effective
        value is ``max(ef, kNN)``.
    k : int
        Number of nearest neighbors for ``algorithm="knn"``.
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
