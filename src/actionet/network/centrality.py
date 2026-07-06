"""Network centrality measures (coreness, PageRank, and label-aware variants)."""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .. import _core
from ..io.persist import persist_updates


def compute_network_centrality(
    X: Union[AnnData, sp.spmatrix],
    algorithm: Literal["coreness", "pagerank", "local_coreness", "local_pagerank"] = "coreness",
    labels: Union[str, np.ndarray, None] = None,
    alpha: float = 0.9,
    max_iter: int = 5,
    tol: float = 1e-8,
    n_threads: int = 0,
    network_key: str = "actionet",
    key_added: Optional[str] = None,
    return_raw: bool = False,
    inplace: bool = True,
) -> Optional[Union[AnnData, np.ndarray]]:
    """Compute network centrality scores.

    Supports global measures (coreness, PageRank) and label-aware local
    variants (local coreness, local PageRank).
    """
    valid_algorithms = {"coreness", "pagerank", "local_coreness", "local_pagerank"}
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"Invalid algorithm '{algorithm}'. Must be one of {sorted(valid_algorithms)}."
        )

    is_anndata = isinstance(X, AnnData)

    if is_anndata:
        from ..tools.anndata import resolve_network
        G = resolve_network(X, network_key)
    else:
        G = X

    n_cells = G.shape[1]

    if algorithm in ("local_coreness", "local_pagerank"):
        if labels is None:
            raise ValueError(
                f"'labels' is required when algorithm='{algorithm}'."
            )

    assignments: Optional[np.ndarray] = None
    if labels is not None:
        if isinstance(labels, str):
            if not is_anndata:
                raise ValueError(
                    "`labels` must be an array when `X` is a sparse matrix, not a string key."
                )
            if labels not in X.obs:
                raise ValueError(f"Labels column '{labels}' not found in adata.obs.")
            raw_labels = X.obs[labels].values
        else:
            raw_labels = np.asarray(labels)
        assignments = pd.Categorical(raw_labels).codes.astype(np.int32)

    if algorithm in ("pagerank", "local_pagerank"):
        alpha = float(np.clip(alpha, 0.0, 0.99))

    if algorithm == "coreness":
        centrality = np.asarray(_core.compute_coreness(G), dtype=np.int32)

    elif algorithm == "pagerank":
        uniform = np.full((n_cells, 1), 1.0 / n_cells)
        centrality = np.asarray(
            _core.compute_network_diffusion(
                G=G, X0=uniform, alpha=alpha, max_it=max_iter,
                thread_no=n_threads, approx=True, norm_method=0, tol=tol,
            )
        ).ravel()

    elif algorithm == "local_coreness":
        centrality = np.asarray(
            _core.compute_archetype_centrality(G, assignments)
        )

    elif algorithm == "local_pagerank":
        unique_codes = np.unique(assignments)
        n_groups = len(unique_codes)
        design = np.zeros((n_cells, n_groups), dtype=np.float64)
        for col_idx, code in enumerate(unique_codes):
            mask = assignments == code
            design[mask, col_idx] = 1.0
        col_sums = design.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        design /= col_sums

        design = np.ascontiguousarray(design)
        scores = np.asarray(
            _core.compute_network_diffusion(
                G=G, X0=design, alpha=alpha, max_it=max_iter,
                thread_no=n_threads, approx=True, norm_method=0, tol=tol,
            )
        )
        col_max = scores.max(axis=0)
        col_max[col_max == 0] = 1.0
        scores /= col_max
        centrality = scores.max(axis=1)

    centrality = centrality.ravel()

    if not is_anndata or return_raw:
        return centrality

    adata = X if inplace else X.copy()
    if key_added is None:
        key_added = f"{algorithm}_{network_key}"
    persist_updates(adata, obs={key_added: centrality})
    if not inplace:
        return adata
    return None
