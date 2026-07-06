"""Network diffusion / smoothing over ACTIONet graphs."""

from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .. import _core
from ..io.persist import persist_updates


def compute_network_diffusion(
    X: Union[AnnData, sp.spmatrix],
    scores: Union[str, np.ndarray],
    norm_method: Literal["pagerank", "pagerank_sym"] = "pagerank",
    alpha: float = 0.85,
    n_threads: int = 0,
    approx: bool = True,
    max_iter: int = 5,
    tol=1e-8,
    network_key: str = "actionet",
    key_added: str = "diffused",
    return_raw: bool = False,
    inplace: bool = True,
) -> Optional[Union[AnnData, np.ndarray]]:
    """Compute network diffusion/smoothing over ACTIONet graph.

    Parameters
    ----------
    X : AnnData or scipy.sparse matrix
        Either an AnnData object (network looked up from ``X.obsp[network_key]``)
        or a raw sparse graph matrix.  When a sparse matrix is passed, the
        result is always returned as a raw array regardless of ``return_raw``;
        ``inplace``, ``key_added``, and ``network_key`` are ignored.
    scores : str or np.ndarray
        Score matrix to diffuse. When ``X`` is an AnnData, this may also be a
        string key into ``X.obsm``; a string is not accepted for a raw matrix
        input.
    network_key : str
        Key in ``X.obsp`` containing the graph.
    alpha : float
        Diffusion parameter (0-1).
    max_iter : int
        Maximum iterations.
    n_threads : int
        Number of threads.
    key_added : str
        Key to store diffused scores in ``X.obsm``. Ignored when
        ``return_raw=True`` or when ``X`` is a sparse matrix.
    return_raw : bool
        If ``True``, return the diffused scores array directly instead of
        writing to the AnnData.
    inplace : bool
        Modify in place or return a copy.
    """
    if norm_method not in ("pagerank", "pagerank_sym"):
        raise ValueError(f"Invalid norm_method '{norm_method}'. Must be 'pagerank' or 'pagerank_sym'.")

    is_anndata = isinstance(X, AnnData)

    if is_anndata:
        from ..tools.anndata import resolve_network
        G = resolve_network(X, network_key)
    else:
        G = X

    if isinstance(scores, str):
        if not is_anndata:
            raise ValueError(
                "`scores` must be an array when `X` is a sparse matrix, not a string key."
            )
        if scores not in X.obsm:
            raise ValueError(f"Scores '{scores}' not found in adata.obsm.")
        X0 = X.obsm[scores]
    else:
        X0 = scores

    if X0.ndim == 1:
        X0 = np.ascontiguousarray(X0.reshape(-1, 1))
    elif not X0.flags["C_CONTIGUOUS"]:
        X0 = np.ascontiguousarray(X0)

    from ..tools.anndata import norm_method_to_int

    X_diffused = _core.compute_network_diffusion(
        G=G,
        X0=X0,
        alpha=alpha,
        max_it=max_iter,
        thread_no=n_threads,
        approx=approx,
        norm_method=norm_method_to_int(norm_method),
        tol=tol,
    )

    if not is_anndata or return_raw:
        return X_diffused

    adata = X if inplace else X.copy()
    persist_updates(adata, obsm={key_added: X_diffused})
    if not inplace:
        return adata
    return None
