"""Advanced ACTIONet functions for lower-level control."""

from typing import Optional, List, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from ._backed_persist import persist_updates


def run_archetypal_analysis(
    data: np.ndarray,
    W0: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> dict:
    """
    Run archetypal analysis on data matrix.

    Parameters
    ----------
    data
        Input data matrix (features x observations).
    W0
        Initial archetype matrix (features x k).
    max_iter
        Maximum iterations.
    tolerance
        Convergence tolerance.

    Returns
    -------
    dict with keys:
        - C: Archetype compositions (observations x k)
        - H: Archetype weights (k x observations)
        - W: Archetypes in original space
    """
    result = _core.run_aa(data, W0, max_iter, tolerance)
    return result


def decompose_action(
    S_r: np.ndarray,
    k_min: int = 2,
    k_max: int = 30,
    max_iter: int = 100,
    tolerance: float = 1e-16,
    n_threads: int = 0,
) -> dict:
    """
    Run ACTION decomposition and return full trace of C and H matrices.

    This is a lower-level function that returns results for all k values
    without filtering or merging.

    Parameters
    ----------
    S_r
        Reduced kernel matrix (cells x components).
    k_min
        Minimum number of archetypes.
    k_max
        Maximum number of archetypes.
    max_iter
        Maximum iterations.
    tolerance
        Convergence tolerance.
    n_threads
        Number of threads (0=auto).

    Returns
    -------
    dict with keys:
        - C: List of C matrices for k=k_min to k_max
        - H: List of H matrices for k=k_min to k_max
        - C_stacked: All C matrices column-stacked (n_cells x T)
        - H_stacked: All H matrices row-stacked (T x n_cells)
    """
    result = _core.decomp_action(S_r, k_min, k_max, max_iter, tolerance, n_threads)
    return result


def collect_archetypes(
    C_stacked: np.ndarray,
    H_stacked: np.ndarray,
    specificity_threshold: float = -3.0,
    min_observations: int = 3,
) -> dict:
    """
    Filter and aggregate multi-level archetypes.

    Parameters
    ----------
    C_stacked
        Pre-stacked C matrix from decomp_action (n_cells x T).
    H_stacked
        Pre-stacked H matrix from decomp_action (T x n_cells).
    specificity_threshold
        Minimum threshold (z-score) to filter archetypes.
    min_observations
        Minimum observations per archetype.

    Returns
    -------
    dict with keys:
        - selected_archs: Indices of selected archetypes
        - C_stacked: Filtered stacked C matrix
        - H_stacked: Filtered stacked H matrix
    """
    result = _core.collect_archetypes(
        C_stacked, H_stacked, specificity_threshold, min_observations
    )
    return result


def merge_archetypes(
    S_r: np.ndarray,
    C_stacked: np.ndarray,
    H_stacked: np.ndarray,
    n_threads: int = 0,
) -> dict:
    """
    Identify and merge redundant archetypes.

    Parameters
    ----------
    S_r
        Reduced kernel matrix (cells x components).
    C_stacked
        Stacked C matrix from collect_archetypes.
    H_stacked
        Stacked H matrix from collect_archetypes.
    n_threads
        Number of threads (0=auto).

    Returns
    -------
    dict with keys:
        - selected_archetypes: Indices of representative archetypes
        - C_merged: Merged C matrix
        - H_merged: Merged H matrix
        - assigned_archetypes: Cell assignments to merged archetypes
    """
    result = _core.merge_archetypes(S_r, C_stacked, H_stacked, n_threads)
    return result


def run_simplex_regression(
    A: np.ndarray,
    B: np.ndarray,
    compute_XtX: bool = False,
) -> np.ndarray:
    """
    Solve simplex-constrained regression: min ||AX - B|| s.t. simplex constraint.

    Parameters
    ----------
    A
        Input matrix A in AX - B.
    B
        Input matrix B in AX - B.
    compute_XtX
        Whether to return X^T X.

    Returns
    -------
    X
        Solution matrix.
    """
    X = _core.run_simplex_regression(A, B, compute_XtX)
    return X


def run_spa(
    data: np.ndarray,
    k: int,
) -> dict:
    """
    Run successive projections algorithm (SPA) for separable NMF.

    Parameters
    ----------
    data
        Input data matrix.
    k
        Number of candidate vertices.

    Returns
    -------
    dict with keys:
        - selected_cols: Selected column indices (0-indexed)
        - norms: Column norms
    """
    result = _core.run_spa(data, k)
    return result


def run_label_propagation(
    X: Union[AnnData, sp.spmatrix],
    initial_labels: Union[str, np.ndarray, pd.Series, List],
    network_key: str = "actionet",
    lambda_param: float = 1.0,
    iterations: int = 3,
    sig_threshold: float = 3.0,
    fixed_labels: Optional[np.ndarray] = None,
    n_threads: int = 0,
    key_added: str = "propagated_labels",
    return_raw: bool = False,
    inplace: bool = True,
) -> Optional[Union[AnnData, np.ndarray]]:
    """
    Run label propagation algorithm on network.

    Parameters
    ----------
    X
        Either an AnnData object (network looked up from ``X.obsp[network_key]``)
        or a raw sparse graph matrix to propagate over directly.  When a sparse
        matrix is passed, the result is always returned as a raw array regardless
        of ``return_raw``; ``inplace``, ``key_added``, and ``network_key`` are
        ignored.
    initial_labels
        Initial label assignments. Can be:

        - A string key into ``X.obs`` (only when ``X`` is AnnData).
        - A list, pd.Series, or np.ndarray of labels (strings or integers).

        Labels are internally factorized to numeric codes for the C++ backend
        and decoded back to the original label space on return.
    network_key
        Key in ``X.obsp`` containing network.  Ignored when ``X`` is a sparse
        matrix.
    lambda_param
        Propagation strength.
    iterations
        Number of iterations.
    sig_threshold
        Significance threshold.
    fixed_labels
        Indices of labels to keep fixed (1-indexed).
    n_threads
        Number of threads (0 = auto).
    key_added
        Key to store propagated labels in ``X.obs``.
        Ignored when ``return_raw=True`` or when ``X`` is a sparse matrix.
    return_raw : bool, default ``False``
        If ``True``, return the propagated labels array directly instead of
        writing to the AnnData.  Always treated as ``True`` when ``X`` is a
        sparse matrix.  When ``True``, ``adata`` is never modified and
        ``inplace`` is ignored.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new
        AnnData object with the results.  Ignored when ``return_raw=True`` or
        when ``X`` is a sparse matrix.

    Returns
    -------
    None
        When ``X`` is AnnData, ``inplace=True``, and ``return_raw=False``.
    AnnData
        A modified copy when ``inplace=False`` and ``return_raw=False``.
    np.ndarray
        Propagated labels array (in the original label space) when
        ``return_raw=True`` or when ``X`` is a sparse matrix.
    """
    is_anndata = isinstance(X, AnnData)

    if is_anndata:
        if network_key not in X.obsp:
            raise ValueError(f"Network '{network_key}' not found in adata.obsp.")
        G = X.obsp[network_key]
    else:
        G = X

    if isinstance(initial_labels, str):
        if not is_anndata:
            raise ValueError(
                "`initial_labels` must be an array when `X` is a sparse matrix, "
                "not a string key."
            )
        if initial_labels not in X.obs:
            raise ValueError(f"Labels column '{initial_labels}' not found in X.obs.")
        raw_labels = X.obs[initial_labels].values
    else:
        raw_labels = np.asarray(initial_labels)

    categories, numeric_codes = np.unique(raw_labels, return_inverse=True)
    numeric_codes = numeric_codes.astype(np.float64)

    new_codes = _core.run_lpa(
        G, numeric_codes, lambda_param, iterations,
        sig_threshold, fixed_labels, n_threads
    )

    new_labels = categories[new_codes.astype(np.intp)]

    if not is_anndata or return_raw:
        return new_labels

    adata = X if inplace else X.copy()
    persist_updates(adata, obs={key_added: new_labels})
    if not inplace:
        return adata
    return None


def compute_coreness(
    adata: AnnData,
    network_key: str = "actionet",
    key_added: str = "coreness",
    inplace: bool = True,
) -> Union[np.ndarray, None]:
    """
    Compute coreness (k-shell decomposition) of graph vertices.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    network_key
        Key in adata.obsp containing network.
    key_added
        Key to store coreness in adata.obs.
    inplace
        If True, adds coreness to adata.obs.
        If False, returns coreness array without modifying adata.

    Returns
    -------
    AnnData or np.ndarray
        If inplace=True,  coreness added to obs.
        If inplace=False, returns coreness array.

    Updates adata (if inplace=True)
    ------
    adata.obs[key_added] : np.ndarray
        Coreness values
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")

    G = adata.obsp[network_key]
    core_num = _core.compute_coreness(G)

    if not inplace:
        return core_num

    adata.obs[key_added] = core_num
    return None


def compute_archetype_centrality(
    adata: AnnData,
    assignments: np.ndarray,
    network_key: str = "actionet",
    key_added: str = "centrality",
) -> AnnData:
    """
    Compute centrality of vertices within archetype-induced subgraphs.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    assignments
        Archetype assignments for each cell.
    network_key
        Key in adata.obsp containing network.
    key_added
        Key to store centrality in adata.obs.

    Returns
    -------
    Updates adata with:
        - adata.obs[key_added]: Centrality values
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")

    G = adata.obsp[network_key]
    conn = _core.compute_archetype_centrality(G, assignments.astype(np.int32))

    adata.obs[key_added] = conn
    return adata
