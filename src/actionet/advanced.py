"""Advanced ACTIONet functions for lower-level control."""

from typing import Optional, List, Tuple
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core


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
        Reduced kernel matrix (components x cells).
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
    """
    result = _core.decomp_action(S_r, k_min, k_max, max_iter, tolerance, n_threads)
    return result


def collect_archetypes(
    C_trace: List[np.ndarray],
    H_trace: List[np.ndarray],
    specificity_threshold: float = -3.0,
    min_observations: int = 3,
) -> dict:
    """
    Filter and aggregate multi-level archetypes.

    Parameters
    ----------
    C_trace
        List of C matrices from ACTION decomposition.
    H_trace
        List of H matrices from ACTION decomposition.
    specificity_threshold
        Minimum threshold (z-score) to filter archetypes.
    min_observations
        Minimum observations per archetype.

    Returns
    -------
    dict with keys:
        - selected_archs: Indices of selected archetypes
        - C_stacked: Stacked C matrix
        - H_stacked: Stacked H matrix
    """
    result = _core.collect_archetypes(
        C_trace, H_trace, specificity_threshold, min_observations
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
        Reduced kernel matrix.
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
        - selected_cols: Selected column indices (1-indexed)
        - norms: Column norms
    """
    result = _core.run_spa(data, k)
    return result


def run_label_propagation(
    adata: AnnData,
    initial_labels: np.ndarray,
    network_key: str = "actionet",
    lambda_param: float = 1.0,
    iterations: int = 3,
    sig_threshold: float = 3.0,
    fixed_labels: Optional[np.ndarray] = None,
    n_threads: int = 0,
    key_added: str = "propagated_labels",
) -> AnnData:
    """
    Run label propagation algorithm on network.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    initial_labels
        Initial label assignments.
    network_key
        Key in adata.obsp containing network.
    lambda_param
        Propagation strength.
    iterations
        Number of iterations.
    sig_threshold
        Significance threshold.
    fixed_labels
        Indices of labels to keep fixed (1-indexed).
    n_threads
        Number of threads.
    key_added
        Key to store propagated labels in adata.obs.

    Returns
    -------
    Updates adata with:
        - adata.obs[key_added]: Propagated labels
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")

    G = adata.obsp[network_key]

    new_labels = _core.run_lpa(
        G, initial_labels, lambda_param, iterations,
        sig_threshold, fixed_labels, n_threads
    )

    adata.obs[key_added] = new_labels
    return adata


def compute_coreness(
    adata: AnnData,
    network_key: str = "actionet",
    key_added: str = "coreness",
) -> AnnData:
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

    Returns
    -------
    Updates adata with:
        - adata.obs[key_added]: Coreness values
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")

    G = adata.obsp[network_key]
    core_num = _core.compute_coreness(G)

    adata.obs[key_added] = core_num
    return adata


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
