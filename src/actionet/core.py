"""High-level Python API wrapping C++ bindings with AnnData integration."""

from typing import Optional, Union, Tuple, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from .anndata_utils import anndata_to_matrix, add_action_results, add_network_to_anndata


def reduce_kernel(
    adata: AnnData,
    n_components: int = 50,
    layer: Optional[str] = None,
    svd_algorithm: int = 0,
    max_iter: int = 0,
    seed: int = 0,
    key_added: str = "action",
    verbose: bool = True,
) -> AnnData:
    """
    Compute reduced kernel matrix for ACTION decomposition.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Number of singular vectors to compute.
    layer
        Layer to use for computation. If None, uses .X.
    svd_algorithm
        SVD algorithm (0=auto).
    max_iter
        Maximum iterations (0=auto).
    seed
        Random seed.
    key_added
        Key to store results in adata.obsm.
    verbose
        Print progress.
        
    Returns
    -------
    Updates adata in-place with:
        - adata.obsm[key_added]: Reduced representation
        - adata.uns[f"{key_added}_params"]: Parameters
    """
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    
    if sp.issparse(S):
        result = _core.reduce_kernel_sparse(S, n_components, svd_algorithm, max_iter, seed, verbose)
    else:
        result = _core.reduce_kernel_dense(S, n_components, svd_algorithm, max_iter, seed, verbose)
    
    # Store results
    adata.obsm[key_added] = result["S_r"].T  # Transpose to cells x components
    adata.uns[f"{key_added}_params"] = {
        "sigma": result["sigma"],
        "V": result["V"],
        "A": result["A"],
        "B": result["B"],
        "n_components": n_components,
    }
    
    return adata


def run_action(
    adata: AnnData,
    k_min: int = 2,
    k_max: int = 30,
    reduction_key: str = "action",
    max_iter: int = 100,
    tolerance: float = 1e-16,
    specificity_threshold: float = -3.0,
    min_observations: int = 3,
    n_threads: int = 0,
    key_added: str = "action_results",
) -> AnnData:
    """
    Run ACTION archetypal analysis decomposition.
    
    Parameters
    ----------
    adata
        Annotated data matrix with reduced representation.
    k_min
        Minimum number of archetypes.
    k_max
        Maximum number of archetypes.
    reduction_key
        Key in adata.obsm containing reduced representation.
    max_iter
        Maximum iterations for AA.
    tolerance
        Convergence tolerance.
    specificity_threshold
        Threshold for filtering archetypes (z-score).
    min_observations
        Minimum observations per archetype.
    n_threads
        Number of threads (0=auto).
    key_added
        Key prefix for storing results.
        
    Returns
    -------
    Updates adata with:
        - adata.obsm["H_stacked"]: Stacked archetype matrix
        - adata.obsm["H_merged"]: Merged archetype matrix
        - adata.obs["assigned_archetype"]: Cell assignments
        - adata.uns[key_added]: Full results
    """
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")
    
    S_r = adata.obsm[reduction_key].T  # Transpose to components x cells
    
    result = _core.run_action(
        S_r, k_min, k_max, max_iter, tolerance,
        specificity_threshold, min_observations, n_threads
    )
    
    add_action_results(adata, result, key_added=key_added)
    return adata


def build_network(
    adata: AnnData,
    archetype_key: str = "H_stacked",
    algorithm: Literal["knn", "k*nn"] = "k*nn",
    distance_metric: Literal["jsd", "l2", "ip"] = "jsd",
    density: float = 1.0,
    k: int = 10,
    mutual_edges_only: bool = True,
    n_threads: int = 0,
    key_added: str = "actionet",
) -> AnnData:
    """
    Build cell-cell interaction network from archetype footprints.
    
    Parameters
    ----------
    adata
        Annotated data matrix with ACTION results.
    archetype_key
        Key in adata.obsm containing archetype matrix.
    algorithm
        Network construction algorithm.
    distance_metric
        Distance metric for similarity.
    density
        Graph density factor.
    k
        Number of nearest neighbors.
    mutual_edges_only
        Only keep mutual nearest neighbors.
    n_threads
        Number of threads (0=auto).
    key_added
        Key to store network in adata.obsp.
        
    Returns
    -------
    Updates adata with:
        - adata.obsp[key_added]: Network adjacency matrix
    """
    if archetype_key not in adata.obsm:
        raise ValueError(f"Archetype matrix '{archetype_key}' not found. Run run_action first.")
    
    H = adata.obsm[archetype_key]
    
    G = _core.build_network(
        H.T, algorithm, distance_metric, density, n_threads,
        16, 200, 50, mutual_edges_only, k
    )
    
    add_network_to_anndata(adata, G, key_added)
    return adata


def compute_network_diffusion(
    adata: AnnData,
    scores: Union[str, np.ndarray],
    network_key: str = "actionet",
    alpha: float = 0.85,
    max_iter: int = 5,
    n_threads: int = 0,
    key_added: str = "diffused",
) -> AnnData:
    """
    Compute network diffusion/smoothing over ACTIONet graph.
    
    Parameters
    ----------
    adata
        Annotated data matrix with network.
    scores
        Either key in adata.obsm or array of scores to diffuse.
    network_key
        Key in adata.obsp containing network.
    alpha
        Diffusion parameter (0-1).
    max_iter
        Maximum iterations.
    n_threads
        Number of threads.
    key_added
        Key to store diffused scores.
        
    Returns
    -------
    Updates adata with:
        - adata.obsm[key_added]: Diffused scores
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found. Run build_network first.")
    
    G = adata.obsp[network_key]
    
    if isinstance(scores, str):
        if scores not in adata.obsm:
            raise ValueError(f"Scores '{scores}' not found in adata.obsm.")
        X0 = adata.obsm[scores]
    else:
        X0 = np.asarray(scores)
    
    if X0.ndim == 1:
        X0 = X0.reshape(-1, 1)
    
    X_diffused = _core.compute_network_diffusion(
        G, X0, alpha, max_iter, n_threads, False, 0, 1e-8
    )
    
    adata.obsm[key_added] = X_diffused
    return adata


def compute_feature_specificity(
    adata: AnnData,
    labels: Union[str, np.ndarray],
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "specificity",
) -> AnnData:
    """
    Compute feature specificity scores for clusters/archetypes.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    labels
        Either key in adata.obs or array of cluster labels.
    layer
        Layer to use (None uses .X).
    n_threads
        Number of threads.
    key_added
        Key prefix for storing results in adata.varm.
        
    Returns
    -------
    Updates adata with:
        - adata.varm[f"{key_added}_upper"]: Upper-tail significance
        - adata.varm[f"{key_added}_lower"]: Lower-tail significance
    """
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    
    if isinstance(labels, str):
        if labels not in adata.obs:
            raise ValueError(f"Labels '{labels}' not found in adata.obs.")
        labels_arr = adata.obs[labels].values
    else:
        labels_arr = np.asarray(labels)
    
    # Convert to integer labels
    from pandas import Categorical
    if not np.issubdtype(labels_arr.dtype, np.integer):
        cat = Categorical(labels_arr)
        labels_int = cat.codes.astype(np.int32)
    else:
        labels_int = labels_arr.astype(np.int32)
    
    result = _core.compute_feature_specificity_sparse(S, labels_int, n_threads)
    
    adata.varm[f"{key_added}_profile"] = result["average_profile"]
    adata.varm[f"{key_added}_upper"] = result["upper_significance"]
    adata.varm[f"{key_added}_lower"] = result["lower_significance"]
    
    return adata


def layout_network(
    adata: AnnData,
    network_key: str = "actionet",
    initial_coords: Optional[np.ndarray] = None,
    method: Literal["umap", "tumap"] = "umap",
    n_components: int = 2,
    spread: float = 1.0,
    min_dist: float = 1.0,
    n_epochs: int = 0,
    seed: int = 0,
    n_threads: int = 0,
    key_added: str = "X_umap",
) -> AnnData:
    """
    Compute 2D/3D layout of ACTIONet graph using UMAP.
    
    Parameters
    ----------
    adata
        Annotated data matrix with network.
    network_key
        Key in adata.obsp containing network.
    initial_coords
        Initial coordinates (if None, random).
    method
        Layout method.
    n_components
        Number of dimensions (2 or 3).
    spread
        UMAP spread parameter.
    min_dist
        UMAP min_dist parameter.
    n_epochs
        Number of optimization epochs (0=auto).
    seed
        Random seed.
    n_threads
        Number of threads.
    key_added
        Key to store layout in adata.obsm.
        
    Returns
    -------
    Updates adata with:
        - adata.obsm[key_added]: Layout coordinates
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")
    
    G = adata.obsp[network_key]
    
    if initial_coords is None:
        rng = np.random.RandomState(seed)
        initial_coords = rng.randn(adata.n_obs, n_components).astype(np.float32)
    
    coords = _core.layout_network(
        G, initial_coords, method, n_components,
        spread, min_dist, n_epochs, seed, n_threads, True
    )
    
    adata.obsm[key_added] = coords
    return adata


def run_svd(
    adata: AnnData,
    n_components: int = 30,
    layer: Optional[str] = None,
    algorithm: int = 0,
    seed: int = 0,
    key_added: str = "X_svd",
) -> AnnData:
    """
    Compute truncated SVD decomposition.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Number of components.
    layer
        Layer to use (None uses .X).
    algorithm
        SVD algorithm (0=auto).
    seed
        Random seed.
    key_added
        Key to store results.
        
    Returns
    -------
    Updates adata with:
        - adata.obsm[key_added]: Right singular vectors
        - adata.uns[f"{key_added}_params"]: SVD parameters
    """
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    
    result = _core.run_svd_sparse(S, n_components, 0, seed, algorithm, True)
    
    adata.obsm[key_added] = result["v"]
    adata.uns[f"{key_added}_params"] = {
        "u": result["u"],
        "d": result["d"],
        "n_components": n_components,
    }
    
    return adata
