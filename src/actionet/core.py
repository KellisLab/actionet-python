"""High-level Python API wrapping C++ bindings with AnnData integration."""

from typing import Literal, Optional, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from ._backed_persist import persist_updates
from .reduction import (
    reduce_kernel,
    reduce_kernel_from_svd,
    run_svd,
)
from .specificity import (
    compute_archetype_feature_specificity,
    compute_feature_specificity,
)
from . import tools


def run_action(
    adata: AnnData,
    k_min: int = 2,
    k_max: int = 30,
    reduction_key: str = "action",
    prenormalize: bool = True,
    max_iter: int = 50,
    tolerance: float = 1e-100,
    specificity_threshold: float = -3.0,
    min_observations: int = 2,
    n_threads: int = 0,
    return_c_matrices: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
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
    return_c_matrices
        If True, persist ``C_stacked`` and ``C_merged`` in ``adata.obsm``.
        If False, only ``H`` matrices and assignments are returned/stored.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm["H_stacked"] : np.ndarray
        Stacked archetype matrix (cells × archetypes).
    adata.obsm["H_merged"] : np.ndarray
        Merged archetype matrix (cells × archetypes, after merging similar archetypes).
    adata.obs["assigned_archetype"] : pd.Series or np.ndarray
        Cell-to-archetype assignments.
    """
    if not inplace:
        adata = adata.copy()
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    S_r = adata.obsm[reduction_key]  # cells x k, native orientation

    if prenormalize:
        # Match the shared AnnData contract and R wrapper: each cell is a row in
        # ``obsm[reduction_key]``, so prenormalization must scale rows, not columns.
        S_r = tools.l1_norm_scale(S_r, axis=1)

    # Ensure C-contiguous memory layout for C++ compatibility
    S_r = np.ascontiguousarray(S_r)

    result = _core.run_action(
        S_r, k_min, k_max, max_iter, tolerance,
        specificity_threshold, min_observations, n_threads, return_c_matrices
    )

    obsm_updates = {
        "H_stacked": result["H_stacked"],    # cells x archetypes, direct
        "H_merged": result["H_merged"],      # direct
    }
    if return_c_matrices:
        obsm_updates["C_stacked"] = result["C_stacked"]
        obsm_updates["C_merged"] = result["C_merged"]

    persist_updates(
        adata,
        obsm=obsm_updates,
        obs={"assigned_archetype": result["assigned_archetypes"]},
    )
    if not inplace:
        return adata
    return None


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
    """
    Build cell-cell interaction network from archetype footprints.

    Parameters
    ----------
    adata
        Annotated data matrix with ACTION results.
    obsm_key
        Key in adata.obsm containing archetype matrix.
    algorithm
        Network construction algorithm.
    distance_metric
        Distance metric for similarity.
    density
        Graph density factor.
    M
        HNSW graph connectivity parameter.
    ef_construction
        HNSW construction search breadth.
        For ``algorithm="k*nn"``, the effective value is
        ``max(ef_construction, kNN)``.
    ef
        HNSW query search breadth.
        For ``algorithm="k*nn"``, the effective value is
        ``max(ef, kNN)``.
    k
        Number of nearest neighbors for ``algorithm="knn"``.
    mutual_edges_only
        Only keep mutual nearest neighbors.
    n_threads
        Number of threads (0=auto).
    key_added
        Key to store network in adata.obsp.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsp[key_added] : scipy.sparse matrix or np.ndarray
        Network adjacency matrix (cells × cells).
    """
    if not inplace:
        adata = adata.copy()
    if obsm_key not in adata.obsm:
        raise ValueError(f"Archetype matrix '{obsm_key}' not found. Run run_action first.")

    H = adata.obsm[obsm_key]

    # buildNetworkCore expects row-major (cells x features) float32 input.
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


def compute_network_diffusion(
    X: Union[AnnData, sp.spmatrix],
    scores: Union[str, np.ndarray],
    norm_method: Literal["pagerank", "pagerank_sym"] = "pagerank",
    alpha: float = 0.85,
    n_threads: int = 0,
    approx: bool = True,
    max_iter: int = 5,
    tol = 1e-8,
    network_key: str = "actionet",
    key_added: str = "diffused",
    return_raw: bool = False,
    inplace: bool = True,
) -> Optional[Union[AnnData, np.ndarray]]:
    """
    Compute network diffusion/smoothing over ACTIONet graph.

    Parameters
    ----------
    X
        Either an AnnData object (network looked up from ``X.obsp[network_key]``)
        or a raw sparse graph matrix to diffuse over directly.  When a sparse
        matrix is passed, the result is always returned as a raw array regardless
        of ``return_raw``; ``inplace``, ``key_added``, and ``network_key`` are
        ignored.
    scores
        Score matrix to diffuse.  When ``X`` is an AnnData, this may also be a
        string key into ``X.obsm``; a string is not accepted for a raw matrix
        input.
    network_key
        Key in ``X.obsp`` containing the graph.  Ignored when ``X`` is a sparse
        matrix.
    alpha
        Diffusion parameter (0-1).
    max_iter
        Maximum iterations.
    n_threads
        Number of threads.
    key_added
        Key to store diffused scores in ``X.obsm``.
        Ignored when ``return_raw=True`` or when ``X`` is a sparse matrix.
    return_raw : bool, default ``False``
        If ``True``, return the diffused scores array directly instead of
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
        When ``X`` is AnnData and ``inplace=True`` (default).
    AnnData
        A modified copy when ``X`` is AnnData and ``inplace=False``.
    np.ndarray
        Diffused scores array (cells × features or cells × 1) when
        ``return_raw=True`` or when ``X`` is a sparse matrix.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Diffused scores (cells × features or cells × 1).
    """
    if norm_method not in ("pagerank", "pagerank_sym"):
        raise ValueError(f"Invalid norm_method '{norm_method}'. Must be 'pagerank' or 'pagerank_sym'.")

    is_anndata = isinstance(X, AnnData)

    if is_anndata:
        if network_key not in X.obsp:
            raise ValueError(f"Network '{network_key}' not found. Run build_network first.")
        G = X.obsp[network_key]
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

    X_diffused = _core.compute_network_diffusion(
        G=G,
        X0=X0,
        alpha=alpha,
        max_it=max_iter,
        thread_no=n_threads,
        approx=approx,
        norm_method=2 if norm_method == "pagerank_sym" else 0,
        tol=tol,
    )

    if not is_anndata or return_raw:
        return X_diffused

    # Defer the AnnData copy until after all validation and computation, so we
    # never pay for a full copy on a validation failure or a raw-return path.
    adata = X if inplace else X.copy()
    persist_updates(adata, obsm={key_added: X_diffused})
    if not inplace:
        return adata
    return None


def layout_network(
    adata: AnnData,
    network_key: str = "actionet",
    initial_coords: Optional[Union[str, np.ndarray]] = None,
    layer: Optional[str] = None,
    method: Literal["umap", "tumap", "largevis", "leopold", "leopold2"] = "umap",
    n_components: int = 2,
    spread: float = 1.0,
    min_dist: float = 1.0,
    n_epochs: int = 0,
    learning_rate: float = 1.0,
    repulsion_strength: float = 1.0,
    negative_sample_rate: float = 5.0,
    approx_pow: bool = False,
    pcg_rand: bool = True,
    rng_type: Optional[str] = None,
    batch: bool = True,
    grain_size: int = 1,
    a: float = 0.0,
    b: float = 0.0,
    opt_method: Literal["adam", "sgd"] = "adam",
    alpha: float = -1.0,
    beta1: float = 0.5,
    beta2: float = 0.9,
    eps: float = 1e-7,
    ai: Optional[np.ndarray] = None,
    aj: Optional[np.ndarray] = None,
    seed: int = 0,
    n_threads: int = 0,
    verbose: bool = True,
    key_added: str = "X_umap",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute 2D/3D layout of ACTIONet graph using uwot methods.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    network_key
        Key in adata.obsp containing network.
    initial_coords
        Initial coordinates. Can be a key in adata.obsm, a numpy array, or None.
        If None, computes initial coordinates via SVD on the specified layer.
    layer
        Layer to use for computing initial coordinates via SVD (if initial_coords is None).
        If None, uses adata.X.
    method
        Layout method: "umap", "tumap", "largevis", "leopold", or "leopold2".
    n_components
        Number of dimensions (2 or 3).
    spread
        UMAP spread parameter.
    min_dist
        UMAP min_dist parameter.
    n_epochs
        Number of optimization epochs (0=auto).
    learning_rate
        Base learning rate.
    repulsion_strength
        Repulsion strength (uwot gamma).
    negative_sample_rate
        Negative sample rate.
    approx_pow
        Use approximate power function in UMAP gradient.
    pcg_rand
        Legacy RNG toggle (kept for backward compatibility).
    rng_type
        RNG implementation ("pcg", "tausworthe", "deterministic"). If provided,
        it takes precedence over `pcg_rand`.
    batch
        Use batch updates.
    grain_size
        Parallel grain size.
    a, b
        UMAP shape parameters. Zero values auto-compute from spread/min_dist.
    opt_method
        Optimizer: "adam" or "sgd".
    alpha, beta1, beta2, eps
        Optimizer hyperparameters.
    ai, aj
        Per-vertex coefficient vectors required for "leopold" (`ai`) and
        "leopold2" (`ai` and `aj`).
    seed
        Random seed.
    n_threads
        Number of threads.
    verbose
        Whether to print progress messages.
    key_added
        Key to store layout in adata.obsm.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Layout coordinates (cells × n_components).
    """
    valid_methods = {"umap", "tumap", "largevis", "leopold", "leopold2"}
    method = method.lower()
    if method not in valid_methods:
        raise ValueError(
            f"Invalid `method` '{method}'. Must be one of {sorted(valid_methods)}."
        )

    rng_value = ""
    if rng_type is not None:
        if not isinstance(rng_type, str):
            raise TypeError("`rng_type` must be a string when provided.")
        rng_value = rng_type.strip().lower()
        valid_rng = {"pcg", "tausworthe", "deterministic"}
        if rng_value not in valid_rng:
            raise ValueError(
                f"Invalid `rng_type` '{rng_type}'. Must be one of {sorted(valid_rng)}."
            )

    def _coerce_optional_vector(values: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"`{name}` must have length {adata.n_obs} (number of observations), "
                f"got {arr.shape[0]}."
            )
        return np.ascontiguousarray(arr, dtype=np.float64)

    ai_arr = _coerce_optional_vector(ai, "ai")
    aj_arr = _coerce_optional_vector(aj, "aj")

    if method == "leopold" and ai_arr is None:
        raise ValueError("`ai` must be provided when `method='leopold'`.")
    if method == "leopold2" and (ai_arr is None or aj_arr is None):
        raise ValueError("`ai` and `aj` must be provided when `method='leopold2'`.")

    if not inplace:
        adata = adata.copy()
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")
    
    G = adata.obsp[network_key]
    
    # Handle initial_coords
    if initial_coords is None:
        # Compute initial coordinates from SVD
        if verbose:
            if layer is not None:
                print(f"Computing initial coordinates from layer '{layer}' via SVD")
            else:
                print("Computing initial coordinates from adata.X via SVD")

        k = max(3, n_components)
        svd_result = run_svd(
            adata,
            n_components=k,
            layer=layer,
            algorithm=None,
            max_iter=0,
            seed=seed,
            verbose=verbose,
            return_operator_compatible=True,
        )

        # Get left singular vectors (cells × k) as initial coords
        initial_coords = svd_result["u"]  # cells x k in new orientation
        # Scale columns to have mean 0 and std 1
        initial_coords = (initial_coords - initial_coords.mean(axis=0)) / initial_coords.std(axis=0)
    elif isinstance(initial_coords, str):
        # initial_coords is a key in adata.obsm
        if initial_coords not in adata.obsm:
            raise ValueError(f"Initial coordinates '{initial_coords}' not found in adata.obsm.")
        initial_coords = adata.obsm[initial_coords]
    else:
        # initial_coords is a numpy array
        initial_coords = np.asarray(initial_coords)

    # Validate initial_coords shape
    if initial_coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"Number of rows in initial_coords ({initial_coords.shape[0]}) "
            f"does not match number of cells in adata ({adata.n_obs})"
        )

    if initial_coords.shape[1] < n_components:
        raise ValueError(
            f"Number of columns in initial_coords ({initial_coords.shape[1]}) "
            f"must be >= n_components ({n_components})"
        )

    # Ensure initial_coords is float64 and C-contiguous (binding accepts double)
    initial_coords = np.ascontiguousarray(initial_coords, dtype=np.float64)

    coords = _core.layout_network(
        G=G,
        initial_coords=initial_coords,
        method=method,
        n_components=n_components,
        spread=spread,
        min_dist=min_dist,
        n_epochs=n_epochs,
        seed=seed,
        thread_no=n_threads,
        verbose=verbose,
        learning_rate=learning_rate,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        approx_pow=approx_pow,
        pcg_rand=pcg_rand,
        rng_type=rng_value,
        batch=batch,
        grain_size=grain_size,
        a=a,
        b=b,
        opt_method=opt_method,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        ai=ai_arr,
        aj=aj_arr,
    )

    persist_updates(adata, obsm={key_added: coords})
    if not inplace:
        return adata
    return None
