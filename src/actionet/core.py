"""High-level Python API wrapping C++ bindings with AnnData integration."""

import os
from typing import Any, Literal, Optional, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from .anndata_utils import anndata_to_matrix
from ._backed_persist import persist_updates
from ._matrix_source import MatrixSource
from .backed_io import (
    _backed_group_path,
    _chunk_target_bytes,
    _flush_backed_handle,
    _is_backed_matrix,
    _maybe_decompress_backed_path,
    _resolve_backed_handle,
    _warn_if_compressed_backed_svd,
)
from .lazy_transform import (
    LazyTransform,
    create_lazy_transform,
    _lazy_params_for_metadata,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)
from .specificity import (
    compute_archetype_feature_specificity,
    compute_feature_specificity,
)
from . import tools

_SVD_ALGORITHM_TO_ID = {
    "irlb": 0,
    "halko": 1,
    "feng": 2,
    "primme": 3,
}
_SVD_ID_TO_ALGORITHM = {v: k for k, v in _SVD_ALGORITHM_TO_ID.items()}


def _normalize_algorithm(algorithm: Optional[str], *, context: str) -> str:
    if algorithm is None:
        return "auto"
    if not isinstance(algorithm, str):
        raise TypeError(f"`{context}` must be a string algorithm name")
    name = algorithm.strip().lower()
    allowed = {"auto", *list(_SVD_ALGORITHM_TO_ID)}
    if name not in allowed:
        raise ValueError(f"Invalid algorithm `{algorithm}`. Allowed: {sorted(allowed)}")
    return name


def _select_svd_algorithm_inmemory(S: Any, algorithm: str, verbose: bool = True) -> int:
    if algorithm != "auto":
        return _SVD_ALGORITHM_TO_ID[algorithm]

    if sp.issparse(S):
        total_elements = S.nnz
    else:
        total_elements = np.prod(S.shape)

    # For in-memory Armadillo sparse matrices (sp_mat), indices are signed 32-bit
    # internally, so NNZ or shape dimensions exceeding 2^31 - 1 would overflow.
    # PRIMME uses external matvec callbacks that bypass Armadillo indexing, so it
    # is safe. (The backed/operator path never hits this function — it always uses
    # Halko via _select_svd_algorithm_backed.)
    max_int32 = 2_147_483_647
    if total_elements > max_int32:
        if verbose:
            print(f"⚠ Matrix exceeds 32-bit indexing limit ({total_elements:,} > {max_int32:,} elements)")
            print("→ Auto-selected PRIMME for safe handling of large matrices")
        return _SVD_ALGORITHM_TO_ID["primme"]

    if sp.issparse(S):
        sparsity = 1.0 - (total_elements / np.prod(S.shape))
        if verbose:
            print(f"Auto-selected IRLB for sparse matrix ({sparsity:.1%} sparse)")
        return _SVD_ALGORITHM_TO_ID["irlb"]

    if verbose:
        print(f"Auto-selected Halko for dense matrix ({total_elements:,} elements)")
    return _SVD_ALGORITHM_TO_ID["halko"]


def _select_svd_algorithm_backed(algorithm: str, verbose: bool = True) -> int:
    if algorithm == "auto":
        # For backed (operator) mode, Halko is unconditionally selected as the
        # default.  The backed operator path bypasses Armadillo's in-memory
        # indexing, so the 32-bit overflow concern that triggers PRIMME in the
        # in-memory path does not apply here.  Halko's peak memory usage depends
        # only on (k+2) * max(n_obs, n_var), not on NNZ, and its matvec count is
        # fixed at 2*(iters+1) passes regardless of matrix conditioning, giving a
        # clean NNZ-proportional I/O cost model at scale.
        #
        # IRLB is also supported for backed operator mode and can be requested
        # explicitly (e.g. for benchmarking), but is not the auto-selected
        # default.  See docs/svd_algorithm_benchmark.md for the backing rationale.
        if verbose:
            print("Detected backed matrix: selecting Halko operator path")
        return _SVD_ALGORITHM_TO_ID["halko"]

    if algorithm not in {"halko", "irlb", "feng", "primme"}:
        raise ValueError(
            "Backed matrices support only 'auto', 'halko', 'irlb', 'feng', or 'primme'"
        )
    return _SVD_ALGORITHM_TO_ID[algorithm]


def reduce_kernel(
    adata: AnnData,
    n_components: int = 30,
    layer: Optional[str] = None,
    key_added: str = "action",
    svd_algorithm: Optional[str] = "auto",
    max_iter: int = 0,
    seed: int = 0,
    verbose: bool = True,
    precomputed_svd: Optional[dict] = None,
    backed_chunk_size: int = 4096,
    allow_compressed: bool = False,
    inplace: bool = True,
    backed_target_chunk_mb: Optional[float] = None,
    backed_n_threads: int = 0,
    lazy_transform: Optional[LazyTransform] = None,
) -> Optional[AnnData]:
    """Compute low-rank kernel reduction and persist outputs to AnnData.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x features). Supports both in-memory and
        backed (HDF5-streamed) AnnData objects.
    n_components : int
        Number of SVD components to retain.
    layer : str or None
        Layer to use (None uses ``.X``).
    key_added : str
        Base key under which to store results in ``adata.obsm``, ``adata.varm``,
        and ``adata.uns``.
    svd_algorithm : str or None
        SVD algorithm: ``"auto"``, ``"irlb"``, ``"halko"``, ``"feng"``, or
        ``"primme"``.  ``"auto"`` selects based on matrix properties.
    max_iter : int
        Maximum iterations for iterative SVD solvers (0 = solver default).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress messages.
    precomputed_svd : dict or None
        Pre-computed SVD with keys ``"u"``, ``"d"``, ``"v"`` (as returned by
        :func:`run_svd`).  If provided, SVD is skipped.
    backed_chunk_size : int
        Row chunk size for backed sparse streaming.
    allow_compressed : bool
        If True, allow compressed backed storage (may be slower). If False
        (default), auto-decompresses to a temporary file.
    inplace : bool
        Modify adata in place or return a copy.
    backed_target_chunk_mb : float or None
        Target chunk size in MiB for backed I/O (None = auto).
    backed_n_threads : int
        Thread count for backed operator compute loops (0 = auto).
        Only used for backed (operator) execution paths; in-memory paths
        use BLAS/library threading.
    lazy_transform : LazyTransform or None
        Pre-computed lazy transform for backed inputs on ``.X`` only.
        Applies row-level normalization and log1p on-the-fly during
        operator matvec, avoiding materialization.

    Returns
    -------
    None or AnnData
        None if ``inplace=True``; modified copy if ``inplace=False``.
    """
    if backed_n_threads < 0:
        raise ValueError("`backed_n_threads` must be >= 0")

    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    # Validate the lazy transform before any expensive operation.
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    use_operator = source.is_backed
    algorithm_name = _normalize_algorithm(svd_algorithm, context="svd_algorithm")
    row_scale_factors: Optional[np.ndarray] = None
    apply_log1p = False
    log_scale = 1.0

    if use_operator:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        _flush_backed_handle(adata, context="reduce_kernel")
        selected_algorithm = _select_svd_algorithm_backed(algorithm_name, verbose)
        io_target_chunk_bytes = _chunk_target_bytes(backed_target_chunk_mb)
        temp_path: Optional[str] = None
        op = None
        file_path = str(adata.filename)
        try:
            temp_path = _maybe_decompress_backed_path(
                adata,
                layer=layer,
                allow_compressed=allow_compressed,
                chunk_size=backed_chunk_size,
                verbose=verbose,
                context="reduce_kernel",
            )
            if temp_path is not None:
                file_path = temp_path

            op = _core.create_backed_operator(
                file_path=file_path,
                group_path=_backed_group_path(layer),
                chunk_size=backed_chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
                io_target_chunk_bytes=io_target_chunk_bytes,
                n_threads=backed_n_threads,
            )

            if precomputed_svd is None:
                result = _core.reduce_kernel_backed_operator(
                    op, n_components, selected_algorithm, max_iter, seed, verbose
                )
            else:
                result = _core.reduce_kernel_from_svd_backed_operator(
                    op,
                    precomputed_svd["u"],
                    precomputed_svd["d"],
                    precomputed_svd["v"],
                    verbose,
                )
        finally:
            op = None
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

        svd_algorithm_id = selected_algorithm
    else:
        S = anndata_to_matrix(adata, layer=layer)
        svd_algorithm_id = _select_svd_algorithm_inmemory(S, algorithm_name, verbose)

        if precomputed_svd is None:
            if sp.issparse(S):
                result = _core.reduce_kernel_sparse(S, n_components, svd_algorithm_id, max_iter, seed, verbose)
            else:
                result = _core.reduce_kernel_dense(S, n_components, svd_algorithm_id, max_iter, seed, verbose)
        else:
            if sp.issparse(S):
                result = _core.reduce_kernel_from_svd_sparse(
                    S, precomputed_svd["u"], precomputed_svd["d"],
                    precomputed_svd["v"], verbose,
                )
            else:
                result = _core.reduce_kernel_from_svd_dense(
                    S, precomputed_svd["u"], precomputed_svd["d"],
                    precomputed_svd["v"], verbose,
                )

    params = {
        "sigma": np.asarray(result["sigma"]).ravel(),
        "n_components": n_components,
        "svd_algorithm": svd_algorithm_id,
        "svd_algorithm_name": _SVD_ID_TO_ALGORITHM.get(svd_algorithm_id, f"unknown({svd_algorithm_id})"),
        "used_precomputed_svd": precomputed_svd is not None,
        "operator_mode": use_operator,
    }
    params.update(_lazy_params_for_metadata(lazy_transform if apply_log1p else None))
    persist_updates(
        adata,
        obsm={
            key_added: result["S_r"],              # cells x k, direct
            f"{key_added}_B": result["B"],
        },
        varm={
            f"{key_added}_U": result["U"],
            f"{key_added}_A": result["A"],
        },
        uns={f"{key_added}_params": params},
    )

    if not inplace:
        return adata
    return None


def reduce_kernel_from_svd(
    adata: AnnData,
    svd_result: dict,
    layer: Optional[str] = None,
    key_added: str = "action",
    verbose: bool = True,
    backed_chunk_size: int = 4096,
    inplace: bool = True,
    lazy_transform: Optional[LazyTransform] = None,
    backed_target_chunk_mb: Optional[float] = None,
    backed_n_threads: int = 0,
) -> Optional[AnnData]:
    """Compute reduced kernel using a precomputed SVD result.

    Convenience wrapper around :func:`reduce_kernel` that infers ``n_components``
    from the SVD result and passes it as ``precomputed_svd``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells × features).
    svd_result : dict
        Precomputed SVD with keys ``"u"``, ``"d"``, ``"v"`` (as returned by
        :func:`run_svd`).
    layer : str or None
        Layer to use (None uses .X).
    key_added : str
        Key under which to store results.
    verbose : bool
        Print progress messages.
    backed_chunk_size : int
        Chunk size for backed sparse streaming.
    inplace : bool
        Modify adata in place or return a copy.
    lazy_transform : LazyTransform or None
        Pre-computed lazy transform for backed inputs on ``.X`` only.
        ``None`` means no transformation is applied.
    backed_target_chunk_mb : float or None
        Target chunk size in MiB for backed I/O (None = auto).
    backed_n_threads : int
        Thread count for backed operator compute loops (0 = auto).
    """
    return reduce_kernel(
        adata=adata,
        n_components=int(np.asarray(svd_result["d"]).size),
        layer=layer,
        key_added=key_added,
        svd_algorithm=None,
        max_iter=0,
        seed=0,
        verbose=verbose,
        precomputed_svd=svd_result,
        backed_chunk_size=backed_chunk_size,
        inplace=inplace,
        backed_target_chunk_mb=backed_target_chunk_mb,
        backed_n_threads=backed_n_threads,
        lazy_transform=lazy_transform,
    )


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


def run_svd(
    X: Union[AnnData, np.ndarray, sp.spmatrix, Any],
    n_components: int = 30,
    algorithm: Optional[str] = "auto",
    max_iter: int = 0,
    seed: int = 0,
    verbose: bool = True,
    return_operator_compatible: bool = True,
    backed_chunk_size: int = 4096,
    layer: Optional[str] = None,
    allow_compressed: bool = False,
    backed_target_chunk_mb: Optional[float] = None,
    backed_n_threads: int = 0,
    lazy_transform: Optional[LazyTransform] = None,
) -> dict:
    """Compute truncated SVD decomposition.

    Parameters
    ----------
    X : AnnData, np.ndarray, scipy.sparse matrix, or backed matrix
        Input data.  AnnData inputs support backed mode (HDF5-streamed).
    n_components : int
        Number of singular values/vectors to compute.
    algorithm : str or None
        SVD algorithm: ``"auto"``, ``"irlb"``, ``"halko"``, ``"feng"``, or
        ``"primme"``.  ``"auto"`` selects based on matrix properties and
        storage mode.
    max_iter : int
        Maximum iterations for iterative solvers (0 = solver default).
    seed : int
        Random seed.
    verbose : bool
        Print progress messages.
    return_operator_compatible : bool
        If True, return only ``{"u", "d", "v"}`` suitable for
        :func:`reduce_kernel_from_svd`.
    backed_chunk_size : int
        Row chunk size for backed sparse streaming.
    layer : str or None
        Layer to use when ``X`` is an AnnData (None uses ``.X``).
    allow_compressed : bool
        Allow compressed backed storage without decompression.
    backed_target_chunk_mb : float or None
        Target chunk size in MiB for backed I/O (None = auto).
    backed_n_threads : int
        Thread count for backed operator compute loops (0 = auto).
        Only used for backed (operator) execution paths.
    lazy_transform : LazyTransform or None
        Pre-computed lazy transform for backed AnnData inputs.

    Returns
    -------
    dict
        SVD result with keys ``"u"`` (left singular vectors, cells x k),
        ``"d"`` (singular values), ``"v"`` (right singular vectors,
        features x k).
    """
    if backed_n_threads < 0:
        raise ValueError("`backed_n_threads` must be >= 0")

    algorithm_name = _normalize_algorithm(algorithm, context="algorithm")

    source_ctx: Optional[MatrixSource] = MatrixSource(X, layer=layer) if isinstance(X, AnnData) else None
    adata_ctx: Optional[AnnData] = X if isinstance(X, AnnData) else None
    matrix = source_ctx.matrix if source_ctx is not None else X

    # Validate the lazy transform before any expensive operation.
    if lazy_transform is not None:
        if adata_ctx is None:
            raise ValueError(
                "`lazy_transform` in `run_svd` requires a backed AnnData input "
                "so row-sum scaling factors can be streamed from the source matrix."
            )
        _validate_lazy_transform(lazy_transform, layer=layer, source=source_ctx)

    if _is_backed_matrix(matrix):
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source_ctx,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        ) if source_ctx is not None else (None, False, 1.0)
        if adata_ctx is not None:
            _flush_backed_handle(adata_ctx, context="run_svd")
        selected_algorithm = _select_svd_algorithm_backed(algorithm_name, verbose)
        io_target_chunk_bytes = _chunk_target_bytes(backed_target_chunk_mb)

        temp_path: Optional[str] = None
        op = None
        try:
            if adata_ctx is not None:
                temp_path = _maybe_decompress_backed_path(
                    adata_ctx,
                    layer=layer,
                    allow_compressed=allow_compressed,
                    chunk_size=backed_chunk_size,
                    verbose=verbose,
                    context="run_svd",
                )
                file_path = temp_path if temp_path is not None else str(adata_ctx.filename)
                group_path = _backed_group_path(layer)
            else:
                metadata = get_storage_metadata_from_matrix(matrix)
                if not allow_compressed and is_compressed_storage(metadata):
                    _warn_if_compressed_backed_svd(
                        metadata,
                        context="run_svd",
                        recommendation="run_svd(..., allow_compressed=True) or pass a backed AnnData object for auto-decompression",
                    )
                file_path, group_path = _resolve_backed_handle(matrix)

            op = _core.create_backed_operator(
                file_path=file_path,
                group_path=group_path,
                chunk_size=backed_chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
                io_target_chunk_bytes=io_target_chunk_bytes,
                n_threads=backed_n_threads,
            )
            result = _core.run_svd_backed_operator(
                op, n_components, max_iter, seed, selected_algorithm, verbose
            )
        finally:
            op = None
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)
    elif sp.issparse(matrix):
        if lazy_transform is not None:
            raise ValueError("`lazy_transform` is supported only for backed matrix inputs.")
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        algorithm_id = _select_svd_algorithm_inmemory(matrix, algorithm_name, verbose)
        result = _core.run_svd_sparse(matrix, n_components, max_iter, seed, algorithm_id, verbose)
    else:
        if lazy_transform is not None:
            raise ValueError("`lazy_transform` is supported only for backed matrix inputs.")
        algorithm_id = _select_svd_algorithm_inmemory(matrix, algorithm_name, verbose)
        result = _core.run_svd_dense(matrix, n_components, max_iter, seed, algorithm_id, verbose)

    if return_operator_compatible:
        result = {"u": result["u"], "d": result["d"], "v": result["v"]}

    return result
