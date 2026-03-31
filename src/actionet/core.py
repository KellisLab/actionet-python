"""High-level Python API wrapping C++ bindings with AnnData integration."""

import os
import shutil
import tempfile
import warnings
from typing import Any, Optional, Union, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from .anndata_utils import anndata_to_matrix
from ._backed_persist import persist_updates
from ._backed_compression import (
    format_compression_summary,
    get_storage_metadata_from_adata,
    get_storage_metadata_from_matrix,
    is_compressed_storage,
)
from ._matrix_source import MatrixSource
from .preprocessing import decompress_backed_storage
from . import tools


_WARNED_COMPRESSED_BACKED_SVD: set[tuple[str, str, str]] = set()
_SVD_ALGORITHM_TO_ID = {
    "irlb": 0,
    "halko": 1,
    "feng": 2,
    "primme": 3,
}
_SVD_ID_TO_ALGORITHM = {v: k for k, v in _SVD_ALGORITHM_TO_ID.items()}


def _is_backed_matrix(X: Any) -> bool:
    """Detect whether ``X`` is backed/on-disk rather than fully in memory."""
    if sp.issparse(X) or isinstance(X, np.ndarray):
        return False

    if hasattr(X, "isbacked"):
        return bool(X.isbacked)

    if hasattr(X, "group"):
        return True

    mod = type(X).__module__
    if mod and mod.startswith("h5py"):
        return True

    return False


def _warn_if_compressed_backed_svd(
    metadata: Optional[dict],
    *,
    context: str,
    recommendation: str,
) -> None:
    """Warn once per (file, matrix key, context) for compressed backed SVD."""
    if not is_compressed_storage(metadata):
        return

    filename = str((metadata or {}).get("filename") or "<unknown>")
    matrix_key = str((metadata or {}).get("matrix_key") or "<unknown>")
    dedupe_key = (filename, matrix_key, context)
    if dedupe_key in _WARNED_COMPRESSED_BACKED_SVD:
        return
    _WARNED_COMPRESSED_BACKED_SVD.add(dedupe_key)

    codecs = format_compression_summary(metadata)
    warnings.warn(
        (
            f"Backed operator SVD in `{context}` is reading compressed storage "
            f"for `{matrix_key}` ({codecs}). This can cause major runtime "
            f"slowdowns due to repeated decompression during matvec passes. "
            f"Recommended: `{recommendation}`."
        ),
        UserWarning,
        stacklevel=3,
    )


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


def _backed_group_path(layer: Optional[str]) -> str:
    return "/X" if layer is None else f"/layers/{layer}"


def _resolve_backed_handle(X: Any, layer: Optional[str] = None) -> tuple[str, str]:
    if isinstance(X, AnnData):
        if not bool(getattr(X, "isbacked", False) and getattr(X, "filename", None)):
            raise ValueError("Backed AnnData expected")
        return str(X.filename), _backed_group_path(layer)

    group = getattr(X, "group", None)
    if group is None or not hasattr(group, "file"):
        raise ValueError("Unable to resolve backed file/group from matrix handle")

    return str(group.file.filename), str(group.name)


def _flush_backed_handle(adata: AnnData, *, context: str) -> None:
    """Best-effort flush for backed AnnData before opening a second HDF5 handle."""
    if not bool(getattr(adata, "isbacked", False)):
        return

    file_obj = getattr(getattr(adata, "file", None), "_file", None)
    if file_obj is None:
        return

    try:
        file_obj.flush()
    except Exception as exc:  # pragma: no cover - depends on h5py backend state
        warnings.warn(
            (
                f"{context}: failed to flush backed AnnData handle before "
                f"operator read ({type(exc).__name__}: {exc})."
            ),
            UserWarning,
            stacklevel=3,
        )


def _chunk_target_bytes(backed_target_chunk_mb: Optional[float]) -> int:
    if backed_target_chunk_mb is None:
        # 0 delegates to the C++ backed-operator auto heuristic that scales with
        # chunk_size and sparse structure.
        return 0
    target = float(backed_target_chunk_mb)
    if target <= 0:
        raise ValueError("`backed_target_chunk_mb` must be > 0 when provided")
    return int(target * 1024 * 1024)


def _maybe_decompress_backed_path(
    adata: AnnData,
    *,
    layer: Optional[str],
    allow_compressed: bool,
    chunk_size: int,
    verbose: bool,
    context: str,
) -> Optional[str]:
    if allow_compressed:
        return None

    metadata = get_storage_metadata_from_adata(adata, layer=layer)
    if not is_compressed_storage(metadata):
        return None

    src_path = str(adata.filename)
    parent = os.path.dirname(src_path) or "."
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = max(int(os.path.getsize(src_path) * 3), 1)
    if free_bytes < required_bytes:
        codecs = format_compression_summary(metadata)
        warnings.warn(
            (
                f"{context}: insufficient disk for auto-decompression "
                f"(need ~{required_bytes / 1e9:.1f} GB free, have {free_bytes / 1e9:.1f} GB). "
                f"Continuing with compressed matrix ({codecs})."
            ),
            UserWarning,
            stacklevel=3,
        )
        return None

    fd, tmp_path = tempfile.mkstemp(prefix="actionet_oom_", suffix=".h5ad", dir=parent)
    os.close(fd)
    os.unlink(tmp_path)
    decompressed = decompress_backed_storage(
        adata,
        layer=layer,
        scope="matrix",
        output_file=tmp_path,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    if decompressed is not None and getattr(decompressed, "file", None) is not None:
        decompressed.file.close()
    return tmp_path


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
) -> Optional[AnnData]:
    """Compute low-rank kernel reduction and persist outputs to AnnData.

    Notes
    -----
    `backed_n_threads` is used only for backed (operator) execution paths.
    In-memory dense/sparse paths continue to use existing BLAS/library threading.
    """
    if backed_n_threads < 0:
        raise ValueError("`backed_n_threads` must be >= 0")

    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)
    X = source.matrix
    use_operator = source.is_backed
    algorithm_name = _normalize_algorithm(svd_algorithm, context="svd_algorithm")

    if use_operator:
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
                row_scale_factors=None,
                apply_log1p=False,
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
        specificity_threshold, min_observations, n_threads
    )

    persist_updates(
        adata,
        obsm={
            "H_stacked": result["H_stacked"],    # cells x archetypes, direct
            "H_merged": result["H_merged"],      # direct
            "C_stacked": result["C_stacked"],
            "C_merged": result["C_merged"],
        },
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
    adata: AnnData,
    scores: Union[str, np.ndarray],
    norm_method: Literal["pagerank", "pagerank_sym"] = "pagerank",
    alpha: float = 0.85,
    n_threads: int = 0,
    approx: bool = True,
    max_iter: int = 5,
    tol = 1e-8,
    network_key: str = "actionet",
    key_added: str = "diffused",
    inplace: bool = True,
) -> Optional[AnnData]:
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
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Diffused scores (cells × features or cells × 1).
    """
    if not inplace:
        adata = adata.copy()
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

    # Ensure C-contiguous memory layout for C++ compatibility
    X0 = np.ascontiguousarray(X0)

    X_diffused = _core.compute_network_diffusion(
        G = G,
        X0 = X0,
        alpha = alpha,
        max_it = max_iter,
        thread_no = n_threads,
        approx = approx,
        norm_method = 2 if norm_method == "pagerank_sym" else 0,
        tol = tol
    )

    persist_updates(adata, obsm={key_added: X_diffused})
    if not inplace:
        return adata
    return None


def _labels_to_membership(labels_int: np.ndarray, n_obs: int) -> np.ndarray:
    labels_int = np.asarray(labels_int, dtype=np.int64).reshape(-1)
    if labels_int.shape[0] != n_obs:
        raise ValueError(
            f"labels length ({labels_int.shape[0]}) does not match number of observations ({n_obs})"
        )

    max_label = int(labels_int.max()) if labels_int.size > 0 else 0
    if max_label <= 0:
        raise ValueError("No valid labels after conversion; ensure labels contain at least one non-missing category.")

    H = np.zeros((n_obs, max_label), dtype=np.float64)
    valid = labels_int > 0
    H[np.arange(n_obs)[valid], labels_int[valid] - 1] = 1.0
    return H


def _run_specificity_backed_sparse(
    adata: AnnData,
    layer: Optional[str],
    chunk_size: int,
    *,
    H: Optional[np.ndarray] = None,
    labels_int: Optional[np.ndarray] = None,
    n_threads: int = 0,
) -> dict:
    """Dispatch backed sparse specificity through the C++ ABI.

    Exactly one of *H* (archetype/membership matrix, shape ``(n_obs, k)``) or
    *labels_int* (1-based integer labels, shape ``(n_obs,)``) must be supplied.

    Returns the raw result dict from the C++ binding.
    """
    file_path = str(adata.filename)
    group_path = _backed_group_path(layer)
    op = None
    try:
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=chunk_size,
            row_scale_factors=None,
            apply_log1p=False,
        )
        if H is not None:
            return _core.archetype_feature_specificity_backed_operator(op, H, n_threads)
        else:
            return _core.compute_feature_specificity_backed_operator(op, labels_int, n_threads)
    finally:
        op = None


def _run_specificity_backed_dense(
    adata: AnnData,
    layer: Optional[str],
    chunk_size: int,
    *,
    H: Optional[np.ndarray] = None,
    labels_int: Optional[np.ndarray] = None,
    n_threads: int = 0,
) -> dict:
    """Dispatch backed dense specificity through the C++ ABI (BackedDenseMatrixOperator).

    Exactly one of *H* (archetype/membership matrix, shape ``(n_obs, k)``) or
    *labels_int* (1-based integer labels, shape ``(n_obs,)``) must be supplied.

    Returns the raw result dict from the C++ binding.
    """
    file_path = str(adata.filename)
    group_path = _backed_group_path(layer)
    op = None
    try:
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=chunk_size,
            row_scale_factors=None,
            apply_log1p=False,
        )
        if H is not None:
            return _core.archetype_feature_specificity_backed_dense_operator(op, H, n_threads)
        else:
            return _core.compute_feature_specificity_backed_dense_operator(op, labels_int, n_threads)
    finally:
        op = None


def _compute_specificity_streamed(
    source: MatrixSource,
    H_cells: np.ndarray,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Pure-Python streamed specificity (kept for reference/debugging only).

    .. deprecated::
        This function is no longer called by any production path.  Both
        sparse-backed and dense-backed specificity now route through C++
        (``BackedSparseMatrixOperator`` and ``BackedDenseMatrixOperator``
        respectively).  This implementation is retained only as a reference
        for testing or debugging; it will be removed in a future cleanup.

    **Algorithm overview** (mirrors ``specificity.cpp``):

    1. Shift the matrix so all values are non-negative (subtract global
       minimum).
    2. Normalise the membership matrix *H* column-wise by dividing each
       column by its mean.
    3. Accumulate, in a single streaming pass over row-chunks:
       - ``row_count``  -- nnz count per feature (column)
       - ``col_count``  -- nnz count per observation (row)
       - ``row_factor_sum`` -- column sums of the shifted matrix
       - ``obs``        -- ``X_shifted.T @ H_norm`` (observed feature--group
         co-occurrence)
    4. Derive per-feature and per-observation density estimates
       ``row_p``, ``col_p`` and a relative-density weight ``beta``.
    5. Compute *expected* co-occurrence ``exp`` and its variance proxy
       ``nu`` under a null model, then evaluate one-sided Bernstein-type
       tail bounds, yielding ``log10``-scaled upper (enrichment) and lower
       (depletion) significance matrices.

    Parameters
    ----------
    source : MatrixSource
        Expression matrix accessor (cells x features).
    H_cells : ndarray, shape ``(n_obs, k)``
        Group-membership or archetype-footprint matrix.
    chunk_size : int
        Rows per streaming chunk.

    Returns
    -------
    dict with keys ``"average_profile"``, ``"upper_significance"``,
    ``"lower_significance"`` -- all ``(n_vars, k)`` arrays.
    """
    H_cells = np.asarray(H_cells, dtype=np.float64, order="C")
    if H_cells.ndim != 2 or H_cells.shape[0] != source.n_obs:
        raise ValueError(
            f"H_cells must have shape (n_obs, k) where n_obs={source.n_obs}, got {H_cells.shape}"
        )
    if H_cells.shape[1] == 0:
        raise ValueError("H_cells must contain at least one archetype/label column.")

    col_mean = H_cells.mean(axis=0)
    col_mean[col_mean == 0] = 1.0
    Ht = H_cells / col_mean[np.newaxis, :]

    min_val = source.global_min(chunk_size=chunk_size)

    row_count = np.zeros(source.n_vars, dtype=np.float64)
    row_factor_sum = np.zeros(source.n_vars, dtype=np.float64)
    col_count = np.zeros(source.n_obs, dtype=np.float64)
    obs = np.zeros((source.n_vars, Ht.shape[1]), dtype=np.float64)

    for chunk in source.iter_row_chunks(chunk_size=chunk_size):
        block = chunk.block
        h_block = Ht[chunk.start:chunk.end, :]

        if sp.issparse(block):
            block_csr = block.tocsr(copy=False)

            # Count nnz on the *original* block -- this matches the C++ sparse
            # iterator which visits all stored elements before and after the
            # min-shift.  Using nnz rather than positivity-after-shift ensures
            # numerical parity with the in-memory C++ path.
            row_count += np.asarray(block_csr.getnnz(axis=0)).ravel()
            col_count[chunk.start:chunk.end] = np.asarray(block_csr.getnnz(axis=1)).ravel()

            if min_val != 0.0:
                block_csr = block_csr.copy()
                block_csr.data = block_csr.data - min_val

            row_factor_sum += np.asarray(block_csr.sum(axis=0)).ravel()
            obs += np.asarray(block_csr.T.dot(h_block))
        else:
            arr = np.asarray(block, dtype=np.float64)
            if min_val != 0.0:
                arr = arr - min_val

            pos = arr > 0
            row_count += pos.sum(axis=0)
            col_count[chunk.start:chunk.end] = pos.sum(axis=1)
            row_factor_sum += arr.sum(axis=0)
            obs += arr.T @ h_block

    row_factor = np.divide(
        row_factor_sum,
        row_count,
        out=np.zeros_like(row_factor_sum),
        where=row_count > 0,
    )

    row_p = row_count / float(source.n_obs if source.n_obs > 0 else 1)
    col_p = col_count / float(source.n_vars if source.n_vars > 0 else 1)

    rho = float(col_p.mean()) if col_p.size > 0 else 0.0
    beta = np.zeros_like(col_p) if rho == 0.0 else (col_p / rho)

    gamma = Ht * beta[:, np.newaxis]
    a = gamma.max(axis=0) if gamma.size > 0 else np.zeros(Ht.shape[1], dtype=np.float64)

    exp = np.outer(row_p * row_factor, gamma.sum(axis=0))
    nu = np.outer(row_p * np.square(row_factor), np.square(gamma).sum(axis=0))
    A = np.outer(row_factor, a)
    lamb = obs - exp

    with np.errstate(divide="ignore", invalid="ignore"):
        log_lower = np.square(lamb) / (2.0 * nu)
        log_upper = np.square(lamb) / (2.0 * (nu + (lamb * A / 3.0)))

    log_lower[lamb >= 0] = 0.0
    log_upper[lamb <= 0] = 0.0

    scale = np.log(10.0)
    log_lower = np.nan_to_num(log_lower, nan=0.0, posinf=0.0, neginf=0.0) / scale
    log_upper = np.nan_to_num(log_upper, nan=0.0, posinf=0.0, neginf=0.0) / scale

    return {
        "average_profile": obs / float(source.n_obs if source.n_obs > 0 else 1),
        "upper_significance": log_upper,
        "lower_significance": log_lower,
    }


def compute_feature_specificity(
    adata: AnnData,
    labels: Union[str, np.ndarray],
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "specificity",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
    return_raw: bool = False,
) -> Optional[Union[AnnData, dict]]:
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
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk for streamed specificity computation on
        backed AnnData.  Ignored for in-memory objects.
    return_raw : bool, optional (default: False)
        If True, returns raw results dict without modifying adata.
        Useful for temporary computations to avoid expensive AnnData copies.

    Returns
    -------
    None, AnnData, or dict
        - If return_raw=True: returns dict with keys "average_profile", "upper_significance", "lower_significance"
        - If inplace=True and return_raw=False: returns None and modifies adata in place
        - If inplace=False and return_raw=False: returns a new AnnData object with the results

    Notes
    -----
    When *adata* is in backed mode (on-disk), the function dispatches
    through the C++ ABI for sparse-backed matrices, performing a
    single streaming scan over the HDF5 data without loading the full
    matrix into memory.  Dense-backed matrices use
    ``BackedDenseMatrixOperator`` (chunked hyperslab reads via HDF5)
    without loading the full matrix into memory.

    Updates AnnData (if return_raw=False)
    --------------
    adata.varm[f"{key_added}_profile"] : np.ndarray
        Average feature profile per cluster/archetype (features × clusters).
    adata.varm[f"{key_added}_upper"] : np.ndarray
        Upper-tail significance scores (features × clusters).
    adata.varm[f"{key_added}_lower"] : np.ndarray
        Lower-tail significance scores (features × clusters).
    """
    if not inplace and not return_raw:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    if isinstance(labels, str):
        if labels not in adata.obs:
            raise ValueError(f"Labels '{labels}' not found in adata.obs.")
        labels_arr = adata.obs[labels].values
    else:
        labels_arr = np.asarray(labels)

    # Normalise pandas Categorical to a plain object array so that the
    # subsequent Categorical() call always produces a deterministic
    # (lexicographic) category ordering, independent of any pre-existing
    # category order stored in the h5ad file.
    if hasattr(labels_arr, 'categories'):
        labels_arr = np.asarray(labels_arr)

    # Convert to integer labels
    from pandas import Categorical
    if not np.issubdtype(labels_arr.dtype, np.integer):
        cat = Categorical(labels_arr)
        labels_int = cat.codes.astype(np.int32)
    else:
        labels_int = labels_arr.astype(np.int32)

    # Function expects 1-based labels, so add 1
    labels_int = labels_int + 1

    if source.is_backed:
        if source.is_sparse:
            # Sparse-backed: dispatch through the C++ ABI for full performance.
            result = _run_specificity_backed_sparse(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                labels_int=labels_int,
                n_threads=n_threads,
            )
        else:
            # Dense-backed: dispatch through the C++ ABI (BackedDenseMatrixOperator).
            result = _run_specificity_backed_dense(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                labels_int=labels_int,
                n_threads=n_threads,
            )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        if sp.issparse(S):
            result = _core.compute_feature_specificity_sparse(S, labels_int, n_threads)
        else:
            result = _core.compute_feature_specificity_dense(S, labels_int, n_threads)

    if return_raw:
        return result

    persist_updates(
        adata,
        varm={
            f"{key_added}_profile": result["average_profile"],
            f"{key_added}_upper": result["upper_significance"],
            f"{key_added}_lower": result["lower_significance"],
        },
    )
    if not inplace:
        return adata
    return None


def compute_archetype_feature_specificity(
    adata: AnnData,
    archetype_key: Union[str, np.ndarray] = "archetype_footprint",
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "archetype",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
) -> Optional[AnnData]:
    """
    Compute feature specificity scores for archetypes using archetype matrix.

    This function is analogous to archetypeFeatureSpecificity() in R.
    It computes feature enrichment for each archetype using the archetype
    footprint matrix (typically the diffused H_merged matrix).

    Parameters
    ----------
    adata
        Annotated data matrix.
    archetype_key
        Either key in adata.obsm containing archetype matrix (cells × archetypes)
        or the archetype matrix itself as numpy array.
    layer
        Layer to use (None uses .X).
    n_threads
        Number of threads.
    key_added
        Prefix for storing results in adata.varm.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk for streamed specificity computation on
        backed AnnData.  Ignored for in-memory objects.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place.
        If inplace=False, returns a new AnnData object with the results.

    Notes
    -----
    When *adata* is in backed mode (on-disk), the function dispatches
    through the C++ ABI for sparse-backed matrices, performing a
    single streaming scan over the HDF5 data without loading the full
    matrix into memory.  Dense-backed matrices use
    ``BackedDenseMatrixOperator`` (chunked hyperslab reads via HDF5)
    without loading the full matrix into memory.

    Updates AnnData
    --------------
    adata.varm[f"{key_added}_feat_profile"] : np.ndarray
        Average feature profile per archetype (features × archetypes).
    adata.varm[f"{key_added}_feat_specificity_upper"] : np.ndarray
        Upper-tail significance scores (features × archetypes).
    adata.varm[f"{key_added}_feat_specificity_lower"] : np.ndarray
        Lower-tail significance scores (features × archetypes).
    """
    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    if isinstance(archetype_key, str):
        if archetype_key not in adata.obsm:
            raise ValueError(f"Archetype matrix '{archetype_key}' not found in adata.obsm.")
        H = adata.obsm[archetype_key]
    else:
        H = np.asarray(archetype_key)

    H = np.ascontiguousarray(H, dtype=np.float64)

    if source.is_backed:
        if source.is_sparse:
            # Sparse-backed: H is (n_obs, k); pass directly (C++ now accepts cells × k).
            result = _run_specificity_backed_sparse(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                H=H,
                n_threads=n_threads,
            )
        else:
            # Dense-backed: dispatch through the C++ ABI (BackedDenseMatrixOperator).
            result_stream = _run_specificity_backed_dense(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                H=H,
                n_threads=n_threads,
            )
            result = {
                "archetypes": result_stream["archetypes"],
                "upper_significance": result_stream["upper_significance"],
                "lower_significance": result_stream["lower_significance"],
            }
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        if sp.issparse(S):
            result = _core.archetype_feature_specificity_sparse(S, H, n_threads)
        else:
            result = _core.archetype_feature_specificity_dense(S, H, n_threads)

    persist_updates(
        adata,
        varm={
            f"{key_added}_feat_profile": result["archetypes"],
            f"{key_added}_feat_specificity_upper": result["upper_significance"],
            f"{key_added}_feat_specificity_lower": result["lower_significance"],
        },
    )

    if not inplace:
        return adata
    return None


def layout_network(
    adata: AnnData,
    network_key: str = "actionet",
    initial_coords: Optional[Union[str, np.ndarray]] = None,
    layer: Optional[str] = None,
    method: Literal["umap", "tumap"] = "umap",
    n_components: int = 2,
    spread: float = 1.0,
    min_dist: float = 1.0,
    n_epochs: int = 0,
    seed: int = 0,
    n_threads: int = 0,
    verbose: bool = True,
    key_added: str = "X_umap",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute 2D/3D layout of ACTIONet graph using UMAP.

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
        G, initial_coords, method, n_components,
        spread, min_dist, n_epochs, seed, n_threads, verbose
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
) -> dict:
    """Compute truncated SVD decomposition.

    Notes
    -----
    `backed_n_threads` is used only for backed (operator) execution paths.
    In-memory dense/sparse paths continue to use existing BLAS/library threading.
    """
    if backed_n_threads < 0:
        raise ValueError("`backed_n_threads` must be >= 0")

    algorithm_name = _normalize_algorithm(algorithm, context="algorithm")

    adata_ctx: Optional[AnnData] = X if isinstance(X, AnnData) else None
    matrix = MatrixSource(X, layer=layer).matrix if isinstance(X, AnnData) else X

    if _is_backed_matrix(matrix):
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
                row_scale_factors=None,
                apply_log1p=False,
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
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        algorithm_id = _select_svd_algorithm_inmemory(matrix, algorithm_name, verbose)
        result = _core.run_svd_sparse(matrix, n_components, max_iter, seed, algorithm_id, verbose)
    else:
        algorithm_id = _select_svd_algorithm_inmemory(matrix, algorithm_name, verbose)
        result = _core.run_svd_dense(matrix, n_components, max_iter, seed, algorithm_id, verbose)

    if return_operator_compatible:
        result = {"u": result["u"], "d": result["d"], "v": result["v"]}

    return result
