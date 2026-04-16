"""Kernel reduction and SVD utilities."""

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

_SVD_ALGORITHM_TO_ID = {
    "irlb": 0,
    "halko": 1,
    "feng": 2,
    "primme": 3,
}
_SVD_ID_TO_ALGORITHM = {v: k for k, v in _SVD_ALGORITHM_TO_ID.items()}

_COMPUTE_BACKEND_TO_ID = {
    "cpu": 0,
    "gpu": 1,
    "auto": 2,
}
_COMPUTE_BACKEND_ID_TO_NAME = {v: k for k, v in _COMPUTE_BACKEND_TO_ID.items()}


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


def _normalize_compute_backend(backend: Optional[str], *, context: str) -> str:
    if backend is None:
        return "auto"
    if not isinstance(backend, str):
        raise TypeError(f"`{context}` must be a string backend name")
    name = backend.strip().lower()
    allowed = {"auto", "cpu", "gpu"}
    if name not in allowed:
        raise ValueError(f"Invalid backend `{backend}`. Allowed: {sorted(allowed)}")
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
    compute_backend: Optional[str] = "auto",
    device_id: int = 0,
    allow_cpu_fallback: bool = True,
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
    compute_backend : str or None
        Backend policy for SVD execution: ``"auto"``, ``"cpu"``, or ``"gpu"``.
        GPU execution currently applies only to in-memory PRIMME SVD paths.
    device_id : int
        CUDA device ordinal used when GPU backend is selected.
    allow_cpu_fallback : bool
        If True, unsupported GPU requests automatically fall back to CPU.
        If False, unsupported GPU requests raise an error.
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
    if device_id < 0:
        raise ValueError("`device_id` must be >= 0")

    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    # Validate the lazy transform before any expensive operation.
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    use_operator = source.is_backed
    algorithm_name = _normalize_algorithm(svd_algorithm, context="svd_algorithm")
    backend_name = _normalize_compute_backend(compute_backend, context="compute_backend")
    backend_id = _COMPUTE_BACKEND_TO_ID[backend_name]
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
                    op, n_components, selected_algorithm, max_iter, seed, verbose,
                    backend_id, device_id, allow_cpu_fallback
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
                result = _core.reduce_kernel_sparse(
                    S, n_components, svd_algorithm_id, max_iter, seed, verbose,
                    backend_id, device_id, allow_cpu_fallback
                )
            else:
                result = _core.reduce_kernel_dense(
                    S, n_components, svd_algorithm_id, max_iter, seed, verbose,
                    backend_id, device_id, allow_cpu_fallback
                )
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
        "compute_backend": backend_id,
        "compute_backend_name": _COMPUTE_BACKEND_ID_TO_NAME.get(backend_id, f"unknown({backend_id})"),
        "device_id": device_id,
        "allow_cpu_fallback": bool(allow_cpu_fallback),
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
    compute_backend: Optional[str] = "auto",
    device_id: int = 0,
    allow_cpu_fallback: bool = True,
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
    compute_backend : str or None
        Backend policy for SVD execution: ``"auto"``, ``"cpu"``, or ``"gpu"``.
        GPU execution currently applies only to in-memory PRIMME SVD paths.
    device_id : int
        CUDA device ordinal used when GPU backend is selected.
    allow_cpu_fallback : bool
        If True, unsupported GPU requests automatically fall back to CPU.
        If False, unsupported GPU requests raise an error.
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
    if device_id < 0:
        raise ValueError("`device_id` must be >= 0")

    algorithm_name = _normalize_algorithm(algorithm, context="algorithm")
    backend_name = _normalize_compute_backend(compute_backend, context="compute_backend")
    backend_id = _COMPUTE_BACKEND_TO_ID[backend_name]

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
                op, n_components, max_iter, seed, selected_algorithm, verbose,
                backend_id, device_id, allow_cpu_fallback
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
        result = _core.run_svd_sparse(
            matrix, n_components, max_iter, seed, algorithm_id, verbose,
            backend_id, device_id, allow_cpu_fallback
        )
    else:
        if lazy_transform is not None:
            raise ValueError("`lazy_transform` is supported only for backed matrix inputs.")
        algorithm_id = _select_svd_algorithm_inmemory(matrix, algorithm_name, verbose)
        result = _core.run_svd_dense(
            matrix, n_components, max_iter, seed, algorithm_id, verbose,
            backend_id, device_id, allow_cpu_fallback
        )

    if return_operator_compatible:
        result = {"u": result["u"], "d": result["d"], "v": result["v"]}

    return result


def smooth_kernel(
    adata: AnnData,
    network_key: str = "actionet",
    reduction_key: str = "action",
    alpha: float = 0.85,
    max_iter: int = 5,
    norm_method: Union[int, Literal["pagerank", "pagerank_sym"]] = "pagerank",
    n_threads: int = 0,
    key_added: str = "action_smoothed",
    return_raw: bool = False,
) -> Union[AnnData, dict]:
    """
    Smooth the reduced kernel over the network.

    This function applies network diffusion to the reduced representation,
    which can improve downstream analysis by leveraging local structure.

    Parameters
    ----------
    adata
        Annotated data matrix with network and reduction.
    network_key
        Key in adata.obsp containing network.
    reduction_key
        Key in adata.obsm containing reduction to smooth.
    alpha
        Diffusion parameter.
    max_iter
        Number of iterations.
    norm_method
        Normalization method.
    n_threads
        Number of threads.
    key_added
        Key to store smoothed reduction.
    return_raw
        If True, return raw diffusion outputs instead of updating adata.

    Returns
    -------
    Updates adata with:
        - adata.obsm[key_added]: Smoothed reduction
    Or, if return_raw=True:
        - Dictionary with raw outputs from diffusion and SVD.
    """
    if isinstance(norm_method, str):
        norm_method_code = 2 if norm_method == "pagerank_sym" else 0
    else:
        norm_method_code = int(norm_method)

    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")

    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found.")

    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")

    G = adata.obsp[network_key]
    S_r = np.asarray(adata.obsm[reduction_key], dtype=float, order="C")
    sigma = np.asarray(adata.uns[params_key]["sigma"], dtype=float).reshape(-1)

    U_left = np.asarray(adata.varm[f"{reduction_key}_U"], dtype=float, order="C")
    A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")

    if sigma.shape[0] != S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    V_right = S_r / sigma[np.newaxis, :]
    svd_out = _core.perturbed_svd(U_left, sigma, V_right, -A, B)

    V_smooth = _core.compute_network_diffusion(
        G, svd_out["v"], alpha, max_iter, n_threads, True, norm_method_code, 1e-8
    )
    H = V_smooth @ np.diag(svd_out["d"])

    if return_raw:
        return {
            "U": U_left,
            "SVD_out": svd_out,
            "V_smooth": V_smooth,
            "H": H,
        }

    persist_updates(adata, obsm={key_added: H})
    return adata
