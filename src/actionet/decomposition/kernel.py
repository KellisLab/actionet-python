"""Reduced-kernel decomposition of expression matrices.

Wraps the C++ reduce_kernel primitive for both in-memory and backed
inputs, with optional lazy transforms and precomputed SVDs.
"""

import os
from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .. import _core
from ..io.lazy_transform import (
    LazyTransform,
    _lazy_params_for_metadata,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)
from ..io.matrix_source import MatrixSource
from ..io.operator import open_backed_operator_for
from ..io.persist import persist_updates
from ..tools.anndata import anndata_to_matrix
from .svd import (
    _SVD_ID_TO_ALGORITHM,
    _chunk_target_bytes,
    _maybe_decompress_backed_path,
    _normalize_algorithm,
    _select_svd_algorithm_backed,
    _select_svd_algorithm_inmemory,
)


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
        SVD algorithm: ``"auto"``, ``"irlb"``, ``"halko"``, or ``"feng"``.
        ``"auto"`` selects based on matrix properties.
    max_iter : int
        Maximum iterations for iterative SVD solvers (0 = solver default).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress messages.
    precomputed_svd : dict or None
        Pre-computed SVD with keys ``"u"``, ``"d"``, ``"v"``.
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
    lazy_transform : LazyTransform or None
        Pre-computed lazy transform for backed inputs on ``.X`` only.

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
        selected_algorithm = _select_svd_algorithm_backed(algorithm_name, verbose)
        io_target_chunk_bytes = _chunk_target_bytes(backed_target_chunk_mb)
        temp_path: Optional[str] = None
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

            with open_backed_operator_for(
                adata,
                layer=layer,
                context="reduce_kernel",
                chunk_size=backed_chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
                file_path=file_path,
                io_target_chunk_bytes=io_target_chunk_bytes,
                n_threads=backed_n_threads,
            ) as op:
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
            key_added: result["S_r"],
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

    Thin wrapper around :func:`reduce_kernel` that infers ``n_components``
    from the SVD result and passes it as ``precomputed_svd``.
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
    """Smooth the reduced kernel over the network.

    Applies network diffusion to the reduced representation, which can
    improve downstream analysis by leveraging local structure.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with network and reduction.
    network_key : str
        Key in adata.obsp containing network.
    reduction_key : str
        Key in adata.obsm containing reduction to smooth.
    alpha : float
        Diffusion parameter.
    max_iter : int
        Number of iterations.
    norm_method : str or int
        Normalization method.
    n_threads : int
        Number of threads.
    key_added : str
        Key to store smoothed reduction.
    return_raw : bool
        If True, return raw diffusion outputs instead of updating adata.

    Returns
    -------
    None or dict
        Updates ``adata.obsm[key_added]`` (or returns raw dict if
        ``return_raw=True``).
    """
    from ..tools.anndata import norm_method_to_int, resolve_network

    norm_method_code = norm_method_to_int(norm_method)

    resolve_network(adata, network_key)

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
