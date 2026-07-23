"""Truncated SVD utilities and algorithm selection.

Provides :func:`run_svd` (public) plus the private algorithm-selection and
compression handling helpers used by both :func:`run_svd` and the kernel
routines in :mod:`.kernel`.
"""

import os
import shutil
import tempfile
import warnings
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .. import _core
from ..io.compression import (
    format_compression_summary,
    get_storage_metadata_from_adata,
    is_compressed_storage,
)
from ..io.chunking import (
    DEFAULT_BACKED_READ_CHUNK_SIZE,
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    resolve_backed_write_chunk_size,
)
from ..io.lazy_transform import (
    LazyTransform,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)
from ..io.matrix_source import MatrixSource
from ..io.operator import open_backed_operator_for

_SVD_ALGORITHM_TO_ID = {
    "irlb": 0,
    "halko": 1,
}
_SVD_ID_TO_ALGORITHM = {v: k for k, v in _SVD_ALGORITHM_TO_ID.items()}
_SVD_BACKEND_CPU = "cpu"


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
    """Auto-select an SVD algorithm for an in-memory matrix.

    Defaults:

    - Sparse inputs -> IRLB. Empirically 3-6x faster than Halko on real
      single-cell matrices at 25k-200k cells with sigma_corr >= 0.999998
      relative to the reference solver. See
      ``tests/benchmark_svd_inmemory_defaults.py`` and the report at
      ``docs/svd_algorithm_benchmark.md``.
    - Dense inputs -> Halko. Halko is roughly 2x faster than IRLB on dense
      input at every tier benchmarked. Same benchmark reference as above.

    Sparse ``nnz`` is 64-bit clean because ``libactionet`` force-defines
    ``ARMA_64BIT_WORD``, so IRLB handles matrices with more than 2^31 non-zero
    entries directly. The remaining hard limit is per-axis: matrix row and
    column counts must fit in ``INT_MAX`` (~2.1B). Inputs that exceed that
    limit will fail inside the C++ SVD entry points with a clear error.
    """
    if algorithm != "auto":
        return _SVD_ALGORITHM_TO_ID[algorithm]

    if sp.issparse(S):
        total_elements = S.nnz
        sparsity = 1.0 - (total_elements / np.prod(S.shape))
        if verbose:
            print(f"Auto-selected IRLB for sparse matrix ({sparsity:.1%} sparse)")
        return _SVD_ALGORITHM_TO_ID["irlb"]

    total_elements = np.prod(S.shape)
    if verbose:
        print(f"Auto-selected Halko for dense matrix ({total_elements:,} elements)")
    return _SVD_ALGORITHM_TO_ID["halko"]


def _select_svd_algorithm_backed(algorithm: str, verbose: bool = True) -> int:
    """Auto-select an SVD algorithm for a backed (HDF5-streamed) matrix.

    For backed inputs, Halko is the unconditional auto-default. Empirical
    evidence across 25k-200k cell tiers shows:

    - IRLB is 6.5-7.3x slower than Halko on backed data at every tier
      (``tests/benchmark_backed_svd_algorithm.py``).
    - Both algorithms produce singular values with correlation >= 0.999998
      relative to Halko.

    Halko is preferred as the default because its fixed matvec count
    (``2*(iters+1)`` passes) gives a predictable NNZ-proportional I/O cost
    model, which is easier to reason about for atlas-scale streaming than
    IRLB's convergence-driven iteration count. See ``context/DECISIONS.md``
    ("Backed SVD algorithm default") and ``docs/svd_algorithm_benchmark.md``.
    IRLB remains available as an explicit backed choice via ``algorithm=``.
    """
    if algorithm == "auto":
        if verbose:
            print("Detected backed matrix: selecting Halko operator path")
        return _SVD_ALGORITHM_TO_ID["halko"]

    if algorithm not in {"halko", "irlb"}:
        raise ValueError("Backed matrices support only 'auto', 'halko', or 'irlb'")
    return _SVD_ALGORITHM_TO_ID[algorithm]


def _chunk_target_bytes(backed_target_chunk_mb: Optional[float]) -> int:
    if backed_target_chunk_mb is None:
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
    write_chunk_size: int,
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

    from ..preprocessing import decompress_backed_storage

    decompressed = decompress_backed_storage(
        adata,
        layer=layer,
        scope="matrix",
        output_file=tmp_path,
        backed_write_chunk_size=write_chunk_size,
        verbose=verbose,
    )
    if decompressed is not None and getattr(decompressed, "file", None) is not None:
        decompressed.file.close()
    return tmp_path


def run_svd(
    X: Union[AnnData, np.ndarray, sp.spmatrix],
    n_components: int = 30,
    algorithm: Optional[str] = "auto",
    max_iter: int = 0,
    seed: int = 0,
    verbose: bool = True,
    return_operator_compatible: bool = True,
    backed_chunk_size: int = DEFAULT_BACKED_READ_CHUNK_SIZE,
    layer: Optional[str] = None,
    allow_compressed: bool = False,
    backed_target_chunk_mb: Optional[float] = None,
    backed_n_threads: int = 0,
    lazy_transform: Optional[LazyTransform] = None,
    backed_write_chunk_size: Optional[int] = None,
) -> dict:
    """Compute truncated SVD decomposition.

    Parameters
    ----------
    X : AnnData, np.ndarray, or scipy.sparse matrix
        Input data. Backed (HDF5-streamed) inputs must be a backed AnnData;
        raw h5py-backed matrix handles are not supported.
    n_components : int
        Number of singular values/vectors to compute.
    algorithm : str or None
        SVD algorithm: ``"auto"``, ``"irlb"``, or ``"halko"``.
        ``"auto"`` selects based on matrix properties and storage mode.
        Sparse inputs with ``nnz > 2^31 - 1`` are supported directly by IRLB;
        per-axis row/column counts must still fit in ``INT_MAX``.
    max_iter : int
        Maximum iterations for iterative solvers (0 = solver default).
    seed : int
        Random seed.
    verbose : bool
        Print progress messages.
    return_operator_compatible : bool
        If True, return only ``{"u", "d", "v"}`` suitable for
        :func:`.kernel.reduce_kernel_from_svd`.
    backed_chunk_size : int
        Row/element chunk size for backed **read/compute** streaming
        (default ``8192``).
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
    backed_write_chunk_size : int or None
        Row/element chunk size for write-heavy backed preparation, currently
        automatic decompression. ``None`` (default) uses the shared write
        default of ``16384``. Atlas-scale HDF5 rewrites may benefit from
        starting with ``32768``; larger values use proportionally more
        temporary memory.

    Returns
    -------
    dict
        SVD result with keys ``"u"`` (left singular vectors, cells x k),
        ``"d"`` (singular values), ``"v"`` (right singular vectors,
        features x k).

        When ``return_operator_compatible=False``, the result additionally
        carries provenance metadata alongside ``"u"``, ``"d"``, ``"v"`` and
        any raw solver diagnostics returned by the C++ layer:

        - ``svd_algorithm``: integer id of the SVD algorithm actually used
          (``0`` = IRLB, ``1`` = Halko).
        - ``svd_algorithm_name``: human-readable name for the resolved id.
        - ``svd_backend_requested`` / ``svd_backend_resolved``: SVD backend
          slot for future GPU dispatch. Both are currently ``"cpu"``.
    """
    backed_chunk_size, backed_write_chunk_size = resolve_backed_write_chunk_size(
        backed_chunk_size,
        backed_write_chunk_size,
    )

    if backed_n_threads < 0:
        raise ValueError("`backed_n_threads` must be >= 0")

    algorithm_name = _normalize_algorithm(algorithm, context="algorithm")

    is_backed_anndata = isinstance(X, AnnData) and bool(getattr(X, "isbacked", False))
    source_ctx: Optional[MatrixSource] = (
        MatrixSource(X, layer=layer) if isinstance(X, AnnData) else None
    )
    matrix = source_ctx.matrix if source_ctx is not None else X

    if lazy_transform is not None:
        if not is_backed_anndata:
            raise ValueError(
                "`lazy_transform` in `run_svd` requires a backed AnnData input "
                "so row-sum scaling factors can be streamed from the source matrix."
            )
        _validate_lazy_transform(lazy_transform, layer=layer, source=source_ctx)

    if is_backed_anndata:
        adata_ctx: AnnData = X  # type: ignore[assignment]
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source_ctx,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        algorithm_id = _select_svd_algorithm_backed(algorithm_name, verbose)
        io_target_chunk_bytes = _chunk_target_bytes(backed_target_chunk_mb)

        temp_path: Optional[str] = None
        try:
            temp_path = _maybe_decompress_backed_path(
                adata_ctx,
                layer=layer,
                allow_compressed=allow_compressed,
                write_chunk_size=backed_write_chunk_size,
                verbose=verbose,
                context="run_svd",
            )
            file_path = temp_path if temp_path is not None else str(adata_ctx.filename)

            with open_backed_operator_for(
                adata_ctx,
                layer=layer,
                context="run_svd",
                chunk_size=backed_chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
                file_path=file_path,
                io_target_chunk_bytes=io_target_chunk_bytes,
                n_threads=backed_n_threads,
            ) as op:
                result = _core.run_svd_backed_operator(
                    op, n_components, max_iter, seed, algorithm_id, verbose
                )
        finally:
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
    else:
        result = dict(result)
        result.update(
            {
                "svd_algorithm": algorithm_id,
                "svd_algorithm_name": _SVD_ID_TO_ALGORITHM.get(
                    algorithm_id, f"unknown({algorithm_id})"
                ),
                "svd_backend_requested": _SVD_BACKEND_CPU,
                "svd_backend_resolved": _SVD_BACKEND_CPU,
            }
        )

    return result
