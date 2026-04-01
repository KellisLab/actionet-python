"""Backed IO/operator helper functions shared across Python front-end modules."""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from ._backed_compression import (
    format_compression_summary,
    get_storage_metadata_from_adata,
    get_storage_metadata_from_matrix,
    is_compressed_storage,
)
from .preprocessing import decompress_backed_storage


_WARNED_COMPRESSED_BACKED_SVD: set[tuple[str, str, str]] = set()


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
    """Flush backed AnnData before opening a second HDF5 handle."""
    if not bool(getattr(adata, "isbacked", False)):
        return

    file_obj = getattr(getattr(adata, "file", None), "_file", None)
    if file_obj is None:
        return

    try:
        file_obj.flush()
    except Exception as exc:
        raise RuntimeError(
            f"{context}: failed to flush backed AnnData handle before operator read "
            f"({type(exc).__name__}: {exc})"
        )


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


def _run_specificity_backed_sparse(
    adata: AnnData,
    layer: Optional[str],
    chunk_size: int,
    *,
    H: Optional[np.ndarray] = None,
    labels_int: Optional[np.ndarray] = None,
    n_threads: int = 0,
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
) -> dict:
    """Dispatch backed sparse specificity through the C++ ABI."""
    file_path = str(adata.filename)
    group_path = _backed_group_path(layer)
    op = None
    try:
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
        )
        if H is not None:
            return _core.archetype_feature_specificity_backed_operator(op, H, n_threads)
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
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
) -> dict:
    """Dispatch backed dense specificity through the C++ ABI."""
    file_path = str(adata.filename)
    group_path = _backed_group_path(layer)
    op = None
    try:
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
        )
        if H is not None:
            return _core.archetype_feature_specificity_backed_dense_operator(op, H, n_threads)
        return _core.compute_feature_specificity_backed_dense_operator(op, labels_int, n_threads)
    finally:
        op = None
