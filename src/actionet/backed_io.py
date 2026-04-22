"""Backed IO/operator helper functions shared across Python front-end modules."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
import warnings
from typing import Any, Generator, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._backed_compression import (
    format_compression_summary,
    get_storage_metadata_from_adata,
    get_storage_metadata_from_matrix,
    is_compressed_storage,
)
from .preprocessing import decompress_backed_storage


_WARNED_COMPRESSED_BACKED_SVD: set[tuple[str, str, str]] = set()
_LOCK_OPEN_ERROR_FRAGMENTS = (
    "createbackedoperator",
    "failed to open h5ad file",
    "resource temporarily unavailable",
    "errno = 11",
    "errno=11",
    "eagain",
    "unable to lock file",
    "file locking disabled",
    "file locking failed",
)


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

    file_attr = getattr(adata, "file", None)
    file_obj = getattr(file_attr, "_file", None)
    if file_obj is None:
        if file_attr is not None:
            warnings.warn(
                f"{context}: backed AnnData file handle appears closed; "
                "operator may read stale data",
                UserWarning,
                stacklevel=3,
            )
        return

    try:
        file_obj.flush()
    except Exception as exc:
        raise RuntimeError(
            f"{context}: failed to flush backed AnnData handle before operator read "
            f"({type(exc).__name__}: {exc})"
        )


def _is_lock_open_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(fragment in msg for fragment in _LOCK_OPEN_ERROR_FRAGMENTS)


def _create_backed_operator(
    *,
    file_path: str,
    group_path: str,
    chunk_size: int,
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
    io_target_chunk_bytes: Optional[int] = None,
    n_threads: Optional[int] = None,
):
    from . import _core

    kwargs = {
        "file_path": file_path,
        "group_path": group_path,
        "chunk_size": int(chunk_size),
        "apply_log1p": bool(apply_log1p),
        "log_scale": float(log_scale),
    }
    if row_scale_factors is not None:
        kwargs["row_scale_factors"] = row_scale_factors
    if io_target_chunk_bytes is not None:
        kwargs["io_target_chunk_bytes"] = int(io_target_chunk_bytes)
    if n_threads is not None:
        kwargs["n_threads"] = int(n_threads)

    return _core.create_backed_operator(**kwargs)


@contextlib.contextmanager
def _open_backed_operator(
    *,
    adata: Optional[AnnData],
    file_path: str,
    group_path: str,
    context: str,
    chunk_size: int,
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
    io_target_chunk_bytes: Optional[int] = None,
    n_threads: Optional[int] = None,
    retry_attempts: int = 3,
    retry_backoff_seconds: float = 0.25,
) -> Generator[Any, None, None]:
    """Open a lock-safe backed operator with retry and temp-copy fallback.

    Use as a context manager::

        with _open_backed_operator(...) as op:
            result = _core.some_call(op, ...)
    """
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be >= 1")

    if adata is not None:
        _flush_backed_handle(adata, context=context)

    lock_errors: list[BaseException] = []
    op = None

    for attempt in range(retry_attempts):
        try:
            op = _create_backed_operator(
                file_path=file_path,
                group_path=group_path,
                chunk_size=chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
                io_target_chunk_bytes=io_target_chunk_bytes,
                n_threads=n_threads,
            )
            break
        except Exception as exc:
            if not _is_lock_open_error(exc):
                raise
            lock_errors.append(exc)
            if attempt + 1 < retry_attempts:
                time.sleep(max(0.0, float(retry_backoff_seconds)) * (attempt + 1))

    fallback_path: Optional[str] = None

    if op is None:
        warnings.warn(
            f"{context}: retries exhausted for '{file_path}'; "
            f"copying to temporary file for lock-free access "
            f"(this may be slow for large files on network storage)",
            UserWarning,
            stacklevel=3,
        )
        parent = os.path.dirname(file_path) or "."
        fd, fallback_path = tempfile.mkstemp(
            prefix="actionet_lock_fallback_",
            suffix=".h5ad",
            dir=parent,
        )
        os.close(fd)

        try:
            shutil.copy2(file_path, fallback_path)
            op = _create_backed_operator(
                file_path=fallback_path,
                group_path=group_path,
                chunk_size=chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
                io_target_chunk_bytes=io_target_chunk_bytes,
                n_threads=n_threads,
            )
        except Exception as exc:
            try:
                if fallback_path is not None and os.path.exists(fallback_path):
                    os.remove(fallback_path)
            except OSError:
                pass

            if not _is_lock_open_error(exc):
                raise

            primary_err = str(lock_errors[-1]) if lock_errors else "unknown primary open error"
            raise RuntimeError(
                f"{context}: failed to open backed operator for '{file_path}' "
                f"after {retry_attempts} retries and fallback copy '{fallback_path}'. "
                f"Primary error: {primary_err}. Fallback error: {exc}"
            ) from exc

    try:
        yield op
    finally:
        op = None
        if fallback_path is not None and os.path.exists(fallback_path):
            try:
                os.remove(fallback_path)
            except OSError as exc:
                warnings.warn(
                    f"{context}: failed to remove fallback operator copy '{fallback_path}' "
                    f"({type(exc).__name__}: {exc})",
                    UserWarning,
                    stacklevel=4,
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
