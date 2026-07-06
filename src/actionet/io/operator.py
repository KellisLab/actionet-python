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
from anndata import AnnData


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


def _backed_group_path(layer: Optional[str]) -> str:
    return "/X" if layer is None else f"/layers/{layer}"


def _flush_backed_handle(adata: AnnData, *, context: str) -> None:
    """Reopen (if needed) and flush backed AnnData before opening a second HDF5 handle.

    anndata 0.12+ may silently close the backing file (e.g. when
    ``.to_memory()`` is called on a view). This function reopens the
    handle when that happens, then flushes to ensure any pending writes
    are visible to a subsequent independent h5py reader.
    """
    if not bool(getattr(adata, "isbacked", False)):
        return

    file_attr = getattr(adata, "file", None)
    if file_attr is None:
        return

    if not getattr(file_attr, "is_open", False):
        mode = getattr(file_attr, "_filemode", None) or "r+"
        try:
            file_attr.open(filemode=mode)
        except Exception as exc:
            warnings.warn(
                f"{context}: backed AnnData file handle was closed and could not "
                f"be reopened ({type(exc).__name__}: {exc}); "
                "operator may read stale data",
                UserWarning,
                stacklevel=3,
            )
            return

    file_obj = getattr(file_attr, "_file", None)
    if file_obj is None:
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
    from .. import _core

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


@contextlib.contextmanager
def open_backed_operator_for(
    adata: AnnData,
    *,
    layer: Optional[str],
    context: str,
    chunk_size: int,
    lazy_transform: Optional[Any] = None,
    source: Optional[Any] = None,
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
    file_path: Optional[str] = None,
    io_target_chunk_bytes: Optional[int] = None,
    n_threads: Optional[int] = None,
    retry_attempts: int = 3,
    retry_backoff_seconds: float = 0.25,
) -> Generator[Any, None, None]:
    """Open a backed operator for the ``.X`` (or a named layer) of ``adata``.

    Consolidates the common preamble used by every backed-operator call site:

    - resolves the HDF5 group path from ``layer``,
    - optionally resolves lazy-transform parameters from ``lazy_transform``
      (overrides any explicit ``row_scale_factors`` / ``apply_log1p`` / ``log_scale``),
    - opens a lock-safe backed operator via :func:`_open_backed_operator`.

    Parameters
    ----------
    file_path
        Override for the source file path. Defaults to ``str(adata.filename)``.
        Used e.g. by ``reduce_kernel`` / ``run_svd`` when routing through a
        temporary uncompressed copy.
    source
        Pre-built :class:`MatrixSource`. Only consulted when ``lazy_transform``
        is not None; passed through to avoid a redundant reconstruction.

    Yields the backed operator handle for use inside a ``with`` block.
    """
    if lazy_transform is not None:
        from .matrix_source import MatrixSource
        from .lazy_transform import _resolve_lazy_backed_transform

        if source is None:
            source = MatrixSource(adata, layer=layer)
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=chunk_size,
        )

    resolved_file_path = file_path if file_path is not None else str(adata.filename)

    with _open_backed_operator(
        adata=adata,
        file_path=resolved_file_path,
        group_path=_backed_group_path(layer),
        context=context,
        chunk_size=chunk_size,
        row_scale_factors=row_scale_factors,
        apply_log1p=apply_log1p,
        log_scale=log_scale,
        io_target_chunk_bytes=io_target_chunk_bytes,
        n_threads=n_threads,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    ) as op:
        yield op

