"""Backed IO/operator helper functions shared across Python front-end modules."""

from __future__ import annotations

import contextlib
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

    from .backed_adapter import BackedAnnDataAdapter
    BackedAnnDataAdapter(adata).flush(context=context)


def _is_lock_open_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(fragment in msg for fragment in _LOCK_OPEN_ERROR_FRAGMENTS)


def _release_backed_handle(adata: Optional[AnnData], *, context: str) -> Optional[str]:
    """Flush and close AnnData's h5py file manager, returning the prior file mode.

    On network filesystems (NFS/GPFS/Lustre) HDF5's in-process reconciliation
    of the h5py handle against a second ``H5Fopen`` request rejects the second
    open even when both handles are read-only and locking is disabled at the
    C++ side. Releasing h5py's handle for the duration of the C++ read removes
    that in-process entry and lets the second open succeed directly.

    Returns the saved ``_filemode`` for a later :func:`_restore_backed_handle`
    call, or ``None`` when no restore is required (adata absent / not backed /
    already closed).
    """
    if adata is None or not bool(getattr(adata, "isbacked", False)):
        return None

    from .backed_adapter import BackedAnnDataAdapter

    _flush_backed_handle(adata, context=context)
    return BackedAnnDataAdapter(adata).release_file_handle()


def _restore_backed_handle(
    adata: Optional[AnnData], saved_mode: Optional[str], *, context: str
) -> None:
    """Re-open AnnData's h5py handle in the mode captured by :func:`_release_backed_handle`.

    Any failure is surfaced as a ``UserWarning`` rather than raised so that a
    successful caller result is not masked by a restore-time error.
    """
    if adata is None or saved_mode is None:
        return

    from .backed_adapter import BackedAnnDataAdapter

    try:
        BackedAnnDataAdapter(adata).restore_file_handle(saved_mode)
    except Exception as exc:
        warnings.warn(
            f"{context}: failed to reopen backed AnnData handle in mode "
            f"'{saved_mode}' ({type(exc).__name__}: {exc})",
            UserWarning,
            stacklevel=3,
        )


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


def _close_backed_operator(op: Any) -> None:
    """Release native operator resources even if its Python wrapper survives."""
    close = getattr(op, "close", None)
    if callable(close):
        # Test doubles and any future directly-exposed concrete operators.
        close()
        return

    from .. import _core

    _core.close_backed_operator(op)


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
    """Open a read-only backed operator for the given HDF5 path.

    Use as a context manager::

        with _open_backed_operator(...) as op:
            result = _core.some_call(op, ...)

    The backing AnnData's h5py file manager (when present) is flushed and
    closed for the duration of the yield, then reopened in its original mode
    on exit. This avoids the in-process HDF5 lock/handle reconciliation that
    rejects a second ``H5Fopen`` on network filesystems (NFS/GPFS/Lustre) even
    when the C++ side disables locking and requests read-only access.

    Callers must not touch ``adata.X``, ``adata.layers``, or any lazy backed
    ``obsm``/``obsp`` datasets inside the ``with`` block; all backed reads
    must go through the yielded C++ operator. In-memory attributes
    (``adata.obs``, ``adata.uns``, dense ``obsm``/``obsp`` values) remain
    accessible because AnnData does not evict them when the file manager is
    closed. The yielded operator is explicitly closed when the block exits
    and must not be reused afterward, even if the caller retains a reference.

    A short retry loop with backoff is preserved to absorb genuinely
    transient lock-open failures (e.g. brief NFS unavailability). No file
    copy or shadow-file fallback is performed: the release-and-reopen
    protocol above is the primary and only remedy for the observed failure
    mode.
    """
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be >= 1")

    saved_mode = _release_backed_handle(adata, context=context)

    op = None
    lock_errors: list[BaseException] = []
    try:
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
                    time.sleep(
                        max(0.0, float(retry_backoff_seconds)) * (attempt + 1)
                    )

        if op is None:
            primary_err = (
                str(lock_errors[-1]) if lock_errors else "unknown open error"
            )
            raise RuntimeError(
                f"{context}: failed to open backed operator for '{file_path}' "
                f"after {retry_attempts} lock-open retries. Last error: "
                f"{primary_err}"
            )

        try:
            yield op
        finally:
            _close_backed_operator(op)
            op = None
    finally:
        _restore_backed_handle(adata, saved_mode, context=context)


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
