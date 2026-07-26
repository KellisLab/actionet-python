"""Structural rewrites of backed AnnData files (subset, materialize).

Public entry points:

- :func:`subset_backed_inplace` — the only safe way to shrink a backed
  AnnData object; atomically rewrites the backing HDF5 file with only the
  selected rows and columns and refreshes the Python handle.
- :func:`materialize_backed` — turn a backed view (created by e.g.
  ``adata[1:1000, :]``) into a proper backed AnnData that is no longer a
  view.

Both are implemented on top of the private :func:`_write_filtered_backed`
helper, which rewrites the HDF5 file with row/col-subsetted chunks. The
sparse-matrix write path (:func:`_write_sparse_subsetted`) also normalizes
mixed ``indices``/``indptr`` dtypes and stores homogeneous index dtypes on
disk so that ``obsp``/``varp``-style pairwise rewrites do not repeatedly
upcast on every chunk.

Companion modules under :mod:`actionet.io` handle annotation persistence
(:mod:`actionet.io.persist`) and periodic checkpointing
(:mod:`actionet.io.checkpoint`).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from time import perf_counter

import anndata as ad
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .chunking import (
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    validate_chunk_size,
)
from .compression import (
    CompressionPolicy,
    get_matrix_compression_policy,
    write_sparse_csr_group_attrs,
)
from .backed_adapter import BackedAnnDataAdapter, backed_view_selection
from .native_h5ad import execute_native_subset, plan_native_subset
from .rewrite import RewriteTransaction
from .persist import (
    _ensure_backed_open,
    _ensure_backed_writable,
    _flush_pending,
    _init_from_reopened,
    _real_layer_keys,
    _refresh_backed_handle,
    is_backed_adata,
)


_INT32_MAX = int(np.iinfo(np.int32).max)
_WriteProfileCallback = Callable[[dict[str, object]], None]


def _emit_write_profile(
    callback: _WriteProfileCallback | None,
    event: str,
    **details: object,
) -> None:
    """Emit one private backed-write profiling event when requested."""
    if callback is not None:
        callback({"event": event, **details})


def _h5_file_size(handle) -> int:
    """Return HDF5's current file-size view, or ``-1`` if unavailable."""
    try:
        return int(handle.id.get_filesize())
    except Exception:
        return -1


def _as_compression_policy(policy) -> CompressionPolicy:
    """Coerce legacy dict / None inputs into a :class:`CompressionPolicy`."""
    if isinstance(policy, CompressionPolicy):
        return policy
    return CompressionPolicy.from_dict(policy)


def _normalize_index_array(
    idx,
    axis_size: int,
    *,
    name: str,
    allow_negative: bool,
) -> np.ndarray:
    """Normalize bool/int selectors to validated int64 indices."""
    arr = np.asarray(idx)
    if arr.dtype == bool:
        arr = arr.ravel()
        if arr.size != axis_size:
            raise ValueError(
                f"Boolean selector for {name} has length {arr.size}, expected {axis_size}"
            )
        return np.flatnonzero(arr).astype(np.int64, copy=False)

    out = arr.astype(np.int64, copy=False).ravel()
    if out.size == 0:
        return out

    if allow_negative:
        out = out.copy()
        neg = out < 0
        if np.any(neg):
            out[neg] += int(axis_size)

    if out.min() < 0 or out.max() >= axis_size:
        raise ValueError(
            f"{name} indices are out of bounds for axis size {axis_size}"
        )
    return out


def _warn_if_duplicates(idx: np.ndarray, *, name: str) -> None:
    """Emit a warning when index arrays contain duplicates."""
    if idx.size == 0:
        return
    n_dupes = int(idx.size - np.unique(idx).size)
    if n_dupes > 0:
        warnings.warn(
            f"{name} indices contain {n_dupes} duplicate(s); "
            "rows/columns will be repeated in the output",
            UserWarning,
            stacklevel=3,
        )


def _is_identity_index(idx: np.ndarray | None, axis_size: int) -> bool:
    """Return whether *idx* selects an axis once, in its original order."""
    return bool(
        idx is not None
        and idx.size == axis_size
        and np.array_equal(idx, np.arange(axis_size, dtype=np.int64))
    )


def _sparse_storage_format(matrix) -> str | None:
    """Return ``csr``/``csc`` for scipy and backed sparse objects."""
    fmt = getattr(matrix, "format", None)
    if isinstance(fmt, str):
        fmt = fmt.lower()
        if fmt in {"csr", "csc"}:
            return fmt

    for node in (matrix, getattr(matrix, "group", None)):
        attrs = getattr(node, "attrs", None)
        if attrs is None:
            continue
        encoding = attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8", errors="ignore")
        if isinstance(encoding, str):
            encoding = encoding.lower()
            if "csr" in encoding:
                return "csr"
            if "csc" in encoding:
                return "csc"
    return None


def _sparse_index_dtype(shape: tuple[int, int], nnz: int) -> np.dtype:
    """Choose one coherent scipy index dtype for a sparse matrix."""
    max_required = max(int(shape[0]), int(shape[1]), int(nnz))
    return np.dtype(np.int32 if max_required <= _INT32_MAX else np.int64)


def _normalize_scipy_sparse_for_row_slicing(matrix):
    """Return a CSR matrix whose ``indices`` and ``indptr`` dtypes agree.

    SciPy fancy row indexing promotes both index arrays to their common
    dtype. A mixed ``indices=int32``/``indptr=int64`` CSR therefore converts
    the *entire* indices array on every row chunk. Harmonizing the two arrays
    once avoids that atlas-scale repeated cost. The returned wrapper does not
    mutate *matrix* and shares data/index arrays whenever their dtype already
    matches the safe target.
    """
    if not sp.issparse(matrix):
        return matrix

    source = matrix if sp.isspmatrix_csr(matrix) else sp.csr_matrix(matrix, copy=False)
    target_dtype = _sparse_index_dtype(source.shape, int(source.nnz))

    if (
        np.dtype(source.indices.dtype) == target_dtype
        and np.dtype(source.indptr.dtype) == target_dtype
    ):
        return source

    indices = np.asarray(source.indices, dtype=target_dtype)
    indptr = np.asarray(source.indptr, dtype=target_dtype)
    normalized = sp.csr_matrix(
        (source.data, indices, indptr),
        shape=source.shape,
        copy=False,
    )

    # Preserve already-computed format flags without forcing an O(nnz) check.
    if hasattr(source, "_has_sorted_indices"):
        normalized._has_sorted_indices = source._has_sorted_indices
    if hasattr(source, "_has_canonical_format"):
        normalized._has_canonical_format = source._has_canonical_format
    return normalized


def _source_sparse_indices_dtype(matrix) -> np.dtype | None:
    """Read the source sparse-index dtype without materializing its values."""
    indices = getattr(matrix, "indices", None)
    if indices is not None and hasattr(indices, "dtype"):
        return np.dtype(indices.dtype)

    group = getattr(matrix, "group", None)
    if group is not None and "indices" in group:
        return np.dtype(group["indices"].dtype)
    return None


def _adaptive_sparse_chunk_size(
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    requested_chunk_size: int,
    *,
    target_block_mb: int = 192,
    overhead_factor: float = 8.0,
    sample_rows: int = 256,
    min_chunk_size: int = 64,
) -> int:
    """Estimate a safer chunk size for backed sparse row operations."""
    req = int(max(1, requested_chunk_size))
    if obs_idx.size == 0:
        return req

    sample_n = int(min(sample_rows, obs_idx.size))
    rows = obs_idx[:sample_n]
    block = matrix[rows, :]

    if var_idx is not None and not _is_identity_index(var_idx, block.shape[1]):
        block = block[:, var_idx]

    block = sp.csr_matrix(block)
    if block.shape[0] == 0 or block.nnz == 0:
        return req

    nnz_per_row = float(block.nnz) / float(block.shape[0])
    bytes_per_nnz = float(block.data.dtype.itemsize + block.indices.dtype.itemsize)
    est_bytes_per_row = max(1.0, nnz_per_row * bytes_per_nnz * float(max(overhead_factor, 1.0)))
    target_bytes = float(max(1, target_block_mb)) * 1024.0 * 1024.0

    safe = int(target_bytes / est_bytes_per_row)
    safe = max(int(max(1, min_chunk_size)), safe)
    return min(req, safe)


def _estimate_total_nnz(matrix, obs_idx: np.ndarray, var_idx: np.ndarray | None) -> int | None:
    """Return exact selected-row nnz from a CSR ``indptr`` without reading data.

    The result depends only on ``obs_idx``; the ``var_idx`` argument is
    retained in the signature for call-site symmetry but is intentionally
    unused, since a CSR ``indptr`` describes row nnz regardless of column
    subsetting. The single caller (``_write_sparse_subsetted``) trusts the
    value as an exact fixed-size allocation only when ``var_idx is None``.
    Returns ``None`` for CSC and other sources that do not expose cheap row
    counts.
    """
    # var_idx is intentionally unused; see docstring.
    _ = var_idx
    import h5py

    if _sparse_storage_format(matrix) != "csr":
        return None

    indptr = None

    group = getattr(matrix, "group", None)
    if group is not None and "indptr" in group:
        indptr_ds = group["indptr"]
        indptr = indptr_ds[:]
    elif isinstance(matrix, h5py.Group) and "indptr" in matrix:
        indptr = matrix["indptr"][:]
    elif sp.issparse(matrix) and hasattr(matrix, "indptr"):
        indptr = np.asarray(matrix.indptr)

    if indptr is None:
        return None

    if obs_idx.size == 0:
        return 0

    row_nnz = np.diff(indptr).astype(np.int64, copy=False)
    total = int(row_nnz[obs_idx].sum())
    return total


def _write_sparse_subsetted(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    encoding: str = "csr_matrix",
    compression_policy: CompressionPolicy | dict | None = None,
    profile_callback: _WriteProfileCallback | None = None,
):
    """Write a row/col-subsetted sparse matrix to *f[h5_key]* in chunks."""
    component_started = perf_counter()
    policy = _as_compression_policy(compression_policy)

    normalize_started = perf_counter()
    source_mat = _normalize_scipy_sparse_for_row_slicing(matrix)
    normalize_s = perf_counter() - normalize_started

    if _is_identity_index(var_idx, source_mat.shape[1]):
        var_idx = None

    n_out = obs_idx.size
    n_vars_out = var_idx.size if var_idx is not None else source_mat.shape[1]

    adaptive_started = perf_counter()
    chunk_size = _adaptive_sparse_chunk_size(
        source_mat,
        obs_idx,
        var_idx,
        chunk_size,
        target_block_mb=128,
        overhead_factor=8.0,
    )
    adaptive_s = perf_counter() - adaptive_started

    estimate_started = perf_counter()
    estimated_nnz = _estimate_total_nnz(source_mat, obs_idx, var_idx)
    estimate_s = perf_counter() - estimate_started

    if profile_callback is not None:
        totals = {
            "source_read_s": 0.0,
            "selection_s": 0.0,
            "conversion_s": 0.0,
            "destination_write_s": 0.0,
            "indptr_update_s": 0.0,
            "resize_s": 0.0,
        }
    else:
        totals = None

    def _read_block(pos: int, end: int):
        rows = obs_idx[pos:end]

        started = perf_counter()
        block = source_mat[rows, :]
        source_read_s = perf_counter() - started

        selection_s = 0.0
        if var_idx is not None:
            started = perf_counter()
            block = block[:, var_idx]
            selection_s = perf_counter() - started

        started = perf_counter()
        block = sp.csr_matrix(block)
        conversion_s = perf_counter() - started
        return block, source_read_s, selection_s, conversion_s

    def _iter_blocks():
        for pos in range(0, n_out, chunk_size):
            end = min(pos + chunk_size, n_out)
            yield pos, end, *_read_block(pos, end)

    blocks = _iter_blocks()
    first_entry = next(blocks, None)
    first_block = first_entry[2] if first_entry is not None else None
    data_dtype = source_mat.dtype if hasattr(source_mat, "dtype") else np.float64
    indices_dtype = _source_sparse_indices_dtype(source_mat) or np.dtype(np.int32)
    if first_block is not None and first_block.nnz > 0:
        data_dtype = first_block.data.dtype
        indices_dtype = np.promote_types(indices_dtype, first_block.indices.dtype)

    if estimated_nnz is not None:
        alloc_size = estimated_nnz
    elif first_block is not None and first_block.nnz > 0:
        avg_nnz_per_row = first_block.nnz / max(1, first_block.shape[0])
        alloc_size = max(1024, int(avg_nnz_per_row * n_out * 1.1))
    else:
        alloc_size = 1024

    use_fixed = estimated_nnz is not None and var_idx is None

    dataset_setup_started = perf_counter()
    grp = f.create_group(h5_key)
    if use_fixed:
        data_ds = grp.create_dataset(
            "data",
            shape=(alloc_size,),
            dtype=data_dtype,
            **policy.sparse_kwargs("data"),
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(alloc_size,),
            dtype=indices_dtype,
            **policy.sparse_kwargs("indices"),
        )
    else:
        data_ds = grp.create_dataset(
            "data",
            shape=(alloc_size,),
            maxshape=(None,),
            dtype=data_dtype,
            **policy.sparse_kwargs("data"),
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(alloc_size,),
            maxshape=(None,),
            dtype=indices_dtype,
            **policy.sparse_kwargs("indices"),
        )
    dataset_setup_s = perf_counter() - dataset_setup_started

    indptr = np.zeros(n_out + 1, dtype=np.int64)
    row_pos = 0
    nnz_pos = 0

    def _ensure_capacity(required: int) -> None:
        if required <= data_ds.shape[0]:
            return
        new_size = max(data_ds.shape[0] * 2, required)
        data_ds.resize((new_size,))
        indices_ds.resize((new_size,))

    def _write_block(
        pos: int,
        end: int,
        block: sp.csr_matrix,
        source_read_s: float,
        selection_s: float,
        conversion_s: float,
    ) -> None:
        nonlocal row_pos, nnz_pos
        n_rows = block.shape[0]
        block_nnz = int(block.nnz)

        started = perf_counter()
        if block_nnz > 0:
            if not use_fixed:
                _ensure_capacity(nnz_pos + block_nnz)
            data_ds[nnz_pos : nnz_pos + block_nnz] = block.data
            indices_ds[nnz_pos : nnz_pos + block_nnz] = block.indices
        destination_write_s = perf_counter() - started

        started = perf_counter()
        row_nnz = np.diff(block.indptr).astype(np.int64, copy=False)
        if n_rows > 0:
            indptr[row_pos + 1 : row_pos + 1 + n_rows] = nnz_pos + np.cumsum(
                row_nnz, dtype=np.int64
            )
        indptr_update_s = perf_counter() - started

        row_pos += n_rows
        nnz_pos += block_nnz

        if profile_callback is not None:
            totals["source_read_s"] += source_read_s
            totals["selection_s"] += selection_s
            totals["conversion_s"] += conversion_s
            totals["destination_write_s"] += destination_write_s
            totals["indptr_update_s"] += indptr_update_s
            _emit_write_profile(
                profile_callback,
                "sparse_chunk",
                component=h5_key,
                row_start=pos,
                row_end=end,
                rows=n_rows,
                nnz=block_nnz,
                source_read_s=source_read_s,
                selection_s=selection_s,
                conversion_s=conversion_s,
                destination_write_s=destination_write_s,
                indptr_update_s=indptr_update_s,
                output_bytes=_h5_file_size(f),
            )

    if first_entry is not None:
        _write_block(*first_entry)
    for entry in blocks:
        _write_block(*entry)

    resize_started = perf_counter()
    if not use_fixed:
        data_ds.resize((nnz_pos,))
        indices_ds.resize((nnz_pos,))
    if totals is not None:
        totals["resize_s"] = perf_counter() - resize_started

    # Keep small outputs coherent on disk so AnnData will not recreate the
    # mixed int32/int64 CSR layout that triggered the repeated SciPy upcast.
    indices_dtype = np.dtype(indices_ds.dtype)
    if (
        indices_dtype == np.dtype(np.int32)
        and max(int(n_out), int(n_vars_out), int(nnz_pos)) <= _INT32_MAX
    ):
        stored_indptr = indptr.astype(np.int32, copy=False)
    else:
        stored_indptr = indptr

    finalize_started = perf_counter()
    grp.create_dataset(
        "indptr",
        data=stored_indptr,
        **policy.sparse_kwargs("indptr"),
    )
    write_sparse_csr_group_attrs(grp, shape=(n_out, n_vars_out), encoding=encoding)
    finalize_s = perf_counter() - finalize_started

    if profile_callback is not None:
        _emit_write_profile(
            profile_callback,
            "sparse_component",
            component=h5_key,
            rows=n_out,
            columns=n_vars_out,
            nnz=nnz_pos,
            chunk_size=chunk_size,
            fixed_allocation=use_fixed,
            normalization_s=normalize_s,
            adaptive_chunk_s=adaptive_s,
            nnz_estimation_s=estimate_s,
            dataset_setup_s=dataset_setup_s,
            finalize_s=finalize_s,
            total_s=perf_counter() - component_started,
            output_bytes=_h5_file_size(f),
            **totals,
        )


def _write_dense_subsetted(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    compression_policy: CompressionPolicy | dict | None = None,
    profile_callback: _WriteProfileCallback | None = None,
):
    """Write a row/col-subsetted dense matrix to *f[h5_key]* in chunks."""
    component_started = perf_counter()
    policy = _as_compression_policy(compression_policy)
    if hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()
    n_out = obs_idx.size
    n_vars_out = var_idx.size if var_idx is not None else matrix.shape[1]
    out_dtype = np.dtype(getattr(matrix, "dtype", np.float64))

    ds = f.create_dataset(
        h5_key,
        shape=(n_out, n_vars_out),
        dtype=out_dtype,
        **policy.dense_kwargs(),
    )
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"

    totals = {
        "source_read_s": 0.0,
        "selection_s": 0.0,
        "conversion_s": 0.0,
        "destination_write_s": 0.0,
    }

    for pos in range(0, n_out, chunk_size):
        end = min(pos + chunk_size, n_out)
        rows = obs_idx[pos:end]

        started = perf_counter()
        block = matrix[rows, :]
        source_read_s = perf_counter() - started

        selection_s = 0.0
        if var_idx is not None:
            started = perf_counter()
            block = block[:, var_idx]
            selection_s = perf_counter() - started

        started = perf_counter()
        if sp.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=out_dtype)
        conversion_s = perf_counter() - started

        started = perf_counter()
        ds[pos:end, :] = block
        destination_write_s = perf_counter() - started
        if profile_callback is not None:
            totals["source_read_s"] += source_read_s
            totals["selection_s"] += selection_s
            totals["conversion_s"] += conversion_s
            totals["destination_write_s"] += destination_write_s
            _emit_write_profile(
                profile_callback,
                "dense_chunk",
                component=h5_key,
                row_start=pos,
                row_end=end,
                rows=end - pos,
                source_read_s=source_read_s,
                selection_s=selection_s,
                conversion_s=conversion_s,
                destination_write_s=destination_write_s,
                output_bytes=_h5_file_size(f),
            )

    if profile_callback is not None:
        _emit_write_profile(
            profile_callback,
            "dense_component",
            component=h5_key,
            rows=n_out,
            columns=n_vars_out,
            chunk_size=chunk_size,
            total_s=perf_counter() - component_started,
            output_bytes=_h5_file_size(f),
            **totals,
        )


def _write_subsetted_matrix(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    compression_policy: CompressionPolicy | dict | None = None,
    profile_callback: _WriteProfileCallback | None = None,
):
    """Dispatch to the legacy Python sparse or dense chunked writer.

    Emits (via ``profile_callback``) exactly one ``sparse_component`` or
    ``dense_component`` event per call, followed by one ``component_flush``
    event that measures the HDF5 flush after the chunked writer completes.

    The native engine treats ``chunk_size`` as a row-count ceiling and also
    enforces its byte-buffer limit. This fallback retains the historical
    coupled read/write stride for rollback compatibility.
    """
    from .matrix_source import _is_sparse_matrix_like

    if _is_identity_index(var_idx, matrix.shape[1]):
        var_idx = None

    if compression_policy is None:
        policy = CompressionPolicy.from_matrix(matrix)
    else:
        policy = _as_compression_policy(compression_policy)

    writer = _write_sparse_subsetted if _is_sparse_matrix_like(matrix) else _write_dense_subsetted

    if profile_callback is None:
        writer(f, h5_key, matrix, obs_idx, var_idx, chunk_size, compression_policy=policy)
        return

    output_bytes_before = _h5_file_size(f)
    write_started = perf_counter()
    writer(
        f,
        h5_key,
        matrix,
        obs_idx,
        var_idx,
        chunk_size,
        compression_policy=policy,
        profile_callback=profile_callback,
    )
    write_s = perf_counter() - write_started

    flush_started = perf_counter()
    f.flush()
    flush_s = perf_counter() - flush_started
    _emit_write_profile(
        profile_callback,
        "component_flush",
        component=h5_key,
        kind="matrix",
        write_s=write_s,
        hdf5_flush_s=flush_s,
        total_s=perf_counter() - write_started,
        output_bytes_before=output_bytes_before,
        output_bytes=_h5_file_size(f),
    )


def _write_filtered_backed(
    adata: AnnData,
    obs_idx: np.ndarray,
    var_idx: np.ndarray,
    dest_path: str,
    chunk_size: int,
    *,
    profile_callback: _WriteProfileCallback | None = None,
) -> None:
    """Write a row/col-subsetted backed AnnData to *dest_path* via h5py.

    Only ``chunk_size`` rows of the expression matrix are in memory at any
    time, so peak RAM is proportional to ``chunk_size * n_vars_filtered``
    rather than the full filtered matrix.

    When ``profile_callback`` is provided, it receives dictionaries for
    matrix chunks, completed components, explicit HDF5 flushes, and file
    close. Profiling is private and opt-in so ordinary rewrites retain their
    existing buffering and flush behavior.
    """
    import h5py
    import pandas as pd

    total_started = perf_counter()
    _flush_pending(adata)
    _ensure_backed_open(adata)

    adapter = BackedAnnDataAdapter(adata)
    obs_sub = adata.obs.iloc[obs_idx].copy()
    var_sub = adata.var.iloc[var_idx].copy()
    h5file = adapter.file_handle
    native_jobs: list[tuple[str, object]] = []

    def _write_or_defer_matrix(
        f,
        h5_key: str,
        matrix,
        row_idx: np.ndarray,
        col_idx: np.ndarray | None,
        *,
        compression_policy=None,
    ) -> None:
        effective_columns = (
            np.arange(matrix.shape[1], dtype=np.int64)
            if col_idx is None
            else col_idx
        )
        location = adapter.matrix_location(matrix, h5_key)
        native_plan = plan_native_subset(location, row_idx, effective_columns)
        if native_plan is not None:
            native_jobs.append((h5_key, native_plan))
            return

        _write_subsetted_matrix(
            f,
            h5_key,
            matrix,
            row_idx,
            col_idx,
            chunk_size,
            compression_policy=compression_policy,
            profile_callback=profile_callback,
        )

    def _write_profiled_component(f, component: str, kind: str, writer):
        if profile_callback is None:
            return writer()

        output_bytes_before = _h5_file_size(f)
        started = perf_counter()
        result = writer()
        write_s = perf_counter() - started
        flush_started = perf_counter()
        f.flush()
        flush_s = perf_counter() - flush_started
        _emit_write_profile(
            profile_callback,
            "component",
            component=component,
            kind=kind,
            write_s=write_s,
            hdf5_flush_s=flush_s,
            total_s=perf_counter() - started,
            output_bytes_before=output_bytes_before,
            output_bytes=_h5_file_size(f),
        )
        return result

    obs_is_identity = _is_identity_index(obs_idx, adata.n_obs)
    var_is_identity = _is_identity_index(var_idx, adata.n_vars)

    def _write_axis_container(
        container,
        row_idx: np.ndarray,
        col_idx: np.ndarray | None,
        is_identity: bool,
        name: str,
    ) -> None:
        """Rewrite a single ``obsm``/``varm``/``obsp``/``varp`` group.

        ``col_idx`` is ``None`` for embeddings (``obsm``/``varm``: row-only
        subsetting) and equals ``row_idx`` for pairwise graphs
        (``obsp``/``varp``: row and column subsetting on the same axis).
        Values that are already an unchanged HDF5 copy of the source get a
        cheap ``h5file.copy`` fast path. DataFrame values are written via
        ``_write_dataframe_to_h5``; every other value type goes through the
        chunked sparse/dense matrix writer.
        """
        keys = list(container.keys())
        if not keys and name not in h5file:
            return

        group = f.create_group(name)
        group.attrs["encoding-type"] = "dict"
        group.attrs["encoding-version"] = "0.1.0"

        for k in keys:
            component = f"{name}/{k}"
            if is_identity and name in h5file and k in h5file[name]:
                _write_profiled_component(
                    f,
                    component,
                    "h5copy",
                    lambda name=name, k=k, group=group: h5file[name].copy(
                        k, group, name=k
                    ),
                )
                continue

            mat = container[k]
            if mat is None:
                continue

            if isinstance(mat, pd.DataFrame):
                mat_sub = mat.iloc[row_idx]
                _write_profiled_component(
                    f,
                    component,
                    "dataframe",
                    lambda component=component, mat_sub=mat_sub: ad.io.write_elem(
                        f, component, mat_sub
                    ),
                )
                continue

            policy = None
            if name in h5file and k in h5file[name]:
                policy = get_matrix_compression_policy(h5file[name][k])
            _write_or_defer_matrix(
                f,
                component,
                mat,
                row_idx,
                col_idx,
                compression_policy=policy,
            )

    with h5py.File(dest_path, "w") as f:
        for key, value in h5file.attrs.items():
            f.attrs[key] = value

        x_policy = get_matrix_compression_policy(h5file["X"]) if "X" in h5file else None
        _write_or_defer_matrix(
            f,
            "X",
            adata.X,
            obs_idx,
            var_idx,
            compression_policy=x_policy,
        )

        _write_profiled_component(
            f, "obs", "dataframe", lambda: ad.io.write_elem(f, "obs", obs_sub)
        )
        _write_profiled_component(
            f, "var", "dataframe", lambda: ad.io.write_elem(f, "var", var_sub)
        )

        layer_keys = _real_layer_keys(adata)
        if layer_keys or "layers" in h5file:
            lg = f.create_group("layers")
            lg.attrs["encoding-type"] = "dict"
            lg.attrs["encoding-version"] = "0.1.0"
            for lk in layer_keys:
                layer_policy = None
                if "layers" in h5file and lk in h5file["layers"]:
                    layer_policy = get_matrix_compression_policy(h5file["layers"][lk])
                _write_or_defer_matrix(
                    f,
                    f"layers/{lk}",
                    adata.layers[lk],
                    obs_idx,
                    var_idx,
                    compression_policy=layer_policy,
                )

        for container, idx, is_identity, name in [
            (adata.obsm, obs_idx, obs_is_identity, "obsm"),
            (adata.varm, var_idx, var_is_identity, "varm"),
        ]:
            _write_axis_container(container, idx, None, is_identity, name)

        for container, idx, is_identity, name in [
            (adata.obsp, obs_idx, obs_is_identity, "obsp"),
            (adata.varp, var_idx, var_is_identity, "varp"),
        ]:
            _write_axis_container(container, idx, idx, is_identity, name)

        if adata.uns:

            _write_profiled_component(
                f,
                "uns",
                "metadata",
                lambda: ad.io.write_elem(f, "uns", dict(adata.uns)),
            )
        elif "uns" in h5file:
            _write_profiled_component(
                f,
                "uns",
                "h5copy",
                lambda: h5file.copy("uns", f, name="uns"),
            )

        raw = getattr(adata, "raw", None)
        if raw is not None:
            raw_grp = f.create_group("raw")
            raw_grp.attrs["encoding-type"] = "raw"
            raw_grp.attrs["encoding-version"] = "0.1.0"

            raw_policy = None
            if "raw" in h5file and "X" in h5file["raw"]:
                raw_policy = get_matrix_compression_policy(h5file["raw"]["X"])
            _write_or_defer_matrix(
                f,
                "raw/X",
                raw.X,
                obs_idx,
                None,
                compression_policy=raw_policy,
            )
            _write_profiled_component(
                f,
                "raw/var",
                "dataframe",
                lambda: ad.io.write_elem(f, "raw/var", raw.var.copy()),
            )

            if "raw" in h5file and "varm" in h5file["raw"]:
                _write_profiled_component(
                    f,
                    "raw/varm",
                    "h5copy",
                    lambda: h5file.copy("raw/varm", raw_grp, name="varm"),
                )

        known_top_level = {
            "X",
            "obs",
            "var",
            "layers",
            "obsm",
            "varm",
            "obsp",
            "varp",
            "uns",
            "raw",
        }
        for top_key in h5file.keys():
            if top_key in known_top_level or top_key in f:
                continue
            _write_profiled_component(
                f,
                top_key,
                "h5copy",
                lambda top_key=top_key: h5file.copy(top_key, f, name=top_key),
            )

        close_started = perf_counter() if profile_callback is not None else 0.0

    close_s = perf_counter() - close_started if profile_callback is not None else 0.0

    for h5_key, native_plan in native_jobs:
        output_bytes_before = os.path.getsize(dest_path)
        transfer_started = perf_counter()
        stats = execute_native_subset(
            native_plan,
            dest_path,
            h5_key,
            max_rows_per_batch=chunk_size,
            collect_span_stats=profile_callback is not None,
        )
        if profile_callback is not None:
            component_event = (
                "dense_component"
                if stats["source"]["encoding"] == "dense"
                else "sparse_component"
            )
            _emit_write_profile(
                profile_callback,
                component_event,
                component=h5_key,
                native=True,
                rows=stats["destination"]["shape"][0],
                columns=stats["destination"]["shape"][1],
                nnz=stats["destination"]["nnz"],
                chunk_size=chunk_size,
                source_read_s=stats["source_read_seconds"],
                selection_s=stats["planning_seconds"],
                conversion_s=stats["packing_seconds"],
                destination_write_s=stats["destination_write_seconds"],
                finalize_s=stats["flush_seconds"],
                total_s=perf_counter() - transfer_started,
                output_bytes=os.path.getsize(dest_path),
            )
            _emit_write_profile(
                profile_callback,
                "component_flush",
                component=h5_key,
                kind="native_matrix",
                write_s=stats["destination_write_seconds"],
                hdf5_flush_s=stats["flush_seconds"],
                total_s=perf_counter() - transfer_started,
                output_bytes_before=output_bytes_before,
                output_bytes=os.path.getsize(dest_path),
            )
            _emit_write_profile(
                profile_callback,
                "native_matrix_component",
                component=h5_key,
                total_s=perf_counter() - transfer_started,
                output_bytes_before=output_bytes_before,
                output_bytes=os.path.getsize(dest_path),
                **stats,
            )

    # AnnData performs the final whole-file structural check. Keep the object
    # backed so validation never materializes the numeric payloads.
    validated = ad.read_h5ad(dest_path, backed="r")
    if getattr(validated, "file", None) is not None:
        validated.file.close()

    if profile_callback is not None:
        output_bytes = os.path.getsize(dest_path)
        _emit_write_profile(
            profile_callback,
            "close",
            component=dest_path,
            close_s=close_s,
            output_bytes=output_bytes,
        )
        _emit_write_profile(
            profile_callback,
            "filtered_write",
            component=dest_path,
            total_s=perf_counter() - total_started,
            output_bytes=output_bytes,
        )


def _atomic_filtered_rewrite(
    adata: AnnData,
    obs_idx: np.ndarray,
    var_idx: np.ndarray,
    destination_path: str,
    chunk_size: int,
    *,
    refresh_source: bool,
    profile_callback: _WriteProfileCallback | None = None,
) -> None:
    """Write, validate, fsync, and atomically publish a filtered H5AD."""
    _flush_pending(adata)
    _ensure_backed_open(adata)
    adapter = BackedAnnDataAdapter(adata)
    source_path = adapter.filename
    destination_path = os.path.realpath(os.fspath(destination_path))
    in_place = source_path == destination_path
    original_mode = adapter.mode

    with RewriteTransaction(source_path, destination_path) as transaction:
        _write_filtered_backed(
            adata,
            obs_idx,
            var_idx,
            transaction.temp_path,
            chunk_size,
            profile_callback=profile_callback,
        )
        commit_stats = transaction.commit(
            close_source=adapter.close if in_place else None,
            restore_source=(
                lambda: adapter.reopen(mode=original_mode)
                if in_place
                else None
            ),
        )
        _emit_write_profile(
            profile_callback,
            "transaction_commit",
            component=destination_path,
            temp_fsync_s=commit_stats.temp_fsync_seconds,
            fingerprint_s=commit_stats.fingerprint_seconds,
            source_close_s=commit_stats.source_close_seconds,
            replace_s=commit_stats.replace_seconds,
            parent_fsync_s=commit_stats.parent_fsync_seconds,
            total_s=commit_stats.total_seconds,
        )

    if refresh_source:
        adapter.reopen(mode=original_mode)


def materialize_backed(
    adata: AnnData,
    filename: str | os.PathLike | None = None,
    *,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
) -> None:
    """Materialize a backed AnnData view into a standalone backed object.

    Turns a backed view (created by e.g. ``adata[1:1000, :]``) into a
    proper backed AnnData (no longer a view).

    If *adata* is already a non-view backed object this is a no-op.

    Parameters
    ----------
    adata : AnnData
        A backed AnnData view (``adata.is_view`` and ``adata.isbacked``).
    filename : path-like or None
        Destination HDF5 path.  When ``None`` (default), the parent backing
        file is atomically rewritten in place.
    backed_write_chunk_size : int, optional (default: 16384)
        Maximum rows per native transfer batch. The native byte-buffer limit
        may reduce the effective batch size. The rollback Python engine uses
        this as its historical read/write stride.

    Raises
    ------
    ValueError
        If *adata* is not backed.
    """
    backed_write_chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )
    if not is_backed_adata(adata):
        raise ValueError(
            "materialize_backed requires a backed AnnData object."
        )
    if not getattr(adata, "is_view", False):
        return

    parent, obs_int, var_int = backed_view_selection(adata)
    _flush_pending(parent)
    _ensure_backed_open(parent)

    parent_path = str(parent.filename)
    dest_path = str(filename) if filename is not None else parent_path
    in_place_parent = (os.path.realpath(dest_path) == os.path.realpath(parent_path))
    if in_place_parent:
        _ensure_backed_writable(parent)

    try:
        _atomic_filtered_rewrite(
            parent,
            obs_int,
            var_int,
            dest_path,
            backed_write_chunk_size,
            refresh_source=in_place_parent,
        )
    except Exception:
        if in_place_parent:
            try:
                _refresh_backed_handle(parent, parent_path, mode="r+")
            except Exception:
                pass
        raise

    if in_place_parent and adata is parent:
        return

    reopened = ad.read_h5ad(dest_path, backed="r+")
    _init_from_reopened(adata, reopened)


def subset_backed_inplace(
    adata: AnnData,
    obs_idx: np.ndarray | None = None,
    var_idx: np.ndarray | None = None,
    *,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
) -> None:
    """Subset a backed AnnData in-place by rewriting the backing file.

    This is the only safe way to shrink the dimensions of a backed AnnData
    object.  The backing HDF5 is atomically rewritten with only the
    selected rows/columns, and the Python handle is refreshed so that
    in-memory metadata and the on-disk file agree on shape.

    Parameters
    ----------
    adata : AnnData
        A backed AnnData opened in ``r+`` mode.
    obs_idx : ndarray of int64 or None
        Row (cell) indices to keep.  ``None`` keeps all rows.
    var_idx : ndarray of int64 or None
        Column (feature) indices to keep.  ``None`` keeps all columns.
    backed_write_chunk_size : int, optional (default: 16384)
        Maximum rows per native transfer batch. The native byte-buffer limit
        may reduce the effective batch size. The rollback Python engine uses
        this as its historical read/write stride.

    Raises
    ------
    ValueError
        If *adata* is not backed or is read-only.
    """
    backed_write_chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )
    if not is_backed_adata(adata):
        raise ValueError(
            "subset_backed_inplace requires a backed AnnData object. "
            "Open with ad.read_h5ad(path, backed='r+')."
        )
    _ensure_backed_open(adata)
    _ensure_backed_writable(adata)
    _flush_pending(adata)

    if getattr(adata, "is_view", False):
        materialize_backed(adata, backed_write_chunk_size=backed_write_chunk_size)
        if obs_idx is None and var_idx is None:
            return

    if obs_idx is None:
        obs_idx = np.arange(adata.n_obs, dtype=np.int64)
    else:
        obs_idx = _normalize_index_array(
            obs_idx,
            adata.n_obs,
            name="obs",
            allow_negative=False,
        )
        _warn_if_duplicates(obs_idx, name="obs")

    if var_idx is None:
        var_idx = np.arange(adata.n_vars, dtype=np.int64)
    else:
        var_idx = _normalize_index_array(
            var_idx,
            adata.n_vars,
            name="var",
            allow_negative=False,
        )
        _warn_if_duplicates(var_idx, name="var")

    if obs_idx.size == 0:
        raise ValueError("obs_idx selects zero observations; empty AnnData is not supported")
    if var_idx.size == 0:
        raise ValueError("var_idx selects zero variables; empty AnnData is not supported")

    if obs_idx.size == adata.n_obs and var_idx.size == adata.n_vars:
        if np.array_equal(obs_idx, np.arange(adata.n_obs, dtype=np.int64)) and \
           np.array_equal(var_idx, np.arange(adata.n_vars, dtype=np.int64)):
            return

    filepath = str(adata.filename)
    _atomic_filtered_rewrite(
        adata,
        obs_idx,
        var_idx,
        filepath,
        backed_write_chunk_size,
        refresh_source=True,
    )
