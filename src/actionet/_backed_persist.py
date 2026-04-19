"""Backed AnnData persistence helpers.

Centralises three concerns:

1. **In-memory updates** -- every API function stores its results on the
   AnnData object so they are immediately visible to the caller.
2. **Disk persistence** -- when the AnnData is in backed mode (HDF5), the
   same results are written to the underlying file via the experimental
   ``_anndata_io.append_to_anndata`` writer so that closing and re-opening
   the file preserves the computed annotations.
3. **Structural rewrites** -- subsetting (obs/var) of backed AnnData
   requires physically rewriting the HDF5 file.  The helpers that
   perform chunked subsetting and atomic file replacement live here so
   that every call-site that changes object shape uses the same safe
   primitive.

The typical call site is simply::

    persist_updates(adata, obsm={"key": array}, uns={"key": value})

which transparently does both steps.

For a full flush of all in-memory state, use :func:`checkpoint_backed`.
For safe subsetting, use :func:`subset_backed_inplace`.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any, Mapping, MutableMapping

import anndata as ad
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._backed_compression import get_matrix_compression_policy

try:
    from .experimental import _anndata_io
except Exception:  # pragma: no cover - optional in some build contexts
    _anndata_io = None


# ---------------------------------------------------------------------------
# Dirty-key tracking: records which annotation keys have been written since
# the last checkpoint, so checkpoint_backed only rewrites what changed.
# ---------------------------------------------------------------------------

import weakref
from collections import defaultdict


class _DirtyTracker:
    """Per-object tracker of annotation keys modified since last flush."""

    def __init__(self):
        self._dirty: dict[int, dict[str, set[str]]] = {}

    def mark(self, adata: AnnData, slot: str, keys: set[str]) -> None:
        """Record that *keys* within *slot* have been modified."""
        obj_id = id(adata)
        if obj_id not in self._dirty:
            self._dirty[obj_id] = defaultdict(set)
            weakref.finalize(adata, self._dirty.pop, obj_id, None)
        self._dirty[obj_id][slot].update(keys)

    def get_dirty(self, adata: AnnData) -> dict[str, set[str]]:
        """Return dirty slots/keys for *adata*, or empty dict if clean."""
        return self._dirty.get(id(adata), {})

    def clear(self, adata: AnnData) -> None:
        """Mark *adata* as fully flushed."""
        self._dirty.pop(id(adata), None)


_dirty_tracker = _DirtyTracker()


def _track_dirty_keys(
    adata: AnnData,
    obs: dict, var: dict,
    obsm: dict, varm: dict,
    obsp: dict, varp: dict,
    layers: dict, uns: dict,
) -> None:
    """Record which slots/keys were written, so checkpoint can skip them."""
    if obs:
        _dirty_tracker.mark(adata, "obs_columns", set(obs.keys()))
    if var:
        _dirty_tracker.mark(adata, "var_columns", set(var.keys()))
    if obsm:
        _dirty_tracker.mark(adata, "obsm_keys", set(obsm.keys()))
    if varm:
        _dirty_tracker.mark(adata, "varm_keys", set(varm.keys()))
    if obsp:
        _dirty_tracker.mark(adata, "obsp_keys", set(obsp.keys()))
    if varp:
        _dirty_tracker.mark(adata, "varp_keys", set(varp.keys()))
    if layers:
        _dirty_tracker.mark(adata, "layers_keys", set(layers.keys()))
    if uns:
        _dirty_tracker.mark(adata, "uns_keys", set(uns.keys()))


def is_backed_adata(adata: AnnData) -> bool:
    """Return True when AnnData is backed and has a filename."""
    return bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None))


def _ensure_backed_writable(adata: AnnData) -> None:
    """Raise if backed AnnData appears to be read-only."""
    if not is_backed_adata(adata):
        return

    file_handle = getattr(getattr(adata, "file", None), "_file", None)
    mode = getattr(file_handle, "mode", None)
    if mode == "r":
        raise ValueError(
            "Backed AnnData was opened read-only (mode='r'). "
            "Re-open with backed='r+' to persist updates."
        )


def _as_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if values is None else dict(values)


def _assign_mapping(
    target: MutableMapping[str, Any],
    values: Mapping[str, Any],
    *,
    tolerate_errors: bool,
) -> None:
    for key, value in values.items():
        try:
            target[key] = value
        except Exception:
            if not tolerate_errors:
                raise


def apply_inmemory_updates(
    adata: AnnData,
    *,
    obs: Mapping[str, Any] | None = None,
    var: Mapping[str, Any] | None = None,
    obsm: Mapping[str, Any] | None = None,
    varm: Mapping[str, Any] | None = None,
    obsp: Mapping[str, Any] | None = None,
    varp: Mapping[str, Any] | None = None,
    layers: Mapping[str, Any] | None = None,
    uns: Mapping[str, Any] | None = None,
    tolerate_errors: bool | None = None,
) -> None:
    """Assign values to the in-memory AnnData object only (no disk write).

    This is the first half of the persist workflow.  Use
    :func:`persist_updates` instead when the caller also needs disk
    persistence for backed objects.

    Parameters
    ----------
    adata : AnnData
        Target object.
    obs, var : dict, optional
        Column name -> 1-D array-like mappings for ``adata.obs`` / ``adata.var``.
    obsm, varm, obsp, varp, layers, uns : dict, optional
        Key -> value mappings for the corresponding AnnData slots.
    tolerate_errors : bool or None
        If ``True``, silently skip assignments that raise.  ``None``
        (default) auto-enables tolerance for backed AnnData where some
        assignments may be unsupported.
    """
    if tolerate_errors is None:
        tolerate_errors = is_backed_adata(adata)

    obs = _as_mapping(obs)
    var = _as_mapping(var)
    obsm = _as_mapping(obsm)
    varm = _as_mapping(varm)
    obsp = _as_mapping(obsp)
    varp = _as_mapping(varp)
    layers = _as_mapping(layers)
    uns = _as_mapping(uns)

    for key, value in obs.items():
        try:
            adata.obs[key] = value
        except Exception:
            if not tolerate_errors:
                raise

    for key, value in var.items():
        try:
            adata.var[key] = value
        except Exception:
            if not tolerate_errors:
                raise

    _assign_mapping(adata.obsm, obsm, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.varm, varm, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.obsp, obsp, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.varp, varp, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.layers, layers, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.uns, uns, tolerate_errors=tolerate_errors)


def persist_updates(
    adata: AnnData,
    *,
    obs: Mapping[str, Any] | None = None,
    var: Mapping[str, Any] | None = None,
    obsm: Mapping[str, Any] | None = None,
    varm: Mapping[str, Any] | None = None,
    obsp: Mapping[str, Any] | None = None,
    varp: Mapping[str, Any] | None = None,
    layers: Mapping[str, Any] | None = None,
    uns: Mapping[str, Any] | None = None,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """Apply updates in-memory and, for backed AnnData, write them to disk.

    This is the primary entry point used by all public API functions to
    store computed results.  For in-memory objects it is equivalent to
    :func:`apply_inmemory_updates`.  For backed objects it additionally
    calls ``_anndata_io.append_to_anndata`` to persist results in the
    underlying HDF5 file.

    Parameters
    ----------
    adata : AnnData
        Target object.
    obs, var, obsm, varm, obsp, varp, layers, uns : dict, optional
        Mappings of keys to values for each AnnData slot.
    validate : bool
        Run ``_anndata_io`` validation before writing (backed only).
    verbose : bool
        Print progress messages during disk writes (backed only).
    """
    obs = _as_mapping(obs)
    var = _as_mapping(var)
    obsm = _as_mapping(obsm)
    varm = _as_mapping(varm)
    obsp = _as_mapping(obsp)
    varp = _as_mapping(varp)
    layers = _as_mapping(layers)
    uns = _as_mapping(uns)

    apply_inmemory_updates(
        adata,
        obs=obs,
        var=var,
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        varp=varp,
        layers=layers,
        uns=uns,
        tolerate_errors=is_backed_adata(adata),
    )

    if not is_backed_adata(adata):
        return

    if _anndata_io is None:
        raise RuntimeError(
            "Backed persistence requested but actionet.experimental._anndata_io "
            "could not be imported."
        )

    _ensure_backed_writable(adata)

    results = {
        "obs_columns": obs,
        "var_columns": var,
        "obsm_keys": obsm,
        "varm_keys": varm,
        "obsp_keys": obsp,
        "varp_keys": varp,
        "layers_keys": layers,
        "uns_keys": uns,
    }

    if not any(len(v) > 0 for v in results.values()):
        return

    # Record which keys are being persisted for dirty tracking.
    _track_dirty_keys(adata, obs, var, obsm, varm, obsp, varp, layers, uns)

    filepath = str(adata.filename)

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    _anndata_io.append_to_anndata(
        filepath,
        results,
        verbose=verbose,
        validate=validate,
    )

    _refresh_backed_handle(adata, filepath, mode="r+")


def persist_layer(
    adata: AnnData,
    layer: str,
    matrix: Any,
    *,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """Convenience wrapper: persist a single layer matrix to a backed file."""
    persist_updates(
        adata,
        layers={layer: matrix},
        validate=validate,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Checkpoint / compact
# ---------------------------------------------------------------------------


def _copy_h5_group_faithful(src_group, dst_group, chunk_size: int) -> None:
    """Recursively copy an HDF5 group preserving compression settings."""
    import h5py

    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            _copy_h5_group_faithful(obj, child, chunk_size=chunk_size)
        elif isinstance(obj, h5py.Dataset):
            kwargs: dict[str, Any] = {
                "shape": obj.shape,
                "dtype": obj.dtype,
            }
            if obj.chunks is not None:
                kwargs["chunks"] = obj.chunks
            if obj.maxshape is not None:
                kwargs["maxshape"] = obj.maxshape
            if obj.compression is not None:
                kwargs["compression"] = obj.compression
            if obj.compression_opts is not None:
                kwargs["compression_opts"] = obj.compression_opts
            if obj.shuffle:
                kwargs["shuffle"] = True
            if obj.fletcher32:
                kwargs["fletcher32"] = True

            dst_ds = dst_group.create_dataset(name, **kwargs)

            # Chunked copy along axis-0.
            if obj.shape == () or obj.ndim == 0:
                dst_ds[()] = obj[()]
            else:
                n_rows = obj.shape[0]
                step = max(1, chunk_size)
                for start in range(0, n_rows, step):
                    end = min(start + step, n_rows)
                    dst_ds[start:end, ...] = obj[start:end, ...]

            for key, value in obj.attrs.items():
                dst_ds.attrs[key] = value
        else:
            raise TypeError(
                f"Unsupported HDF5 object type for key '{name}': {type(obj)}"
            )


def _repack_h5ad(
    adata: AnnData,
    *,
    chunk_size: int = 4096,
    verbose: bool = False,
) -> None:
    """Repack a backed H5AD file to reclaim dead space, then refresh the handle.

    Performs an atomic copy to a temp file, replaces the original, and
    re-opens the AnnData handle so it points at the compacted file.
    """
    import h5py

    src_path = str(adata.filename)

    parent_dir = os.path.dirname(src_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".h5ad", dir=parent_dir, prefix=".compact_"
    )
    os.close(fd)

    try:
        with h5py.File(src_path, "r") as src_f, h5py.File(tmp_path, "w") as dst_f:
            _copy_h5_group_faithful(src_f, dst_f, chunk_size=chunk_size)

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        os.replace(tmp_path, src_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    import anndata as ad

    reopened = ad.read_h5ad(src_path, backed="r+")
    _init_from_reopened(adata, reopened)

    if verbose:
        print(f"[INFO] Compacted {src_path}")


def _checkpoint_collect_args(adata: AnnData) -> dict:
    """Build collect_annotation_results kwargs using only dirty keys.

    If nothing is dirty (e.g. user calls checkpoint without prior persist_updates),
    falls back to collecting all annotation keys for a full flush.
    """
    dirty = _dirty_tracker.get_dirty(adata)

    if not dirty:
        return {
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "obsm_keys": list(adata.obsm.keys()),
            "varm_keys": list(adata.varm.keys()),
            "obsp_keys": list(adata.obsp.keys()),
            "varp_keys": list(adata.varp.keys()),
            "layers_keys": [],
            "uns_keys": list(adata.uns.keys()),
        }

    return {
        "obs_columns": sorted(dirty.get("obs_columns", set())),
        "var_columns": sorted(dirty.get("var_columns", set())),
        "obsm_keys": sorted(dirty.get("obsm_keys", set())),
        "varm_keys": sorted(dirty.get("varm_keys", set())),
        "obsp_keys": sorted(dirty.get("obsp_keys", set())),
        "varp_keys": sorted(dirty.get("varp_keys", set())),
        "layers_keys": sorted(dirty.get("layers_keys", set())),
        "uns_keys": sorted(dirty.get("uns_keys", set())),
    }


def checkpoint_backed(
    adata: AnnData,
    *,
    compact: bool = False,
    chunk_size: int = 4096,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """Flush all in-memory annotations to the backing HDF5 file.

    This is the recommended way to checkpoint a backed AnnData object.
    Unlike ``adata.write_h5ad()``, which rewrites the full object and
    nearly doubles file size due to HDF5 dead-space accumulation,
    ``checkpoint_backed`` writes only in-memory annotation slots and
    optionally repacks the file to reclaim any dead space.

    Parameters
    ----------
    adata : AnnData
        A backed AnnData object opened in ``r+`` mode.
    compact : bool, optional (default: False)
        If ``True``, repack the HDF5 file after writing to reclaim
        dead space from prior delete-then-create overwrites.  This
        requires a full file copy and is expensive for large files.
    chunk_size : int, optional (default: 4096)
        Row-chunk size used during the compact file copy.
    validate : bool, optional (default: False)
        Run ``_anndata_io`` validation before writing.
    verbose : bool, optional (default: False)
        Print progress messages.

    Raises
    ------
    ValueError
        If *adata* is not backed or is opened read-only.
    RuntimeError
        If the experimental IO module is unavailable.
    """
    if not is_backed_adata(adata):
        raise ValueError(
            "checkpoint_backed requires a backed AnnData object. "
            "Open with ad.read_h5ad(path, backed='r+')."
        )

    _ensure_backed_writable(adata)

    if _anndata_io is None:
        raise RuntimeError(
            "Backed persistence requested but actionet.experimental._anndata_io "
            "could not be imported."
        )

    results = _anndata_io.collect_annotation_results(
        adata,
        **_checkpoint_collect_args(adata),
        verbose=verbose,
    )

    has_data = any(len(v) > 0 for v in results.values())

    if has_data:
        if verbose:
            print(f"[INFO] Checkpointing to {adata.filename}")

        filepath = str(adata.filename)

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        _anndata_io.append_to_anndata(
            filepath,
            results,
            verbose=verbose,
            validate=validate,
        )

        _refresh_backed_handle(adata, filepath, mode="r+")

    _dirty_tracker.clear(adata)

    if compact:
        _repack_h5ad(adata, chunk_size=chunk_size, verbose=verbose)


# ---------------------------------------------------------------------------
# Backed subsetting infrastructure (relocated from preprocessing.py)
# ---------------------------------------------------------------------------


def _refresh_backed_handle(adata: AnnData, path: str, mode: str = "r+") -> None:
    """Close and re-open a backed AnnData handle in-place."""
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    reopened = ad.read_h5ad(path, backed=mode)
    _init_from_reopened(adata, reopened)


def _init_from_reopened(adata: AnnData, reopened: AnnData) -> None:
    """Reinitialize *adata* from *reopened*, handling backed-raw edge cases."""
    raw_obj = getattr(reopened, "raw", None)
    if raw_obj is not None and getattr(raw_obj, "_X", None) is None:
        try:
            raw_obj._X = raw_obj.X
        except Exception:
            pass
    adata._init_as_actual(reopened)
    adata.file = reopened.file


def _dataset_create_kwargs_from_spec(spec: dict | None) -> dict:
    """Translate compression metadata into h5py create_dataset kwargs."""
    if not spec:
        return {}

    kwargs: dict = {}
    codec = spec.get("compression")
    if codec is not None:
        kwargs["compression"] = codec
        if spec.get("compression_opts") is not None:
            kwargs["compression_opts"] = spec["compression_opts"]
    return kwargs


def _dense_compression_kwargs(compression_policy: dict | None) -> dict:
    """Return compression kwargs for dense datasets."""
    if not compression_policy:
        return {}
    datasets = compression_policy.get("datasets", {})
    if not datasets:
        return {}
    first_spec = next(iter(datasets.values()))
    return _dataset_create_kwargs_from_spec(first_spec)


def _sparse_dataset_compression_kwargs(
    compression_policy: dict | None,
    dataset_name: str,
) -> dict:
    """Return compression kwargs for one sparse component dataset."""
    if not compression_policy:
        return {}
    spec = (compression_policy.get("datasets", {}) or {}).get(dataset_name)
    return _dataset_create_kwargs_from_spec(spec)


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

    if var_idx is not None:
        n_vars = block.shape[1]
        is_full_var = (
            var_idx.size == n_vars
            and np.array_equal(var_idx, np.arange(n_vars, dtype=np.int64))
        )
        if not is_full_var:
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
    """Try to compute exact output nnz from source indptr without reading data.

    Returns None if the source format doesn't support cheap nnz estimation
    (e.g. CSC with row subsetting, or non-sparse backed objects).
    """
    import h5py

    indptr = None

    # Backed sparse CSR: the indptr is accessible via the HDF5 group.
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

    # For CSR, indptr gives nnz per row directly. If we also subset columns,
    # the exact nnz can only be known after reading data, so return an upper
    # bound (the row-selected nnz before column filtering).
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
    compression_policy: dict | None = None,
):
    """Write a row/col-subsetted sparse matrix to *f[h5_key]* in chunks."""
    source_mat = matrix
    n_out = obs_idx.size
    n_vars_out = var_idx.size if var_idx is not None else source_mat.shape[1]
    chunk_size = _adaptive_sparse_chunk_size(
        source_mat,
        obs_idx,
        var_idx,
        chunk_size,
        target_block_mb=128,
        overhead_factor=8.0,
    )

    estimated_nnz = _estimate_total_nnz(source_mat, obs_idx, var_idx)

    def _iter_blocks():
        for pos in range(0, n_out, chunk_size):
            end = min(pos + chunk_size, n_out)
            rows = obs_idx[pos:end]
            block = source_mat[rows, :]
            if var_idx is not None:
                block = block[:, var_idx]
            yield sp.csr_matrix(block)

    blocks = _iter_blocks()
    first_block = next(blocks, None)
    data_dtype = source_mat.dtype if hasattr(source_mat, "dtype") else np.float64
    indices_dtype = np.int32
    if first_block is not None and first_block.nnz > 0:
        data_dtype = first_block.data.dtype
        indices_dtype = first_block.indices.dtype

    # Pre-allocate to estimated size when available; otherwise use a generous
    # initial capacity to minimize resize calls.
    if estimated_nnz is not None and estimated_nnz > 0:
        alloc_size = estimated_nnz
    elif first_block is not None and first_block.nnz > 0:
        avg_nnz_per_row = first_block.nnz / max(1, first_block.shape[0])
        alloc_size = max(1024, int(avg_nnz_per_row * n_out * 1.1))
    else:
        alloc_size = 1024

    # When we have exact pre-allocation (no var subsetting), use fixed-size
    # datasets which are faster than resizable ones.
    use_fixed = (estimated_nnz is not None and var_idx is None)

    grp = f.create_group(h5_key)
    if use_fixed:
        data_ds = grp.create_dataset(
            "data",
            shape=(alloc_size,),
            dtype=data_dtype,
            **_sparse_dataset_compression_kwargs(compression_policy, "data"),
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(alloc_size,),
            dtype=indices_dtype,
            **_sparse_dataset_compression_kwargs(compression_policy, "indices"),
        )
    else:
        data_ds = grp.create_dataset(
            "data",
            shape=(alloc_size,),
            maxshape=(None,),
            dtype=data_dtype,
            **_sparse_dataset_compression_kwargs(compression_policy, "data"),
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(alloc_size,),
            maxshape=(None,),
            dtype=indices_dtype,
            **_sparse_dataset_compression_kwargs(compression_policy, "indices"),
        )

    indptr = np.zeros(n_out + 1, dtype=np.int64)
    row_pos = 0
    nnz_pos = 0

    def _ensure_capacity(required: int) -> None:
        if required <= data_ds.shape[0]:
            return
        new_size = max(data_ds.shape[0] * 2, required)
        data_ds.resize((new_size,))
        indices_ds.resize((new_size,))

    def _write_block(block: sp.csr_matrix) -> None:
        nonlocal row_pos, nnz_pos
        n_rows = block.shape[0]
        block_nnz = int(block.nnz)

        if block_nnz > 0:
            if not use_fixed:
                _ensure_capacity(nnz_pos + block_nnz)
            data_ds[nnz_pos:nnz_pos + block_nnz] = block.data
            indices_ds[nnz_pos:nnz_pos + block_nnz] = block.indices

        row_nnz = np.diff(block.indptr).astype(np.int64, copy=False)
        if n_rows > 0:
            indptr[row_pos + 1:row_pos + 1 + n_rows] = nnz_pos + np.cumsum(row_nnz, dtype=np.int64)

        row_pos += n_rows
        nnz_pos += block_nnz

    if first_block is not None:
        _write_block(first_block)
    for block in blocks:
        _write_block(block)

    if not use_fixed:
        data_ds.resize((nnz_pos,))
        indices_ds.resize((nnz_pos,))

    grp.create_dataset(
        "indptr",
        data=indptr,
        **_sparse_dataset_compression_kwargs(compression_policy, "indptr"),
    )
    grp.attrs["shape"] = np.array([n_out, n_vars_out])
    grp.attrs["encoding-type"] = encoding
    grp.attrs["encoding-version"] = "0.1.0"


def _write_dense_subsetted(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    compression_policy: dict | None = None,
):
    """Write a row/col-subsetted dense matrix to *f[h5_key]* in chunks."""
    n_out = obs_idx.size
    n_vars_out = var_idx.size if var_idx is not None else matrix.shape[1]
    out_dtype = np.dtype(getattr(matrix, "dtype", np.float64))

    ds = f.create_dataset(
        h5_key,
        shape=(n_out, n_vars_out),
        dtype=out_dtype,
        **_dense_compression_kwargs(compression_policy),
    )
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"

    for pos in range(0, n_out, chunk_size):
        end = min(pos + chunk_size, n_out)
        rows = obs_idx[pos:end]
        block = matrix[rows, :]
        if var_idx is not None:
            block = block[:, var_idx]
        if sp.issparse(block):
            block = block.toarray()
        ds[pos:end, :] = np.asarray(block, dtype=out_dtype)


def _write_subsetted_matrix(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    compression_policy: dict | None = None,
):
    """Dispatch to sparse or dense chunked writer."""
    from ._matrix_source import _is_sparse_matrix_like

    policy = compression_policy if compression_policy is not None else get_matrix_compression_policy(matrix)

    if _is_sparse_matrix_like(matrix):
        _write_sparse_subsetted(
            f,
            h5_key,
            matrix,
            obs_idx,
            var_idx,
            chunk_size,
            compression_policy=policy,
        )
    else:
        _write_dense_subsetted(
            f,
            h5_key,
            matrix,
            obs_idx,
            var_idx,
            chunk_size,
            compression_policy=policy,
        )


def _write_filtered_backed(
    adata: AnnData,
    obs_idx: np.ndarray,
    var_idx: np.ndarray,
    dest_path: str,
    chunk_size: int,
) -> None:
    """Write a row/col-subsetted backed AnnData to *dest_path* via h5py.

    Only ``chunk_size`` rows of the expression matrix are in memory at any
    time, so peak RAM is proportional to ``chunk_size * n_vars_filtered``
    rather than the full filtered matrix.
    """
    import h5py
    from .experimental._anndata_io import _write_dataframe_to_h5, _write_dict_value

    obs_sub = adata.obs.iloc[obs_idx].copy()
    var_sub = adata.var.iloc[var_idx].copy()
    h5file = adata.file._file if is_backed_adata(adata) else None

    obs_is_identity = (
        obs_idx.size == adata.n_obs
        and np.array_equal(obs_idx, np.arange(adata.n_obs, dtype=np.int64))
    )
    var_is_identity = (
        var_idx.size == adata.n_vars
        and np.array_equal(var_idx, np.arange(adata.n_vars, dtype=np.int64))
    )

    with h5py.File(dest_path, "w") as f:
        if h5file is not None:
            for key, value in h5file.attrs.items():
                f.attrs[key] = value
        else:
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"

        # -- X -----------------------------------------------------------
        x_policy = get_matrix_compression_policy(h5file["X"]) if h5file is not None and "X" in h5file else None
        _write_subsetted_matrix(
            f,
            "X",
            adata.X,
            obs_idx,
            var_idx,
            chunk_size,
            compression_policy=x_policy,
        )

        # -- obs / var (small DataFrames) --------------------------------
        _write_dataframe_to_h5(f, "obs", obs_sub)
        _write_dataframe_to_h5(f, "var", var_sub)

        # -- layers ------------------------------------------------------
        layer_keys = list(adata.layers.keys())
        if layer_keys or (h5file is not None and "layers" in h5file):
            lg = f.create_group("layers")
            lg.attrs["encoding-type"] = "dict"
            lg.attrs["encoding-version"] = "0.1.0"
            for lk in layer_keys:
                layer_policy = None
                if h5file is not None and "layers" in h5file and lk in h5file["layers"]:
                    layer_policy = get_matrix_compression_policy(h5file["layers"][lk])
                _write_subsetted_matrix(
                    f,
                    f"layers/{lk}",
                    adata.layers[lk],
                    obs_idx,
                    var_idx,
                    chunk_size,
                    compression_policy=layer_policy,
                )

        # -- obsm / varm (typically small dense arrays) ------------------
        for container, idx, is_identity, name in [
            (adata.obsm, obs_idx, obs_is_identity, "obsm"),
            (adata.varm, var_idx, var_is_identity, "varm"),
        ]:
            keys = list(container.keys())
            if keys or (h5file is not None and name in h5file):
                group = f.create_group(name)
                group.attrs["encoding-type"] = "dict"
                group.attrs["encoding-version"] = "0.1.0"
                for k in keys:
                    # Fast path: copy directly when no row subsetting needed
                    if is_identity and h5file is not None and name in h5file and k in h5file[name]:
                        h5file[name].copy(k, group, name=k)
                        continue
                    mat = container[k]
                    if mat is not None:
                        emb_policy = None
                        if h5file is not None and name in h5file and k in h5file[name]:
                            emb_policy = get_matrix_compression_policy(h5file[name][k])
                        _write_subsetted_matrix(
                            f,
                            f"{name}/{k}",
                            mat,
                            idx,
                            None,
                            chunk_size,
                            compression_policy=emb_policy,
                        )

        # -- obsp / varp (pairwise, row+col subsetted) -------------------
        for container, idx, is_identity, name in [
            (adata.obsp, obs_idx, obs_is_identity, "obsp"),
            (adata.varp, var_idx, var_is_identity, "varp"),
        ]:
            keys = list(container.keys())
            if keys or (h5file is not None and name in h5file):
                group = f.create_group(name)
                group.attrs["encoding-type"] = "dict"
                group.attrs["encoding-version"] = "0.1.0"
                for k in keys:
                    # Fast path: copy directly when no subsetting needed
                    if is_identity and h5file is not None and name in h5file and k in h5file[name]:
                        h5file[name].copy(k, group, name=k)
                        continue
                    mat = container[k]
                    if mat is not None:
                        pair_policy = None
                        if h5file is not None and name in h5file and k in h5file[name]:
                            pair_policy = get_matrix_compression_policy(h5file[name][k])
                        _write_subsetted_matrix(
                            f,
                            f"{name}/{k}",
                            mat,
                            idx,
                            idx,
                            chunk_size,
                            compression_policy=pair_policy,
                        )

        # -- uns (copy verbatim via anndata's own write) -----------------
        if adata.uns:
            uns_grp = f.create_group("uns")
            uns_grp.attrs["encoding-type"] = "dict"
            uns_grp.attrs["encoding-version"] = "0.1.0"
            for k, v in adata.uns.items():
                _write_dict_value(uns_grp, k, v)
        elif h5file is not None and "uns" in h5file:
            h5file.copy("uns", f, name="uns")

        # -- raw (subset by obs only; keep raw var/varm unchanged) -------
        raw = getattr(adata, "raw", None)
        if raw is not None:
            raw_grp = f.create_group("raw")
            raw_grp.attrs["encoding-type"] = "raw"
            raw_grp.attrs["encoding-version"] = "0.1.0"

            raw_policy = None
            if h5file is not None and "raw" in h5file and "X" in h5file["raw"]:
                raw_policy = get_matrix_compression_policy(h5file["raw"]["X"])
            _write_subsetted_matrix(
                f,
                "raw/X",
                raw.X,
                obs_idx,
                None,
                chunk_size,
                compression_policy=raw_policy,
            )
            _write_dataframe_to_h5(f, "raw/var", raw.var.copy())

            if h5file is not None and "raw" in h5file and "varm" in h5file["raw"]:
                h5file.copy("raw/varm", raw_grp, name="varm")

        # -- pass through unhandled top-level groups ----------------------
        if h5file is not None:
            for top_key in h5file.keys():
                if top_key in f:
                    continue
                h5file.copy(top_key, f, name=top_key)


def _view_idx_to_int(idx, axis_size: int) -> np.ndarray:
    """Convert a view index (slice or ndarray) to an int64 index array."""
    if isinstance(idx, slice):
        return np.arange(*idx.indices(axis_size), dtype=np.int64)
    return _normalize_index_array(
        idx,
        axis_size,
        name="view selector",
        allow_negative=True,
    )


def materialize_backed(
    adata: AnnData,
    filename: str | os.PathLike | None = None,
    *,
    chunk_size: int = 4096,
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
    chunk_size : int
        Rows per chunk during the backed write.

    Raises
    ------
    ValueError
        If *adata* is not backed.
    """
    if not is_backed_adata(adata):
        raise ValueError(
            "materialize_backed requires a backed AnnData object."
        )
    if not getattr(adata, "is_view", False):
        return

    parent = adata._adata_ref
    obs_int = _view_idx_to_int(adata._oidx, parent.n_obs)
    var_int = _view_idx_to_int(adata._vidx, parent.n_vars)

    parent_path = str(parent.filename)
    dest_path = str(filename) if filename is not None else parent_path
    in_place_parent = (os.path.realpath(dest_path) == os.path.realpath(parent_path))
    if in_place_parent:
        _ensure_backed_writable(parent)

    dest_dir = os.path.dirname(dest_path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".h5ad")
    os.close(tmp_fd)

    closed_parent = False
    try:
        _write_filtered_backed(parent, obs_int, var_int, tmp_path, chunk_size)
        if in_place_parent and hasattr(parent, "file") and parent.file is not None:
            parent.file.close()
            closed_parent = True
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if in_place_parent and closed_parent:
            try:
                _refresh_backed_handle(parent, parent_path, mode="r+")
            except Exception:
                pass
        raise

    if in_place_parent:
        parent_reopened = ad.read_h5ad(dest_path, backed="r+")
        _init_from_reopened(parent, parent_reopened)
        if adata is parent:
            return

    reopened = ad.read_h5ad(dest_path, backed="r+")
    _init_from_reopened(adata, reopened)


def subset_backed_inplace(
    adata: AnnData,
    obs_idx: np.ndarray | None = None,
    var_idx: np.ndarray | None = None,
    *,
    chunk_size: int = 4096,
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
    chunk_size : int
        Rows per chunk during the backed write.

    Raises
    ------
    ValueError
        If *adata* is not backed or is read-only.
    """
    if not is_backed_adata(adata):
        raise ValueError(
            "subset_backed_inplace requires a backed AnnData object. "
            "Open with ad.read_h5ad(path, backed='r+')."
        )
    _ensure_backed_writable(adata)

    if getattr(adata, "is_view", False):
        materialize_backed(adata, chunk_size=chunk_size)
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
    parent_dir = os.path.dirname(filepath) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=parent_dir, suffix=".h5ad")
    os.close(tmp_fd)

    try:
        _write_filtered_backed(adata, obs_idx, var_idx, tmp_path, chunk_size)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    os.replace(tmp_path, filepath)

    _refresh_backed_handle(adata, filepath, mode="r+")
