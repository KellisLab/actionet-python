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
import shutil
import tempfile
from typing import Any, Mapping, MutableMapping, Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._backed_compression import get_matrix_compression_policy

try:
    from .experimental import _anndata_io
except Exception:  # pragma: no cover - optional in some build contexts
    _anndata_io = None


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

    _anndata_io.append_to_anndata(
        str(adata.filename),
        results,
        verbose=verbose,
        validate=validate,
    )


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
    adata._init_as_actual(reopened)
    adata.file = reopened.file

    if verbose:
        print(f"[INFO] Compacted {src_path}")


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
        obs_columns=list(adata.obs.columns),
        var_columns=list(adata.var.columns),
        obsm_keys=list(adata.obsm.keys()),
        varm_keys=list(adata.varm.keys()),
        obsp_keys=list(adata.obsp.keys()),
        varp_keys=list(adata.varp.keys()),
        layers_keys=[],
        uns_keys=list(adata.uns.keys()),
        verbose=verbose,
    )

    has_data = any(len(v) > 0 for v in results.values())

    if has_data:
        if verbose:
            print(f"[INFO] Checkpointing to {adata.filename}")

        _anndata_io.append_to_anndata(
            str(adata.filename),
            results,
            verbose=verbose,
            validate=validate,
        )

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
    initial_capacity = 1024
    if first_block is not None and first_block.nnz > 0:
        data_dtype = first_block.data.dtype
        indices_dtype = first_block.indices.dtype
        initial_capacity = max(initial_capacity, int(first_block.nnz) * 2)

    grp = f.create_group(h5_key)
    data_ds = grp.create_dataset(
        "data",
        shape=(initial_capacity,),
        maxshape=(None,),
        dtype=data_dtype,
        **_sparse_dataset_compression_kwargs(compression_policy, "data"),
    )
    indices_ds = grp.create_dataset(
        "indices",
        shape=(initial_capacity,),
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
        new_size = data_ds.shape[0]
        while new_size < required:
            new_size = max(new_size * 2, required)
        data_ds.resize((new_size,))
        indices_ds.resize((new_size,))

    def _write_block(block: sp.csr_matrix) -> None:
        nonlocal row_pos, nnz_pos
        n_rows = block.shape[0]
        block_nnz = int(block.nnz)

        if block_nnz > 0:
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

    ds = f.create_dataset(
        h5_key,
        shape=(n_out, n_vars_out),
        dtype=np.float64,
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
        ds[pos:end, :] = np.asarray(block, dtype=np.float64)


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
    from .experimental._anndata_io import _write_dataframe_to_h5

    obs_sub = adata.obs.iloc[obs_idx].copy()
    var_sub = adata.var.iloc[var_idx].copy()
    h5file = adata.file._file if is_backed_adata(adata) else None

    with h5py.File(dest_path, "w") as f:
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
        if layer_keys:
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
        for container, idx, name in [
            (adata.obsm, obs_idx, "obsm"),
            (adata.varm, var_idx, "varm"),
        ]:
            keys = list(container.keys())
            if keys:
                f.create_group(name)
                for k in keys:
                    mat = container[k]
                    sub = mat[idx] if mat is not None else None
                    if sub is not None:
                        h5k = f"{name}/{k}"
                        if sp.issparse(sub):
                            sub = sub.tocsr()
                            grp = f.create_group(h5k)
                            grp.create_dataset("data", data=sub.data, compression="gzip")
                            grp.create_dataset("indices", data=sub.indices, compression="gzip")
                            grp.create_dataset("indptr", data=sub.indptr, compression="gzip")
                            grp.attrs["shape"] = np.array(sub.shape)
                            grp.attrs["encoding-type"] = "csr_matrix"
                            grp.attrs["encoding-version"] = "0.1.0"
                        else:
                            ds = f.create_dataset(h5k, data=np.asarray(sub), compression="gzip")
                            ds.attrs["encoding-type"] = "array"
                            ds.attrs["encoding-version"] = "0.2.0"

        # -- obsp / varp (pairwise, row+col subsetted) -------------------
        for container, idx, name in [
            (adata.obsp, obs_idx, "obsp"),
            (adata.varp, var_idx, "varp"),
        ]:
            keys = list(container.keys())
            if keys:
                f.create_group(name)
                for k in keys:
                    mat = container[k]
                    if mat is not None:
                        sub = mat[np.ix_(idx, idx)]
                        h5k = f"{name}/{k}"
                        if sp.issparse(sub):
                            sub = sub.tocsr()
                            grp = f.create_group(h5k)
                            grp.create_dataset("data", data=sub.data, compression="gzip")
                            grp.create_dataset("indices", data=sub.indices, compression="gzip")
                            grp.create_dataset("indptr", data=sub.indptr, compression="gzip")
                            grp.attrs["shape"] = np.array(sub.shape)
                            grp.attrs["encoding-type"] = "csr_matrix"
                            grp.attrs["encoding-version"] = "0.1.0"
                        else:
                            ds = f.create_dataset(h5k, data=np.asarray(sub), compression="gzip")
                            ds.attrs["encoding-type"] = "array"
                            ds.attrs["encoding-version"] = "0.2.0"

        # -- uns (copy verbatim via anndata's own write) -----------------
        if adata.uns:
            from .experimental._anndata_io import _write_dict_value
            uns_grp = f.create_group("uns")
            uns_grp.attrs["encoding-type"] = "dict"
            uns_grp.attrs["encoding-version"] = "0.1.0"
            for k, v in adata.uns.items():
                _write_dict_value(uns_grp, k, v)


def _view_idx_to_int(idx, axis_size: int) -> np.ndarray:
    """Convert a view index (slice or ndarray) to an int64 index array."""
    if isinstance(idx, slice):
        return np.arange(*idx.indices(axis_size), dtype=np.int64)
    return np.asarray(idx, dtype=np.int64).ravel()


def materialize_backed(
    adata: AnnData,
    filename: str | os.PathLike | None = None,
    *,
    chunk_size: int = 4096,
) -> None:
    """Materialize a backed AnnData view into a standalone backed object.

    Turns a backed view (created by e.g. ``adata[1:1000, :]``) into a
    proper backed AnnData that owns its own HDF5 file.  The parent object
    is not modified.

    If *adata* is already a non-view backed object this is a no-op.

    Parameters
    ----------
    adata : AnnData
        A backed AnnData view (``adata.is_view`` and ``adata.isbacked``).
    filename : path-like or None
        Destination HDF5 path.  When ``None`` a sibling file is created
        next to the parent's backing file.
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
    parent_dir = os.path.dirname(parent_path) or "."

    if filename is not None:
        dest_path = str(filename)
    else:
        import uuid
        stem = os.path.splitext(os.path.basename(parent_path))[0]
        tag = uuid.uuid4().hex[:8]
        dest_path = os.path.join(parent_dir, f"{stem}_subset_{tag}.h5ad")

    try:
        _write_filtered_backed(parent, obs_int, var_int, dest_path, chunk_size)
    except Exception:
        if os.path.exists(dest_path):
            os.unlink(dest_path)
        raise

    reopened = ad.read_h5ad(dest_path, backed="r+")
    adata._init_as_actual(reopened)
    adata.file = reopened.file


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
        obs_idx = np.asarray(obs_idx, dtype=np.int64).ravel()

    if var_idx is None:
        var_idx = np.arange(adata.n_vars, dtype=np.int64)
    else:
        var_idx = np.asarray(var_idx, dtype=np.int64).ravel()

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
    shutil.move(tmp_path, filepath)

    _refresh_backed_handle(adata, filepath, mode="r+")
