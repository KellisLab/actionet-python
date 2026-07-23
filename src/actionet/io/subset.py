"""Structural rewrites of backed AnnData files (subset, materialize).

This is one third of the former ``_backed_persist.py``. It provides:

- :func:`subset_backed_inplace` — the only safe way to shrink a backed
  AnnData object.
- :func:`materialize_backed` — turn a backed view into a proper backed
  AnnData.

Both are implemented on top of :func:`_write_filtered_backed`, which
atomically rewrites the HDF5 file with row/col-subsetted chunks.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import anndata as ad
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .compression import (
    CompressionPolicy,
    get_matrix_compression_policy,
    write_sparse_csr_group_attrs,
)
from .persist import (
    _ensure_backed_open,
    _ensure_backed_writable,
    _flush_pending,
    _init_from_reopened,
    _real_layer_keys,
    _refresh_backed_handle,
    is_backed_adata,
)


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
):
    """Write a row/col-subsetted sparse matrix to *f[h5_key]* in chunks."""
    policy = _as_compression_policy(compression_policy)
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

    if estimated_nnz is not None and estimated_nnz > 0:
        alloc_size = estimated_nnz
    elif first_block is not None and first_block.nnz > 0:
        avg_nnz_per_row = first_block.nnz / max(1, first_block.shape[0])
        alloc_size = max(1024, int(avg_nnz_per_row * n_out * 1.1))
    else:
        alloc_size = 1024

    use_fixed = (estimated_nnz is not None and var_idx is None)

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
        **policy.sparse_kwargs("indptr"),
    )
    write_sparse_csr_group_attrs(grp, shape=(n_out, n_vars_out), encoding=encoding)


def _write_dense_subsetted(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    compression_policy: CompressionPolicy | dict | None = None,
):
    """Write a row/col-subsetted dense matrix to *f[h5_key]* in chunks."""
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
    compression_policy: CompressionPolicy | dict | None = None,
):
    """Dispatch to sparse or dense chunked writer."""
    from .matrix_source import _is_sparse_matrix_like

    if compression_policy is None:
        policy = CompressionPolicy.from_matrix(matrix)
    else:
        policy = _as_compression_policy(compression_policy)

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
    from .anndata_io import _write_dataframe_to_h5, _write_dict_value

    _flush_pending(adata)
    _ensure_backed_open(adata)

    obs_sub = adata.obs.iloc[obs_idx].copy()
    var_sub = adata.var.iloc[var_idx].copy()
    h5file = adata.file._file

    obs_is_identity = (
        obs_idx.size == adata.n_obs
        and np.array_equal(obs_idx, np.arange(adata.n_obs, dtype=np.int64))
    )
    var_is_identity = (
        var_idx.size == adata.n_vars
        and np.array_equal(var_idx, np.arange(adata.n_vars, dtype=np.int64))
    )

    with h5py.File(dest_path, "w") as f:
        for key, value in h5file.attrs.items():
            f.attrs[key] = value

        x_policy = get_matrix_compression_policy(h5file["X"]) if "X" in h5file else None
        _write_subsetted_matrix(
            f,
            "X",
            adata.X,
            obs_idx,
            var_idx,
            chunk_size,
            compression_policy=x_policy,
        )

        _write_dataframe_to_h5(f, "obs", obs_sub)
        _write_dataframe_to_h5(f, "var", var_sub)

        layer_keys = _real_layer_keys(adata)
        if layer_keys or "layers" in h5file:
            lg = f.create_group("layers")
            lg.attrs["encoding-type"] = "dict"
            lg.attrs["encoding-version"] = "0.1.0"
            for lk in layer_keys:
                layer_policy = None
                if "layers" in h5file and lk in h5file["layers"]:
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

        for container, idx, is_identity, name in [
            (adata.obsm, obs_idx, obs_is_identity, "obsm"),
            (adata.varm, var_idx, var_is_identity, "varm"),
        ]:
            keys = list(container.keys())
            if keys or name in h5file:
                group = f.create_group(name)
                group.attrs["encoding-type"] = "dict"
                group.attrs["encoding-version"] = "0.1.0"
                for k in keys:
                    if is_identity and name in h5file and k in h5file[name]:
                        h5file[name].copy(k, group, name=k)
                        continue
                    mat = container[k]
                    if mat is not None:
                        import pandas as pd
                        if isinstance(mat, pd.DataFrame):
                            mat_sub = mat.iloc[idx]
                            _write_dataframe_to_h5(f, f"{name}/{k}", mat_sub)
                        else:
                            emb_policy = None
                            if name in h5file and k in h5file[name]:
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

        for container, idx, is_identity, name in [
            (adata.obsp, obs_idx, obs_is_identity, "obsp"),
            (adata.varp, var_idx, var_is_identity, "varp"),
        ]:
            keys = list(container.keys())
            if keys or name in h5file:
                group = f.create_group(name)
                group.attrs["encoding-type"] = "dict"
                group.attrs["encoding-version"] = "0.1.0"
                for k in keys:
                    if is_identity and name in h5file and k in h5file[name]:
                        h5file[name].copy(k, group, name=k)
                        continue
                    mat = container[k]
                    if mat is not None:
                        pair_policy = None
                        if name in h5file and k in h5file[name]:
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

        if adata.uns:
            uns_grp = f.create_group("uns")
            uns_grp.attrs["encoding-type"] = "dict"
            uns_grp.attrs["encoding-version"] = "0.1.0"
            for k, v in adata.uns.items():
                _write_dict_value(uns_grp, k, v)
        elif "uns" in h5file:
            h5file.copy("uns", f, name="uns")

        raw = getattr(adata, "raw", None)
        if raw is not None:
            raw_grp = f.create_group("raw")
            raw_grp.attrs["encoding-type"] = "raw"
            raw_grp.attrs["encoding-version"] = "0.1.0"

            raw_policy = None
            if "raw" in h5file and "X" in h5file["raw"]:
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

            if "raw" in h5file and "varm" in h5file["raw"]:
                h5file.copy("raw/varm", raw_grp, name="varm")

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
        Rows per chunk during the backed write. Atlas-scale sparse rewrites
        may benefit from starting with ``32768``; larger values can increase
        temporary-memory use.

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
    _flush_pending(parent)
    _ensure_backed_open(parent)

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
        Rows per chunk during the backed write. Atlas-scale sparse rewrites
        may benefit from starting with ``32768``; larger values can increase
        temporary-memory use.

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
    _ensure_backed_open(adata)
    _ensure_backed_writable(adata)
    _flush_pending(adata)

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
