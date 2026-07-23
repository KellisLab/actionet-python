"""Iterative QC filtering and axis-safe subsetting of AnnData objects."""

import os
import pathlib
import shutil
import tempfile
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..io.persist import (
    is_backed_adata,
    _refresh_backed_handle,
)
from ..io.subset import (
    _adaptive_sparse_chunk_size,
    _write_filtered_backed,
    _normalize_index_array,
    _warn_if_duplicates,
    _view_idx_to_int,
    materialize_backed,
    subset_backed_inplace,
)
from ..io.matrix_source import MatrixSource
from ..io.chunking import (
    DEFAULT_BACKED_READ_CHUNK_SIZE,
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    resolve_backed_write_chunk_size,
    validate_chunk_size,
)


def _compute_filter_stats(
    source: MatrixSource,
    obs_idx: np.ndarray,
    var_idx: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass computation of row sums, row nnz, and column nnz.

    Reads each chunk once and accumulates all three statistics
    simultaneously, cutting I/O by 3x compared to three separate passes.

    Returns ``(row_sums, row_nnz, col_nnz)`` where ``row_sums`` and
    ``row_nnz`` have length ``obs_idx.size`` and ``col_nnz`` has length
    ``var_idx.size``.
    """
    row_sums = np.zeros(obs_idx.size, dtype=np.float64)
    row_nnz = np.zeros(obs_idx.size, dtype=np.int64)
    col_nnz = np.zeros(var_idx.size, dtype=np.int64)

    obs_is_full = obs_idx.size == source.n_obs and np.array_equal(
        obs_idx, np.arange(source.n_obs, dtype=np.int64)
    )
    var_is_full = var_idx.size == source.n_vars and np.array_equal(
        var_idx, np.arange(source.n_vars, dtype=np.int64)
    )
    col_indices = None if var_is_full else var_idx

    pos = 0
    if obs_is_full:
        for chunk in source.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices):
            block = chunk.block
            sz = chunk.end - chunk.start
            if sp.issparse(block):
                row_sums[pos : pos + sz] = np.asarray(block.sum(axis=1)).ravel()
                row_nnz[pos : pos + sz] = np.asarray(block.getnnz(axis=1)).ravel()
                col_nnz += np.asarray(block.getnnz(axis=0)).ravel().astype(np.int64, copy=False)
            else:
                arr = np.asarray(block, dtype=np.float64)
                row_sums[pos : pos + sz] = arr.sum(axis=1)
                row_nnz[pos : pos + sz] = np.count_nonzero(arr, axis=1)
                col_nnz += np.count_nonzero(arr, axis=0)
            pos += sz
    else:
        for _rows, block in source.iter_selected_row_chunks(
            obs_idx,
            chunk_size=chunk_size,
            col_indices=col_indices,
        ):
            sz = _rows.size
            if sp.issparse(block):
                row_sums[pos : pos + sz] = np.asarray(block.sum(axis=1)).ravel()
                row_nnz[pos : pos + sz] = np.asarray(block.getnnz(axis=1)).ravel()
                col_nnz += np.asarray(block.getnnz(axis=0)).ravel().astype(np.int64, copy=False)
            else:
                arr = np.asarray(block, dtype=np.float64)
                row_sums[pos : pos + sz] = arr.sum(axis=1)
                row_nnz[pos : pos + sz] = np.count_nonzero(arr, axis=1)
                col_nnz += np.count_nonzero(arr, axis=0)
            pos += sz

    return row_sums, row_nnz, col_nnz


def compute_filter_masks(
    adata: AnnData,
    layer_name: str | None = None,
    *,
    min_cells_per_feat: int | float | None = None,
    min_feats_per_cell: int | None = None,
    min_umis_per_cell: int | None = None,
    max_umis_per_cell: int | None = None,
    backed_chunk_size: int = DEFAULT_BACKED_READ_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute iterative filtering masks without modifying *adata*.

    Alternately evaluates cell and feature QC thresholds until the set of
    passing cells/features stabilises (typically 1--3 iterations).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_name : str or None, optional (default: None)
        Layer to compute statistics from.  ``None`` uses ``adata.X``.
    min_cells_per_feat : int or float or None, optional
        Minimum cells expressing a feature.  A ``float`` in ``(0, 1)``
        is interpreted as a fraction of the current number of passing cells.
    min_feats_per_cell : int or None, optional
        Minimum features detected per cell.
    min_umis_per_cell : int or None, optional
        Minimum total UMI count per cell.
    max_umis_per_cell : int or None, optional
        Maximum total UMI count per cell.
    backed_chunk_size : int, optional (default: 8192)
        Rows per streaming chunk (backed mode only). Read-only path.

    Returns
    -------
    obs_mask : ndarray of bool, shape ``(n_obs,)``
    var_mask : ndarray of bool, shape ``(n_vars,)``
    """
    if min_feats_per_cell is not None and min_feats_per_cell == 0:
        min_feats_per_cell = None
    if min_cells_per_feat is not None and min_cells_per_feat == 0:
        min_cells_per_feat = None

    source = MatrixSource(adata, layer=layer_name)

    obs_idx = np.arange(source.n_obs, dtype=np.int64)
    var_idx = np.arange(source.n_vars, dtype=np.int64)
    prev_shape = None

    while True:
        row_mask = np.ones(obs_idx.size, dtype=bool)
        col_mask = np.ones(var_idx.size, dtype=bool)

        chunk_size = int(max(1, backed_chunk_size))
        if source.is_backed and source.is_sparse:
            chunk_size = _adaptive_sparse_chunk_size(
                source.matrix,
                obs_idx,
                var_idx,
                chunk_size,
                target_block_mb=128,
                overhead_factor=10.0,
            )

        row_sums, row_nnz, col_nnz = _compute_filter_stats(
            source,
            obs_idx,
            var_idx,
            chunk_size,
        )

        if min_umis_per_cell is not None:
            row_mask &= row_sums >= min_umis_per_cell
        if max_umis_per_cell is not None:
            row_mask &= row_sums <= max_umis_per_cell
        if min_feats_per_cell is not None:
            row_mask &= row_nnz >= min_feats_per_cell
        if min_cells_per_feat is not None:
            if isinstance(min_cells_per_feat, float) and 0 < min_cells_per_feat < 1:
                min_fc = int(np.ceil(min_cells_per_feat * obs_idx.size))
            else:
                min_fc = int(min_cells_per_feat)
            col_mask &= col_nnz >= min_fc

        new_shape = (int(row_mask.sum()), int(col_mask.sum()))
        if prev_shape == new_shape:
            break
        prev_shape = new_shape

        obs_idx = obs_idx[row_mask]
        var_idx = var_idx[col_mask]

    obs_mask = np.zeros(adata.n_obs, dtype=bool)
    var_mask = np.zeros(adata.n_vars, dtype=bool)
    obs_mask[obs_idx] = True
    var_mask[var_idx] = True
    return obs_mask, var_mask


# ---------------------------------------------------------------------------
# Generic backed-safe subsetting
# ---------------------------------------------------------------------------


def _coerce_to_int_idx(
    mask_or_idx: np.ndarray,
    axis_size: int,
    *,
    name: str,
) -> np.ndarray:
    """Normalise a boolean mask **or** integer index array to int64 indices.

    Parameters
    ----------
    mask_or_idx : ndarray
        Boolean mask of length *axis_size*, or integer index array.
    axis_size : int
        Length of the axis being subsetted (for validation).
    name : str
        Human-readable axis name for error messages.
    """
    arr = np.asarray(mask_or_idx).ravel()
    if arr.dtype != bool and arr.size > 0 and arr.astype(np.int64, copy=False).min() < 0:
        raise ValueError(f"{name} indices contain negative values")
    idx = _normalize_index_array(
        mask_or_idx,
        axis_size,
        name=name,
        allow_negative=False,
    )
    if arr.dtype != bool:
        _warn_if_duplicates(idx, name=name)
    return idx


def subset_anndata(
    adata: AnnData,
    obs_idx: np.ndarray | None = None,
    var_idx: np.ndarray | None = None,
    *,
    inplace: bool = True,
    output_file: str | None = None,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
) -> AnnData | None:
    """Subset an AnnData safely on both axes (backed and in-memory).

    This is the recommended way to shrink a backed AnnData object.
    For backed objects the HDF5 file is atomically rewritten so that the
    on-disk dimensions match the Python object.  For in-memory objects
    the subset is a standard copy.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_idx : ndarray or None, optional
        Cells to keep -- boolean mask *or* integer index array.
        ``None`` keeps all cells.
    var_idx : ndarray or None, optional
        Features to keep -- boolean mask *or* integer index array.
        ``None`` keeps all features.
    inplace : bool, optional (default: True)
        If ``True``, modify *adata* in place and return ``None``.
        If ``False``, return a new AnnData (see ``output_file`` for backed
        objects).
    output_file : str or None, optional
        Only used when *adata* is backed and ``inplace=False``.
        If ``None``, the filtered data is written to a temporary file,
        loaded into memory, and the temporary file is deleted — the
        returned AnnData is **in-memory**.
        If a path is given, the filtered data is written to that path and
        a new **backed** AnnData opened at that path is returned.  The
        returned handle is always opened in ``r+`` (read-write) mode,
        regardless of the mode of the input object.
        Ignored for in-memory objects and when ``inplace=True``.
    backed_write_chunk_size : int, optional (default: 16384)
        Rows per chunk during backed writes. Atlas-scale sparse rewrites may
        benefit from starting with ``32768``; larger values can increase
        temporary-memory use.

        .. note::
           The backed filtered-rewrite path currently uses a single coupled
           read+write chunk stride internally; this parameter drives that
           stride. Decoupling the read stride from the write stride is
           tracked as a follow-up (see ``TODO`` in
           ``src/actionet/io/subset.py::_write_subsetted_matrix``).

    Returns
    -------
    AnnData or None
        ``None`` when ``inplace=True``; modified copy otherwise.
    """
    backed_write_chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )
    obs_int = (
        _coerce_to_int_idx(obs_idx, adata.n_obs, name="obs")
        if obs_idx is not None
        else np.arange(adata.n_obs, dtype=np.int64)
    )
    var_int = (
        _coerce_to_int_idx(var_idx, adata.n_vars, name="var")
        if var_idx is not None
        else np.arange(adata.n_vars, dtype=np.int64)
    )

    if obs_int.size == 0 or var_int.size == 0:
        raise ValueError(
            "Subset selects zero observations or variables; empty AnnData is not supported"
        )

    backed = is_backed_adata(adata)

    if backed:
        is_view = getattr(adata, "is_view", False)

        if is_view:
            parent = adata._adata_ref
            view_obs = _view_idx_to_int(adata._oidx, parent.n_obs)
            view_var = _view_idx_to_int(adata._vidx, parent.n_vars)
            combined_obs = view_obs[obs_int]
            combined_var = view_var[var_int]
            source = parent
        else:
            combined_obs = obs_int
            combined_var = var_int
            source = adata

        if inplace:
            if is_view:
                materialize_backed(adata, backed_write_chunk_size=backed_write_chunk_size)
                no_extra_obs = obs_idx is None
                no_extra_var = var_idx is None
                if not (no_extra_obs and no_extra_var):
                    subset_backed_inplace(
                        adata,
                        obs_int,
                        var_int,
                        backed_write_chunk_size=backed_write_chunk_size,
                    )
            else:
                subset_backed_inplace(
                    adata,
                    obs_int,
                    var_int,
                    backed_write_chunk_size=backed_write_chunk_size,
                )
            return None

        filepath = str(source.filename)

        if output_file is None:
            # No destination given: load subset into memory and clean up temp.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(pathlib.Path(filepath).parent),
                suffix=".h5ad",
            )
            os.close(tmp_fd)
            try:
                _write_filtered_backed(
                    source,
                    combined_obs,
                    combined_var,
                    tmp_path,
                    backed_write_chunk_size,
                )
                return ad.read_h5ad(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # output_file given: write to that path and return a backed handle.
        dest = str(output_file)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(pathlib.Path(dest).parent),
            suffix=".h5ad",
        )
        os.close(tmp_fd)
        try:
            _write_filtered_backed(
                source,
                combined_obs,
                combined_var,
                tmp_path,
                backed_write_chunk_size,
            )
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        shutil.move(tmp_path, dest)
        return ad.read_h5ad(dest, backed="r+")

    # In-memory path.
    if inplace:
        subset = adata[obs_int, var_int].copy()
        adata.__dict__.update(subset.__dict__)
        return None

    return adata[obs_int, var_int].copy()


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------


def apply_filter(
    adata: AnnData,
    obs_mask: np.ndarray,
    var_mask: np.ndarray,
    *,
    inplace: bool = True,
    output_file: str | None = None,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
) -> AnnData | None:
    """Subset *adata* using precomputed boolean masks.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_mask : ndarray of bool, shape ``(n_obs,)``
        Cells to keep.
    var_mask : ndarray of bool, shape ``(n_vars,)``
        Features to keep.
    inplace : bool, optional (default: True)
        If True, modify *adata* in place (returns ``None``).
        If False, return a new (possibly in-memory) AnnData.
    output_file : str or None, optional
        For backed AnnData, write the filtered result to this path using
        chunked h5py I/O (constant-memory).  If ``None`` and ``inplace=True``,
        the backing file is overwritten in place.  If ``None`` and
        ``inplace=False``, an in-memory AnnData is returned and the source
        backing file is left unchanged.
        Ignored for in-memory objects.
    backed_write_chunk_size : int, optional (default: 16384)
        Rows per chunk during backed writes. Atlas-scale sparse rewrites may
        benefit from starting with ``32768``; larger values can increase
        temporary-memory use.

        .. note::
           The backed filtered-rewrite path currently uses a single coupled
           read+write chunk stride internally; this parameter drives that
           stride. Decoupling the read stride from the write stride is
           tracked as a follow-up (see ``TODO`` in
           ``src/actionet/io/subset.py::_write_subsetted_matrix``).
    """
    backed_write_chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )
    obs_idx = np.where(obs_mask)[0].astype(np.int64)
    var_idx = np.where(var_mask)[0].astype(np.int64)
    backed = is_backed_adata(adata)

    if backed:
        if inplace and output_file is None:
            subset_backed_inplace(adata, obs_idx, var_idx, backed_write_chunk_size=backed_write_chunk_size)
            return None

        if not inplace and output_file is None:
            filepath = str(adata.filename)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(pathlib.Path(filepath).parent),
                suffix=".h5ad",
            )
            os.close(tmp_fd)
            try:
                _write_filtered_backed(adata, obs_idx, var_idx, tmp_path, backed_write_chunk_size)
                return ad.read_h5ad(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        dest = str(output_file)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(pathlib.Path(dest).parent),
            suffix=".h5ad",
        )
        os.close(tmp_fd)
        try:
            _write_filtered_backed(adata, obs_idx, var_idx, tmp_path, backed_write_chunk_size)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if inplace:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            shutil.move(tmp_path, dest)
            _refresh_backed_handle(adata, dest, mode="r+")
            return None

        shutil.move(tmp_path, dest)
        return ad.read_h5ad(dest, backed="r+")

    # In-memory path.
    if inplace:
        subset = adata[obs_mask, var_mask].copy()
        adata._init_as_actual(subset)
        return None

    return adata[obs_mask, var_mask].copy()


# ---------------------------------------------------------------------------
# filter_anndata (backward-compatible wrapper)
# ---------------------------------------------------------------------------


def filter_anndata(
    adata: AnnData,
    layer_name: str | None = None,
    min_cells_per_feat: int | float | None = None,
    min_feats_per_cell: int | None = None,
    min_umis_per_cell: int | None = None,
    max_umis_per_cell: int | None = None,
    inplace: bool = True,
    filter_adata: bool = True,
    backed_chunk_size: int = DEFAULT_BACKED_READ_CHUNK_SIZE,
    backed_write_chunk_size: int | None = None,
) -> Union[AnnData, dict, None]:
    """Iterative QC filtering -- backed-safe, single-pass per iteration.

    Thin wrapper around :func:`compute_filter_masks` and
    :func:`apply_filter`.  Iteratively removes cells and features that
    fail specified thresholds until the dimensions stabilise.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_name : str or None, optional (default: None)
        Layer to compute statistics from.  ``None`` uses ``adata.X``.
    min_cells_per_feat : int or float or None, optional
        Minimum cells expressing a feature.  A ``float`` in ``(0, 1)``
        is interpreted as a fraction of the current number of passing
        cells (recomputed each iteration), not of the original ``n_obs``.
    min_feats_per_cell : int or None, optional
        Minimum features detected per cell.
    min_umis_per_cell : int or None, optional
        Minimum total UMI count per cell.
    max_umis_per_cell : int or None, optional
        Maximum total UMI count per cell.
    inplace : bool, optional (default: True)
        Subset ``adata`` in place.  When False, return a new object.
    filter_adata : bool, optional (default: True)
        If True, apply the filter.  If False, return a dict of masks.
    backed_chunk_size : int, optional (default: 8192)
        Rows per streaming chunk while computing backed filter statistics
        (read-only path).
    backed_write_chunk_size : int or None, optional (default: None)
        Rows per chunk during the backed structural rewrite. ``None`` uses
        the shared write default of ``16384``. Atlas-scale writes may benefit
        from starting with ``32768``; larger values use proportionally more
        temporary memory.
    """
    backed_chunk_size, backed_write_chunk_size = resolve_backed_write_chunk_size(
        backed_chunk_size,
        backed_write_chunk_size,
    )

    obs_mask, var_mask = compute_filter_masks(
        adata,
        layer_name=layer_name,
        min_cells_per_feat=min_cells_per_feat,
        min_feats_per_cell=min_feats_per_cell,
        min_umis_per_cell=min_umis_per_cell,
        max_umis_per_cell=max_umis_per_cell,
        backed_chunk_size=backed_chunk_size,
    )

    if filter_adata:
        return apply_filter(
            adata,
            obs_mask,
            var_mask,
            inplace=inplace,
            backed_write_chunk_size=backed_write_chunk_size,
        )

    obs_names = np.array(adata.obs_names)
    var_names = np.array(adata.var_names)
    filtered_obs = pd.DataFrame(
        {"name": obs_names, "idx": np.arange(adata.n_obs), "mask": obs_mask}
    )
    filtered_vars = pd.DataFrame(
        {"name": var_names, "idx": np.arange(adata.n_vars), "mask": var_mask}
    )
    return {"fil_vars": filtered_vars, "fil_obs": filtered_obs}
