"""Data preprocessing functions."""

import os
import pathlib
import shutil
import tempfile
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse

from ._matrix_source import MatrixSource


def import_anndata_generic(
    input_path: str,
    mtx_file: str,
    gene_annotations: str,
    sample_annotations: str,
    gene_headers: list[str] | None = None,
    sample_headers: list[str] | None = None,
    sep: str = "\t",
    prefilter: bool = False,
    prefil_params: dict | None = None,
) -> AnnData:
    """Python implementation of import.se.generic from R, using AnnData."""
    gene_path = os.path.join(input_path, gene_annotations)
    gene_table = pd.read_csv(gene_path, sep=sep, header=None, dtype=str)
    if gene_headers is not None:
        gene_table.columns = gene_headers

    sample_path = os.path.join(input_path, sample_annotations)
    sample_annots = pd.read_csv(sample_path, sep=sep, header=None, dtype=str)
    if sample_headers is not None:
        sample_annots.columns = sample_headers

    obs_names = sample_annots.iloc[:, 0].tolist()
    var_names = gene_table.iloc[:, 0].tolist()

    mtx_path = os.path.join(input_path, mtx_file)
    X = mmread(mtx_path).transpose()
    adata = AnnData(X=csr_matrix(X), obs=sample_annots, var=gene_table)
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obs_names_make_unique(join="_")
    adata.var_names_make_unique(join="_")

    if prefilter and prefil_params is not None:
        adata = filter_anndata(
            adata,
            layer_name=None,
            min_cells_per_feat=prefil_params.get("min_cells_per_feat", None),
            min_feats_per_cell=prefil_params.get("min_feats_per_cell", None),
            min_umis_per_cell=prefil_params.get("min_umis_per_cell", None),
            max_umis_per_cell=prefil_params.get("max_umis_per_cell", None),
            filter_adata=True,
            inplace=False,
        )

    return adata


def normalize_anndata(
    adata: AnnData,
    target_sum: float = 1e4,
    log_transform: bool = True,
    log_base: Optional[float] = None,
    layer: str | None = None,
    backed_chunk_size: int = 4096,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Total-count normalization with optional log1p transform.

    Mimics the R ``normalize.ace`` function: each cell's counts are scaled
    to *target_sum*, then (by default) a ``log1p`` transform is applied.
    Both steps are chunked and backed-safe, so this function works
    identically on in-memory and HDF5-backed AnnData objects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    target_sum : float, optional (default: 1e4)
        Target total count per cell after scaling.
    log_transform : bool, optional (default: True)
        If ``True``, apply ``log1p`` (optionally with a custom base) after
        scaling.  Set to ``False`` for scaling only.
    log_base : float or None, optional (default: None)
        Logarithm base for the log1p step.  ``None`` uses the natural
        logarithm.  Ignored when *log_transform* is ``False``.
    layer : str or None, optional (default: None)
        Layer to normalize.  ``None`` uses ``adata.X``.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when operating on backed AnnData.
        Ignored for in-memory objects.
    inplace : bool, optional (default: True)
        If ``True``, normalize in place and return ``None``; otherwise
        return a modified copy.

    Returns
    -------
    AnnData or None
        Modified copy if ``inplace=False``, otherwise ``None``.
    """
    if target_sum <= 0:
        raise ValueError("target_sum must be positive.")
    if log_base is not None and log_base <= 0:
        raise ValueError("log_base must be positive.")

    if not inplace:
        if getattr(adata, "isbacked", False):
            adata = adata.to_memory()
        else:
            adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    if not source.is_backed:
        _normalize_in_memory(adata, source, target_sum, log_transform, log_base, layer)
    else:
        _normalize_backed(source, target_sum, log_transform, log_base, backed_chunk_size)

    if inplace:
        return None
    return adata


# ---------------------------------------------------------------------------
# In-memory fast path (no chunking)
# ---------------------------------------------------------------------------

def _normalize_in_memory(
    adata: AnnData,
    source: MatrixSource,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    layer: str | None,
) -> None:
    """Normalize an in-memory AnnData matrix without chunked iteration.

    Operates directly on the underlying sparse or dense arrays, avoiding
    the per-chunk slicing, copying, and write-back overhead that the
    backed path requires.
    """
    X = source.matrix

    if issparse(X):
        X = X.tocsr()
        X = X.astype(np.float64, copy=False)
        # Ensure the converted matrix is stored back
        if layer is None:
            adata.X = X
        else:
            adata.layers[layer] = X

        # Row sums via scipy (single sparse mat-vec, no chunking)
        row_sums = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        scaling = np.divide(
            target_sum,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 0,
        )

        if X.nnz > 0:
            row_nnz = np.diff(X.indptr)
            X.data *= np.repeat(scaling, row_nnz)

            if log_transform:
                if X.data.min() < 0:
                    import warnings
                    warnings.warn(
                        f"Matrix contains negative values (min={X.data.min():.4g}). "
                        "log1p is only meaningful for non-negative data; "
                        "results may contain NaN.",
                        stacklevel=2,
                    )
                np.log1p(X.data, out=X.data)
                if log_base is not None:
                    X.data /= np.log(log_base)
    else:
        arr = np.asarray(X, dtype=np.float64)
        row_sums = arr.sum(axis=1)
        scaling = np.divide(
            target_sum,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 0,
        )

        arr *= scaling[:, np.newaxis]

        if log_transform:
            np.log1p(arr, out=arr)
            if log_base is not None:
                arr /= np.log(log_base)

        if layer is None:
            adata.X = arr
        else:
            adata.layers[layer] = arr


# ---------------------------------------------------------------------------
# Backed (disk-backed) chunked path
# ---------------------------------------------------------------------------

def _normalize_backed(
    source: MatrixSource,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    chunk_size: int,
) -> None:
    """Normalize a backed AnnData matrix using chunked streaming I/O."""
    row_sums = source.row_sums(chunk_size=chunk_size)
    scaling = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )

    if log_transform:
        log_scale = 1.0 if log_base is None else 1.0 / np.log(log_base)

        if source.is_sparse:
            global_min = source.global_min(chunk_size=chunk_size)
            if global_min < 0:
                import warnings
                warnings.warn(
                    f"Matrix contains negative values (min={global_min:.4g}). "
                    "log1p is only meaningful for non-negative data; "
                    "results may contain NaN.",
                    stacklevel=2,
                )

        def _normalize_block(block, start: int, end: int):
            scale = scaling[start:end]
            if issparse(block):
                block = block.tocsr(copy=True)
                block = block.astype(np.float64, copy=False)
                if block.nnz > 0:
                    row_nnz = np.diff(block.indptr)
                    block.data *= np.repeat(scale, row_nnz)
                    np.log1p(block.data, out=block.data)
                    if log_scale != 1.0:
                        block.data *= log_scale
                return block
            arr = np.asarray(block, dtype=np.float64)
            arr *= scale[:, np.newaxis]
            np.log1p(arr, out=arr)
            if log_scale != 1.0:
                arr *= log_scale
            return arr
    else:
        def _normalize_block(block, start: int, end: int):
            scale = scaling[start:end]
            if issparse(block):
                block = block.tocsr(copy=True)
                block = block.astype(np.float64, copy=False)
                if block.nnz > 0:
                    row_nnz = np.diff(block.indptr)
                    block.data *= np.repeat(scale, row_nnz)
                return block
            arr = np.asarray(block, dtype=np.float64)
            arr *= scale[:, np.newaxis]
            return arr

    source.apply_rowwise(_normalize_block, chunk_size=chunk_size)


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

    pos = 0
    for _rows, block in source.iter_selected_row_chunks(
        obs_idx, chunk_size=chunk_size, col_indices=var_idx,
    ):
        sz = _rows.size
        if sp.issparse(block):
            row_sums[pos:pos + sz] = np.asarray(block.sum(axis=1)).ravel()
            row_nnz[pos:pos + sz] = np.asarray(block.getnnz(axis=1)).ravel()
            col_nnz += np.asarray(block.getnnz(axis=0)).ravel().astype(np.int64, copy=False)
        else:
            arr = np.asarray(block, dtype=np.float64)
            row_sums[pos:pos + sz] = arr.sum(axis=1)
            row_nnz[pos:pos + sz] = np.count_nonzero(arr, axis=1)
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
    backed_chunk_size: int = 4096,
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
    backed_chunk_size : int, optional (default: 4096)
        Rows per streaming chunk (backed mode only).

    Returns
    -------
    obs_mask : ndarray of bool, shape ``(n_obs,)``
    var_mask : ndarray of bool, shape ``(n_vars,)``
    """
    source = MatrixSource(adata, layer=layer_name)

    obs_idx = np.arange(source.n_obs, dtype=np.int64)
    var_idx = np.arange(source.n_vars, dtype=np.int64)
    prev_shape = None

    while True:
        row_mask = np.ones(obs_idx.size, dtype=bool)
        col_mask = np.ones(var_idx.size, dtype=bool)

        row_sums, row_nnz, col_nnz = _compute_filter_stats(
            source, obs_idx, var_idx, backed_chunk_size,
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
# Chunked h5py write for backed subsetting
# ---------------------------------------------------------------------------

def _write_sparse_subsetted(
    f,
    h5_key: str,
    matrix,
    obs_idx: np.ndarray,
    var_idx: np.ndarray | None,
    chunk_size: int,
    encoding: str = "csr_matrix",
):
    """Write a row/col-subsetted sparse matrix to *f[h5_key]* in chunks.

    Only ``chunk_size`` rows are loaded into memory at a time.
    """
    source_mat = matrix
    n_out = obs_idx.size
    n_vars_out = var_idx.size if var_idx is not None else source_mat.shape[1]

    # Accumulate CSR components in lists, then concatenate once.
    all_data = []
    all_indices = []
    indptr = [0]
    nnz_so_far = 0

    for pos in range(0, n_out, chunk_size):
        end = min(pos + chunk_size, n_out)
        rows = obs_idx[pos:end]
        block = source_mat[rows, :]
        if var_idx is not None:
            block = block[:, var_idx]
        block = sp.csr_matrix(block)
        all_data.append(block.data)
        all_indices.append(block.indices)
        for row_nnz in np.diff(block.indptr):
            nnz_so_far += int(row_nnz)
            indptr.append(nnz_so_far)

    data = np.concatenate(all_data) if all_data else np.array([], dtype=np.float64)
    indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    indptr = np.array(indptr, dtype=np.int64)

    grp = f.create_group(h5_key)
    grp.create_dataset("data", data=data, compression="gzip")
    grp.create_dataset("indices", data=indices, compression="gzip")
    grp.create_dataset("indptr", data=indptr, compression="gzip")
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
):
    """Write a row/col-subsetted dense matrix to *f[h5_key]* in chunks."""
    n_out = obs_idx.size
    n_vars_out = var_idx.size if var_idx is not None else matrix.shape[1]

    ds = f.create_dataset(
        h5_key,
        shape=(n_out, n_vars_out),
        dtype=np.float64,
        compression="gzip",
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
):
    """Dispatch to sparse or dense chunked writer."""
    from ._matrix_source import _is_sparse_matrix_like

    if _is_sparse_matrix_like(matrix):
        _write_sparse_subsetted(f, h5_key, matrix, obs_idx, var_idx, chunk_size)
    else:
        _write_dense_subsetted(f, h5_key, matrix, obs_idx, var_idx, chunk_size)


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

    with h5py.File(dest_path, "w") as f:
        # -- X -----------------------------------------------------------
        _write_subsetted_matrix(f, "X", adata.X, obs_idx, var_idx, chunk_size)

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
                _write_subsetted_matrix(
                    f, f"layers/{lk}", adata.layers[lk],
                    obs_idx, var_idx, chunk_size,
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
    backed_chunk_size: int = 4096,
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
        chunked h5py I/O (constant-memory).  If ``None`` and the object
        is backed, the backing file is overwritten in place.
        Ignored for in-memory objects.
    backed_chunk_size : int, optional (default: 4096)
        Rows per chunk during backed writes.
    """
    is_backed = bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None))
    obs_idx = np.where(obs_mask)[0].astype(np.int64)
    var_idx = np.where(var_mask)[0].astype(np.int64)

    if is_backed:
        filepath = str(adata.filename)
        dest = output_file if output_file is not None else filepath

        # Write to a temp file first (atomic replace).
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(pathlib.Path(dest).parent), suffix=".h5ad",
        )
        os.close(tmp_fd)
        try:
            _write_filtered_backed(adata, obs_idx, var_idx, tmp_path, backed_chunk_size)
        except Exception:
            os.unlink(tmp_path)
            raise

        if inplace:
            # Close the old backing file, replace it, reopen.
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            shutil.move(tmp_path, dest)
            reopened = ad.read_h5ad(dest, backed="r+")
            adata._init_as_actual(reopened)
            adata.file = reopened.file
            return None

        # Not inplace: move temp to dest, return backed or in-memory.
        shutil.move(tmp_path, dest)
        if output_file is not None:
            return ad.read_h5ad(dest, backed="r+")
        return ad.read_h5ad(dest)

    # In-memory path: single view-to-copy instead of two _inplace_subset calls.
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
    backed_chunk_size: int = 4096,
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
        is interpreted as a fraction of ``n_obs``.
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
    backed_chunk_size : int, optional (default: 4096)
        Rows per streaming chunk (backed mode only).
    """
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
            adata, obs_mask, var_mask,
            inplace=inplace,
            backed_chunk_size=backed_chunk_size,
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
