"""Data preprocessing functions."""

import os
import pathlib
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
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


def normalize_ace(
    adata: AnnData,
    target_sum: float = 1e4,
    layer: str | None = None,
    backed_chunk_size: int = 4096,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Chunked, backed-safe total-count normalization for ``.X`` or a layer.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    target_sum : float
        Target total count per cell after normalization.
    layer : str or None, optional (default: None)
        Layer to normalize.  ``None`` uses ``adata.X``.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when operating on backed AnnData.
        Ignored for in-memory objects.
    inplace : bool, optional (default: True)
        If True, normalize in place; otherwise return a modified copy.
    """
    if target_sum <= 0:
        raise ValueError("target_sum must be positive.")

    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)
    row_sums = source.row_sums(chunk_size=backed_chunk_size)
    scaling = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )

    def _scale_block(block, start: int, end: int):
        scale = scaling[start:end]
        if issparse(block):
            block = block.tocsr(copy=True)
            if block.nnz > 0:
                row_nnz = np.diff(block.indptr)
                block.data *= np.repeat(scale, row_nnz)
            return block
        arr = np.asarray(block, dtype=np.float64)
        arr *= scale[:, np.newaxis]
        return arr

    source.apply_rowwise(_scale_block, chunk_size=backed_chunk_size)

    if inplace:
        return None
    return adata


def log1p_ace(
    adata: AnnData,
    layer: str | None = None,
    base: Optional[float] = None,
    backed_chunk_size: int = 4096,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Chunked, backed-safe ``log1p`` transform for .X or a layer.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str or None, optional (default: None)
        Layer to transform.  ``None`` uses ``adata.X``.
    base : float or None, optional (default: None)
        Logarithm base.  ``None`` uses natural log.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when operating on backed AnnData.
        Ignored for in-memory objects.
    inplace : bool, optional (default: True)
        If True, transform in place; otherwise return a modified copy.
    """
    if base is not None and base <= 0:
        raise ValueError("base must be positive.")

    if not inplace:
        adata = adata.copy()

    if base is None:
        scale = 1.0
    else:
        scale = 1.0 / np.log(base)

    source = MatrixSource(adata, layer=layer)

    # Guard: log1p on negative values yields NaN/complex; warn early.
    if source.is_sparse:
        global_min = source.global_min(chunk_size=backed_chunk_size)
        if global_min < 0:
            import warnings
            warnings.warn(
                f"Matrix contains negative values (min={global_min:.4g}). "
                "log1p is only meaningful for non-negative data; "
                "results may contain NaN.",
                stacklevel=2,
            )

    def _log_block(block, _start: int, _end: int):
        if issparse(block):
            block = block.copy()
            block.data = np.log1p(block.data)
            if scale != 1.0:
                block.data *= scale
            return block
        arr = np.asarray(block, dtype=np.float64)
        np.log1p(arr, out=arr)
        if scale != 1.0:
            arr *= scale
        return arr

    source.apply_rowwise(_log_block, chunk_size=backed_chunk_size)

    if inplace:
        return None
    return adata


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
    """Iterative R-style filtering, implemented with chunked backed-safe metrics.

    The function alternately removes cells and features that fail the
    specified thresholds, then re-evaluates the remaining cells/features until
    the dimensions stabilise.  Convergence is typically reached in 1--3
    iterations; the worst case is ``min(n_obs, n_vars)`` iterations, but this
    is never observed in practice because removing cells/features only
    monotonically shrinks the counts used by subsequent checks.

    For backed AnnData, every iteration performs a full streaming pass
    through the matrix (row sums, nnz counts, column counts).  Keep
    ``backed_chunk_size`` large enough to amortise per-chunk overhead.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_name : str or None, optional (default: None)
        Layer to compute statistics from.  ``None`` uses ``adata.X``.
    min_cells_per_feat : int or float or None, optional
        Minimum number of cells expressing a feature.  A ``float`` in
        ``(0, 1)`` is interpreted as a fraction of ``n_obs``.
    min_feats_per_cell : int or None, optional
        Minimum number of features detected per cell.
    min_umis_per_cell : int or None, optional
        Minimum total UMI count per cell.
    max_umis_per_cell : int or None, optional
        Maximum total UMI count per cell.
    inplace : bool, optional (default: True)
        Subset ``adata`` in place.  When False, return a new object.
    filter_adata : bool, optional (default: True)
        If True, return the filtered AnnData.  If False, return a dict
        of boolean masks and index DataFrames.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when operating on backed AnnData.
        Ignored for in-memory objects.
    """
    source = MatrixSource(adata, layer=layer_name)
    is_backed = source.is_backed

    obs_names = np.array(adata.obs_names)
    var_names = np.array(adata.var_names)

    obs_idx = np.arange(source.n_obs, dtype=np.int64)
    var_idx = np.arange(source.n_vars, dtype=np.int64)
    prev_shape = None

    while True:
        row_mask = np.ones(obs_idx.size, dtype=bool)
        col_mask = np.ones(var_idx.size, dtype=bool)

        row_sums = source.row_sums(
            chunk_size=backed_chunk_size,
            col_indices=var_idx,
            row_indices=obs_idx,
        )
        row_nnz = source.nnz_row_counts(
            chunk_size=backed_chunk_size,
            col_indices=var_idx,
            row_indices=obs_idx,
        )
        col_nnz = source.nnz_col_counts(
            chunk_size=backed_chunk_size,
            col_indices=var_idx,
            row_indices=obs_idx,
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

    final_obs_mask = np.zeros(adata.n_obs, dtype=bool)
    final_var_mask = np.zeros(adata.n_vars, dtype=bool)
    final_obs_mask[obs_idx] = True
    final_var_mask[var_idx] = True

    if filter_adata:
        if inplace:
            if is_backed:
                # Backed AnnData cannot use _inplace_subset_obs/var because
                # AnnData.copy() is unsupported in backed mode.  Instead we
                # materialise the filtered subset, overwrite the backing file,
                # and reinitialise the object from the new file.
                #
                # Note: backed AnnData also forbids "view of a view", so both
                # obs and var masks must be applied in a single indexing step.
                import tempfile, shutil
                filepath = str(adata.filename)
                subset = adata[final_obs_mask, final_var_mask].to_memory()
                # Close the backing file before overwriting
                if hasattr(adata, "file") and adata.file is not None:
                    adata.file.close()
                with tempfile.NamedTemporaryFile(
                    dir=str(pathlib.Path(filepath).parent),
                    suffix=".h5ad",
                    delete=False,
                ) as tmp:
                    tmp_path = tmp.name
                subset.write_h5ad(tmp_path)
                del subset
                shutil.move(tmp_path, filepath)
                reopened = ad.read_h5ad(filepath, backed="r+")
                adata._init_as_actual(reopened)
                # _init_as_actual copies data but drops the file handle;
                # restore it so the object stays backed.
                adata.file = reopened.file
            else:
                adata._inplace_subset_obs(final_obs_mask)
                adata._inplace_subset_var(final_var_mask)
            return None
        if is_backed:
            subset = adata[final_obs_mask, final_var_mask].to_memory()
            return subset
        adata_copy = adata.copy()
        adata_copy._inplace_subset_obs(final_obs_mask)
        adata_copy._inplace_subset_var(final_var_mask)
        return adata_copy

    filtered_obs = pd.DataFrame(
        {"name": obs_names, "idx": np.arange(adata.n_obs), "mask": final_obs_mask}
    )
    filtered_vars = pd.DataFrame(
        {"name": var_names, "idx": np.arange(adata.n_vars), "mask": final_var_mask}
    )
    return {"fil_vars": filtered_vars, "fil_obs": filtered_obs}
