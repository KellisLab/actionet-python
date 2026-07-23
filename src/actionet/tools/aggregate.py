"""Grouped aggregation over matrices and AnnData objects.

Covers :func:`aggregate_matrix` (grouped sums/means/vars on a matrix),
:func:`aggregate_anndata` (AnnData-level wrapper that dispatches per
``func`` and returns a new AnnData), and :func:`matrix_sums` (streaming
row/column/global sums for backed or in-memory matrices).
"""

from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .. import _core


def _check_group_vec_missing(group_vec: np.ndarray) -> None:
    """Check for missing values in group vector."""
    if group_vec.dtype.kind in ("f", "c"):
        if np.isnan(group_vec).any():
            raise ValueError("NA values in group_vec.")
    elif group_vec.dtype == object:
        if any(val is None for val in group_vec):
            raise ValueError("NA values in group_vec.")


def _as_csr_sorted(X: sp.spmatrix) -> sp.csr_matrix:
    """Return sparse matrix in canonical CSR form with sorted indices."""
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    if not X.has_sorted_indices:
        X.sort_indices()
    return X


def aggregate_matrix(
    X: Union[np.ndarray, sp.spmatrix],
    group_vec: Union[np.ndarray, list],
    dim: int = 1,
    method: Literal["sum", "mean", "var"] = "sum",
    return_sparse: bool = False,
    return_inverse: bool = False,
) -> Union[np.ndarray, sp.spmatrix, tuple]:
    """Aggregate a matrix by group labels along rows or columns.

    Mirrors the R aggregateMatrix() API and uses the C++ core.

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Input matrix (rows x columns).
    group_vec : array-like
        Group labels for columns (dim=1) or rows (dim=2).
    dim : {1, 2}, default=1
        1 aggregates columns, 2 aggregates rows.
    method : {"sum", "mean", "var"}, default="sum"
        Aggregation method. "var" uses sample variance (ddof=1) and returns
        zeros for singleton groups.
    return_sparse : bool, default=False
        If True and X is sparse, return a sparse matrix.
    return_inverse : bool, default=False
        If True, also return the inverse indices and unique labels from
        numpy.unique used to order the aggregated dimension.

    Returns
    -------
    np.ndarray or sp.spmatrix
        Aggregated matrix.
    tuple
        If return_inverse=True, returns (aggregated, inverse, unique_labels).
    """
    if dim not in (1, 2):
        raise ValueError("'dim' must be either 1 (columns) or 2 (rows).")
    if method not in ("sum", "mean", "var"):
        raise ValueError("method must be 'sum', 'mean', or 'var'.")

    group_vec = np.asarray(group_vec)
    _check_group_vec_missing(group_vec)

    expected_len = X.shape[1] if dim == 1 else X.shape[0]
    if group_vec.shape[0] != expected_len:
        raise ValueError(
            f"Length of group_vec ({group_vec.shape[0]}) does not match the number of "
            f"{'columns' if dim == 1 else 'rows'} ({expected_len}) in X."
        )

    unique_labels, inverse = np.unique(group_vec, return_inverse=True)
    labels = (inverse.astype(np.int64) + 1).astype(np.float64)
    axis = dim - 1

    if sp.issparse(X):
        if return_sparse:
            if method == "sum":
                result = _as_csr_sorted(_core.compute_grouped_sums_sparse2(X, labels, axis))
                return (result, inverse, unique_labels) if return_inverse else result
            if method == "mean":
                result = _as_csr_sorted(_core.compute_grouped_means_sparse2(X, labels, axis))
                return (result, inverse, unique_labels) if return_inverse else result
            result = _as_csr_sorted(_core.compute_grouped_vars_sparse2(X, labels, axis))
            return (result, inverse, unique_labels) if return_inverse else result

        if method == "sum":
            result = _core.compute_grouped_sums_sparse(X, labels, axis)
            return (result, inverse, unique_labels) if return_inverse else result
        if method == "mean":
            result = _core.compute_grouped_means_sparse(X, labels, axis)
            return (result, inverse, unique_labels) if return_inverse else result
        result = _core.compute_grouped_vars_sparse(X, labels, axis)
        return (result, inverse, unique_labels) if return_inverse else result

    X = np.asarray(X, dtype=np.float64)
    if method == "sum":
        result = _core.compute_grouped_sums_dense(X, labels, axis)
        return (result, inverse, unique_labels) if return_inverse else result
    if method == "mean":
        result = _core.compute_grouped_means_dense(X, labels, axis)
        return (result, inverse, unique_labels) if return_inverse else result
    result = _core.compute_grouped_vars_dense(X, labels, axis)
    return (result, inverse, unique_labels) if return_inverse else result


def _count_nonzero_grouped(
    X: Union[np.ndarray, sp.spmatrix],
    group_labels: np.ndarray,
    dim: int,
    todense: bool,
    return_inverse: bool = False,
) -> Union[np.ndarray, sp.spmatrix, tuple]:
    """Count nonzero entries by group. Used by :func:`aggregate_anndata`."""
    if sp.issparse(X):
        if todense:
            X_bool = X.toarray() != 0
            return aggregate_matrix(
                X_bool.astype(np.float64),
                group_labels,
                dim=dim,
                method="sum",
                return_sparse=False,
                return_inverse=return_inverse,
            )
        X_nz = X.copy()
        X_nz.data = np.ones_like(X_nz.data)
        return aggregate_matrix(
            X_nz,
            group_labels,
            dim=dim,
            method="sum",
            return_sparse=True,
            return_inverse=return_inverse,
        )

    X_bool = np.asarray(X) != 0
    return aggregate_matrix(
        X_bool.astype(np.float64),
        group_labels,
        dim=dim,
        method="sum",
        return_sparse=False,
        return_inverse=return_inverse,
    )


def aggregate_anndata(
    adata: AnnData,
    by: Union[str, list[str]],
    func: Union[str, list, None] = None,
    layer: Optional[str] = None,
    min_count: int = 0,
    axis: int = 0,
    todense: bool = False,
) -> AnnData:
    """Aggregate an AnnData object by groups defined in obs or var annotations.

    Parameters
    ----------
    adata
        Input AnnData object.
    by
        Column name(s) in obs (axis=0) or var (axis=1) to group by.
        Multiple columns will be combined with '_'.
    func
        Aggregation function(s): 'sum', 'mean', 'var', or 'count_nonzero'.
        If None, defaults to ['sum', 'mean', 'var', 'count_nonzero'].
        If a single string, applies that function only.
        If a list, applies all specified functions and stores in layers.
    layer
        Name of layer to aggregate. If None, uses .X.
    min_count
        Minimum count threshold.
    axis
        0 to aggregate observations (default), 1 to aggregate variables.
    todense
        If input matrix is sparse, convert to and return dense aggregation.

    Returns
    -------
    AnnData
        Aggregated AnnData object with .X=None and all results stored as named
        layers (one per func).
    """
    if func is None:
        func = ["sum", "mean", "var", "count_nonzero"]
    elif isinstance(func, str):
        func = [func]
    elif not isinstance(func, list):
        raise TypeError(f"func must be a string, list of strings, or None, got {type(func)}")

    valid_funcs = ["sum", "mean", "var", "count_nonzero"]
    for f in func:
        if not isinstance(f, str):
            raise TypeError(f"All elements in func must be strings, got {type(f)}")
        if f not in valid_funcs:
            raise ValueError(
                f"Invalid aggregation function '{f}'. "
                f"Must be one of {valid_funcs}"
            )

    if isinstance(by, str):
        by = [by]
    elif not isinstance(by, list):
        raise TypeError(f"by must be a string or list of strings, got {type(by)}")

    if axis not in (0, 1):
        raise ValueError("axis must be 0 (obs) or 1 (var)")

    X = adata.layers[layer] if layer is not None else adata.X
    metadata_df = adata.obs if axis == 0 else adata.var

    if len(by) == 1:
        group_labels = metadata_df[by[0]].astype(str)
    else:
        group_labels = metadata_df[by].astype(str).agg("_".join, axis=1)

    group_labels = group_labels.values
    dim = 2 if axis == 0 else 1

    if min_count > 0:
        unique_labels, inverse = np.unique(group_labels, return_inverse=True)
        group_counts = np.bincount(inverse, minlength=len(unique_labels))
        keep_mask = group_counts >= min_count
        if not keep_mask.all():
            dropped = unique_labels[~keep_mask]
            dropped_list = ", ".join(map(str, dropped))
            print(f"Dropped groups: {dropped_list}")
            keep_labels = unique_labels[keep_mask]
            axis_mask = np.isin(group_labels, keep_labels)
            if axis == 0:
                adata = adata[axis_mask, :]
            else:
                adata = adata[:, axis_mask]
            X = adata.layers[layer] if layer is not None else adata.X
            metadata_df = adata.obs if axis == 0 else adata.var
            group_labels = (metadata_df[by[0]].astype(str) if len(by) == 1
                            else metadata_df[by].astype(str).agg("_".join, axis=1))
            group_labels = group_labels.values

    unique_labels, inverse = np.unique(group_labels, return_inverse=True)
    group_counts = np.bincount(inverse, minlength=len(unique_labels))

    layers_dict = {}
    for f in func:
        if f == "count_nonzero":
            X_agg = _count_nonzero_grouped(
                X,
                group_labels,
                dim=dim,
                todense=todense,
                return_inverse=False,
            )
        else:
            return_sparse = sp.issparse(X) and not todense
            X_agg = aggregate_matrix(
                X,
                group_labels,
                dim=dim,
                method=f,
                return_sparse=return_sparse,
                return_inverse=False,
            )
        layers_dict[f] = X_agg

    agg_metadata_df = metadata_df[by].copy()
    agg_metadata_df["_group_label"] = group_labels
    agg_metadata_df = (
        agg_metadata_df
        .drop_duplicates(subset="_group_label")
        .set_index("_group_label")
        .loc[list(unique_labels)]
        .reset_index(drop=True)
    )
    agg_metadata_df.index = list(unique_labels)

    count_col = "n_obs" if axis == 0 else "n_var"
    agg_metadata_df[count_col] = group_counts

    if axis == 0:
        adata_agg = AnnData(X=None, obs=agg_metadata_df, var=adata.var.copy())
    else:
        adata_agg = AnnData(X=None, obs=adata.obs.copy(), var=agg_metadata_df)

    adata_agg.layers.update(layers_dict)

    return adata_agg


def matrix_sums(
    adata: AnnData,
    axis: Optional[int] = None,
    layer: Optional[str] = None,
    nonzero: bool = False,
    chunk_size: int = 8192,
) -> Union[np.ndarray, np.floating, np.integer]:
    """Compute sums or non-zero counts of the expression matrix.

    Works transparently for both in-memory (dense or sparse) and backed
    (on-disk HDF5) AnnData objects.  Backed matrices are streamed in
    constant-memory chunks rather than materialised wholesale.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    axis : {0, 1, None}, default=None
        Axis along which to compute, following numpy conventions:

        - ``1`` — sum across columns, returning one value per row (per-cell).
        - ``0`` — sum across rows, returning one value per column (per-gene).
        - ``None`` — scalar global total over the entire matrix.
    layer : str, optional
        Layer to use.  If ``None`` (default), uses ``.X``.
    nonzero : bool, default=False
        If ``False``, compute the sum of all values (float64).
        If ``True``, count the number of non-zero entries (int64) instead
        of summing values.
    chunk_size : int, default=8192
        Number of rows per streaming chunk.
    """
    if axis not in (0, 1, None):
        raise ValueError("axis must be 0, 1, or None.")

    from ..io.matrix_source import MatrixSource
    source = MatrixSource(adata, layer=layer)

    if nonzero:
        if axis == 1:
            return source.nnz_row_counts(chunk_size=chunk_size)
        if axis == 0:
            return source.nnz_col_counts(chunk_size=chunk_size)
        return source.nnz_row_counts(chunk_size=chunk_size).sum()

    if axis == 1:
        return source.row_sums(chunk_size=chunk_size)
    if axis == 0:
        return source.col_sums(chunk_size=chunk_size)
    return source.row_sums(chunk_size=chunk_size).sum()
