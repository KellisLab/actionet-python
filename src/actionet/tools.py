"""Matrix tools for scaling and normalization."""

from typing import Optional, Union, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core

def l1_norm_scale(
    X: Union[np.ndarray, sp.spmatrix],
    axis: Literal[0, 1] = 1
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Scale the rows or columns of a matrix to have L1 norm 1.

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Input matrix (dense or sparse).
    axis : {0, 1}, default=1
        Axis to normalize (0=columns, 1=rows).

    Returns
    -------
    np.ndarray or sp.spmatrix
        L1-normalized matrix of the same type as input.
    """
    if sp.issparse(X):
        # Work with CSR for row, CSC for column
        if axis == 1:
            X = X.tocsr(copy=True)
            norms = np.abs(X).sum(axis=1).A1
            norms[norms == 0] = 1
            X = X.multiply(1 / norms[:, None])
        else:
            X = X.tocsc(copy=True)
            norms = np.abs(X).sum(axis=0).A1
            norms[norms == 0] = 1
            X = X.multiply(1 / norms)
        return X
    else:
        X = np.asarray(X, dtype=np.float64)
        norms = np.abs(X).sum(axis=axis, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

def scale(
    X: Union[np.ndarray, list],
    center: Union[bool, np.ndarray] = True,
    scale: Union[bool, np.ndarray] = True
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_cols = X.shape[1]

    # Handle center argument
    if isinstance(center, bool):
        if center:
            center_vals = X.mean(axis=0)
        else:
            center_vals = np.zeros(n_cols)
    else:
        center_vals = np.asarray(center)
        if center_vals.shape[0] != n_cols:
            raise ValueError("Length of center does not match number of columns")

    # Handle scale argument
    if isinstance(scale, bool):
        if scale:
            scale_vals = X.std(axis=0, ddof=1)
        else:
            scale_vals = np.ones(n_cols)
    else:
        scale_vals = np.asarray(scale)
        if scale_vals.shape[0] != n_cols:
            raise ValueError("Length of scale does not match number of columns")

    # Avoid division by zero
    scale_vals = np.where(scale_vals == 0, 1, scale_vals)

    return (X - center_vals) / scale_vals


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
    """
    Aggregate a matrix by group labels along rows or columns.

    This mirrors the R aggregateMatrix() API and uses the C++ core for aggregation.

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


def matrix_sums(
    adata: AnnData,
    axis: Optional[int] = None,
    layer: Optional[str] = None,
    nonzero: bool = False,
    chunk_size: int = 4096,
) -> Union[np.ndarray, np.floating, np.integer]:
    """
    Compute sums or non-zero counts of the expression matrix.

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
          Result shape: ``(n_obs,)``.
        - ``0`` — sum across rows, returning one value per column (per-gene).
          Result shape: ``(n_vars,)``.
        - ``None`` — scalar global total over the entire matrix.
    layer : str, optional
        Layer to use.  If ``None`` (default), uses ``.X``.
    nonzero : bool, default=False
        If ``False``, compute the sum of all values (float64).
        If ``True``, count the number of non-zero entries (int64) instead
        of summing values.
    chunk_size : int, default=4096
        Number of rows per streaming chunk.  Tune to balance memory usage
        and I/O throughput for backed objects.

    Returns
    -------
    np.ndarray or scalar
        - ``axis=1`` → 1-D float64 (or int64 when ``nonzero=True``) array
          of length ``n_obs``.
        - ``axis=0`` → 1-D float64 (or int64 when ``nonzero=True``) array
          of length ``n_vars``.
        - ``axis=None`` → scalar float64 (or int64 when ``nonzero=True``).

    Examples
    --------
    Per-cell total counts (works backed or in-memory):

    >>> cell_totals = matrix_sums(adata)

    Per-gene total counts:

    >>> gene_totals = matrix_sums(adata, axis=0)

    Number of detected genes per cell:

    >>> n_genes = matrix_sums(adata, nonzero=True)

    Global total across the whole matrix:

    >>> total = matrix_sums(adata, axis=None)
    """
    if axis not in (0, 1, None):
        raise ValueError("axis must be 0, 1, or None.")

    from ._matrix_source import MatrixSource
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
