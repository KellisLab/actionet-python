"""Matrix tools for scaling and normalization."""

from typing import Union, Literal
import numpy as np
import scipy.sparse as sp

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
                result = _core.compute_grouped_sums_sparse2(X, labels, axis)
                return (result, inverse, unique_labels) if return_inverse else result
            if method == "mean":
                result = _core.compute_grouped_means_sparse2(X, labels, axis)
                return (result, inverse, unique_labels) if return_inverse else result
            result = _core.compute_grouped_vars_sparse2(X, labels, axis)
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
