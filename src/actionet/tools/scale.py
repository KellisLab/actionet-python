"""Matrix scaling helpers.

Column-wise :func:`scale` (mirroring R's ``base::scale``) and L1-norm
row/column normalization :func:`l1_norm_scale`.
"""

from typing import Literal, Union

import numpy as np
import scipy.sparse as sp


def l1_norm_scale(
    X: Union[np.ndarray, sp.spmatrix],
    axis: Literal[0, 1] = 1,
) -> Union[np.ndarray, sp.spmatrix]:
    """Scale the rows or columns of a matrix to have L1 norm 1.

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
    X = np.asarray(X, dtype=np.float64)
    norms = np.abs(X).sum(axis=axis, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def scale(
    X,
    center: Union[bool, np.ndarray] = True,
    scale: Union[bool, np.ndarray] = True,
) -> np.ndarray:
    """Column-wise centering and scaling, mirroring R's base ``scale()``.

    Parameters
    ----------
    X : array-like
        Numeric vector, matrix, or any object convertible via
        ``np.asarray``.  Sparse matrices are densified first.
        1-D inputs are treated as single-column matrices and the result
        is returned with the same 1-D shape.
    center : bool or array-like, default True
        If ``True``, center each column by its mean.  If an array, use
        those values directly.  ``False`` disables centering.
    scale : bool or array-like, default True
        If ``True``, scale each column by its sample standard deviation
        (``ddof=1``).  If an array, use those values directly.  ``False``
        disables scaling.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as *X*.
    """
    if sp.issparse(X):
        X = np.asarray(X.toarray(), dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)

    squeezed = X.ndim == 1
    if squeezed:
        X = X[:, np.newaxis]
    elif X.ndim == 0:
        return X

    n_cols = X.shape[1]

    if isinstance(center, bool):
        center_vals = X.mean(axis=0) if center else np.zeros(n_cols)
    else:
        center_vals = np.asarray(center, dtype=np.float64).ravel()
        if center_vals.shape[0] != n_cols:
            raise ValueError("Length of center does not match number of columns")

    if isinstance(scale, bool):
        scale_vals = X.std(axis=0, ddof=1) if scale else np.ones(n_cols)
    else:
        scale_vals = np.asarray(scale, dtype=np.float64).ravel()
        if scale_vals.shape[0] != n_cols:
            raise ValueError("Length of scale does not match number of columns")

    scale_vals = np.where(scale_vals == 0, 1, scale_vals)

    result = (X - center_vals) / scale_vals
    return result.ravel() if squeezed else result
