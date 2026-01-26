"""Matrix tools for scaling and normalization."""

from typing import Union, Literal
import numpy as np
import scipy.sparse as sp

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
        norms = np.abs(X).sum(axis=axis, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

import numpy as np

def scale(X, center=True, scale=True):
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
