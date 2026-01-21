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
