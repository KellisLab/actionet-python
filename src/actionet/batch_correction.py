"""Batch correction and normalization utilities."""

from typing import Optional, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import pandas as pd

from . import _core
from .anndata_utils import anndata_to_matrix
from ._backed_persist import persist_updates
from ._matrix_source import MatrixSource


def _load_reduction_state(adata: AnnData, reduction_key: str) -> tuple[np.ndarray, ...]:
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")

    params = adata.uns[params_key]
    old_S_r = np.asarray(adata.obsm[reduction_key], dtype=float, order="C")  # cells x k
    old_U = np.asarray(adata.varm[f"{reduction_key}_U"], dtype=float, order="C")
    old_A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    old_B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")
    old_sigma = np.asarray(params["sigma"], dtype=float).reshape(-1)

    if old_S_r.shape[0] != adata.n_obs:
        raise ValueError("Reduction matrix has unexpected shape; expected cells x components.")
    if old_sigma.shape[0] != old_S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    return old_S_r, old_U, old_A, old_B, old_sigma


def _orthonormalize_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """QR-orthonormalise the columns of *X*, dropping near-zero-norm columns."""
    if X.size == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)

    Q, R = np.linalg.qr(np.asarray(X, dtype=np.float64), mode="reduced")
    keep = np.abs(np.diag(R)) > eps
    if np.any(keep):
        return Q[:, keep]
    return np.zeros((X.shape[0], 0), dtype=np.float64)


def _deflate_terms(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Augment perturbation matrices with a mean-deflation column.

    Given perturbation pair ``(A, B)`` representing a rank-*q* additive
    correction ``A @ B.T``, this prepends a column that removes the column
    mean of A:

        A_aug = [1, A]          (n_vars, q+1)
        B_aug = [-B @ mean(A), B]   (n_obs, q+1)

    This ensures that the perturbedSVD call simultaneously centres the
    correction and applies the orthogonalisation.
    """
    mu_A = np.asarray(A.mean(axis=0), dtype=np.float64).reshape(-1)
    mu = np.asarray(B @ mu_A, dtype=np.float64).reshape(-1, 1)
    A_aug = np.column_stack([np.ones((A.shape[0], 1), dtype=np.float64), A])
    B_aug = np.column_stack([-mu, B])
    return np.ascontiguousarray(A_aug), np.ascontiguousarray(B_aug)


def _perturbed_with_prior(
    old_S_r: np.ndarray,
    old_U: np.ndarray,
    old_A: np.ndarray,
    old_B: np.ndarray,
    old_sigma: np.ndarray,
    A_new: np.ndarray,
    B_new: np.ndarray,
) -> dict:
    """Apply a new additive perturbation on top of an existing corrected SVD.

    Reconstructs the right singular vectors ``V`` from ``S_r / sigma``,
    calls the C++ ``perturbedSVD`` with both the prior perturbation terms
    ``(old_A, old_B)`` and the new terms ``(A_new, B_new)``, and returns the
    updated reduction in the same dict format as the in-memory C++ path.

    Returns
    -------
    dict with keys ``"U"``, ``"sigma"``, ``"S_r"``, ``"A"``, ``"B"``.
    ``S_r`` has shape ``(k, n_obs)`` -- the same orientation as the
    in-memory C++ binding output.
    """
    old_V = old_S_r / old_sigma[np.newaxis, :]
    out = _core.perturbed_svd_with_prior(
        np.ascontiguousarray(old_U, dtype=np.float64),
        np.ascontiguousarray(old_sigma, dtype=np.float64),
        np.ascontiguousarray(old_V, dtype=np.float64),
        np.ascontiguousarray(old_A, dtype=np.float64),
        np.ascontiguousarray(old_B, dtype=np.float64),
        np.ascontiguousarray(A_new, dtype=np.float64),
        np.ascontiguousarray(B_new, dtype=np.float64),
    )

    sigma = np.asarray(out["d"], dtype=np.float64).reshape(-1)
    V = np.asarray(out["v"], dtype=np.float64, order="C")
    S_r = (V * sigma[np.newaxis, :]).T

    # Orientation contract: S_r is (k, n_obs), matching the in-memory C++ path.
    assert S_r.shape[0] == sigma.shape[0], (
        f"S_r orientation mismatch: expected ({sigma.shape[0]}, n_obs), "
        f"got {S_r.shape}"
    )

    return {
        "U": np.asarray(out["u"], dtype=np.float64, order="C"),
        "sigma": sigma,
        "S_r": np.asarray(S_r, dtype=np.float64, order="C"),
        "A": np.asarray(out["A"], dtype=np.float64, order="C"),
        "B": np.asarray(out["B"], dtype=np.float64, order="C"),
    }


def _streamed_batch_terms(
    source: MatrixSource, design: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute batch-correction perturbation terms from a backed matrix.

    Two streaming passes are required:
      1. ``Z = orth(X.T @ design)``  -- accumulates ``X.T @ design`` row-wise.
      2. ``B = -X @ Z``              -- needs ``Z`` from pass 1.

    Returns ``(Z, B)`` with shapes ``(n_vars, q)`` and ``(n_obs, q)``
    where *q* is the rank of the orthonormalised design projection.
    """
    Z = source.xt_dot(design, chunk_size=chunk_size)  # genes x covariates
    Z = _orthonormalize_columns(Z)
    if Z.shape[1] == 0:
        raise ValueError("Design matrix is rank-deficient after orthonormalization.")
    B = -source.x_dot(Z, chunk_size=chunk_size)  # cells x covariates
    return Z, B


def _streamed_basal_terms(
    source: MatrixSource, basal: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute basal-correction perturbation terms from a backed matrix.

    Only one streaming pass is needed because ``Z = orth(basal)`` does not
    require reading the matrix.

    Returns ``(Z, B)`` with shapes ``(n_vars, q)`` and ``(n_obs, q)``.
    """
    Z = _orthonormalize_columns(basal)
    if Z.shape[1] == 0:
        raise ValueError("Basal state matrix is rank-deficient after orthonormalization.")
    B = -source.x_dot(Z, chunk_size=chunk_size)
    return Z, B


def correct_batch_effect(
    adata: AnnData,
    batch_key: Optional[str] = None,
    design: Optional[np.ndarray] = None,
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_suffix: str = "corrected",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
) -> AnnData:
    """Correct batch effects using orthogonalization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed reduction.
    batch_key : str or None
        Column in ``adata.obs`` containing batch labels (categorical).
        Mutually exclusive with *design*.
    design : ndarray or None
        Explicit design matrix ``(n_obs, n_covariates)``.
        Mutually exclusive with *batch_key*.
    reduction_key : str
        Key prefix for the existing reduction in ``adata.obsm``.
    layer : str or None
        Layer to use.  ``None`` uses ``adata.X``.
    corrected_suffix : str
        Suffix appended to *reduction_key* for storing corrected results.
    inplace : bool
        Modify *adata* in place or return a copy.
    backed_chunk_size : int
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    """
    corrected_key = f"{reduction_key}_{corrected_suffix}"

    if (batch_key is None) == (design is None):
        raise ValueError("Provide exactly one of 'batch_key' or 'design'.")

    if not inplace:
        adata = adata.copy()

    if batch_key is not None and batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    old_S_r, old_U, old_A, old_B, old_sigma = _load_reduction_state(adata, reduction_key)

    if batch_key is not None:
        batch_labels = pd.Series(adata.obs[batch_key].to_numpy(), index=adata.obs_names).astype("category")
        if len(batch_labels.cat.categories) < 2:
            raise ValueError("'batch_key' must have at least 2 categories for correction.")
        design = pd.get_dummies(batch_labels, drop_first=False).to_numpy(dtype=float)
    else:
        design = np.asarray(design, dtype=float)
        if design.ndim != 2:
            raise ValueError("'design' must be a 2D matrix (cells x covariates).")

    if design.shape[0] != adata.n_obs:
        raise ValueError("Design matrix does not match number of observations.")
    design = np.ascontiguousarray(design)

    source = MatrixSource(adata, layer=layer)
    if source.is_backed:
        A_new, B_new = _streamed_batch_terms(source, design, chunk_size=backed_chunk_size)
        A_deflate, B_deflate = _deflate_terms(A_new, B_new)
        result = _perturbed_with_prior(old_S_r, old_U, old_A, old_B, old_sigma, A_deflate, B_deflate)
    else:
        S = anndata_to_matrix(adata, layer=layer, transpose=True)
        if sp.issparse(S):
            result = _core.orthogonalize_batch_effect_sparse(
                S, old_S_r, old_U, old_A, old_B, old_sigma, design
            )
        else:
            S = np.asarray(S, dtype=float, order="C")
            result = _core.orthogonalize_batch_effect_dense(
                S, old_S_r, old_U, old_A, old_B, old_sigma, design
            )

    persist_updates(
        adata,
        obsm={
            corrected_key: result["S_r"].T,  # cells x components
            f"{corrected_key}_B": result["B"],
        },
        varm={
            f"{corrected_key}_U": result["U"],
            f"{corrected_key}_A": result["A"],
        },
        uns={
            f"{corrected_key}_params": {
                "sigma": result["sigma"],
                "batch_key": batch_key,
                "original_reduction": reduction_key,
            }
        },
    )
    return adata


def correct_basal_expression(
    adata: AnnData,
    basal_genes: Union[list, np.ndarray],
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_key: str = "action_basal_corrected",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
) -> AnnData:
    """Correct for basal expression levels by orthogonalizing their effects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed reduction.
    basal_genes : list or ndarray
        Gene names whose basal expression should be removed.
    reduction_key : str
        Key prefix for the existing reduction.
    layer : str or None
        Layer to use.  ``None`` uses ``adata.X``.
    corrected_key : str
        Key for storing corrected results.
    inplace : bool
        Modify *adata* in place or return a copy.
    backed_chunk_size : int
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    """
    if not inplace:
        adata = adata.copy()

    old_S_r, old_U, old_A, old_B, old_sigma = _load_reduction_state(adata, reduction_key)

    basal_genes = np.array(basal_genes)
    gene_mask = np.isin(adata.var_names, basal_genes)
    if gene_mask.sum() == 0:
        raise ValueError("None of the specified basal genes found in adata.var_names")

    basal = np.zeros((adata.n_vars, 1), dtype=float)
    basal[gene_mask, 0] = 1.0
    basal = np.ascontiguousarray(basal)

    source = MatrixSource(adata, layer=layer)
    if source.is_backed:
        A_new, B_new = _streamed_basal_terms(source, basal, chunk_size=backed_chunk_size)
        A_deflate, B_deflate = _deflate_terms(A_new, B_new)
        result = _perturbed_with_prior(old_S_r, old_U, old_A, old_B, old_sigma, A_deflate, B_deflate)
    else:
        S = anndata_to_matrix(adata, layer=layer, transpose=True)
        if sp.issparse(S):
            result = _core.orthogonalize_basal_sparse(
                S, old_S_r, old_U, old_A, old_B, old_sigma, basal
            )
        else:
            S = np.asarray(S, dtype=float, order="C")
            result = _core.orthogonalize_basal_dense(
                S, old_S_r, old_U, old_A, old_B, old_sigma, basal
            )

    persist_updates(
        adata,
        obsm={
            corrected_key: result["S_r"].T,
            f"{corrected_key}_B": result["B"],
        },
        varm={
            f"{corrected_key}_U": result["U"],
            f"{corrected_key}_A": result["A"],
        },
        uns={
            f"{corrected_key}_params": {
                "sigma": result["sigma"],
                "basal_genes": basal_genes[np.isin(basal_genes, adata.var_names)].tolist(),
                "original_reduction": reduction_key,
            }
        },
    )

    return adata
