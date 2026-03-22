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


def _backed_group_path(layer: Optional[str]) -> str:
    return "/X" if layer is None else f"/layers/{layer}"


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
        file_path = str(adata.filename)
        group_path = _backed_group_path(layer)
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=backed_chunk_size,
        )
        result = _core.orthogonalize_batch_effect_operator(
            op, old_S_r, old_U, old_A, old_B, old_sigma, design,
        )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
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
            corrected_key: result["S_r"],          # cells x k, direct
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
        file_path = str(adata.filename)
        group_path = _backed_group_path(layer)
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=backed_chunk_size,
        )
        result = _core.orthogonalize_basal_operator(
            op, old_S_r, old_U, old_A, old_B, old_sigma, basal,
        )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
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
            corrected_key: result["S_r"],
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
