"""Batch correction and normalization utilities."""

from typing import Optional, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import pandas as pd

from . import _core
from .anndata_utils import anndata_to_matrix


def correct_batch_effect(
    adata: AnnData,
    batch_key: Optional[str] = None,
    design: Optional[np.ndarray] = None,
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_suffix: str = "corrected",
    inplace: bool = True,
) -> AnnData:
    """
    Correct batch effects using orthogonalization.
    
    This function removes batch effects from a reduced representation by
    orthogonalizing the batch covariate effects from the data.
    
    Parameters
    ----------
    adata
        Annotated data matrix with reduced representation.
    batch_key
        Key in adata.obs containing batch labels. Mutually exclusive with design.
    design
        Design matrix (cells x covariates). Mutually exclusive with batch_key.
    reduction_key
        Key in adata.obsm containing reduced representation from reduce_kernel().
    layer
        Layer to use for correction (None uses .X).
    corrected_suffix
        Suffix for corrected key.
    inplace
        If True, modifies adata in place. If False, returns a new AnnData.

    Returns
    -------
    Updates adata with:
        - adata.obsm[corrected_key]: Batch-corrected reduction
        - adata.uns[f"{corrected_key}_params"]: Correction parameters
    """
    corrected_key = f"{reduction_key}_{corrected_suffix}"

    if (batch_key is None) == (design is None):
        raise ValueError("Provide exactly one of 'batch_key' or 'design'.")

    if not inplace:
        adata = adata.copy()

    if batch_key is not None and batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")
    
    # Get reduction parameters and matrices
    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")
    
    params = adata.uns[params_key]
    old_S_r = np.asarray(adata.obsm[reduction_key], dtype=float, order="C")  # Cells x components
    old_V = np.asarray(adata.varm[f"{reduction_key}_V"], dtype=float, order="C")
    old_A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    old_B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")
    old_sigma = np.asarray(params["sigma"], dtype=float).reshape(-1)

    if old_S_r.shape[0] != adata.n_obs:
        raise ValueError("Reduction matrix has unexpected shape; expected cells x components.")
    if old_sigma.shape[0] != old_S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    # Create design matrix from batch labels or use provided design
    if batch_key is not None:
        batch_labels = pd.Series(adata.obs[batch_key].to_numpy(), index=adata.obs_names).astype("category")
        if len(batch_labels.cat.categories) < 2:
            raise ValueError("'batch_key' must have at least 2 categories for correction.")

        # One-hot encode batches (no intercept)
        design = pd.get_dummies(batch_labels, drop_first=False).to_numpy(dtype=float)
    else:
        design = np.asarray(design, dtype=float)
        if design.ndim != 2:
            raise ValueError("'design' must be a 2D matrix (cells x covariates).")

    if design.shape[0] != adata.n_obs:
        raise ValueError("Design matrix does not match number of observations.")
    design = np.ascontiguousarray(design)

    # Get expression matrix
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    
    # Run orthogonalization
    if sp.issparse(S):
        result = _core.orthogonalize_batch_effect_sparse(
            S, old_S_r, old_V, old_A, old_B, old_sigma, design
        )
    else:
        S = np.asarray(S, dtype=float, order="C")
        result = _core.orthogonalize_batch_effect_dense(
            S, old_S_r, old_V, old_A, old_B, old_sigma, design
        )
    
    # Store corrected reduction
    adata.obsm[corrected_key] = result["S_r"].T  # Transpose to cells x components
    adata.varm[f"{corrected_key}_V"] = result["V"]
    adata.varm[f"{corrected_key}_A"] = result["A"]
    adata.obsm[f"{corrected_key}_B"] = result["B"]
    adata.uns[f"{corrected_key}_params"] = {
        "sigma": result["sigma"],
        "batch_key": batch_key,
        "original_reduction": reduction_key,
    }
    
    return adata


def correct_basal_expression(
    adata: AnnData,
    basal_genes: Union[list, np.ndarray],
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_key: str = "action_basal_corrected",
) -> AnnData:
    """
    Correct for basal expression levels by orthogonalizing their effects.

    This function removes the effect of specified basal/housekeeping genes
    from the reduced representation.

    Parameters
    ----------
    adata
        Annotated data matrix with reduced representation.
    basal_genes
        List of gene names representing basal expression.
    reduction_key
        Key in adata.obsm containing reduced representation.
    layer
        Layer to use for correction (None uses .X).
    corrected_key
        Key to store corrected reduction.

    Returns
    -------
    Updates adata with:
        - adata.obsm[corrected_key]: Basal-corrected reduction
        - adata.uns[f"{corrected_key}_params"]: Correction parameters
    """
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    # Find basal genes in the data
    basal_genes = np.array(basal_genes)
    gene_mask = np.isin(adata.var_names, basal_genes)

    if gene_mask.sum() == 0:
        raise ValueError("None of the specified basal genes found in adata.var_names")

    # Get reduction parameters and matrices
    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")

    params = adata.uns[params_key]
    old_S_r = np.asarray(adata.obsm[reduction_key], dtype=float, order="C")
    old_V = np.asarray(adata.varm[f"{reduction_key}_V"], dtype=float, order="C")
    old_A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    old_B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")
    old_sigma = np.asarray(params["sigma"], dtype=float).reshape(-1)

    if old_S_r.shape[0] != adata.n_obs:
        raise ValueError("Reduction matrix has unexpected shape; expected cells x components.")
    if old_sigma.shape[0] != old_S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    # Get expression matrix
    S = anndata_to_matrix(adata, layer=layer, transpose=True)

    # Create basal matrix (indicator of basal genes)
    basal = np.zeros((adata.n_vars, 1), dtype=float)
    basal[gene_mask, 0] = 1.0
    basal = np.ascontiguousarray(basal)

    # Run orthogonalization
    if sp.issparse(S):
        result = _core.orthogonalize_basal_sparse(
            S, old_S_r, old_V, old_A, old_B, old_sigma, basal
        )
    else:
        S = np.asarray(S, dtype=float, order="C")
        result = _core.orthogonalize_basal_dense(
            S, old_S_r, old_V, old_A, old_B, old_sigma, basal
        )

    # Store corrected reduction
    adata.obsm[corrected_key] = result["S_r"].T
    adata.varm[f"{corrected_key}_V"] = result["V"]
    adata.varm[f"{corrected_key}_A"] = result["A"]
    adata.obsm[f"{corrected_key}_B"] = result["B"]
    adata.uns[f"{corrected_key}_params"] = {
        "sigma": result["sigma"],
        "basal_genes": basal_genes[np.isin(basal_genes, adata.var_names)].tolist(),
        "original_reduction": reduction_key,
    }

    return adata
