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
    batch_key: str,
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_key: str = "action_corrected",
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
        Key in adata.obs containing batch labels.
    reduction_key
        Key in adata.obsm containing reduced representation from reduce_kernel().
    layer
        Layer to use for correction (None uses .X).
    corrected_key
        Key to store corrected reduction in adata.obsm.
        
    Returns
    -------
    Updates adata with:
        - adata.obsm[corrected_key]: Batch-corrected reduction
        - adata.uns[f"{corrected_key}_params"]: Correction parameters
    """
    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")
    
    # Get reduction parameters
    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")
    
    params = adata.uns[params_key]
    old_S_r = adata.obsm[reduction_key].T  # Components x cells
    old_V = params["V"]
    old_A = params["A"]
    old_B = params["B"]
    old_sigma = params["sigma"]
    
    # Create design matrix from batch labels
    batch_labels = adata.obs[batch_key]
    if not isinstance(batch_labels.dtype, pd.CategoricalDtype):
        batch_labels = pd.Categorical(batch_labels)
    
    # One-hot encode batches (no intercept)
    n_batches = len(batch_labels.categories)
    design = np.zeros((len(batch_labels), n_batches))
    for i, cat in enumerate(batch_labels.categories):
        design[batch_labels == cat, i] = 1
    
    # Get expression matrix
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    
    # Run orthogonalization
    if sp.issparse(S):
        result = _core.orthogonalize_batch_effect_sparse(
            S, old_S_r, old_V, old_A, old_B, old_sigma, design
        )
    else:
        result = _core.orthogonalize_batch_effect_dense(
            S, old_S_r, old_V, old_A, old_B, old_sigma, design
        )
    
    # Store corrected reduction
    adata.obsm[corrected_key] = result["S_r"].T  # Transpose to cells x components
    adata.uns[f"{corrected_key}_params"] = {
        "V": result["V"],
        "sigma": result["sigma"],
        "A": result["A"],
        "B": result["B"],
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
    
    # Get reduction parameters
    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")
    
    params = adata.uns[params_key]
    old_S_r = adata.obsm[reduction_key].T
    old_V = params["V"]
    old_A = params["A"]
    old_B = params["B"]
    old_sigma = params["sigma"]
    
    # Get expression matrix
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    
    # Create basal matrix (indicator of basal genes)
    basal = np.zeros((adata.n_vars, 1))
    basal[gene_mask, 0] = 1
    
    # Run orthogonalization
    result = _core.orthogonalize_basal_sparse(
        S, old_S_r, old_V, old_A, old_B, old_sigma, basal
    )
    
    # Store corrected reduction
    adata.obsm[corrected_key] = result["S_r"].T
    adata.uns[f"{corrected_key}_params"] = {
        "V": result["V"],
        "sigma": result["sigma"],
        "A": result["A"],
        "B": result["B"],
        "basal_genes": basal_genes[np.isin(basal_genes, adata.var_names)].tolist(),
        "original_reduction": reduction_key,
    }
    
    return adata
