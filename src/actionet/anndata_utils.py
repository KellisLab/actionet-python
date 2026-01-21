"""Utilities for translating between AnnData and C++ data structures."""

from typing import Optional, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def anndata_to_matrix(
    adata: AnnData,
    layer: Optional[str] = None,
    transpose: bool = False,
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Extract matrix from AnnData for C++ input.
    
    Parameters
    ----------
    adata
        AnnData object.
    layer
        Layer name (None uses .X).
    transpose
        Transpose to genes x cells format.
        
    Returns
    -------
    Matrix (sparse or dense).
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    
    if transpose:
        if sp.issparse(X):
            X = X.T.tocsr()
        else:
            X = X.T
    
    return X


def matrix_to_anndata(
    X: Union[np.ndarray, sp.spmatrix],
    obs_names: Optional[np.ndarray] = None,
    var_names: Optional[np.ndarray] = None,
) -> AnnData:
    """
    Create AnnData from matrix.
    
    Parameters
    ----------
    X
        Expression matrix (cells x genes).
    obs_names
        Cell names.
    var_names
        Gene names.
        
    Returns
    -------
    AnnData object.
    """
    
    adata = AnnData(X)
    
    if obs_names is not None:
        adata.obs_names = obs_names
    if var_names is not None:
        adata.var_names = var_names
    
    return adata


def add_action_results(
    adata: AnnData,
    result: dict,
) -> None:
    """
    Add ACTION decomposition results to AnnData.
    
    Parameters
    ----------
    adata
        AnnData object to update in-place.
    result
        Result dictionary from run_action().
    """
    # Store archetype matrices
    adata.obsm["H_stacked"] = result["H_stacked"].T
    adata.obsm["H_merged"] = result["H_merged"].T

    adata.obsm["C_stacked"] = result["C_stacked"]
    adata.obsm["C_merged"] = result["C_merged"]

    # Store archetype assignments
    adata.obs["assigned_archetype"] = result["assigned_archetypes"]


def add_network_to_anndata(
    adata: AnnData,
    network: sp.spmatrix,
    key: str = "actionet",
) -> None:
    """
    Add network to AnnData.obsp.
    
    Parameters
    ----------
    adata
        AnnData object.
    network
        Sparse adjacency matrix.
    key
        Key in adata.obsp.
    """
    adata.obsp[key] = network
