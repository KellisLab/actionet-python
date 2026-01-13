"""Gene expression imputation utilities."""

from typing import Optional, Union, List
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from .anndata_utils import anndata_to_matrix


def impute_features(
    adata: AnnData,
    features: Union[List[str], np.ndarray],
    network_key: str = "actionet",
    layer: Optional[str] = None,
    alpha: float = 0.85,
    max_iter: int = 5,
    norm_method: int = 2,
    n_threads: int = 0,
    key_added: str = "imputed",
) -> np.ndarray:
    """
    Impute gene expression using network diffusion.
    
    This function smooths gene expression over the ACTIONet cell-cell network
    to impute dropout events and reduce noise.
    
    Parameters
    ----------
    adata
        Annotated data matrix with ACTIONet network.
    features
        List of gene names to impute.
    network_key
        Key in adata.obsp containing cell-cell network.
    layer
        Layer to use (None uses .X).
    alpha
        Diffusion parameter (0-1). Higher values = deeper diffusion.
    max_iter
        Number of diffusion iterations.
    norm_method
        Normalization method for network (0=none, 1=symmetric, 2=random walk).
    n_threads
        Number of threads.
    key_added
        Optional key to store results in adata.layers.
        
    Returns
    -------
    imputed_expr : np.ndarray
        Imputed expression matrix (cells x features).
        If key_added is provided, also stores in adata.layers[key_added].
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found. Run build_network first.")
    
    # Find features in data
    features = np.array(features)
    feature_mask = np.isin(adata.var_names, features)
    
    if feature_mask.sum() == 0:
        raise ValueError("None of the specified features found in adata.var_names")
    
    matched_features = adata.var_names[feature_mask]
    feature_indices = np.where(feature_mask)[0]
    
    # Get expression matrix for selected features
    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    X0 = S[feature_indices, :]
    
    if sp.issparse(X0):
        X0 = X0.toarray()
    
    # Get network
    G = adata.obsp[network_key]
    
    # Transpose to cells x features for diffusion
    X0_T = X0.T
    
    # Run network diffusion
    X_imputed = _core.compute_network_diffusion(
        G, X0_T, alpha, max_iter, n_threads, True, norm_method, 1e-8
    )
    
    # Rescale imputed values to match original max expression
    original_max = X0.max(axis=1)
    imputed_max = X_imputed.max(axis=0)
    
    # Avoid division by zero
    scale_factors = np.where(imputed_max > 0, original_max / imputed_max, 1.0)
    X_imputed = X_imputed * scale_factors[np.newaxis, :]
    
    # Ensure non-negative
    X_imputed = np.maximum(X_imputed, 0)
    
    # Optionally store in adata
    if key_added:
        if key_added not in adata.layers:
            adata.layers[key_added] = np.zeros_like(adata.X)
        
        # Update only the imputed features
        for i, gene_idx in enumerate(feature_indices):
            adata.layers[key_added][:, gene_idx] = X_imputed[:, i]
    
    return X_imputed


def impute_from_archetypes(
    adata: AnnData,
    features: Union[List[str], np.ndarray],
    archetype_profile_key: str = "specificity_profile",
    archetype_matrix_key: str = "H_merged",
) -> np.ndarray:
    """
    Impute gene expression by interpolating over archetype profiles.
    
    This method uses the learned archetype-gene associations to impute
    expression based on cell-archetype memberships.
    
    Parameters
    ----------
    adata
        Annotated data matrix with ACTION results.
    features
        List of gene names to impute.
    archetype_profile_key
        Key in adata.varm containing gene-archetype profiles.
    archetype_matrix_key
        Key in adata.obsm containing cell-archetype matrix (H).
        
    Returns
    -------
    imputed_expr : np.ndarray
        Imputed expression matrix (cells x features).
    """
    if archetype_matrix_key not in adata.obsm:
        raise ValueError(f"Archetype matrix '{archetype_matrix_key}' not found. Run run_action first.")
    
    if archetype_profile_key not in adata.varm:
        raise ValueError(f"Archetype profiles '{archetype_profile_key}' not found. Run compute_feature_specificity first.")
    
    # Find features in data
    features = np.array(features)
    feature_mask = np.isin(adata.var_names, features)
    
    if feature_mask.sum() == 0:
        raise ValueError("None of the specified features found in adata.var_names")
    
    matched_features = adata.var_names[feature_mask]
    
    # Get archetype profiles for selected features (genes x archetypes)
    Z = adata.varm[archetype_profile_key][feature_mask, :]
    
    # Get cell-archetype matrix (cells x archetypes)
    H = adata.obsm[archetype_matrix_key]
    
    # Impute: cells x genes = (cells x archetypes) @ (archetypes x genes).T
    imputed_expr = H @ Z.T
    
    # Ensure non-negative
    imputed_expr = np.maximum(imputed_expr, 0)
    
    return imputed_expr


def smooth_kernel(
    adata: AnnData,
    network_key: str = "actionet",
    reduction_key: str = "action",
    alpha: float = 0.85,
    max_iter: int = 5,
    norm_method: int = 2,
    n_threads: int = 0,
    key_added: str = "action_smoothed",
) -> AnnData:
    """
    Smooth the reduced kernel over the network.
    
    This function applies network diffusion to the reduced representation,
    which can improve downstream analysis by leveraging local structure.
    
    Parameters
    ----------
    adata
        Annotated data matrix with network and reduction.
    network_key
        Key in adata.obsp containing network.
    reduction_key
        Key in adata.obsm containing reduction to smooth.
    alpha
        Diffusion parameter.
    max_iter
        Number of iterations.
    norm_method
        Normalization method.
    n_threads
        Number of threads.
    key_added
        Key to store smoothed reduction.
        
    Returns
    -------
    Updates adata with:
        - adata.obsm[key_added]: Smoothed reduction
    """
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")
    
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found.")
    
    G = adata.obsp[network_key]
    X0 = adata.obsm[reduction_key]
    
    # Run diffusion
    X_smoothed = _core.compute_network_diffusion(
        G, X0, alpha, max_iter, n_threads, False, norm_method, 1e-8
    )
    
    adata.obsm[key_added] = X_smoothed
    return adata
