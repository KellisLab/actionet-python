"""Gene expression imputation utilities."""

from typing import Optional, Union, List, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from .anndata_utils import anndata_to_matrix


def impute_features(
    adata: AnnData,
    features: Union[List[str], np.ndarray],
    algorithm: Literal["actionet", "pca"] = "actionet",
    features_use: Optional[str] = None,
    network_key: str = "actionet",
    layer: Optional[str] = None,
    reduction_key: str = "action",
    alpha: float = 0.85,
    max_iter: int = 5,
    norm_method: Union[int, Literal["pagerank", "pagerank_sym"]] = "pagerank_sym",
    n_threads: int = 0,
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
    algorithm
        Imputation algorithm to use ('actionet' or 'pca').
    features_use
        Column in adata.var to use for feature matching (if not using var_names directly).
    network_key
        Key in adata.obsp containing cell-cell network.
    layer
        Layer to use (None uses .X).
    reduction_key
        Key in adata.obsm containing reduction to use.
    alpha
        Diffusion parameter (0-1). Higher values = deeper diffusion.
    max_iter
        Number of diffusion iterations.
    norm_method
        Normalization method for network (0=none, 1=symmetric, 2=random walk, "pagerank", "pagerank_sym").
    n_threads
        Number of threads.

    Returns
    -------
    imputed_expr : np.ndarray
        Imputed expression matrix (cells x features).
        If key_added is provided, also stores in adata.layers[key_added].
    """
    if isinstance(norm_method, str):
        norm_method_code = 2 if norm_method == "pagerank_sym" else 0
    else:
        norm_method_code = int(norm_method)

    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found. Run build_network first.")

    features = np.array(features)
    if features_use is None:
        feature_labels = adata.var_names.to_numpy()
    else:
        if features_use not in adata.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        feature_labels = adata.var[features_use].to_numpy()

    feature_mask = np.isin(feature_labels, features)

    if feature_mask.sum() == 0:
        raise ValueError("None of the specified features found in adata.var_names")

    feature_indices = np.where(feature_mask)[0]

    S = anndata_to_matrix(adata, layer=layer, transpose=True)
    X0 = S[feature_indices, :]

    if sp.issparse(X0):
        X0 = X0.toarray()

    G = adata.obsp[network_key]

    if algorithm == "pca":
        smooth_out = smooth_kernel(
            adata,
            network_key=network_key,
            reduction_key=reduction_key,
            alpha=alpha,
            max_iter=max_iter,
            norm_method=norm_method_code,
            n_threads=n_threads,
            return_raw=True,
        )
        W = smooth_out["SVD_out"]["u"][feature_indices, :]
        H = smooth_out["H"]
        X_imputed = (W @ H.T).T
    else:
        X0_T = X0.T
        X_imputed = _core.compute_network_diffusion(
            G, X0_T, alpha, max_iter, n_threads, True, norm_method_code, 1e-8
        )

    original_max = X0.max(axis=1)
    imputed_max = X_imputed.max(axis=0)

    scale_factors = np.where(imputed_max > 0, original_max / imputed_max, 1.0)
    X_imputed = X_imputed * scale_factors[np.newaxis, :]

    X_imputed = np.maximum(X_imputed, 0)

    return X_imputed


def impute_from_archetypes(
    adata: AnnData,
    features: Union[List[str], np.ndarray],
    features_use: Optional[str] = None,
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
    features_use
        Column in adata.var to use for feature matching (if not using var_names directly).
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

    features = np.array(features)
    if features_use is None:
        feature_labels = adata.var_names.to_numpy()
    else:
        if features_use not in adata.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        feature_labels = adata.var[features_use].to_numpy()

    feature_mask = np.isin(feature_labels, features)

    if feature_mask.sum() == 0:
        raise ValueError("None of the specified features found in adata.var_names")

    Z = adata.varm[archetype_profile_key][feature_mask, :]
    H = adata.obsm[archetype_matrix_key]

    imputed_expr = H @ Z.T
    imputed_expr = np.maximum(imputed_expr, 0)

    return imputed_expr


def smooth_kernel(
    adata: AnnData,
    network_key: str = "actionet",
    reduction_key: str = "action",
    alpha: float = 0.85,
    max_iter: int = 5,
    norm_method: Union[int, Literal["pagerank", "pagerank_sym"]] = "pagerank",
    n_threads: int = 0,
    key_added: str = "action_smoothed",
    return_raw: bool = False,
) -> Union[AnnData, dict]:
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
    return_raw
        If True, return raw diffusion outputs instead of updating adata.

    Returns
    -------
    Updates adata with:
        - adata.obsm[key_added]: Smoothed reduction
    Or, if return_raw=True:
        - Dictionary with raw outputs from diffusion and SVD.
    """
    if isinstance(norm_method, str):
        norm_method_code = 2 if norm_method == "pagerank_sym" else 0
    else:
        norm_method_code = int(norm_method)

    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")

    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found.")

    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")

    G = adata.obsp[network_key]
    S_r = np.asarray(adata.obsm[reduction_key], dtype=float, order="C")
    sigma = np.asarray(adata.uns[params_key]["sigma"], dtype=float).reshape(-1)

    V = np.asarray(adata.varm[f"{reduction_key}_V"], dtype=float, order="C")
    A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")

    if sigma.shape[0] != S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    U = S_r / sigma[np.newaxis, :]
    svd_out = _core.perturbed_svd(V, sigma, U, -A, B)

    V_smooth = _core.compute_network_diffusion(
        G, svd_out["v"], alpha, max_iter, n_threads, True, norm_method_code, 1e-8
    )
    H = V_smooth @ np.diag(svd_out["d"])

    if return_raw:
        return {
            "U": U,
            "SVD_out": svd_out,
            "V_smooth": V_smooth,
            "H": H,
        }

    adata.obsm[key_added] = H
    return adata
