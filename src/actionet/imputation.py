"""Gene expression imputation utilities."""

from typing import Optional, Union, List, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import pandas as pd
import warnings

from . import _core
from .anndata_utils import anndata_to_matrix
from ._matrix_source import MatrixSource
from ._backed_persist import persist_updates
from .backed_io import _backed_group_path
from .lazy_transform import (
    LazyTransform,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)


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
    backed_chunk_size: int = 4096,
    lazy_transform: Optional[LazyTransform] = None,
) -> pd.DataFrame:
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
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        When provided, the backed operator applies per-row normalization
        and log1p on-the-fly without requiring a persisted ``logcounts``
        layer.  Only valid when ``layer=None`` and the input is backed.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.

    Returns
    -------
    imputed_expr : pd.DataFrame
        Imputed expression matrix (cells x features) with adata.obs_names as index
        and the input features as columns.
    """
    if isinstance(norm_method, str):
        norm_method_code = 2 if norm_method == "pagerank_sym" else 0
    else:
        norm_method_code = int(norm_method)

    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found. Run build_network first.")

    from ._feature_lookup import resolve_feature_space, resolve_requested_features
    space = resolve_feature_space(adata, features_use, context="impute_features")
    resolved = resolve_requested_features(features, space, context="impute_features")

    if len(resolved.matched_names) == 0:
        raise ValueError("None of the specified features found in adata.var_names")

    matched_features = resolved.matched_names
    feature_indices = resolved.matched_indices

    source = MatrixSource(adata, layer=layer)
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    if source.is_backed:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        file_path = str(adata.filename)
        group_path = _backed_group_path(layer)
        op = _core.create_backed_operator(
            file_path=file_path,
            group_path=group_path,
            chunk_size=backed_chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
            n_threads=n_threads,
        )
        X0 = np.asarray(
            _core.backed_take_columns(op, feature_indices, prefer_sparse=False),
            dtype=np.float64,
        )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        X0 = S[:, feature_indices]
        if sp.issparse(X0):
            X0 = X0.toarray()
        X0 = np.asarray(X0, dtype=np.float64)  # cells x features

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
        X_imputed = _core.compute_network_diffusion(
            G, X0, alpha, max_iter, n_threads, True, norm_method_code, 1e-8
        )

    original_max = X0.max(axis=0)
    imputed_max = X_imputed.max(axis=0)

    scale_factors = np.where(imputed_max > 0, original_max / imputed_max, 1.0)
    X_imputed = X_imputed * scale_factors[np.newaxis, :]

    X_imputed = np.maximum(X_imputed, 0)

    return pd.DataFrame(X_imputed, index=adata.obs_names, columns=matched_features)


def impute_features_from_archetypes(
    adata: AnnData,
    features: Union[List[str], np.ndarray],
    features_use: Optional[str] = None,
    archetype_profile_key: Optional[str] = None,
    archetype_matrix_key: str = "H_merged",
    archetype_key: str = "archetype_footprint",
    layer: Optional[str] = None,
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
    lazy_transform: Optional[LazyTransform] = None,
) -> pd.DataFrame:
    """Impute gene expression by interpolating over archetype profiles.

    Each cell's imputed expression is a weighted combination of per-archetype
    average expression profiles, where the weights come from the cell's
    archetype membership vector (``adata.obsm[archetype_matrix_key]``).

    If ``archetype_profile_key`` is ``None`` (the default),
    :func:`~actionet.specificity.compute_archetype_feature_specificity` is
    called automatically with ``return_raw=True`` to compute the profiles
    on-the-fly without modifying ``adata``.  The ``"archetypes"`` field of the
    returned dict (genes x archetypes average expression profiles) is used
    directly.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with ACTION results.  Must contain the archetype
        membership matrix in ``adata.obsm[archetype_matrix_key]``.
    features : list of str or np.ndarray
        Gene names to impute.
    features_use : str, optional
        Column in ``adata.var`` to use for feature matching.  If ``None``,
        ``adata.var_names`` is used directly.
    archetype_profile_key : str, optional
        Key in ``adata.varm`` containing pre-computed gene-archetype average
        expression profiles (genes x archetypes).  The default stored key from
        :func:`~actionet.specificity.compute_archetype_feature_specificity` is
        ``"archetype_feat_profile"`` (i.e. ``key_added="archetype"``).
        When ``None`` (default), profiles are computed on-the-fly via
        :func:`~actionet.specificity.compute_archetype_feature_specificity`.
    archetype_matrix_key : str, default ``"H_merged"``
        Key in ``adata.obsm`` containing the cell-archetype membership matrix
        (cells x archetypes) used as imputation weights.
    archetype_key : str or np.ndarray, default ``"archetype_footprint"``
        Key in ``adata.obsm`` holding the archetype membership matrix, or the
        matrix itself, passed to
        :func:`~actionet.specificity.compute_archetype_feature_specificity`
        when profiles are computed on-the-fly.  This is the matrix used to
        *score* feature specificity per archetype and may differ from
        ``archetype_matrix_key``.  Ignored if ``archetype_profile_key`` is
        provided.
    layer : str, optional
        Layer of ``adata`` to use as the expression matrix when computing
        profiles on-the-fly.  If ``None``, ``adata.X`` is used.  Ignored if
        ``archetype_profile_key`` is provided.
    n_threads : int, default 0
        Number of threads for the C++ backend when computing profiles
        on-the-fly.  ``0`` lets the backend choose.  Ignored if
        ``archetype_profile_key`` is provided.
    backed_chunk_size : int, default 4096
        Row chunk size used when streaming a backed (HDF5-on-disk) AnnData
        during on-the-fly profile computation.  Ignored if
        ``archetype_profile_key`` is provided.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform forwarded to
        :func:`~actionet.specificity.compute_archetype_feature_specificity`
        when profiles are computed on-the-fly.  Only valid when ``layer=None``
        and the input is backed.  Ignored if ``archetype_profile_key`` is
        provided.

    Returns
    -------
    imputed_expr : pd.DataFrame
        Imputed expression matrix (cells x features) with ``adata.obs_names``
        as the index and matched feature names as columns, clipped to
        non-negative values.  Columns correspond to the subset of ``features``
        found in ``adata.var_names`` (or ``features_use``), preserving the
        order in ``adata.var``.

    See Also
    --------
    actionet.specificity.compute_archetype_feature_specificity :
        Computes the per-archetype average expression profiles used here.
    """
    if archetype_matrix_key not in adata.obsm:
        raise ValueError(f"Archetype matrix '{archetype_matrix_key}' not found. Run run_action first.")

    if archetype_profile_key is None:
        from .specificity import compute_archetype_feature_specificity

        print(
            f"archetype_profile_key not provided — computing archetype feature "
            f"specificity on-the-fly from adata.obsm['{archetype_key}']."
        )
        raw = compute_archetype_feature_specificity(
            adata,
            archetype_key=archetype_key,
            layer=layer,
            n_threads=n_threads,
            backed_chunk_size=backed_chunk_size,
            return_raw=True,
            lazy_transform=lazy_transform,
        )
        feat_profiles = raw["archetypes"]  # genes x archetypes
    else:
        if archetype_profile_key not in adata.varm:
            raise ValueError(
                f"Archetype profiles '{archetype_profile_key}' not found in adata.varm. "
                "Run compute_archetype_feature_specificity first or leave "
                "archetype_profile_key=None to compute on-the-fly."
            )
        feat_profiles = adata.varm[archetype_profile_key]  # genes x archetypes

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

    matched_features = adata.var_names[feature_mask] if features_use is None else adata.var[features_use][feature_mask]

    Z = feat_profiles[feature_mask, :]  # (n_features_matched, archetypes)
    H = adata.obsm[archetype_matrix_key]  # (cells, archetypes)

    imputed_expr = H @ Z.T  # (cells, n_features_matched)
    imputed_expr = np.maximum(imputed_expr, 0)

    return pd.DataFrame(imputed_expr, index=adata.obs_names, columns=matched_features)


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

    U_left = np.asarray(adata.varm[f"{reduction_key}_U"], dtype=float, order="C")
    A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")

    if sigma.shape[0] != S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    V_right = S_r / sigma[np.newaxis, :]
    svd_out = _core.perturbed_svd(U_left, sigma, V_right, -A, B)

    V_smooth = _core.compute_network_diffusion(
        G, svd_out["v"], alpha, max_iter, n_threads, True, norm_method_code, 1e-8
    )
    H = V_smooth @ np.diag(svd_out["d"])

    if return_raw:
        return {
            "U": U_left,
            "SVD_out": svd_out,
            "V_smooth": V_smooth,
            "H": H,
        }

    persist_updates(adata, obsm={key_added: H})
    return adata
