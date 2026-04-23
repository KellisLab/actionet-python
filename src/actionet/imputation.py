"""Gene expression imputation utilities."""

from typing import Optional, Union, List, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import pandas as pd
from . import _core
from .anndata_utils import anndata_to_matrix
from ._matrix_source import MatrixSource
from .backed_io import _backed_group_path, _open_backed_operator
from .lazy_transform import (
    LazyTransform,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)
from .reduction import smooth_kernel


def impute_features(
    adata: AnnData,
    features: Union[List[str], np.ndarray],
    method: Literal["diffusion", "pca", "archetypes"] = "diffusion",
    features_use: Optional[str] = None,
    # --- network-diffusion params (method="diffusion" or "pca") ---
    network_key: str = "actionet",
    layer: Optional[str] = None,
    reduction_key: str = "action",
    alpha: float = 0.85,
    max_iter: int = 5,
    norm_method: Union[int, Literal["pagerank", "pagerank_sym"]] = "pagerank_sym",
    # --- archetype params (method="archetypes") ---
    archetype_profile_key: str = "archetype_feat_profile",
    archetype_matrix_key: str = "H_merged",
    # --- common ---
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
    lazy_transform: Optional[LazyTransform] = None,
) -> pd.DataFrame:
    """Impute gene expression values for a set of features.

    Three imputation methods are available, selected via ``method``:

    - ``"diffusion"`` (default) — network diffusion over the ACTIONet
      cell-cell graph.  Smooths expression to reduce dropout noise.
    - ``"pca"`` — PCA-based smoothing via the ACTIONet kernel.
    - ``"archetypes"`` — fast archetype interpolation: each cell's imputed
      expression is a weighted combination of per-archetype average profiles.
      Requires pre-computed profiles in ``adata.varm`` (see
      :func:`~actionet.specificity.compute_archetype_feature_specificity`).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with ACTIONet results.
    features : list of str or np.ndarray
        Feature names to impute.
    method : {"diffusion", "pca", "archetypes"}, default ``"diffusion"``
        Imputation method.
    features_use : str, optional
        Column in ``adata.var`` to use for feature matching.  If ``None``,
        ``adata.var_names`` is used directly.
    network_key : str, default ``"actionet"``
        Key in ``adata.obsp`` containing the cell-cell network.
        Used by ``method="diffusion"`` and ``method="pca"``.
    layer : str, optional
        Layer of ``adata`` to use as the expression matrix.  If ``None``,
        ``adata.X`` is used.  Used by ``method="diffusion"`` and ``method="pca"``.
    reduction_key : str, default ``"action"``
        Key in ``adata.obsm`` containing the reduction.
        Used only by ``method="pca"``.
    alpha : float, default 0.85
        Diffusion parameter (0–1).  Higher values produce deeper smoothing.
        Used by ``method="diffusion"`` and ``method="pca"``.
    max_iter : int, default 5
        Number of diffusion iterations.
        Used by ``method="diffusion"`` and ``method="pca"``.
    norm_method : int or {"pagerank", "pagerank_sym"}, default ``"pagerank_sym"``
        Network normalization method.
        Used by ``method="diffusion"`` and ``method="pca"``.
    archetype_profile_key : str, default ``"archetype_feat_profile"``
        Key in ``adata.varm`` containing per-archetype average expression
        profiles (genes x archetypes).  Written by
        :func:`~actionet.specificity.compute_archetype_feature_specificity`
        with the default ``key_added="archetype"``.
        Used only by ``method="archetypes"``.
    archetype_matrix_key : str, default ``"H_merged"``
        Key in ``adata.obsm`` containing the cell-archetype membership matrix
        (cells x archetypes) used as imputation weights.
        Used only by ``method="archetypes"``.
    n_threads : int, default 0
        Number of threads for the C++ backend.  ``0`` lets the backend choose.
        Used by ``method="diffusion"`` and ``method="pca"``.
    backed_chunk_size : int, default 4096
        Number of rows per chunk when streaming a backed (HDF5-on-disk) AnnData.
        Ignored for in-memory objects.
        Used by ``method="diffusion"`` and ``method="pca"``.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        When provided, the backed operator applies per-row normalization and
        log scaling on-the-fly without requiring a persisted ``logcounts`` layer.
        Only valid when ``layer=None`` and the input is backed.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.
        Used by ``method="diffusion"`` and ``method="pca"``.

    Returns
    -------
    imputed_expr : pd.DataFrame
        Imputed expression matrix (cells x features) with ``adata.obs_names``
        as the index and matched feature names as columns.

    See Also
    --------
    actionet.specificity.compute_archetype_feature_specificity :
        Computes the per-archetype profiles required for ``method="archetypes"``.
    """
    from ._feature_lookup import resolve_feature_space, resolve_requested_features
    space = resolve_feature_space(adata, features_use, context="impute_features")
    resolved = resolve_requested_features(features, space, context="impute_features")

    if len(resolved.matched_names) == 0:
        raise ValueError("None of the specified features found in adata.var_names")

    matched_features = resolved.matched_names
    feature_indices = resolved.matched_indices

    # ------------------------------------------------------------------
    # Archetype interpolation path — pure lookup + matmul, no expression
    # matrix access needed.
    # ------------------------------------------------------------------
    if method == "archetypes":
        if archetype_matrix_key not in adata.obsm:
            raise ValueError(
                f"Archetype matrix '{archetype_matrix_key}' not found in adata.obsm. "
                "Run run_action() first."
            )
        if archetype_profile_key not in adata.varm:
            raise ValueError(
                f"Archetype profiles '{archetype_profile_key}' not found in adata.varm. "
                "Run compute_archetype_feature_specificity(adata) first."
            )
        print(
            f"impute_features: method=archetypes | "
            f"profile=varm['{archetype_profile_key}'] | "
            f"weights=obsm['{archetype_matrix_key}']"
        )
        feat_profiles = adata.varm[archetype_profile_key]  # genes x archetypes
        Z = feat_profiles[feature_indices, :]               # matched_features x archetypes
        H = adata.obsm[archetype_matrix_key]                # cells x archetypes
        imputed_expr = np.maximum(H @ Z.T, 0)               # cells x matched_features
        return pd.DataFrame(imputed_expr, index=adata.obs_names, columns=matched_features)

    # ------------------------------------------------------------------
    # Network-diffusion paths ("diffusion" and "pca")
    # ------------------------------------------------------------------
    if isinstance(norm_method, str):
        norm_method_code = 2 if norm_method == "pagerank_sym" else 0
    else:
        norm_method_code = int(norm_method)

    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found. Run build_network first.")

    print(
        f"impute_features: method={method} | "
        f"alpha={alpha} | network=obsp['{network_key}']"
        + (f" | layer='{layer}'" if layer is not None else "")
    )

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
        with _open_backed_operator(
            adata=adata,
            file_path=file_path,
            group_path=group_path,
            context="impute_features",
            chunk_size=backed_chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
            n_threads=n_threads,
        ) as op:
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

    if method == "pca":
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
