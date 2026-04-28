"""Feature expression plotting utilities for ACTIONet."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ..anndata_utils import anndata_to_matrix
from ..imputation import impute_features
from .._matrix_source import MatrixSource
from ..backed_io import _backed_group_path, _open_backed_operator
from ..lazy_transform import (
    LazyTransform,
    _resolve_lazy_backed_transform,
)
from .. import _core
from .umap import (
    _prepare_umap_context,
    _render_umap_raster,
    plot_umap,
    plot_umap_raster,
)


def _flatten_features(features: Union[str, Sequence[Union[str, Iterable[str]]]]) -> list[str]:
    if isinstance(features, str):
        return [features]
    out: list[str] = []
    for item in features:
        if isinstance(item, str):
            out.append(item)
        else:
            out.extend([str(val) for val in item])
    return out


def _resolve_feature_labels(adata: AnnData, features_use: Optional[str]) -> np.ndarray:
    if features_use is None:
        return adata.var_names.to_numpy()
    if features_use not in adata.var.columns:
        raise ValueError(f"Column '{features_use}' not found in adata.var")
    return adata.var[features_use].to_numpy()


def _select_features(
    adata: AnnData,
    features: Union[str, Sequence[Union[str, Iterable[str]]]],
    features_use: Optional[str],
    sort_features: bool,
) -> list[str]:
    feature_labels = _resolve_feature_labels(adata, features_use)
    feature_set = set(feature_labels.tolist())
    flat = _flatten_features(features)
    unique = list(dict.fromkeys(flat))
    matched = [feat for feat in unique if feat in feature_set]
    if sort_features:
        matched = sorted(matched)
    return matched


def _extract_expression(
    adata: AnnData,
    features: Sequence[str],
    features_use: Optional[str],
    layer: Optional[str],
    lazy_transform: Optional[LazyTransform] = None,
    backed_chunk_size: int = 4096,
) -> pd.DataFrame:
    feature_labels = _resolve_feature_labels(adata, features_use)
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_labels)}
    feature_indices = np.array([feature_to_idx[feat] for feat in features], dtype=np.int64)

    source = MatrixSource(adata, layer=layer)
    if source.is_backed:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        with _open_backed_operator(
            adata=adata,
            file_path=str(adata.filename),
            group_path=_backed_group_path(layer),
            context="plot_feature_expression",
            chunk_size=backed_chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
        ) as op:
            expr = np.asarray(
                _core.backed_take_columns(op, feature_indices, prefer_sparse=False),
                dtype=np.float64,
            )
    else:
        matrix = anndata_to_matrix(adata, layer=layer, transpose=True)
        expr = matrix[feature_indices, :]
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr).T
    return pd.DataFrame(expr, index=adata.obs_names, columns=features)


def _grid_shape(n_plots: int) -> tuple[int, int]:
    ncol = int(np.ceil(np.sqrt(n_plots)))
    nrow = int(np.ceil(n_plots / ncol))
    return nrow, ncol


def plot_feature_expression(
    adata: AnnData,
    features: Union[str, Sequence[Union[str, Iterable[str]]]],
    features_use: Optional[str] = None,
    method: Literal["diffusion", "pca", "archetypes", "none"] = "diffusion",
    alpha: float = 0.9,
    layer: Optional[str] = None,
    trans_attr: Optional[Union[str, Sequence, np.ndarray]] = None,
    trans_th: float = -0.5,
    trans_fac: float = 3.0,
    cmap: Union[str, Sequence[str]] = "magma",
    size: float = 1.0,
    network_key: str = "actionet",
    basis: str = "umap_2d_actionet",
    single_plot: bool = False,
    legend: bool = False,
    sort_features: bool = True,
    archetype_profile_key: str = "archetype_feat_profile",
    archetype_matrix_key: str = "H_merged",
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
    lazy_transform: Optional[LazyTransform] = None,
) -> Union[Any, dict[str, Any]]:
    """Plot feature expression values on the UMAP embedding.

    Parameters
    ----------
    adata
        AnnData with expression data and embedding in ``adata.obsm[basis]``.
    features
        Feature name or collection of features to plot.
    features_use
        Column in ``adata.var`` to use for feature matching.
    method : {"diffusion", "pca", "archetypes", "none"}, default ``"diffusion"``
        Imputation method passed to :func:`~actionet.imputation.impute_features`.

        - ``"diffusion"`` — network diffusion smoothing.
        - ``"pca"`` — PCA-based kernel smoothing.
        - ``"archetypes"`` — fast archetype interpolation; requires
          pre-computed profiles (run
          :func:`~actionet.specificity.compute_archetype_feature_specificity`
          first).
        - ``"none"`` — raw expression, no imputation.

        Setting ``alpha=0`` also forces raw mode regardless of ``method``
        (except ``"archetypes"``), for backward compatibility.
    alpha
        Diffusion parameter used by ``method="diffusion"`` and ``method="pca"``.
        Setting ``alpha=0`` disables imputation (equivalent to ``method="none"``).
    layer
        Layer to read expression from (``None`` uses ``.X``).
        Used by ``method="diffusion"``, ``method="pca"``, and ``method="none"``.
    trans_attr
        Optional continuous attribute controlling transparency.
    trans_th
        Z-score threshold for transparency mapping.
    trans_fac
        Transparency scale factor for the logistic mapping.
    cmap
        Continuous palette for expression values.
    size
        Marker size for UMAP scatter.
    network_key
        Key in ``adata.obsp`` for the ACTIONet network.
    basis
        Key in ``adata.obsm`` containing 2D coordinates.
    single_plot
        If True, arrange multiple plots into a grid.
    legend
        Whether to show the color legend.
    sort_features
        If True, sort matched features alphabetically.
    archetype_profile_key : str, default ``"archetype_feat_profile"``
        Key in ``adata.varm`` for archetype expression profiles.
        Used only by ``method="archetypes"``.
    archetype_matrix_key : str, default ``"H_merged"``
        Key in ``adata.obsm`` for cell-archetype membership weights.
        Used only by ``method="archetypes"``.
    n_threads
        Number of threads for imputation.
    backed_chunk_size : int, default 4096
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.

    Returns
    -------
    lets_plot.PlotSpec or dict
        A lets-plot object, a grid if ``single_plot`` is True, or a dict of plots.
    """

    requested = _flatten_features(features)
    marker_set = _select_features(adata, requested, features_use, sort_features)
    if len(marker_set) == 0:
        raise ValueError("No features found in 'features_use'.")

    use_raw = method == "none" or (alpha == 0 and method != "archetypes")

    if use_raw:
        expr_profile = _extract_expression(
            adata, marker_set, features_use, layer,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
    else:
        expr_profile = impute_features(
            adata,
            features=requested,
            method=method,
            features_use=features_use,
            network_key=network_key,
            layer=layer,
            alpha=alpha,
            n_threads=n_threads,
            backed_chunk_size=backed_chunk_size,
            lazy_transform=lazy_transform,
            archetype_profile_key=archetype_profile_key,
            archetype_matrix_key=archetype_matrix_key,
        )
        if sort_features:
            expr_profile = expr_profile.loc[:, [f for f in marker_set if f in expr_profile.columns]]

    out = {}
    for feat_name in expr_profile.columns:
        values = expr_profile[feat_name].to_numpy()
        out[feat_name] = plot_umap(
            adata,
            color=values,
            color_source=None,
            cmap=cmap,
            size=size,
            trans_attr=trans_attr,
            trans_fac=trans_fac,
            trans_th=trans_th,
            basis=basis,
            legend=legend,
            title=feat_name,
        )

    if single_plot and len(out) > 1:
        try:
            from lets_plot import gggrid
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("lets-plot is required for plot grids.") from exc
        nrow, ncol = _grid_shape(len(out))
        scale_size = size / max(nrow, 1)
        plots = []
        for key in list(out.keys()):
            plots.append(
                plot_umap(
                    adata,
                    color=expr_profile[key].to_numpy(),
                    color_source=None,
                    cmap=cmap,
                    size=scale_size,
                    trans_attr=trans_attr,
                    trans_fac=trans_fac,
                    trans_th=trans_th,
                    basis=basis,
                    legend=legend,
                    title=key,
                )
            )
        total_cells = nrow * ncol
        if len(plots) < total_cells:
            plots.extend([None] * (total_cells - len(plots)))
        return gggrid(plots, ncol=ncol)

    if len(out) == 1:
        return next(iter(out.values()))
    return out


def plot_feature_expression_raster(
    adata: AnnData,
    features: Union[str, Sequence[Union[str, Iterable[str]]]],
    features_use: Optional[str] = None,
    method: Literal["diffusion", "pca", "archetypes", "none"] = "diffusion",
    alpha: float = 0.9,
    layer: Optional[str] = None,
    trans_attr: Optional[Union[str, Sequence, np.ndarray]] = None,
    trans_th: float = -0.5,
    trans_fac: float = 3.0,
    cmap: Union[str, Sequence[str]] = "magma",
    size: float = 1.0,
    network_key: str = "actionet",
    basis: str = "umap_2d_actionet",
    single_plot: bool = False,
    legend: bool = False,
    sort_features: bool = True,
    archetype_profile_key: str = "archetype_feat_profile",
    archetype_matrix_key: str = "H_merged",
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
    lazy_transform: Optional[LazyTransform] = None,
) -> Union[Any, dict[str, Any]]:
    """Plot feature expression values on the UMAP embedding using a raster backend.

    Parameters
    ----------
    adata
        AnnData with expression data and embedding in ``adata.obsm[basis]``.
    features
        Feature name or collection of features to plot.
    features_use
        Column in ``adata.var`` to use for feature matching.
    method : {"diffusion", "pca", "archetypes", "none"}, default ``"diffusion"``
        Imputation method passed to :func:`~actionet.imputation.impute_features`.

        - ``"diffusion"`` — network diffusion smoothing.
        - ``"pca"`` — PCA-based kernel smoothing.
        - ``"archetypes"`` — fast archetype interpolation; requires
          pre-computed profiles (run
          :func:`~actionet.specificity.compute_archetype_feature_specificity`
          first).
        - ``"none"`` — raw expression, no imputation.

        Setting ``alpha=0`` also forces raw mode regardless of ``method``
        (except ``"archetypes"``), for backward compatibility.
    alpha
        Diffusion parameter used by ``method="diffusion"`` and ``method="pca"``.
        Setting ``alpha=0`` disables imputation (equivalent to ``method="none"``).
    layer
        Layer to read expression from (``None`` uses ``.X``).
        Used by ``method="diffusion"``, ``method="pca"``, and ``method="none"``.
    trans_attr
        Optional continuous attribute controlling transparency.
    trans_th
        Z-score threshold for transparency mapping.
    trans_fac
        Transparency scale factor for the logistic mapping.
    cmap
        Continuous palette for expression values.
    size
        Marker size for UMAP scatter.
    network_key
        Key in ``adata.obsp`` for the ACTIONet network.
    basis
        Key in ``adata.obsm`` containing 2D coordinates.
    single_plot
        If True, arrange multiple plots into a single matplotlib figure.
    legend
        Whether to show the color legend.
    sort_features
        If True, sort matched features alphabetically.
    archetype_profile_key : str, default ``"archetype_feat_profile"``
        Key in ``adata.varm`` for archetype expression profiles.
        Used only by ``method="archetypes"``.
    archetype_matrix_key : str, default ``"H_merged"``
        Key in ``adata.obsm`` for cell-archetype membership weights.
        Used only by ``method="archetypes"``.
    n_threads
        Number of threads for imputation.
    backed_chunk_size : int, default 4096
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.
    """

    requested = _flatten_features(features)
    marker_set = _select_features(adata, requested, features_use, sort_features)
    if len(marker_set) == 0:
        raise ValueError("No features found in 'features_use'.")

    use_raw = method == "none" or (alpha == 0 and method != "archetypes")

    if use_raw:
        expr_profile = _extract_expression(
            adata, marker_set, features_use, layer,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
    else:
        expr_profile = impute_features(
            adata,
            features=requested,
            method=method,
            features_use=features_use,
            network_key=network_key,
            layer=layer,
            alpha=alpha,
            n_threads=n_threads,
            backed_chunk_size=backed_chunk_size,
            lazy_transform=lazy_transform,
            archetype_profile_key=archetype_profile_key,
            archetype_matrix_key=archetype_matrix_key,
        )
        if sort_features:
            expr_profile = expr_profile.loc[:, [f for f in marker_set if f in expr_profile.columns]]

    out = {}
    for feat_name in expr_profile.columns:
        values = expr_profile[feat_name].to_numpy()
        out[feat_name] = plot_umap_raster(
            adata,
            color=values,
            color_source=None,
            cmap=cmap,
            size=size,
            trans_attr=trans_attr,
            trans_fac=trans_fac,
            trans_th=trans_th,
            basis=basis,
            legend=legend,
            title=feat_name,
        )

    if single_plot and len(out) > 1:
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("matplotlib is required for raster UMAP plotting.") from exc
        nrow, ncol = _grid_shape(len(out))
        panel_width = 6.0
        panel_height = 5.0
        fig = Figure(
            figsize=(panel_width * ncol, panel_height * nrow),
            dpi=100.0,
            facecolor="white",
            layout="constrained",
        )
        FigureCanvasAgg(fig)
        axes = np.asarray(fig.subplots(nrow, ncol, squeeze=False))
        scale_size = size / max(nrow, 1)

        for ax, key in zip(axes.flat, list(expr_profile.columns)):
            ctx = _prepare_umap_context(
                adata,
                color=expr_profile[key].to_numpy(),
                color_source=None,
                color_type="continuous",
                basis=basis,
                alpha=1.0,
                fig_dpi=100.0,
                fig_size=(panel_width, panel_height),
                trans_attr=trans_attr,
                trans_fac=trans_fac,
                trans_th=trans_th,
                color_slot=None,
            )
            _render_umap_raster(
                ax,
                ctx,
                cmap=cmap,
                palette="tab20",
                size=scale_size,
                legend=legend,
                title=key,
                vmin=None,
                vmax=None,
                order=None,
                na_color="#cccccc",
                hide_na=False,
                text_labels=False,
                text_label_size=9.0,
                nudge_text=False,
            )

        total_cells = nrow * ncol
        if len(expr_profile.columns) < total_cells:
            for ax in axes.flat[len(expr_profile.columns):]:
                ax.set_visible(False)

        return fig

    if len(out) == 1:
        return next(iter(out.values()))
    return out
