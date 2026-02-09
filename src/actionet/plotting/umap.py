"""UMAP plotting utilities for ACTIONet."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .utils import (
    apply_point_order,
    build_discrete_color_map,
    compute_transparency,
    ensure_rgb_hex,
    normalize_cmap_spec,
    resolve_embedding,
    resolve_numeric_vector,
)


def _prepare_alpha(alpha: Union[float, Sequence[float]], n_obs: int) -> np.ndarray:
    """Return per-point alpha values, validating length for vectors."""
    if isinstance(alpha, (list, tuple, np.ndarray, pd.Series)):
        values = np.asarray(alpha)
        if values.shape[0] != n_obs:
            raise ValueError("Alpha vector length does not match number of observations.")
        return values
    return np.full(n_obs, float(alpha))


def _figsize_to_px(figsize: tuple[float, float], dpi: float) -> tuple[float, float]:
    """Convert an inches-based figsize to pixels for lets-plot."""
    return float(figsize[0]) * dpi, float(figsize[1]) * dpi


def _classify_color_values(
    values: Union[np.ndarray, Sequence, pd.Series]
) -> tuple[Union[np.ndarray, pd.Series], str, Optional[Sequence[str]]]:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, "rgb", None

    series = pd.Series(values)
    if series.dtype.name == "category":
        return series, "categorical", list(series.cat.categories)

    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(), "continuous", None

    non_na = series.dropna()
    if not non_na.empty and non_na.map(lambda v: isinstance(v, str) and v.startswith("#")).all():
        return series, "rgb", None

    categorical = series.astype("category")
    return categorical, "categorical", list(categorical.cat.categories)


def _resolve_var_values(adata: AnnData, key: str, layer: Optional[str]) -> np.ndarray:
    if key not in adata.var_names:
        raise KeyError(f"Feature {key} not found in adata.var_names.")
    idx = int(adata.var_names.get_loc(key))
    matrix = adata.layers[layer] if layer else adata.X
    col = matrix[:, idx]
    if hasattr(col, "toarray"):
        col = col.toarray()
    return np.asarray(col).ravel()


def _resolve_color_input(
    adata: AnnData,
    *,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]],
    color_source: Optional[Literal["obs", "obsm"]],
    color_slot: Optional[str],
) -> tuple[Optional[Union[np.ndarray, pd.Series]], str, Optional[Sequence[str]]]:
    if isinstance(color, str):
        if color_source is None:
            raise ValueError("color_source must be set when color is a key string.")
        if color_source == "obs":
            return _classify_color_values(adata.obs[color])
        if color_source == "obsm":
            values = np.asarray(adata.obsm[color])
            if values.ndim == 2 and values.shape[1] == 3:
                return values, "rgb", None
            if values.ndim == 1:
                return _classify_color_values(values)
            raise ValueError("Unsupported obsm color shape; expected 1D or Nx3.")
        raise ValueError("color_source must be one of obs or obsm.")

    if color is not None:
        return _classify_color_values(color)

    if color_slot and color_slot in adata.obsm:
        slot_vals = np.asarray(adata.obsm[color_slot])
        if slot_vals.ndim == 2 and slot_vals.shape[1] == 3 and slot_vals.shape[0] == adata.n_obs:
            return slot_vals, "rgb", None

    return None, "none", None


def plot_umap(
    adata: AnnData,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    color_source: Optional[Literal["obs", "var", "obsm"]] = "obs",
    basis: str = "umap_2d_actionet",
    cmap: Optional[Union[str, Sequence[str]]] = "magma",
    palette: Optional[Union[str, Sequence[str], dict]] = "tab20",
    size: float = 1.5,
    alpha: Union[float, Sequence[float]] = 1,
    legend: bool = True,
    figsize: tuple[float, float] = (6, 5),
    fig_dpi: float = 100.0,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    order: Optional[Union[str, Sequence[int]]] = None,
    na_color: str = "#cccccc",
    trans_attr: Optional[Union[str, Sequence[float], np.ndarray]] = None,
    trans_fac: float = 1.5,
    trans_th: float = -0.5,
    hide_na: bool = False,
    color_slot: Optional[str] = "colors_actionet",
    add_text_labels: bool = False,
    label_text_size: float = 9.0,
    nudge_text_labels: bool = False,
):
    """Plot a UMAP embedding with discrete or continuous coloring.

    Parameters
    ----------
    adata
        AnnData with a 2D embedding in ``adata.obsm[basis]``.
    color
        Color key string or vector-like values with length ``adata.n_obs``.
    color_source
        When ``color`` is a key string, where to resolve it from: ``"obs"`` or ``"obsm"``.
    basis
        Key in ``adata.obsm`` containing 2D coordinates (default: ``"X_umap"``).
    cmap
        Continuous colormap name or list of colors for gradients.
    palette
        Discrete palette name, list of colors, or dict mapping category to color.
    size
        Marker size.
    alpha
        Scalar alpha or per-point alpha vector (length ``adata.n_obs``).
    legend
        Whether to show the legend.
    figsize
        Figure size in inches.
    fig_dpi
        DPI used to convert inches to pixels for lets-plot.
    title
        Optional plot title.
    vmin, vmax
        Optional clamping bounds for continuous values.
    order
        Overplot ordering: ``"random"``, ``"sort"`` (ascending), or explicit indices.
    na_color
        Color for missing values when categorical labels contain NA.
    trans_attr
        Continuous attribute used to compute point transparency.
    trans_fac
        Transparency scale factor for the logistic mapping.
    trans_th
        Z-score threshold for transparency mapping.
    hide_na
        If True, remove points with missing categorical labels.
    color_slot
        If no color specified, try ``adata.obsm[color_slot]`` for RGB colors.
    add_text_labels
        If True, add categorical label text at cluster centers.
    label_text_size
        Font size for text labels (points).
    nudge_text_labels
        If True, apply a small offset to label positions.

    Returns
    -------
    lets_plot.PlotSpec
        A lets-plot object.
    """

    try:
        from lets_plot import (
            aes,
            geom_point,
            geom_text,
            ggplot,
            ggsize,
            labs,
            scale_alpha_identity,
            scale_color_gradientn,
            scale_color_identity,
            scale_color_manual,
            theme,
            theme_void,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("lets-plot is required for static UMAP plotting.") from exc

    coords = resolve_embedding(adata, basis)

    values, kind, categories = _resolve_color_input(
        adata,
        color=color,
        color_source=color_source,
        color_slot=color_slot,
    )

    alpha_values = _prepare_alpha(alpha, adata.n_obs)
    if trans_attr is not None:
        trans_vals = resolve_numeric_vector(adata, trans_attr, "trans_attr")
        alpha_values = alpha_values * compute_transparency(trans_vals, trans_fac, trans_th)
    alpha_is_array = isinstance(alpha, (list, tuple, np.ndarray, pd.Series)) or trans_attr is not None
    alpha_mapping = aes(alpha="alpha") if alpha_is_array else None
    alpha_scale = scale_alpha_identity() if alpha_is_array else None
    alpha_arg = None if alpha_is_array else float(alpha_values[0])
    point_kwargs = {"size": size}
    if alpha_arg is not None:
        point_kwargs["alpha"] = alpha_arg

    def _base_aes(color_key: Optional[str] = None):
        if color_key is None:
            if alpha_mapping is not None:
                return aes("x", "y", alpha="alpha")
            return aes("x", "y")
        if alpha_mapping is not None:
            return aes("x", "y", color=color_key, alpha="alpha")
        return aes("x", "y", color=color_key)

    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "alpha": alpha_values})

    if kind == "none":
        plot_df["color"] = "default"
        color_map = {"default": "#4c72b0"}
        base_aes = _base_aes("color")
        plot = ggplot(plot_df, base_aes) + geom_point(**point_kwargs)
        plot = plot + scale_color_manual(values=color_map)
    elif kind == "rgb":
        plot_df["color"] = ensure_rgb_hex(values)
        base_aes = _base_aes("color")
        plot = ggplot(plot_df, base_aes) + geom_point(show_legend=legend, **point_kwargs)
        plot = plot + scale_color_identity()
    elif kind == "categorical":
        series = pd.Series(values)
        series = series.astype("category")
        if series.isna().any():
            series = series.cat.add_categories(["NA"]).fillna("NA")
        categories = list(series.cat.categories) if categories is None else list(categories)
        if "NA" not in categories and series.isna().any():
            categories.append("NA")
        color_map = build_discrete_color_map(categories, palette)
        if "NA" in color_map:
            color_map["NA"] = na_color
        color_map = {str(key): value for key, value in color_map.items()}
        labels = series.astype(str).to_numpy()
        plot_df["color"] = labels
        if hide_na and "NA" in plot_df["color"].values:
            plot_df = plot_df[plot_df["color"] != "NA"].copy()
        base_aes = _base_aes("color")
        plot = ggplot(plot_df, base_aes) + geom_point(**point_kwargs)
        plot = plot + scale_color_manual(values=color_map)
        if add_text_labels:
            label_df = plot_df.groupby("color", as_index=False)[["x", "y"]].median()
            if nudge_text_labels and not label_df.empty:
                x_range = plot_df["x"].max() - plot_df["x"].min()
                y_range = plot_df["y"].max() - plot_df["y"].min()
                label_df["x"] += (label_df["x"].rank() / len(label_df)) * x_range * 0.02
                label_df["y"] += (label_df["y"].rank() / len(label_df)) * y_range * 0.02
            plot = plot + geom_text(
                data=label_df,
                mapping=aes("x", "y", label="color"),
                size=label_text_size,
                color="black",
                show_legend=False,
            )
    else:
        values = np.asarray(values, dtype=float)
        if vmin is not None:
            values = np.maximum(values, vmin)
        if vmax is not None:
            values = np.minimum(values, vmax)
        plot_df["value"] = values
        non_na = plot_df[~np.isnan(values)]
        na_df = plot_df[np.isnan(values)]
        if order is None:
            order = "sort"
        non_na = apply_point_order(non_na, order, values[~np.isnan(values)])
        base_aes = _base_aes("value")
        plot = ggplot(non_na, base_aes) + geom_point(**point_kwargs)
        if not na_df.empty:
            na_aes = _base_aes()
            plot = plot + geom_point(
                data=na_df,
                mapping=na_aes,
                color=na_color,
                show_legend=False,
                **point_kwargs,
            )
        gradient = normalize_cmap_spec(cmap)
        plot = plot + scale_color_gradientn(colors=gradient)

    if alpha_scale is not None:
        plot = plot + alpha_scale

    width_px, height_px = _figsize_to_px(figsize, fig_dpi)
    plot = plot + theme_void() + ggsize(width_px, height_px)

    if legend:
        plot = plot + theme(legend_position="right")
    else:
        plot = plot + theme(legend_position="none")

    if title:
        plot = plot + labs(title=title)

    return plot


def plot_umap_interactive(
    adata: AnnData,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    color_source: Optional[Literal["obs", "var", "obsm"]] = None,
    layer: Optional[str] = None,
    basis: str = "X_umap",
    palette: Optional[Union[str, Sequence[str], dict]] = "tab20",
    cmap: Optional[Union[str, Sequence[str]]] = "viridis",
    size: float = 6,
    alpha: float = 0.9,
    hover_data: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    trans_attr: Optional[Union[str, Sequence[float], np.ndarray]] = None,
    trans_fac: float = 1.5,
    trans_th: float = -0.5,
):
    """Plot an interactive UMAP embedding using Plotly WebGL.

    Parameters
    ----------
    adata
        AnnData with a 2D embedding in ``adata.obsm[basis]``.
    color
        Color key string or vector-like values with length ``adata.n_obs``.
    color_source
        When ``color`` is a key string, where to resolve it from: ``"obs"``, ``"var"``, or ``"obsm"``.
    layer
        Layer to pull expression values from when ``color_source="var"``.
    basis
        Key in ``adata.obsm`` containing 2D coordinates (default: ``"X_umap"``).
    palette
        Discrete palette name, list of colors, or dict mapping category to color.
    cmap
        Continuous colormap name or list of colors for gradients.
    size
        Marker size.
    alpha
        Marker opacity.
    hover_data
        Columns in ``adata.obs`` to include in hover tooltips.
    title
        Optional plot title.
    vmin, vmax
        Optional clamping bounds for continuous values.
    trans_attr
        Continuous attribute used to compute point transparency.
    trans_fac
        Transparency scale factor for the logistic mapping.
    trans_th
        Z-score threshold for transparency mapping.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure (WebGL).
    """

    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotly is required for interactive UMAP plotting.") from exc

    values, kind, categories = _resolve_color_input(
        adata,
        color=color,
        color_source=color_source,
        layer=layer,
        color_slot=None,
    )

    coords = resolve_embedding(adata, basis)
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})

    alpha_values = None
    if trans_attr is not None:
        trans_vals = resolve_numeric_vector(adata, trans_attr, "trans_attr")
        alpha_values = compute_transparency(trans_vals, trans_fac, trans_th) * float(alpha)

    color_args = {}
    if kind == "none":
        plot_df["color"] = "default"
        color_args["color_discrete_map"] = {"default": "#4c72b0"}
        color_key = "color"
    elif kind == "rgb":
        plot_df["color"] = ensure_rgb_hex(values)
        color_args["color_discrete_map"] = {
            value: value for value in plot_df["color"].unique().tolist()
        }
        color_key = "color"
    elif kind == "categorical":
        series = pd.Series(values).astype("category")
        if series.isna().any():
            series = series.cat.add_categories(["NA"]).fillna("NA")
        categories = list(series.cat.categories) if categories is None else list(categories)
        if "NA" not in categories and series.isna().any():
            categories.append("NA")
        color_map = build_discrete_color_map(categories, palette)
        if "NA" in color_map:
            color_map["NA"] = "#cccccc"
        plot_df["color"] = series.astype(str).to_numpy()
        color_args["color_discrete_map"] = color_map
        color_key = "color"
    else:
        values = np.asarray(values, dtype=float)
        plot_df["value"] = values
        gradient = normalize_cmap_spec(cmap)
        color_args["color_continuous_scale"] = gradient
        if vmin is not None or vmax is not None:
            color_args["range_color"] = (
                vmin if vmin is not None else float(np.nanmin(values)),
                vmax if vmax is not None else float(np.nanmax(values)),
            )
        color_key = "value"

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_key,
        hover_data=list(hover_data) if hover_data else None,
        render_mode="webgl",
        **color_args,
    )
    marker_opts = {"size": size, "opacity": alpha}
    if alpha_values is not None:
        marker_opts["opacity"] = alpha_values
    fig.update_traces(marker=marker_opts)
    if title:
        fig.update_layout(title=title)
    return fig
