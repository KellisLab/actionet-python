"""UMAP plotting utilities for ACTIONet."""

from __future__ import annotations

import io
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _PreparedUmapContext:
    """Common UMAP plot inputs shared by lets-plot and raster renderers."""

    coords: np.ndarray
    values: Optional[Union[np.ndarray, pd.Series]]
    kind: Literal["none", "rgb", "categorical", "continuous"]
    categories: Optional[list[str]]
    alpha_values: np.ndarray
    alpha_is_array: bool
    alpha_arg: Optional[float]
    width_px: float
    height_px: float


class _ActionetRasterFigureMixin:
    """Provide a PNG-rich display hook for notebook frontends."""

    def _repr_png_(self):
        buf = io.BytesIO()
        self.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {"image/png": self._repr_png_()}, {}


def _prepare_alpha(alpha: Union[float, Sequence[float]], n_obs: int) -> np.ndarray:
    """Return per-point alpha values, validating length for vectors."""
    if isinstance(alpha, (list, tuple, np.ndarray, pd.Series)):
        values = np.asarray(alpha)
        if values.shape[0] != n_obs:
            raise ValueError("Alpha vector length does not match number of observations.")
        return values.astype(float)
    return np.full(n_obs, float(alpha), dtype=float)


def _figsize_to_px(figsize: tuple[float, float], dpi: float) -> tuple[float, float]:
    """Convert an inches-based figsize to pixels for plotting backends."""
    return float(figsize[0]) * dpi, float(figsize[1]) * dpi


def _classify_color_values(
    values: Union[np.ndarray, Sequence, pd.Series],
    force: Optional[Literal["categorical", "continuous"]] = None,
) -> tuple[Union[np.ndarray, pd.Series], Literal["rgb", "categorical", "continuous"], Optional[list[str]]]:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, "rgb", None

    series = pd.Series(values)

    if force == "categorical":
        categorical = series.astype("category")
        return categorical, "categorical", list(categorical.cat.categories)

    if force == "continuous":
        return series.to_numpy(), "continuous", None

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
    color_source: Optional[Literal["obs", "var", "obsm"]],
    color_slot: Optional[str],
    color_type: Optional[Literal["auto", "categorical", "continuous"]] = "auto",
    layer: Optional[str] = None,
) -> tuple[Optional[Union[np.ndarray, pd.Series]], Literal["none", "rgb", "categorical", "continuous"], Optional[list[str]]]:
    force = None if (color_type is None or color_type == "auto") else color_type
    if isinstance(color, str):
        if color_source is None:
            raise ValueError("color_source must be set when color is a key string.")
        if color_source == "obs":
            return _classify_color_values(adata.obs[color], force=force)
        if color_source == "var":
            return _classify_color_values(_resolve_var_values(adata, color, layer), force=force)
        if color_source == "obsm":
            values = np.asarray(adata.obsm[color])
            if values.ndim == 2 and values.shape[1] == 3:
                return values, "rgb", None
            if values.ndim == 1:
                return _classify_color_values(values, force=force)
            raise ValueError("Unsupported obsm color shape; expected 1D or Nx3.")
        raise ValueError("color_source must be one of obs, var, or obsm.")

    if color is not None:
        return _classify_color_values(color, force=force)

    if color_slot and color_slot in adata.obsm:
        slot_vals = np.asarray(adata.obsm[color_slot])
        if slot_vals.ndim == 2 and slot_vals.shape[1] == 3 and slot_vals.shape[0] == adata.n_obs:
            return slot_vals, "rgb", None

    return None, "none", None


def _prepare_umap_context(
    adata: AnnData,
    *,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]],
    color_source: Optional[Literal["obs", "var", "obsm"]],
    color_type: Optional[Literal["auto", "categorical", "continuous"]],
    basis: str,
    alpha: Union[float, Sequence[float]],
    fig_dpi: float,
    figsize: tuple[float, float],
    trans_attr: Optional[Union[str, Sequence[float], np.ndarray]],
    trans_fac: float,
    trans_th: float,
    color_slot: Optional[str],
) -> _PreparedUmapContext:
    coords = resolve_embedding(adata, basis)
    values, kind, categories = _resolve_color_input(
        adata,
        color=color,
        color_source=color_source,
        color_slot=color_slot,
        color_type=color_type,
        layer=None,
    )

    alpha_values = _prepare_alpha(alpha, adata.n_obs)
    if trans_attr is not None:
        trans_vals = resolve_numeric_vector(adata, trans_attr, "trans_attr")
        alpha_values = alpha_values * compute_transparency(trans_vals, trans_fac, trans_th)
    alpha_is_array = isinstance(alpha, (list, tuple, np.ndarray, pd.Series)) or trans_attr is not None
    alpha_arg = None if alpha_is_array else float(alpha_values[0])
    width_px, height_px = _figsize_to_px(figsize, fig_dpi)

    return _PreparedUmapContext(
        coords=coords,
        values=values,
        kind=kind,
        categories=categories,
        alpha_values=alpha_values,
        alpha_is_array=alpha_is_array,
        alpha_arg=alpha_arg,
        width_px=width_px,
        height_px=height_px,
    )


def _build_base_frame(ctx: _PreparedUmapContext) -> pd.DataFrame:
    data = {"x": ctx.coords[:, 0], "y": ctx.coords[:, 1]}
    if ctx.alpha_is_array:
        data["alpha"] = ctx.alpha_values
    return pd.DataFrame(data)


def _prepare_categorical_payload(
    values: Union[np.ndarray, pd.Series, Sequence],
    *,
    categories: Optional[Sequence[str]],
    palette: Optional[Union[str, Sequence[str], dict]],
    na_color: str,
) -> tuple[pd.Series, np.ndarray, list[str], dict[str, str]]:
    series = pd.Series(values).astype("category")
    had_na = bool(series.isna().any())
    if had_na:
        series = series.cat.add_categories(["NA"]).fillna("NA")
    categories_out = list(series.cat.categories) if categories is None else list(categories)
    if had_na and "NA" not in categories_out:
        categories_out.append("NA")
    color_map = build_discrete_color_map(categories_out, palette)
    if "NA" in color_map:
        color_map["NA"] = na_color
    color_map = {str(key): value for key, value in color_map.items()}
    labels = series.astype(str).to_numpy()
    return series, labels, categories_out, color_map


def _compute_label_positions(
    plot_df: pd.DataFrame,
    *,
    nudge_text_labels: bool,
) -> pd.DataFrame:
    label_df = plot_df.groupby("color", as_index=False)[["x", "y"]].median()
    if nudge_text_labels and not label_df.empty:
        x_range = plot_df["x"].max() - plot_df["x"].min()
        y_range = plot_df["y"].max() - plot_df["y"].min()
        label_df["x"] += (label_df["x"].rank() / len(label_df)) * x_range * 0.02
        label_df["y"] += (label_df["y"].rank() / len(label_df)) * y_range * 0.02
    return label_df


def _prepare_continuous_values(
    values: Union[np.ndarray, Sequence, pd.Series],
    *,
    vmin: Optional[float],
    vmax: Optional[float],
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if vmin is not None:
        arr = np.maximum(arr, vmin)
    if vmax is not None:
        arr = np.minimum(arr, vmax)
    return arr


def _prepare_rgb_values(values: Union[np.ndarray, pd.Series, Sequence]) -> list[str]:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return ensure_rgb_hex(arr)
    return pd.Series(values).astype(str).tolist()


def _validate_sampling_args(
    sampling: Literal["none", "random"],
    sample_n: Optional[int],
    tooltips: Literal["none", "default"],
) -> None:
    if sampling not in {"none", "random"}:
        raise ValueError("sampling must be one of 'none' or 'random'.")
    if sampling == "random":
        if sample_n is None:
            raise ValueError("sample_n must be provided when sampling='random'.")
        if int(sample_n) <= 0:
            raise ValueError("sample_n must be a positive integer when sampling='random'.")
    if tooltips not in {"none", "default"}:
        raise ValueError("tooltips must be one of 'none' or 'default'.")


def _point_render_kwargs(
    *,
    size: float,
    alpha_arg: Optional[float],
    sampling: Literal["none", "random"],
    sample_n: Optional[int],
    sampling_seed: int,
    tooltips: Literal["none", "default"],
    sampling_random,
) -> dict:
    kwargs = {"size": size}
    if alpha_arg is not None:
        kwargs["alpha"] = alpha_arg
    if sampling == "random":
        kwargs["sampling"] = sampling_random(int(sample_n), seed=sampling_seed)
    else:
        kwargs["sampling"] = "none"
    if tooltips == "none":
        kwargs["tooltips"] = "none"
    return kwargs


def _mpl_to_rgba_array():
    try:
        from matplotlib.colors import to_rgba_array
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for raster UMAP plotting.") from exc
    return to_rgba_array


def _rgba_values(
    colors: Union[str, Sequence[str], np.ndarray],
    *,
    n_obs: int,
    alpha_values: Optional[np.ndarray],
    alpha_arg: Optional[float],
) -> np.ndarray:
    to_rgba_array = _mpl_to_rgba_array()
    if isinstance(colors, str):
        rgba = np.repeat(to_rgba_array([colors]), n_obs, axis=0)
    else:
        rgba = to_rgba_array(np.asarray(colors))
        if rgba.shape[0] == 1 and n_obs > 1:
            rgba = np.repeat(rgba, n_obs, axis=0)
    if alpha_values is not None:
        rgba[:, 3] = np.asarray(alpha_values, dtype=float)
    elif alpha_arg is not None:
        rgba[:, 3] = float(alpha_arg)
    return rgba


def _style_raster_axes(ax, *, title: Optional[str]) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def _raster_marker_area(size: float) -> float:
    size = max(float(size), 0.1)
    return size * size


def _new_raster_figure(
    *,
    figsize: tuple[float, float],
    fig_dpi: float,
):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for raster UMAP plotting.") from exc

    class _ActionetRasterFigure(_ActionetRasterFigureMixin, Figure):
        pass

    fig = _ActionetRasterFigure(figsize=figsize, dpi=fig_dpi, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _render_umap_raster(
    ax,
    ctx: _PreparedUmapContext,
    *,
    cmap: Optional[Union[str, Sequence[str]]],
    palette: Optional[Union[str, Sequence[str], dict]],
    size: float,
    legend: bool,
    title: Optional[str],
    vmin: Optional[float],
    vmax: Optional[float],
    order: Optional[Union[str, Sequence[int]]],
    na_color: str,
    hide_na: bool,
    add_text_labels: bool,
    label_text_size: float,
    nudge_text_labels: bool,
) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for raster UMAP plotting.") from exc

    if ax.figure.canvas is None:  # pragma: no branch - canvas exists for direct Figure() usage
        FigureCanvasAgg(ax.figure)

    marker_area = _raster_marker_area(size)
    per_point_alpha = ctx.alpha_values if ctx.alpha_is_array else None
    scalar_alpha = None if ctx.alpha_is_array else ctx.alpha_arg

    if ctx.kind == "none":
        rgba = _rgba_values(
            "#4c72b0",
            n_obs=ctx.coords.shape[0],
            alpha_values=per_point_alpha,
            alpha_arg=scalar_alpha,
        )
        ax.scatter(
            ctx.coords[:, 0],
            ctx.coords[:, 1],
            s=marker_area,
            c=rgba,
            linewidths=0,
            edgecolors="none",
        )

    elif ctx.kind == "rgb":
        colors = _prepare_rgb_values(ctx.values)
        rgba = _rgba_values(
            colors,
            n_obs=ctx.coords.shape[0],
            alpha_values=per_point_alpha,
            alpha_arg=scalar_alpha,
        )
        ax.scatter(
            ctx.coords[:, 0],
            ctx.coords[:, 1],
            s=marker_area,
            c=rgba,
            linewidths=0,
            edgecolors="none",
        )

    elif ctx.kind == "categorical":
        _, labels, _, color_map = _prepare_categorical_payload(
            ctx.values,
            categories=ctx.categories,
            palette=palette,
            na_color=na_color,
        )
        plot_df = _build_base_frame(ctx)
        plot_df["color"] = labels
        if hide_na and "NA" in plot_df["color"].values:
            plot_df = plot_df[plot_df["color"] != "NA"].copy()
        point_colors = plot_df["color"].map(color_map).to_numpy()
        alpha_vec = plot_df["alpha"].to_numpy() if ctx.alpha_is_array else None
        rgba = _rgba_values(
            point_colors,
            n_obs=len(plot_df),
            alpha_values=alpha_vec,
            alpha_arg=scalar_alpha,
        )
        ax.scatter(
            plot_df["x"].to_numpy(),
            plot_df["y"].to_numpy(),
            s=marker_area,
            c=rgba,
            linewidths=0,
            edgecolors="none",
        )

        if add_text_labels and not plot_df.empty:
            label_df = _compute_label_positions(plot_df, nudge_text_labels=nudge_text_labels)
            for _, row in label_df.iterrows():
                ax.text(row["x"], row["y"], row["color"], fontsize=label_text_size, color="black")

        if legend:
            present_labels = set(plot_df["color"].unique().tolist())
            legend_labels = [label for label in color_map if label in present_labels]
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color_map[label],
                    markeredgecolor="none",
                    markersize=max(np.sqrt(marker_area), 4.0),
                    label=label,
                )
                for label in legend_labels
            ]
            if handles:
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                )

    else:
        values = _prepare_continuous_values(ctx.values, vmin=vmin, vmax=vmax)
        base_df = _build_base_frame(ctx)
        plot_df = base_df.copy()
        plot_df["value"] = values
        non_na = plot_df[~np.isnan(values)].copy()
        na_df = base_df[np.isnan(values)].copy()

        order_use = "sort" if order is None else order
        non_na = apply_point_order(non_na, order_use, values[~np.isnan(values)])

        gradient = normalize_cmap_spec(cmap)
        cmap_obj = LinearSegmentedColormap.from_list("actionet_gradient", gradient)
        if non_na.empty:
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            vmin_eff = float(np.nanmin(non_na["value"])) if vmin is None else float(vmin)
            vmax_eff = float(np.nanmax(non_na["value"])) if vmax is None else float(vmax)
            if np.isclose(vmin_eff, vmax_eff):
                vmax_eff = vmin_eff + 1e-12
            norm = Normalize(vmin=vmin_eff, vmax=vmax_eff)

        if not non_na.empty:
            if ctx.alpha_is_array:
                rgba = cmap_obj(norm(non_na["value"].to_numpy()))
                rgba[:, 3] = non_na["alpha"].to_numpy()
                ax.scatter(
                    non_na["x"].to_numpy(),
                    non_na["y"].to_numpy(),
                    s=marker_area,
                    c=rgba,
                    linewidths=0,
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    non_na["x"].to_numpy(),
                    non_na["y"].to_numpy(),
                    s=marker_area,
                    c=non_na["value"].to_numpy(),
                    cmap=cmap_obj,
                    norm=norm,
                    alpha=scalar_alpha,
                    linewidths=0,
                    edgecolors="none",
                )

        if not na_df.empty:
            alpha_vec = na_df["alpha"].to_numpy() if ctx.alpha_is_array else None
            rgba = _rgba_values(
                na_color,
                n_obs=len(na_df),
                alpha_values=alpha_vec,
                alpha_arg=scalar_alpha,
            )
            ax.scatter(
                na_df["x"].to_numpy(),
                na_df["y"].to_numpy(),
                s=marker_area,
                c=rgba,
                linewidths=0,
                edgecolors="none",
            )

        if legend and not non_na.empty:
            mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
            mappable.set_array(non_na["value"].to_numpy())
            ax.figure.colorbar(mappable, ax=ax)

    _style_raster_axes(ax, title=title)


def plot_umap(
    adata: AnnData,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    color_source: Optional[Literal["obs", "var", "obsm"]] = "obs",
    color_type: Optional[Literal["auto", "categorical", "continuous"]] = "auto",
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
    sampling: Literal["none", "random"] = "none",
    sample_n: Optional[int] = None,
    sampling_seed: int = 37,
    tooltips: Literal["none", "default"] = "none",
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
    color_type
        Override the automatic color classification. ``"auto"`` (default) infers the type
        from the data: numeric arrays become continuous gradients, string/category arrays
        become discrete. Use ``"categorical"`` to force discrete coloring for numeric
        labels such as Leiden cluster integers. Use ``"continuous"`` to force gradient
        coloring regardless of dtype.
    basis
        Key in ``adata.obsm`` containing 2D coordinates.
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
        Continuous plots default to ``"sort"`` when ``None``.
    na_color
        Color for missing values when categorical or continuous values contain NA.
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
    sampling
        lets-plot point sampling strategy. Defaults to ``"none"``.
    sample_n
        Number of sampled points when ``sampling="random"``.
    sampling_seed
        Seed used for lets-plot random sampling.
    tooltips
        lets-plot tooltip mode. Defaults to ``"none"`` for faster static rendering.

    Returns
    -------
    lets_plot.PlotSpec
        A lets-plot object.
    """

    _validate_sampling_args(sampling, sample_n, tooltips)

    try:
        from lets_plot import (
            aes,
            geom_point,
            geom_text,
            ggplot,
            ggsize,
            labs,
            sampling_random,
            scale_alpha_identity,
            scale_color_gradientn,
            scale_color_identity,
            scale_color_manual,
            theme,
            theme_void,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("lets-plot is required for static UMAP plotting.") from exc

    ctx = _prepare_umap_context(
        adata,
        color=color,
        color_source=color_source,
        color_type=color_type,
        basis=basis,
        alpha=alpha,
        fig_dpi=fig_dpi,
        figsize=figsize,
        trans_attr=trans_attr,
        trans_fac=trans_fac,
        trans_th=trans_th,
        color_slot=color_slot,
    )

    point_kwargs = _point_render_kwargs(
        size=size,
        alpha_arg=ctx.alpha_arg,
        sampling=sampling,
        sample_n=sample_n,
        sampling_seed=sampling_seed,
        tooltips=tooltips,
        sampling_random=sampling_random,
    )

    def _base_aes(color_key: Optional[str] = None):
        mapping_kwargs = {}
        if color_key is not None:
            mapping_kwargs["color"] = color_key
        if ctx.alpha_is_array:
            mapping_kwargs["alpha"] = "alpha"
        return aes("x", "y", **mapping_kwargs)

    if ctx.kind == "none":
        plot_df = _build_base_frame(ctx)
        plot = ggplot() + geom_point(
            data=plot_df,
            mapping=_base_aes(),
            color="#4c72b0",
            inherit_aes=False,
            **point_kwargs,
        )

    elif ctx.kind == "rgb":
        plot_df = _build_base_frame(ctx)
        plot_df["color"] = _prepare_rgb_values(ctx.values)
        plot = ggplot(plot_df, _base_aes("color")) + geom_point(show_legend=legend, **point_kwargs)
        plot = plot + scale_color_identity()

    elif ctx.kind == "categorical":
        _, labels, _, color_map = _prepare_categorical_payload(
            ctx.values,
            categories=ctx.categories,
            palette=palette,
            na_color=na_color,
        )
        plot_df = _build_base_frame(ctx)
        plot_df["color"] = labels
        if hide_na and "NA" in plot_df["color"].values:
            plot_df = plot_df[plot_df["color"] != "NA"].copy()
        plot = ggplot(plot_df, _base_aes("color")) + geom_point(**point_kwargs)
        plot = plot + scale_color_manual(values=color_map)
        if add_text_labels and not plot_df.empty:
            label_df = _compute_label_positions(plot_df, nudge_text_labels=nudge_text_labels)
            plot = plot + geom_text(
                data=label_df,
                mapping=aes("x", "y", label="color"),
                size=label_text_size,
                color="black",
                show_legend=False,
                inherit_aes=False,
            )

    else:
        values = _prepare_continuous_values(ctx.values, vmin=vmin, vmax=vmax)
        base_df = _build_base_frame(ctx)
        plot_df = base_df.copy()
        plot_df["value"] = values
        non_na = plot_df[~np.isnan(values)].copy()
        na_df = base_df[np.isnan(values)].copy()
        order_use = "sort" if order is None else order
        non_na = apply_point_order(non_na, order_use, values[~np.isnan(values)])
        plot = ggplot(non_na, _base_aes("value")) + geom_point(**point_kwargs)
        if not na_df.empty:
            plot = plot + geom_point(
                data=na_df,
                mapping=_base_aes(),
                color=na_color,
                show_legend=False,
                inherit_aes=False,
                **point_kwargs,
            )
        plot = plot + scale_color_gradientn(colors=normalize_cmap_spec(cmap))

    if ctx.alpha_is_array:
        plot = plot + scale_alpha_identity()

    plot = plot + theme_void() + ggsize(ctx.width_px, ctx.height_px)
    plot = plot + theme(legend_position="right" if legend else "none")

    if title:
        plot = plot + labs(title=title)

    return plot


def plot_umap_raster(
    adata: AnnData,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    color_source: Optional[Literal["obs", "var", "obsm"]] = "obs",
    color_type: Optional[Literal["auto", "categorical", "continuous"]] = "auto",
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
    """Plot a rasterized static UMAP embedding using Matplotlib Agg."""

    fig = _new_raster_figure(figsize=figsize, fig_dpi=fig_dpi)
    ax = fig.add_subplot(111)

    ctx = _prepare_umap_context(
        adata,
        color=color,
        color_source=color_source,
        color_type=color_type,
        basis=basis,
        alpha=alpha,
        fig_dpi=fig_dpi,
        figsize=figsize,
        trans_attr=trans_attr,
        trans_fac=trans_fac,
        trans_th=trans_th,
        color_slot=color_slot,
    )

    _render_umap_raster(
        ax,
        ctx,
        cmap=cmap,
        palette=palette,
        size=size,
        legend=legend,
        title=title,
        vmin=vmin,
        vmax=vmax,
        order=order,
        na_color=na_color,
        hide_na=hide_na,
        add_text_labels=add_text_labels,
        label_text_size=label_text_size,
        nudge_text_labels=nudge_text_labels,
    )

    return fig


def plot_umap_interactive(
    adata: AnnData,
    color: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    color_source: Optional[Literal["obs", "var", "obsm"]] = None,
    color_type: Optional[Literal["auto", "categorical", "continuous"]] = "auto",
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
    """Plot an interactive UMAP embedding using Plotly WebGL."""

    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotly is required for interactive UMAP plotting.") from exc

    values, kind, categories = _resolve_color_input(
        adata,
        color=color,
        color_source=color_source,
        color_slot=None,
        color_type=color_type,
        layer=layer,
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
        plot_df["color"] = _prepare_rgb_values(values)
        color_args["color_discrete_map"] = {
            value: value for value in plot_df["color"].unique().tolist()
        }
        color_key = "color"
    elif kind == "categorical":
        _, labels, _, color_map = _prepare_categorical_payload(
            values,
            categories=categories,
            palette=palette,
            na_color="#cccccc",
        )
        plot_df["color"] = labels
        color_args["color_discrete_map"] = color_map
        color_key = "color"
    else:
        values = _prepare_continuous_values(values, vmin=vmin, vmax=vmax)
        plot_df["value"] = values
        color_args["color_continuous_scale"] = normalize_cmap_spec(cmap)
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
