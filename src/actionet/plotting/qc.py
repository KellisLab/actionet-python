"""QC plotting utilities for ACTIONet."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .utils import build_discrete_color_map


def _resolve_group_labels(
    adata: AnnData, groupby: Optional[str]
) -> tuple[pd.Series, list[str]]:
    if groupby is None:
        labels = pd.Series(["all"] * adata.n_obs)
    else:
        if groupby not in adata.obs:
            raise ValueError(f"groupby '{groupby}' not found in adata.obs.")
        labels = adata.obs[groupby]
    labels = labels.astype("category")
    if labels.isna().any():
        labels = labels.cat.add_categories(["NA"]).fillna("NA")
    categories = list(labels.cat.categories)
    return labels.astype(str), categories


def _resolve_values(
    adata: AnnData, key: Union[str, Sequence, np.ndarray, pd.Series]
) -> np.ndarray:
    if isinstance(key, str):
        if key not in adata.obs:
            raise ValueError(f"key '{key}' not found in adata.obs.")
        return pd.to_numeric(adata.obs[key], errors="coerce").to_numpy()
    values = np.asarray(key)
    if values.shape[0] != adata.n_obs:
        raise ValueError("values length does not match number of observations.")
    return values.astype(float)


def _normalize_keys(
    keys: Union[str, Sequence[Union[str, Sequence, np.ndarray, pd.Series]], np.ndarray, pd.Series]
) -> list[Union[str, Sequence, np.ndarray, pd.Series]]:
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, (np.ndarray, pd.Series)):
        return [keys]
    return list(keys)


def _apply_log_transform(values: np.ndarray, log_trans: str) -> np.ndarray:
    if log_trans == "none":
        return values
    safe = values.copy()
    safe[safe == 0] = 1
    if log_trans == "log":
        return np.log(safe)
    if log_trans == "log2":
        return np.log2(safe)
    if log_trans == "log10":
        return np.log10(safe)
    raise ValueError("log_trans must be one of 'none', 'log', 'log2', 'log10'.")


def plot_qc_violin(
    adata: AnnData,
    keys: Union[str, Sequence[Union[str, Sequence, np.ndarray, pd.Series]], np.ndarray, pd.Series],
    groupby: Optional[str] = None,
    palette: Optional[Union[str, Sequence[str], dict]] = None,
    log_trans: Literal["none", "log", "log2", "log10"] = "none",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: tuple[float, float] = (6, 4),
    fig_dpi: float = 100.0,
    show_boxplot: bool = True,
    legend: bool = False,
):
    """Plot QC violin plots for per-cell metrics.

    Parameters
    ----------
    adata
        AnnData with per-cell metrics stored in ``adata.obs``.
    keys
        A key (or list of keys) in ``adata.obs`` or a vector-like per-cell array.
    groupby
        Optional ``adata.obs`` column to group violins by.
    palette
        Discrete palette name, list of colors, or dict mapping category to color.
    log_trans
        Log transform to apply to values: ``"none"``, ``"log"``, ``"log2"``, or ``"log10"``.
    title
        Optional plot title.
    x_label
        X-axis label (defaults to blank).
    y_label
        Y-axis label (defaults to the key name).
    figsize
        Figure size in inches.
    fig_dpi
        DPI used to convert inches to pixels for lets-plot.
    show_boxplot
        If True, overlay a boxplot inside the violin.
    legend
        If True, show the legend (default False).

    Returns
    -------
    lets_plot.PlotSpec or dict
        A lets-plot object, or a dict of plots if ``keys`` is a list.
    """

    key_list = _normalize_keys(keys)
    if len(key_list) > 1:
        return {
            key: plot_qc_violin(
                adata,
                key,
                groupby=groupby,
                palette=palette,
                log_trans=log_trans,
                title=title,
                x_label=x_label,
                y_label=y_label,
                figsize=figsize,
                fig_dpi=fig_dpi,
                show_boxplot=show_boxplot,
                legend=legend,
            )
            for key in key_list
        }

    try:
        from lets_plot import (
            aes,
            element_rect,
            element_text,
            geom_boxplot,
            geom_violin,
            ggplot,
            ggsize,
            labs,
            scale_fill_manual,
            theme,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("lets-plot is required for QC violin plotting.") from exc

    key = key_list[0]
    values = _resolve_values(adata, key)
    labels, categories = _resolve_group_labels(adata, groupby)
    values = _apply_log_transform(values, log_trans)

    plot_df = pd.DataFrame({"label": labels.to_numpy(), "value": values})
    plot_df["label"] = plot_df["label"].astype(str)

    color_map = build_discrete_color_map(categories, palette)
    color_map = {str(key): value for key, value in color_map.items()}

    base = ggplot(plot_df, aes(x="label", y="value", fill="label"))
    base = base + geom_violin(
        quantile_lines=True,
        quantiles=[0.25, 0.5, 0.75],
        scale="width",
    )
    if show_boxplot:
        base = base + geom_boxplot(width=0.2, fill="white", alpha=0.7)
    base = base + scale_fill_manual(values=color_map)

    if x_label is None:
        x_label = ""
    if y_label is None:
        if isinstance(key, str):
            y_label = key
        else:
            y_label = "value"

    plot = base + labs(x=x_label, y=y_label)
    plot = plot + theme(
        axis_text_x=element_text(angle=90, vjust=0.5, hjust=1),
        panel_border=element_rect(color="black", fill=None),
        legend_position="right" if legend else "none",
    )

    width_px = float(figsize[0]) * fig_dpi
    height_px = float(figsize[1]) * fig_dpi
    plot = plot + ggsize(width_px, height_px)

    if title:
        plot = plot + labs(title=title) + theme(plot_title=element_text(hjust=0.5))

    return plot

