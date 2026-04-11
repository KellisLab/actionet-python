"""QC plotting utilities for ACTIONet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .utils import build_discrete_color_map, sort_categories

if TYPE_CHECKING:
    from .._matrix_source import MatrixSource
    from ..lazy_transform import LazyTransform


# ---------------------------------------------------------------------------
# Mitochondrial feature helpers (mirrors R .get_mito_feats)
# ---------------------------------------------------------------------------

#: Species aliases accepted by :func:`get_mito_feats`.
_SPECIES_ALIASES: dict[str, str] = {
    "human": "hsapiens",
    "mouse": "mmusculus",
    "hsapiens": "hsapiens",
    "mmusculus": "mmusculus",
}

#: Supported ``id_type`` values and the matching column in the feature table.
_ID_TYPE_COLUMNS: dict[str, str] = {
    "gene_name": "gene_name",
    "ensembl_id": "ensembl_id",
}


def get_mito_feats(
    id_type: Literal["gene_name", "ensembl_id"] = "gene_name",
    species: Literal["hsapiens", "mmusculus", "human", "mouse"] = "hsapiens",
    protein_coding: bool = False,
) -> list[str]:
    """Return bundled mitochondrial feature names for a target organism.

    The feature table is sourced from the ``mito_human_mouse_biomart_240924``
    dataset in the R ``actionet`` package and covers all annotated mitochondrial
    transcripts: protein-coding genes, Mt_tRNA, and Mt_rRNA.

    Parameters
    ----------
    id_type
        Which identifier column to return:

        - ``"gene_name"`` (default) — HGNC / MGI gene symbols
          (e.g. ``MT-ND1`` for human, ``mt-Nd1`` for mouse).
        - ``"ensembl_id"`` — Ensembl stable gene IDs.
    species
        Target organism.  Accepts ``"hsapiens"``/``"human"`` or
        ``"mmusculus"``/``"mouse"``.
    protein_coding
        If ``True``, restrict to protein-coding features only (excludes
        Mt_tRNA and Mt_rRNA).

    Returns
    -------
    list of str
        Feature identifiers, in the same order as the bundled table.

    Raises
    ------
    ValueError
        On unrecognised ``id_type`` or ``species``.

    Examples
    --------
    >>> from actionet.plotting.qc import get_mito_feats
    >>> get_mito_feats()                          # human gene symbols
    >>> get_mito_feats("ensembl_id", "mouse")     # mouse Ensembl IDs
    >>> get_mito_feats(protein_coding=True)        # protein-coding only
    """
    id_type_lower = id_type.lower()
    if id_type_lower not in _ID_TYPE_COLUMNS:
        raise ValueError(
            f"id_type must be one of {list(_ID_TYPE_COLUMNS)}; got {id_type!r}."
        )

    species_lower = species.lower()
    if species_lower not in _SPECIES_ALIASES:
        raise ValueError(
            f"species must be one of {list(_SPECIES_ALIASES)}; got {species!r}."
        )
    canonical_species = _SPECIES_ALIASES[species_lower]

    from .._data import load_mito_features

    df = load_mito_features()
    mask = df["species"] == canonical_species
    if protein_coding:
        mask = mask & (df["feature_type"] == "protein_coding")

    col = _ID_TYPE_COLUMNS[id_type_lower]
    return df.loc[mask, col].tolist()


# ---------------------------------------------------------------------------
# Feature-abundance computation (mirrors R getFeatureAbundance)
# ---------------------------------------------------------------------------

def _resolve_feature_indices(
    adata: AnnData,
    features: Union[str, Sequence[str], None],
    features_use: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return column indices for ``features``, or ``None`` when ``features="all"``.

    Parameters
    ----------
    adata
        AnnData object.
    features
        ``"all"`` to use all features; otherwise a sequence of feature names
        matched against the label vector selected by ``features_use``.
        Features not present in the label vector are silently skipped
        (matching R ``which(features_use %in% features)`` semantics).
    features_use
        Column of ``adata.var`` whose values are used as feature labels.
        ``None`` (default) uses ``adata.var_names`` (the index).

    Raises
    ------
    ValueError
        If ``features_use`` is not a column in ``adata.var``, if ``features``
        is empty, or if *none* of the requested features matched at all.
    """
    if features is None or (isinstance(features, str) and features == "all"):
        return None

    feat_list = [features] if isinstance(features, str) else list(features)
    if len(feat_list) == 0:
        raise ValueError("'features' must be 'all' or a non-empty list of feature names.")

    if features_use is None:
        label_values = adata.var_names.tolist()
        label_desc = "adata.var_names"
    else:
        if features_use not in adata.var.columns:
            raise ValueError(
                f"features_use column '{features_use}' not found in adata.var. "
                f"Available columns: {list(adata.var.columns)}"
            )
        label_values = adata.var[features_use].astype(str).tolist()
        label_desc = f"adata.var['{features_use}']"

    # Build first-occurrence lookup (mirrors R match() / which() semantics).
    # pd.Index.get_indexer() raises InvalidIndexError on duplicate labels.
    first_occurrence: dict[str, int] = {}
    for i, lab in enumerate(label_values):
        if lab not in first_occurrence:
            first_occurrence[lab] = i

    found = np.array(
        [first_occurrence[f] for f in feat_list if f in first_occurrence],
        dtype=np.int64,
    )

    if found.size == 0:
        missing_sample = [f for f in feat_list if f not in first_occurrence]
        raise ValueError(
            f"No features matched {label_desc}. "
            f"First few requested: {missing_sample[:5]}"
            + (" ..." if len(missing_sample) > 5 else "")
        )

    return found


def _apply_lazy_transform_to_block(
    block,
    row_slice: slice,
    row_scale_factors: np.ndarray,
    apply_log1p: bool,
    log_scale: float,
) -> np.ndarray:
    """Apply lazy logcount transform to a single row block (in-place safe)."""
    scales = row_scale_factors[row_slice]  # (n_rows,)
    if sp.issparse(block):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float64)
    block = block * scales[:, np.newaxis]
    if apply_log1p:
        np.log1p(block, out=block)
        if log_scale != 1.0:
            block *= log_scale
    return block


def get_feature_abundance(
    adata: AnnData,
    features: Union[str, Sequence[str]] = "all",
    *,
    features_use: Optional[str] = None,
    nonzero: bool = False,
    metric: Literal["counts", "fraction", "percent", "ratio"] = "counts",
    layer: Optional[str] = None,
    lazy_transform: Optional["LazyTransform"] = None,
    groupby: Optional[str] = None,
    groups_use: Optional[Sequence[str]] = None,
    chunk_size: int = 4096,
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """Compute per-cell feature abundance metrics, streaming over backed data.

    This is the Python equivalent of the R ``getFeatureAbundance()`` function.
    It supports in-memory and backed AnnData objects without materialising the
    full matrix.  When ``lazy_transform`` is provided the raw counts are
    normalized on-the-fly (identical to the transform used for SVD / ACTION).

    Parameters
    ----------
    adata
        AnnData object.
    features
        ``"all"`` to use all features, or a list of feature names to subset.
        When ``features="all"`` only the ``"counts"`` metric is valid (total UMI
        per cell), matching R semantics.
    features_use
        Column of ``adata.var`` whose values are used as feature labels when
        matching ``features``.  ``None`` (default) matches against
        ``adata.var_names`` (the index).  Useful when ``var_names`` are
        numeric IDs but a ``"gene_name"`` or ``"ensembl_id"`` column holds the
        human-readable labels.
    nonzero
        If ``True``, binarise the matrix before aggregating so that the result
        is the number (or fraction) of *detected* features per cell.
    metric
        Aggregation metric:

        - ``"counts"`` — raw sum per cell over selected features.
        - ``"fraction"`` — sum over selected features divided by total UMI.
        - ``"percent"`` — 100 × fraction.
        - ``"ratio"`` — selected sum / (total sum − selected sum).
    layer
        Layer to read from.  ``None`` uses ``adata.X``.  Mutually exclusive
        with ``lazy_transform``.
    lazy_transform
        Pre-initialized :class:`~actionet.LazyTransform` for on-the-fly
        normalization of backed ``.X``.  Mutually exclusive with ``layer``.
    groupby
        ``adata.obs`` column used to split output by group.  When ``None``
        a single array is returned; when set a ``dict[group_name, array]``
        is returned.
    groups_use
        Subset of group labels to include (only used when ``groupby`` is set).
    chunk_size
        Rows per streaming chunk (backed data).

    Returns
    -------
    ndarray or dict of ndarray
        Per-cell abundance values (length = ``adata.n_obs`` or subset per
        group), dtype float64.

    Raises
    ------
    ValueError
        On parameter conflicts or missing keys.
    """
    from .._matrix_source import MatrixSource
    from ..lazy_transform import _validate_lazy_transform, _resolve_lazy_backed_transform

    if features == "all" and metric != "counts":
        raise ValueError(
            "metric must be 'counts' when features='all'. "
            "Use a feature list for fraction/percent/ratio metrics."
        )

    _validate_lazy_transform(lazy_transform, layer=layer, source=MatrixSource(adata, layer=layer))

    source = MatrixSource(adata, layer=layer)
    col_indices = _resolve_feature_indices(adata, features, features_use=features_use)

    # Resolve lazy transform scales (None if not used)
    row_scale_factors: Optional[np.ndarray] = None
    apply_log1p = False
    log_scale = 1.0
    if lazy_transform is not None:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=chunk_size,
        )

    # --- streaming pass 1: per-cell total sum (needed for fraction / ratio) ---
    if metric in ("fraction", "percent", "ratio"):
        total_sums = _streaming_row_sums(
            source,
            col_indices=None,  # always total
            nonzero=nonzero,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
            chunk_size=chunk_size,
        )
    else:
        total_sums = None

    # --- streaming pass 2: per-cell feature sum ---
    feature_sums = _streaming_row_sums(
        source,
        col_indices=col_indices,
        nonzero=nonzero,
        row_scale_factors=row_scale_factors,
        apply_log1p=apply_log1p,
        log_scale=log_scale,
        chunk_size=chunk_size,
    )

    # --- compute metric ---
    if metric == "counts":
        per_cell = feature_sums
    elif metric == "fraction":
        per_cell = np.divide(
            feature_sums, total_sums,
            out=np.zeros_like(feature_sums),
            where=total_sums > 0,
        )
    elif metric == "percent":
        per_cell = 100.0 * np.divide(
            feature_sums, total_sums,
            out=np.zeros_like(feature_sums),
            where=total_sums > 0,
        )
    elif metric == "ratio":
        denom = total_sums - feature_sums
        per_cell = np.divide(
            feature_sums, denom,
            out=np.zeros_like(feature_sums),
            where=denom > 0,
        )
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    # --- split by group if requested ---
    if groupby is None:
        return per_cell

    if groupby not in adata.obs:
        raise ValueError(f"groupby '{groupby}' not found in adata.obs.")

    labels = adata.obs[groupby].astype(str)
    unique_labels = list(dict.fromkeys(labels))  # insertion-order unique

    if groups_use is not None:
        groups_use_set = set(str(g) for g in groups_use)
        unique_labels = [g for g in unique_labels if g in groups_use_set]

    out: dict[str, np.ndarray] = {}
    for grp in unique_labels:
        mask = labels == grp
        out[grp] = per_cell[mask.to_numpy()]

    return out


def _streaming_row_sums(
    source: "MatrixSource",
    *,
    col_indices: Optional[np.ndarray],
    nonzero: bool,
    row_scale_factors: Optional[np.ndarray],
    apply_log1p: bool,
    log_scale: float,
    chunk_size: int,
) -> np.ndarray:
    """Stream-compute per-row sums with optional column subset, binarisation, and transform."""
    out = np.zeros(source.n_obs, dtype=np.float64)

    for chunk in source.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices):
        block = chunk.block
        start, end = chunk.start, chunk.end

        # Apply lazy transform before any other operation (only when scales are given)
        if row_scale_factors is not None:
            block = _apply_lazy_transform_to_block(
                block,
                slice(start, end),
                row_scale_factors,
                apply_log1p,
                log_scale,
            )
        elif sp.issparse(block):
            block = block  # keep sparse; binarise below if needed
        else:
            block = np.asarray(block, dtype=np.float64)

        if nonzero:
            if sp.issparse(block):
                block = (block > 0).astype(np.float64)
            else:
                block = (block > 0).astype(np.float64)

        if sp.issparse(block):
            out[start:end] = np.asarray(block.sum(axis=1)).ravel()
        else:
            out[start:end] = np.asarray(block, dtype=np.float64).sum(axis=1)

    return out


# ---------------------------------------------------------------------------
# Shared helpers (also used by the legacy keys path)
# ---------------------------------------------------------------------------

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
    categories = sort_categories(list(labels.cat.categories))
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


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_qc_violin(
    adata: AnnData,
    # --- on-the-fly computation path (mirrors R plotFeatureDist) ---
    features: Union[str, Sequence[str], None] = None,
    groupby: Optional[str] = None,
    features_use: Optional[str] = None,
    metric: Literal["counts", "fraction", "percent", "ratio"] = "counts",
    nonzero: bool = False,
    log_trans: Literal["none", "log", "log2", "log10"] = "none",
    layer: Optional[str] = None,
    lazy_transform: Optional["LazyTransform"] = None,
    groups_use: Optional[Sequence[str]] = None,
    chunk_size: int = 4096,
    # --- precomputed path (legacy / fast path) ---
    keys: Optional[Union[str, Sequence[Union[str, Sequence, np.ndarray, pd.Series]], np.ndarray, pd.Series]] = None,
    # --- visual options ---
    palette: Optional[Union[str, Sequence[str], dict]] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: tuple[float, float] = (6, 4),
    fig_dpi: float = 100.0,
    show_boxplot: bool = True,
    legend: bool = False,
):
    """Plot QC violin distributions for per-cell metrics.

    Two usage modes are supported:

    **On-the-fly mode** — compute feature distributions directly from the
    expression matrix without pre-computing anything.  Works on both in-memory
    and backed (read-only) AnnData objects.  Use this when you have not already
    stored QC metrics in ``adata.obs``.

    **Precomputed mode** — ``keys`` points to existing columns in ``adata.obs``
    (or raw arrays).  Use this when QC stats have already been computed and
    stored.

    Parameters
    ----------
    adata
        AnnData object.
    features
        *On-the-fly mode only.*  Features to aggregate over.  ``"all"`` uses
        every feature (only valid for ``metric="counts"``); a list of feature
        names restricts aggregation to that subset.  When ``None`` and no
        ``keys`` are given, ``"all"`` is assumed.
    groupby
        ``adata.obs`` column used to split violins by group.  ``None`` plots
        a single ``"all"`` violin.
    features_use
        Column of ``adata.var`` whose values are matched against ``features``.
        ``None`` (default) uses ``adata.var_names`` (the index).  Set this
        when ``var_names`` are numeric/Ensembl IDs but ``features`` contains
        gene symbols stored in a ``adata.var`` column (or vice versa).
    metric
        Aggregation metric for on-the-fly computation:

        - ``"counts"`` — raw sum per cell over selected features (total UMI).
        - ``"fraction"`` — feature sum / total UMI.
        - ``"percent"`` — 100 × fraction.
        - ``"ratio"`` — feature sum / (total − feature sum).
    nonzero
        Binarise the matrix before aggregating so the result measures the
        number/fraction of *detected* (non-zero) features.
    log_trans
        Log transform applied to the plotted values:
        ``"none"``, ``"log"``, ``"log2"``, or ``"log10"``.
        When ``log_trans != "none"`` zeros are replaced with 1 before taking
        the logarithm (matching R behaviour).
    layer
        Layer to read from.  ``None`` uses ``adata.X``.  Mutually exclusive
        with ``lazy_transform``.
    lazy_transform
        Pre-initialized :class:`~actionet.LazyTransform` for on-the-fly
        normalization of a backed ``.X`` (library-size scaling + log1p).
        Mutually exclusive with ``layer``.
    groups_use
        Subset of group labels to include in the plot.  ``None`` includes all.
    chunk_size
        Streaming chunk size (rows) for backed data.
    keys
        *Precomputed mode.*  A key or list of keys in ``adata.obs``, or raw
        per-cell arrays.  When provided, ``features``, ``layer``,
        ``lazy_transform``, ``metric``, and ``nonzero`` are ignored.
        If multiple keys are given a ``dict`` of plots is returned.
    palette
        Discrete palette name, list of colors, or ``dict`` mapping group label
        to color.
    title
        Optional plot title.
    x_label
        X-axis label.  Defaults to blank.
    y_label
        Y-axis label.  Defaults to the key name or metric string.
    figsize
        Figure size in inches ``(width, height)``.
    fig_dpi
        DPI used to convert ``figsize`` to pixel dimensions for lets-plot.
    show_boxplot
        Overlay a miniature boxplot inside each violin.
    legend
        Show the fill legend.

    Returns
    -------
    lets_plot.PlotSpec or dict
        A single lets-plot object, or a ``dict`` keyed by ``keys`` when
        multiple keys are provided.

    Examples
    --------
    On-the-fly, all cells, total UMI:

    >>> act.plot_qc_violin(adata)

    Mitochondrial fraction by cluster (backed, read-only):

    >>> lt = act.create_lazy_transform(adata)
    >>> act.plot_qc_violin(
    ...     adata,
    ...     features=mito_genes,
    ...     metric="fraction",
    ...     groupby="cluster",
    ...     lazy_transform=lt,
    ... )

    Using pre-stored QC metrics:

    >>> act.plot_qc_violin(adata, keys=["n_counts", "pct_mito"], groupby="batch")
    """
    # ------------------------------------------------------------------
    # Dispatch: precomputed keys path
    # ------------------------------------------------------------------
    if keys is not None:
        key_list = _normalize_keys(keys)
        if len(key_list) > 1:
            return {
                k: plot_qc_violin(
                    adata,
                    keys=k,
                    groupby=groupby,
                    groups_use=groups_use,
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
                for k in key_list
            }

        key = key_list[0]
        per_cell = _resolve_values(adata, key)
        per_cell = _apply_log_transform(per_cell, log_trans)
        labels, categories = _resolve_group_labels(adata, groupby)

        if groups_use is not None:
            groups_use_set = set(str(g) for g in groups_use)
            mask = labels.isin(groups_use_set)
            labels = labels[mask]
            per_cell = per_cell[mask.to_numpy()]
            categories = [c for c in categories if c in groups_use_set]

        if y_label is None:
            y_label = key if isinstance(key, str) else "value"

        return _build_violin_plot(
            labels=labels,
            values=per_cell,
            categories=categories,
            palette=palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            figsize=figsize,
            fig_dpi=fig_dpi,
            show_boxplot=show_boxplot,
            legend=legend,
        )

    # ------------------------------------------------------------------
    # Dispatch: on-the-fly computation path
    # ------------------------------------------------------------------
    eff_features: Union[str, Sequence[str]] = "all" if features is None else features

    abundance = get_feature_abundance(
        adata,
        features=eff_features,
        features_use=features_use,
        nonzero=nonzero,
        metric=metric,
        layer=layer,
        lazy_transform=lazy_transform,
        groupby=groupby,
        groups_use=groups_use,
        chunk_size=chunk_size,
    )

    # Determine y-axis label
    if y_label is None:
        if log_trans != "none":
            y_label = f"{log_trans}({metric})"
        else:
            y_label = metric

    if groupby is None:
        # abundance is a flat array; wrap into a single-group structure
        all_values = _apply_log_transform(np.asarray(abundance, dtype=np.float64), log_trans)
        labels = pd.Series(["all"] * adata.n_obs, dtype=str)
        categories = ["all"]
        if groups_use is not None:
            if "all" not in [str(g) for g in groups_use]:
                labels = pd.Series([], dtype=str)
                all_values = np.array([], dtype=np.float64)
                categories = []
        return _build_violin_plot(
            labels=labels,
            values=all_values,
            categories=categories,
            palette=palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            figsize=figsize,
            fig_dpi=fig_dpi,
            show_boxplot=show_boxplot,
            legend=legend,
        )

    # abundance is a dict; flatten into a long-form DataFrame
    assert isinstance(abundance, dict)
    categories = sort_categories(list(abundance.keys()))
    all_labels: list[str] = []
    all_values_list: list[np.ndarray] = []
    for grp in categories:
        vals = _apply_log_transform(np.asarray(abundance[grp], dtype=np.float64), log_trans)
        all_labels.extend([grp] * len(vals))
        all_values_list.append(vals)

    labels_s = pd.Series(all_labels, dtype=str)
    values_arr = np.concatenate(all_values_list) if all_values_list else np.array([], dtype=np.float64)

    return _build_violin_plot(
        labels=labels_s,
        values=values_arr,
        categories=categories,
        palette=palette,
        title=title,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        fig_dpi=fig_dpi,
        show_boxplot=show_boxplot,
        legend=legend,
    )


# ---------------------------------------------------------------------------
# lets-plot rendering helper
# ---------------------------------------------------------------------------

def _build_violin_plot(
    *,
    labels: pd.Series,
    values: np.ndarray,
    categories: list[str],
    palette,
    title: Optional[str],
    x_label: Optional[str],
    y_label: Optional[str],
    figsize: tuple[float, float],
    fig_dpi: float,
    show_boxplot: bool,
    legend: bool,
):
    """Construct a lets-plot violin figure from long-form data."""
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
            scale_x_discrete,
            theme,
        )
    except ImportError as exc:
        raise ImportError("lets-plot is required for QC violin plotting.") from exc

    plot_df = pd.DataFrame({"label": labels.to_numpy(), "value": values})
    plot_df["label"] = plot_df["label"].astype(str)

    color_map = build_discrete_color_map(categories, palette)
    color_map = {str(k): v for k, v in color_map.items()}

    base = ggplot(plot_df, aes(x="label", y="value", fill="label"))
    base = base + geom_violin(
        quantile_lines=True,
        quantiles=[0.25, 0.5, 0.75],
        scale="width",
    )
    if show_boxplot:
        base = base + geom_boxplot(width=0.2, fill="white", alpha=0.7)
    base = base + scale_fill_manual(values=color_map)
    base = base + scale_x_discrete(limits=[str(c) for c in categories])

    plot = base + labs(
        x="" if x_label is None else x_label,
        y="" if y_label is None else y_label,
    )
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


# ---------------------------------------------------------------------------
# Mitochondrial distribution plot (mirrors R plotMitoDist)
# ---------------------------------------------------------------------------

def plot_mito_violin(
    adata: AnnData,
    groupby: Optional[str] = None,
    id_type: Literal["gene_name", "ensembl_id"] = "gene_name",
    species: Literal["hsapiens", "mmusculus", "human", "mouse"] = "hsapiens",
    protein_coding: bool = False,
    features_use: Optional[str] = None,
    metric: Literal["counts", "fraction", "percent", "ratio"] = "fraction",
    log_trans: Literal["none", "log", "log2", "log10"] = "none",
    layer: Optional[str] = None,
    lazy_transform: Optional["LazyTransform"] = None,
    groups_use: Optional[Sequence[str]] = None,
    chunk_size: int = 4096,
    palette: Optional[Union[str, Sequence[str], dict]] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: tuple[float, float] = (6, 4),
    fig_dpi: float = 100.0,
    show_boxplot: bool = True,
    legend: bool = False,
):
    """Plot mitochondrial transcript distribution as a violin plot.

    A convenience shim over :func:`plot_qc_violin` that automatically
    resolves mitochondrial feature names from the bundled reference table
    (``mito_human_mouse_biomart_240924``), identical to the R
    ``plotMitoDist()`` function.

    Parameters
    ----------
    adata
        AnnData object.
    groupby
        ``adata.obs`` column used to split violins by group.
    id_type
        Feature identifier type used to match ``adata.var_names``:

        - ``"gene_name"`` (default) — HGNC/MGI gene symbols.
        - ``"ensembl_id"`` — Ensembl stable gene IDs.
    species
        Target organism: ``"hsapiens"``/``"human"`` or
        ``"mmusculus"``/``"mouse"``.
    protein_coding
        If ``True``, restrict to protein-coding mitochondrial genes only
        (excludes Mt_tRNA and Mt_rRNA entries).
    features_use
        Column of ``adata.var`` whose values are matched against the resolved
        mitochondrial feature names.  ``None`` (default) matches against
        ``adata.var_names``.  Must be consistent with ``id_type``: use
        ``features_use="gene_name"`` when ``var_names`` are Ensembl IDs but a
        gene-symbol column exists, or ``features_use="ensembl_id"`` for the
        reverse.
    metric
        Aggregation metric.  Defaults to ``"fraction"`` (mitochondrial
        fraction of total UMI), matching R default behaviour.

        - ``"counts"`` — raw mito UMI sum per cell.
        - ``"fraction"`` — mito sum / total UMI.
        - ``"percent"`` — 100 × fraction.
        - ``"ratio"`` — mito sum / (total − mito sum).
    log_trans
        Log transform applied to the plotted values.
    layer
        Layer to read from.  ``None`` uses ``adata.X``.
    lazy_transform
        Pre-initialized :class:`~actionet.LazyTransform` for on-the-fly
        normalization of backed ``.X``.
    groups_use
        Subset of group labels to include.
    chunk_size
        Streaming chunk size for backed data.
    palette
        Discrete palette name, list of colors, or ``dict``.
    title
        Plot title.  Defaults to ``"Mitochondrial features (<metric>)"``.
    x_label
        X-axis label.
    y_label
        Y-axis label.  Defaults to the metric string.
    figsize
        Figure size ``(width, height)`` in inches.
    fig_dpi
        DPI used to convert ``figsize`` to pixels for lets-plot.
    show_boxplot
        Overlay a miniature boxplot inside each violin.
    legend
        Show the fill legend.

    Returns
    -------
    lets_plot.PlotSpec
        A single violin plot.

    Examples
    --------
    Mitochondrial fraction by cluster, in-memory:

    >>> act.plot_mito_violin(adata, groupby="cluster")

    Mitochondrial percent from backed h5ad with lazy normalization:

    >>> lt = act.create_lazy_transform(adata)
    >>> act.plot_mito_violin(
    ...     adata,
    ...     groupby="cluster",
    ...     metric="percent",
    ...     lazy_transform=lt,
    ... )

    Protein-coding mito genes only (mouse, Ensembl IDs):

    >>> act.plot_mito_violin(
    ...     adata,
    ...     id_type="ensembl_id",
    ...     species="mouse",
    ...     protein_coding=True,
    ... )
    """
    mito_feats = get_mito_feats(id_type=id_type, species=species, protein_coding=protein_coding)

    if not mito_feats:
        raise ValueError(
            f"No mitochondrial features found for species={species!r}, "
            f"id_type={id_type!r}, protein_coding={protein_coding}."
        )

    if title is None:
        title = f"Mitochondrial features ({metric})"

    return plot_qc_violin(
        adata,
        features=mito_feats,
        features_use=features_use,
        groupby=groupby,
        metric=metric,
        nonzero=False,
        log_trans=log_trans,
        layer=layer,
        lazy_transform=lazy_transform,
        groups_use=groups_use,
        chunk_size=chunk_size,
        palette=palette,
        title=title,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        fig_dpi=fig_dpi,
        show_boxplot=show_boxplot,
        legend=legend,
    )
