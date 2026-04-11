"""Plotting utilities for ACTIONet."""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse as sp


_DEFAULT_DISCRETE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

_DISCRETE_PALETTES = {
    "tab20": _DEFAULT_DISCRETE,
    "Set1": [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ],
    "Set2": [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ],
    "Set3": [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    ],
}

_CONTINUOUS_PALETTES = {
    "viridis": [
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    ],
    "magma": [
        "#000004",
        "#1b0c41",
        "#4f0c6b",
        "#781c6d",
        "#a52c60",
        "#cf4446",
        "#ed6925",
        "#fb9b06",
        "#f7d13d",
        "#fcfdbf",
    ],
    "inferno": [
        "#000004",
        "#1b0c41",
        "#4a0c6b",
        "#781c6d",
        "#a52c60",
        "#cf4446",
        "#ed6925",
        "#fca50a",
        "#f6d746",
        "#fcffa4",
    ],
    "plasma": [
        "#0d0887",
        "#41049d",
        "#6a00a8",
        "#8f0da4",
        "#b12a90",
        "#cc4778",
        "#e16462",
        "#f2844b",
        "#fca636",
        "#fcce25",
    ],
    "cividis": [
        "#00204c",
        "#2c3f7c",
        "#4f5d8b",
        "#6f7b8c",
        "#8a9a8d",
        "#a7b98b",
        "#c5d881",
        "#e3f76f",
    ],
    "greys": [
        "#000000",
        "#2f2f2f",
        "#555555",
        "#7f7f7f",
        "#a6a6a6",
        "#c7c7c7",
        "#e0e0e0",
        "#ffffff",
    ],
    "BlGrRd": ["#2b6cb0", "#9e9e9e", "#c53030"],
    "RdYlBu": ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"],
    "Spectral": [
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",
    ],
}


def _is_array_like(value: object) -> bool:
    """Return True if value is a list/tuple/array-like container."""
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _to_numpy_1d(values: Union[Sequence, np.ndarray, pd.Series]) -> np.ndarray:
    """Coerce a 1D sequence or Series into a NumPy array."""
    if isinstance(values, pd.Series):
        return values.to_numpy()
    return np.asarray(values)


def _rgb_to_hex(colors: np.ndarray) -> list[str]:
    """Convert RGB array (0-1 or 0-255) to hex color strings."""
    arr = np.asarray(colors)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Expected an (n, 3) array of RGB values.")
    if arr.max() <= 1.0:
        arr = (arr * 255).round()
    arr = np.clip(arr, 0, 255).astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in arr]


def resolve_embedding(adata: AnnData, basis: str) -> np.ndarray:
    """Return 2D coordinates from adata.obsm for the given basis."""
    if basis not in adata.obsm:
        raise ValueError(
            f"Embedding '{basis}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[basis])
    if coords.shape[1] < 2:
        raise ValueError(f"Embedding '{basis}' must have at least 2 columns.")
    return coords[:, :2]


def resolve_color_vector(
    adata: AnnData,
    color: Optional[Union[str, Sequence, np.ndarray]],
    layer: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], str, Optional[pd.Index]]:
    """Resolve a color spec into values, kind, and categories if applicable."""
    if color is None:
        return None, "none", None

    if isinstance(color, str):
        if color in adata.obs:
            series = adata.obs[color]
            return series.to_numpy(), _infer_color_kind(series), series.astype("category").cat.categories
        if color in adata.var_names:
            idx = int(np.where(adata.var_names == color)[0][0])
            matrix = adata.layers[layer] if layer is not None else adata.X
            if sp.issparse(matrix):
                values = np.asarray(matrix[:, idx].toarray()).ravel()
            else:
                values = np.asarray(matrix[:, idx]).ravel()
            return values, "continuous", None
        if color in adata.obsm:
            values = np.asarray(adata.obsm[color])
            if values.ndim == 2 and values.shape[1] == 3:
                return values, "rgb", None
            if values.ndim == 2 and values.shape[1] == 1:
                return values[:, 0], "continuous", None
        raise ValueError(
            f"Color key '{color}' not found in adata.obs, adata.var_names, or adata.obsm."
        )

    if _is_array_like(color):
        values = _to_numpy_1d(color)
        if values.shape[0] != adata.n_obs:
            raise ValueError("Color vector length does not match number of observations.")
        return values, _infer_color_kind(values), None

    raise TypeError("Unsupported color specification.")


def _infer_color_kind(values: Union[pd.Series, np.ndarray]) -> str:
    """Infer whether values represent categorical, continuous, or rgb data."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_categorical_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        if unique_count <= 50 and pd.api.types.is_integer_dtype(series):
            return "categorical"
        return "continuous"
    return "categorical"


def build_discrete_color_map(
    categories: Sequence,
    palette: Optional[Union[str, Sequence[str], Mapping[object, str]]],
) -> dict:
    """Return a category-to-color map from a palette name, list, or dict."""
    if palette is None:
        colors = _DEFAULT_DISCRETE
    elif isinstance(palette, str):
        colors = _DISCRETE_PALETTES.get(palette, _DEFAULT_DISCRETE)
    elif isinstance(palette, Mapping):
        return {cat: palette.get(cat, _DEFAULT_DISCRETE[i % len(_DEFAULT_DISCRETE)]) for i, cat in enumerate(categories)}
    elif isinstance(palette, Sequence):
        colors = list(palette)
    else:
        raise TypeError("Unsupported palette specification.")

    if len(colors) < len(categories):
        warnings.warn(
            f"Palette has {len(colors)} color(s) but {len(categories)} categories were found. "
            "Colors will be cycled, causing some categories to share the same color. "
            "Pass a larger palette to avoid ambiguity.",
            UserWarning,
            stacklevel=3,
        )
        repeats = (len(categories) // len(colors)) + 1
        colors = (colors * repeats)[: len(categories)]

    return {cat: colors[i] for i, cat in enumerate(categories)}


def normalize_cmap_spec(cmap: Optional[Union[str, Sequence[str]]]) -> list[str]:
    """Return a list of colors for a named or explicit continuous palette."""
    if cmap is None:
        return _CONTINUOUS_PALETTES["viridis"]
    if isinstance(cmap, str):
        return _CONTINUOUS_PALETTES.get(cmap, _CONTINUOUS_PALETTES["viridis"])
    if isinstance(cmap, Sequence):
        return list(cmap)
    raise TypeError("Unsupported cmap specification.")


def ensure_rgb_hex(values: np.ndarray) -> list[str]:
    """Normalize RGB array to hex color strings."""
    return _rgb_to_hex(values)


def sort_categories(categories: Sequence) -> list:
    """Sort a list of category labels, using numeric order when all are numeric.

    When every label in *categories* (after stripping whitespace) can be parsed
    as a finite float, they are sorted by their numeric value.  Otherwise the
    original lexicographic order from ``pd.Categorical`` is preserved.  NA/NaN
    labels are always placed last.
    """
    na_labels = [c for c in categories if pd.isna(c) or str(c).strip().upper() == "NA"]
    non_na = [c for c in categories if c not in na_labels]

    def _try_float(v) -> Optional[float]:
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    floats = [_try_float(c) for c in non_na]
    if non_na and all(f is not None for f in floats):
        paired = sorted(zip(floats, non_na), key=lambda x: x[0])
        sorted_non_na = [c for _, c in paired]
    else:
        sorted_non_na = sorted(non_na, key=str)

    return sorted_non_na + na_labels


def apply_point_order(
    df: pd.DataFrame,
    order: Optional[Union[str, Sequence[int]]],
    values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Order rows for overplotting control (random, sort, or explicit indices)."""
    if order is None:
        return df
    if isinstance(order, str):
        if order == "random":
            return df.sample(frac=1.0, random_state=None)
        if order == "sort":
            if values is None:
                return df
            idx = np.argsort(values)
            return df.iloc[idx]
        raise ValueError("order must be 'random', 'sort', or an index array.")
    order_idx = np.asarray(order)
    return df.iloc[order_idx]


def resolve_numeric_vector(
    adata: AnnData,
    values: Union[str, Sequence, np.ndarray],
    name: str,
) -> np.ndarray:
    """Resolve a numeric vector from adata.obs or a supplied array-like."""
    if isinstance(values, str):
        if values not in adata.obs:
            raise ValueError(f"{name} '{values}' not found in adata.obs.")
        return pd.to_numeric(adata.obs[values], errors="coerce").to_numpy()
    if _is_array_like(values):
        arr = _to_numpy_1d(values)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(f"{name} length does not match number of observations.")
        return arr.astype(float)
    raise TypeError(f"Unsupported {name} specification.")


def compute_transparency(
    values: np.ndarray,
    trans_fac: float = 1.5,
    trans_th: float = -0.5,
    scale: bool = True,
) -> np.ndarray:
    """Compute per-point alpha values using a logistic transform."""
    vec = np.asarray(values, dtype=float)
    if scale:
        mean = np.nanmean(vec)
        std = np.nanstd(vec)
        if std == 0 or np.isnan(std):
            z = np.zeros_like(vec)
        else:
            z = (vec - mean) / std
    else:
        z = vec
    alpha_val = 1 / (1 + np.exp(-trans_fac * (z - trans_th)))
    alpha_val[z > trans_th] = 1
    alpha_val = alpha_val ** trans_fac
    return alpha_val


def darken_hex(color: str, factor: float = 0.1) -> str:
    """Darken a hex color by a given factor in [0, 1]."""
    if not color.startswith("#") or len(color) != 7:
        return color
    factor = max(0.0, min(1.0, float(factor)))
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    r = max(0, int(r * (1 - factor)))
    g = max(0, int(g * (1 - factor)))
    b = max(0, int(b * (1 - factor)))
    return f"#{r:02x}{g:02x}{b:02x}"
