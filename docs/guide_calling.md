# Guide Calling API (Python)

This document describes the fit-first guide-calling API added in `actionet`.

## Overview

The guide-calling stack models each guide independently with a 1D, 2-component Gaussian mixture model (shared variance):

- Input orientation is fixed to **`cells x guides`**.
- The fit stage operates on stored nonzero values passing `min_counts`.
- With `apply_log10p1=True` (default), fitting happens in transformed space (`log10(1 + count)`).
- Thresholds are derived from compact fit parameters without refitting.

Public functions:

- `actionet.fit_guides_gmm`
- `actionet.derive_guide_thresholds`
- `actionet.sweep_guide_thresholds`
- `actionet.guide_call_gmm`

## `fit_guides_gmm`

```python
fit_guides_gmm(
    X,
    *,
    layer=None,
    guide_names=None,
    min_points=5,
    min_counts=10.0,
    n_init=8,
    max_iter=200,
    tol=1e-6,
    variance_floor=1e-3,
    apply_log10p1=True,
    seed=0,
    n_threads=0,
    backed_chunk_size=4096,
    backed_chunk_guides=256,
    return_table=True,
)
```

Fits per-guide shared-variance GMMs and returns compact arrays.

Accepted inputs:

- `anndata.AnnData` (in-memory or backed)
- `scipy.sparse` matrix
- `numpy.ndarray` (internally converted to sparse)
- ACTIONet `MatrixOperator`

Returned dictionary fields:

- `weights`: `float64[n_guides, 2]` (`background`, `foreground`)
- `means`: `float64[n_guides, 2]` (ordered ascending)
- `sigma`: `float64[n_guides]` (shared per guide)
- `log_likelihood`: `float64[n_guides]`
- `n_points`: `int64[n_guides]`
- `status`: `int32[n_guides]`
- `status_codes`: mapping with keys:
  - `"ok"` = 0
  - `"insufficient_points"` = 1
  - `"degenerate"` = 2
  - `"numerical_failure"` = 3
- `fit_params`: fit metadata used for reproducibility
- `guide_names`: optional guide names
- `table`: optional `pandas.DataFrame` summary

## `derive_guide_thresholds`

```python
derive_guide_thresholds(
    fits,
    *,
    method="quantile",        # "quantile" | "equal_density" | "valley"
    bg_quantile=0.99,
    fg_quantile=0.01,
    valley_grid_size=256,
    output_scale="raw",       # "raw" | "transformed"
    return_table=True,
)
```

Derives thresholds from a prior fit payload (no refit).

Methods:

- `quantile`: guide-specific `neg_<q>` and `pos_<q>` columns.
- `equal_density`: intersection of the two fitted components.
- `valley`: numeric minimum of mixture density between component means.

Return fields:

- `background`: `float64[n_guides]`
- `foreground`: `float64[n_guides]`
- `method`, `scale`
- `column_names`: threshold column names
- `table`: optional `DataFrame`

If a guide fit status is not `"ok"`, thresholds are returned as `NaN` for that guide.

## `sweep_guide_thresholds`

```python
sweep_guide_thresholds(
    fits,
    *,
    bg_quantiles,
    fg_quantiles,
    output_scale="raw",
    return_tables=False,
)
```

Sweeps quantile thresholds from existing fits (no refit), useful for threshold tuning and curve analysis.

Return fields:

- `background`: `float64[n_guides, len(bg_quantiles)]`
- `foreground`: `float64[n_guides, len(fg_quantiles)]`
- `bg_quantiles`, `fg_quantiles`
- `background_labels`, `foreground_labels`
- `scale`
- optional `background_table`, `foreground_table`

## `guide_call_gmm`

```python
guide_call_gmm(
    X,
    *,
    layer=None,
    guide_names=None,
    method="quantile",
    bg_quantile=0.99,
    fg_quantile=0.01,
    valley_grid_size=256,
    background_thresholds=None,
    foreground_thresholds=None,
    threshold_scale="raw",
    result_mode="auto",       # "auto" | "full" | "simple"
    return_fits=False,
    min_points=5,
    min_counts=10.0,
    n_init=8,
    max_iter=200,
    tol=1e-6,
    variance_floor=1e-3,
    apply_log10p1=True,
    seed=0,
    n_threads=0,
    backed_chunk_size=4096,
    backed_chunk_guides=256,
)
```

Convenience wrapper that resolves input, optionally fits, derives or accepts thresholds, and applies thresholds to produce sparse assignments.

Return fields:

- `thresholds`: per-guide threshold table
- `threshold_arrays`: raw arrays for `background` and `foreground`
- `assignments`:
  - `background`: sparse indicator matrix (`cells x guides`)
  - `foreground`: sparse indicator matrix (`cells x guides`)
- `metadata`: result mode and execution details
- optional `fits`: included in full mode or when requested

Result mode behavior:

- `result_mode="full"`: include fit payload.
- `result_mode="simple"`: omit fit payload.
- `result_mode="auto"`:
  - explicit thresholds provided: skip fitting when possible, return simple payload.
  - no explicit thresholds: fit once, derive thresholds, return simple unless `return_fits=True`.

## Typical Fit-First Workflow

```python
import numpy as np
import actionet as an

# 1) Fit once (compact per-guide parameters)
fits = an.fit_guides_gmm(
    adata,                  # AnnData, sparse matrix, ndarray, or MatrixOperator
    layer=None,
    min_counts=10,
    n_init=8,
    n_threads=16,
)

# 2) Derive baseline thresholds without refit
thr = an.derive_guide_thresholds(
    fits,
    method="quantile",
    bg_quantile=0.99,
    fg_quantile=0.01,
    output_scale="raw",
)

# 3) Sweep quantiles for tuning/diagnostics (still no refit)
sweep = an.sweep_guide_thresholds(
    fits,
    bg_quantiles=np.array([0.95, 0.99]),
    fg_quantiles=np.array([0.01, 0.05]),
)

# 4) Apply explicit chosen thresholds in simple mode
res = an.guide_call_gmm(
    adata,
    background_thresholds=thr["background"],
    foreground_thresholds=thr["foreground"],
    threshold_scale="raw",
    result_mode="simple",   # no fit payload returned
)

foreground_calls = res["assignments"]["foreground"]  # sparse cells x guides
```

This pattern keeps runtime low for repeated threshold tuning because expensive fitting is done once.
