"""High-performance per-guide GMM calling (fit-first workflow).

This module provides a fit-first guide-calling stack backed by ACTIONet C++ APIs.
The default flow fits compact per-guide mixture parameters once, then derives
thresholds post-hoc without refitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from ._matrix_source import MatrixSource
from .anndata_utils import anndata_to_matrix
from .backed_io import _backed_group_path, _open_backed_operator


ArrayLike1D = Union[np.ndarray, Sequence[float]]
ThresholdMethod = Literal["quantile", "equal_density", "valley"]
ResultMode = Literal["auto", "full", "simple"]
ThresholdScale = Literal["raw", "transformed"]


@dataclass
class _ResolvedInput:
    kind: Literal["sparse", "operator"]
    data: Any
    n_guides: int
    guide_names: Optional[np.ndarray]
    _exit_stack: list = field(default_factory=list, repr=False)


def _as_float64_1d(x: ArrayLike1D, *, name: str, expected_len: Optional[int] = None) -> np.ndarray:
    """Coerce *x* to a contiguous float64 1-D array, optionally length-checked."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"{name} length ({arr.size}) does not match expected length ({expected_len})")
    return arr


def _fmt_quantile_label(prefix: str, q: float) -> str:
    return f"{prefix}_{q}"


def _transform_to_raw(values: np.ndarray, *, apply_log10p1: bool) -> np.ndarray:
    """Inverse-transform thresholds from log10(1+x) space back to raw counts.

    Non-finite values (NaN from failed fits) are preserved unchanged.
    When *apply_log10p1* is False the array is returned as-is (identity).
    """
    out = np.asarray(values, dtype=np.float64).copy()
    if not apply_log10p1:
        return out

    finite = np.isfinite(out)
    out[finite] = np.power(10.0, out[finite]) - 1.0
    return out


def _fit_uses_log10p1(fits: dict) -> bool:
    """Return whether the fit payload was produced with ``apply_log10p1=True``."""
    meta = fits.get("fit_params")
    if isinstance(meta, dict) and "apply_log10p1" in meta:
        return bool(meta["apply_log10p1"])
    return True


def _resolve_input(
    X: Union[AnnData, sp.spmatrix, np.ndarray, Any],
    *,
    layer: Optional[str],
    backed_chunk_size: int,
    n_threads: int,
    guide_names: Optional[Sequence[str]] = None,
) -> _ResolvedInput:
    """Normalise heterogeneous input into a ``_ResolvedInput``.

    Accepts AnnData (in-memory or backed), scipy sparse, numpy dense, or a
    pre-built ``MatrixOperator``.  Returns the underlying data handle, guide
    count, and (optional) guide names in a uniform container.
    """
    if isinstance(X, AnnData):
        source = MatrixSource(X, layer=layer)
        inferred_names = np.asarray(X.var_names, dtype=object)
        if guide_names is not None:
            inferred_names = np.asarray(guide_names, dtype=object)
            if inferred_names.size != source.n_vars:
                raise ValueError(
                    f"guide_names length ({inferred_names.size}) does not match n_guides ({source.n_vars})"
                )

        if source.is_backed:
            cm = _open_backed_operator(
                adata=X,
                file_path=str(X.filename),
                group_path=_backed_group_path(layer),
                context="guide_call_gmm",
                chunk_size=max(1, int(backed_chunk_size)),
                n_threads=int(n_threads),
            )
            op = cm.__enter__()
            resolved = _ResolvedInput(
                kind="operator",
                data=op,
                n_guides=source.n_vars,
                guide_names=inferred_names,
            )
            resolved._exit_stack.append(cm)
            return resolved

        mat = anndata_to_matrix(X, layer=layer)
        if not sp.issparse(mat):
            mat = sp.csr_matrix(np.asarray(mat, dtype=np.float64))
        if guide_names is not None:
            inferred_names = np.asarray(guide_names, dtype=object)
            if inferred_names.size != mat.shape[1]:
                raise ValueError(
                    f"guide_names length ({inferred_names.size}) does not match n_guides ({mat.shape[1]})"
                )
        return _ResolvedInput(kind="sparse", data=mat, n_guides=mat.shape[1], guide_names=inferred_names)

    matrix_operator_cls = getattr(_core, "MatrixOperator", None)
    if matrix_operator_cls is not None and isinstance(X, matrix_operator_cls):
        shape = getattr(X, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("Operator input must expose a 2D shape")
        n_guides = int(shape[1])
        inferred_names = np.asarray(guide_names, dtype=object) if guide_names is not None else None
        if inferred_names is not None and inferred_names.size != n_guides:
            raise ValueError(
                f"guide_names length ({inferred_names.size}) does not match n_guides ({n_guides})"
            )
        return _ResolvedInput(kind="operator", data=X, n_guides=n_guides, guide_names=inferred_names)

    if sp.issparse(X):
        inferred_names = np.asarray(guide_names, dtype=object) if guide_names is not None else None
        if inferred_names is not None and inferred_names.size != X.shape[1]:
            raise ValueError(
                f"guide_names length ({inferred_names.size}) does not match n_guides ({X.shape[1]})"
            )
        return _ResolvedInput(kind="sparse", data=X, n_guides=X.shape[1], guide_names=inferred_names)

    if isinstance(X, np.ndarray):
        X_sp = sp.csr_matrix(np.asarray(X, dtype=np.float64))
        inferred_names = np.asarray(guide_names, dtype=object) if guide_names is not None else None
        if inferred_names is not None and inferred_names.size != X_sp.shape[1]:
            raise ValueError(
                f"guide_names length ({inferred_names.size}) does not match n_guides ({X_sp.shape[1]})"
            )
        return _ResolvedInput(kind="sparse", data=X_sp, n_guides=X_sp.shape[1], guide_names=inferred_names)

    raise TypeError(
        "Unsupported input type. Expected AnnData, scipy.sparse matrix, numpy.ndarray, "
        "or ACTIONet MatrixOperator."
    )


def _cleanup_resolved(resolved: _ResolvedInput) -> None:
    for cm in reversed(resolved._exit_stack):
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
    resolved._exit_stack.clear()


def _fit_from_resolved(
    resolved: _ResolvedInput,
    *,
    min_points: int,
    min_counts: float,
    n_init: int,
    max_iter: int,
    tol: float,
    variance_floor: float,
    apply_log10p1: bool,
    seed: int,
    n_threads: int,
    backed_chunk_guides: int,
) -> dict:
    """Dispatch the C++ GMM fit to the correct backend (sparse or operator)."""
    if resolved.kind == "operator":
        fits = _core.fit_guides_gmm_backed_operator(
            resolved.data,
            min_points=min_points,
            min_counts=min_counts,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            variance_floor=variance_floor,
            apply_log10p1=apply_log10p1,
            seed=seed,
            n_threads=n_threads,
            backed_chunk_guides=backed_chunk_guides,
        )
    else:
        fits = _core.fit_guides_gmm_sparse(
            resolved.data,
            min_points=min_points,
            min_counts=min_counts,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            variance_floor=variance_floor,
            apply_log10p1=apply_log10p1,
            seed=seed,
            n_threads=n_threads,
        )
    return fits


def _apply_from_resolved(
    resolved: _ResolvedInput,
    *,
    background_thresholds_raw: np.ndarray,
    foreground_thresholds_raw: np.ndarray,
    backed_chunk_guides: int,
) -> dict:
    """Dispatch threshold application to the correct C++ backend.

    Thresholds must be in raw count space.  Returns a dict with sparse
    indicator matrices ``"background"`` and ``"foreground"``.
    """
    if resolved.kind == "operator":
        return _core.apply_guide_thresholds_backed_operator(
            resolved.data,
            background_thresholds_raw,
            foreground_thresholds_raw,
            chunk_guides=backed_chunk_guides,
        )
    return _core.apply_guide_thresholds_sparse(
        resolved.data,
        background_thresholds_raw,
        foreground_thresholds_raw,
    )


def _dress_fit_result(
    fits: dict,
    resolved: _ResolvedInput,
    *,
    apply_log10p1: bool,
    min_points: int,
    min_counts: float,
    n_init: int,
    max_iter: int,
    tol: float,
    variance_floor: float,
    seed: int,
    n_threads: int,
    backed_chunk_guides: int,
) -> dict:
    """Attach metadata, guide names, and a summary table to raw C++ fit output.

    Mutates *fits* in place and returns it.  Used by both
    :func:`fit_guides_gmm` and :func:`guide_call_gmm` to avoid duplicating
    the post-fit decoration logic.
    """
    guide_index = None
    if resolved.guide_names is not None:
        guide_index = pd.Index(resolved.guide_names, name="guide")

    fits["fit_params"] = {
        "min_points": int(min_points),
        "min_counts": float(min_counts),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "variance_floor": float(variance_floor),
        "apply_log10p1": bool(apply_log10p1),
        "seed": int(seed),
        "n_threads": int(n_threads),
        "backed_chunk_guides": int(backed_chunk_guides),
    }
    fits["guide_names"] = resolved.guide_names

    table = pd.DataFrame(
        {
            "weight_background": np.asarray(fits["weights"])[:, 0],
            "weight_foreground": np.asarray(fits["weights"])[:, 1],
            "mean_background": np.asarray(fits["means"])[:, 0],
            "mean_foreground": np.asarray(fits["means"])[:, 1],
            "sigma": np.asarray(fits["sigma"]),
            "log_likelihood": np.asarray(fits["log_likelihood"]),
            "n_points": np.asarray(fits["n_points"]),
            "status": np.asarray(fits["status"]),
        },
        index=guide_index,
    )
    fits["table"] = table
    return fits


def fit_guides_gmm(
    X: Union[AnnData, sp.spmatrix, np.ndarray, Any],
    *,
    layer: Optional[str] = None,
    guide_names: Optional[Sequence[str]] = None,
    min_points: int = 5,
    min_counts: float = 10.0,
    n_init: int = 8,
    max_iter: int = 200,
    tol: float = 1e-6,
    variance_floor: float = 1e-3,
    apply_log10p1: bool = True,
    seed: int = 0,
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
    backed_chunk_guides: int = 256,
    return_table: bool = True,
) -> dict:
    """Fit per-guide 2-component shared-variance GMMs (cells x guides).

    Parameters
    ----------
    X
        AnnData, scipy sparse matrix, numpy array, or MatrixOperator with
        orientation fixed to cells x guides.
    layer
        AnnData layer to use when ``X`` is AnnData.
    guide_names
        Optional guide names. Must match number of guide columns.
    min_points, min_counts, n_init, max_iter, tol, variance_floor
        EM fit controls.
    apply_log10p1
        If True, fits in log10(1 + count) space.
    seed
        Base random seed (deterministic composition per guide/init in C++).
    n_threads
        Number of threads (0 = auto).
    backed_chunk_size, backed_chunk_guides
        Backed I/O and guide-chunk controls for operator paths.
    return_table
        If True, include a pandas summary table in ``result["table"]``.

    Returns
    -------
    dict
        Compact fit payload with keys:
        ``weights``, ``means``, ``sigma``, ``log_likelihood``, ``n_points``,
        ``status``, ``status_codes``, plus fit metadata and optional table.
    """
    resolved = _resolve_input(
        X,
        layer=layer,
        backed_chunk_size=backed_chunk_size,
        n_threads=n_threads,
        guide_names=guide_names,
    )
    try:
        fits = _fit_from_resolved(
            resolved,
            min_points=min_points,
            min_counts=min_counts,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            variance_floor=variance_floor,
            apply_log10p1=apply_log10p1,
            seed=seed,
            n_threads=n_threads,
            backed_chunk_guides=backed_chunk_guides,
        )

        fits = _dress_fit_result(
            fits, resolved,
            apply_log10p1=apply_log10p1,
            min_points=min_points, min_counts=min_counts,
            n_init=n_init, max_iter=max_iter, tol=tol,
            variance_floor=variance_floor, seed=seed,
            n_threads=n_threads, backed_chunk_guides=backed_chunk_guides,
        )

        if not return_table:
            fits.pop("table", None)

        return fits
    finally:
        _cleanup_resolved(resolved)


def derive_guide_thresholds(
    fits: dict,
    *,
    method: ThresholdMethod = "quantile",
    bg_quantile: float = 0.99,
    fg_quantile: float = 0.01,
    valley_grid_size: int = 256,
    output_scale: ThresholdScale = "raw",
    return_table: bool = True,
) -> dict:
    """Derive per-guide thresholds from an existing fit payload (no refit).

    Parameters
    ----------
    fits
        Output from :func:`fit_guides_gmm`.
    method
        ``"quantile"``, ``"equal_density"``, or ``"valley"``.
    bg_quantile, fg_quantile
        Quantiles used when ``method="quantile"``.
    valley_grid_size
        Search grid size used when ``method="valley"``.
    output_scale
        ``"raw"`` returns count-scale thresholds (inverse-transformed when
        fits used ``apply_log10p1=True``); ``"transformed"`` returns thresholds
        in fit space.
    return_table
        If True, include ``result["table"]``.

    Returns
    -------
    dict
        ``background`` and ``foreground`` threshold arrays, method metadata,
        and optional table. Failed-fit guides remain NaN.
    """
    weights = np.asarray(fits["weights"], dtype=np.float64)
    means = np.asarray(fits["means"], dtype=np.float64)
    sigma = np.asarray(fits["sigma"], dtype=np.float64)
    status = np.asarray(fits["status"], dtype=np.int32)

    if method == "quantile":
        out = _core.derive_guide_thresholds_quantile(
            weights, means, sigma, status, bg_quantile=bg_quantile, fg_quantile=fg_quantile
        )
        bg_name = _fmt_quantile_label("neg", float(bg_quantile))
        fg_name = _fmt_quantile_label("pos", float(fg_quantile))
    elif method == "equal_density":
        out = _core.derive_guide_thresholds_equal_density(weights, means, sigma, status)
        bg_name = "background"
        fg_name = "foreground"
    elif method == "valley":
        out = _core.derive_guide_thresholds_valley(
            weights, means, sigma, status, grid_size=int(valley_grid_size)
        )
        bg_name = "background"
        fg_name = "foreground"
    else:
        raise ValueError("method must be one of {'quantile', 'equal_density', 'valley'}")

    background = np.asarray(out["background"], dtype=np.float64)
    foreground = np.asarray(out["foreground"], dtype=np.float64)

    apply_log10p1 = _fit_uses_log10p1(fits)
    if output_scale == "raw":
        background_out = _transform_to_raw(background, apply_log10p1=apply_log10p1)
        foreground_out = _transform_to_raw(foreground, apply_log10p1=apply_log10p1)
    elif output_scale == "transformed":
        background_out = background
        foreground_out = foreground
    else:
        raise ValueError("output_scale must be 'raw' or 'transformed'")

    result = {
        "background": background_out,
        "foreground": foreground_out,
        "method": method,
        "scale": output_scale,
        "column_names": (bg_name, fg_name),
    }

    if return_table:
        guide_names = fits.get("guide_names")
        guide_index = pd.Index(guide_names, name="guide") if guide_names is not None else None
        result["table"] = pd.DataFrame(
            {bg_name: background_out, fg_name: foreground_out},
            index=guide_index,
        )

    return result


def sweep_guide_thresholds(
    fits: dict,
    *,
    bg_quantiles: ArrayLike1D,
    fg_quantiles: ArrayLike1D,
    output_scale: ThresholdScale = "raw",
    return_tables: bool = False,
) -> dict:
    """Sweep quantile threshold curves from existing fits (no refit).

    Parameters
    ----------
    fits
        Output from :func:`fit_guides_gmm`.
    bg_quantiles, fg_quantiles
        1D arrays of quantiles in (0, 1) for background and foreground
        threshold curves.
    output_scale
        ``"raw"`` or ``"transformed"``.
    return_tables
        If True, attach DataFrame views to the result.

    Returns
    -------
    dict
        Threshold sweep arrays with shape ``(n_guides, n_quantiles)`` and
        corresponding quantile labels.
    """
    weights = np.asarray(fits["weights"], dtype=np.float64)
    means = np.asarray(fits["means"], dtype=np.float64)
    sigma = np.asarray(fits["sigma"], dtype=np.float64)
    status = np.asarray(fits["status"], dtype=np.int32)

    bg_q = _as_float64_1d(bg_quantiles, name="bg_quantiles")
    fg_q = _as_float64_1d(fg_quantiles, name="fg_quantiles")

    out = _core.sweep_guide_thresholds_quantile(weights, means, sigma, status, bg_q, fg_q)
    background = np.asarray(out["background"], dtype=np.float64)
    foreground = np.asarray(out["foreground"], dtype=np.float64)

    apply_log10p1 = _fit_uses_log10p1(fits)
    if output_scale == "raw":
        background_out = _transform_to_raw(background, apply_log10p1=apply_log10p1)
        foreground_out = _transform_to_raw(foreground, apply_log10p1=apply_log10p1)
    elif output_scale == "transformed":
        background_out = background
        foreground_out = foreground
    else:
        raise ValueError("output_scale must be 'raw' or 'transformed'")

    bg_labels = [_fmt_quantile_label("neg", float(q)) for q in bg_q]
    fg_labels = [_fmt_quantile_label("pos", float(q)) for q in fg_q]
    result = {
        "background": background_out,
        "foreground": foreground_out,
        "bg_quantiles": bg_q,
        "fg_quantiles": fg_q,
        "background_labels": bg_labels,
        "foreground_labels": fg_labels,
        "scale": output_scale,
    }

    if return_tables:
        guide_names = fits.get("guide_names")
        guide_index = pd.Index(guide_names, name="guide") if guide_names is not None else None
        result["background_table"] = pd.DataFrame(background_out, columns=bg_labels, index=guide_index)
        result["foreground_table"] = pd.DataFrame(foreground_out, columns=fg_labels, index=guide_index)

    return result


def guide_call_gmm(
    X: Union[AnnData, sp.spmatrix, np.ndarray, Any],
    *,
    layer: Optional[str] = None,
    guide_names: Optional[Sequence[str]] = None,
    method: ThresholdMethod = "quantile",
    bg_quantile: float = 0.99,
    fg_quantile: float = 0.01,
    valley_grid_size: int = 256,
    background_thresholds: Optional[ArrayLike1D] = None,
    foreground_thresholds: Optional[ArrayLike1D] = None,
    threshold_scale: ThresholdScale = "raw",
    result_mode: ResultMode = "auto",
    return_fits: bool = False,
    min_points: int = 5,
    min_counts: float = 10.0,
    n_init: int = 8,
    max_iter: int = 200,
    tol: float = 1e-6,
    variance_floor: float = 1e-3,
    apply_log10p1: bool = True,
    seed: int = 0,
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
    backed_chunk_guides: int = 256,
) -> dict:
    """Run fit/derive/apply guide calling with full/simple/auto result modes.

    This is the main entry point for end-to-end guide calling.  It fits
    per-guide GMMs, derives thresholds, and produces sparse indicator
    assignment matrices in a single call.

    Parameters
    ----------
    X
        Cells-by-guides count data.  Accepts AnnData (in-memory or backed),
        scipy sparse, numpy dense, or a pre-built ACTIONet MatrixOperator.
    layer
        AnnData layer to use when ``X`` is AnnData (None = ``.X``).
    guide_names
        Optional guide labels; must match the number of guide columns.
    method
        Threshold derivation method: ``"quantile"`` (default),
        ``"equal_density"``, or ``"valley"``.
    bg_quantile, fg_quantile
        Quantiles for the background and foreground components when
        ``method="quantile"``.
    valley_grid_size
        Grid resolution when ``method="valley"``.
    background_thresholds, foreground_thresholds
        Pre-computed per-guide thresholds.  When supplied the fit step can
        be skipped entirely (in auto/simple mode without ``return_fits``).
    threshold_scale
        Whether explicit thresholds are in ``"raw"`` count space or
        ``"transformed"`` (log10(1+x)) space.
    result_mode
        ``"auto"`` returns a minimal payload by default, ``"full"`` always
        includes fits, ``"simple"`` never includes fits.
    return_fits
        Force inclusion of the fit payload regardless of *result_mode*.
    min_points, min_counts, n_init, max_iter, tol, variance_floor
        EM fit controls forwarded to the C++ backend.
    apply_log10p1
        Fit in log10(1 + count) space (default True).
    seed
        Base random seed for deterministic per-guide/init seed composition.
    n_threads
        Thread count (0 = auto-detect).
    backed_chunk_size, backed_chunk_guides
        I/O chunking controls for backed-mode operators.

    Returns
    -------
    dict
        ``"thresholds"`` : DataFrame of per-guide thresholds.
        ``"threshold_arrays"`` : raw numpy threshold vectors.
        ``"assignments"`` : sparse indicator matrices (background / foreground).
        ``"metadata"`` : run parameters and status flags.
        ``"fits"`` : (optional) full fit payload from :func:`fit_guides_gmm`.
    """
    mode = str(result_mode).lower()
    if mode not in {"auto", "full", "simple"}:
        raise ValueError("result_mode must be one of {'auto', 'full', 'simple'}")

    resolved = _resolve_input(
        X,
        layer=layer,
        backed_chunk_size=backed_chunk_size,
        n_threads=n_threads,
        guide_names=guide_names,
    )
    try:
        bg_in = background_thresholds
        fg_in = foreground_thresholds
        if bg_in is None and fg_in is not None:
            bg_in = fg_in
        if fg_in is None and bg_in is not None:
            fg_in = bg_in
        explicit_thresholds = (bg_in is not None) or (fg_in is not None)

        bg_raw: Optional[np.ndarray] = None
        fg_raw: Optional[np.ndarray] = None
        threshold_table: Optional[pd.DataFrame] = None

        if explicit_thresholds:
            if bg_in is None or fg_in is None:
                raise ValueError("Both background and foreground thresholds must be set together")
            bg = _as_float64_1d(bg_in, name="background_thresholds", expected_len=resolved.n_guides)
            fg = _as_float64_1d(fg_in, name="foreground_thresholds", expected_len=resolved.n_guides)

            if threshold_scale == "raw":
                bg_raw = bg
                fg_raw = fg
            elif threshold_scale == "transformed":
                bg_raw = _transform_to_raw(bg, apply_log10p1=apply_log10p1)
                fg_raw = _transform_to_raw(fg, apply_log10p1=apply_log10p1)
            else:
                raise ValueError("threshold_scale must be 'raw' or 'transformed'")

            guide_index = pd.Index(resolved.guide_names, name="guide") if resolved.guide_names is not None else None
            threshold_table = pd.DataFrame(
                {"background": bg_raw, "foreground": fg_raw},
                index=guide_index,
            )

        needs_fit = True
        if explicit_thresholds and mode in {"auto", "simple"} and not return_fits:
            needs_fit = False

        fits: Optional[dict] = None
        if needs_fit:
            raw_fits = _fit_from_resolved(
                resolved,
                min_points=min_points,
                min_counts=min_counts,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                variance_floor=variance_floor,
                apply_log10p1=apply_log10p1,
                seed=seed,
                n_threads=n_threads,
                backed_chunk_guides=backed_chunk_guides,
            )
            fits = _dress_fit_result(
                raw_fits,
                resolved,
                apply_log10p1=apply_log10p1,
                min_points=min_points,
                min_counts=min_counts,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                variance_floor=variance_floor,
                seed=seed,
                n_threads=n_threads,
                backed_chunk_guides=backed_chunk_guides,
            )

        if bg_raw is None or fg_raw is None:
            if fits is None:
                raise RuntimeError("Internal error: thresholds not provided and fits not available")
            derived = derive_guide_thresholds(
                fits,
                method=method,
                bg_quantile=bg_quantile,
                fg_quantile=fg_quantile,
                valley_grid_size=valley_grid_size,
                output_scale="raw",
                return_table=True,
            )
            bg_raw = np.asarray(derived["background"], dtype=np.float64)
            fg_raw = np.asarray(derived["foreground"], dtype=np.float64)
            threshold_table = derived.get("table")

        assignments = _apply_from_resolved(
            resolved,
            background_thresholds_raw=bg_raw,
            foreground_thresholds_raw=fg_raw,
            backed_chunk_guides=backed_chunk_guides,
        )

        final_mode = mode
        if mode == "auto":
            final_mode = "full" if (fits is not None and (return_fits or not explicit_thresholds)) else "simple"
            if fits is not None and not return_fits:
                final_mode = "simple"

        out = {
            "thresholds": threshold_table,
            "threshold_arrays": {
                "background": bg_raw,
                "foreground": fg_raw,
            },
            "assignments": {
                "background": assignments["background"],
                "foreground": assignments["foreground"],
            },
            "metadata": {
                "result_mode": final_mode,
                "explicit_thresholds": bool(explicit_thresholds),
                "threshold_method": method if not explicit_thresholds else "explicit",
                "threshold_scale": "raw",
                "n_guides": int(resolved.n_guides),
                "n_threads": int(n_threads),
                "apply_log10p1": bool(apply_log10p1),
                "fit_performed": bool(fits is not None),
            },
        }

        if (mode == "full") or return_fits or (mode == "auto" and fits is not None and not explicit_thresholds):
            out["fits"] = fits

        return out
    finally:
        _cleanup_resolved(resolved)
