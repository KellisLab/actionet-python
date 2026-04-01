"""Reusable lazy-transform object and helpers for backed operators."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._matrix_source import MatrixSource
from .backed_io import _backed_group_path


class LazyTransform:
    """Reusable lazy logcount transform state for backed matrix operators."""

    def __init__(
        self,
        adata: AnnData,
        *,
        layer: Optional[str] = None,
        target_sum: float = 1e4,
        log_base: Optional[float] = None,
        pseudocount: float = 1.0,
        key: Optional[str] = None,
        backed_chunk_size: int = 4096,
        validation_samples: int = 16,
    ) -> None:
        _validate_lazy_logcounts_params(
            lazy_logcounts=True,
            lazy_target_sum=target_sum,
            lazy_log_base=log_base,
            lazy_pseudocount=pseudocount,
        )

        source = MatrixSource(adata, layer=layer)
        if not source.is_backed:
            raise ValueError(
                "Lazy logcount transform is supported only for backed AnnData inputs."
            )

        chunk_size = int(max(1, backed_chunk_size))
        row_sums = source.row_sums(chunk_size=chunk_size)
        row_scale_factors = np.divide(
            float(target_sum),
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 0,
        )

        sample_indices = _sample_row_indices(source.n_obs, max_samples=validation_samples)
        sample_row_sums = np.asarray(row_sums[sample_indices], dtype=np.float64)

        self.target_sum = float(target_sum)
        self.log_base = None if log_base is None else float(log_base)
        self.pseudocount = float(pseudocount)
        self.key = None if key is None else str(key)

        # Operator parameters are materialized once during initialization.
        self.row_scale_factors = np.ascontiguousarray(row_scale_factors, dtype=np.float64)
        self.apply_log1p = True
        self.log_scale = 1.0 if self.log_base is None else float(1.0 / np.log(self.log_base))

        # Validation state used by downstream operators.
        self.source_group_path = _backed_group_path(layer)
        self.source_shape = (int(source.n_obs), int(source.n_vars))
        self.matrix_fingerprint = _matrix_fingerprint(source)
        self.validation_row_indices = np.ascontiguousarray(sample_indices, dtype=np.int64)
        self.validation_row_sums = np.ascontiguousarray(sample_row_sums, dtype=np.float64)

        self._validated = False

    @property
    def is_initialized(self) -> bool:
        return (
            self.row_scale_factors is not None
            and self.matrix_fingerprint is not None
            and self.source_shape is not None
            and self.source_group_path is not None
            and self.validation_row_indices is not None
            and self.validation_row_sums is not None
        )

    def cache_size(self) -> int:
        # Backward-compatible shim from stage-2A cache API.
        return 1 if self.row_scale_factors is not None else 0


def create_lazy_transform(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    target_sum: float = 1e4,
    log_base: Optional[float] = None,
    pseudocount: float = 1.0,
    key: Optional[str] = None,
    backed_chunk_size: int = 4096,
    validation_samples: int = 16,
) -> LazyTransform:
    """Create and initialize a reusable lazy transform for a backed matrix source."""
    return LazyTransform(
        adata,
        layer=layer,
        target_sum=target_sum,
        log_base=log_base,
        pseudocount=pseudocount,
        key=key,
        backed_chunk_size=backed_chunk_size,
        validation_samples=validation_samples,
    )


def _validate_lazy_logcounts_params(
    *,
    lazy_logcounts: bool,
    lazy_target_sum: float,
    lazy_log_base: Optional[float],
    lazy_pseudocount: float,
) -> None:
    if not lazy_logcounts:
        return

    if lazy_target_sum <= 0:
        raise ValueError("`lazy_target_sum` must be > 0 when `lazy_logcounts=True`.")
    if lazy_log_base is not None:
        if lazy_log_base <= 0:
            raise ValueError("`lazy_log_base` must be > 0 when provided.")
        if np.isclose(lazy_log_base, 1.0):
            raise ValueError("`lazy_log_base` cannot be 1.0.")
    if lazy_pseudocount <= 0:
        raise ValueError("`lazy_pseudocount` must be > 0 when `lazy_logcounts=True`.")
    if not np.isclose(lazy_pseudocount, 1.0):
        raise ValueError(
            "Stage-1 lazy logcounts currently supports only `lazy_pseudocount=1.0`."
        )


def _lazy_log_base_mode(lazy_log_base: Optional[float]) -> str:
    return "natural" if lazy_log_base is None else "numeric"


def _matrix_nnz(source: MatrixSource) -> Optional[int]:
    matrix = source.matrix

    if sp.issparse(matrix):
        return int(matrix.nnz)

    nnz = getattr(matrix, "nnz", None)
    if nnz is not None:
        try:
            return int(nnz)
        except Exception:
            pass

    group = getattr(matrix, "group", None)
    if group is not None:
        try:
            if "indptr" in group:
                indptr = group["indptr"]
                return int(indptr[-1])
        except Exception:
            pass

    return None


def _matrix_dtype(source: MatrixSource) -> str:
    matrix = source.matrix
    dtype = getattr(matrix, "dtype", None)
    if dtype is None:
        return "unknown"
    try:
        return str(np.dtype(dtype))
    except Exception:
        return str(dtype)


def _matrix_fingerprint(source: MatrixSource) -> dict[str, Any]:
    file_path = os.path.realpath(str(source.adata.filename))
    payload: dict[str, Any] = {
        "file_path": file_path,
        "group_path": _backed_group_path(source.layer),
        "shape": [int(source.n_obs), int(source.n_vars)],
        "is_sparse": bool(source.is_sparse),
        "dtype": _matrix_dtype(source),
    }
    nnz = _matrix_nnz(source)
    if nnz is not None:
        payload["nnz"] = int(nnz)
    return payload


def _stable_hash(payload: Mapping[str, Any]) -> str:
    serial = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def _sample_row_indices(n_obs: int, max_samples: int = 16) -> np.ndarray:
    sample_n = int(min(max_samples, max(n_obs, 0)))
    if sample_n <= 0:
        return np.zeros(0, dtype=np.int64)
    if sample_n == n_obs:
        return np.arange(n_obs, dtype=np.int64)
    return np.unique(np.linspace(0, n_obs - 1, num=sample_n, dtype=np.int64))


def _row_sums_for_rows(source: MatrixSource, row_indices: np.ndarray) -> np.ndarray:
    out = np.zeros(row_indices.size, dtype=np.float64)
    for i, row_idx in enumerate(row_indices):
        block = source.get_rows(int(row_idx), int(row_idx) + 1)
        if sp.issparse(block):
            out[i] = float(block.sum())
        else:
            out[i] = float(np.asarray(block, dtype=np.float64).sum())
    return out


def _lazy_params_for_metadata(
    lazy_transform: Optional[LazyTransform],
) -> dict[str, Any]:
    if lazy_transform is None:
        return {"lazy_logcounts": False}

    payload: dict[str, Any] = {
        "lazy_logcounts": True,
        "lazy_target_sum": float(lazy_transform.target_sum),
        "lazy_pseudocount": float(lazy_transform.pseudocount),
        "lazy_log_base_mode": _lazy_log_base_mode(lazy_transform.log_base),
        "lazy_apply_log1p": bool(lazy_transform.apply_log1p),
        "lazy_log_scale": float(lazy_transform.log_scale),
    }
    if lazy_transform.log_base is not None:
        payload["lazy_log_base"] = float(lazy_transform.log_base)
    if lazy_transform.key is not None:
        payload["lazy_transform_key"] = str(lazy_transform.key)
    if lazy_transform.matrix_fingerprint is not None:
        payload["lazy_matrix_fingerprint"] = dict(lazy_transform.matrix_fingerprint)
        payload["lazy_scale_cache_key"] = _stable_hash(lazy_transform.matrix_fingerprint)[:24]
    return payload


def _resolve_lazy_backed_transform(
    source: MatrixSource,
    *,
    lazy_transform: Optional[LazyTransform],
    backed_chunk_size: int,
) -> tuple[Optional[np.ndarray], bool, float]:
    del backed_chunk_size  # The transform must already be fully initialized.

    if lazy_transform is None:
        return None, False, 1.0

    if not source.is_backed:
        raise ValueError("Lazy logcount transform is supported only for backed AnnData inputs.")

    _validate_lazy_logcounts_params(
        lazy_logcounts=True,
        lazy_target_sum=lazy_transform.target_sum,
        lazy_log_base=lazy_transform.log_base,
        lazy_pseudocount=lazy_transform.pseudocount,
    )

    if not lazy_transform.is_initialized:
        raise ValueError(
            "Lazy transform is not initialized. Recreate it with "
            "`create_lazy_transform(backed_adata, ...)`."
        )

    current_group_path = _backed_group_path(source.layer)
    if lazy_transform.source_group_path != current_group_path:
        raise ValueError(
            "Lazy transform source mismatch: requested matrix path "
            f"'{current_group_path}' but transform targets "
            f"'{lazy_transform.source_group_path}'. Recreate the lazy transform "
            "for this source layer."
        )

    shape = (int(source.n_obs), int(source.n_vars))
    if lazy_transform.source_shape != shape:
        raise ValueError(
            "Lazy transform shape mismatch: current source has shape "
            f"{shape}, but transform was initialized with "
            f"{lazy_transform.source_shape}. Recreate the lazy transform."
        )

    current_fingerprint = _matrix_fingerprint(source)
    if current_fingerprint != lazy_transform.matrix_fingerprint:
        raise ValueError(
            "Lazy transform fingerprint mismatch: source matrix metadata changed "
            "since transform initialization. Recreate the lazy transform."
        )

    row_scale_factors = np.asarray(lazy_transform.row_scale_factors, dtype=np.float64)
    if row_scale_factors.ndim != 1 or row_scale_factors.shape[0] != source.n_obs:
        raise ValueError(
            "Lazy transform row scales are invalid for the current source shape. "
            "Recreate the lazy transform."
        )

    if not lazy_transform._validated:
        indices = np.asarray(lazy_transform.validation_row_indices, dtype=np.int64)
        expected = np.asarray(lazy_transform.validation_row_sums, dtype=np.float64)
        if indices.size != expected.size:
            raise ValueError(
                "Lazy transform validation state is corrupt (sample index/row-sum size mismatch). "
                "Recreate the lazy transform."
            )
        if indices.size > 0:
            observed = _row_sums_for_rows(source, indices)
            if not np.allclose(observed, expected, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    "Lazy transform validation failed: sampled source rows changed since "
                    "transform initialization. Recreate the lazy transform."
                )
        lazy_transform._validated = True

    return (
        np.ascontiguousarray(row_scale_factors, dtype=np.float64),
        bool(lazy_transform.apply_log1p),
        float(lazy_transform.log_scale),
    )
