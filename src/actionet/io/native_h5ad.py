"""Private Python control plane for libactionet's H5AD matrix data plane."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .. import _core

from .backed_adapter import BackedMatrixLocation


NativeEngine = Literal["auto", "native", "python"]
_ENGINE_ENV = "ACTIONET_BACKED_IO_ENGINE"
_MAX_BUFFER_BYTES = 128 * 1024 * 1024
_GAP_MERGE_BYTES = 64 * 1024


class NativeCapabilityError(RuntimeError):
    """A selector or encoding is unsupported before transfer starts."""


def native_layout_capability(
    info: dict[str, object],
    *,
    preserve_layout: bool,
) -> tuple[bool, str]:
    """Check filter decoder/encoder availability from native inspection."""
    for dataset in info.get("datasets", []):
        for filter_info in dataset.get("filters", []):
            name = filter_info.get("name") or filter_info.get("id")
            if not filter_info.get("decode_available", False):
                return False, f"HDF5 filter {name!r} has no decoder"
            if preserve_layout and not filter_info.get(
                "encode_available", False
            ):
                return False, f"HDF5 filter {name!r} has no encoder"
    return True, ""


def backed_io_engine() -> NativeEngine:
    value = os.environ.get(_ENGINE_ENV, "auto").strip().lower()
    if value not in {"auto", "native", "python"}:
        raise ValueError(
            f"{_ENGINE_ENV} must be one of auto, native, or python; got {value!r}"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class NativeSubsetPlan:
    source: BackedMatrixLocation
    source_info: dict[str, object]
    rows: np.ndarray
    columns: np.ndarray


def plan_native_subset(
    source: BackedMatrixLocation | None,
    rows: np.ndarray,
    columns: np.ndarray,
) -> NativeSubsetPlan | None:
    """Preflight native support without touching a destination file."""
    engine = backed_io_engine()
    if engine == "python" or source is None:
        return None

    try:
        info = _core.h5ad_inspect_matrix(source.filename, source.h5_path)
    except Exception as exc:
        if engine == "native":
            raise NativeCapabilityError(
                f"native H5AD inspection rejected {source.h5_path}: {exc}"
            ) from exc
        return None

    supported, reason = native_layout_capability(
        dict(info),
        preserve_layout=True,
    )
    if not supported:
        if engine == "native":
            raise NativeCapabilityError(
                f"native H5AD transfer rejected {source.h5_path}: {reason}"
            )
        return None

    return NativeSubsetPlan(
        source=source,
        source_info=dict(info),
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
    )


def execute_native_subset(
    plan: NativeSubsetPlan,
    destination_file: str,
    destination_h5_path: str,
    *,
    max_rows_per_batch: int,
    collect_span_stats: bool,
) -> dict[str, object]:
    """Execute a preflighted transfer.

    Any exception here is a transaction failure. It must never trigger a
    silent Python retry because native writing may already have started.
    """
    return dict(
        _core.h5ad_subset_matrix(
            plan.source.filename,
            plan.source.h5_path,
            os.fspath(destination_file),
            destination_h5_path,
            plan.rows,
            plan.columns,
            max_buffer_bytes=_MAX_BUFFER_BYTES,
            gap_merge_bytes=_GAP_MERGE_BYTES,
            max_rows_per_batch=max_rows_per_batch,
            preserve_layout=True,
            collect_span_stats=collect_span_stats,
        )
    )


def native_copy_matrix(
    source_file: str,
    source_h5_path: str,
    destination_file: str,
    destination_h5_path: str,
    *,
    max_rows_per_batch: int,
    preserve_layout: bool,
    collect_span_stats: bool = False,
) -> dict[str, object]:
    return dict(
        _core.h5ad_copy_matrix(
            source_file,
            source_h5_path,
            destination_file,
            destination_h5_path,
            max_buffer_bytes=_MAX_BUFFER_BYTES,
            max_rows_per_batch=max_rows_per_batch,
            preserve_layout=preserve_layout,
            collect_span_stats=collect_span_stats,
        )
    )


def native_transform_matrix(
    source_file: str,
    source_h5_path: str,
    destination_file: str,
    destination_h5_path: str,
    *,
    row_scale: np.ndarray,
    apply_log: bool,
    pseudocount: float,
    log_scale: float,
    output_dtype: np.dtype,
    max_rows_per_batch: int,
    destination_structure_path: str | None = None,
) -> dict[str, object]:
    dtype = np.dtype(output_dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("Native H5AD transforms support float32 or float64 output")
    return dict(
        _core.h5ad_transform_matrix(
            source_file,
            source_h5_path,
            destination_file,
            destination_h5_path,
            np.asarray(row_scale, dtype=np.float64),
            apply_log=apply_log,
            pseudocount=float(pseudocount),
            log_scale=float(log_scale),
            output_dtype=dtype.name,
            max_buffer_bytes=_MAX_BUFFER_BYTES,
            max_rows_per_batch=max_rows_per_batch,
            preserve_layout=True,
            destination_structure_path=destination_structure_path,
            destination_structure_is_exact_copy=(
                destination_structure_path is not None
            ),
        )
    )
