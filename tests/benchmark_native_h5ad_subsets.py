#!/usr/bin/env python3
"""Benchmark native H5AD selection across the sub-settable workload space.

The source is always opened read-only. Every run creates a unique output in
the source filesystem (or ``--output-dir``), validates it, and deletes it.
JSON records are written to stdout so callers can retain only the results.

Example
-------
python tests/benchmark_native_h5ad_subsets.py atlas.h5ad \
  --fractions 0.05 0.5 0.999 \
  --patterns random alternating position-shifted
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from actionet import _core


MIB = 1024 * 1024
DEFAULT_FRACTIONS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)
DEFAULT_PATTERNS = ("contiguous", "random", "clustered", "alternating", "position-shifted")


def _fingerprint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns)


def _selection(size: int, fraction: float, pattern: str, seed: int) -> np.ndarray:
    count = min(size, max(0, int(round(size * fraction))))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == size:
        return np.arange(size, dtype=np.int64)

    if pattern == "contiguous":
        return np.arange(count, dtype=np.int64)
    if pattern == "position-shifted":
        start = min(size - count, max(0, int(round(size * 0.67 - count / 2))))
        return np.arange(start, start + count, dtype=np.int64)
    if pattern == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(size, size=count, replace=False)).astype(np.int64)
    if pattern == "alternating":
        # linspace is strictly increasing when count <= size.
        return np.linspace(0, size - 1, count, dtype=np.int64)
    if pattern == "clustered":
        # Spread bounded contiguous clusters over the source axis.
        cluster_count = min(64, count)
        base, remainder = divmod(count, cluster_count)
        occupied: list[np.ndarray] = []
        cursor = 0
        for cluster in range(cluster_count):
            width = base + (cluster < remainder)
            remaining = count - cursor
            available_start = size - remaining
            target = int(round(cluster * available_start / max(1, cluster_count - 1)))
            start = max(cursor, target)
            occupied.append(np.arange(start, start + width, dtype=np.int64))
            cursor = start + width
        return np.concatenate(occupied)
    raise ValueError(f"unknown selector pattern: {pattern}")


def _temporary_h5(output_dir: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".actionet-native-benchmark-",
        suffix=".h5ad",
        dir=output_dir,
    )
    os.close(descriptor)
    return Path(raw_path)


def _run_subset(
    source: Path,
    matrix_path: str,
    output_dir: Path,
    rows: np.ndarray,
    columns: np.ndarray | None,
    row_ceiling: int,
    collect_spans: bool,
) -> dict[str, object]:
    destination = _temporary_h5(output_dir)
    try:
        with h5py.File(destination, "w"):
            pass
        started = perf_counter()
        stats = dict(
            _core.h5ad_subset_matrix(
                str(source),
                matrix_path,
                str(destination),
                matrix_path,
                rows,
                columns,
                max_buffer_bytes=128 * MIB,
                gap_merge_bytes=64 * 1024,
                max_rows_per_batch=row_ceiling,
                preserve_layout=True,
                collect_span_stats=collect_spans,
            )
        )
        wall_seconds = perf_counter() - started
        validation = dict(
            _core.h5ad_validate_matrix(
                str(destination),
                matrix_path,
                full=False,
            )
        )
        if not validation["valid"]:
            raise RuntimeError(f"native output validation failed: {validation['error']}")
        stats.pop("spans", None)
        return {
            "wall_seconds": wall_seconds,
            "output_bytes": destination.stat().st_size,
            **stats,
        }
    finally:
        destination.unlink(missing_ok=True)


def _dataset_creation_options(source: h5py.Dataset) -> dict[str, object]:
    options: dict[str, object] = {}
    if source.chunks is not None:
        options["chunks"] = source.chunks
    if source.compression is not None:
        options["compression"] = source.compression
        options["compression_opts"] = source.compression_opts
    if source.shuffle:
        options["shuffle"] = True
    if source.fletcher32:
        options["fletcher32"] = True
    if source.scaleoffset is not None:
        options["scaleoffset"] = source.scaleoffset
    return options


def _copy_dataset_sequential(
    source: h5py.Dataset,
    destination: h5py.Group,
    name: str,
    max_buffer_bytes: int,
) -> tuple[float, float, int]:
    target = destination.create_dataset(
        name,
        shape=source.shape,
        dtype=source.dtype,
        **_dataset_creation_options(source),
    )
    for key, value in source.attrs.items():
        target.attrs[key] = value

    if source.ndim == 1:
        rows_per_batch = max(1, max_buffer_bytes // max(1, source.dtype.itemsize))
    else:
        row_bytes = max(1, int(np.prod(source.shape[1:])) * source.dtype.itemsize)
        rows_per_batch = max(1, max_buffer_bytes // row_bytes)

    read_seconds = 0.0
    write_seconds = 0.0
    calls = 0
    for start in range(0, source.shape[0], rows_per_batch):
        stop = min(source.shape[0], start + rows_per_batch)
        started = perf_counter()
        block = source[start:stop]
        read_seconds += perf_counter() - started
        started = perf_counter()
        target[start:stop] = block
        write_seconds += perf_counter() - started
        calls += 1
    return read_seconds, write_seconds, calls


def _run_raw_sequential_reference(
    source_path: Path,
    matrix_path: str,
    output_dir: Path,
) -> dict[str, object]:
    destination_path = _temporary_h5(output_dir)
    try:
        started = perf_counter()
        read_seconds = 0.0
        write_seconds = 0.0
        calls = 0
        with (
            h5py.File(source_path, "r") as source_file,
            h5py.File(destination_path, "w") as destination_file,
        ):
            source = source_file[matrix_path]
            normalized_path = matrix_path.strip("/")
            if "/" in normalized_path:
                parent_path, name = normalized_path.rsplit("/", 1)
                parent = destination_file.require_group(parent_path)
            else:
                name = normalized_path
                parent = destination_file
            if isinstance(source, h5py.Dataset):
                read_seconds, write_seconds, calls = _copy_dataset_sequential(
                    source, parent, name, 128 * MIB
                )
            else:
                target = parent.create_group(name)
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                for dataset_name in ("data", "indices", "indptr"):
                    read_s, write_s, dataset_calls = _copy_dataset_sequential(
                        source[dataset_name], target, dataset_name, 128 * MIB
                    )
                    read_seconds += read_s
                    write_seconds += write_s
                    calls += dataset_calls
            destination_file.flush()

        validation = dict(
            _core.h5ad_validate_matrix(
                str(destination_path),
                matrix_path,
                full=False,
            )
        )
        if not validation["valid"]:
            raise RuntimeError(f"sequential reference validation failed: {validation['error']}")
        return {
            "kind": "raw-sequential-reference",
            "wall_seconds": perf_counter() - started,
            "source_read_seconds": read_seconds,
            "destination_write_seconds": write_seconds,
            "hdf5_batches": calls,
            "output_bytes": destination_path.stat().st_size,
        }
    finally:
        destination_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--matrix-path", default="/X")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--patterns", nargs="+", choices=DEFAULT_PATTERNS, default=DEFAULT_PATTERNS)
    parser.add_argument("--joint-columns", action="store_true")
    parser.add_argument("--row-ceiling", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--collect-spans", action="store_true")
    parser.add_argument("--skip-raw-reference", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output_dir = (args.output_dir or source.parent).resolve(strict=True)
    before = _fingerprint(source)
    info = dict(_core.h5ad_inspect_matrix(str(source), args.matrix_path))
    rows, columns = map(int, info["shape"])

    if not args.skip_raw_reference:
        print(json.dumps(_run_raw_sequential_reference(source, args.matrix_path, output_dir), default=str), flush=True)

    try:
        for fraction in args.fractions:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"retained fractions must be in [0, 1], got {fraction}")
            for pattern in args.patterns:
                row_selection = _selection(rows, fraction, pattern, args.seed)
                column_selection = (
                    _selection(columns, fraction, pattern, args.seed + 1)
                    if args.joint_columns
                    else None
                )
                result = _run_subset(
                    source,
                    args.matrix_path,
                    output_dir,
                    row_selection,
                    column_selection,
                    args.row_ceiling,
                    args.collect_spans,
                )
                result.update(
                    {
                        "kind": "native-subset",
                        "retained_fraction": fraction,
                        "pattern": pattern,
                        "selected_rows": row_selection.size,
                        "selected_columns": (
                            columns if column_selection is None else column_selection.size
                        ),
                        "joint_columns": args.joint_columns,
                    }
                )
                print(json.dumps(result, default=str), flush=True)
    finally:
        after = _fingerprint(source)
        if after != before:
            raise RuntimeError(
                f"source fingerprint changed during benchmark: {before!r} -> {after!r}"
            )


if __name__ == "__main__":
    main()
