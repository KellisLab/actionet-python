"""Benchmark native backed column extraction without loading AnnData metadata.

The source H5AD is always opened read-only.  Results are emitted as JSON lines
with source fingerprints and process I/O counters so cached and storage-backed
runs can be distinguished.

Examples
--------
PYTHONPATH=/tmp/actionet-core-preload:src \
  /data/tau_project/.venv_tau/bin/python \
  tests/benchmark_backed_take_columns.py \
  /data/adatas/adata_agg_ALL_pass132_post.h5ad \
  --feature-counts 1 10 100

Pass ``--drop-cache`` only for an intentionally scheduled cold-cache run.  It
uses read-only ``POSIX_FADV_DONTNEED`` and never mutates the source file.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from actionet import _core


def _fingerprint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _process_io() -> dict[str, int]:
    """Return process I/O counters when available.

    ``/proc/self/io`` only exists on Linux. On macOS and any other host that
    lacks it the benchmark still runs; we simply omit the corresponding
    deltas from the reported timings.
    """
    values: dict[str, int] = {}
    try:
        handle = open("/proc/self/io", encoding="utf-8")
    except OSError:
        return values
    with handle:
        for line in handle:
            key, value = line.rstrip().split(": ", maxsplit=1)
            values[key] = int(value)
    return values


def _drop_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("--drop-cache requires os.posix_fadvise")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--group-path", default="/X")
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-sparse", action="store_true")
    parser.add_argument("--drop-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.h5ad.resolve()
    initial_fingerprint = _fingerprint(source)
    info = _core.h5ad_inspect_matrix(str(source), args.group_path)
    n_rows, n_cols = (int(value) for value in info["shape"])
    rng = np.random.default_rng(args.seed)

    print(
        json.dumps(
            {
                "event": "configuration",
                "source": str(source),
                "group_path": args.group_path,
                "encoding": info["encoding"],
                "shape": [n_rows, n_cols],
                "nnz": int(info["nnz"]),
                "chunk_size": args.chunk_size,
                "drop_cache": args.drop_cache,
                "extension": _core.__file__,
                "source_fingerprint": initial_fingerprint,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for feature_count in args.feature_counts:
        if not 0 <= feature_count <= n_cols:
            raise ValueError(
                f"feature count {feature_count} is outside [0, {n_cols}]"
            )
        columns = np.sort(
            rng.choice(n_cols, size=feature_count, replace=False).astype(np.int64)
        )
        if args.drop_cache:
            _drop_cache(source)

        before_io = _process_io()
        before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.perf_counter()
        operator = _core.create_backed_operator(
            file_path=str(source),
            group_path=args.group_path,
            chunk_size=args.chunk_size,
        )
        opened = time.perf_counter()
        result = _core.backed_take_columns(
            operator,
            columns,
            prefer_sparse=args.prefer_sparse,
        )
        finished = time.perf_counter()
        after_io = _process_io()
        after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        checksum = float(result.sum())
        print(
            json.dumps(
                {
                    "event": "result",
                    "feature_count": feature_count,
                    "columns": columns.tolist(),
                    "shape": list(result.shape),
                    "checksum": checksum,
                    "open_s": opened - started,
                    "take_s": finished - opened,
                    "io": {
                        key: after_io[key] - before_io.get(key, 0)
                        for key in after_io
                    },
                    "peak_rss_delta_kib": max(0, after_rss - before_rss),
                    "source_fingerprint_unchanged": (
                        _fingerprint(source) == initial_fingerprint
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if _fingerprint(source) != initial_fingerprint:
        raise RuntimeError("source H5AD fingerprint changed during benchmark")


if __name__ == "__main__":
    main()
