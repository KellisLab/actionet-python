#!/usr/bin/env python3
"""Compare backed HDF5 write-heavy operations at 4K and 32K chunks.

The work directory must be on the same filesystem as the input so elapsed and
I/O measurements are not confounded by different storage devices. Timing
gates are opt-in because shared/HPC filesystem performance is inherently
variable and should not be enforced in the unit-test suite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

import actionet as an
from actionet.io.checkpoint import _repack_h5ad

from benchmark_support import StageProfiler, _io_counters_mb

DEFAULT_OPERATIONS = (
    "repack",
    "decompress",
    "subset",
    "normalize",
    "svd_decompress",
)


def _has_compressed_dataset(path: Path) -> bool:
    compressed = False
    with h5py.File(path, "r") as handle:

        def visitor(_name, node):
            nonlocal compressed
            if isinstance(node, h5py.Dataset) and node.compression is not None:
                compressed = True

        handle.visititems(visitor)
    return compressed


def _run_operation(
    operation: str,
    path: Path,
    *,
    chunk_size: int,
    compute_chunk_size: int,
    n_components: int,
) -> None:
    if operation == "repack":
        adata = ad.read_h5ad(path, backed="r+")
        try:
            _repack_h5ad(adata, chunk_size=chunk_size, verbose=False)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return

    if operation == "decompress":
        adata = ad.read_h5ad(path, backed="r+")
        try:
            an.decompress_backed_storage(
                adata,
                scope="file",
                chunk_size=chunk_size,
                verbose=False,
            )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return

    if operation == "subset":
        adata = ad.read_h5ad(path, backed="r+")
        try:
            obs_idx = np.arange(0, adata.n_obs, 2, dtype=np.int64)
            an.subset_anndata(
                adata,
                obs_idx=obs_idx,
                inplace=True,
                backed_chunk_size=chunk_size,
            )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return

    if operation == "normalize":
        adata = ad.read_h5ad(path, backed="r+")
        try:
            an.normalize_anndata(
                adata,
                target_sum=1e4,
                log_transform=False,
                backed_chunk_size=compute_chunk_size,
                backed_write_chunk_size=chunk_size,
                inplace=True,
            )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return

    if operation == "svd_decompress":
        adata = ad.read_h5ad(path, backed="r")
        try:
            an.run_svd(
                adata,
                n_components=min(n_components, adata.n_obs - 1, adata.n_vars - 1),
                backed_chunk_size=compute_chunk_size,
                backed_write_chunk_size=chunk_size,
                allow_compressed=False,
                verbose=False,
            )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return

    raise ValueError(f"Unknown operation: {operation}")


def _measure(operation: str, source: Path, work_path: Path, **kwargs) -> dict:
    shutil.copy2(source, work_path)
    try:
        io_read_before, io_write_before = _io_counters_mb()
        with StageProfiler(operation) as profiler:
            _run_operation(operation, work_path, **kwargs)
        io_read_after, io_write_after = _io_counters_mb()
    finally:
        work_path.unlink(missing_ok=True)
    return {
        "operation": operation,
        "chunk_size": kwargs["chunk_size"],
        "compute_chunk_size": kwargs["compute_chunk_size"],
        "wall_s": profiler.elapsed,
        "peak_rss_mb": profiler.peak_rss_mb,
        "io_read_mb": max(0.0, io_read_after - io_read_before),
        "io_write_mb": max(0.0, io_write_after - io_write_before),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source H5AD file")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--compute-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--operations", nargs="+", choices=DEFAULT_OPERATIONS, default=list(DEFAULT_OPERATIONS)
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--min-repack-speedup",
        type=float,
        default=0.0,
        help="Optional 4096/32768 repack speedup gate; 0 disables it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.input.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if any(size <= 0 for size in args.chunk_sizes):
        raise ValueError("All chunk sizes must be > 0")
    if args.compute_chunk_size <= 0:
        raise ValueError("--compute-chunk-size must be > 0")

    owns_work_dir = args.work_dir is None
    if owns_work_dir:
        work_dir = Path(tempfile.mkdtemp(prefix="actionet_write_chunks_", dir=source.parent))
    else:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    if os.stat(source).st_dev != os.stat(work_dir).st_dev:
        raise ValueError("The input and work directory must be on the same filesystem")

    compressed = _has_compressed_dataset(source)
    results = []
    skipped = []
    try:
        for operation in args.operations:
            if operation in {"decompress", "svd_decompress"} and not compressed:
                skipped.append({"operation": operation, "reason": "input is uncompressed"})
                continue
            for chunk_size in args.chunk_sizes:
                work_path = work_dir / f"{operation}_{chunk_size}.h5ad"
                row = _measure(
                    operation,
                    source,
                    work_path,
                    chunk_size=chunk_size,
                    compute_chunk_size=args.compute_chunk_size,
                    n_components=args.n_components,
                )
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        if owns_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    payload = {
        "input": str(source),
        "input_bytes": source.stat().st_size,
        "same_filesystem": True,
        "results": results,
        "skipped": skipped,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.min_repack_speedup > 0:
        times = {
            row["chunk_size"]: row["wall_s"] for row in results if row["operation"] == "repack"
        }
        if 4096 not in times or 32768 not in times:
            raise ValueError("The repack speedup gate requires chunk sizes 4096 and 32768")
        speedup = times[4096] / times[32768]
        print(f"repack_speedup_4096_over_32768={speedup:.3f}", flush=True)
        if speedup < args.min_repack_speedup:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
