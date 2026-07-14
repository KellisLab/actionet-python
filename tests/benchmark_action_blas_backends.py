#!/usr/bin/env python3
"""Reproducible ACTION benchmark for MKL and OpenBLAS builds.

Run this script from an environment containing exactly the actionet wheel to
measure.  The script verifies the extension's ELF dependencies, creates one
reusable reduction from ``data/test_adata.h5ad``, reports medians of three
trials, and checks thread/output determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

# The experiment assigns parallelism to ACTION's explicit OpenMP loop.  These
# variables are set before NumPy/actionet load a BLAS runtime so the "one
# ACTION thread" sample does not silently include a vendor-owned inner pool.
for _blas_thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_blas_thread_variable] = "1"

import anndata as ad  # noqa: E402
from anndata import AnnData  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import actionet as an  # noqa: E402
from actionet import _core  # noqa: E402


THREAD_LABELS = ("1", "2", "4", "8", "16", "auto")
RTOL = 1e-8
ATOL = 1e-10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/test_adata.h5ad"))
    parser.add_argument(
        "--reduction-cache",
        type=Path,
        default=Path("data/test_adata.action_reduction.npy"),
    )
    parser.add_argument("--output", type=Path, default=Path("action_blas_benchmark.json"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", nargs="+", choices=THREAD_LABELS, default=THREAD_LABELS)
    parser.add_argument("--mode", choices=("run-action", "pipeline", "both"), default="run-action")
    parser.add_argument("--expect-backend", choices=("openblas", "mkl"))
    parser.add_argument("--expect-openblas-threading", choices=("pthread", "openmp"))
    parser.add_argument("--parity-output", type=Path)
    parser.add_argument("--parity-reference", type=Path)
    parser.add_argument("--timing-reference", type=Path)
    parser.add_argument("--enforce-acceptance", action="store_true")
    return parser.parse_args()


def _run_checked(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def _verify_elf_linkage(
    expected_backend: str | None,
    expected_openblas_threading: str | None,
) -> dict[str, Any]:
    extension = Path(_core.__file__).resolve()
    if platform.system() != "Linux":
        raise RuntimeError(
            "This benchmark's linkage verification currently requires Linux ELF tools"
        )

    dynamic = _run_checked(["readelf", "-d", str(extension)])
    resolved = _run_checked(["ldd", str(extension)])
    lowered = (dynamic + resolved).lower()
    has_openblas = "openblas" in lowered
    has_mkl = "libmkl" in lowered
    if has_openblas == has_mkl:
        raise RuntimeError("expected exactly one of OpenBLAS or MKL in the extension dependencies")
    backend = "openblas" if has_openblas else "mkl"
    if expected_backend is not None and backend != expected_backend:
        raise RuntimeError(f"expected {expected_backend}, but ELF linkage resolves to {backend}")

    openblas_threading = None
    if backend == "openblas":
        for line in resolved.splitlines():
            if "openblas" not in line.lower() or "=>" not in line:
                continue
            library_path = Path(line.split("=>", 1)[1].strip().split()[0]).resolve()
            real_path = str(library_path).lower()
            if "openmp" in real_path:
                openblas_threading = "openmp"
            elif "pthread" in real_path or "openblasp" in library_path.name.lower():
                openblas_threading = "pthread"
            break
        if (
            expected_openblas_threading is not None
            and openblas_threading != expected_openblas_threading
        ):
            raise RuntimeError(
                f"expected OpenBLAS-{expected_openblas_threading}, resolved {openblas_threading}"
            )

    return {
        "extension": str(extension),
        "backend": backend,
        "openblas_threading": openblas_threading,
        "readelf_dynamic": dynamic,
        "ldd": resolved,
    }


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _load_or_create_reduction(dataset: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.ascontiguousarray(np.load(cache, allow_pickle=False), dtype=np.float64)
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found and no reduction cache exists: {dataset}")

    source = ad.read_h5ad(dataset)
    if "action" not in source.obsm:
        an.reduce_kernel(source, n_components=30, key_added="action", seed=0, verbose=False)
    reduction = np.ascontiguousarray(source.obsm["action"], dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, reduction, allow_pickle=False)
    return reduction


def _minimal_adata(reduction: np.ndarray) -> AnnData:
    result = AnnData(X=sp.csr_matrix((reduction.shape[0], 0), dtype=np.float64))
    result.obsm["action"] = reduction.copy()
    return result


def _extract_action_outputs(adata: AnnData) -> dict[str, np.ndarray]:
    return {
        "assignments": np.asarray(adata.obs["assigned_archetype"], dtype=np.int64),
        "C_stacked": np.asarray(adata.obsm["C_stacked"]),
        "C_merged": np.asarray(adata.obsm["C_merged"]),
        "H_stacked": np.asarray(adata.obsm["H_stacked"]),
        "H_merged": np.asarray(adata.obsm["H_merged"]),
    }


def _assert_parity(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
    if not np.array_equal(reference["assignments"], candidate["assignments"]):
        mismatches = int(np.count_nonzero(reference["assignments"] != candidate["assignments"]))
        raise AssertionError(f"ACTION assignments differ at {mismatches} observations")
    for key in ("C_stacked", "C_merged", "H_stacked", "H_merged"):
        np.testing.assert_allclose(reference[key], candidate[key], rtol=RTOL, atol=ATOL)


def _benchmark_run_action(
    reduction: np.ndarray,
    thread_count: int,
) -> tuple[float, dict[str, np.ndarray]]:
    adata = _minimal_adata(reduction)
    started = time.perf_counter()
    an.run_action(adata, n_threads=thread_count, inplace=True)
    elapsed = time.perf_counter() - started
    return elapsed, _extract_action_outputs(adata)


def _benchmark_pipeline(reduction: np.ndarray, thread_count: int) -> float:
    adata = _minimal_adata(reduction)
    started = time.perf_counter()
    an.run_actionet(adata, n_threads=thread_count, inplace=True)
    return time.perf_counter() - started


def _load_parity(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _thread_value(label: str) -> int:
    return 0 if label == "auto" else int(label)


def _check_scaling(medians: dict[str, float]) -> None:
    if "1" not in medians or "auto" not in medians:
        raise AssertionError("scaling acceptance requires thread counts 1 and auto")
    multithread = [value for key, value in medians.items() if key not in ("1", "auto")]
    if not multithread:
        raise AssertionError("scaling acceptance requires at least one explicit multithread count")
    if medians["auto"] > medians["1"] / 2.0:
        raise AssertionError("automatic threading is less than 2x faster than one ACTION thread")
    if medians["auto"] > 1.2 * min(multithread):
        raise AssertionError(
            "automatic threading is more than 20% slower than the best thread count"
        )


def main() -> None:
    args = _parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.enforce_acceptance and args.mode == "pipeline":
        raise ValueError("--enforce-acceptance requires --mode run-action or both")

    linkage = _verify_elf_linkage(args.expect_backend, args.expect_openblas_threading)
    reduction = _load_or_create_reduction(args.dataset, args.reduction_cache)
    modes = ("run-action", "pipeline") if args.mode == "both" else (args.mode,)
    timings: dict[str, dict[str, list[float]]] = {mode: {} for mode in modes}
    medians: dict[str, dict[str, float]] = {mode: {} for mode in modes}
    parity: dict[str, np.ndarray] | None = None

    for mode in modes:
        for thread_label in args.threads:
            thread_count = _thread_value(thread_label)
            samples: list[float] = []
            for trial in range(args.trials):
                if mode == "run-action":
                    elapsed, outputs = _benchmark_run_action(reduction, thread_count)
                    if parity is None:
                        parity = outputs
                    else:
                        _assert_parity(parity, outputs)
                else:
                    elapsed = _benchmark_pipeline(reduction, thread_count)
                samples.append(elapsed)
                print(
                    f"{mode} threads={thread_label} trial={trial + 1}/{args.trials}: "
                    f"{elapsed:.3f}s",
                    flush=True,
                )
            timings[mode][thread_label] = samples
            medians[mode][thread_label] = statistics.median(samples)

    if args.parity_reference is not None:
        if parity is None:
            raise AssertionError("cross-backend parity requires --mode run-action or both")
        _assert_parity(_load_parity(args.parity_reference), parity)
    if args.parity_output is not None:
        if parity is None:
            raise AssertionError("parity output requires --mode run-action or both")
        args.parity_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.parity_output, **parity)

    if args.enforce_acceptance:
        _check_scaling(medians["run-action"])
        if linkage["backend"] == "mkl" and medians["run-action"]["auto"] > 9.40 * 1.10:
            raise AssertionError(
                "MKL run_action regressed by more than 10% from the 9.40s baseline"
            )
        if linkage["backend"] == "openblas":
            if args.timing_reference is None:
                raise AssertionError("OpenBLAS acceptance requires an MKL --timing-reference")
            reference = json.loads(args.timing_reference.read_text())
            if reference["linkage"]["backend"] != "mkl":
                raise AssertionError("timing reference is not an MKL benchmark")
            for mode in modes:
                if medians[mode]["auto"] > 1.5 * reference["medians"][mode]["auto"]:
                    raise AssertionError(f"OpenBLAS {mode} exceeds 1.5x the MKL median")

    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "dataset": str(args.dataset.resolve()),
        "reduction_cache": str(args.reduction_cache.resolve()),
        "reduction_shape": list(reduction.shape),
        "reduction_sha256": _sha256_array(reduction),
        "trials": args.trials,
        "linkage": linkage,
        "timings": timings,
        "medians": medians,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"backend": linkage["backend"], "medians": medians}, indent=2))


if __name__ == "__main__":
    main()
