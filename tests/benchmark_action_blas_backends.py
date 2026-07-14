#!/usr/bin/env python3
"""Reproducible ACTION benchmark for MKL and OpenBLAS builds.

Run this script from an environment containing exactly the actionet wheel to
measure.  The script verifies the extension's ELF dependencies, creates one
reusable reduction from ``data/test_adata.h5ad``, reports medians of three
trials, checks thread/output determinism, and records ACTION decision margins
from one untimed decomposition trace.
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
COEFFICIENT_SUPPORT_TOLERANCE = 1e-6
LANDMARK_PROXIMITY_TOLERANCE = 1e-3


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


def _decision_diagnostics(
    reduction: np.ndarray,
    outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Recompute an untimed trace and report margins at each decision gate.

    ``run_action`` returns H after merge-time normalization, so the returned
    matrices cannot reconstruct the earlier landmark predicate. Recomputing
    one decomposition outside the timed region keeps performance measurements
    honest while measuring every margin on the matrices that actually enter
    pruning and merging.
    """
    normalized = np.ascontiguousarray(an.tools.l1_norm_scale(reduction, axis=1))
    trace = _core.decomp_action(normalized, 2, 30, 50, 1e-100, 0)
    C_full = np.asarray(trace["C_stacked"], dtype=np.float64)
    H_full = np.asarray(trace["H_stacked"], dtype=np.float64)
    collected = _core.collect_archetypes(C_full, H_full, -3.0, 2)
    retained = np.asarray(collected["selected_archs"], dtype=np.int64)
    H_merged = np.asarray(outputs["H_merged"], dtype=np.float64)

    if C_full.shape != H_full.T.shape:
        raise AssertionError(
            "decision diagnostics require transposed full C/H decomposition buffers"
        )
    if C_full.ndim != 2 or retained.size == 0:
        raise AssertionError("decision diagnostics require retained archetypes")
    if retained.size != outputs["C_stacked"].shape[1]:
        raise AssertionError("diagnostic retained count disagrees with run_action output")

    h_max = np.max(H_full, axis=1, keepdims=True)
    landmark_distance = h_max - H_full
    landmark_mask = landmark_distance < LANDMARK_PROXIMITY_TOLERANCE
    strongest_landmark_support = np.max(
        np.where(landmark_mask, C_full.T, -np.inf),
        axis=1,
    )

    backbone = np.corrcoef(H_full)
    np.fill_diagonal(backbone, 0.0)
    backbone = np.maximum(np.nan_to_num(backbone, nan=0.0), 0.0)
    adjacency = (backbone > 0.0).astype(np.float64)
    adjacency_squared = adjacency @ adjacency
    strength = np.sum(backbone, axis=1)
    degree = np.sum(adjacency, axis=1)
    denominator = strength * (degree - 1.0)
    transitivity = np.zeros(H_full.shape[0], dtype=np.float64)
    valid = denominator > 0.0
    transitivity[valid] = np.sum(
        backbone[valid] * adjacency_squared[valid], axis=1
    ) / denominator[valid]
    transitivity_std = float(np.std(transitivity, ddof=1))
    if transitivity_std > 0.0:
        specificity_z = (transitivity - np.mean(transitivity)) / transitivity_std
    else:
        specificity_z = np.zeros_like(transitivity)

    C_retained = C_full[:, retained]
    H_retained = H_full[retained, :].copy()
    column_sums = np.sum(H_retained, axis=0)
    column_sums = np.where(column_sums > 0.0, column_sums, 1.0)
    H_retained /= column_sums[np.newaxis, :]
    H_arch = np.ascontiguousarray(H_retained @ C_retained)
    H_arch = np.nan_to_num(H_arch, nan=0.0)
    spa = _core.run_spa(H_arch, H_arch.shape[1])
    merge_scores = np.asarray(spa["norms"], dtype=np.float64)
    score_sum = float(np.sum(merge_scores))
    score_sq_sum = float(np.sum(np.square(merge_scores)))
    effective_rank = score_sum * score_sum / score_sq_sum if score_sq_sum > 0.0 else None

    assignment_gap = None
    if H_merged.ndim == 2 and H_merged.shape[1] >= 2:
        top_two = np.partition(H_merged, kth=-2, axis=1)[:, -2:]
        top_two.sort(axis=1)
        assignment_gap = float(np.min(top_two[:, 1] - top_two[:, 0]))

    return {
        "scope": "full-decomposition",
        "policy": {
            "coefficient_support_tolerance": COEFFICIENT_SUPPORT_TOLERANCE,
            "landmark_proximity_tolerance": LANDMARK_PROXIMITY_TOLERANCE,
        },
        "full_archetype_count": int(C_full.shape[1]),
        "retained_archetype_count": int(retained.size),
        "merged_archetype_count": int(outputs["C_merged"].shape[1]),
        "minimum_retained_landmark_support_margin": float(
            np.min(
                strongest_landmark_support[retained]
                - COEFFICIENT_SUPPORT_TOLERANCE
            )
        ),
        "minimum_landmark_support_boundary_margin": float(
            np.min(
                np.abs(
                    strongest_landmark_support
                    - COEFFICIENT_SUPPORT_TOLERANCE
                )
            )
        ),
        "minimum_coefficient_support_boundary_margin": float(
            np.min(np.abs(C_full - COEFFICIENT_SUPPORT_TOLERANCE))
        ),
        "minimum_landmark_proximity_boundary_margin": float(
            np.min(np.abs(landmark_distance - LANDMARK_PROXIMITY_TOLERANCE))
        ),
        "minimum_specificity_threshold_margin": float(
            np.min(np.abs(specificity_z - (-3.0)))
        ),
        "merge_effective_rank": effective_rank,
        "merge_effective_rank_selected_count": (
            int(np.floor(effective_rank + 0.5)) if effective_rank is not None else None
        ),
        "merge_effective_rank_rounding_margin": (
            float(abs(effective_rank - (np.floor(effective_rank) + 0.5)))
            if effective_rank is not None
            else None
        ),
        "assignment_top_two_minimum_gap": assignment_gap,
    }


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

    decision_diagnostics = (
        _decision_diagnostics(reduction, parity) if parity is not None else None
    )
    report = {
        "schema_version": 2,
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
        "decision_diagnostics": decision_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "backend": linkage["backend"],
                "medians": medians,
                "decision_diagnostics": decision_diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
