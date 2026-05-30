#!/usr/bin/env python3
"""benchmark_batch_correction.py — Sweep `correct_batch_effect` over batch counts.

Loads a real dataset and measures `actionet.correct_batch_effect` wall time
and peak RSS while sweeping the number of batches `b`. Designed to be run
twice — once before optimization (baseline) and once after — and to compare
results side-by-side.

Pipeline (matches tests/test_batchcorr.ipynb):
    filter_anndata → normalize_total → log1p → reduce_kernel(k=30)

For each `b` in --batch-counts:
    - Build a synthetic batch label vector by stratified random assignment over
      the existing `--batch-key` (default `UID`) categories so cells from a
      given UID stay together when `b <= n_unique_uid`. For `b > n_unique_uid`
      we randomly subdivide the largest UIDs.
    - Run `correct_batch_effect(adata.copy(), batch_key=...)` for `--trials`.
    - Record wall, peak RSS, sigma, and the perturbed S_r/U for parity probes.

Usage:
    python tests/benchmark_batch_correction.py \\
        --dataset data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad \\
        --output baseline.json --tag baseline

    python tests/benchmark_batch_correction.py \\
        --dataset data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad \\
        --output optimized.json --tag optimized \\
        --baseline-json baseline.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class StageProfiler:
    """Wall time + peak RSS delta context manager."""

    SAMPLE_INTERVAL = 0.05

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self.peak_rss_mb: float = 0.0
        self._peak_abs: float = 0.0

    def _sampler(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / 1e6
                if rss > self._peak_abs:
                    self._peak_abs = rss
            except Exception:
                pass
            time.sleep(self.SAMPLE_INTERVAL)

    def __enter__(self) -> "StageProfiler":
        gc.collect()
        self._proc = psutil.Process()
        self._rss0 = self._proc.memory_info().rss / 1e6
        self._peak_abs = self._rss0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.elapsed = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            rss = self._proc.memory_info().rss / 1e6
            self._peak_abs = max(self._peak_abs, rss)
        except Exception:
            pass
        self.peak_rss_mb = max(0.0, self._peak_abs - self._rss0)
        return False


# ---------------------------------------------------------------------------
# Batch label synthesis
# ---------------------------------------------------------------------------

def synthesize_batches(
    adata,
    batch_key: str,
    n_batches: int,
    seed: int = 0,
) -> np.ndarray:
    """Build a length-n_obs string vector with exactly `n_batches` unique levels.

    Strategy:
      - Group cells by the natural batch key (e.g. UID) so cells within a UID
        cluster together.
      - If n_batches <= n_unique_uid: merge UIDs into n_batches buckets by
        round-robin assignment of UIDs sorted by frequency (balances bucket
        sizes).
      - If n_batches  > n_unique_uid: subdivide the largest UIDs uniformly at
        random.
    """
    rng = np.random.default_rng(seed)
    obs = pd.Series(adata.obs[batch_key].astype(str).to_numpy(), name="orig")
    uids = obs.value_counts().index.tolist()  # sorted by frequency desc
    n_unique = len(uids)

    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if n_batches == 1:
        return np.full(adata.n_obs, "b0", dtype=object)

    if n_batches <= n_unique:
        bucket_of_uid: Dict[str, int] = {}
        for i, uid in enumerate(uids):
            bucket_of_uid[uid] = i % n_batches
        out = np.array([f"b{bucket_of_uid[v]}" for v in obs], dtype=object)
        return out

    out = np.empty(adata.n_obs, dtype=object)
    extra = n_batches - n_unique
    sizes = np.array([int((obs == u).sum()) for u in uids])
    splits_per_uid = np.zeros(n_unique, dtype=int)
    splits_per_uid[:] = 1
    for _ in range(extra):
        idx = int(np.argmax(sizes / np.maximum(splits_per_uid, 1)))
        splits_per_uid[idx] += 1

    next_label = 0
    obs_idx = obs.to_numpy()
    for j, uid in enumerate(uids):
        mask = (obs_idx == uid)
        n_uid = int(mask.sum())
        s = splits_per_uid[j]
        if s == 1:
            out[mask] = f"b{next_label}"
            next_label += 1
        else:
            sub = rng.integers(0, s, size=n_uid)
            for k in range(s):
                out[np.where(mask)[0][sub == k]] = f"b{next_label + k}"
            next_label += s
    return out


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    n_batches: int
    trial: int
    wall_s: float
    peak_rss_mb: float
    sigma: List[float] = field(default_factory=list)


@dataclass
class SweepResult:
    tag: str
    dataset: str
    n_obs: int
    n_vars: int
    nnz: int
    k: int
    trials_per_b: int
    seed: int
    git_sha: Optional[str]
    timestamp: str
    trials: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prepare_adata(dataset_path: Path, k: int, verbose: bool = True):
    """Load dataset, run filter/normalize/log1p/reduce_kernel once."""
    import anndata as ad
    import actionet
    import scanpy as sc

    if verbose:
        print(f"[load] {dataset_path}", flush=True)
    adata = ad.read_h5ad(str(dataset_path))

    if verbose:
        print(f"[prep] filter_anndata", flush=True)
    actionet.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
    if verbose:
        print(f"[prep] normalize_total + log1p", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata, base=2, copy=False)

    if verbose:
        print(f"[prep] reduce_kernel(k={k})", flush=True)
    actionet.reduce_kernel(adata, n_components=k, key_added="action", inplace=True)
    return adata


def run_sweep(
    dataset_path: Path,
    output_path: Path,
    tag: str,
    batch_counts: List[int],
    trials: int,
    seed: int,
    k: int,
    natural_batch_key: str,
) -> SweepResult:
    import actionet

    adata = prepare_adata(dataset_path, k=k, verbose=True)

    nnz = int(adata.X.nnz) if sp.issparse(adata.X) else int(np.prod(adata.X.shape))
    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)

    git_sha = _git_sha()
    out = SweepResult(
        tag=tag,
        dataset=str(dataset_path),
        n_obs=n_obs,
        n_vars=n_vars,
        nnz=nnz,
        k=k,
        trials_per_b=trials,
        seed=seed,
        git_sha=git_sha,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print(
        f"[info] dataset n_obs={n_obs:,} n_vars={n_vars:,} nnz={nnz:,} k={k}",
        flush=True,
    )

    # Cache the (post-reduce_kernel) state so we restore exactly the same
    # reduction inputs for every (b, trial) iteration.
    snap_S_r = adata.obsm["action"].copy()
    snap_B = adata.obsm["action_B"].copy()
    snap_U = adata.varm["action_U"].copy()
    snap_A = adata.varm["action_A"].copy()
    snap_params = dict(adata.uns["action_params"])

    for b in batch_counts:
        labels = synthesize_batches(adata, natural_batch_key, b, seed=seed)
        adata.obs["_bench_batch"] = pd.Categorical(labels)
        n_unique = int(adata.obs["_bench_batch"].cat.categories.size)

        for trial in range(trials):
            adata.obsm["action"] = snap_S_r.copy()
            adata.obsm["action_B"] = snap_B.copy()
            adata.varm["action_U"] = snap_U.copy()
            adata.varm["action_A"] = snap_A.copy()
            adata.uns["action_params"] = dict(snap_params)

            gc.collect()
            with StageProfiler() as prof:
                actionet.correct_batch_effect(
                    adata,
                    batch_key="_bench_batch",
                    reduction_key="action",
                    inplace=True,
                )

            sigma = list(map(float, adata.uns["action_corrected_params"]["sigma"]))
            tr = TrialResult(
                n_batches=n_unique,
                trial=trial,
                wall_s=prof.elapsed,
                peak_rss_mb=prof.peak_rss_mb,
                sigma=sigma,
            )
            out.trials.append(asdict(tr))

            # Clean corrected outputs to avoid bloat between trials
            for key in (
                "action_corrected",
                "action_corrected_B",
            ):
                if key in adata.obsm:
                    del adata.obsm[key]
            for key in ("action_corrected_U", "action_corrected_A"):
                if key in adata.varm:
                    del adata.varm[key]
            if "action_corrected_params" in adata.uns:
                del adata.uns["action_corrected_params"]

            print(
                f"[run] b={n_unique:>4d} trial={trial} "
                f"wall={prof.elapsed:7.3f}s rss_delta={prof.peak_rss_mb:7.1f}MB",
                flush=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(asdict(out), fh, indent=2, default=_json_safe)
    print(f"[done] wrote {output_path}", flush=True)
    return out


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Non-serialisable: {type(obj)}")


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> str:
    """Pretty-print a baseline vs optimized comparison table."""
    base_by_b = _group_by_b(baseline["trials"])
    opt_by_b = _group_by_b(optimized["trials"])

    bs = sorted(set(base_by_b) | set(opt_by_b))
    lines: List[str] = []
    lines.append(f"# Comparison: {baseline.get('tag','baseline')} vs {optimized.get('tag','optimized')}")
    lines.append(f"dataset: {baseline.get('dataset')}")
    lines.append(
        f"n_obs={baseline.get('n_obs')} n_vars={baseline.get('n_vars')} nnz={baseline.get('nnz')} k={baseline.get('k')}"
    )
    lines.append("")
    lines.append(
        "| n_batches | base_wall_s | opt_wall_s | speedup | base_rss_mb | opt_rss_mb | sigma_max_abs_diff |"
    )
    lines.append(
        "|-----------|-------------|------------|---------|-------------|------------|--------------------|"
    )
    for b in bs:
        base_rows = base_by_b.get(b, [])
        opt_rows = opt_by_b.get(b, [])
        bw = _median([r["wall_s"] for r in base_rows]) if base_rows else float("nan")
        ow = _median([r["wall_s"] for r in opt_rows]) if opt_rows else float("nan")
        br = _median([r["peak_rss_mb"] for r in base_rows]) if base_rows else float("nan")
        orr = _median([r["peak_rss_mb"] for r in opt_rows]) if opt_rows else float("nan")
        speedup = (bw / ow) if ow and not np.isnan(ow) and not np.isnan(bw) else float("nan")
        sd = _max_sigma_diff(base_rows, opt_rows)
        lines.append(
            f"| {b:>9d} | {bw:11.3f} | {ow:10.3f} | {speedup:7.2f}x | "
            f"{br:11.1f} | {orr:10.1f} | {sd:.3e} |"
        )
    return "\n".join(lines)


def _group_by_b(trials: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in trials:
        out.setdefault(int(r["n_batches"]), []).append(r)
    return out


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(values))


def _max_sigma_diff(base_rows, opt_rows) -> float:
    if not base_rows or not opt_rows:
        return float("nan")
    bs = np.asarray(base_rows[0]["sigma"], dtype=float)
    os_ = np.asarray(opt_rows[0]["sigma"], dtype=float)
    if bs.size != os_.size:
        return float("inf")
    return float(np.max(np.abs(bs - os_)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch-correction batch-count sweep benchmark")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, required=True, help="JSON output path")
    p.add_argument("--tag", type=str, default="run", help="Label for this run (e.g. baseline / optimized)")
    p.add_argument(
        "--batch-counts",
        type=int,
        nargs="+",
        default=[2, 5, 10, 25, 50, 100],
        help="Values of b to sweep",
    )
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=30, help="Reduction rank")
    p.add_argument("--natural-batch-key", type=str, default="UID")
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="If given, also print a baseline-vs-this comparison table",
    )
    args = p.parse_args(argv)

    if not args.dataset.exists():
        print(f"[err] dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    res = run_sweep(
        dataset_path=args.dataset,
        output_path=args.output,
        tag=args.tag,
        batch_counts=list(args.batch_counts),
        trials=args.trials,
        seed=args.seed,
        k=args.k,
        natural_batch_key=args.natural_batch_key,
    )

    if args.baseline_json is not None:
        if args.baseline_json.exists():
            with open(args.baseline_json) as fh:
                baseline = json.load(fh)
            print()
            print(compare(baseline, asdict(res)))
        else:
            print(f"[warn] --baseline-json not found: {args.baseline_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
