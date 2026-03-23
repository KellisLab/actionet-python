#!/usr/bin/env python3
"""benchmark_backed_svd_algorithm.py — Backed SVD algorithm comparison: Halko vs IRLB.

Benchmarks run_svd() with algorithm='halko' and algorithm='irlb' on backed
(HDF5-streamed) AnnData objects across dataset size tiers.

Metrics collected per trial:
  - wall_s          : wall-clock seconds
  - peak_rss_mb     : peak RSS increase (MB) during the call
  - io_read_mb      : bytes read from storage (MB)
  - sigma_corr      : Pearson correlation of singular values vs reference (halko)
  - reconstruction_err : relative Frobenius reconstruction error ||A - U D V'||_F / ||A||_F
                         estimated via 500-row random probe

Usage:
  python tests/benchmark_backed_svd_algorithm.py [--output-dir PATH] [--tiers 25k 50k 100k]
                                                  [--n-components 30] [--trials 2]
                                                  [--chunk-size 4096]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import psutil
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
BENCHMARK_DATA_DIR = Path("/data/actionet_benchmark")

PYTHON_EXE = str(REPO_ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON_EXE).exists():
    PYTHON_EXE = sys.executable

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

TIERS = [25_000, 50_000, 100_000, 150_000, 200_000]
TIER_LABELS = {t: f"{t // 1000}k" for t in TIERS}

ALGORITHMS = ["halko", "irlb"]
DEFAULT_N_COMPONENTS = 30
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_TRIALS = 2

# Accuracy probe: sample this many rows to estimate reconstruction error
PROBE_ROWS = 500


# ---------------------------------------------------------------------------
# Per-stage profiler (wall time + peak RSS delta)
# ---------------------------------------------------------------------------

class StageProfiler:
    """Context manager: wall time + peak RSS delta for one stage."""

    SAMPLE_INTERVAL = 0.05

    def __init__(self):
        self.elapsed: float = 0.0
        self.peak_rss_mb: float = 0.0
        self._peak_abs: float = 0.0

    def _sampler(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / 1e6
                if rss > self._peak_abs:
                    self._peak_abs = rss
            except Exception:
                pass
            time.sleep(self.SAMPLE_INTERVAL)

    def __enter__(self):
        gc.collect()
        self._proc = psutil.Process()
        self._rss0 = self._proc.memory_info().rss / 1e6
        self._peak_abs = self._rss0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
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


def _io_counters_mb() -> Tuple[float, float]:
    try:
        c = psutil.Process().io_counters()
        return c.read_bytes / 1e6, c.write_bytes / 1e6
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    dataset: str
    n_obs: int
    n_vars: int
    nnz: int
    n_components: int
    chunk_size: int
    algorithm: str
    trial: int
    wall_s: float
    peak_rss_mb: float
    io_read_mb: float
    sigma_corr: float          # vs halko reference; NaN for halko itself
    reconstruction_err: float  # relative Frobenius on probe rows; NaN if skipped
    status: str
    failure_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _sigma_correlation(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    """Pearson correlation between two singular-value vectors (same length k)."""
    k = min(len(sigma_a), len(sigma_b))
    if k < 2:
        return float("nan")
    a, b = sigma_a[:k], sigma_b[:k]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _reconstruction_error(
    adata: ad.AnnData,
    svd_result: Dict[str, np.ndarray],
    layer: Optional[str],
    probe_rows: int,
    rng: np.random.Generator,
) -> float:
    """Relative Frobenius reconstruction error on a random row probe.

    ||A_probe - U_probe D V'||_F / ||A_probe||_F

    Uses in-memory materialisation of probe_rows rows only.
    """
    try:
        n_obs = adata.n_obs
        idx = rng.choice(n_obs, size=min(probe_rows, n_obs), replace=False)
        idx_sorted = np.sort(idx)

        if layer is None:
            X_probe = adata.X[idx_sorted]
        else:
            X_probe = adata.layers[layer][idx_sorted]

        if sp.issparse(X_probe):
            X_probe = np.asarray(X_probe.todense(), dtype=float)
        else:
            X_probe = np.asarray(X_probe, dtype=float)

        u = np.asarray(svd_result["u"], dtype=float)
        d = np.asarray(svd_result["d"], dtype=float).ravel()
        v = np.asarray(svd_result["v"], dtype=float)

        # Reconstruct: A_approx = U D V';  probe rows use u[idx_sorted, :]
        u_probe = u[idx_sorted, :]
        A_approx = (u_probe * d) @ v.T  # (probe_rows, n_vars)

        diff = X_probe - A_approx
        denom = np.linalg.norm(X_probe, "fro")
        if denom == 0:
            return float("nan")
        return float(np.linalg.norm(diff, "fro") / denom)
    except Exception as exc:
        print(f"    [accuracy] reconstruction_error failed: {exc}", flush=True)
        return float("nan")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _dataset_path(tier_label: str) -> Path:
    return BENCHMARK_DATA_DIR / f"scale_subset_{tier_label}.h5ad"


def _get_nnz(adata: ad.AnnData) -> int:
    try:
        if sp.issparse(adata.X):
            return int(adata.X.nnz)
        return int(np.prod(adata.X.shape))
    except Exception:
        return 0


def _make_backed_copy(src_path: Path, work_dir: Path, label: str) -> Path:
    """Copy h5ad to a writable work dir for backed access."""
    import shutil
    dst = work_dir / f"_svdalg_{label}.h5ad"
    shutil.copy2(str(src_path), str(dst))
    return dst


def _decompress_if_needed(adata: ad.AnnData) -> None:
    try:
        from actionet._backed_compression import (
            get_storage_metadata_from_adata,
            is_compressed_storage,
        )
        import actionet as an
        if is_compressed_storage(get_storage_metadata_from_adata(adata)):
            print("      decompressing backed storage ...", flush=True)
            an.decompress_backed_storage(adata, scope="file", chunk_size=4096, verbose=False)
    except Exception as exc:
        print(f"      decompression check failed (ignored): {exc}", flush=True)


# ---------------------------------------------------------------------------
# Core: run one (dataset, algorithm, trial) inside a child process
# ---------------------------------------------------------------------------

def run_one_in_process(
    tier_label: str,
    algorithm: str,
    trial: int,
    n_components: int,
    chunk_size: int,
    ref_sigma: Optional[np.ndarray],  # halko reference for sigma_corr (passed as list)
    output_jsonl: str,
    work_dir: str,
) -> None:
    """Full benchmark run for one (tier, algorithm, trial) combination.

    Intended to run inside a child subprocess.
    """
    import actionet as an

    src_path = _dataset_path(tier_label)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    backed_path = _make_backed_copy(src_path, work, f"{tier_label}_{algorithm}_t{trial}")
    adata = ad.read_h5ad(str(backed_path), backed="r+")

    try:
        n_obs = int(adata.n_obs)
        n_vars = int(adata.n_vars)
        nnz = _get_nnz(adata)

        # Preprocessing (not timed — same for all algorithms)
        _decompress_if_needed(adata)
        an.filter_anndata(adata, min_cells_per_feat=0.01,
                          backed_chunk_size=chunk_size, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2,
                             backed_chunk_size=chunk_size, inplace=True)

        # Close Python handle before C++ backed operator opens same file
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass

        # --- Timed SVD ---
        io_r0, _ = _io_counters_mb()
        with StageProfiler() as prof:
            result = an.run_svd(
                adata,
                n_components=n_components,
                algorithm=algorithm,
                seed=42,
                verbose=False,
                backed_chunk_size=chunk_size,
            )
        io_r1, _ = _io_counters_mb()

        wall_s = prof.elapsed
        peak_rss_mb = prof.peak_rss_mb
        io_read_mb = max(0.0, io_r1 - io_r0)

        sigma = np.asarray(result["d"]).ravel()

        # Sigma correlation vs reference
        if ref_sigma is not None:
            ref_arr = np.array(ref_sigma)
            sigma_corr = _sigma_correlation(sigma, ref_arr)
        else:
            sigma_corr = float("nan")

        # Reconstruction error
        adata2 = ad.read_h5ad(str(backed_path), backed="r")
        rng = np.random.default_rng(99)
        rec_err = _reconstruction_error(adata2, result, layer=None, probe_rows=PROBE_ROWS, rng=rng)
        try:
            if hasattr(adata2, "file") and adata2.file is not None:
                adata2.file.close()
        except Exception:
            pass
        del adata2

        row = BenchRow(
            dataset=tier_label,
            n_obs=n_obs,
            n_vars=n_vars,
            nnz=nnz,
            n_components=n_components,
            chunk_size=chunk_size,
            algorithm=algorithm,
            trial=trial,
            wall_s=wall_s,
            peak_rss_mb=peak_rss_mb,
            io_read_mb=io_read_mb,
            sigma_corr=sigma_corr,
            reconstruction_err=rec_err,
            status="ok",
            failure_reason=None,
        )

        # Write result row
        with open(output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(row.to_json() + "\n")

        print(
            f"  [{tier_label} {algorithm} t{trial}] "
            f"wall={wall_s:.2f}s  rss={peak_rss_mb:.0f}MB  "
            f"io_read={io_read_mb:.0f}MB  sigma_corr={sigma_corr:.6f}  rec_err={rec_err:.6f}",
            flush=True,
        )

    except Exception as exc:
        row = BenchRow(
            dataset=tier_label,
            n_obs=0, n_vars=0, nnz=0,
            n_components=n_components,
            chunk_size=chunk_size,
            algorithm=algorithm,
            trial=trial,
            wall_s=0.0, peak_rss_mb=0.0, io_read_mb=0.0,
            sigma_corr=float("nan"),
            reconstruction_err=float("nan"),
            status="failed",
            failure_reason=str(exc),
        )
        with open(output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(row.to_json() + "\n")
        print(f"  [{tier_label} {algorithm} t{trial}] FAILED: {exc}", flush=True)
        traceback.print_exc()
    finally:
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        try:
            backed_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Halko reference sigma collector (in-process, used as baseline)
# ---------------------------------------------------------------------------

def collect_halko_sigma(
    tier_label: str,
    n_components: int,
    chunk_size: int,
    work_dir: Path,
) -> Optional[np.ndarray]:
    """Run halko once in-process to get a reference sigma vector for accuracy comparison."""
    import actionet as an

    src_path = _dataset_path(tier_label)
    backed_path = _make_backed_copy(src_path, work_dir, f"{tier_label}_halko_ref")
    adata = ad.read_h5ad(str(backed_path), backed="r+")
    try:
        _decompress_if_needed(adata)
        an.filter_anndata(adata, min_cells_per_feat=0.01,
                          backed_chunk_size=chunk_size, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2,
                             backed_chunk_size=chunk_size, inplace=True)
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        result = an.run_svd(
            adata,
            n_components=n_components,
            algorithm="halko",
            seed=42,
            verbose=False,
            backed_chunk_size=chunk_size,
        )
        return np.asarray(result["d"]).ravel()
    except Exception as exc:
        print(f"  [ref sigma] failed for {tier_label}: {exc}", flush=True)
        return None
    finally:
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        try:
            backed_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Child-process dispatch
# ---------------------------------------------------------------------------

def _child_main(kwargs_json: str) -> None:
    import json as _json
    kw = _json.loads(kwargs_json)
    ref_sigma = kw.pop("ref_sigma", None)
    if ref_sigma is not None:
        ref_sigma = np.array(ref_sigma)
    kw["work_dir"] = str(kw["work_dir"])
    run_one_in_process(ref_sigma=ref_sigma, **kw)


def dispatch_child(
    tier_label: str,
    algorithm: str,
    trial: int,
    n_components: int,
    chunk_size: int,
    ref_sigma: Optional[np.ndarray],
    output_jsonl: Path,
    work_dir: Path,
    timeout_s: float = 3600.0,
) -> str:
    """Run one (tier, algorithm, trial) in a subprocess. Returns status string."""
    import subprocess

    kw: Dict[str, Any] = {
        "tier_label": tier_label,
        "algorithm": algorithm,
        "trial": trial,
        "n_components": n_components,
        "chunk_size": chunk_size,
        "ref_sigma": ref_sigma.tolist() if ref_sigma is not None else None,
        "output_jsonl": str(output_jsonl),
        "work_dir": str(work_dir),
    }
    kwargs_json = json.dumps(kw)
    tests_dir = str(Path(__file__).resolve().parent)
    cmd = [
        PYTHON_EXE, "-c",
        f"import sys; sys.path.insert(0,{repr(tests_dir)}); "
        f"from benchmark_backed_svd_algorithm import _child_main; "
        f"_child_main({repr(kwargs_json)})"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout_s
    while True:
        line = proc.stdout.readline()
        if line:
            print(line, end="", flush=True)
        if proc.poll() is not None:
            try:
                rest = proc.stdout.read()
                if rest:
                    print(rest, end="", flush=True)
            except Exception:
                pass
            break
        if time.monotonic() > deadline:
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait(timeout=5)
            return "timeout"
        time.sleep(0.3)
    return "ok" if proc.returncode == 0 else "failed"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(output_dir: Path, jsonl_path: Path) -> None:
    import pandas as pd

    rows = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not rows:
        print("  [report] No results to report.", flush=True)
        return

    df = pd.DataFrame(rows)
    for col in ["wall_s", "peak_rss_mb", "io_read_mb", "sigma_corr",
                "reconstruction_err", "n_obs", "n_vars", "nnz"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    csv_path = output_dir / "svd_algorithm_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [report] CSV: {csv_path}", flush=True)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("  [report] No successful rows.", flush=True)
        return

    # Summary table: mean across trials, grouped by (dataset, algorithm)
    summary = (
        ok.groupby(["dataset", "n_obs", "algorithm"])[
            ["wall_s", "peak_rss_mb", "io_read_mb", "sigma_corr", "reconstruction_err"]
        ]
        .mean()
        .reset_index()
        .sort_values(["n_obs", "algorithm"])
    )

    # Speed ratio: irlb_wall / halko_wall per dataset
    pivot_wall = ok.groupby(["dataset", "algorithm"])["wall_s"].mean().unstack("algorithm")
    if "irlb" in pivot_wall.columns and "halko" in pivot_wall.columns:
        pivot_wall["irlb_vs_halko"] = pivot_wall["irlb"] / pivot_wall["halko"]

    # Memory ratio
    pivot_mem = ok.groupby(["dataset", "algorithm"])["peak_rss_mb"].mean().unstack("algorithm")
    if "irlb" in pivot_mem.columns and "halko" in pivot_mem.columns:
        pivot_mem["irlb_vs_halko"] = pivot_mem["irlb"] / pivot_mem["halko"]

    report_path = output_dir / "svd_algorithm_benchmark.md"
    lines = [
        "# Backed SVD Algorithm Benchmark: Halko vs IRLB",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
    ]

    if not ok.empty:
        lines += [
            f"- n_components: {int(ok['n_components'].iloc[0])}",
            f"- chunk_size: {int(ok['chunk_size'].iloc[0])}",
            f"- trials per config: {int(ok.groupby(['dataset','algorithm'])['trial'].nunique().max())}",
            f"- reconstruction probe rows: {PROBE_ROWS}",
            "",
        ]

    lines += ["## Summary (mean across trials)", ""]
    lines.append(summary.to_markdown(index=False))
    lines.append("")

    lines += ["## Speed Comparison (wall seconds)", ""]
    lines.append(pivot_wall.to_markdown())
    lines.append("")

    lines += ["## Memory Comparison (peak RSS MB)", ""]
    lines.append(pivot_mem.to_markdown())
    lines.append("")

    # Accuracy section
    acc = ok[ok["algorithm"] == "irlb"][["dataset", "n_obs", "sigma_corr", "reconstruction_err"]]
    if not acc.empty:
        lines += [
            "## Accuracy (IRLB vs Halko reference)",
            "",
            "> `sigma_corr`: Pearson correlation of singular values between IRLB and Halko.",
            "> `reconstruction_err`: relative Frobenius error ||A_probe - U D V'||_F / ||A_probe||_F",
            "> estimated on a random 500-row probe of the normalised matrix.",
            "",
        ]
        acc_agg = acc.groupby(["dataset", "n_obs"])[["sigma_corr", "reconstruction_err"]].mean().reset_index()
        lines.append(acc_agg.to_markdown(index=False))
        lines.append("")

    # Recommendation
    lines += ["## Recommendation", ""]
    try:
        if "irlb" in pivot_wall.columns and "halko" in pivot_wall.columns:
            med_ratio = float(pivot_wall["irlb_vs_halko"].dropna().median())
            med_corr = float(
                ok[ok["algorithm"] == "irlb"]["sigma_corr"].dropna().median()
            )
            if med_ratio < 0.85 and med_corr > 0.9999:
                rec = (
                    f"**Change default to IRLB.** "
                    f"IRLB is {1/med_ratio:.2f}x faster than Halko (median ratio {med_ratio:.2f}) "
                    f"with near-identical accuracy (sigma_corr median={med_corr:.6f})."
                )
            elif med_ratio < 1.10 and med_corr > 0.9999:
                rec = (
                    f"**Keep Halko as default.** "
                    f"IRLB speed is comparable (ratio {med_ratio:.2f}) but offers no clear "
                    f"advantage. Halko has a fixed and predictable matvec count regardless of "
                    f"matrix conditioning, making its I/O cost easier to reason about at scale."
                )
            elif med_ratio >= 1.10:
                rec = (
                    f"**Keep Halko as default.** "
                    f"IRLB is {med_ratio:.2f}x slower than Halko (median). "
                    f"The additional matvec passes imposed by iterative refinement hurt "
                    f"backed I/O throughput more than they help accuracy."
                )
            else:
                rec = (
                    f"**Inconclusive.** "
                    f"IRLB speed ratio={med_ratio:.2f}, sigma_corr={med_corr:.6f}. "
                    f"Manual inspection of the full table is recommended."
                )
            lines.append(rec)
            lines.append("")
    except Exception:
        pass

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [report] Report: {report_path}", flush=True)

    # Terminal summary
    print("\n" + "="*70, flush=True)
    print("SVD ALGORITHM BENCHMARK SUMMARY", flush=True)
    print("="*70, flush=True)
    print(summary.to_string(index=False), flush=True)
    if "irlb_vs_halko" in pivot_wall.columns:
        print("\nSpeed ratio (IRLB wall / Halko wall) — <1.0 means IRLB is faster:", flush=True)
        print(pivot_wall[["halko", "irlb", "irlb_vs_halko"]].to_string(), flush=True)
    print("="*70, flush=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    tiers: List[str],
    n_components: int,
    chunk_size: int,
    trials: int,
    output_dir: Path,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "raw_results.jsonl"

    # Load already-completed cases for resume
    completed: set = set()
    if resume and jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        key = (row["dataset"], row["algorithm"], int(row["trial"]))
                        completed.add(key)
                except Exception:
                    pass
        print(f"  [resume] {len(completed)} completed cases found.", flush=True)

    for tier_label in tiers:
        src_path = _dataset_path(tier_label)
        if not src_path.exists():
            print(f"  [skip] dataset not found: {src_path}", flush=True)
            continue

        print(f"\n{'='*70}", flush=True)
        print(f"  TIER: {tier_label}", flush=True)
        print(f"{'='*70}", flush=True)

        # Collect halko reference sigma (one run, in-process, not counted in timing)
        print(f"  Collecting halko reference sigma for accuracy comparison ...", flush=True)
        ref_sigma = collect_halko_sigma(tier_label, n_components, chunk_size, work_dir)
        if ref_sigma is None:
            print(f"  WARNING: could not collect halko reference sigma for {tier_label}.", flush=True)

        for algorithm in ALGORITHMS:
            for trial in range(1, trials + 1):
                key = (tier_label, algorithm, trial)
                if resume and key in completed:
                    print(f"  [skip] {tier_label} {algorithm} t{trial} (already complete)", flush=True)
                    continue

                print(f"\n  Running: {tier_label} {algorithm} trial={trial}", flush=True)
                # For halko, sigma_corr is NaN (it is its own reference); pass ref anyway for consistency
                ref = ref_sigma if algorithm == "irlb" else None
                status = dispatch_child(
                    tier_label=tier_label,
                    algorithm=algorithm,
                    trial=trial,
                    n_components=n_components,
                    chunk_size=chunk_size,
                    ref_sigma=ref,
                    output_jsonl=jsonl_path,
                    work_dir=work_dir,
                )
                if status != "ok":
                    print(f"  WARN: {tier_label} {algorithm} t{trial} returned status={status}", flush=True)

    print("\nGenerating report ...", flush=True)
    generate_report(output_dir, jsonl_path)
    print(f"\nDone. Results in: {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark backed SVD: Halko vs IRLB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tiers", nargs="+",
        default=["25k", "50k", "100k"],
        help="Tier labels to benchmark (e.g. 25k 50k 100k). "
             "Corresponding scale_subset_<tier>.h5ad must exist in "
             f"{BENCHMARK_DATA_DIR}",
    )
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: tests/benchmark_results/svd_alg_<timestamp>)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases already completed in an existing output dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = time.strftime("svd_alg_%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "tests" / "benchmark_results" / run_id

    print(f"Backed SVD Algorithm Benchmark: Halko vs IRLB", flush=True)
    print(f"Tiers     : {args.tiers}", flush=True)
    print(f"Components: {args.n_components}", flush=True)
    print(f"Chunk size: {args.chunk_size}", flush=True)
    print(f"Trials    : {args.trials}", flush=True)
    print(f"Output    : {output_dir}", flush=True)
    print(flush=True)

    run_benchmark(
        tiers=args.tiers,
        n_components=args.n_components,
        chunk_size=args.chunk_size,
        trials=args.trials,
        output_dir=output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
