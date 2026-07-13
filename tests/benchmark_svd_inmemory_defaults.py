#!/usr/bin/env python3
"""benchmark_svd_inmemory_defaults.py -- In-memory SVD algorithm comparison.

Benchmarks run_svd() with algorithm in {irlb, halko} across the two
in-memory storage forms (sparse CSR and dense float64) using the same
tier-subset datasets the backed benchmark consumes. Purpose: settle the
sparse-in-mem and dense-in-mem defaults in
`actionet.decomposition.svd._select_svd_algorithm_inmemory`.

Metrics per (dataset, storage_form, algorithm, trial):
  - wall_s              : wall-clock seconds
  - peak_rss_mb         : peak RSS increase (MB) during the SVD call
  - sigma_corr          : Pearson correlation of singular values vs IRLB reference
  - reconstruction_err  : relative Frobenius reconstruction error ||A - U D V'||_F / ||A||_F
                          estimated on a random 500-row probe

Each (tier, storage, algorithm, trial) case runs in a fresh subprocess so
peak RSS is a clean per-case delta and C++ side effects cannot leak.

Usage:
  python tests/benchmark_svd_inmemory_defaults.py \\
      [--tiers 25k 50k 100k 150k 200k] \\
      [--n-components 30] [--trials 2] \\
      [--skip-dense-above 100k] \\
      [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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

_BENCHMARK_DATA_DIR_ENV = os.environ.get("ACTIONET_BENCHMARK_DATA_DIR", "").strip()
if _BENCHMARK_DATA_DIR_ENV:
    BENCHMARK_DATA_DIR = Path(_BENCHMARK_DATA_DIR_ENV)
elif Path("/data/actionet_benchmark").exists():
    BENCHMARK_DATA_DIR = Path("/data/actionet_benchmark")
else:
    BENCHMARK_DATA_DIR = REPO_ROOT / "data" / "actionet_benchmark"

PYTHON_EXE = str(REPO_ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON_EXE).exists():
    PYTHON_EXE = sys.executable

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

DEFAULT_TIERS = ["25k", "50k", "100k", "150k", "200k"]
ALGORITHMS = ["irlb", "halko"]
STORAGE_FORMS = ["sparse", "dense"]
DEFAULT_N_COMPONENTS = 30
DEFAULT_TRIALS = 2

# Rough dense-matrix memory guard: dense = n_obs * n_vars * 8 bytes (float64).
# Skip dense cases whose dense matrix exceeds this many GB; also configurable
# via --skip-dense-above.
DEFAULT_DENSE_TIER_CAP = "100k"

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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    dataset: str
    n_obs: int
    n_vars: int
    nnz: int
    storage_form: str
    n_components: int
    algorithm: str
    trial: int
    wall_s: float
    peak_rss_mb: float
    sigma_corr: float          # vs IRLB reference; NaN for IRLB itself
    reconstruction_err: float
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
    k = min(len(sigma_a), len(sigma_b))
    if k < 2:
        return float("nan")
    a, b = sigma_a[:k], sigma_b[:k]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _reconstruction_error_inmem(
    X: Any,
    svd_result: Dict[str, np.ndarray],
    probe_rows: int,
    rng: np.random.Generator,
) -> float:
    """Relative Frobenius reconstruction error on a random row probe.

    Works uniformly for both sparse (CSR) and dense in-memory matrices.
    """
    try:
        n_obs = X.shape[0]
        idx = rng.choice(n_obs, size=min(probe_rows, n_obs), replace=False)
        idx_sorted = np.sort(idx)

        X_probe = X[idx_sorted]
        if sp.issparse(X_probe):
            X_probe = np.asarray(X_probe.todense(), dtype=float)
        else:
            X_probe = np.asarray(X_probe, dtype=float)

        u = np.asarray(svd_result["u"], dtype=float)
        d = np.asarray(svd_result["d"], dtype=float).ravel()
        v = np.asarray(svd_result["v"], dtype=float)

        u_probe = u[idx_sorted, :]
        A_approx = (u_probe * d) @ v.T

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


def _tier_int(tier_label: str) -> int:
    """Parse '25k' / '200k' / '1M' into an int cell count."""
    lbl = tier_label.strip().lower()
    if lbl.endswith("k"):
        return int(float(lbl[:-1]) * 1_000)
    if lbl.endswith("m"):
        return int(float(lbl[:-1]) * 1_000_000)
    return int(lbl)


def _get_nnz_sparse(X: sp.spmatrix) -> int:
    try:
        return int(X.nnz)
    except Exception:
        return 0


def _prepare_sparse_matrix(tier_label: str, verbose: bool = False) -> Tuple[Any, int, int, int]:
    """Load the tier subset in-memory and run the standard preprocessing.

    Returns (sparse_csr, n_obs, n_vars, nnz).
    """
    import actionet as an

    src_path = _dataset_path(tier_label)
    if verbose:
        print(f"      loading {src_path.name} into memory ...", flush=True)
    adata = ad.read_h5ad(str(src_path))

    an.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
    an.normalize_anndata(
        adata,
        target_sum=1e4,
        log_transform=True,
        log_base=2,
        inplace=True,
    )

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sp.isspmatrix_csr(X):
        X = X.tocsr()

    n_obs, n_vars = X.shape
    nnz = _get_nnz_sparse(X)

    del adata
    gc.collect()
    return X, int(n_obs), int(n_vars), int(nnz)


# ---------------------------------------------------------------------------
# Core: run one (dataset, storage, algorithm, trial) inside a child process
# ---------------------------------------------------------------------------

def run_one_in_process(
    tier_label: str,
    storage_form: str,
    algorithm: str,
    trial: int,
    n_components: int,
    ref_sigma: Optional[np.ndarray],  # IRLB reference for sigma_corr (passed as list from parent)
    output_jsonl: str,
) -> None:
    """Full benchmark run for one (tier, storage, algorithm, trial) combo.

    Intended to run inside a child subprocess for clean RSS accounting.
    """
    import actionet as an

    try:
        X_sparse, n_obs, n_vars, nnz = _prepare_sparse_matrix(tier_label, verbose=True)

        if storage_form == "sparse":
            X = X_sparse
        elif storage_form == "dense":
            X = np.asarray(X_sparse.todense(), dtype=np.float64, order="C")
            del X_sparse
            gc.collect()
        else:
            raise ValueError(f"Unknown storage_form: {storage_form}")

        # --- Timed SVD ---
        with StageProfiler() as prof:
            result = an.run_svd(
                X,
                n_components=n_components,
                algorithm=algorithm,
                seed=42,
                verbose=False,
            )

        wall_s = prof.elapsed
        peak_rss_mb = prof.peak_rss_mb

        sigma = np.asarray(result["d"]).ravel()

        # Sigma correlation vs reference
        if ref_sigma is not None:
            ref_arr = np.array(ref_sigma)
            sigma_corr = _sigma_correlation(sigma, ref_arr)
        else:
            sigma_corr = float("nan")

        rng = np.random.default_rng(99)
        rec_err = _reconstruction_error_inmem(X, result, probe_rows=PROBE_ROWS, rng=rng)

        row = BenchRow(
            dataset=tier_label,
            n_obs=n_obs,
            n_vars=n_vars,
            nnz=nnz,
            storage_form=storage_form,
            n_components=n_components,
            algorithm=algorithm,
            trial=trial,
            wall_s=wall_s,
            peak_rss_mb=peak_rss_mb,
            sigma_corr=sigma_corr,
            reconstruction_err=rec_err,
            status="ok",
            failure_reason=None,
        )
        with open(output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(row.to_json() + "\n")

        print(
            f"  [{tier_label} {storage_form} {algorithm} t{trial}] "
            f"wall={wall_s:.2f}s  rss={peak_rss_mb:.0f}MB  "
            f"sigma_corr={sigma_corr:.6f}  rec_err={rec_err:.6f}",
            flush=True,
        )

    except Exception as exc:
        row = BenchRow(
            dataset=tier_label,
            n_obs=0, n_vars=0, nnz=0,
            storage_form=storage_form,
            n_components=n_components,
            algorithm=algorithm,
            trial=trial,
            wall_s=0.0, peak_rss_mb=0.0,
            sigma_corr=float("nan"),
            reconstruction_err=float("nan"),
            status="failed",
            failure_reason=str(exc),
        )
        with open(output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(row.to_json() + "\n")
        print(f"  [{tier_label} {storage_form} {algorithm} t{trial}] FAILED: {exc}", flush=True)
        traceback.print_exc()
    finally:
        gc.collect()


def collect_reference_sigma(
    tier_label: str,
    storage_form: str,
    n_components: int,
) -> Optional[np.ndarray]:
    """Run IRLB once in-process for a (tier, storage_form) to get a reference sigma."""
    import actionet as an
    try:
        X_sparse, _, _, _ = _prepare_sparse_matrix(tier_label, verbose=False)
        if storage_form == "sparse":
            X = X_sparse
        else:
            X = np.asarray(X_sparse.todense(), dtype=np.float64, order="C")
            del X_sparse
            gc.collect()
        result = an.run_svd(
            X,
            n_components=n_components,
            algorithm="irlb",
            seed=42,
            verbose=False,
        )
        return np.asarray(result["d"]).ravel()
    except Exception as exc:
        print(f"  [ref sigma] failed for {tier_label}/{storage_form}: {exc}", flush=True)
        return None
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Child-process dispatch
# ---------------------------------------------------------------------------

def _child_main(kwargs_json: str) -> None:
    kw = json.loads(kwargs_json)
    ref_sigma = kw.pop("ref_sigma", None)
    if ref_sigma is not None:
        ref_sigma = np.array(ref_sigma)
    run_one_in_process(ref_sigma=ref_sigma, **kw)


def _child_ref_main(kwargs_json: str) -> None:
    """Collect IRLB reference sigma in a subprocess and write it to a JSON file."""
    kw = json.loads(kwargs_json)
    out_path = kw.pop("out_path")
    sigma = collect_reference_sigma(**kw)
    payload = {"sigma": sigma.tolist() if sigma is not None else None}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def dispatch_child(
    tier_label: str,
    storage_form: str,
    algorithm: str,
    trial: int,
    n_components: int,
    ref_sigma: Optional[np.ndarray],
    output_jsonl: Path,
    timeout_s: float = 7200.0,
) -> str:
    import subprocess

    kw: Dict[str, Any] = {
        "tier_label": tier_label,
        "storage_form": storage_form,
        "algorithm": algorithm,
        "trial": trial,
        "n_components": n_components,
        "ref_sigma": ref_sigma.tolist() if ref_sigma is not None else None,
        "output_jsonl": str(output_jsonl),
    }
    kwargs_json = json.dumps(kw)
    tests_dir = str(Path(__file__).resolve().parent)
    cmd = [
        PYTHON_EXE, "-c",
        f"import sys; sys.path.insert(0,{repr(tests_dir)}); "
        f"from benchmark_svd_inmemory_defaults import _child_main; "
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


def dispatch_ref_child(
    tier_label: str,
    storage_form: str,
    n_components: int,
    scratch_dir: Path,
    timeout_s: float = 7200.0,
) -> Optional[np.ndarray]:
    """Compute reference IRLB sigma in a subprocess (avoids parent RSS bloat)."""
    import subprocess

    out_path = scratch_dir / f"_ref_sigma_{tier_label}_{storage_form}.json"
    kw = {
        "tier_label": tier_label,
        "storage_form": storage_form,
        "n_components": n_components,
        "out_path": str(out_path),
    }
    kwargs_json = json.dumps(kw)
    tests_dir = str(Path(__file__).resolve().parent)
    cmd = [
        PYTHON_EXE, "-c",
        f"import sys; sys.path.insert(0,{repr(tests_dir)}); "
        f"from benchmark_svd_inmemory_defaults import _child_ref_main; "
        f"_child_ref_main({repr(kwargs_json)})"
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
            return None
        time.sleep(0.3)
    if not out_path.exists():
        return None
    try:
        with open(out_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    finally:
        try:
            out_path.unlink()
        except Exception:
            pass
    sigma = payload.get("sigma")
    if sigma is None:
        return None
    return np.asarray(sigma, dtype=float)


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
    for col in ["wall_s", "peak_rss_mb", "sigma_corr",
                "reconstruction_err", "n_obs", "n_vars", "nnz"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    csv_path = output_dir / "svd_inmemory_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [report] CSV: {csv_path}", flush=True)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("  [report] No successful rows.", flush=True)
        return

    report_path = output_dir / "svd_inmemory_benchmark.md"
    lines = [
        "# In-Memory SVD Algorithm Benchmark: IRLB vs Halko",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- n_components: {int(ok['n_components'].iloc[0])}",
        f"- trials per config: {int(ok.groupby(['dataset','storage_form','algorithm'])['trial'].nunique().max())}",
        f"- reconstruction probe rows: {PROBE_ROWS}",
        f"- reference algorithm for sigma_corr: IRLB",
        "",
    ]

    # Summary table
    summary = (
        ok.groupby(["storage_form", "dataset", "n_obs", "algorithm"])[
            ["wall_s", "peak_rss_mb", "sigma_corr", "reconstruction_err"]
        ]
        .mean()
        .reset_index()
        .sort_values(["storage_form", "n_obs", "algorithm"])
    )
    lines += ["## Summary (mean across trials)", "", summary.to_markdown(index=False), ""]

    # Per storage-form ratio tables (wall time and RSS), each vs IRLB reference.
    for storage in STORAGE_FORMS:
        sub = ok[ok["storage_form"] == storage]
        if sub.empty:
            continue
        lines += [f"## {storage.title()} in-memory", ""]

        pivot_wall = sub.groupby(["dataset", "algorithm"])["wall_s"].mean().unstack("algorithm")
        if "irlb" in pivot_wall.columns:
            for alg in [c for c in pivot_wall.columns if c != "irlb"]:
                pivot_wall[f"{alg}_vs_irlb"] = pivot_wall[alg] / pivot_wall["irlb"]
        lines += ["### Wall time (s), ratios vs IRLB", "", pivot_wall.to_markdown(), ""]

        pivot_mem = sub.groupby(["dataset", "algorithm"])["peak_rss_mb"].mean().unstack("algorithm")
        if "irlb" in pivot_mem.columns:
            for alg in [c for c in pivot_mem.columns if c != "irlb"]:
                pivot_mem[f"{alg}_vs_irlb"] = pivot_mem[alg] / pivot_mem["irlb"]
        lines += ["### Peak RSS (MB), ratios vs IRLB", "", pivot_mem.to_markdown(), ""]

        acc = sub[sub["algorithm"] != "irlb"][
            ["dataset", "n_obs", "algorithm", "sigma_corr", "reconstruction_err"]
        ]
        if not acc.empty:
            lines += [
                "### Accuracy (vs IRLB reference)",
                "",
                "> `sigma_corr`: Pearson correlation of singular values vs IRLB.",
                "> `reconstruction_err`: relative Frobenius error on a random 500-row probe.",
                "",
            ]
            acc_agg = (
                acc.groupby(["dataset", "n_obs", "algorithm"])[
                    ["sigma_corr", "reconstruction_err"]
                ]
                .mean()
                .reset_index()
                .sort_values(["n_obs", "algorithm"])
            )
            lines.append(acc_agg.to_markdown(index=False))
            lines.append("")

    # Recommendations per storage form
    lines += ["## Recommendations", ""]
    current_defaults = {"sparse": "irlb", "dense": "halko"}
    acc_threshold = 0.9999

    for storage in STORAGE_FORMS:
        sub = ok[ok["storage_form"] == storage]
        if sub.empty:
            continue

        wall_medians: Dict[str, float] = {}
        corr_medians: Dict[str, float] = {}
        rss_medians: Dict[str, float] = {}
        for alg in sub["algorithm"].unique():
            wall_medians[alg] = float(sub[sub["algorithm"] == alg]["wall_s"].dropna().median())
            rss_medians[alg] = float(sub[sub["algorithm"] == alg]["peak_rss_mb"].dropna().median())
            if alg == "irlb":
                corr_medians[alg] = float("inf")
            else:
                corr_medians[alg] = float(sub[sub["algorithm"] == alg]["sigma_corr"].dropna().median())

        # Accuracy filter: keep algs whose median sigma_corr vs IRLB > 0.9999
        candidates = {
            alg: w for alg, w in wall_medians.items()
            if alg == "irlb" or corr_medians.get(alg, 0.0) > acc_threshold
        }

        cur = current_defaults[storage]

        if not candidates:
            rec = f"**{storage.title()}: inconclusive** -- no algorithm passed the accuracy filter."
        else:
            winner = min(candidates, key=candidates.get)
            winner_wall = candidates[winner]
            cur_wall = wall_medians.get(cur, float("nan"))
            winner_rss = rss_medians.get(winner, float("nan"))
            cur_rss = rss_medians.get(cur, float("nan"))

            if winner == cur:
                rec = (
                    f"**{storage.title()}: keep current default ({cur.upper()}).** "
                    f"Median wall={winner_wall:.3f}s beats all accuracy-qualified alternatives."
                )
            else:
                wall_gain = 1.0 - (winner_wall / cur_wall) if cur_wall > 0 else 0.0
                rss_ratio = (winner_rss / cur_rss) if cur_rss > 0 else float("nan")
                if wall_gain >= 0.10:
                    rec = (
                        f"**{storage.title()}: change default to {winner.upper()}.** "
                        f"Winner median wall={winner_wall:.3f}s vs current {cur.upper()} "
                        f"median wall={cur_wall:.3f}s ({wall_gain*100:.1f}% faster). "
                        f"Peak RSS ratio winner/current = {rss_ratio:.2f}. "
                        f"Winner sigma_corr vs IRLB median = "
                        f"{corr_medians[winner] if corr_medians[winner] != float('inf') else float('nan')}."
                    )
                else:
                    rec = (
                        f"**{storage.title()}: keep current default ({cur.upper()}).** "
                        f"Fastest accuracy-qualified alt was {winner.upper()} "
                        f"({winner_wall:.3f}s vs {cur_wall:.3f}s, only {wall_gain*100:.1f}% faster). "
                        f"Below the 10% wall-time threshold for switching."
                    )
        lines.append(rec)
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [report] Report: {report_path}", flush=True)

    # Terminal summary
    print("\n" + "=" * 70, flush=True)
    print("IN-MEMORY SVD ALGORITHM BENCHMARK SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(summary.to_string(index=False), flush=True)
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _should_skip_dense(tier_label: str, skip_above: Optional[str]) -> bool:
    if skip_above is None:
        return False
    return _tier_int(tier_label) > _tier_int(skip_above)


def run_benchmark(
    tiers: List[str],
    n_components: int,
    trials: int,
    skip_dense_above: Optional[str],
    output_dir: Path,
    resume: bool,
    algorithms: Optional[List[str]] = None,
) -> None:
    if algorithms is None:
        algorithms = list(ALGORITHMS)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "raw_results.jsonl"
    scratch_dir = output_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

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
                        key = (row["dataset"], row["storage_form"], row["algorithm"], int(row["trial"]))
                        completed.add(key)
                except Exception:
                    pass
        print(f"  [resume] {len(completed)} completed cases found.", flush=True)

    for tier_label in tiers:
        src_path = _dataset_path(tier_label)
        if not src_path.exists():
            print(f"  [skip] dataset not found: {src_path}", flush=True)
            continue

        print(f"\n{'=' * 70}", flush=True)
        print(f"  TIER: {tier_label}", flush=True)
        print(f"{'=' * 70}", flush=True)

        for storage_form in STORAGE_FORMS:
            if storage_form == "dense" and _should_skip_dense(tier_label, skip_dense_above):
                print(
                    f"  [skip] dense/{tier_label} exceeds --skip-dense-above={skip_dense_above}",
                    flush=True,
                )
                continue

            print(f"\n  storage_form: {storage_form}", flush=True)
            print(f"  Collecting IRLB reference sigma for {tier_label}/{storage_form} ...", flush=True)
            ref_sigma = dispatch_ref_child(
                tier_label=tier_label,
                storage_form=storage_form,
                n_components=n_components,
                scratch_dir=scratch_dir,
            )
            if ref_sigma is None:
                print(
                    f"  WARNING: could not collect IRLB reference sigma for {tier_label}/{storage_form}.",
                    flush=True,
                )

            for algorithm in algorithms:
                for trial in range(1, trials + 1):
                    key = (tier_label, storage_form, algorithm, trial)
                    if resume and key in completed:
                        print(
                            f"  [skip] {tier_label} {storage_form} {algorithm} t{trial} (already complete)",
                            flush=True,
                        )
                        continue

                    print(
                        f"\n  Running: {tier_label} {storage_form} {algorithm} trial={trial}",
                        flush=True,
                    )
                    # IRLB is its own reference (sigma_corr = NaN); others compare against it.
                    ref = ref_sigma if algorithm != "irlb" else None
                    status = dispatch_child(
                        tier_label=tier_label,
                        storage_form=storage_form,
                        algorithm=algorithm,
                        trial=trial,
                        n_components=n_components,
                        ref_sigma=ref,
                        output_jsonl=jsonl_path,
                    )
                    if status != "ok":
                        print(
                            f"  WARN: {tier_label} {storage_form} {algorithm} t{trial} returned status={status}",
                            flush=True,
                        )

    print("\nGenerating report ...", flush=True)
    generate_report(output_dir, jsonl_path)
    print(f"\nDone. Results in: {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark in-memory SVD: IRLB vs Halko (sparse and dense)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tiers", nargs="+",
        default=DEFAULT_TIERS,
        help=(
            "Tier labels to benchmark (e.g. 25k 50k 100k 150k 200k). "
            "Corresponding scale_subset_<tier>.h5ad must exist in "
            f"{BENCHMARK_DATA_DIR}"
        ),
    )
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--skip-dense-above",
        default=DEFAULT_DENSE_TIER_CAP,
        help=(
            "Skip dense-in-memory cases for tiers strictly larger than this "
            "(default: 100k). Set to a large value like '10M' or empty '' to "
            "disable the guard."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: tests/benchmark_results/svd_inmem_<timestamp>)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cases already completed in an existing output dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = time.strftime("svd_inmem_%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "tests" / "benchmark_results" / run_id

    skip_dense_above = args.skip_dense_above.strip() if args.skip_dense_above else None
    if skip_dense_above == "":
        skip_dense_above = None

    algorithms = list(ALGORITHMS)

    banner_algs = " vs ".join(a.upper() for a in algorithms)
    print(f"In-Memory SVD Algorithm Benchmark: {banner_algs}", flush=True)
    print(f"Tiers            : {args.tiers}", flush=True)
    print(f"Components       : {args.n_components}", flush=True)
    print(f"Trials           : {args.trials}", flush=True)
    print(f"Algorithms       : {algorithms}", flush=True)
    print(f"Skip dense above : {skip_dense_above}", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print(flush=True)

    run_benchmark(
        tiers=args.tiers,
        n_components=args.n_components,
        trials=args.trials,
        skip_dense_above=skip_dense_above,
        output_dir=output_dir,
        resume=args.resume,
        algorithms=algorithms,
    )


if __name__ == "__main__":
    main()

