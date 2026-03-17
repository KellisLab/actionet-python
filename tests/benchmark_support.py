"""benchmark_support.py — shared helpers for the ACTIONet Scaling Benchmark Suite.

Responsibilities:
- Dataset manifests and stratified subset generation
- Per-stage metrics collection (psutil sampling thread)
- Child-process case dispatch with timeout and RSS kill
- Scaling-model fitting (power-law, batch-augmented)
- Report and plot generation

See: tests/ACTIONet Scaling Benchmark Suite.md
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scipy.sparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
BENCHMARK_DATA_DIR = Path("/data/actionet_benchmark")
BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Results go inside the workspace (per spec: tests/benchmark_results/<run_id>/)
BENCHMARK_RESULTS_DIR = REPO_ROOT / "tests" / "benchmark_results"
BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = str(REPO_ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON_EXE).exists():
    PYTHON_EXE = sys.executable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERS = [25_000, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000]
TIER_LABELS = {t: f"{t // 1000}k" if t < 1_000_000 else f"{t // 1_000_000}M" for t in TIERS}
TIER_LABELS[300_000] = "300k"

SUBSET_SEED = 42
MIN_CELLS_PER_SAMPLE = 100
MIN_LABEL_CELLS = 50
MIN_LABELS_FOR_ANNOTATION = 2

N_THREADS = 44
# Halko (randomised power iteration) is used throughout the benchmark instead of
# PRIMME. For scale evaluation the critical property is that matvec count is fixed
# at 2*(iters+1) passes regardless of matrix conditioning, giving a clean
# NNZ-proportional I/O cost model. PRIMME's adaptive convergence introduces
# iteration-count variance that contaminates the backed-vs-in-memory signal and
# makes the scaling curve non-monotone. Use "primme" only if the goal is to
# benchmark production-workflow wall time rather than storage-layer overhead.
SVD_ALGORITHM = "halko"
N_COMPONENTS = 30

# Stopping rules
TIMEOUT_SECONDS = 12 * 3600
RSS_KILL_GB = 150
STAGE_RSS_LIMIT_GB = 150
STAGE_TIME_LIMIT_SECONDS = 4 * 3600

# Trials per tier
def trials_for_tier(n_obs: int) -> int:
    if n_obs <= 50_000:
        return 3
    elif n_obs <= 200_000:
        return 2
    else:
        return 1


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------

DATASET_MANIFEST: Dict[str, Dict[str, Any]] = {
    "small_full": {
        "file": "test_adata.h5ad",
        "data_dir": DATA_DIR,
        "batch_key": None,
        "label_key": "CellLabel",
        "features_key": "Gene",
        "batch_correction": False,
        "tier": None,
    },
    "sparse_medium": {
        "file": "adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad",
        "data_dir": DATA_DIR,
        "batch_key": "UID",
        "label_key": "CellType",
        "features_key": "Gene",
        "batch_correction": True,
        "tier": None,
    },
    "scale_full": {
        "file": "adata_agg_Scn4b_OX_fil.h5ad",
        "data_dir": DATA_DIR,
        "batch_key": "UID",
        "label_key": "CellType",
        "features_key": "Gene",
        "batch_correction": True,
        "tier": None,
    },
}

# Subset handles (populated lazily)
for _t in TIERS:
    _lbl = TIER_LABELS[_t]
    DATASET_MANIFEST[f"scale_subset_{_lbl}"] = {
        "file": f"scale_subset_{_lbl}.h5ad",
        "data_dir": BENCHMARK_DATA_DIR,
        "batch_key": "UID",
        "label_key": "CellType",
        "features_key": "Gene",
        "batch_correction": True,
        "tier": _t,
    }


def dataset_path(handle: str) -> Path:
    cfg = DATASET_MANIFEST[handle]
    return Path(cfg["data_dir"]) / cfg["file"]


# ---------------------------------------------------------------------------
# Sparse-medium one-time conversion
# ---------------------------------------------------------------------------

def ensure_sparse_medium() -> None:
    """Convert sparse_medium to CSR if it has a dense backing store."""
    path = dataset_path("sparse_medium")
    if not path.exists():
        print(f"  [sparse_medium] file not found: {path}", flush=True)
        return
    adata = ad.read_h5ad(str(path), backed="r")
    is_sparse = scipy.sparse.issparse(adata.X)
    if hasattr(adata, "file") and adata.file is not None:
        try:
            adata.file.close()
        except Exception:
            pass
    del adata
    gc.collect()

    if is_sparse:
        print("  [sparse_medium] already sparse — no conversion needed.", flush=True)
        return

    print("  [sparse_medium] dense backing detected — converting to CSR in-place ...", flush=True)
    adata = ad.read_h5ad(str(path))
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.write_h5ad(str(path))
    del adata
    gc.collect()
    print("  [sparse_medium] conversion complete.", flush=True)


# ---------------------------------------------------------------------------
# Stratified subset generation
# ---------------------------------------------------------------------------

def _make_subset(handle: str, overwrite: bool = False) -> Optional[Path]:
    """Generate a stratified subset for a scale_subset_* handle.

    Returns the path to the h5ad file on success, None if the tier is
    infeasible (all samples below min_cells threshold).
    """
    cfg = DATASET_MANIFEST[handle]
    out_path = Path(cfg["data_dir"]) / cfg["file"]

    if out_path.exists() and not overwrite:
        return out_path

    tier = cfg["tier"]
    assert tier is not None, f"make_subset called on non-subset handle: {handle}"

    src_path = dataset_path("scale_full")
    if not src_path.exists():
        print(f"  [subset] scale_full not found: {src_path}", flush=True)
        return None

    print(f"  [subset] Generating {handle} (tier={tier:,}) ...", flush=True)
    adata = ad.read_h5ad(str(src_path), backed="r")
    batch_key = cfg["batch_key"]
    obs = adata.obs.copy()
    if hasattr(adata, "file") and adata.file is not None:
        try:
            adata.file.close()
        except Exception:
            pass
    del adata
    gc.collect()

    # Count cells per sample
    counts = obs[batch_key].value_counts()
    eligible = counts[counts >= MIN_CELLS_PER_SAMPLE].index.tolist()
    if not eligible:
        print(f"  [subset] No eligible samples for tier {tier:,}.", flush=True)
        return None

    eligible_obs = obs[obs[batch_key].isin(eligible)].copy()
    total_eligible = len(eligible_obs)
    if total_eligible < tier:
        print(
            f"  [subset] Only {total_eligible:,} eligible cells for tier {tier:,} — using all.",
            flush=True,
        )
        selected_idx = eligible_obs.index.tolist()
    else:
        rng = np.random.default_rng(SUBSET_SEED)
        # Proportional stratified sampling
        sample_sizes = {}
        for uid in eligible:
            n_uid = int(counts[uid])
            sample_sizes[uid] = max(0, int(round(n_uid / total_eligible * tier)))

        # Adjust to hit exactly `tier` cells
        total_sampled = sum(sample_sizes.values())
        diff = tier - total_sampled
        if diff != 0:
            sorted_uids = sorted(eligible, key=lambda u: -counts[u])
            for i in range(abs(diff)):
                uid = sorted_uids[i % len(sorted_uids)]
                sample_sizes[uid] += int(math.copysign(1, diff))
                sample_sizes[uid] = max(0, min(int(counts[uid]), sample_sizes[uid]))

        selected_idx = []
        for uid in eligible:
            n_take = sample_sizes.get(uid, 0)
            if n_take <= 0:
                continue
            uid_idx = eligible_obs.index[eligible_obs[batch_key] == uid].tolist()
            chosen = rng.choice(uid_idx, size=min(n_take, len(uid_idx)), replace=False).tolist()
            selected_idx.extend(chosen)

    # Load full adata and subset
    adata_full = ad.read_h5ad(str(src_path))
    adata_sub = adata_full[selected_idx].copy()
    del adata_full
    gc.collect()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_sub.write_h5ad(str(out_path))
    del adata_sub
    gc.collect()
    print(f"  [subset] Written: {out_path} ({len(selected_idx):,} cells)", flush=True)
    return out_path


def generate_all_subsets(tiers: Optional[List[int]] = None, overwrite: bool = False) -> Dict[str, Optional[Path]]:
    """Generate all scale_subset_* h5ad files. Returns {handle: path_or_None}."""
    if tiers is None:
        tiers = TIERS
    results = {}
    for t in tiers:
        lbl = TIER_LABELS[t]
        handle = f"scale_subset_{lbl}"
        results[handle] = _make_subset(handle, overwrite=overwrite)
    return results


# ---------------------------------------------------------------------------
# Profiler: per-stage wall time + peak RSS delta
# ---------------------------------------------------------------------------

class StageProfiler:
    """Context manager: records wall time and peak RSS increase for one stage."""

    def __init__(self, label: str = "", sample_interval: float = 0.05):
        self.label = label
        self.sample_interval = sample_interval
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
            time.sleep(self.sample_interval)

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
        return False  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Result row dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResultRow:
    case_id: str
    dataset: str
    tier: Optional[int]
    profile: str
    mode: str
    stage: str
    params: Dict[str, Any]
    n_obs: int
    n_vars: int
    nnz: int
    n_batches: int
    representation_dim: int
    execution_kind: str
    wall_s: float
    peak_rss_mb: float
    io_read_mb: float
    io_write_mb: float
    status: str
    failure_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _io_counters_mb() -> tuple[float, float]:
    """Return (read_mb, write_mb) for current process since process start."""
    try:
        c = psutil.Process().io_counters()
        return c.read_bytes / 1e6, c.write_bytes / 1e6
    except Exception:
        return 0.0, 0.0


def _get_n_batches(adata: ad.AnnData, batch_key: Optional[str]) -> int:
    if batch_key and batch_key in adata.obs:
        return int(pd.Series(adata.obs[batch_key]).nunique())
    return 0


def _get_nnz(adata: ad.AnnData) -> int:
    try:
        if scipy.sparse.issparse(adata.X):
            return int(adata.X.nnz)
        if hasattr(adata.X, "shape"):
            return int(np.prod(adata.X.shape))
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------

def append_result(jsonl_path: Path, row: ResultRow) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(row.to_json() + "\n")


def load_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def is_case_complete(jsonl_path: Path, case_id: str) -> bool:
    """Return True if case_id has a status=ok total row in the JSONL."""
    for row in load_results(jsonl_path):
        if row.get("case_id") == case_id and row.get("stage") == "total" and row.get("status") == "ok":
            return True
    return False


# ---------------------------------------------------------------------------
# Case ID generation
# ---------------------------------------------------------------------------

def make_case_id(dataset: str, profile: str, mode: str, trial: int, sweep: str = "") -> str:
    parts = [dataset, profile, mode, f"t{trial}"]
    if sweep:
        parts.append(sweep)
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Scaling model fitting
# ---------------------------------------------------------------------------

def fit_power_law(ns: np.ndarray, ys: np.ndarray) -> Optional[Dict[str, float]]:
    """Fit T(N) = a * N^b via log-linear regression. Returns {a, b, r2}."""
    mask = (ns > 0) & (ys > 0) & np.isfinite(ns) & np.isfinite(ys)
    if mask.sum() < 2:
        return None
    log_n = np.log(ns[mask])
    log_y = np.log(ys[mask])
    coeffs = np.polyfit(log_n, log_y, 1)
    b, log_a = coeffs
    a = np.exp(log_a)
    y_pred = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": float(a), "b": float(b), "r2": float(r2)}


def extrapolate_power_law(model: Dict[str, float], ns: List[int]) -> Dict[int, float]:
    """Predict T(N) for a list of N values using a fitted power-law model."""
    return {n: model["a"] * (n ** model["b"]) for n in ns}


EXTRAPOLATE_AT = [500_000, 1_000_000, 5_000_000, 10_000_000]


def fit_batch_augmented(
    ns: np.ndarray,
    bs: np.ndarray,
    ys: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Fit T(N, B) = a * N^b * B^c via multi-variate log-linear regression."""
    mask = (ns > 0) & (bs > 0) & (ys > 0) & np.isfinite(ns) & np.isfinite(bs) & np.isfinite(ys)
    if mask.sum() < 3:
        return None
    log_n = np.log(ns[mask])
    log_b = np.log(bs[mask])
    log_y = np.log(ys[mask])
    X = np.column_stack([log_n, log_b, np.ones_like(log_n)])
    result = np.linalg.lstsq(X, log_y, rcond=None)
    coeffs = result[0]
    b_exp, c_exp, log_a = coeffs
    a = np.exp(log_a)
    return {"a": float(a), "b": float(b_exp), "c": float(c_exp)}


# ---------------------------------------------------------------------------
# Summary CSV builder
# ---------------------------------------------------------------------------

def build_summary_csv(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        rows.extend(load_results(jsonl_file))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Report & plot generation
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_report(run_dir: Path) -> None:
    """Generate report.md and plots/ from raw JSONL files under run_dir/raw/."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as _plt

    raw_dir = run_dir / "raw"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    summary_path = run_dir / "summary.csv"
    report_path = run_dir / "report.md"

    df = build_summary_csv(raw_dir, summary_path)
    if df.empty:
        with open(report_path, "w") as fh:
            fh.write("# ACTIONet Scaling Benchmark Report\n\nNo results yet.\n")
        return

    # Normalise types
    for col in ["wall_s", "peak_rss_mb", "n_obs", "n_vars", "nnz", "tier"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    lines = []
    lines.append("# ACTIONet Scaling Benchmark Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- Full-workflow scaling tables per profile/mode ---
    wf_df = df[(df["stage"] != "total") & (df["status"] == "ok") & df["tier"].notna()].copy()
    if not wf_df.empty:
        lines.append("## Full-Workflow Scaling (per stage)")
        for profile in sorted(wf_df["profile"].dropna().unique()):
            p_df = wf_df[wf_df["profile"] == profile]
            lines.append(f"\n### Profile: {profile}")
            pivot = (
                p_df.groupby(["stage", "tier"])["wall_s"]
                .mean()
                .unstack("tier")
            )
            lines.append("\n**Wall time (s) — mean across trials**\n")
            lines.append(pivot.to_markdown())

            pivot_mem = (
                p_df.groupby(["stage", "tier"])["peak_rss_mb"]
                .mean()
                .unstack("tier")
            )
            lines.append("\n**Peak RSS (MB) — mean across trials**\n")
            lines.append(pivot_mem.to_markdown())

            # Plot
            stages_of_interest = [s for s in pivot.index if s not in ("total",)]
            if len(stages_of_interest) > 0 and len(pivot.columns) > 1:
                fig, axes = _plt.subplots(1, 2, figsize=(16, 6))
                for stage in stages_of_interest:
                    tiers = sorted([c for c in pivot.columns if pd.notna(c)])
                    ys = [pivot.loc[stage, t] if t in pivot.columns else np.nan for t in tiers]
                    axes[0].plot(tiers, ys, marker="o", label=stage)
                axes[0].set_xlabel("Cell count (N)")
                axes[0].set_ylabel("Wall time (s)")
                axes[0].set_title(f"Runtime scaling — {profile}")
                axes[0].legend(fontsize=6, ncol=2)
                axes[0].set_xscale("log")
                axes[0].set_yscale("log")

                for stage in stages_of_interest:
                    tiers = sorted([c for c in pivot_mem.columns if pd.notna(c)])
                    ys = [pivot_mem.loc[stage, t] if t in pivot_mem.columns and stage in pivot_mem.index else np.nan for t in tiers]
                    axes[1].plot(tiers, ys, marker="o", label=stage)
                axes[1].set_xlabel("Cell count (N)")
                axes[1].set_ylabel("Peak RSS (MB)")
                axes[1].set_title(f"Memory scaling — {profile}")
                axes[1].legend(fontsize=6, ncol=2)
                axes[1].set_xscale("log")
                axes[1].set_yscale("log")

                _plt.tight_layout()
                fig_name = f"scaling_{profile}.png"
                fig.savefig(str(plots_dir / fig_name), dpi=150)
                _plt.close(fig)
                lines.append(f"\n![Scaling {profile}](plots/{fig_name})\n")

    # --- Power-law fits ---
    lines.append("\n## Power-Law Scaling Models\n")
    stages_to_fit = [
        "filter", "normalize", "reduce_kernel", "batch_correction",
        "action_decomposition", "network_construction", "archetype_diffusion",
        "layout_2d", "feature_specificity", "marker_detection",
        "annotation", "imputation",
    ]
    fit_rows = []
    for stage in stages_to_fit:
        s_df = df[(df["stage"] == stage) & (df["status"] == "ok") & df["tier"].notna() & df["wall_s"].notna()]
        if s_df.empty:
            continue
        agg = s_df.groupby("tier")["wall_s"].mean().reset_index()
        model = fit_power_law(agg["tier"].values, agg["wall_s"].values)
        if model:
            preds = extrapolate_power_law(model, EXTRAPOLATE_AT)
            fit_rows.append({
                "stage": stage,
                "a": f"{model['a']:.3e}",
                "b (exponent)": f"{model['b']:.3f}",
                "R²": f"{model['r2']:.4f}",
                **{f"pred_{n//1000}k_s": f"{preds[n]:.1f}" for n in EXTRAPOLATE_AT},
            })
    if fit_rows:
        lines.append(pd.DataFrame(fit_rows).to_markdown(index=False))
        lines.append("")

    # --- Unoptimized stage section ---
    legacy_stages = ["marker_detection", "annotation", "imputation"]
    leg_df = df[df["stage"].isin(legacy_stages) & (df["status"] == "ok")].copy()
    if not leg_df.empty:
        lines.append("\n## Legacy-Unoptimized Stages\n")
        lines.append(
            "> `marker_detection`, `annotation`, and `imputation` have had no "
            "algorithmic optimizations since the original sub-million-cell "
            "implementation. Python-level backed streaming paths exist but the "
            "C++ core complexity is unchanged.\n"
        )
        pivot_leg = leg_df.groupby(["stage", "tier"])["wall_s"].mean().unstack("tier")
        lines.append(pivot_leg.to_markdown())
        lines.append("")

    # --- network_construction frontier curves ---
    net_df = df[(df["stage"] == "network_construction") & (df["status"] == "ok")].copy()
    if not net_df.empty and "tier" in net_df.columns and net_df["tier"].notna().any():
        lines.append("\n## Network Construction Frontier\n")
        for profile in sorted(net_df["profile"].dropna().unique()):
            sub = net_df[(net_df["profile"] == profile) & net_df["tier"].notna()]
            agg = sub.groupby("tier")["wall_s"].mean().reset_index()
            lines.append(f"\n### {profile}\n")
            lines.append(agg.to_markdown(index=False))
        lines.append("")

    # --- Focused sweep sections (thread sweep, ef sweep, etc.) ---
    sweep_df = df[df["profile"].isin(["thread_sweep", "ef_sweep", "chunk_sweep", "batch_sweep", "kmax_sweep"])].copy()
    if not sweep_df.empty:
        lines.append("\n## Focused Sweeps\n")
        for sweep_profile in sorted(sweep_df["profile"].unique()):
            sub = sweep_df[sweep_df["profile"] == sweep_profile]
            lines.append(f"\n### {sweep_profile}\n")
            lines.append(sub[["dataset", "stage", "params", "n_obs", "wall_s", "peak_rss_mb", "status"]].to_markdown(index=False))
            lines.append("")

    # Write report
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [report] Written: {report_path}", flush=True)
    print(f"  [report] Summary CSV: {summary_path}", flush=True)
