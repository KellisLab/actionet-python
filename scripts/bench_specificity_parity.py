"""
Parity and performance benchmark: compute_feature_specificity()
  in-memory (C++ ABI) vs backed (C++ ABI via backed-operator).

Usage:
    python scripts/bench_specificity_parity.py
"""

import time
import gc
import traceback

import numpy as np
import anndata as ad
import scipy.sparse as sp

import actionet

H5AD_PATH = "data/test_adata.h5ad"
LABEL_KEY  = "CellLabel"
N_REPEATS  = 3          # timed repeats per mode
CHUNK_SIZE = 4096
N_THREADS  = 0          # 0 = auto

BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_in_memory() -> ad.AnnData:
    """Load the full dataset into RAM (no backing)."""
    return ad.read_h5ad(H5AD_PATH)


def _load_backed() -> ad.AnnData:
    """Open the dataset in backed/read mode."""
    return ad.read_h5ad(H5AD_PATH, backed="r")


def _run(adata: ad.AnnData) -> dict:
    return actionet.compute_feature_specificity(
        adata,
        labels=LABEL_KEY,
        n_threads=N_THREADS,
        backed_chunk_size=CHUNK_SIZE,
        return_raw=True,
    )


def _time_runs(adata: ad.AnnData, label: str) -> tuple[dict, float, float]:
    """Run _run() N_REPEATS times; return last result + (mean, std) elapsed."""
    result = None
    times = []
    for _ in range(N_REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        result = _run(adata)
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    print(f"  {label}: runs={times} → mean={arr.mean():.3f}s  std={arr.std():.3f}s")
    return result, float(arr.mean()), float(arr.std())


def _maxabsdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a.astype(np.float64)), 1e-12)
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)) / denom))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    finite = np.isfinite(a_flat) & np.isfinite(b_flat)
    if finite.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a_flat[finite], b_flat[finite])[0, 1])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}=== compute_feature_specificity — parity & performance benchmark ==={RESET}")
    print(f"  h5ad : {H5AD_PATH}")
    print(f"  label: {LABEL_KEY!r}  |  repeats={N_REPEATS}  |  chunk={CHUNK_SIZE}  |  threads={N_THREADS}\n")

    # ── 1. Dataset info ───────────────────────────────────────────────────────
    print(f"{BOLD}[1] Dataset info{RESET}")
    adata_back = _load_backed()
    n_obs, n_vars = adata_back.shape
    n_labels = int(adata_back.obs[LABEL_KEY].nunique())
    x_type   = type(adata_back.X).__name__
    print(f"  shape  : {n_obs} obs × {n_vars} vars")
    print(f"  n_labels: {n_labels}")
    print(f"  X type (backed): {x_type}")
    print(f"  is_backed: {adata_back.isbacked}\n")

    # Confirm backed X is sparse-backed (dispatches to C++ ABI)
    from actionet._matrix_source import MatrixSource
    src_back = MatrixSource(adata_back, layer=None)
    print(f"  MatrixSource.is_backed={src_back.is_backed}  is_sparse={src_back.is_sparse}")
    print(f"  → backed dispatch path: {'C++ ABI' if src_back.is_backed and src_back.is_sparse else 'Python streamed'}\n")

    # ── 2. In-memory run ──────────────────────────────────────────────────────
    print(f"{BOLD}[2] In-memory (C++ ABI){RESET}")
    adata_mem = _load_in_memory()
    src_mem = MatrixSource(adata_mem, layer=None)
    print(f"  X type (in-memory): {type(adata_mem.X).__name__}  is_sparse={sp.issparse(adata_mem.X)}")
    print(f"  MatrixSource.is_backed={src_mem.is_backed}")
    result_mem, mean_mem, std_mem = _time_runs(adata_mem, "in-memory")

    # ── 3. Backed run ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}[3] Backed / C++ ABI (compute_feature_specificity_backed_operator){RESET}")
    result_bck, mean_bck, std_bck = _time_runs(adata_back, "backed  ")

    # ── 4. Parity analysis ────────────────────────────────────────────────────
    print(f"\n{BOLD}[4] Parity analysis{RESET}")
    KEYS = ["average_profile", "upper_significance", "lower_significance"]
    all_pass = True
    ATOL = 1e-6   # absolute tolerance for numerical parity

    for key in KEYS:
        a = result_mem[key]
        b = result_bck[key]
        shape_ok = a.shape == b.shape
        mad      = _maxabsdiff(a, b) if shape_ok else float("inf")
        rel      = _rel_err(a, b)   if shape_ok else float("inf")
        cor      = _corr(a, b)      if shape_ok else float("nan")
        passed   = shape_ok and mad < ATOL

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        all_pass = all_pass and passed

        print(f"  {key}:")
        print(f"    shape   mem={a.shape}  backed={b.shape}  {'✓' if shape_ok else '✗'}")
        print(f"    max|Δ|  = {mad:.3e}  (tol={ATOL:.0e})  [{status}]")
        print(f"    max|Δ|/|a| = {rel:.3e}")
        print(f"    Pearson r  = {cor:.8f}")

    # ── 5. Performance summary ────────────────────────────────────────────────
    print(f"\n{BOLD}[5] Performance summary{RESET}")
    speedup = mean_mem / mean_bck if mean_bck > 0 else float("inf")
    faster_mode = "backed" if mean_bck < mean_mem else "in-memory"
    print(f"  in-memory : {mean_mem:.3f}s ± {std_mem:.3f}s")
    print(f"  backed    : {mean_bck:.3f}s ± {std_bck:.3f}s")
    print(f"  speedup   : {speedup:.2f}×  ({faster_mode} is faster)")

    # ── 6. Overall verdict ────────────────────────────────────────────────────
    print(f"\n{BOLD}[6] Overall verdict{RESET}")
    if all_pass:
        print(f"  {GREEN}{BOLD}PARITY: PASS{RESET} — all outputs agree within tolerance {ATOL:.0e}")
    else:
        print(f"  {RED}{BOLD}PARITY: FAIL{RESET} — numerical discrepancies exceed tolerance")

    print()


if __name__ == "__main__":
    main()
