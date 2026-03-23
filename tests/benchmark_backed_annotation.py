"""Focused benchmark: annotate_cells + impute_features across storage modes.

Usage:
    python tests/benchmark_backed_annotation.py [--n-cells N] [--n-genes N]

Generates a synthetic dataset in three backed modes (CSR, CSC, dense) plus
in-memory, and times annotate_cells (vision + actionet) and impute_features.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_synthetic(n_cells: int, n_genes: int, seed: int = 42) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_types = 5
    labels = np.array([f"CT_{i % n_types}" for i in range(n_cells)])

    X = rng.poisson(0.3, size=(n_cells, n_genes)).astype(np.float64)
    n_markers = min(20, n_genes // n_types)
    for ct in range(n_types):
        rows = np.where(labels == f"CT_{ct}")[0]
        col_start = ct * n_markers
        col_end = min(col_start + n_markers, n_genes)
        if col_start < n_genes:
            X[np.ix_(rows, np.arange(col_start, col_end))] += rng.poisson(4.0, size=(len(rows), col_end - col_start))

    var_names = np.array([f"G{i}" for i in range(n_genes)])
    obs = pd.DataFrame({"CellLabel": labels})
    var = pd.DataFrame({"Gene": var_names})
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.var_names = var_names
    adata.obs_names = np.array([f"cell_{i}" for i in range(n_cells)])
    return adata


def _make_markers(n_genes: int, n_types: int = 5) -> dict:
    n_markers = min(20, n_genes // n_types)
    markers = {}
    for ct in range(n_types):
        start = ct * n_markers
        end = min(start + n_markers, n_genes)
        markers[f"CT_{ct}"] = [f"G{i}" for i in range(start, end)]
    return markers


def _prep_adata(adata):
    """Run reduce + build_network in-memory."""
    import actionet
    actionet.reduce(adata, n_comps=min(20, adata.n_obs - 1), verbose=False)
    actionet.build_network(adata, verbose=False)


def _write_backed(adata, tmp_dir, fmt):
    path = tmp_dir / f"bench_{fmt}.h5ad"
    if fmt == "dense":
        adata_copy = adata.copy()
        adata_copy.X = adata_copy.X.toarray() if sp.issparse(adata_copy.X) else adata_copy.X
        adata_copy.write_h5ad(path)
    elif fmt == "csc":
        adata_copy = adata.copy()
        adata_copy.X = sp.csc_matrix(adata_copy.X) if not sp.isspmatrix_csc(adata_copy.X) else adata_copy.X
        adata_copy.write_h5ad(path)
    else:
        adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r+")


def _time_fn(fn, label: str, repeats: int = 1) -> float:
    gc.collect()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    elapsed = min(times)
    print(f"  {label}: {elapsed:.3f}s (best of {repeats})")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=2000)
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    import actionet

    tmp_dir = Path("/tmp/actionet_bench")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic data: {args.n_cells} cells x {args.n_genes} genes")
    adata = _make_synthetic(args.n_cells, args.n_genes)
    _prep_adata(adata)

    markers = _make_markers(args.n_genes)
    features = [f"G{i}" for i in range(20)]

    results = []

    # In-memory
    print("\n[in-memory]")
    results.append(("in-memory", "annotate_vision",
                     _time_fn(lambda: actionet.annotate_cells(adata, markers, method="vision"), "annotate_cells(vision)", args.repeats)))
    results.append(("in-memory", "annotate_actionet",
                     _time_fn(lambda: actionet.annotate_cells(adata, markers, method="actionet"), "annotate_cells(actionet)", args.repeats)))
    results.append(("in-memory", "impute_features",
                     _time_fn(lambda: actionet.impute_features(adata, features), "impute_features", args.repeats)))

    # Backed modes
    for fmt in ["csr", "csc", "dense"]:
        print(f"\n[backed-{fmt}]")
        adata_b = _write_backed(adata, tmp_dir, fmt)

        results.append((f"backed-{fmt}", "annotate_vision",
                         _time_fn(lambda: actionet.annotate_cells(adata_b, markers, method="vision"), "annotate_cells(vision)", args.repeats)))
        results.append((f"backed-{fmt}", "annotate_actionet",
                         _time_fn(lambda: actionet.annotate_cells(adata_b, markers, method="actionet"), "annotate_cells(actionet)", args.repeats)))
        results.append((f"backed-{fmt}", "impute_features",
                         _time_fn(lambda: actionet.impute_features(adata_b, features), "impute_features", args.repeats)))

    # Summary table
    print("\n--- Summary ---")
    df = pd.DataFrame(results, columns=["mode", "function", "time_s"])
    pivot = df.pivot(index="function", columns="mode", values="time_s")
    print(pivot.to_string(float_format="%.3f"))

    # Save
    out_path = Path(__file__).resolve().parent / "benchmark_results" / "backed_annotation_bench.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
