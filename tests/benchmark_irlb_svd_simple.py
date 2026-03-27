#!/usr/bin/env python3
"""benchmark_irlb_svd_simple.py - Simple benchmark for all IRLB SVD variants.

Benchmarks all SVD implementations (sparse, dense, in-memory, disk-backed)
across all supported algorithms (IRLB, Halko, Feng, PRIMME).

Usage:
    python tests/benchmark_irlb_svd_simple.py [--size small|medium|large] [--output results.csv]
"""

import argparse
import gc
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import actionet as an


# Dataset size presets
SIZE_PRESETS = {
    "small": {"n_obs": 500, "n_vars": 200, "n_components": 15, "density": 0.3},
    "medium": {"n_obs": 2000, "n_vars": 500, "n_components": 30, "density": 0.2},
    "large": {"n_obs": 5000, "n_vars": 1000, "n_components": 50, "density": 0.1},
}


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    matrix_type: str      # sparse/dense
    storage_mode: str     # inmemory/backed
    algorithm: str        # irlb/halko/feng/primme
    n_obs: int
    n_vars: int
    n_components: int
    density: float
    wall_time_s: float
    top_5_sigma: List[float]

    def to_dict(self) -> Dict:
        return {
            "matrix_type": self.matrix_type,
            "storage_mode": self.storage_mode,
            "algorithm": self.algorithm,
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "n_components": self.n_components,
            "density": self.density,
            "wall_time_s": self.wall_time_s,
            "sigma_1": self.top_5_sigma[0] if len(self.top_5_sigma) > 0 else None,
            "sigma_2": self.top_5_sigma[1] if len(self.top_5_sigma) > 1 else None,
            "sigma_3": self.top_5_sigma[2] if len(self.top_5_sigma) > 2 else None,
            "sigma_4": self.top_5_sigma[3] if len(self.top_5_sigma) > 3 else None,
            "sigma_5": self.top_5_sigma[4] if len(self.top_5_sigma) > 4 else None,
        }


def create_test_matrix(
    n_obs: int,
    n_vars: int,
    density: float,
    as_sparse: bool,
    random_state: int = 42,
) -> np.ndarray:
    """Create synthetic test matrix."""
    rng = np.random.RandomState(random_state)

    if as_sparse:
        X = sp.random(n_obs, n_vars, density=density, random_state=random_state, format='csr')
        X.data = X.data * 100
        return X
    else:
        rank = min(n_obs, n_vars) // 4
        U = rng.randn(n_obs, rank)
        V = rng.randn(n_vars, rank)
        sigma = np.linspace(100, 10, rank)
        X = U @ np.diag(sigma) @ V.T
        return X


def benchmark_inmemory(
    X: np.ndarray,
    algorithm: str,
    n_components: int,
) -> BenchmarkResult:
    """Benchmark in-memory SVD."""
    matrix_type = "sparse" if sp.issparse(X) else "dense"
    n_obs, n_vars = X.shape
    density = X.nnz / (n_obs * n_vars) if sp.issparse(X) else 1.0

    gc.collect()
    t0 = time.perf_counter()

    result = an.run_svd(
        X,
        n_components=n_components,
        algorithm=algorithm,
        seed=42,
        verbose=False,
    )

    wall_time = time.perf_counter() - t0

    sigma = np.asarray(result["d"]).ravel()
    top_5 = sigma[:5].tolist()

    return BenchmarkResult(
        matrix_type=matrix_type,
        storage_mode="inmemory",
        algorithm=algorithm,
        n_obs=n_obs,
        n_vars=n_vars,
        n_components=n_components,
        density=density,
        wall_time_s=wall_time,
        top_5_sigma=top_5,
    )


def benchmark_backed(
    X: np.ndarray,
    algorithm: str,
    n_components: int,
    tmp_dir: Path,
) -> BenchmarkResult:
    """Benchmark backed SVD."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    matrix_type = "sparse" if sp.issparse(X) else "dense"
    n_obs, n_vars = X.shape
    density = X.nnz / (n_obs * n_vars) if sp.issparse(X) else 1.0

    # Create backed AnnData
    if sp.issparse(X):
        adata = ad.AnnData(X=X.tocsr())
    else:
        adata = ad.AnnData(X=X)

    h5ad_path = tmp_dir / f"test_{matrix_type}_{algorithm}.h5ad"
    adata.write_h5ad(h5ad_path)

    # Reopen in backed mode
    adata_backed = ad.read_h5ad(h5ad_path, backed="r+")

    try:
        gc.collect()
        t0 = time.perf_counter()

        result = an.run_svd(
            adata_backed,
            n_components=n_components,
            algorithm=algorithm,
            seed=42,
            verbose=False,
            backed_chunk_size=4096,
        )

        wall_time = time.perf_counter() - t0

        sigma = np.asarray(result["d"]).ravel()
        top_5 = sigma[:5].tolist()

        return BenchmarkResult(
            matrix_type=matrix_type,
            storage_mode="backed",
            algorithm=algorithm,
            n_obs=n_obs,
            n_vars=n_vars,
            n_components=n_components,
            density=density,
            wall_time_s=wall_time,
            top_5_sigma=top_5,
        )
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()


def run_benchmarks(
    size: str = "small",
    algorithms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run comprehensive benchmark suite."""
    if algorithms is None:
        algorithms = ["irlb", "halko", "feng", "primme"]

    preset = SIZE_PRESETS[size]
    n_obs = preset["n_obs"]
    n_vars = preset["n_vars"]
    n_components = preset["n_components"]
    density = preset["density"]

    print(f"Benchmark configuration: {size}")
    print(f"  n_obs: {n_obs}, n_vars: {n_vars}, n_components: {n_components}")
    print(f"  Algorithms: {algorithms}")
    print()

    # Generate test matrices once
    print("Generating test matrices...")
    X_sparse = create_test_matrix(n_obs, n_vars, density, as_sparse=True)
    X_dense = create_test_matrix(n_obs, n_vars, density, as_sparse=False)
    print(f"  Sparse: {X_sparse.shape} with {X_sparse.nnz:,} nonzeros ({X_sparse.nnz / (n_obs*n_vars):.1%} dense)")
    print(f"  Dense: {X_dense.shape}")
    print()

    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for alg in algorithms:
            print(f"Algorithm: {alg}")

            # In-memory sparse
            try:
                print(f"  Running: sparse in-memory...")
                res = benchmark_inmemory(X_sparse, alg, n_components)
                results.append(res)
                print(f"    ✓ {res.wall_time_s:.3f}s")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

            # In-memory dense
            try:
                print(f"  Running: dense in-memory...")
                res = benchmark_inmemory(X_dense, alg, n_components)
                results.append(res)
                print(f"    ✓ {res.wall_time_s:.3f}s")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

            # Backed sparse
            try:
                print(f"  Running: sparse backed...")
                res = benchmark_backed(X_sparse, alg, n_components, tmp_path)
                results.append(res)
                print(f"    ✓ {res.wall_time_s:.3f}s")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

            # Backed dense
            try:
                print(f"  Running: dense backed...")
                res = benchmark_backed(X_dense, alg, n_components, tmp_path)
                results.append(res)
                print(f"    ✓ {res.wall_time_s:.3f}s")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

            print()

    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print benchmark summary."""
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Group by configuration
    summary = df.pivot_table(
        index=["matrix_type", "storage_mode"],
        columns="algorithm",
        values="wall_time_s",
        aggfunc="mean",
    )

    print("Wall Time (seconds)")
    print("-" * 80)
    print(summary.to_string())
    print()

    # Speed comparisons (relative to IRLB)
    if "irlb" in df["algorithm"].values:
        print("Speed Comparison (relative to IRLB, lower is faster)")
        print("-" * 80)
        for (mtype, smode), group in df.groupby(["matrix_type", "storage_mode"]):
            irlb_time = group[group["algorithm"] == "irlb"]["wall_time_s"].values
            if len(irlb_time) > 0:
                irlb_time = irlb_time[0]
                print(f"\n{mtype} {smode}:")
                for _, row in group.iterrows():
                    ratio = row["wall_time_s"] / irlb_time
                    marker = "✓" if ratio <= 1.2 else "⚠" if ratio <= 2.0 else "✗"
                    print(f"  {marker} {row['algorithm']:8s}: {ratio:5.2f}x  ({row['wall_time_s']:.3f}s)")
        print()

    # Singular value consistency
    print("Singular Value Consistency (top singular value)")
    print("-" * 80)
    for (mtype, smode), group in df.groupby(["matrix_type", "storage_mode"]):
        sigmas = group["sigma_1"].values
        if len(sigmas) > 0:
            mean_sigma = np.mean(sigmas)
            std_sigma = np.std(sigmas)
            cv = (std_sigma / mean_sigma) * 100
            print(f"{mtype} {smode}: mean={mean_sigma:.1f}, std={std_sigma:.1f}, CV={cv:.2f}%")

    print()
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark all IRLB SVD implementations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size preset",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["irlb", "halko"],
        choices=["irlb", "halko", "feng", "primme"],
        help="Algorithms to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )

    args = parser.parse_args()

    # Run benchmarks
    df = run_benchmarks(size=args.size, algorithms=args.algorithms)

    # Print summary
    print_summary(df)

    # Save results
    if args.output:
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nFull results:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
