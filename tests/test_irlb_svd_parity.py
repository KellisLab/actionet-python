#!/usr/bin/env python3
"""test_irlb_svd_parity.py - Parity tests for IRLB SVD implementations.

Tests all SVD algorithm variants (sparse, dense, in-memory, disk-backed) to verify
that outputs are consistent across implementations. Validates that:
  - Singular values are numerically similar
  - Reconstruction error is consistent
  - U, D, V decomposition is valid

Covers:
  - In-memory sparse (CSR)
  - In-memory dense (numpy array)
  - Disk-backed sparse (HDF5)
  - Disk-backed dense (HDF5)
  - All supported algorithms: IRLB, Halko
  - Rejection of retired algorithms (Feng, PRIMME) at every API surface
  - SVD backend / algorithm provenance metadata attached to ``run_svd``
    results and persisted by ``reduce_kernel``
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import actionet as an
from actionet import _core


# Tolerance settings for comparing DIFFERENT algorithms (IRLB vs Halko)
# These are intentionally relaxed because different algorithms produce slightly different results
SIGMA_RTOL = 0.05      # 5% relative tolerance for singular values (cross-algorithm)
SIGMA_ATOL = 1e-3      # Absolute tolerance for singular values
RECON_RTOL = 0.95      # Relative reconstruction error tolerance (relaxed for full-rank test matrices)
SIGMA_CORR_THRESHOLD = 0.999  # Pearson correlation threshold for sigma vectors (cross-algorithm)

# Halko's in-memory C++ default (``default_max_it=5``) is tuned for real
# single-cell matrices where spectra decay quickly. On the small full-rank
# synthetic matrices used here the tail singular values have not converged
# after 5 iterations, so the cross-algorithm parity checks below pass
# ``max_iter=HALKO_SYNTHETIC_MAX_IT`` to give Halko room to match IRLB.
# The single-algorithm smoke tests keep the default max_iter=0 so the
# public default is still exercised.
HALKO_SYNTHETIC_MAX_IT = 20


def _create_test_matrix(
    n_obs: int = 200,
    n_vars: int = 100,
    density: float = 0.3,
    rank: int = 20,
    random_state: int = 42,
    as_sparse: bool = True,
) -> np.ndarray:
    """Create a synthetic test matrix.

    For sparse matrices, generates a random sparse matrix directly.
    For dense matrices, creates a low-rank matrix.
    """
    rng = np.random.RandomState(random_state)

    if as_sparse:
        # Generate sparse matrix directly using random sparse structure
        X = sp.random(n_obs, n_vars, density=density, random_state=random_state, format='csr')
        # Scale values to have reasonable magnitudes
        X.data = X.data * 100
        return X
    else:
        # Create low-rank dense matrix
        U = rng.randn(n_obs, rank)
        V = rng.randn(n_vars, rank)
        sigma = np.linspace(100, 10, rank)
        X = U @ np.diag(sigma) @ V.T
        return X


def _create_backed_anndata(
    X: np.ndarray,
    tmp_dir: Path,
    prefix: str = "test",
) -> Tuple[ad.AnnData, Path]:
    """Create a backed AnnData from a matrix and return (adata, path)."""
    if sp.issparse(X):
        adata = ad.AnnData(X=X.tocsr())
    else:
        adata = ad.AnnData(X=X)

    h5ad_path = tmp_dir / f"{prefix}.h5ad"
    adata.write_h5ad(h5ad_path)

    # Reopen in backed mode
    adata_backed = ad.read_h5ad(h5ad_path, backed="r+")
    return adata_backed, h5ad_path


def _validate_svd_result(
    result: Dict[str, np.ndarray],
    X: np.ndarray,
    n_components: int,
    rtol: float = RECON_RTOL,
) -> None:
    """Validate that SVD result is well-formed and reconstructs X."""
    u = np.asarray(result["u"])
    d = np.asarray(result["d"]).ravel()
    v = np.asarray(result["v"])

    # Check shapes
    if sp.issparse(X):
        n_obs, n_vars = X.shape
    else:
        n_obs, n_vars = X.shape

    assert u.shape == (n_obs, n_components), f"U shape mismatch: {u.shape}"
    assert d.shape == (n_components,), f"D shape mismatch: {d.shape}"
    # V is returned as (n_vars, n_components) in this implementation
    assert v.shape == (n_vars, n_components), f"V shape mismatch: {v.shape}"

    # Check that singular values are non-negative and sorted
    assert np.all(d >= 0), "Singular values must be non-negative"
    assert np.all(d[:-1] >= d[1:]), "Singular values must be sorted descending"

    # Reconstruction check (sample rows to avoid OOM on large matrices)
    probe_size = min(50, n_obs)
    idx = np.random.RandomState(99).choice(n_obs, probe_size, replace=False)

    if sp.issparse(X):
        X_probe = X[idx].toarray()
    else:
        X_probe = X[idx]

    u_probe = u[idx]
    # V is (n_vars, n_components), so we need V.T for reconstruction
    X_recon = (u_probe * d) @ v.T

    diff_norm = np.linalg.norm(X_probe - X_recon, "fro")
    orig_norm = np.linalg.norm(X_probe, "fro")
    rel_err = diff_norm / (orig_norm + 1e-12)

    assert rel_err < rtol, f"Reconstruction error {rel_err:.6f} exceeds tolerance {rtol}"


def _compare_svd_results(
    result_a: Dict[str, np.ndarray],
    result_b: Dict[str, np.ndarray],
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Compare two SVD results for parity."""
    d_a = np.asarray(result_a["d"]).ravel()
    d_b = np.asarray(result_b["d"]).ravel()

    # Compare singular values
    assert d_a.shape == d_b.shape, f"Sigma shape mismatch: {d_a.shape} vs {d_b.shape}"

    # Relative/absolute tolerance
    np.testing.assert_allclose(
        d_a, d_b,
        rtol=SIGMA_RTOL,
        atol=SIGMA_ATOL,
        err_msg=f"Singular values differ between {label_a} and {label_b}",
    )

    # Correlation check
    if len(d_a) >= 2:
        corr = np.corrcoef(d_a, d_b)[0, 1]
        assert corr > SIGMA_CORR_THRESHOLD, (
            f"Singular value correlation {corr:.6f} below threshold "
            f"{SIGMA_CORR_THRESHOLD} ({label_a} vs {label_b})"
        )


# ============================================================================
# Test Cases: In-Memory
# ============================================================================

@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_inmemory_sparse_svd_algorithms(algorithm):
    """Test in-memory sparse SVD for all algorithms."""
    n_components = 10
    X_sparse = _create_test_matrix(
        n_obs=200, n_vars=100, density=0.1, rank=20, as_sparse=True, random_state=42
    )

    result = an.run_svd(
        X_sparse,
        n_components=n_components,
        algorithm=algorithm,
        seed=42,
        verbose=False,
    )

    _validate_svd_result(result, X_sparse, n_components)


@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_inmemory_dense_svd_algorithms(algorithm):
    """Test in-memory dense SVD for all algorithms."""
    n_components = 10
    X_dense = _create_test_matrix(
        n_obs=200, n_vars=100, density=1.0, rank=20, as_sparse=False, random_state=42
    )

    result = an.run_svd(
        X_dense,
        n_components=n_components,
        algorithm=algorithm,
        seed=42,
        verbose=False,
    )

    _validate_svd_result(result, X_dense, n_components)


def test_inmemory_sparse_parity_irlb_vs_halko():
    """Verify IRLB and Halko produce consistent results for in-memory sparse."""
    n_components = 15
    X_sparse = _create_test_matrix(
        n_obs=300, n_vars=150, density=0.1, rank=30, as_sparse=True, random_state=123
    )

    result_irlb = an.run_svd(
        X_sparse, n_components=n_components, algorithm="irlb", seed=42, verbose=False
    )
    result_halko = an.run_svd(
        X_sparse,
        n_components=n_components,
        algorithm="halko",
        seed=42,
        verbose=False,
        max_iter=HALKO_SYNTHETIC_MAX_IT,
    )

    _validate_svd_result(result_irlb, X_sparse, n_components)
    _validate_svd_result(result_halko, X_sparse, n_components)
    _compare_svd_results(result_irlb, result_halko, "IRLB", "Halko")


def test_inmemory_dense_parity_irlb_vs_halko():
    """Verify IRLB and Halko produce consistent results for in-memory dense."""
    n_components = 15
    X_dense = _create_test_matrix(
        n_obs=300, n_vars=150, density=1.0, rank=30, as_sparse=False, random_state=123
    )

    result_irlb = an.run_svd(
        X_dense, n_components=n_components, algorithm="irlb", seed=42, verbose=False
    )
    result_halko = an.run_svd(
        X_dense,
        n_components=n_components,
        algorithm="halko",
        seed=42,
        verbose=False,
        max_iter=HALKO_SYNTHETIC_MAX_IT,
    )

    _validate_svd_result(result_irlb, X_dense, n_components)
    _validate_svd_result(result_halko, X_dense, n_components)
    _compare_svd_results(result_irlb, result_halko, "IRLB", "Halko")


# ============================================================================
# Test Cases: Disk-Backed
# ============================================================================

@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_backed_sparse_svd_algorithms(algorithm, tmp_path):
    """Test backed sparse SVD for all algorithms."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 10
    X_sparse = _create_test_matrix(
        n_obs=200, n_vars=100, density=0.1, rank=20, as_sparse=True, random_state=42
    )

    adata_backed, h5ad_path = _create_backed_anndata(X_sparse, tmp_path, f"backed_sparse_{algorithm}")

    try:
        result = an.run_svd(
            adata_backed,
            n_components=n_components,
            algorithm=algorithm,
            seed=42,
            verbose=False,
            backed_chunk_size=4096,
        )

        _validate_svd_result(result, X_sparse, n_components)
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()


@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_backed_dense_svd_algorithms(algorithm, tmp_path):
    """Test backed dense SVD for all algorithms."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 10
    X_dense = _create_test_matrix(
        n_obs=200, n_vars=100, density=1.0, rank=20, as_sparse=False, random_state=42
    )

    adata_backed, h5ad_path = _create_backed_anndata(X_dense, tmp_path, f"backed_dense_{algorithm}")

    try:
        result = an.run_svd(
            adata_backed,
            n_components=n_components,
            algorithm=algorithm,
            seed=42,
            verbose=False,
            backed_chunk_size=4096,
        )

        _validate_svd_result(result, X_dense, n_components)
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()


def test_backed_sparse_parity_irlb_vs_halko(tmp_path):
    """Verify IRLB and Halko produce consistent results for backed sparse."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 15
    X_sparse = _create_test_matrix(
        n_obs=300, n_vars=150, density=0.1, rank=30, as_sparse=True, random_state=456
    )

    # Create two separate backed files (avoid file locking issues)
    adata_irlb, h5ad_irlb = _create_backed_anndata(X_sparse, tmp_path, "backed_sparse_irlb")
    adata_halko, h5ad_halko = _create_backed_anndata(X_sparse, tmp_path, "backed_sparse_halko")

    try:
        result_irlb = an.run_svd(
            adata_irlb, n_components=n_components, algorithm="irlb", seed=42, verbose=False
        )
        result_halko = an.run_svd(
            adata_halko,
            n_components=n_components,
            algorithm="halko",
            seed=42,
            verbose=False,
            max_iter=HALKO_SYNTHETIC_MAX_IT,
        )

        _validate_svd_result(result_irlb, X_sparse, n_components)
        _validate_svd_result(result_halko, X_sparse, n_components)
        _compare_svd_results(result_irlb, result_halko, "IRLB-backed", "Halko-backed")
    finally:
        for adata, path in [(adata_irlb, h5ad_irlb), (adata_halko, h5ad_halko)]:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            if path.exists():
                path.unlink()


def test_backed_dense_parity_irlb_vs_halko(tmp_path):
    """Verify IRLB and Halko produce consistent results for backed dense."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 15
    X_dense = _create_test_matrix(
        n_obs=300, n_vars=150, density=1.0, rank=30, as_sparse=False, random_state=456
    )

    # Create two separate backed files
    adata_irlb, h5ad_irlb = _create_backed_anndata(X_dense, tmp_path, "backed_dense_irlb")
    adata_halko, h5ad_halko = _create_backed_anndata(X_dense, tmp_path, "backed_dense_halko")

    try:
        result_irlb = an.run_svd(
            adata_irlb, n_components=n_components, algorithm="irlb", seed=42, verbose=False
        )
        result_halko = an.run_svd(
            adata_halko,
            n_components=n_components,
            algorithm="halko",
            seed=42,
            verbose=False,
            max_iter=HALKO_SYNTHETIC_MAX_IT,
        )

        _validate_svd_result(result_irlb, X_dense, n_components)
        _validate_svd_result(result_halko, X_dense, n_components)
        _compare_svd_results(result_irlb, result_halko, "IRLB-backed", "Halko-backed")
    finally:
        for adata, path in [(adata_irlb, h5ad_irlb), (adata_halko, h5ad_halko)]:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            if path.exists():
                path.unlink()


# ============================================================================
# Cross-Implementation Parity: In-Memory vs Backed
# ============================================================================

def test_parity_inmemory_vs_backed_sparse_irlb(tmp_path):
    """Verify in-memory and backed sparse produce consistent IRLB results."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 12
    X_sparse = _create_test_matrix(
        n_obs=250, n_vars=120, density=0.1, rank=25, as_sparse=True, random_state=789
    )

    # In-memory
    result_mem = an.run_svd(
        X_sparse, n_components=n_components, algorithm="irlb", seed=42, verbose=False
    )

    # Backed
    adata_backed, h5ad_path = _create_backed_anndata(X_sparse, tmp_path, "parity_backed_sparse")
    try:
        result_backed = an.run_svd(
            adata_backed, n_components=n_components, algorithm="irlb", seed=42, verbose=False
        )

        _validate_svd_result(result_mem, X_sparse, n_components)
        _validate_svd_result(result_backed, X_sparse, n_components)
        _compare_svd_results(result_mem, result_backed, "In-Memory-IRLB", "Backed-IRLB")
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()


def test_parity_inmemory_vs_backed_dense_halko(tmp_path):
    """Verify in-memory and backed dense produce consistent Halko results."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 12
    X_dense = _create_test_matrix(
        n_obs=250, n_vars=120, density=1.0, rank=25, as_sparse=False, random_state=789
    )

    # In-memory
    result_mem = an.run_svd(
        X_dense, n_components=n_components, algorithm="halko", seed=42, verbose=False
    )

    # Backed
    adata_backed, h5ad_path = _create_backed_anndata(X_dense, tmp_path, "parity_backed_dense")
    try:
        result_backed = an.run_svd(
            adata_backed, n_components=n_components, algorithm="halko", seed=42, verbose=False
        )

        _validate_svd_result(result_mem, X_dense, n_components)
        _validate_svd_result(result_backed, X_dense, n_components)
        _compare_svd_results(result_mem, result_backed, "In-Memory-Halko", "Backed-Halko")
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()


# ============================================================================
# Stress Test: All Combinations
# ============================================================================

def test_all_combinations_consistency(tmp_path):
    """Comprehensive test: all algorithms on all data types."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    n_components = 8
    algorithms = ["irlb", "halko"]

    # Generate test data
    X_sparse = _create_test_matrix(
        n_obs=150, n_vars=80, density=0.15, rank=20, as_sparse=True, random_state=111
    )
    X_dense = _create_test_matrix(
        n_obs=150, n_vars=80, density=1.0, rank=20, as_sparse=False, random_state=111
    )

    results = {}

    # In-memory sparse
    for alg in algorithms:
        key = f"sparse_mem_{alg}"
        results[key] = an.run_svd(
            X_sparse, n_components=n_components, algorithm=alg, seed=42, verbose=False
        )
        _validate_svd_result(results[key], X_sparse, n_components)

    # In-memory dense
    for alg in algorithms:
        key = f"dense_mem_{alg}"
        results[key] = an.run_svd(
            X_dense, n_components=n_components, algorithm=alg, seed=42, verbose=False
        )
        _validate_svd_result(results[key], X_dense, n_components)

    # Backed sparse
    adata_backed_sparse, h5ad_sparse = _create_backed_anndata(
        X_sparse, tmp_path, "all_comb_sparse"
    )
    try:
        for alg in algorithms:
            key = f"sparse_backed_{alg}"
            results[key] = an.run_svd(
                adata_backed_sparse, n_components=n_components, algorithm=alg, seed=42, verbose=False
            )
            _validate_svd_result(results[key], X_sparse, n_components)
    finally:
        if hasattr(adata_backed_sparse, "file") and adata_backed_sparse.file is not None:
            adata_backed_sparse.file.close()
        if h5ad_sparse.exists():
            h5ad_sparse.unlink()

    # Backed dense
    adata_backed_dense, h5ad_dense = _create_backed_anndata(
        X_dense, tmp_path, "all_comb_dense"
    )
    try:
        for alg in algorithms:
            key = f"dense_backed_{alg}"
            results[key] = an.run_svd(
                adata_backed_dense, n_components=n_components, algorithm=alg, seed=42, verbose=False
            )
            _validate_svd_result(results[key], X_dense, n_components)
    finally:
        if hasattr(adata_backed_dense, "file") and adata_backed_dense.file is not None:
            adata_backed_dense.file.close()
        if h5ad_dense.exists():
            h5ad_dense.unlink()

    # Cross-validate all sparse results
    sparse_keys = [k for k in results if "sparse" in k]
    for i, key_a in enumerate(sparse_keys):
        for key_b in sparse_keys[i+1:]:
            _compare_svd_results(results[key_a], results[key_b], key_a, key_b)

    # Cross-validate all dense results
    dense_keys = [k for k in results if "dense" in k]
    for i, key_a in enumerate(dense_keys):
        for key_b in dense_keys[i+1:]:
            _compare_svd_results(results[key_a], results[key_b], key_a, key_b)

    print(f"\nAll {len(results)} combinations validated successfully!")


# ============================================================================
# Regression tests: PRIMME removal and 64-bit sparse support (Phases 1 & 2)
# ============================================================================


@pytest.mark.parametrize("algorithm", ["primme", "PRIMME", "Primme"])
def test_primme_algorithm_rejected(algorithm):
    """Requesting the removed `primme` algorithm must raise ValueError.

    PRIMME was removed from the public Python SVD API. The C++ sources
    remain compiled for one release cycle but are unreachable from Python
    (see context/DECISIONS.md - "SVD algorithm strategy"). Any request for
    "primme" (in any casing) must fail during `_normalize_algorithm`.
    """
    X = _create_test_matrix(n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=0)
    with pytest.raises(ValueError, match=r"Invalid algorithm"):
        an.run_svd(X, n_components=4, algorithm=algorithm, verbose=False)


def test_reduce_kernel_rejects_primme():
    """`reduce_kernel` must also refuse `svd_algorithm="primme"`."""
    X = _create_test_matrix(n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=1)
    adata = ad.AnnData(X=X)
    with pytest.raises(ValueError, match=r"Invalid algorithm") as excinfo:
        an.reduce_kernel(adata, n_components=4, svd_algorithm="primme", verbose=False)

    message = str(excinfo.value)
    for name in ("auto", "halko", "irlb"):
        assert name in message, f"Expected {name!r} in allowed set of error message: {message}"


@pytest.mark.parametrize("algorithm", ["feng", "FENG", "Feng"])
def test_feng_algorithm_rejected(algorithm):
    """Requesting the removed `feng` algorithm must raise ValueError.

    Feng was retired from the public Python SVD API. The C++ sources remain
    compiled for one release cycle but are unreachable from Python (see
    context/DECISIONS.md - "SVD algorithm strategy update: Feng retired
    from public API"). Any request for "feng" (in any casing) must fail
    during `_normalize_algorithm` with a message listing the allowed set
    ``{auto, halko, irlb}``.
    """
    X = _create_test_matrix(n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=0)
    with pytest.raises(ValueError, match=r"Invalid algorithm") as excinfo:
        an.run_svd(X, n_components=4, algorithm=algorithm, verbose=False)
    message = str(excinfo.value)
    for name in ("auto", "halko", "irlb"):
        assert name in message, f"Expected {name!r} in allowed set of error message: {message}"


def test_reduce_kernel_rejects_feng():
    """`reduce_kernel` must also refuse `svd_algorithm="feng"`."""
    X = _create_test_matrix(n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=1)
    adata = ad.AnnData(X=X)
    with pytest.raises(ValueError, match=r"Invalid algorithm") as excinfo:
        an.reduce_kernel(adata, n_components=4, svd_algorithm="feng", verbose=False)

    message = str(excinfo.value)
    for name in ("auto", "halko", "irlb"):
        assert name in message, f"Expected {name!r} in allowed set of error message: {message}"


@pytest.mark.parametrize("algorithm_id", [2, 3])
def test_core_run_svd_rejects_retired_algorithm_ids(algorithm_id):
    """Private `_core.run_svd_*` calls must not bypass Python SVD policy."""
    X_sparse = _create_test_matrix(
        n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=2
    ).tocsr()
    X_dense = X_sparse.toarray()

    with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
        _core.run_svd_sparse(X_sparse, 4, 0, 0, algorithm_id, False)

    with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
        _core.run_svd_dense(X_dense, 4, 0, 0, algorithm_id, False)


@pytest.mark.parametrize("algorithm_id", [2, 3])
def test_core_reduce_kernel_rejects_retired_algorithm_ids(algorithm_id):
    """Private `_core.reduce_kernel_*` calls must also reject retired SVD IDs."""
    X_sparse = _create_test_matrix(
        n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=3
    ).tocsr()
    X_dense = X_sparse.toarray()

    with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
        _core.reduce_kernel_sparse(X_sparse, 4, algorithm_id, 0, 0, False)

    with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
        _core.reduce_kernel_dense(X_dense, 4, algorithm_id, 0, 0, False)


@pytest.mark.parametrize("algorithm_id", [2, 3])
def test_core_backed_operator_rejects_retired_algorithm_ids(tmp_path, algorithm_id):
    """Backed `_core` SVD entry points expose only IRLB/Halko to Python."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    X_sparse = _create_test_matrix(
        n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=4
    ).tocsr()
    adata_backed, h5ad_path = _create_backed_anndata(
        X_sparse, tmp_path, prefix=f"core_retired_{algorithm_id}"
    )

    # Close the anndata `r+` handle before the C++ side opens the same file,
    # otherwise HDF5 refuses to co-open the file on Linux even with locking
    # disabled at the environment level.
    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    try:
        op = _core.create_backed_operator(str(h5ad_path), "/X", 16)

        with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
            _core.run_svd_backed_operator(op, 4, 0, 0, algorithm_id, False)

        with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
            _core.reduce_kernel_backed_operator(op, 4, algorithm_id, 0, 0, False)
    finally:
        if h5ad_path.exists():
            h5ad_path.unlink()


def test_irlb_sparse_accepts_int64_indices():
    """IRLB on scipy CSR with int64 `indices`/`indptr` completes correctly.

    Phase 2 documents that sparse ``nnz > 2^31 - 1`` is supported end-to-end.
    Materializing such a matrix in CI is impractical, but the plumbing that
    would enable it is: ``scipy_to_arma_sparse`` reads ``indices``/``indptr``
    as ``py::ssize_t`` (int64), and Armadillo's ``sp_mat`` uses 64-bit
    ``uword`` under ``ARMA_64BIT_WORD``. This test exercises the plumbing at
    a small scale by forcing a scipy CSR to use int64 index arrays and
    verifying IRLB produces the same singular values as the int32-indexed
    equivalent.

    scipy downcasts index arrays to int32 whenever the shape/nnz allow it,
    so we assign int64 arrays post-construction to preserve them.
    """
    X32 = _create_test_matrix(
        n_obs=200, n_vars=120, density=0.15, as_sparse=True, random_state=7
    )
    X32 = sp.csr_matrix(X32).astype(np.float64)
    assert X32.indices.dtype == np.int32
    assert X32.indptr.dtype == np.int32

    X64 = sp.csr_matrix(X32, copy=True)
    X64.indices = X32.indices.astype(np.int64, copy=True)
    X64.indptr = X32.indptr.astype(np.int64, copy=True)
    assert X64.indices.dtype == np.int64
    assert X64.indptr.dtype == np.int64

    n_components = 10
    result32 = an.run_svd(X32, n_components=n_components, algorithm="irlb", seed=42, verbose=False)
    result64 = an.run_svd(X64, n_components=n_components, algorithm="irlb", seed=42, verbose=False)

    np.testing.assert_allclose(result32["d"], result64["d"], rtol=1e-10, atol=1e-12)
    _validate_svd_result(result64, X64, n_components)


# ============================================================================
# Regression tests: SVD backend / algorithm provenance metadata (Phase 2)
# ============================================================================


_SVD_METADATA_KEYS = (
    "svd_algorithm",
    "svd_algorithm_name",
    "svd_backend_requested",
    "svd_backend_resolved",
)


@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_run_svd_emits_backend_metadata_inmemory_sparse(algorithm):
    """`run_svd` must attach backend/algorithm provenance on the non-operator path."""
    X = _create_test_matrix(
        n_obs=64, n_vars=32, density=0.25, as_sparse=True, random_state=17
    )
    result = an.run_svd(
        X,
        n_components=6,
        algorithm=algorithm,
        seed=42,
        verbose=False,
        return_operator_compatible=False,
    )

    for key in _SVD_METADATA_KEYS:
        assert key in result, f"expected {key!r} in run_svd result, got {sorted(result)}"

    assert result["svd_algorithm"] == (0 if algorithm == "irlb" else 1)
    assert result["svd_algorithm_name"] == algorithm
    assert result["svd_backend_requested"] == "cpu"
    assert result["svd_backend_resolved"] == "cpu"


@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_run_svd_emits_backend_metadata_inmemory_dense(algorithm):
    """Dense in-memory path must also emit backend/algorithm provenance."""
    X = _create_test_matrix(
        n_obs=64, n_vars=32, density=1.0, as_sparse=False, random_state=19
    )
    result = an.run_svd(
        X,
        n_components=6,
        algorithm=algorithm,
        seed=42,
        verbose=False,
        return_operator_compatible=False,
    )

    for key in _SVD_METADATA_KEYS:
        assert key in result
    assert result["svd_algorithm_name"] == algorithm


@pytest.mark.parametrize("algorithm", ["irlb", "halko"])
def test_run_svd_emits_backend_metadata_backed(algorithm, tmp_path):
    """Backed streaming path must emit the same provenance keys as in-memory."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    X = _create_test_matrix(
        n_obs=80, n_vars=48, density=0.2, as_sparse=True, random_state=21
    )
    adata_backed, h5ad_path = _create_backed_anndata(
        X, tmp_path, prefix=f"backend_meta_{algorithm}"
    )
    try:
        result = an.run_svd(
            adata_backed,
            n_components=6,
            algorithm=algorithm,
            seed=42,
            verbose=False,
            return_operator_compatible=False,
            backed_chunk_size=4096,
        )
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()

    for key in _SVD_METADATA_KEYS:
        assert key in result
    assert result["svd_algorithm_name"] == algorithm
    assert result["svd_backend_resolved"] == "cpu"


def test_run_svd_operator_compatible_omits_backend_metadata():
    """`return_operator_compatible=True` returns only the u/d/v triple."""
    X = _create_test_matrix(
        n_obs=48, n_vars=24, density=0.3, as_sparse=True, random_state=23
    )
    result = an.run_svd(
        X,
        n_components=5,
        algorithm="irlb",
        seed=42,
        verbose=False,
        return_operator_compatible=True,
    )
    assert set(result) == {"u", "d", "v"}


def test_reduce_kernel_persists_backend_metadata_inmemory():
    """`reduce_kernel` writes backend/algorithm provenance into ``uns[<key>_params]``."""
    X = _create_test_matrix(
        n_obs=64, n_vars=32, density=0.25, as_sparse=True, random_state=25
    )
    adata = ad.AnnData(X=X)
    an.reduce_kernel(adata, n_components=6, svd_algorithm="halko", seed=42, verbose=False)

    params = adata.uns["action_params"]
    for key in _SVD_METADATA_KEYS:
        assert key in params, f"expected {key!r} in reduce_kernel params, got {sorted(params)}"

    assert params["svd_algorithm"] == 1
    assert params["svd_algorithm_name"] == "halko"
    assert params["svd_backend_requested"] == "cpu"
    assert params["svd_backend_resolved"] == "cpu"
    assert params["used_precomputed_svd"] is False
    assert params["operator_mode"] is False


def test_reduce_kernel_precomputed_svd_marks_algorithm_none():
    """When a precomputed SVD is supplied, no in-house algorithm was run.

    The persisted metadata must therefore report ``svd_algorithm=None`` /
    ``svd_algorithm_name="none"`` instead of the resolved-but-unused id;
    ``used_precomputed_svd`` should be True.
    """
    X = _create_test_matrix(
        n_obs=64, n_vars=32, density=0.25, as_sparse=True, random_state=27
    )
    svd = an.run_svd(X, n_components=6, algorithm="irlb", seed=42, verbose=False)

    adata = ad.AnnData(X=X)
    an.reduce_kernel(
        adata,
        n_components=6,
        svd_algorithm="halko",
        precomputed_svd=svd,
        verbose=False,
    )

    params = adata.uns["action_params"]
    assert params["svd_algorithm"] is None
    assert params["svd_algorithm_name"] == "none"
    assert params["used_precomputed_svd"] is True
    assert params["svd_backend_requested"] == "cpu"
    assert params["svd_backend_resolved"] == "cpu"


def test_reduce_kernel_persists_backend_metadata_backed(tmp_path):
    """Backed `reduce_kernel` writes the same provenance keys as in-memory,
    and reports ``operator_mode=True``.

    Locks in that the streaming branch of ``reduce_kernel`` funnels into the
    same ``params`` dict as the in-memory branch and preserves the full
    backend/algorithm contract exposed by Phase 2.
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    X = _create_test_matrix(
        n_obs=80, n_vars=48, density=0.2, as_sparse=True, random_state=29
    )
    adata_backed, h5ad_path = _create_backed_anndata(
        X, tmp_path, prefix="reduce_kernel_backed_meta"
    )
    try:
        an.reduce_kernel(
            adata_backed,
            n_components=6,
            svd_algorithm="halko",
            seed=42,
            verbose=False,
        )
        params = dict(adata_backed.uns["action_params"])
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()

    for key in _SVD_METADATA_KEYS:
        assert key in params, f"expected {key!r} in backed reduce_kernel params, got {sorted(params)}"

    assert params["svd_algorithm"] == 1
    assert params["svd_algorithm_name"] == "halko"
    assert params["svd_backend_requested"] == "cpu"
    assert params["svd_backend_resolved"] == "cpu"
    assert params["used_precomputed_svd"] is False
    assert params["operator_mode"] is True


def test_reduce_kernel_precomputed_svd_marks_algorithm_none_backed(tmp_path):
    """Backed ``reduce_kernel(precomputed_svd=...)`` also records
    ``svd_algorithm=None`` / ``svd_algorithm_name="none"``.

    The backed precomputed-SVD short-circuit runs a different C++ entry point
    (``_core.reduce_kernel_from_svd_backed_operator``) than the in-memory
    variant; this test guards its provenance contract explicitly.
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    X = _create_test_matrix(
        n_obs=80, n_vars=48, density=0.2, as_sparse=True, random_state=31
    )
    svd = an.run_svd(X, n_components=6, algorithm="irlb", seed=42, verbose=False)

    adata_backed, h5ad_path = _create_backed_anndata(
        X, tmp_path, prefix="reduce_kernel_backed_precomputed"
    )
    try:
        an.reduce_kernel(
            adata_backed,
            n_components=6,
            svd_algorithm="halko",
            precomputed_svd=svd,
            verbose=False,
        )
        params = dict(adata_backed.uns["action_params"])
    finally:
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()
        if h5ad_path.exists():
            h5ad_path.unlink()

    assert params["svd_algorithm_name"] == "none"
    assert params["used_precomputed_svd"] is True
    assert params["operator_mode"] is True
    assert params["svd_backend_requested"] == "cpu"
    assert params["svd_backend_resolved"] == "cpu"
    # ``svd_algorithm`` is written as ``None`` for the precomputed path but
    # AnnData's HDF5 writer drops ``None`` entries from ``uns`` on backed
    # persistence; either "absent" or "None" is acceptable for the backed
    # contract. Assert the *absence* of a numeric algorithm id so the
    # provenance stays honest.
    assert params.get("svd_algorithm") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
