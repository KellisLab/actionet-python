"""Tests for Step 4 (L4): fused rowStats in BackedSparseMatrixOperator.

Validates that the backed sparse VISION annotation path (which now uses
the fused rowStats method internally) produces results identical to the
in-memory VISION path, both with and without lazy transforms.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import actionet._core as _core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sym_graph(n: int, density: float = 0.15, seed: int = 42) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)
    G = sp.random(n, n, density=density, random_state=rng, format="csc",
                  dtype=np.float64)
    G = G + G.T
    G.setdiag(0)
    G.eliminate_zeros()
    return G


def _write_sparse_h5ad(path, S: sp.spmatrix, fmt: str = "csr"):
    """Write a minimal h5ad with sparse X suitable for backed reading."""
    n_obs, n_var = S.shape
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    if fmt == "csr":
        X = sp.csr_matrix(S, dtype=np.float64)
    else:
        X = sp.csc_matrix(S, dtype=np.float64)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return adata


# ---------------------------------------------------------------------------
# Test: backed sparse VISION parity with in-memory (no transform)
# ---------------------------------------------------------------------------

class TestBackedVisionRowStatsParity:
    @pytest.fixture(params=["csr", "csc"], ids=["csr", "csc"])
    def sparse_fmt(self, request):
        return request.param

    def test_backed_vs_inmemory_no_transform(self, tmp_path, sparse_fmt):
        """Backed VISION result must match in-memory VISION result exactly."""
        n_cells, n_genes, n_labels = 80, 40, 5
        rng = np.random.default_rng(11)
        S = sp.random(n_cells, n_genes, density=0.3,
                      random_state=rng, format="csc", dtype=np.float64)
        G = _make_sym_graph(n_cells, seed=22)
        X = sp.random(n_genes, n_labels, density=0.4,
                      random_state=rng, format="csc", dtype=np.float64)

        ref = _core.compute_feature_stats_vision(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )

        h5_path = str(tmp_path / f"test_{sparse_fmt}.h5ad")
        _write_sparse_h5ad(h5_path, S, fmt=sparse_fmt)

        op = _core.create_backed_operator(
            file_path=h5_path, group_path="/X", chunk_size=32,
            n_threads=1,
        )
        backed = _core.compute_feature_stats_vision_backed_operator(
            op, G, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )

        np.testing.assert_allclose(backed, ref, rtol=1e-8, atol=1e-10)

    def test_backed_vs_inmemory_approx(self, tmp_path, sparse_fmt):
        """Chebyshev (approx=True) backed vs in-memory."""
        n_cells, n_genes, n_labels = 60, 30, 3
        rng = np.random.default_rng(33)
        S = sp.random(n_cells, n_genes, density=0.25,
                      random_state=rng, format="csc", dtype=np.float64)
        G = _make_sym_graph(n_cells, seed=44)
        X = sp.random(n_genes, n_labels, density=0.35,
                      random_state=rng, format="csc", dtype=np.float64)

        ref = _core.compute_feature_stats_vision(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=True, thread_no=1,
        )

        h5_path = str(tmp_path / f"test_approx_{sparse_fmt}.h5ad")
        _write_sparse_h5ad(h5_path, S, fmt=sparse_fmt)

        op = _core.create_backed_operator(
            file_path=h5_path, group_path="/X", chunk_size=32,
            n_threads=1,
        )
        backed = _core.compute_feature_stats_vision_backed_operator(
            op, G, X, norm_method=2, alpha=0.85, max_it=3,
            approx=True, thread_no=1,
        )

        np.testing.assert_allclose(backed, ref, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: backed sparse VISION with lazy transform (row_scale + log1p)
# ---------------------------------------------------------------------------

class TestBackedVisionLazyTransformParity:
    @pytest.fixture(params=["csr", "csc"], ids=["csr", "csc"])
    def sparse_fmt(self, request):
        return request.param

    def _apply_transform_inmemory(self, S: sp.spmatrix,
                                  row_scale: np.ndarray,
                                  log_base: float) -> sp.csc_matrix:
        """Apply the same transform that the backed operator does:
        scale each row, then log1p, then multiply by 1/log(base)."""
        S_csc = sp.csc_matrix(S, dtype=np.float64, copy=True)
        log_scale = 1.0 / np.log(log_base)
        # Scale rows
        diag = sp.diags(row_scale, format="csc")
        S_scaled = diag @ S_csc
        # log1p element-wise (on data array)
        S_out = S_scaled.copy()
        S_out.data = np.log1p(S_out.data) * log_scale
        return S_out

    def test_lazy_transform_parity(self, tmp_path, sparse_fmt):
        """Backed VISION with row_scale + log1p must match in-memory with
        the equivalent transform pre-applied."""
        n_cells, n_genes, n_labels = 64, 32, 4
        rng = np.random.default_rng(55)

        S_raw = sp.random(n_cells, n_genes, density=0.3,
                          random_state=rng, format="csc", dtype=np.float64)
        # Make all values positive (counts-like)
        S_raw.data = np.abs(S_raw.data) * 10.0
        S_raw.eliminate_zeros()

        G = _make_sym_graph(n_cells, seed=66)
        X = sp.random(n_genes, n_labels, density=0.4,
                      random_state=rng, format="csc", dtype=np.float64)

        # Row scale factors: 1 / total_counts per cell
        row_sums = np.asarray(S_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        row_scale = 1e4 / row_sums
        log_base = 2.0

        # In-memory reference: apply transform, then compute VISION stats
        S_transformed = self._apply_transform_inmemory(S_raw, row_scale, log_base)
        ref = _core.compute_feature_stats_vision(
            G, S_transformed, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )

        # Backed path with lazy transform
        h5_path = str(tmp_path / f"test_lazy_{sparse_fmt}.h5ad")
        _write_sparse_h5ad(h5_path, S_raw, fmt=sparse_fmt)

        log_scale = 1.0 / np.log(log_base)
        op = _core.create_backed_operator(
            file_path=h5_path, group_path="/X", chunk_size=32,
            row_scale_factors=row_scale,
            apply_log1p=True,
            log_scale=log_scale,
            n_threads=1,
        )
        backed = _core.compute_feature_stats_vision_backed_operator(
            op, G, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )

        # The backed operator uses fastlog approximation (~0.3% relative error),
        # so we relax tolerance accordingly.
        np.testing.assert_allclose(backed, ref, rtol=5e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: backed result is deterministic across repeated calls
# ---------------------------------------------------------------------------

class TestBackedVisionDeterminism:
    def test_repeated_calls_identical(self, tmp_path):
        n_cells, n_genes, n_labels = 50, 25, 3
        rng = np.random.default_rng(77)
        S = sp.random(n_cells, n_genes, density=0.25,
                      random_state=rng, format="csc", dtype=np.float64)
        G = _make_sym_graph(n_cells, seed=88)
        X = sp.random(n_genes, n_labels, density=0.3,
                      random_state=rng, format="csc", dtype=np.float64)

        h5_path = str(tmp_path / "test_det.h5ad")
        _write_sparse_h5ad(h5_path, S, fmt="csr")

        results = []
        for _ in range(3):
            op = _core.create_backed_operator(
                file_path=h5_path, group_path="/X", chunk_size=16,
                n_threads=1,
            )
            r = _core.compute_feature_stats_vision_backed_operator(
                op, G, X, norm_method=2, alpha=0.85, max_it=3,
                approx=False, thread_no=1,
            )
            results.append(r)

        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])


# ---------------------------------------------------------------------------
# Test: output shape and sanity
# ---------------------------------------------------------------------------

class TestBackedVisionSanity:
    def test_output_shape(self, tmp_path):
        n_cells, n_genes, n_labels = 50, 25, 7
        rng = np.random.default_rng(99)
        S = sp.random(n_cells, n_genes, density=0.2,
                      random_state=rng, format="csc", dtype=np.float64)
        G = _make_sym_graph(n_cells)
        X = sp.random(n_genes, n_labels, density=0.3,
                      random_state=rng, format="csc", dtype=np.float64)

        h5_path = str(tmp_path / "test_shape.h5ad")
        _write_sparse_h5ad(h5_path, S, fmt="csr")

        op = _core.create_backed_operator(
            file_path=h5_path, group_path="/X", chunk_size=32,
            n_threads=1,
        )
        result = _core.compute_feature_stats_vision_backed_operator(
            op, G, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )
        assert result.shape == (n_cells, n_labels)

    def test_no_nan_in_output(self, tmp_path):
        n_cells, n_genes, n_labels = 40, 20, 3
        rng = np.random.default_rng(101)
        S = sp.random(n_cells, n_genes, density=0.2,
                      random_state=rng, format="csc", dtype=np.float64)
        G = _make_sym_graph(n_cells)
        X = sp.random(n_genes, n_labels, density=0.3,
                      random_state=rng, format="csc", dtype=np.float64)

        h5_path = str(tmp_path / "test_nan.h5ad")
        _write_sparse_h5ad(h5_path, S, fmt="csc")

        op = _core.create_backed_operator(
            file_path=h5_path, group_path="/X", chunk_size=32,
            n_threads=1,
        )
        result = _core.compute_feature_stats_vision_backed_operator(
            op, G, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )
        assert not np.any(np.isnan(result))
