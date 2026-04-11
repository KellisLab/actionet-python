"""Tests for Step 3 (L2+L3): non-mutating diffusion and loop determinism.

These tests verify that:
1. computeNetworkDiffusion never modifies the input graph G.
2. Repeated calls with the same G produce identical results.
3. The VISION and ACTIONet paths see the original G even when called
   multiple times (the "computeFeatureStats loop bug" regression test).
"""

import numpy as np
import pytest
from scipy import sparse

import actionet._core as _core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sym_graph(n: int, density: float = 0.1, seed: int = 42) -> sparse.csc_matrix:
    """Create a symmetric non-negative sparse graph."""
    rng = np.random.default_rng(seed)
    G = sparse.random(n, n, density=density, random_state=rng, format="csc",
                      dtype=np.float64)
    G = G + G.T
    G.setdiag(0)
    G.eliminate_zeros()
    return G


def _make_scores(n: int, k: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((n, k)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Test: G is not modified by compute_network_diffusion
# ---------------------------------------------------------------------------

class TestDiffusionNonMutation:
    @pytest.fixture(params=[
        dict(approx=False, label="power_iter"),
        dict(approx=True, label="chebyshev"),
    ], ids=lambda d: d["label"])
    def diffusion_params(self, request):
        return request.param

    def test_graph_unchanged_after_diffusion(self, diffusion_params):
        n, k = 100, 5
        G = _make_sym_graph(n)
        X0 = _make_scores(n, k)

        G_data_before = G.data.copy()
        G_indices_before = G.indices.copy()
        G_indptr_before = G.indptr.copy()

        _ = _core.compute_network_diffusion(
            G, X0, alpha=0.85, max_it=5,
            thread_no=1, approx=diffusion_params["approx"],
            norm_method=0, tol=1e-8,
        )

        np.testing.assert_array_equal(G.data, G_data_before)
        np.testing.assert_array_equal(G.indices, G_indices_before)
        np.testing.assert_array_equal(G.indptr, G_indptr_before)

    def test_graph_unchanged_with_sym_norm(self, diffusion_params):
        """Sym-pagerank normalization (norm_method=2)."""
        n, k = 80, 3
        G = _make_sym_graph(n)
        X0 = _make_scores(n, k)

        G_data_before = G.data.copy()

        _ = _core.compute_network_diffusion(
            G, X0, alpha=0.85, max_it=5,
            thread_no=1, approx=diffusion_params["approx"],
            norm_method=2, tol=1e-8,
        )

        np.testing.assert_array_equal(G.data, G_data_before)


# ---------------------------------------------------------------------------
# Test: repeated calls produce identical results (determinism)
# ---------------------------------------------------------------------------

class TestDiffusionDeterminism:
    @pytest.fixture(params=[False, True], ids=["power_iter", "chebyshev"])
    def approx(self, request):
        return request.param

    def test_repeated_calls_identical(self, approx):
        n, k = 100, 5
        G = _make_sym_graph(n)
        X0 = _make_scores(n, k)

        results = []
        for _ in range(3):
            r = _core.compute_network_diffusion(
                G, X0, alpha=0.85, max_it=5,
                thread_no=1, approx=approx,
                norm_method=0, tol=1e-8,
            )
            results.append(r)

        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])

    def test_alpha_zero_returns_input(self):
        n, k = 50, 3
        G = _make_sym_graph(n)
        X0 = _make_scores(n, k)

        result = _core.compute_network_diffusion(
            G, X0, alpha=0.0, max_it=5, thread_no=1,
        )
        np.testing.assert_array_equal(result, X0)


# ---------------------------------------------------------------------------
# Test: VISION path (compute_feature_stats_vision) doesn't mutate G
# ---------------------------------------------------------------------------

class TestVisionNonMutation:
    def test_graph_unchanged_after_vision(self):
        n_cells, n_genes, n_labels = 60, 30, 4
        rng = np.random.default_rng(99)
        G = _make_sym_graph(n_cells)
        S = sparse.random(n_cells, n_genes, density=0.2,
                          random_state=rng, format="csc", dtype=np.float64)
        X = sparse.random(n_genes, n_labels, density=0.3,
                          random_state=rng, format="csc", dtype=np.float64)

        G_data_before = G.data.copy()
        G_indices_before = G.indices.copy()
        G_indptr_before = G.indptr.copy()

        _ = _core.compute_feature_stats_vision(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )

        np.testing.assert_array_equal(G.data, G_data_before)
        np.testing.assert_array_equal(G.indices, G_indices_before)
        np.testing.assert_array_equal(G.indptr, G_indptr_before)

    def test_vision_repeated_determinism(self):
        n_cells, n_genes, n_labels = 60, 30, 4
        rng = np.random.default_rng(99)
        G = _make_sym_graph(n_cells)
        S = sparse.random(n_cells, n_genes, density=0.2,
                          random_state=rng, format="csc", dtype=np.float64)
        X = sparse.random(n_genes, n_labels, density=0.3,
                          random_state=rng, format="csc", dtype=np.float64)

        r1 = _core.compute_feature_stats_vision(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )
        r2 = _core.compute_feature_stats_vision(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Test: ACTIONet path (compute_feature_stats) — the loop determinism fix
# ---------------------------------------------------------------------------

class TestACTIONetLoopDeterminism:
    def test_feature_stats_determinism(self):
        """compute_feature_stats calls diffusion in a per-label loop.
        Before the fix, G was mutated on the first iteration, so
        subsequent iterations saw a different graph.  This test
        checks that the result is deterministic across calls."""
        n_cells, n_genes, n_labels = 50, 20, 3
        rng = np.random.default_rng(123)
        G = _make_sym_graph(n_cells)
        S = sparse.random(n_cells, n_genes, density=0.3,
                          random_state=rng, format="csc", dtype=np.float64)
        X = sparse.random(n_genes, n_labels, density=0.4,
                          random_state=rng, format="csc", dtype=np.float64)

        r1 = _core.compute_feature_stats(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )
        r2 = _core.compute_feature_stats(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    def test_graph_unchanged_after_feature_stats(self):
        n_cells, n_genes, n_labels = 50, 20, 3
        rng = np.random.default_rng(123)
        G = _make_sym_graph(n_cells)
        S = sparse.random(n_cells, n_genes, density=0.3,
                          random_state=rng, format="csc", dtype=np.float64)
        X = sparse.random(n_genes, n_labels, density=0.4,
                          random_state=rng, format="csc", dtype=np.float64)

        G_data_before = G.data.copy()

        _ = _core.compute_feature_stats(
            G, S, X, norm_method=2, alpha=0.85, max_it=3,
            approx=False, thread_no=1,
        )

        np.testing.assert_array_equal(G.data, G_data_before)


# ---------------------------------------------------------------------------
# Test: diffusion output sanity — non-negative input stays non-negative
# ---------------------------------------------------------------------------

class TestDiffusionSanity:
    def test_nonneg_input_nonneg_output_chebyshev(self):
        n, k = 100, 5
        G = _make_sym_graph(n)
        X0 = np.abs(_make_scores(n, k))

        result = _core.compute_network_diffusion(
            G, X0, alpha=0.85, max_it=5,
            thread_no=1, approx=True, norm_method=0, tol=1e-8,
        )
        assert np.all(result >= -1e-14), f"Negative values found: min={result.min()}"

    def test_output_shape(self):
        n, k = 100, 7
        G = _make_sym_graph(n)
        X0 = _make_scores(n, k)

        result = _core.compute_network_diffusion(
            G, X0, alpha=0.85, max_it=3, thread_no=1,
        )
        assert result.shape == (n, k)
