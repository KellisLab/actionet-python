"""Parity tests for the rewritten computeGraphLabelEnrichment (L1).

Verifies the fused single-NNZ-pass + in-place Bennett implementation
produces identical results to a pure-Python reference implementation
of the original Armadillo-based code.
"""

import numpy as np
import pytest
import scipy.sparse as sp

try:
    from actionet import _core
    _has_ext = True
except Exception:
    _has_ext = False

requires_ext = pytest.mark.skipif(not _has_ext, reason="C extension not built")

RTOL = 1e-8
ATOL = 1e-10


def _reference_enrichment(G_dense: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Pure-Python reference matching the original Armadillo implementation."""
    Obs = G_dense @ scores
    p = scores.mean(axis=0, keepdims=True)                   # (1, n_labels)
    row_sum = G_dense.sum(axis=1, keepdims=True)              # (n, 1)
    Exp = row_sum @ p                                         # (n, n_labels)
    Lambda = Obs - Exp

    row_sum_sq = (G_dense ** 2).sum(axis=1, keepdims=True)    # (n, 1)
    Nu = row_sum_sq @ p                                       # (n, n_labels)
    a = G_dense.max(axis=1, keepdims=True)                    # (n, 1)
    Lambda_scaled = Lambda * (a / 3.0)

    denom = 2.0 * (Nu + Lambda_scaled)
    logPvals = np.where(
        (Lambda > 0) & (denom > 0),
        Lambda ** 2 / denom,
        0.0,
    )
    logPvals = np.nan_to_num(logPvals, nan=0.0)
    return logPvals


def _make_graph_and_scores(n, n_labels, density=0.05, seed=42):
    """Create a random sparse non-negative graph and random scores."""
    rng = np.random.default_rng(seed)
    nnz = int(n * n * density)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    data = rng.exponential(1.0, size=nnz)
    G = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
    G.sum_duplicates()

    scores = rng.random((n, n_labels))
    scores = np.asarray(scores, dtype=np.float64)
    return G, scores


@requires_ext
class TestEnrichmentParity:
    """C++ result matches the pure-Python reference implementation."""

    @pytest.fixture(params=[
        (100, 5, 0.1),
        (500, 20, 0.03),
        (200, 50, 0.05),
    ], ids=["100x5", "500x20", "200x50"])
    def graph_and_scores(self, request):
        n, n_labels, density = request.param
        return _make_graph_and_scores(n, n_labels, density)

    def test_parity(self, graph_and_scores):
        G, scores = graph_and_scores
        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        expected = _reference_enrichment(G.toarray(), scores)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_shape(self, graph_and_scores):
        G, scores = graph_and_scores
        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        assert result.shape == scores.shape

    def test_non_negative(self, graph_and_scores):
        G, scores = graph_and_scores
        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        assert np.all(result >= 0.0)

    def test_no_nan(self, graph_and_scores):
        G, scores = graph_and_scores
        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        assert not np.any(np.isnan(result))


@requires_ext
class TestEnrichmentEdgeCases:

    def test_identity_graph(self):
        """Identity graph: each cell only sees itself."""
        n, n_labels = 50, 3
        G = sp.eye(n, format="csc")
        rng = np.random.default_rng(7)
        scores = rng.random((n, n_labels)).astype(np.float64)

        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        expected = _reference_enrichment(G.toarray(), scores)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_uniform_scores(self):
        """When all scores are identical, enrichment should be zero."""
        n = 100
        G, _ = _make_graph_and_scores(n, 1, density=0.05, seed=11)
        scores = np.ones((n, 3), dtype=np.float64)

        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        assert np.allclose(result, 0.0, atol=1e-12)

    def test_single_label(self):
        n = 80
        G, scores = _make_graph_and_scores(n, 1, density=0.08, seed=99)
        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        expected = _reference_enrichment(G.toarray(), scores)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_empty_graph(self):
        """Graph with no edges."""
        n, n_labels = 30, 4
        G = sp.csc_matrix((n, n))
        scores = np.random.default_rng(3).random((n, n_labels)).astype(np.float64)

        result = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        assert np.allclose(result, 0.0, atol=1e-12)

    def test_csr_input(self):
        """CSR input should produce the same result as CSC."""
        G, scores = _make_graph_and_scores(100, 5, density=0.1, seed=77)
        G_csr = G.tocsr()
        result_csc = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        result_csr = _core.compute_graph_label_enrichment(G_csr, scores, thread_no=1)
        np.testing.assert_allclose(result_csr, result_csc, rtol=RTOL, atol=ATOL)


@requires_ext
class TestEnrichmentThreadSafety:
    """Multi-threaded results match single-threaded."""

    def test_multithreaded_parity(self):
        G, scores = _make_graph_and_scores(200, 10, density=0.05, seed=55)
        result_1t = _core.compute_graph_label_enrichment(G, scores, thread_no=1)
        result_mt = _core.compute_graph_label_enrichment(G, scores, thread_no=4)
        np.testing.assert_allclose(result_mt, result_1t, rtol=RTOL, atol=ATOL)
