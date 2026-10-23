"""Tests for auto-resolved HNSW parameters in build_network.

``M``, ``ef_construction``, and ``ef`` default to ``None`` and are resolved in
libactionet:

- ``M``: 16.
- ``knn``: ``ef_construction = max(200, 4k)``, ``ef = max(200, 5k)``.
- ``k*nn``: both ``max(200, kNN)``; explicit values are floored at ``kNN``.

Graph equality is checked with ``threads=1`` so HNSW insertion order (and
therefore the index) is deterministic.
"""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import actionet as an
import actionet._core as _core


def _make_embedding(n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = rng.random((n, k)).astype(np.float32)
    H /= H.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(H, dtype=np.float32)


def _core_build(H, algorithm, *, k=10, M=None, efc=None, ef=None):
    G = _core.build_network(
        H, algorithm=algorithm, distance_metric="jsd", density=1.0, thread_no=1,
        M=M, ef_construction=efc, ef=ef, mutual_edges_only=True, k=k,
    )
    return sparse.csr_matrix(G)


def _assert_same_graph(G1, G2):
    np.testing.assert_array_equal(G1.indptr, G2.indptr)
    np.testing.assert_array_equal(G1.indices, G2.indices)
    np.testing.assert_allclose(G1.data, G2.data, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Default resolution parity
# ---------------------------------------------------------------------------
class TestAutoResolution:
    @pytest.mark.parametrize("algorithm", ["knn", "k*nn"])
    def test_auto_matches_legacy_defaults(self, algorithm):
        """At k=10 (knn) and small n (k*nn), auto must equal the former
        (16, 200, 200) defaults."""
        H = _make_embedding(1000, 12, seed=1)
        G_auto = _core_build(H, algorithm, k=10)
        G_legacy = _core_build(H, algorithm, k=10, M=16, efc=200, ef=200)
        _assert_same_graph(G_auto, G_legacy)

    def test_knn_large_k_uses_scaled_ef(self):
        H = _make_embedding(2000, 12, seed=2)
        G_auto = _core_build(H, "knn", k=100)
        G_explicit = _core_build(H, "knn", k=100, M=16, efc=400, ef=500)
        _assert_same_graph(G_auto, G_explicit)

    def test_kstar_explicit_values_floored_at_knn(self):
        """n=1000 gives kNN = 5 * round(sqrt(1000)) = 160; explicit values
        below it are raised to kNN, so they match an explicit kNN request."""
        H = _make_embedding(1000, 12, seed=3)
        G_low = _core_build(H, "k*nn", efc=10, ef=10)
        G_floor = _core_build(H, "k*nn", efc=160, ef=160)
        _assert_same_graph(G_low, G_floor)

    def test_knn_explicit_values_honored(self):
        """knn passes explicit values through; a tiny ef_construction builds a
        different index than the auto value."""
        H = _make_embedding(2000, 12, seed=4)
        G_auto = _core_build(H, "knn", k=30)
        G_low = _core_build(H, "knn", k=30, efc=4, ef=4)
        assert (G_auto != G_low).nnz > 0

    def test_public_api_auto_matches_core(self):
        H = _make_embedding(800, 10, seed=5)
        adata = AnnData(X=np.zeros((H.shape[0], 1), dtype=np.float32))
        adata.obsm["H_stacked"] = H
        an.build_network(adata, algorithm="knn", k=50, n_threads=1)
        G_core = _core_build(H, "knn", k=50, M=16, efc=200, ef=250)
        _assert_same_graph(sparse.csr_matrix(adata.obsp["actionet"]), G_core)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.fixture
    def adata(self):
        H = _make_embedding(100, 6, seed=6)
        adata = AnnData(X=np.zeros((H.shape[0], 1), dtype=np.float32))
        adata.obsm["H_stacked"] = H
        return adata

    @pytest.mark.parametrize("name", ["M", "ef_construction", "ef"])
    @pytest.mark.parametrize("value", [0, -5, True, 12.5, "16"])
    def test_python_rejects_invalid(self, adata, name, value):
        with pytest.raises(ValueError, match=name):
            an.build_network(adata, **{name: value})

    def test_python_accepts_integral_float(self, adata):
        an.build_network(adata, algorithm="knn", M=16.0, ef_construction=200.0, ef=200.0)
        assert "actionet" in adata.obsp

    def test_core_accepts_none(self):
        H = _make_embedding(100, 6, seed=7)
        G = _core.build_network(H, "knn", "jsd", 1.0, 1, None, None, None, True, 5)
        assert G.shape == (100, 100)

    def test_core_rejects_m_below_two(self):
        H = _make_embedding(100, 6, seed=8)
        with pytest.raises(RuntimeError, match="M must be at least 2"):
            _core_build(H, "knn", M=1)

    @pytest.mark.parametrize("name", ["M", "efc", "ef"])
    def test_core_rejects_negative(self, name):
        H = _make_embedding(100, 6, seed=9)
        with pytest.raises(RuntimeError):
            _core_build(H, "knn", **{name: -1})
