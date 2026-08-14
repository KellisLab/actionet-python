"""Tests for build_network correctness invariants.

These protect the network construction rewrite:

- The JSD kernel rewrite (LUT removal + vectorizable fasterlog2) must keep
  producing a valid, symmetric similarity graph with sane weights.
- The parallel symmetrization (parallel_sort + two-pass CSR) must preserve the
  output contract: symmetric adjacency, sorted CSR column indices, zero
  diagonal, and mutual-edges-only semantics.
- Results must be deterministic across repeated calls and stable across thread
  counts (topology-level), and independent of the JSD lookup path that used to
  depend on a 4 MB table.
"""

import numpy as np
import pytest
from scipy import sparse

import actionet._core as _core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_embedding(n: int, k: int, seed: int = 0, simplex: bool = True) -> np.ndarray:
    """Row-major float32 embedding.  When ``simplex`` (the JSD-appropriate
    case), each row is non-negative and L1-normalized to sum to 1."""
    rng = np.random.default_rng(seed)
    H = rng.random((n, k)).astype(np.float32)
    if simplex:
        H = np.abs(H)
        H /= H.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(H, dtype=np.float32)


def _build(H, algorithm="knn", metric="jsd", *, k=10, threads=1,
           mutual=True, M=16.0, efc=200.0, ef=200.0, density=1.0):
    G = _core.build_network(H, algorithm, metric, density, threads,
                            M, efc, ef, mutual, k)
    return sparse.csr_matrix(G)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
class TestGraphStructure:
    @pytest.mark.parametrize("algorithm", ["knn", "k*nn"])
    @pytest.mark.parametrize("metric", ["jsd", "l2", "ip"])
    def test_symmetric_zero_diagonal(self, algorithm, metric):
        H = _make_embedding(400, 12, seed=1)
        G = _build(H, algorithm, metric, k=15, threads=4)
        # Symmetric adjacency.
        diff = (G - G.T)
        assert abs(diff).max() < 1e-5, "graph is not symmetric"
        # No self loops.
        assert G.diagonal().sum() == 0.0, "graph has nonzero diagonal"

    @pytest.mark.parametrize("algorithm", ["knn", "k*nn"])
    def test_sorted_csr_indices(self, algorithm):
        """The two-pass CSR builder promises per-row sorted column indices
        without a post-fill sort; verify that contract holds."""
        H = _make_embedding(500, 10, seed=2)
        G = _build(H, algorithm, "jsd", k=12, threads=4)
        assert G.has_sorted_indices or _indices_sorted(G), \
            "CSR column indices are not sorted within rows"

    def test_weights_positive_and_finite(self):
        H = _make_embedding(300, 8, seed=3)
        G = _build(H, "knn", "jsd", k=10, threads=4)
        assert np.all(np.isfinite(G.data))
        assert np.all(G.data > 0.0), "similarity weights must be positive"


def _indices_sorted(G: sparse.csr_matrix) -> bool:
    for r in range(G.shape[0]):
        seg = G.indices[G.indptr[r]:G.indptr[r + 1]]
        if seg.size > 1 and np.any(np.diff(seg) <= 0):
            return False
    return True


# ---------------------------------------------------------------------------
# Determinism and thread-invariance
# ---------------------------------------------------------------------------
class TestDeterminism:
    @pytest.mark.parametrize("algorithm", ["knn", "k*nn"])
    def test_repeated_calls_identical(self, algorithm):
        H = _make_embedding(400, 10, seed=4)
        G1 = _build(H, algorithm, "jsd", k=12, threads=1)
        G2 = _build(H, algorithm, "jsd", k=12, threads=1)
        np.testing.assert_array_equal(G1.indptr, G2.indptr)
        np.testing.assert_array_equal(G1.indices, G2.indices)
        np.testing.assert_allclose(G1.data, G2.data, rtol=1e-6, atol=1e-7)

    def test_thread_count_topology_stable(self):
        """HNSW insertion order varies with threads, so exact edges may differ
        slightly, but the bulk of the graph must be stable (high Jaccard)."""
        H = _make_embedding(800, 12, seed=5)
        G1 = _build(H, "knn", "jsd", k=15, threads=1)
        G8 = _build(H, "knn", "jsd", k=15, threads=8)
        e1 = set(zip(*G1.nonzero()))
        e8 = set(zip(*G8.nonzero()))
        jac = len(e1 & e8) / len(e1 | e8)
        assert jac > 0.9, f"thread topology diverged: jaccard={jac:.3f}"


# ---------------------------------------------------------------------------
# Mutual-edges semantics
# ---------------------------------------------------------------------------
class TestMutualEdges:
    def test_mutual_subset_of_union(self):
        """The mutual-only graph must be a subset of the non-mutual graph."""
        H = _make_embedding(500, 10, seed=6)
        G_mut = _build(H, "knn", "jsd", k=12, threads=4, mutual=True)
        G_all = _build(H, "knn", "jsd", k=12, threads=4, mutual=False)
        e_mut = set(zip(*G_mut.nonzero()))
        e_all = set(zip(*G_all.nonzero()))
        # Allow a small slack for HNSW nondeterminism between the two runs.
        leaked = len(e_mut - e_all) / max(1, len(e_mut))
        assert leaked < 0.05, f"mutual edges not ~subset of union: leaked={leaked:.3f}"


# ---------------------------------------------------------------------------
# JSD kernel sanity (LUT-removal regression guard)
# ---------------------------------------------------------------------------
class TestJSDKernel:
    def test_identical_rows_are_closest(self):
        """Duplicated rows must be near-neighbors with near-maximal similarity;
        catches gross errors in the JSD distance rewrite."""
        base = _make_embedding(200, 10, seed=7)
        # Duplicate the first row into the second slot.
        H = base.copy()
        H[1] = H[0]
        G = _build(H, "knn", "jsd", k=10, threads=1)
        # Row 0 and row 1 are identical -> should be mutual neighbors.
        assert G[0, 1] > 0.0 and G[1, 0] > 0.0, \
            "identical rows are not connected in the JSD graph"

    def test_jsd_weights_bounded(self):
        """JSD similarity = max(eps, 1 - d); with d in [0,1], weights are in
        (0, 1]. Verify the rewrite keeps weights in range."""
        H = _make_embedding(400, 12, seed=8)
        G = _build(H, "knn", "jsd", k=12, threads=4)
        assert G.data.min() > 0.0
        assert G.data.max() <= 1.0 + 1e-5, f"max weight {G.data.max()} exceeds 1"
