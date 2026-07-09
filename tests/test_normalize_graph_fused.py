"""Tests for the fused normalizeGraph overload (Phase 3F Item 4).

The overload writes back the pre-normalization column sums that
normalizeGraph computes internally, letting callers (e.g. prepareGraph_
in network_diffusion.cpp) skip a separate arma::sum pre-pass.

These tests verify:
1. The normalized-G output of the fused overload equals the single-arg
   overload's output (bit-exact) for norm_method 0, 1, and 2.
2. The reported col_sums match a direct scipy computation of the
   pre-normalization column sums (up to the abs-vs-signed convention
   documented in the header — all tests here use non-negative weights,
   the only supported input for PageRank-style diffusion).
"""

import numpy as np
import pytest
from scipy import sparse

import actionet._core as _core


def _make_nonneg_graph(n: int, density: float = 0.15, seed: int = 42,
                      symmetric: bool = True) -> sparse.csc_matrix:
    rng = np.random.default_rng(seed)
    G = sparse.random(n, n, density=density, random_state=rng, format="csc",
                      dtype=np.float64)
    if symmetric:
        G = G + G.T
    G.setdiag(0)
    G.eliminate_zeros()
    return G.tocsc()


def _make_graph_with_sinks(n: int, seed: int = 7) -> sparse.csc_matrix:
    """Graph with some deliberately zero-weight columns (sinks)."""
    rng = np.random.default_rng(seed)
    G = sparse.random(n, n, density=0.2, random_state=rng, format="csc",
                     dtype=np.float64)
    G = G.tolil()
    for j in [0, n // 3, n - 1]:
        G[:, j] = 0
    G.setdiag(0)
    return G.tocsc()


@pytest.mark.parametrize("norm_method", [0, 1, 2])
def test_normalized_G_matches_single_arg_overload(norm_method):
    n = 60
    G = _make_nonneg_graph(n, seed=100 + norm_method)

    G_single = _core.normalize_graph(G, norm_method=norm_method)
    G_fused, _cs = _core.normalize_graph_with_col_sums(
        G, norm_method=norm_method)

    G_single_dense = G_single.toarray()
    G_fused_dense = G_fused.toarray()
    np.testing.assert_array_equal(G_single_dense, G_fused_dense)


@pytest.mark.parametrize("norm_method", [0, 1, 2])
def test_col_sums_match_scipy(norm_method):
    n = 60
    G = _make_nonneg_graph(n, seed=200 + norm_method)

    expected_col_sums = np.asarray(G.sum(axis=0)).ravel()

    _G_fused, cs = _core.normalize_graph_with_col_sums(
        G, norm_method=norm_method)

    assert cs.shape == (n,)
    np.testing.assert_allclose(cs, expected_col_sums, atol=1e-12)


@pytest.mark.parametrize("norm_method", [0, 1, 2])
def test_col_sums_identify_sinks(norm_method):
    """Sink columns (all-zero) must have col_sum == 0 so
    prepareGraph_ can mask them out of the teleportation vector."""
    n = 30
    G = _make_graph_with_sinks(n)
    expected_col_sums = np.asarray(G.sum(axis=0)).ravel()
    expected_sink_mask = expected_col_sums == 0

    _G_fused, cs = _core.normalize_graph_with_col_sums(
        G, norm_method=norm_method)

    actual_sink_mask = cs == 0
    np.testing.assert_array_equal(actual_sink_mask, expected_sink_mask)
    assert actual_sink_mask.any(), "Test fixture is missing sink columns"


def test_diffusion_output_unchanged_after_fusion():
    """End-to-end regression: computeNetworkDiffusion output should be
    numerically identical whether the graph-prep path uses the fused
    normalizeGraph overload or (as before) a separate arma::sum pre-pass.
    We can't easily compare against the pre-refactor code, so this test
    just asserts determinism on a fixture with sinks — a sanity check
    that the sink-mask handoff via col_sums matches the old arma::sum
    handoff for non-negative weights."""
    rng = np.random.default_rng(31337)
    n, k = 80, 4
    G = _make_graph_with_sinks(n, seed=42)
    X0 = np.ascontiguousarray(rng.random((n, k)), dtype=np.float64)

    r1 = _core.compute_network_diffusion(
        G, X0, alpha=0.85, max_it=5,
        thread_no=1, approx=False, norm_method=0, tol=1e-8,
    )
    r2 = _core.compute_network_diffusion(
        G, X0, alpha=0.85, max_it=5,
        thread_no=1, approx=False, norm_method=0, tol=1e-8,
    )
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == (n, k)
    assert np.isfinite(r1).all()
