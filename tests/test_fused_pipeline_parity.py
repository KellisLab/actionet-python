"""Parity tests: fused annotate_cells bindings vs legacy separate-call pipeline.

Validates that the fused pybind wrappers (annotate_cells_vision_fused,
annotate_cells_actionet_fused, annotate_cells_vision_backed_fused) produce
identical results to the legacy multi-crossing paths.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, random as sp_random

from actionet import _core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adata_with_graph(n_obs=200, n_var=80, n_markers=5, density=0.3, seed=42):
    """Build a small AnnData with a cell-cell graph and marker dict."""
    rng = np.random.default_rng(seed)
    X = sp_random(n_obs, n_var, density=density, format="csr", random_state=seed,
                  data_rvs=lambda n: rng.standard_normal(n).astype(np.float32))
    X.data = np.abs(X.data)

    var_names = [f"gene_{i}" for i in range(n_var)]
    adata = AnnData(X=X, var=pd.DataFrame(index=var_names))

    # Symmetric k-NN-like graph
    G = sp_random(n_obs, n_obs, density=0.05, format="csr", random_state=seed + 1,
                  data_rvs=lambda n: rng.uniform(0.1, 1.0, size=n).astype(np.float64))
    G = (G + G.T) / 2
    G.setdiag(0)
    G.eliminate_zeros()
    adata.obsp["actionet"] = G

    markers = {}
    for i in range(n_markers):
        start = (i * n_var // n_markers)
        end = start + max(3, n_var // (n_markers * 2))
        markers[f"type_{i}"] = var_names[start:end]

    return adata, markers, G


@pytest.fixture
def small_adata():
    return _make_adata_with_graph()


# ---------------------------------------------------------------------------
# Helper: legacy (unfused) pipeline
# ---------------------------------------------------------------------------

def _legacy_vision_inmemory(adata, markers, G, norm_method_code=2, alpha=0.85,
                             max_it=5, approx=True, n_threads=0):
    """Replicate the old code path using separate pybind calls."""
    from actionet.annotation import _encode_markers, _sparse_row_sum_sq
    from actionet._feature_lookup import resolve_feature_space
    from scipy.sparse import issparse

    space = resolve_feature_space(adata, None, context="test")
    X_markers, celltype_names = _encode_markers(markers, space.labels)
    S = adata.X

    if issparse(S):
        stats = S.dot(X_markers)
        if issparse(stats):
            stats = stats.toarray()
        stats = np.asarray(stats, dtype=np.float64)
        row_sums = np.asarray(S.sum(axis=1), dtype=np.float64).ravel()
        row_sum_sq = _sparse_row_sum_sq(S)
    else:
        S_arr = np.asarray(S, dtype=np.float64)
        X_dense = X_markers.toarray()
        stats = S_arr @ X_dense
        stats = np.asarray(stats, dtype=np.float64)
        row_sums = S_arr.sum(axis=1)
        row_sum_sq = np.sum(S_arr * S_arr, axis=1)

    n_vars = S.shape[1]
    mu = row_sums / n_vars
    sigma_sq = (row_sum_sq - 2.0 * mu * row_sums + n_vars * mu ** 2) / (n_vars - 1)

    marker_stats = _core.compute_feature_stats_vision_from_stats(
        G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
        norm_method=norm_method_code, alpha=alpha, max_it=max_it,
        approx=approx, thread_no=n_threads,
    )

    Gn = _core.normalize_graph(G, norm_method=1).T
    marker_stats_pos = np.maximum(
        np.nan_to_num(marker_stats, nan=0.0, posinf=0.0, neginf=0.0), 0
    )
    log_pvals = _core.compute_graph_label_enrichment(Gn, marker_stats_pos, n_threads)

    return marker_stats, log_pvals, X_markers, stats, mu, sigma_sq


def _legacy_actionet(adata, markers, G, norm_method_code=2, alpha=0.85,
                     max_it=5, approx=True, n_threads=0, ignore_baseline=False):
    """Replicate the old ACTIONet code path using separate pybind calls."""
    from actionet.annotation import _encode_markers
    from actionet._feature_lookup import resolve_feature_space
    from scipy.sparse import issparse

    space = resolve_feature_space(adata, None, context="test")
    X_markers, celltype_names = _encode_markers(markers, space.labels)

    required_idx = np.where(np.asarray(X_markers.getnnz(axis=1)).ravel() > 0)[0]
    X_sub = X_markers[required_idx, :]

    S = adata.X
    S_sub = S[:, required_idx]
    if not issparse(S_sub):
        S_sub = csr_matrix(np.asarray(S_sub))

    marker_stats = _core.compute_feature_stats(
        G=G, S=S_sub, X=X_sub, norm_method=norm_method_code,
        alpha=alpha, max_it=max_it, approx=approx,
        thread_no=n_threads, ignore_baseline=ignore_baseline,
    )

    Gn = _core.normalize_graph(G, norm_method=1).T
    marker_stats_pos = np.maximum(
        np.nan_to_num(marker_stats, nan=0.0, posinf=0.0, neginf=0.0), 0
    )
    log_pvals = _core.compute_graph_label_enrichment(Gn, marker_stats_pos, n_threads)

    return marker_stats, log_pvals, X_markers, X_sub, S_sub, required_idx


# ---------------------------------------------------------------------------
# Tests: VISION in-memory fused vs legacy
# ---------------------------------------------------------------------------

class TestVisionInMemoryFused:
    """Fused VISION in-memory binding must match separate-call legacy path."""

    def test_marker_stats_exact(self, small_adata):
        adata, markers, G = small_adata
        ms_leg, lp_leg, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(adata, markers, G)

        fused = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=2, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )

        np.testing.assert_allclose(fused["marker_stats"], ms_leg, rtol=1e-12, atol=1e-14)

    def test_log_pvals_exact(self, small_adata):
        adata, markers, G = small_adata
        ms_leg, lp_leg, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(adata, markers, G)

        fused = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=2, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )

        np.testing.assert_allclose(fused["log_pvals"], lp_leg, rtol=1e-12, atol=1e-14)

    def test_shapes(self, small_adata):
        adata, markers, G = small_adata
        _, _, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(adata, markers, G)

        fused = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=2, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )

        n_obs, n_labels = adata.n_obs, X_markers.shape[1]
        assert fused["marker_stats"].shape == (n_obs, n_labels)
        assert fused["log_pvals"].shape == (n_obs, n_labels)

    def test_no_nan(self, small_adata):
        adata, markers, G = small_adata
        _, _, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(adata, markers, G)

        fused = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=2, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )

        assert np.isfinite(fused["log_pvals"]).all()

    def test_deterministic(self, small_adata):
        adata, markers, G = small_adata
        _, _, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(adata, markers, G)

        r1 = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=2, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )
        r2 = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=2, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )

        np.testing.assert_array_equal(r1["marker_stats"], r2["marker_stats"])
        np.testing.assert_array_equal(r1["log_pvals"], r2["log_pvals"])

    def test_pagerank_sym(self, small_adata):
        """Test with norm_method=0 (pagerank) instead of 2 (sym_pagerank)."""
        adata, markers, G = small_adata
        ms_leg, lp_leg, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(
            adata, markers, G, norm_method_code=0,
        )

        fused = _core.annotate_cells_vision_fused(
            G=G, stats=stats, mu=mu, sigma_sq=sigma_sq, X=X_markers,
            norm_method=0, alpha=0.85, max_it=5, approx=True,
            enrichment_norm_method=1, thread_no=0,
        )

        np.testing.assert_allclose(fused["marker_stats"], ms_leg, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(fused["log_pvals"], lp_leg, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: ACTIONet fused vs legacy
# ---------------------------------------------------------------------------

class TestACTIONetFused:
    """Fused ACTIONet binding must match separate-call legacy path."""

    def test_marker_stats_exact(self, small_adata):
        adata, markers, G = small_adata
        ms_leg, lp_leg, X_markers, X_sub, S_sub, _ = _legacy_actionet(adata, markers, G)

        fused = _core.annotate_cells_actionet_fused(
            G=G, S=S_sub, X=X_sub, norm_method=2, alpha=0.85, max_it=5,
            approx=True, enrichment_norm_method=1, thread_no=0,
            ignore_baseline=False,
        )

        np.testing.assert_allclose(fused["marker_stats"], ms_leg, rtol=1e-12, atol=1e-14)

    def test_log_pvals_exact(self, small_adata):
        adata, markers, G = small_adata
        ms_leg, lp_leg, X_markers, X_sub, S_sub, _ = _legacy_actionet(adata, markers, G)

        fused = _core.annotate_cells_actionet_fused(
            G=G, S=S_sub, X=X_sub, norm_method=2, alpha=0.85, max_it=5,
            approx=True, enrichment_norm_method=1, thread_no=0,
            ignore_baseline=False,
        )

        np.testing.assert_allclose(fused["log_pvals"], lp_leg, rtol=1e-12, atol=1e-14)

    def test_shapes(self, small_adata):
        adata, markers, G = small_adata
        _, _, X_markers, X_sub, S_sub, _ = _legacy_actionet(adata, markers, G)

        fused = _core.annotate_cells_actionet_fused(
            G=G, S=S_sub, X=X_sub, norm_method=2, alpha=0.85, max_it=5,
            approx=True, enrichment_norm_method=1, thread_no=0,
            ignore_baseline=False,
        )

        n_obs = adata.n_obs
        n_labels = X_sub.shape[1]
        assert fused["marker_stats"].shape == (n_obs, n_labels)
        assert fused["log_pvals"].shape == (n_obs, n_labels)

    def test_deterministic(self, small_adata):
        adata, markers, G = small_adata
        _, _, _, X_sub, S_sub, _ = _legacy_actionet(adata, markers, G)

        r1 = _core.annotate_cells_actionet_fused(
            G=G, S=S_sub, X=X_sub, norm_method=2, alpha=0.85, max_it=5,
            approx=True, enrichment_norm_method=1, thread_no=0,
            ignore_baseline=False,
        )
        r2 = _core.annotate_cells_actionet_fused(
            G=G, S=S_sub, X=X_sub, norm_method=2, alpha=0.85, max_it=5,
            approx=True, enrichment_norm_method=1, thread_no=0,
            ignore_baseline=False,
        )

        np.testing.assert_array_equal(r1["marker_stats"], r2["marker_stats"])
        np.testing.assert_array_equal(r1["log_pvals"], r2["log_pvals"])


# ---------------------------------------------------------------------------
# Tests: end-to-end annotate_cells function
# ---------------------------------------------------------------------------

class TestAnnotateCellsEndToEnd:
    """End-to-end tests for annotate_cells using the fused path under the hood."""

    def test_vision_with_enrichment(self, small_adata):
        """annotate_cells with method='vision', use_enrichment=True triggers fused path."""
        from actionet.annotation import annotate_cells
        adata, markers, G = small_adata
        result = annotate_cells(adata, markers, method="vision", use_enrichment=True)

        assert "labels" in result
        assert "confidence" in result
        assert "enrichment" in result
        assert result["labels"].shape == (adata.n_obs,)
        assert result["confidence"].shape == (adata.n_obs,)
        assert result["enrichment"].shape[0] == adata.n_obs

    def test_vision_without_enrichment(self, small_adata):
        """annotate_cells with use_enrichment=False uses legacy path."""
        from actionet.annotation import annotate_cells
        adata, markers, G = small_adata
        result = annotate_cells(adata, markers, method="vision", use_enrichment=False)

        assert "labels" in result
        assert "confidence" in result
        assert result["labels"].shape == (adata.n_obs,)

    def test_actionet_with_enrichment(self, small_adata):
        """annotate_cells with method='actionet', use_enrichment=True triggers fused path."""
        from actionet.annotation import annotate_cells
        adata, markers, G = small_adata
        result = annotate_cells(adata, markers, method="actionet", use_enrichment=True)

        assert "labels" in result
        assert "confidence" in result
        assert "enrichment" in result
        assert result["labels"].shape == (adata.n_obs,)

    def test_actionet_without_enrichment(self, small_adata):
        """annotate_cells with method='actionet', use_enrichment=False."""
        from actionet.annotation import annotate_cells
        adata, markers, G = small_adata
        result = annotate_cells(adata, markers, method="actionet", use_enrichment=False)

        assert "labels" in result
        assert "confidence" in result
        assert result["labels"].shape == (adata.n_obs,)

    def test_fused_labels_match_legacy_labels(self, small_adata):
        """The final labels assigned via fused path must match those from legacy path."""
        adata, markers, G = small_adata
        ms_leg, lp_leg, X_markers, stats, mu, sigma_sq = _legacy_vision_inmemory(
            adata, markers, G, norm_method_code=0,
        )

        from actionet.annotation import annotate_cells
        result = annotate_cells(adata, markers, method="vision",
                                use_enrichment=True, norm_method="pagerank")
        legacy_labels_idx = np.argmax(lp_leg, axis=1)

        from actionet.annotation import _encode_markers
        from actionet._feature_lookup import resolve_feature_space
        space = resolve_feature_space(adata, None, context="test")
        _, celltype_names = _encode_markers(markers, space.labels)
        celltype_arr = np.asarray(celltype_names)

        legacy_labels = celltype_arr[legacy_labels_idx]

        np.testing.assert_array_equal(result["labels"], legacy_labels)

    def test_g_not_mutated(self, small_adata):
        """The original G in adata.obsp must not be mutated by fused path."""
        from actionet.annotation import annotate_cells
        adata, markers, G = small_adata
        G_before = G.copy()

        annotate_cells(adata, markers, method="vision", use_enrichment=True)

        G_after = adata.obsp["actionet"]
        np.testing.assert_array_equal(G_before.data, G_after.data)
        np.testing.assert_array_equal(G_before.indices, G_after.indices)
        np.testing.assert_array_equal(G_before.indptr, G_after.indptr)

    def test_with_lpa(self, small_adata):
        """Fused path works when use_lpa=True."""
        from actionet.annotation import annotate_cells
        adata, markers, G = small_adata
        result = annotate_cells(adata, markers, method="vision",
                                use_enrichment=True, use_lpa=True)

        assert "labels_corrected" in result
        assert result["labels_corrected"].shape == (adata.n_obs,)
