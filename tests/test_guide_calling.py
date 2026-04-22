"""Tests for fit-first guide-calling APIs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import anndata as ad

try:
    import actionet as an
    from actionet import _core  # noqa: F401
    import actionet.guide_calling as gc

    _has_ext = True
except Exception:
    _has_ext = False


requires_ext = pytest.mark.skipif(not _has_ext, reason="C extension not built")


def _make_sparse_counts_from_log_mixture(
    n_cells: int,
    mus: tuple[float, float],
    sigma: float,
    w_bg: float,
    seed: int = 0,
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    labels = rng.random(n_cells) < w_bg
    log_vals = np.empty(n_cells, dtype=np.float64)
    log_vals[labels] = rng.normal(mus[0], sigma, size=labels.sum())
    log_vals[~labels] = rng.normal(mus[1], sigma, size=(~labels).sum())
    counts = np.maximum(np.rint(np.power(10.0, log_vals) - 1.0), 0.0)
    rows = np.arange(n_cells, dtype=np.int64)
    mask = counts > 0
    return sp.csr_matrix((counts[mask], (rows[mask], np.zeros(mask.sum(), dtype=np.int64))), shape=(n_cells, 1))


def _small_sparse_matrix(seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    n_cells = 300
    n_guides = 12
    rows = rng.integers(0, n_cells, size=1200)
    cols = rng.integers(0, n_guides, size=1200)
    data = rng.poisson(20, size=1200).astype(np.float64)
    X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_guides)).tocsr()
    X.sum_duplicates()
    X.eliminate_zeros()
    return X


@requires_ext
def test_fit_guides_gmm_recovers_reasonable_means():
    X = _make_sparse_counts_from_log_mixture(
        n_cells=8000,
        mus=(0.5, 1.5),
        sigma=0.22,
        w_bg=0.75,
        seed=7,
    )
    fits = an.fit_guides_gmm(
        X,
        min_points=20,
        min_counts=1,
        n_init=10,
        max_iter=300,
        tol=1e-6,
        seed=11,
        n_threads=1,
    )

    status = int(np.asarray(fits["status"])[0])
    codes = fits["status_codes"]
    assert status == int(codes["ok"])

    means = np.asarray(fits["means"])[0]
    assert means[0] < means[1]
    assert np.isclose(means[0], 0.5, atol=0.20)
    assert np.isclose(means[1], 1.5, atol=0.20)


@requires_ext
def test_quantile_threshold_names_are_stable():
    X = _small_sparse_matrix(seed=12)
    fits = an.fit_guides_gmm(X, min_counts=1, n_init=4, n_threads=1)
    out = an.derive_guide_thresholds(fits, method="quantile", bg_quantile=0.99, fg_quantile=0.01)
    table = out["table"]
    assert "neg_0.99" in table.columns
    assert "pos_0.01" in table.columns


@requires_ext
def test_fit_first_derive_multiple_times_without_refit():
    X = _small_sparse_matrix(seed=18)
    fits = an.fit_guides_gmm(X, min_counts=1, n_init=3, n_threads=1)

    q = an.derive_guide_thresholds(fits, method="quantile", bg_quantile=0.95, fg_quantile=0.05)
    e = an.derive_guide_thresholds(fits, method="equal_density")
    v = an.derive_guide_thresholds(fits, method="valley", valley_grid_size=128)

    assert q["background"].shape == e["background"].shape == v["background"].shape
    assert fits["weights"].shape[0] == q["background"].shape[0]


@requires_ext
def test_explicit_thresholds_bypass_fit_in_auto_mode(monkeypatch):
    X = _small_sparse_matrix(seed=21)
    n_guides = X.shape[1]

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("fit_guides_gmm should not be called for explicit-threshold auto mode")

    monkeypatch.setattr(gc, "_fit_from_resolved", _should_not_be_called)

    res = an.guide_call_gmm(
        X,
        background_thresholds=np.zeros(n_guides, dtype=np.float64),
        foreground_thresholds=np.ones(n_guides, dtype=np.float64),
        threshold_scale="raw",
        result_mode="auto",
    )

    assert "fits" not in res
    assert res["metadata"]["fit_performed"] is False
    assert sp.issparse(res["assignments"]["background"])
    assert sp.issparse(res["assignments"]["foreground"])


@requires_ext
def test_result_mode_behaviour():
    X = _small_sparse_matrix(seed=33)

    res_simple = an.guide_call_gmm(
        X,
        result_mode="simple",
        min_counts=1,
        n_init=3,
        n_threads=1,
    )
    assert "fits" not in res_simple
    assert res_simple["metadata"]["fit_performed"] is True

    res_full = an.guide_call_gmm(
        X,
        result_mode="full",
        min_counts=1,
        n_init=3,
        n_threads=1,
    )
    assert "fits" in res_full
    assert res_full["metadata"]["fit_performed"] is True


@requires_ext
def test_backed_and_in_memory_fit_parity(tmp_path):
    X = _small_sparse_matrix(seed=44).tocsc()
    adata = ad.AnnData(X=X.copy())
    adata.var_names = np.array([f"g{i}" for i in range(adata.n_vars)], dtype=object)

    path = tmp_path / "guide_calling_backed.h5ad"
    adata.write_h5ad(path)
    adata_backed = ad.read_h5ad(path, backed="r+")

    fits_mem = an.fit_guides_gmm(adata, min_counts=1, n_init=3, n_threads=1)
    fits_backed = an.fit_guides_gmm(
        adata_backed,
        min_counts=1,
        n_init=3,
        n_threads=1,
        backed_chunk_guides=4,
    )

    np.testing.assert_allclose(np.asarray(fits_mem["weights"]), np.asarray(fits_backed["weights"]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fits_mem["means"]), np.asarray(fits_backed["means"]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fits_mem["sigma"]), np.asarray(fits_backed["sigma"]), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(fits_mem["status"]), np.asarray(fits_backed["status"]))

    call_mem = an.guide_call_gmm(adata, result_mode="simple", min_counts=1, n_init=3, n_threads=1)
    call_backed = an.guide_call_gmm(
        adata_backed,
        result_mode="simple",
        min_counts=1,
        n_init=3,
        n_threads=1,
        backed_chunk_guides=4,
    )
    np.testing.assert_allclose(
        call_mem["assignments"]["foreground"].toarray(),
        call_backed["assignments"]["foreground"].toarray(),
        atol=0,
    )

    adata_backed.file.close()


@requires_ext
def test_sweep_guide_thresholds():
    X = _small_sparse_matrix(seed=50)
    fits = an.fit_guides_gmm(X, min_counts=1, n_init=3, n_threads=1)
    bg_q = np.array([0.90, 0.95, 0.99])
    fg_q = np.array([0.01, 0.05, 0.10])
    sweep = an.sweep_guide_thresholds(fits, bg_quantiles=bg_q, fg_quantiles=fg_q, return_tables=True)

    assert sweep["background"].shape == (X.shape[1], 3)
    assert sweep["foreground"].shape == (X.shape[1], 3)
    assert "background_table" in sweep
    assert "foreground_table" in sweep
    assert list(sweep["background_table"].columns) == ["neg_0.9", "neg_0.95", "neg_0.99"]
    assert list(sweep["foreground_table"].columns) == ["pos_0.01", "pos_0.05", "pos_0.1"]

    ok = np.asarray(fits["status"]) == fits["status_codes"]["ok"]
    bg_ok = sweep["background"][ok]
    assert np.all(bg_ok[:, 0] <= bg_ok[:, 1]) and np.all(bg_ok[:, 1] <= bg_ok[:, 2])


@requires_ext
def test_valley_and_equal_density_with_assignments():
    X = _small_sparse_matrix(seed=55)
    fits = an.fit_guides_gmm(X, min_counts=1, n_init=3, n_threads=1)

    for method in ("equal_density", "valley"):
        thr = an.derive_guide_thresholds(fits, method=method, output_scale="raw")
        bg = thr["background"]
        fg = thr["foreground"]
        ok = np.asarray(fits["status"]) == fits["status_codes"]["ok"]
        assert np.all(np.isfinite(bg[ok]))
        assert np.all(np.isfinite(fg[ok]))
        assert np.all(np.isnan(bg[~ok]))
        assert np.all(np.isnan(fg[~ok]))

        res = an.guide_call_gmm(
            X, method=method, min_counts=1, n_init=3, n_threads=1, result_mode="simple",
        )
        assert sp.issparse(res["assignments"]["foreground"])
        assert res["assignments"]["foreground"].shape == X.shape


@requires_ext
def test_apply_log10p1_false_path():
    X = _small_sparse_matrix(seed=60)
    fits = an.fit_guides_gmm(X, min_counts=1, n_init=3, n_threads=1, apply_log10p1=False)
    thr = an.derive_guide_thresholds(fits, method="quantile", output_scale="raw")

    ok = np.asarray(fits["status"]) == fits["status_codes"]["ok"]
    means_ok = np.asarray(fits["means"])[ok]
    assert np.all(means_ok > 1.0), "Without log transform, means should be in raw count space"

    res = an.guide_call_gmm(
        X, min_counts=1, n_init=3, n_threads=1, apply_log10p1=False, result_mode="simple",
    )
    assert sp.issparse(res["assignments"]["foreground"])
    assert res["metadata"]["apply_log10p1"] is False


@requires_ext
def test_failed_guides_return_nan_thresholds():
    rng = np.random.default_rng(70)
    n_cells = 200
    n_guides = 4
    data = np.zeros((n_cells, n_guides), dtype=np.float64)
    data[:, 0] = rng.poisson(50, size=n_cells).astype(np.float64)
    data[:, 1] = 5.0  # below min_counts=10
    # cols 2 and 3 are all zeros
    X = sp.csr_matrix(data)

    fits = an.fit_guides_gmm(X, min_counts=10, min_points=5, n_init=3, n_threads=1)
    status = np.asarray(fits["status"])
    codes = fits["status_codes"]

    assert status[1] in (codes["insufficient_points"], codes["degenerate"])
    assert status[2] == codes["insufficient_points"]
    assert status[3] == codes["insufficient_points"]

    thr = an.derive_guide_thresholds(fits, method="quantile", output_scale="transformed")
    assert np.isnan(thr["background"][2])
    assert np.isnan(thr["foreground"][2])
    assert np.isnan(thr["background"][3])
    assert np.isnan(thr["foreground"][3])

    res = an.guide_call_gmm(
        X, min_counts=10, min_points=5, n_init=3, n_threads=1, result_mode="simple",
    )
    fg_assign = res["assignments"]["foreground"]
    assert fg_assign[:, 2].nnz == 0
    assert fg_assign[:, 3].nnz == 0




@requires_ext
def test_backed_guide_call_handles_retryable_open_conflict(tmp_path, monkeypatch):
    X = _small_sparse_matrix(seed=91).tocsc()
    adata = ad.AnnData(X=X.copy())
    adata.var_names = np.array([f"g{i}" for i in range(adata.n_vars)], dtype=object)

    path = tmp_path / "guide_calling_backed_retry.h5ad"
    adata.write_h5ad(path)
    adata_backed = ad.read_h5ad(path, backed="r+")

    import actionet.backed_io as backed_io

    sentinel = object()
    create_calls = {"count": 0}

    def flaky_create(**kwargs):
        create_calls["count"] += 1
        if create_calls["count"] == 1:
            raise RuntimeError("createBackedOperator: failed to open h5ad file: simulated lock")
        return sentinel

    monkeypatch.setattr(backed_io, "_create_backed_operator", flaky_create)

    def fake_apply(
        resolved,
        *,
        background_thresholds_raw,
        foreground_thresholds_raw,
        backed_chunk_guides,
    ):
        assert resolved.kind == "operator"
        assert resolved.data is sentinel
        shape = adata_backed.shape
        background = sp.csr_matrix(shape, dtype=np.float64)
        foreground = sp.csr_matrix(shape, dtype=np.float64)
        return {"background": background, "foreground": foreground}

    monkeypatch.setattr(gc, "_apply_from_resolved", fake_apply)

    try:
        result = an.guide_call_gmm(
            adata_backed,
            result_mode="simple",
            background_thresholds=np.zeros(adata_backed.n_vars, dtype=np.float64),
            foreground_thresholds=np.ones(adata_backed.n_vars, dtype=np.float64),
            threshold_scale="raw",
            n_threads=1,
            backed_chunk_size=64,
            backed_chunk_guides=4,
        )
    finally:
        adata_backed.file.close()

    assert create_calls["count"] >= 2
    assert sp.issparse(result["assignments"]["foreground"])
    assert result["assignments"]["foreground"].shape == adata_backed.shape
