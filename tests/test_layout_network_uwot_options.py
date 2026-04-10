"""Tests for extended uwot layout options exposed in the Python API."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import actionet as act


def _make_toy_adata(n_obs: int = 12, n_vars: int = 5) -> ad.AnnData:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)

    rows = []
    cols = []
    data = []
    for i in range(n_obs):
        j = (i + 1) % n_obs
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])
    adata.obsp["actionet"] = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    return adata


def _init_coords(adata: ad.AnnData, n_dims: int = 3) -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.ascontiguousarray(
        rng.normal(size=(adata.n_obs, n_dims)),
        dtype=np.float64,
    )


def test_layout_network_deterministic_rng_is_reproducible():
    adata = _make_toy_adata()
    init = _init_coords(adata, n_dims=3)

    out_a = act.layout_network(
        adata,
        initial_coords=init,
        method="umap",
        n_components=2,
        n_epochs=20,
        rng_type="deterministic",
        seed=1,
        n_threads=1,
        verbose=False,
        key_added="X_det",
        inplace=False,
    )
    out_b = act.layout_network(
        adata,
        initial_coords=init,
        method="umap",
        n_components=2,
        n_epochs=20,
        rng_type="deterministic",
        seed=999,
        n_threads=1,
        verbose=False,
        key_added="X_det",
        inplace=False,
    )

    np.testing.assert_allclose(out_a.obsm["X_det"], out_b.obsm["X_det"], rtol=0.0, atol=0.0)


def test_layout_network_largevis_runs():
    adata = _make_toy_adata()
    init = _init_coords(adata, n_dims=3)

    out = act.layout_network(
        adata,
        initial_coords=init,
        method="largevis",
        n_components=2,
        n_epochs=20,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_largevis",
        inplace=False,
    )

    assert out.obsm["X_largevis"].shape == (adata.n_obs, 2)


def test_layout_network_leopold_requires_ai():
    adata = _make_toy_adata()
    init = _init_coords(adata, n_dims=3)

    with pytest.raises(ValueError, match="`ai` must be provided"):
        act.layout_network(
            adata,
            initial_coords=init,
            method="leopold",
            n_components=2,
            n_epochs=10,
            seed=0,
            n_threads=1,
            verbose=False,
            inplace=False,
        )


def test_layout_network_leopold_runs_with_ai():
    adata = _make_toy_adata()
    init = _init_coords(adata, n_dims=3)
    ai = np.linspace(0.1, 1.0, adata.n_obs)

    out = act.layout_network(
        adata,
        initial_coords=init,
        method="leopold",
        ai=ai,
        n_components=2,
        n_epochs=20,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_leopold",
        inplace=False,
    )

    assert out.obsm["X_leopold"].shape == (adata.n_obs, 2)


def test_layout_network_leopold2_requires_ai_and_aj():
    adata = _make_toy_adata()
    init = _init_coords(adata, n_dims=3)
    ai = np.linspace(0.1, 1.0, adata.n_obs)

    with pytest.raises(ValueError, match="`ai` and `aj` must be provided"):
        act.layout_network(
            adata,
            initial_coords=init,
            method="leopold2",
            ai=ai,
            n_components=2,
            n_epochs=10,
            seed=0,
            n_threads=1,
            verbose=False,
            inplace=False,
        )


def test_layout_network_leopold2_runs_with_ai_and_aj():
    adata = _make_toy_adata()
    init = _init_coords(adata, n_dims=3)
    ai = np.linspace(0.1, 1.0, adata.n_obs)
    aj = np.linspace(0.2, 1.1, adata.n_obs)

    out = act.layout_network(
        adata,
        initial_coords=init,
        method="leopold2",
        ai=ai,
        aj=aj,
        n_components=2,
        n_epochs=20,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_leopold2",
        inplace=False,
    )

    assert out.obsm["X_leopold2"].shape == (adata.n_obs, 2)
