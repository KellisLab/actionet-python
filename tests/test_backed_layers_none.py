"""Regression test for anndata >= 0.13 `layers[None]` alias for X.

Anndata 0.13 exposes `adata.X` as `adata.layers[None]`, so
`list(adata.layers.keys())` now always contains ``None``. Code paths
that iterate the layer keys and treat them as HDF5 sub-paths must
filter that alias out, otherwise ``h5py`` raises
``TypeError: Argument 'path' must not be None`` and (worse) we would
attempt to write ``X`` a second time under ``layers/None``.

This test locks in that behavior for
:func:`actionet.io.subset.subset_backed_inplace`.
"""

from __future__ import annotations

import os
import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from actionet.io.subset import subset_backed_inplace


def _make_adata_with_empty_layers(n_obs: int = 30, n_var: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.random(n_obs, n_var, density=0.4, random_state=rng, format="csr", dtype=np.float32)
    X.data = np.abs(X.data)
    obs = pd.DataFrame(
        {"group": rng.choice(["A", "B"], size=n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_var)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_subset_backed_inplace_ignores_layers_none_alias(tmp_path):
    adata = _make_adata_with_empty_layers()
    path = str(tmp_path / "adata_none_layer.h5ad")
    adata.write_h5ad(path)

    with h5py.File(path, "r") as f:
        assert "layers" not in f or len(list(f["layers"].keys())) == 0

    backed = ad.read_h5ad(path, backed="r+")

    keys = list(backed.layers.keys())
    real_keys = [k for k in keys if k is not None]
    assert real_keys == [], (
        f"fixture should have no real layers; got {real_keys}"
    )

    obs_idx = np.array([0, 1, 2, 5, 7], dtype=np.int64)
    var_idx = np.array([0, 3, 4], dtype=np.int64)

    subset_backed_inplace(backed, obs_idx=obs_idx, var_idx=var_idx)

    assert backed.n_obs == obs_idx.size
    assert backed.n_vars == var_idx.size

    backed.file.close()

    with h5py.File(path, "r") as f:
        assert "X" in f
        if "layers" in f:
            layer_keys = list(f["layers"].keys())
            assert layer_keys == [], (
                f"layers group should be empty; got {layer_keys}"
            )
