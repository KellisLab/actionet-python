"""Regression test for anndata >= 0.13 `layers[None]` alias for X.

Anndata 0.13 exposes `adata.X` as `adata.layers[None]`, so
`list(adata.layers.keys())` now always contains ``None``. Code paths
that iterate the layer keys and treat them as HDF5 sub-paths must
filter that alias out, otherwise ``h5py`` raises
``TypeError: Argument 'path' must not be None`` and (worse) we would
attempt to write ``X`` a second time under ``layers/None``.

This module locks in that behavior for the io-layer filter sites:

- :func:`actionet.io.subset.subset_backed_inplace` (via helper
  ``_real_layer_keys``).
- :func:`actionet.io.persist._real_layer_keys` (the primary shared
  abstraction; also drives ``_include_all_inmemory_annotations`` and
  the ``checkpoint_backed`` layer enumeration).
- :func:`actionet.io.anndata_io.collect_annotation_results` (inline
  ``if key is None: continue`` filter).
"""

from __future__ import annotations

import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from actionet.io.anndata_io import collect_annotation_results
from actionet.io.persist import _real_layer_keys
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


def test_real_layer_keys_filters_none_alias_on_backed_and_inmemory():
    """`_real_layer_keys` filters the ``layers[None]`` alias for both
    in-memory and backed AnnData objects.

    ``_real_layer_keys`` is the primary shared abstraction used by
    :mod:`actionet.io.subset`, :mod:`actionet.io.checkpoint`, and
    :func:`actionet.io.persist._include_all_inmemory_annotations`; its
    behavior is the invariant every backed HDF5 rewrite path relies on.
    """
    adata = _make_adata_with_empty_layers()

    real_keys_inmem = _real_layer_keys(adata)
    assert real_keys_inmem == [], (
        f"in-memory empty layers should filter None; got {real_keys_inmem}"
    )

    adata.layers["real_layer"] = np.ones((adata.n_obs, adata.n_vars), dtype=np.float32)
    real_keys_inmem = _real_layer_keys(adata)
    assert real_keys_inmem == ["real_layer"], (
        f"in-memory should skip None alias; got {real_keys_inmem}"
    )


def test_real_layer_keys_backed_skips_none_alias(tmp_path):
    """Backed AnnData exhibits ``layers.keys()`` = ``[None]`` for a file
    with no real layers; ``_real_layer_keys`` must return an empty list.
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    adata = _make_adata_with_empty_layers()
    path = str(tmp_path / "adata_real_layer_keys.h5ad")
    adata.write_h5ad(path)

    backed = ad.read_h5ad(path, backed="r+")
    try:
        raw_keys = list(backed.layers.keys())
        real_keys = _real_layer_keys(backed)
        assert real_keys == [], (
            f"raw keys={raw_keys!r}; filtered keys should be empty, got {real_keys!r}"
        )
    finally:
        if hasattr(backed, "file") and backed.file is not None:
            backed.file.close()


def test_collect_annotation_results_skips_layers_none_alias():
    """`collect_annotation_results` must skip the ``None`` key when it
    appears in the requested ``layers_keys`` list.

    Uses in-memory AnnData where ``layers[None]`` is aliased to ``.X`` by
    anndata >= 0.13, then explicitly requests both the ``None`` alias and a
    real layer. Result must contain the real layer only.
    """
    adata = _make_adata_with_empty_layers()
    adata.layers["real_layer"] = adata.X.copy()

    results = collect_annotation_results(
        adata,
        layers_keys=[None, "real_layer"],
        verbose=False,
    )

    layers_result = results["layers_keys"]
    assert None not in layers_result, (
        f"None key must be filtered from collected layers; got keys {list(layers_result)}"
    )
    assert "real_layer" in layers_result
    assert list(layers_result) == ["real_layer"]
