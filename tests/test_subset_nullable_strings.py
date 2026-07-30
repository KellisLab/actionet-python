"""Regression: backed subset must handle pandas StringDtype obs/var.

anndata >= 0.12 reads string obs/var columns as :class:`pd.StringArray`
by default, and ``anndata.io.write_elem`` refuses to write them unless
``anndata.settings.allow_write_nullable_strings`` is enabled. The
``_write_filtered_backed`` path in :mod:`actionet.io.subset` coerces
nullable string columns/indices to numpy object dtype before writing so
the on-disk contract stays compatible with anndata < 0.11 and anndataR.

Reproduces the traceback from calling
``subset_anndata(..., inplace=False, output_file=None)`` on an AnnData
whose obs/var carry :class:`pd.StringDtype` columns.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import actionet as an


def _make_object_string_adata(seed: int = 7) -> ad.AnnData:
    """Source AnnData written with legacy object-dtype string columns.

    Anndata >= 0.12 promotes these to ``StringDtype`` on read; that is the
    state that trips the write guard when we subsequently subset.
    """
    rng = np.random.default_rng(seed)
    n_obs, n_var = 5, 4
    X = sp.random(n_obs, n_var, density=0.5, random_state=rng, format="csr", dtype=np.float64)
    X.data = np.abs(X.data) + 1.0

    obs = pd.DataFrame(
        {
            "label": np.array(["a", "b", "c", "d", "e"], dtype=object),
            "cluster": pd.Categorical(["x", "y", "x", "y", "x"]),
            "score": np.arange(n_obs, dtype=np.float32),
        },
        index=pd.Index([f"cell_{i}" for i in range(n_obs)], dtype=object),
    )
    var = pd.DataFrame(
        {
            "symbol": np.array(["g0", "g1", "g2", "g3"], dtype=object),
            "chrom": np.array(["1", "1", "2", "3"], dtype=object),
        },
        index=pd.Index([f"gene_{i}" for i in range(n_var)], dtype=object),
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _force_nullable_string_columns(adata: ad.AnnData) -> None:
    """Coerce obs/var string columns/indices to ``pd.StringDtype`` in place.

    Simulates the anndata >= 0.12 read behaviour that would otherwise
    require an environment-dependent read to reproduce.
    """
    for frame_name in ("obs", "var"):
        frame = getattr(adata, frame_name)
        for col in frame.columns:
            if frame[col].dtype == object:
                frame[col] = frame[col].astype("string")
        frame.index = pd.Index(frame.index.astype(str), dtype="string")
        setattr(adata, frame_name, frame)


def test_subset_backed_with_nullable_string_columns(tmp_path):
    """subset_anndata(inplace=False, output_file=None) on a backed AnnData
    whose obs/var carry ``pd.StringDtype`` columns must not raise
    ``allow_write_nullable_strings`` and must return a valid subset.
    """
    src = _make_object_string_adata()
    path = str(tmp_path / "nullable_strings.h5ad")
    src.write_h5ad(path)

    backed = ad.read_h5ad(path, backed="r+")
    _force_nullable_string_columns(backed)

    obs_mask = np.array([True, False, True, False, True])

    try:
        result = an.subset_anndata(
            backed,
            obs_idx=obs_mask,
            inplace=False,
            output_file=None,
        )

        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 3
        assert result.n_vars == backed.n_vars
        assert list(result.obs.index) == ["cell_0", "cell_2", "cell_4"]
        assert list(result.obs["label"]) == ["a", "c", "e"]
        assert list(result.var.index) == list(backed.var.index)
    finally:
        try:
            backed.file.close()
        except Exception:
            pass


def test_coerce_helper_preserves_non_string_dtypes():
    """The helper must only touch StringDtype; numeric/categorical/bool stay put."""
    from actionet.io import coerce_nullable_strings_for_write

    df = pd.DataFrame(
        {
            "s": pd.array(["a", pd.NA, "b"], dtype="string"),
            "n": np.array([1, 2, 3], dtype=np.int64),
            "f": np.array([0.5, 1.5, 2.5], dtype=np.float32),
            "b": np.array([True, False, True]),
            "c": pd.Categorical(["x", "y", "x"]),
            "o": np.array(["p", "q", "r"], dtype=object),
        },
        index=pd.Index(["r0", "r1", "r2"], dtype="string"),
    )

    out = coerce_nullable_strings_for_write(df)

    assert out["s"].dtype == object
    assert list(out["s"]) == ["a", None, "b"] or list(out["s"]) == ["a", pd.NA, "b"]
    assert out["n"].dtype == np.int64
    assert out["f"].dtype == np.float32
    assert out["b"].dtype == bool
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert out["o"].dtype == object
    assert out.index.dtype == object


def test_coerce_helper_handles_string_categorical_categories():
    """Categorical whose .categories are StringDtype must be rewritten to object."""
    from actionet.io import coerce_nullable_strings_for_write

    cats = pd.array(["x", "y", "z"], dtype="string")
    series = pd.Categorical(["x", "y", "x", "z"], categories=cats)
    df = pd.DataFrame({"c": series})

    out = coerce_nullable_strings_for_write(df)

    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert out["c"].cat.categories.dtype == object
    assert list(out["c"]) == ["x", "y", "x", "z"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
