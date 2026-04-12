"""Tests for subset_anndata and subset_backed_inplace.

Validates that backed AnnData subsetting correctly rewrites the HDF5 file
and refreshes the handle, so that downstream operations (e.g. reduce_kernel)
see consistent dimensions.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad
import tempfile
import os
import shutil

import actionet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_adata(n_obs: int = 200, n_var: int = 100, seed: int = 42) -> ad.AnnData:
    """Create a minimal sparse AnnData suitable for testing."""
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_var, density=0.3, random_state=rng, format="csr", dtype=np.float64)
    X.data = np.abs(X.data) * 100
    obs = pd.DataFrame(
        {"group": rng.choice(["A", "B", "C"], size=n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_var)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _write_backed(adata: ad.AnnData, path: str) -> ad.AnnData:
    """Write an in-memory AnnData to disk and reopen as backed r+."""
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r+")


@pytest.fixture
def backed_adata(tmp_path):
    """Provide a backed AnnData in a temporary directory."""
    adata = _make_test_adata()
    path = str(tmp_path / "test.h5ad")
    return _write_backed(adata, path)


@pytest.fixture
def inmemory_adata():
    """Provide an in-memory AnnData."""
    return _make_test_adata()


# ---------------------------------------------------------------------------
# subset_anndata: backed mode
# ---------------------------------------------------------------------------

class TestSubsetAnndataBacked:
    def test_obs_subset_bool_mask(self, backed_adata):
        orig_n_obs = backed_adata.n_obs
        mask = np.zeros(orig_n_obs, dtype=bool)
        mask[:50] = True

        actionet.subset_anndata(backed_adata, obs_idx=mask, inplace=True)

        assert backed_adata.n_obs == 50
        assert backed_adata.shape[0] == 50
        assert backed_adata.isbacked

    def test_var_subset_bool_mask(self, backed_adata):
        orig_n_var = backed_adata.n_vars
        mask = np.zeros(orig_n_var, dtype=bool)
        mask[:30] = True

        actionet.subset_anndata(backed_adata, var_idx=mask, inplace=True)

        assert backed_adata.n_vars == 30
        assert backed_adata.shape[1] == 30
        assert backed_adata.isbacked

    def test_both_axes_int_indices(self, backed_adata):
        obs_sel = np.array([0, 5, 10, 15, 20], dtype=np.int64)
        var_sel = np.array([0, 1, 2, 3], dtype=np.int64)

        actionet.subset_anndata(backed_adata, obs_idx=obs_sel, var_idx=var_sel, inplace=True)

        assert backed_adata.n_obs == 5
        assert backed_adata.n_vars == 4
        assert backed_adata.isbacked

    def test_noop_subset_is_safe(self, backed_adata):
        """Passing None for both axes should be a no-op."""
        orig_shape = backed_adata.shape
        actionet.subset_anndata(backed_adata, inplace=True)
        assert backed_adata.shape == orig_shape

    def test_not_inplace_returns_new(self, backed_adata):
        mask = np.zeros(backed_adata.n_obs, dtype=bool)
        mask[:30] = True

        result = actionet.subset_anndata(backed_adata, obs_idx=mask, inplace=False)

        assert result is not None
        assert result.n_obs == 30
        assert backed_adata.n_obs == 200  # original unchanged

        if hasattr(result, "file") and result.file is not None:
            result.file.close()

    def test_obs_dataframe_preserved(self, backed_adata):
        obs_sel = np.array([0, 1, 2], dtype=np.int64)
        expected_index = list(backed_adata.obs_names[obs_sel])

        actionet.subset_anndata(backed_adata, obs_idx=obs_sel, inplace=True)

        assert list(backed_adata.obs_names) == expected_index

    def test_file_dimensions_match_python(self, backed_adata):
        """The critical invariant: on-disk shape == Python shape after subset."""
        mask = np.zeros(backed_adata.n_obs, dtype=bool)
        mask[::2] = True  # keep every other cell
        expected_n = int(mask.sum())

        actionet.subset_anndata(backed_adata, obs_idx=mask, inplace=True)

        reopened = ad.read_h5ad(str(backed_adata.filename), backed="r")
        assert reopened.n_obs == expected_n
        assert reopened.n_obs == backed_adata.n_obs
        reopened.file.close()


# ---------------------------------------------------------------------------
# subset_anndata: in-memory mode
# ---------------------------------------------------------------------------

class TestSubsetAnndataInMemory:
    def test_obs_subset_inplace(self, inmemory_adata):
        mask = np.zeros(inmemory_adata.n_obs, dtype=bool)
        mask[:50] = True

        actionet.subset_anndata(inmemory_adata, obs_idx=mask, inplace=True)
        assert inmemory_adata.n_obs == 50

    def test_both_axes_not_inplace(self, inmemory_adata):
        obs_sel = np.arange(0, 100, dtype=np.int64)
        var_sel = np.arange(0, 50, dtype=np.int64)

        result = actionet.subset_anndata(inmemory_adata, obs_idx=obs_sel, var_idx=var_sel, inplace=False)

        assert result.n_obs == 100
        assert result.n_vars == 50
        assert inmemory_adata.n_obs == 200  # unchanged


# ---------------------------------------------------------------------------
# subset_backed_inplace: error handling
# ---------------------------------------------------------------------------

class TestSubsetBackedInplaceErrors:
    def test_rejects_inmemory(self, inmemory_adata):
        with pytest.raises(ValueError, match="backed"):
            actionet.subset_backed_inplace(inmemory_adata, obs_idx=np.array([0, 1]))

    def test_rejects_readonly(self, tmp_path):
        adata = _make_test_adata(n_obs=20, n_var=10)
        path = str(tmp_path / "ro.h5ad")
        adata.write_h5ad(path)
        ro = ad.read_h5ad(path, backed="r")
        with pytest.raises(ValueError, match="read-only"):
            actionet.subset_backed_inplace(ro, obs_idx=np.array([0, 1]))
        ro.file.close()


# ---------------------------------------------------------------------------
# Integration: subset_anndata -> filter_anndata parity
# ---------------------------------------------------------------------------

class TestSubsetFilterParity:
    def test_filter_and_subset_produce_same_shape(self, tmp_path):
        """subset_anndata with the same masks as filter_anndata should yield identical shapes."""
        adata = _make_test_adata(n_obs=200, n_var=100)

        path_a = str(tmp_path / "a.h5ad")
        path_b = str(tmp_path / "b.h5ad")
        adata.write_h5ad(path_a)
        adata.write_h5ad(path_b)

        adata_a = ad.read_h5ad(path_a, backed="r+")
        adata_b = ad.read_h5ad(path_b, backed="r+")

        obs_mask, var_mask = actionet.compute_filter_masks(
            adata_a, min_cells_per_feat=5,
        )

        actionet.apply_filter(adata_a, obs_mask, var_mask, inplace=True)
        actionet.subset_anndata(adata_b, obs_idx=obs_mask, var_idx=var_mask, inplace=True)

        assert adata_a.shape == adata_b.shape
        assert list(adata_a.obs_names) == list(adata_b.obs_names)
        assert list(adata_a.var_names) == list(adata_b.var_names)


# ---------------------------------------------------------------------------
# Input validation edge cases
# ---------------------------------------------------------------------------

class TestSubsetInputValidation:
    """Guards for negative, OOB, duplicate, and empty indices."""

    def test_negative_indices_raises(self, inmemory_adata):
        with pytest.raises(ValueError, match="negative"):
            actionet.subset_anndata(
                inmemory_adata, obs_idx=np.array([0, -1, 5]), inplace=True,
            )

    def test_negative_var_indices_raises(self, inmemory_adata):
        with pytest.raises(ValueError, match="negative"):
            actionet.subset_anndata(
                inmemory_adata, var_idx=np.array([-3, 0, 2]), inplace=True,
            )

    def test_oob_obs_indices_raises(self, inmemory_adata):
        with pytest.raises(ValueError, match="axis size"):
            actionet.subset_anndata(
                inmemory_adata, obs_idx=np.array([0, 999]), inplace=True,
            )

    def test_oob_var_indices_raises(self, inmemory_adata):
        with pytest.raises(ValueError, match="axis size"):
            actionet.subset_anndata(
                inmemory_adata, var_idx=np.array([0, 100]), inplace=True,
            )

    def test_duplicate_indices_warns(self, inmemory_adata):
        with pytest.warns(UserWarning, match="duplicate"):
            actionet.subset_anndata(
                inmemory_adata, obs_idx=np.array([0, 0, 1, 1, 2]), inplace=True,
            )

    def test_empty_obs_mask_raises(self, inmemory_adata):
        mask = np.zeros(inmemory_adata.n_obs, dtype=bool)
        with pytest.raises(ValueError, match="zero"):
            actionet.subset_anndata(inmemory_adata, obs_idx=mask, inplace=True)

    def test_empty_var_indices_raises(self, inmemory_adata):
        with pytest.raises(ValueError, match="zero"):
            actionet.subset_anndata(
                inmemory_adata, var_idx=np.array([], dtype=np.int64), inplace=True,
            )

    def test_empty_backed_obs_raises(self, backed_adata):
        with pytest.raises(ValueError, match="zero"):
            actionet.subset_backed_inplace(
                backed_adata, obs_idx=np.array([], dtype=np.int64),
            )

    def test_empty_backed_var_raises(self, backed_adata):
        with pytest.raises(ValueError, match="zero"):
            actionet.subset_backed_inplace(
                backed_adata, var_idx=np.array([], dtype=np.int64),
            )


# ---------------------------------------------------------------------------
# materialize_backed
# ---------------------------------------------------------------------------

class TestMaterializeBacked:
    def test_view_creates_independent_file(self, backed_adata):
        """Materializing a view gives it its own file; parent is untouched."""
        parent_path = str(backed_adata.filename)
        parent_shape = backed_adata.shape
        view = backed_adata[10:20, :5]

        actionet.materialize_backed(view)

        assert not view.is_view
        assert view.isbacked
        assert view.n_obs == 10
        assert view.n_vars == 5
        assert str(view.filename) != parent_path

        assert backed_adata.shape == parent_shape
        assert not backed_adata.is_view

    def test_reassigned_variable(self, tmp_path):
        """adata = adata[...]; materialize_backed(adata) works."""
        adata = _make_test_adata(n_obs=100, n_var=50)
        path = str(tmp_path / "test.h5ad")
        adata.write_h5ad(path)
        adata = ad.read_h5ad(path, backed="r+")

        adata = adata[5:25, :10]
        assert adata.is_view

        actionet.materialize_backed(adata)

        assert not adata.is_view
        assert adata.isbacked
        assert adata.n_obs == 20
        assert adata.n_vars == 10

        reopened = ad.read_h5ad(str(adata.filename), backed="r")
        assert reopened.shape == (20, 10)
        reopened.file.close()

    def test_noop_on_non_view(self, backed_adata):
        """Non-view backed AnnData is a no-op."""
        orig_path = str(backed_adata.filename)
        orig_shape = backed_adata.shape

        actionet.materialize_backed(backed_adata)

        assert backed_adata.shape == orig_shape
        assert str(backed_adata.filename) == orig_path

    def test_rejects_in_memory(self, inmemory_adata):
        with pytest.raises(ValueError, match="backed"):
            actionet.materialize_backed(inmemory_adata)

    def test_custom_filename(self, backed_adata, tmp_path):
        """materialize_backed respects an explicit filename argument."""
        view = backed_adata[:50, :30]
        dest = str(tmp_path / "custom_output.h5ad")

        actionet.materialize_backed(view, filename=dest)

        assert not view.is_view
        assert str(view.filename) == dest
        assert view.n_obs == 50
        assert view.n_vars == 30

    def test_data_integrity(self, tmp_path):
        """Materialized view contains the correct data values."""
        adata = _make_test_adata(n_obs=50, n_var=20)
        path = str(tmp_path / "test.h5ad")
        adata.write_h5ad(path)
        backed = ad.read_h5ad(path, backed="r+")

        expected_obs_names = list(backed.obs_names[5:15])
        expected_var_names = list(backed.var_names[:10])

        view = backed[5:15, :10]
        actionet.materialize_backed(view)

        assert list(view.obs_names) == expected_obs_names
        assert list(view.var_names) == expected_var_names


# ---------------------------------------------------------------------------
# subset_anndata on backed views
# ---------------------------------------------------------------------------

class TestSubsetAnndataBackedView:
    def test_inplace_no_extra_subset(self, backed_adata):
        """subset_anndata on a view with no extra idx materializes the view."""
        view = backed_adata[10:30, :20]
        actionet.subset_anndata(view, inplace=True)

        assert not view.is_view
        assert view.isbacked
        assert view.n_obs == 20
        assert view.n_vars == 20

    def test_inplace_with_extra_subset(self, backed_adata):
        """subset_anndata on a view with additional obs_idx works."""
        view = backed_adata[10:30, :20]
        extra_obs = np.array([0, 2, 4], dtype=np.int64)

        actionet.subset_anndata(view, obs_idx=extra_obs, inplace=True)

        assert not view.is_view
        assert view.n_obs == 3
        assert view.n_vars == 20

    def test_not_inplace_returns_new(self, backed_adata):
        """subset_anndata(view, inplace=False) returns a new backed object."""
        view = backed_adata[10:30, :20]
        obs_sel = np.array([0, 5, 10], dtype=np.int64)

        result = actionet.subset_anndata(view, obs_idx=obs_sel, inplace=False)

        assert result is not None
        assert result.n_obs == 3
        assert result.n_vars == 20
        assert result.isbacked
        assert not result.is_view

        assert backed_adata.shape == (200, 100)

        result.file.close()
