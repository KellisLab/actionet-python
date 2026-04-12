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
import os

import actionet
import actionet._backed_persist as _backed_persist


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


def _make_dense_test_adata(
    n_obs: int = 40,
    n_var: int = 25,
    *,
    dtype=np.float32,
) -> ad.AnnData:
    """Create a minimal dense AnnData with configurable dtype."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(n_obs, n_var)).astype(dtype, copy=False)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_var)])
    return ad.AnnData(X=X, obs=obs, var=var)


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

    def test_preserves_order_and_duplicates(self, inmemory_adata):
        obs_sel = np.array([5, 0, 0], dtype=np.int64)
        var_sel = np.array([8, 3, 3, 1], dtype=np.int64)

        result = actionet.subset_anndata(inmemory_adata, obs_idx=obs_sel, var_idx=var_sel, inplace=False)

        assert list(result.obs_names) == ["cell_5", "cell_0", "cell_0"]
        assert list(result.var_names) == ["gene_8", "gene_3", "gene_3", "gene_1"]


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

    def test_apply_filter_backed_not_inplace_keeps_source(self, tmp_path):
        adata = _make_test_adata(n_obs=80, n_var=30)
        path = str(tmp_path / "source.h5ad")
        adata.write_h5ad(path)
        backed = ad.read_h5ad(path, backed="r+")
        original_shape = backed.shape

        obs_mask = np.zeros(backed.n_obs, dtype=bool)
        obs_mask[:25] = True
        var_mask = np.zeros(backed.n_vars, dtype=bool)
        var_mask[:10] = True

        result = actionet.apply_filter(
            backed,
            obs_mask,
            var_mask,
            inplace=False,
            output_file=None,
        )

        assert result.shape == (25, 10)
        assert not result.isbacked

        reopened = ad.read_h5ad(path, backed="r")
        assert reopened.shape == original_shape
        reopened.file.close()
        backed.file.close()


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
    def test_view_materializes_in_place_by_default(self, backed_adata):
        """Default materialization rewrites parent file and refreshes handles."""
        parent_path = str(backed_adata.filename)
        view = backed_adata[10:20, :5]

        actionet.materialize_backed(view)

        assert not view.is_view
        assert view.isbacked
        assert view.n_obs == 10
        assert view.n_vars == 5
        assert str(view.filename) == parent_path

        assert backed_adata.shape == (10, 5)
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

    def test_bool_view_materialization_preserves_selection(self, backed_adata):
        """Boolean-backed views materialize to the intended obs/var subsets."""
        obs_mask = np.zeros(backed_adata.n_obs, dtype=bool)
        obs_mask[[0, 2, 5, 7, 11]] = True
        var_mask = np.zeros(backed_adata.n_vars, dtype=bool)
        var_mask[[1, 4, 9, 15]] = True

        view = backed_adata[obs_mask, var_mask]
        expected_obs_names = list(backed_adata.obs_names[obs_mask])
        expected_var_names = list(backed_adata.var_names[var_mask])

        actionet.materialize_backed(view)

        assert not view.is_view
        assert list(view.obs_names) == expected_obs_names
        assert list(view.var_names) == expected_var_names

    def test_custom_filename_failure_keeps_existing_file(self, backed_adata, tmp_path, monkeypatch):
        """Writer failures must not delete or truncate a pre-existing destination."""
        view = backed_adata[:20, :10]
        dest = str(tmp_path / "existing_dest.h5ad")
        original_payload = b"existing-content"
        with open(dest, "wb") as fh:
            fh.write(original_payload)

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic write failure")

        monkeypatch.setattr(_backed_persist, "_write_filtered_backed", _boom)

        with pytest.raises(RuntimeError, match="synthetic"):
            actionet.materialize_backed(view, filename=dest)

        assert os.path.exists(dest)
        with open(dest, "rb") as fh:
            assert fh.read() == original_payload


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


# ---------------------------------------------------------------------------
# Backed rewrite payload/regression checks
# ---------------------------------------------------------------------------

class TestBackedRewriteRegressions:
    def test_subset_backed_preserves_raw_payload(self, tmp_path):
        adata = _make_test_adata(n_obs=60, n_var=35)
        adata.raw = adata.copy()
        expected_raw_var_names = list(adata.raw.var_names)
        path = str(tmp_path / "raw_payload.h5ad")
        adata.write_h5ad(path)
        backed = ad.read_h5ad(path, backed="r+")

        actionet.subset_backed_inplace(
            backed,
            obs_idx=np.arange(25, dtype=np.int64),
            var_idx=np.arange(10, dtype=np.int64),
        )

        reopened = ad.read_h5ad(path, backed="r")
        assert reopened.shape == (25, 10)
        assert reopened.raw is not None
        assert reopened.raw.shape == (25, 35)
        assert list(reopened.raw.var_names) == expected_raw_var_names
        reopened.file.close()
        backed.file.close()

    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_subset_backed_dense_dtype_preserved(self, tmp_path, dtype):
        adata = _make_dense_test_adata(dtype=dtype)
        path = str(tmp_path / f"dense_{np.dtype(dtype).name}.h5ad")
        backed = _write_backed(adata, path)
        dtype_before = np.dtype(backed.X.dtype)

        actionet.subset_backed_inplace(
            backed,
            obs_idx=np.arange(20, dtype=np.int64),
            var_idx=np.arange(12, dtype=np.int64),
        )

        reopened = ad.read_h5ad(path, backed="r")
        assert np.dtype(reopened.X.dtype) == dtype_before
        reopened.file.close()
        backed.file.close()

    def test_pairwise_subsetting_path_does_not_use_np_ix(self, tmp_path, monkeypatch):
        adata = _make_test_adata(n_obs=70, n_var=40)
        obs_graph = sp.random(70, 70, density=0.15, random_state=0, format="csr")
        adata.obsp["obs_graph"] = (obs_graph + obs_graph.T).tocsr()
        var_graph = sp.random(40, 40, density=0.2, random_state=1, format="csr")
        adata.varp["var_graph"] = (var_graph + var_graph.T).tocsr()

        path = str(tmp_path / "pairwise.h5ad")
        backed = _write_backed(adata, path)

        def _forbid_ix(*args, **kwargs):
            raise AssertionError("np.ix_ should not be used in pairwise rewrite path")

        monkeypatch.setattr(_backed_persist.np, "ix_", _forbid_ix)

        actionet.subset_backed_inplace(
            backed,
            obs_idx=np.arange(30, dtype=np.int64),
            var_idx=np.arange(18, dtype=np.int64),
        )

        assert backed.obsp["obs_graph"].shape == (30, 30)
        assert backed.varp["var_graph"].shape == (18, 18)
