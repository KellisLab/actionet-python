"""Unit tests for MatrixSource and _backed_persist infrastructure."""

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import pytest

from actionet.io.matrix_source import MatrixSource
from actionet.io.persist import (
    is_backed_adata,
    persist_updates,
    apply_inmemory_updates,
)

from .conftest import make_test_adata, open_backed


# ---------------------------------------------------------------------------
# MatrixSource: basic in-memory behaviour
# ---------------------------------------------------------------------------

class TestMatrixSourceInMemory:
    """MatrixSource on in-memory AnnData (dense and sparse)."""

    @pytest.fixture(params=["dense", "csr", "csc"])
    def adata(self, request):
        return make_test_adata(n_cells=30, n_genes=20, sparse_fmt=request.param, seed=7)

    def test_shape(self, adata):
        src = MatrixSource(adata)
        assert src.shape == (30, 20)
        assert src.n_obs == 30
        assert src.n_vars == 20

    def test_layer(self, adata):
        src = MatrixSource(adata, layer="logcounts")
        assert src.shape == (30, 20)

    def test_missing_layer_raises(self, adata):
        with pytest.raises(KeyError, match="no_such_layer"):
            MatrixSource(adata, layer="no_such_layer").matrix

    def test_row_sums(self, adata):
        src = MatrixSource(adata)
        rs = src.row_sums(chunk_size=8)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X.sum(axis=1)).ravel()
        else:
            expected = X.sum(axis=1)
        np.testing.assert_allclose(rs, expected, rtol=1e-10)

    def test_col_sums(self, adata):
        src = MatrixSource(adata)
        cs = src.col_sums(chunk_size=8)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X.sum(axis=0)).ravel()
        else:
            expected = X.sum(axis=0)
        np.testing.assert_allclose(cs, expected, rtol=1e-10)

    def test_nnz_row_counts(self, adata):
        src = MatrixSource(adata)
        rc = src.nnz_row_counts(chunk_size=8)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X.getnnz(axis=1)).ravel()
        else:
            expected = np.count_nonzero(np.asarray(X), axis=1)
        np.testing.assert_array_equal(rc, expected)

    def test_nnz_col_counts(self, adata):
        src = MatrixSource(adata)
        cc = src.nnz_col_counts(chunk_size=8)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X.getnnz(axis=0)).ravel()
        else:
            expected = np.count_nonzero(np.asarray(X), axis=0)
        np.testing.assert_array_equal(cc, expected)

    def test_feature_subset_inmemory_not_implemented(self, adata):
        src = MatrixSource(adata)
        with pytest.raises(NotImplementedError):
            src.feature_subset(np.array([0, 5, 10]), chunk_size=8)

    def test_row_sums_with_row_indices(self, adata):
        src = MatrixSource(adata)
        idx = np.array([1, 3, 7])
        rs = src.row_sums(chunk_size=4, row_indices=idx)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X[idx, :].sum(axis=1)).ravel()
        else:
            expected = X[idx, :].sum(axis=1)
        np.testing.assert_allclose(rs, expected, rtol=1e-10)

    def test_col_sums_with_col_indices(self, adata):
        src = MatrixSource(adata)
        cidx = np.array([2, 4, 6])
        cs = src.col_sums(chunk_size=8, col_indices=cidx)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X[:, cidx].sum(axis=0)).ravel()
        else:
            expected = X[:, cidx].sum(axis=0)
        np.testing.assert_allclose(cs, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# MatrixSource: backed mode
# ---------------------------------------------------------------------------

class TestMatrixSourceBacked:
    """MatrixSource on backed AnnData."""

    def test_backed_row_sums(self, tmp_path):
        mem = make_test_adata(n_cells=24, n_genes=16, sparse_fmt="csr", seed=5)
        bk = open_backed(tmp_path, mem)

        src_mem = MatrixSource(mem)
        src_bk = MatrixSource(bk)

        np.testing.assert_allclose(
            src_bk.row_sums(chunk_size=8),
            src_mem.row_sums(chunk_size=8),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# _backed_persist
# ---------------------------------------------------------------------------

class TestBackedPersist:
    """Test persist_updates and apply_inmemory_updates."""

    def test_inmemory_updates_obs(self):
        adata = make_test_adata(n_cells=10, n_genes=5, sparse_fmt="dense", seed=1)
        arr = np.arange(10, dtype=float)
        apply_inmemory_updates(adata, obs={"test_col": arr})
        np.testing.assert_array_equal(adata.obs["test_col"].values, arr)

    def test_inmemory_updates_obsm(self):
        adata = make_test_adata(n_cells=10, n_genes=5, sparse_fmt="dense", seed=1)
        mat = np.ones((10, 3))
        apply_inmemory_updates(adata, obsm={"test_key": mat})
        np.testing.assert_array_equal(adata.obsm["test_key"], mat)

    def test_is_backed_false_for_inmemory(self):
        adata = make_test_adata(n_cells=10, n_genes=5, sparse_fmt="dense", seed=1)
        assert not is_backed_adata(adata)

    def test_is_backed_true_for_backed(self, tmp_path):
        mem = make_test_adata(n_cells=10, n_genes=5, sparse_fmt="csr", seed=1)
        bk = open_backed(tmp_path, mem)
        assert is_backed_adata(bk)

    def test_persist_updates_inmemory_noop(self):
        """persist_updates on in-memory data should just set attributes."""
        adata = make_test_adata(n_cells=10, n_genes=5, sparse_fmt="dense", seed=1)
        persist_updates(adata, uns={"foo": "bar"})
        assert adata.uns["foo"] == "bar"

    def test_persist_updates_backed_supports_dataframe_in_obsm_and_varm(self, tmp_path):
        """Backed writes should preserve DataFrame payloads stored in obsm/varm."""
        adata_mem = make_test_adata(n_cells=12, n_genes=8, sparse_fmt="csr", seed=2)
        path = tmp_path / "df_slots.h5ad"
        adata_mem.write_h5ad(path)
        adata = ad.read_h5ad(path, backed="r+")

        obs_df = pd.DataFrame(
            {
                "dc1": np.linspace(0.0, 1.1, adata.n_obs),
                "dc2": np.linspace(2.0, 3.1, adata.n_obs),
            },
            index=adata.obs_names.copy(),
        )
        var_df = pd.DataFrame(
            {"loading": np.linspace(-1.0, 1.0, adata.n_vars)},
            index=adata.var_names.copy(),
        )

        persist_updates(
            adata,
            obsm={"df_embed": obs_df},
            varm={"df_loadings": var_df},
            validate=True,
        )
        adata.file.close()

        reloaded = ad.read_h5ad(path)
        assert isinstance(reloaded.obsm["df_embed"], pd.DataFrame)
        assert isinstance(reloaded.varm["df_loadings"], pd.DataFrame)
        pd.testing.assert_index_equal(reloaded.obsm["df_embed"].index, obs_df.index)
        pd.testing.assert_index_equal(reloaded.varm["df_loadings"].index, var_df.index)
        pd.testing.assert_index_equal(reloaded.obsm["df_embed"].columns, obs_df.columns)
        pd.testing.assert_index_equal(reloaded.varm["df_loadings"].columns, var_df.columns)
        pd.testing.assert_frame_equal(reloaded.obsm["df_embed"], obs_df)
        pd.testing.assert_frame_equal(reloaded.varm["df_loadings"], var_df)
        # Explicit column-order round-trip: names must come back as strings, not bytes.
        assert reloaded.obsm["df_embed"].columns.tolist() == ["dc1", "dc2"]
        assert reloaded.varm["df_loadings"].columns.tolist() == ["loading"]

    def test_persist_updates_backed_dataframe_overwrite(self, tmp_path):
        """Overwriting an existing ndarray slot with a DataFrame should succeed."""
        adata_mem = make_test_adata(n_cells=10, n_genes=6, sparse_fmt="csr", seed=3)
        path = tmp_path / "overwrite.h5ad"
        adata_mem.write_h5ad(path)
        adata = ad.read_h5ad(path, backed="r+")

        arr = np.ones((adata.n_obs, 3), dtype=float)
        persist_updates(adata, obsm={"slot": arr})
        adata.file.close()

        adata = ad.read_h5ad(path, backed="r+")
        df = pd.DataFrame(
            {"a": np.arange(adata.n_obs, dtype=float)},
            index=adata.obs_names.copy(),
        )
        persist_updates(adata, obsm={"slot": df})
        adata.file.close()

        reloaded = ad.read_h5ad(path)
        assert isinstance(reloaded.obsm["slot"], pd.DataFrame)
        pd.testing.assert_frame_equal(reloaded.obsm["slot"], df)

    def test_persist_updates_backed_dataframe_string_columns(self, tmp_path):
        """DataFrames with string columns in obsm should round-trip correctly."""
        adata_mem = make_test_adata(n_cells=8, n_genes=4, sparse_fmt="csr", seed=4)
        path = tmp_path / "str_cols.h5ad"
        adata_mem.write_h5ad(path)
        adata = ad.read_h5ad(path, backed="r+")

        labels = [f"label_{i % 3}" for i in range(adata.n_obs)]
        df = pd.DataFrame({"cluster": labels}, index=adata.obs_names.copy())

        persist_updates(adata, obsm={"str_embed": df})
        adata.file.close()

        reloaded = ad.read_h5ad(path)
        assert isinstance(reloaded.obsm["str_embed"], pd.DataFrame)
        assert reloaded.obsm["str_embed"]["cluster"].tolist() == labels

    def test_persist_updates_backed_dataframe_rejected_in_obsp(self, tmp_path):
        """Passing a DataFrame to obsp should fail public validation."""
        from actionet.io.anndata_io import (
            ValidationError,
            append_to_anndata,
        )

        n = 6
        df = pd.DataFrame(np.eye(n))
        tmp_h5 = tmp_path / "reject.h5ad"
        ad.AnnData(np.eye(n)).write_h5ad(tmp_h5)
        with pytest.raises(ValidationError, match="Must be numpy array"):
            append_to_anndata(
                tmp_h5,
                {"obsp_keys": {"conn": df}},
            )


# ---------------------------------------------------------------------------
# Validation (_anndata_io)
# ---------------------------------------------------------------------------

class TestValidation:
    """Test _anndata_io validation accepts various input types."""

    def test_validate_obs_column_accepts_ndarray(self):
        from actionet.io.anndata_io import _validate_obs_var_column
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise
        _validate_obs_var_column("test", arr, 3, "obs", verbose=False)

    def test_validate_obs_column_accepts_list(self):
        from actionet.io.anndata_io import _validate_obs_var_column
        vals = [1, 2, 3]
        _validate_obs_var_column("test", vals, 3, "obs", verbose=False)

    def test_validate_obs_column_accepts_series(self):
        from actionet.io.anndata_io import _validate_obs_var_column
        s = pd.Series([1, 2, 3])
        _validate_obs_var_column("test", s, 3, "obs", verbose=False)

    def test_validate_obs_column_length_mismatch(self):
        from actionet.io.anndata_io import (
            _validate_obs_var_column,
            ValidationError,
        )
        with pytest.raises(ValidationError, match="Length mismatch"):
            _validate_obs_var_column("test", np.array([1, 2]), 3, "obs", verbose=False)

    def test_validate_matrix_accepts_sparse(self):
        from actionet.io.anndata_io import _validate_matrix
        mat = sp.csr_matrix(np.eye(5))
        _validate_matrix("test", mat, 5, "obsm", "obs", verbose=False)

    def test_validate_matrix_accepts_dense(self):
        from actionet.io.anndata_io import _validate_matrix
        mat = np.eye(5)
        _validate_matrix("test", mat, 5, "obsm", "obs", verbose=False)

    def test_validate_matrix_rejects_wrong_dim(self):
        from actionet.io.anndata_io import (
            _validate_matrix,
            ValidationError,
        )
        mat = np.eye(5)
        with pytest.raises(ValidationError, match="First dimension"):
            _validate_matrix("test", mat, 3, "obsm", "obs", verbose=False)

    def test_validate_matrix_accepts_dataframe(self):
        from actionet.io.anndata_io import _validate_matrix
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        _validate_matrix("test", df, 3, "obsm", "obs", verbose=False)

    def test_validate_matrix_dataframe_nan_names_column(self):
        from actionet.io.anndata_io import (
            _validate_matrix,
            ValidationError,
        )
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        with pytest.raises(ValidationError, match="Column 'a'.*NaN"):
            _validate_matrix("test", df, 3, "obsm", "obs", verbose=False)

    def test_validate_matrix_dataframe_shape_mismatch(self):
        from actionet.io.anndata_io import (
            _validate_matrix,
            ValidationError,
        )
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValidationError, match="First dimension"):
            _validate_matrix("test", df, 5, "obsm", "obs", verbose=False)
