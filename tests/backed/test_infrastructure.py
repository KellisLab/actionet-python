"""Unit tests for MatrixSource and _backed_persist infrastructure."""

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import pytest

from actionet._matrix_source import MatrixSource
from actionet._backed_persist import (
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

    def test_xt_dot(self, adata):
        src = MatrixSource(adata)
        rng = np.random.default_rng(1)
        right = rng.standard_normal((src.n_obs, 3))
        result = src.xt_dot(right, chunk_size=10)

        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X.T.dot(right))
        else:
            expected = X.T @ right
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_x_dot(self, adata):
        src = MatrixSource(adata)
        rng = np.random.default_rng(2)
        right = rng.standard_normal((src.n_vars, 3))
        result = src.x_dot(right, chunk_size=10)

        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X.dot(right))
        else:
            expected = X @ right
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_global_min_nonneg(self, adata):
        src = MatrixSource(adata)
        gmin = src.global_min(chunk_size=8)
        assert gmin == 0.0  # Poisson data is non-negative

    def test_global_min_with_negatives(self):
        X = np.array([[-1.5, 2.0], [0.0, 3.0]])
        adata = ad.AnnData(X=X)
        src = MatrixSource(adata)
        assert src.global_min(chunk_size=1) == -1.5

    def test_feature_subset(self, adata):
        src = MatrixSource(adata)
        idx = np.array([0, 5, 10])
        sub = src.feature_subset(idx, chunk_size=8)
        X = adata.X
        if sp.issparse(X):
            expected = np.asarray(X[:, idx].todense())
        else:
            expected = X[:, idx]
        np.testing.assert_allclose(
            np.asarray(sub.todense()) if sp.issparse(sub) else sub,
            expected,
            rtol=1e-10,
        )

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

    def test_backed_xt_dot(self, tmp_path):
        mem = make_test_adata(n_cells=24, n_genes=16, sparse_fmt="csr", seed=5)
        bk = open_backed(tmp_path, mem)

        rng = np.random.default_rng(9)
        right = rng.standard_normal((24, 3))

        res_mem = MatrixSource(mem).xt_dot(right, chunk_size=8)
        res_bk = MatrixSource(bk).xt_dot(right, chunk_size=8)
        np.testing.assert_allclose(res_bk, res_mem, rtol=1e-8)

    def test_backed_x_dot(self, tmp_path):
        mem = make_test_adata(n_cells=24, n_genes=16, sparse_fmt="csr", seed=5)
        bk = open_backed(tmp_path, mem)

        rng = np.random.default_rng(10)
        right = rng.standard_normal((16, 3))

        res_mem = MatrixSource(mem).x_dot(right, chunk_size=8)
        res_bk = MatrixSource(bk).x_dot(right, chunk_size=8)
        np.testing.assert_allclose(res_bk, res_mem, rtol=1e-8)


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


# ---------------------------------------------------------------------------
# Validation (_anndata_io)
# ---------------------------------------------------------------------------

class TestValidation:
    """Test _anndata_io validation accepts various input types."""

    def test_validate_obs_column_accepts_ndarray(self):
        from actionet.experimental._anndata_io import _validate_obs_var_column
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise
        _validate_obs_var_column("test", arr, 3, "obs", verbose=False)

    def test_validate_obs_column_accepts_list(self):
        from actionet.experimental._anndata_io import _validate_obs_var_column
        vals = [1, 2, 3]
        _validate_obs_var_column("test", vals, 3, "obs", verbose=False)

    def test_validate_obs_column_accepts_series(self):
        from actionet.experimental._anndata_io import _validate_obs_var_column
        s = pd.Series([1, 2, 3])
        _validate_obs_var_column("test", s, 3, "obs", verbose=False)

    def test_validate_obs_column_length_mismatch(self):
        from actionet.experimental._anndata_io import (
            _validate_obs_var_column,
            ValidationError,
        )
        with pytest.raises(ValidationError, match="Length mismatch"):
            _validate_obs_var_column("test", np.array([1, 2]), 3, "obs", verbose=False)

    def test_validate_matrix_accepts_sparse(self):
        from actionet.experimental._anndata_io import _validate_matrix
        mat = sp.csr_matrix(np.eye(5))
        _validate_matrix("test", mat, 5, "obsm", "obs", verbose=False)

    def test_validate_matrix_accepts_dense(self):
        from actionet.experimental._anndata_io import _validate_matrix
        mat = np.eye(5)
        _validate_matrix("test", mat, 5, "obsm", "obs", verbose=False)

    def test_validate_matrix_rejects_wrong_dim(self):
        from actionet.experimental._anndata_io import (
            _validate_matrix,
            ValidationError,
        )
        mat = np.eye(5)
        with pytest.raises(ValidationError, match="First dimension"):
            _validate_matrix("test", mat, 3, "obsm", "obs", verbose=False)
