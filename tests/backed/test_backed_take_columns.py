"""Tests for the C++ backed_take_columns binding."""

import numpy as np
import pytest
import scipy.sparse as sp

from .conftest import make_test_adata, open_backed


@pytest.fixture(params=["csr", "csc"])
def backed_sparse(request, tmp_path):
    """Yield (backed_adata, in_memory_X) tuples for CSR and CSC."""
    fmt = request.param
    adata = make_test_adata(n_cells=96, n_genes=72, sparse_fmt=fmt)
    X_mem = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    backed = open_backed(tmp_path, adata)
    return backed, X_mem, fmt


@pytest.fixture
def backed_dense(tmp_path):
    """Yield (backed_adata, in_memory_X) for a dense-backed file."""
    adata = make_test_adata(n_cells=50, n_genes=30, sparse_fmt="dense")
    X_mem = np.asarray(adata.X)
    backed = open_backed(tmp_path, adata)
    return backed, X_mem


def _create_op(adata, group_path="/X"):
    from actionet import _core
    return _core.create_backed_operator(
        file_path=str(adata.filename),
        group_path=group_path,
    )


class TestBackedTakeColumnsSparse:
    """Tests for sparse-backed takeColumns via backed_take_columns."""

    def test_dense_output_all_rows(self, backed_sparse):
        from actionet import _core
        backed, X_mem, fmt = backed_sparse
        op = _create_op(backed)
        cols = np.array([0, 5, 10, 2], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=False)
        expected = X_mem[:, cols]
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_sparse_output_all_rows(self, backed_sparse):
        from actionet import _core
        backed, X_mem, fmt = backed_sparse
        op = _create_op(backed)
        cols = np.array([1, 3, 7], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=True)
        assert sp.issparse(result)
        expected = X_mem[:, cols]
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)

    def test_row_indices(self, backed_sparse):
        from actionet import _core
        backed, X_mem, fmt = backed_sparse
        op = _create_op(backed)
        cols = np.array([0, 5, 10], dtype=np.int64)
        rows = np.array([2, 10, 50, 90], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, row_indices=rows, prefer_sparse=False)
        expected = X_mem[np.ix_(rows, cols)]
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_empty_columns(self, backed_sparse):
        from actionet import _core
        backed, X_mem, fmt = backed_sparse
        op = _create_op(backed)
        cols = np.array([], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=False)
        assert result.shape == (96, 0)

    def test_duplicate_columns_preserved(self, backed_sparse):
        from actionet import _core
        backed, X_mem, fmt = backed_sparse
        op = _create_op(backed)
        cols = np.array([3, 3, 5, 3], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=False)
        expected = X_mem[:, cols]
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_single_column(self, backed_sparse):
        from actionet import _core
        backed, X_mem, fmt = backed_sparse
        op = _create_op(backed)
        cols = np.array([42], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=False)
        expected = X_mem[:, cols]
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestBackedTakeColumnsDense:
    """Tests for dense-backed takeColumns."""

    def test_dense_output_all_rows(self, backed_dense):
        from actionet import _core
        backed, X_mem = backed_dense
        op = _create_op(backed)
        cols = np.array([0, 5, 10, 2], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=False)
        expected = X_mem[:, cols]
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_sparse_output(self, backed_dense):
        from actionet import _core
        backed, X_mem = backed_dense
        op = _create_op(backed)
        cols = np.array([1, 3, 7], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, prefer_sparse=True)
        assert sp.issparse(result)
        expected = X_mem[:, cols]
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)

    def test_row_indices(self, backed_dense):
        from actionet import _core
        backed, X_mem = backed_dense
        op = _create_op(backed)
        cols = np.array([0, 5, 10], dtype=np.int64)
        rows = np.array([2, 10, 30], dtype=np.int64)
        result = _core.backed_take_columns(op, cols, row_indices=rows, prefer_sparse=False)
        expected = X_mem[np.ix_(rows, cols)]
        np.testing.assert_allclose(result, expected, atol=1e-12)
