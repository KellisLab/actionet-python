"""Tests for _sparse_row_sum_sq: NNZ-based row sum-of-squares without S.power(2)."""

import numpy as np
import pytest
import scipy.sparse as sp

from actionet.annotation import _sparse_row_sum_sq


RTOL = 1e-8
ATOL = 1e-10


def _reference_row_sum_sq(S):
    """Ground truth via dense computation."""
    D = S.toarray() if sp.issparse(S) else np.asarray(S)
    return np.sum(D * D, axis=1).astype(np.float64)


class TestSparseRowSumSqParity:
    """Parity with S.power(2).sum(axis=1) across formats."""

    @pytest.fixture(params=["csr", "csc", "coo"])
    def sparse_matrix(self, request):
        rng = np.random.default_rng(42)
        X = rng.poisson(0.5, size=(200, 100)).astype(np.float64)
        X[X < 0.3] = 0
        fmt = request.param
        if fmt == "csr":
            return sp.csr_matrix(X)
        elif fmt == "csc":
            return sp.csc_matrix(X)
        return sp.coo_matrix(X)

    def test_parity_with_reference(self, sparse_matrix):
        result = _sparse_row_sum_sq(sparse_matrix)
        expected = _reference_row_sum_sq(sparse_matrix)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_dtype_is_float64(self, sparse_matrix):
        result = _sparse_row_sum_sq(sparse_matrix)
        assert result.dtype == np.float64

    def test_shape_is_1d_nrows(self, sparse_matrix):
        result = _sparse_row_sum_sq(sparse_matrix)
        assert result.shape == (sparse_matrix.shape[0],)


class TestSparseRowSumSqEdgeCases:
    """Edge cases: empty rows, all-zero, single element, large indices."""

    def test_empty_rows(self):
        """Matrix where some rows have no stored entries."""
        row = np.array([0, 0, 2, 4, 4, 4])
        col = np.array([1, 3, 0, 2, 5, 7])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        S = sp.csr_matrix((data, (row, col)), shape=(6, 10))
        result = _sparse_row_sum_sq(S)
        expected = _reference_row_sum_sq(S)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)
        assert result[1] == 0.0
        assert result[3] == 0.0
        assert result[5] == 0.0

    def test_all_zero_matrix(self):
        S_csr = sp.csr_matrix((50, 30))
        result_csr = _sparse_row_sum_sq(S_csr)
        assert np.all(result_csr == 0.0)
        assert result_csr.shape == (50,)

        S_csc = sp.csc_matrix((50, 30))
        result_csc = _sparse_row_sum_sq(S_csc)
        assert np.all(result_csc == 0.0)

    def test_single_element(self):
        S = sp.csr_matrix(([7.5], ([0], [0])), shape=(1, 1))
        result = _sparse_row_sum_sq(S)
        np.testing.assert_allclose(result, [7.5 ** 2], rtol=RTOL, atol=ATOL)

    def test_single_row(self):
        S = sp.csr_matrix(np.array([[1.0, 0, 3.0, 0, 2.0]]))
        result = _sparse_row_sum_sq(S)
        np.testing.assert_allclose(result, [1.0 + 9.0 + 4.0], rtol=RTOL, atol=ATOL)

    def test_large_index_sparse(self):
        """Wide matrix (100k columns) with few nonzeros."""
        row = np.array([0, 1, 1])
        col = np.array([99999, 0, 50000])
        data = np.array([1.5, 2.5, 3.5])
        S = sp.csr_matrix((data, (row, col)), shape=(3, 100000))
        result = _sparse_row_sum_sq(S)
        expected = _reference_row_sum_sq(S)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)
        assert result[2] == 0.0

    def test_mixed_sparsity(self):
        """One fully dense row, one fully empty row."""
        rng = np.random.default_rng(7)
        D = np.zeros((4, 20), dtype=np.float64)
        D[0, :] = rng.standard_normal(20)
        D[2, 5] = 42.0
        for fmt in [sp.csr_matrix, sp.csc_matrix]:
            S = fmt(D)
            result = _sparse_row_sum_sq(S)
            expected = _reference_row_sum_sq(S)
            np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)
            assert result[1] == 0.0
            assert result[3] == 0.0


class TestSparseRowSumSqInputTypes:
    """Non-float and integer input dtypes are handled correctly in float64."""

    def test_int_data(self):
        S = sp.csr_matrix(np.array([[1, 0, 3], [0, 2, 0]], dtype=np.int32))
        result = _sparse_row_sum_sq(S)
        np.testing.assert_allclose(result, [10.0, 4.0], rtol=RTOL, atol=ATOL)
        assert result.dtype == np.float64

    def test_float32_data(self):
        S = sp.csr_matrix(np.array([[1.5, 0, 2.5]], dtype=np.float32))
        result = _sparse_row_sum_sq(S)
        expected = np.float64(1.5) ** 2 + np.float64(2.5) ** 2
        np.testing.assert_allclose(result, [expected], rtol=RTOL, atol=ATOL)
        assert result.dtype == np.float64


class TestSigmaSqParity:
    """End-to-end: verify sigma_sq computed with _sparse_row_sum_sq matches
    the original S.power(2).sum(axis=1) formula."""

    @pytest.fixture(params=["csr", "csc"])
    def test_matrix(self, request):
        rng = np.random.default_rng(99)
        X = rng.poisson(0.3, size=(300, 150)).astype(np.float64)
        if request.param == "csr":
            return sp.csr_matrix(X)
        return sp.csc_matrix(X)

    def test_sigma_sq_parity(self, test_matrix):
        S = test_matrix
        n_vars = S.shape[1]

        row_sums = np.asarray(S.sum(axis=1), dtype=np.float64).ravel()

        row_sum_sq_new = _sparse_row_sum_sq(S)
        row_sum_sq_old = np.asarray(S.power(2).sum(axis=1), dtype=np.float64).ravel()

        np.testing.assert_allclose(row_sum_sq_new, row_sum_sq_old, rtol=RTOL, atol=ATOL)

        mu = row_sums / n_vars
        sigma_sq_new = (row_sum_sq_new - 2.0 * mu * row_sums
                        + n_vars * mu ** 2) / (n_vars - 1)
        sigma_sq_old = (row_sum_sq_old - 2.0 * mu * row_sums
                        + n_vars * mu ** 2) / (n_vars - 1)

        np.testing.assert_allclose(sigma_sq_new, sigma_sq_old, rtol=RTOL, atol=ATOL)
