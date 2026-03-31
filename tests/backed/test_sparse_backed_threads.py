"""Sparse-backed threading parity tests for backed operator SVD/reduction."""

import numpy as np

import actionet as an

from .conftest import make_test_adata, open_backed


def test_sparse_backed_svd_thread_parity_csr(tmp_path):
    """CSR-backed singular values remain stable across backed thread settings."""
    adata_t1 = open_backed(tmp_path / "t1", make_test_adata(sparse_fmt="csr", seed=101))
    adata_t4 = open_backed(tmp_path / "t4", make_test_adata(sparse_fmt="csr", seed=101))

    k = 10
    res_t1 = an.run_svd(adata_t1, n_components=k, backed_chunk_size=32, backed_n_threads=1)
    res_t4 = an.run_svd(adata_t4, n_components=k, backed_chunk_size=32, backed_n_threads=4)

    np.testing.assert_allclose(
        np.sort(np.asarray(res_t4["d"]).reshape(-1))[::-1],
        np.sort(np.asarray(res_t1["d"]).reshape(-1))[::-1],
        rtol=1e-4,
    )


def test_sparse_backed_svd_thread_parity_csc(tmp_path):
    """CSC-backed singular values remain stable across backed thread settings."""
    adata_t1 = open_backed(tmp_path / "t1", make_test_adata(sparse_fmt="csc", seed=102))
    adata_t4 = open_backed(tmp_path / "t4", make_test_adata(sparse_fmt="csc", seed=102))

    k = 10
    res_t1 = an.run_svd(adata_t1, n_components=k, backed_chunk_size=32, backed_n_threads=1)
    res_t4 = an.run_svd(adata_t4, n_components=k, backed_chunk_size=32, backed_n_threads=4)

    np.testing.assert_allclose(
        np.sort(np.asarray(res_t4["d"]).reshape(-1))[::-1],
        np.sort(np.asarray(res_t1["d"]).reshape(-1))[::-1],
        rtol=1e-4,
    )


def test_sparse_backed_reduce_kernel_thread_parity(tmp_path):
    """CSR-backed reduce_kernel sigma remains stable across thread settings."""
    adata_t1 = open_backed(tmp_path / "t1", make_test_adata(sparse_fmt="csr", seed=103))
    adata_t4 = open_backed(tmp_path / "t4", make_test_adata(sparse_fmt="csr", seed=103))

    k = 10
    an.reduce_kernel(adata_t1, n_components=k, backed_chunk_size=32, backed_n_threads=1, inplace=True)
    an.reduce_kernel(adata_t4, n_components=k, backed_chunk_size=32, backed_n_threads=4, inplace=True)

    sigma_t1 = np.asarray(adata_t1.uns["action_params"]["sigma"]).reshape(-1)
    sigma_t4 = np.asarray(adata_t4.uns["action_params"]["sigma"]).reshape(-1)
    np.testing.assert_allclose(
        np.sort(sigma_t4)[::-1],
        np.sort(sigma_t1)[::-1],
        rtol=1e-4,
    )


def test_sparse_backed_svd_parity_vs_inmemory_csr(tmp_path):
    """CSR-backed run_svd singular values match in-memory reference."""
    adata_mem = make_test_adata(sparse_fmt="csr", seed=104)
    adata_backed = open_backed(tmp_path / "parity_svd", make_test_adata(sparse_fmt="csr", seed=104))

    k = 10
    res_mem = an.run_svd(adata_mem, n_components=k, algorithm="halko", seed=123, verbose=False)
    res_backed = an.run_svd(
        adata_backed,
        n_components=k,
        algorithm="halko",
        seed=123,
        verbose=False,
        backed_chunk_size=32,
        backed_n_threads=1,
    )

    np.testing.assert_allclose(
        np.sort(np.asarray(res_backed["d"]).reshape(-1))[::-1],
        np.sort(np.asarray(res_mem["d"]).reshape(-1))[::-1],
        rtol=1e-4,
    )


def test_sparse_backed_reduce_kernel_parity_vs_inmemory_csr(tmp_path):
    """CSR-backed reduce_kernel sigma matches in-memory reference."""
    adata_mem = make_test_adata(sparse_fmt="csr", seed=105)
    adata_backed = open_backed(tmp_path / "parity_reduce", make_test_adata(sparse_fmt="csr", seed=105))

    k = 10
    an.reduce_kernel(
        adata_mem,
        n_components=k,
        svd_algorithm="halko",
        seed=234,
        verbose=False,
        inplace=True,
    )
    an.reduce_kernel(
        adata_backed,
        n_components=k,
        svd_algorithm="halko",
        seed=234,
        verbose=False,
        backed_chunk_size=32,
        backed_n_threads=1,
        inplace=True,
    )

    sigma_mem = np.asarray(adata_mem.uns["action_params"]["sigma"]).reshape(-1)
    sigma_backed = np.asarray(adata_backed.uns["action_params"]["sigma"]).reshape(-1)
    np.testing.assert_allclose(
        np.sort(sigma_backed)[::-1],
        np.sort(sigma_mem)[::-1],
        rtol=1e-4,
    )
