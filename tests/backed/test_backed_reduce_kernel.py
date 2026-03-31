"""Backed reduce_kernel smoke test (originally tests/test_backed.py)."""

import numpy as np
import pytest
import actionet as an

from .conftest import make_test_adata, open_backed


def test_backed_reduce_kernel_smoke(tmp_path):
    """reduce_kernel runs on a backed AnnData without error."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=99))

    an.normalize_anndata(adata, target_sum=1e4, backed_chunk_size=32, inplace=True)

    an.reduce_kernel(
        adata,
        n_components=10,
        layer=None,
        key_added="action",
        backed_chunk_size=32,
        inplace=True,
    )

    assert "action" in adata.obsm
    assert adata.obsm["action"].shape == (adata.n_obs, 10)
    assert not np.any(np.isnan(adata.obsm["action"]))


def _procrustes_rel_error(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """Relative Frobenius error after best orthogonal alignment."""
    U, _, VT = np.linalg.svd(X_test.T @ X_ref, full_matrices=False)
    Q = U @ VT
    denom = np.linalg.norm(X_ref, ord="fro") + 1e-12
    return float(np.linalg.norm(X_test @ Q - X_ref, ord="fro") / denom)


@pytest.mark.parametrize("sparse_fmt", ["csr", "csc"])
def test_backed_reduce_kernel_from_svd_parity(tmp_path, sparse_fmt):
    """Backed reduce_kernel_from_svd matches in-memory reduction output."""
    n_components = 12

    adata_mem = make_test_adata(n_cells=96, n_genes=72, sparse_fmt=sparse_fmt, seed=123)
    an.normalize_anndata(adata_mem, target_sum=1e4, inplace=True)
    svd = an.run_svd(
        adata_mem,
        n_components=n_components,
        algorithm="halko",
        verbose=False,
    )

    an.reduce_kernel(
        adata_mem,
        n_components=n_components,
        precomputed_svd=svd,
        key_added="action_mem",
        verbose=False,
        inplace=True,
    )
    action_mem = np.asarray(adata_mem.obsm["action_mem"], dtype=float)
    sigma_mem = np.asarray(adata_mem.uns["action_mem_params"]["sigma"], dtype=float).reshape(-1)

    adata_backed = open_backed(tmp_path / sparse_fmt, adata_mem)
    an.reduce_kernel(
        adata_backed,
        n_components=n_components,
        precomputed_svd=svd,
        key_added="action_backed",
        backed_chunk_size=24,
        verbose=False,
        inplace=True,
    )
    action_backed = np.asarray(adata_backed.obsm["action_backed"], dtype=float)
    sigma_backed = np.asarray(adata_backed.uns["action_backed_params"]["sigma"], dtype=float).reshape(-1)

    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    sigma_rel = np.linalg.norm(sigma_backed - sigma_mem) / (np.linalg.norm(sigma_mem) + 1e-12)
    action_rel = _procrustes_rel_error(action_mem, action_backed)

    assert sigma_rel < 1e-10
    assert action_rel < 1e-10


def test_backed_reduce_kernel_first_pass_matches_second_pass_after_backed_normalize(tmp_path):
    """First backed reduce_kernel pass must not lag behind the second pass."""
    n_components = 12

    adata_mem = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=123)
    an.normalize_anndata(adata_mem, target_sum=1e4, inplace=True)
    an.reduce_kernel(
        adata_mem,
        n_components=n_components,
        key_added="action_mem",
        svd_algorithm="halko",
        seed=2,
        verbose=False,
        inplace=True,
    )

    adata_backed = open_backed(tmp_path / "first_vs_second", make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=123))
    an.normalize_anndata(adata_backed, target_sum=1e4, backed_chunk_size=24, inplace=True)

    an.reduce_kernel(
        adata_backed,
        n_components=n_components,
        key_added="action_first",
        svd_algorithm="halko",
        seed=2,
        backed_chunk_size=24,
        backed_n_threads=1,
        verbose=False,
        inplace=True,
    )
    first = np.asarray(adata_backed.obsm["action_first"], dtype=float)
    sigma_first = np.asarray(adata_backed.uns["action_first_params"]["sigma"], dtype=float).reshape(-1)

    an.reduce_kernel(
        adata_backed,
        n_components=n_components,
        key_added="action_second",
        svd_algorithm="halko",
        seed=2,
        backed_chunk_size=24,
        backed_n_threads=1,
        verbose=False,
        inplace=True,
    )
    second = np.asarray(adata_backed.obsm["action_second"], dtype=float)
    sigma_second = np.asarray(adata_backed.uns["action_second_params"]["sigma"], dtype=float).reshape(-1)

    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    sigma_rel = np.linalg.norm(sigma_second - sigma_first) / (np.linalg.norm(sigma_second) + 1e-12)
    action_rel = _procrustes_rel_error(second, first)

    assert sigma_rel < 1e-10
    assert action_rel < 1e-10
