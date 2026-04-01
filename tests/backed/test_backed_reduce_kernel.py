"""Backed reduce_kernel smoke test (originally tests/test_backed.py)."""

import numpy as np
import pytest
import actionet as an
import actionet.core as actionet_core

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


def test_flush_backed_handle_raises_on_flush_failure():
    """Flush failures should fail fast to avoid stale operator reads."""
    class _BrokenHandle:
        def flush(self):
            raise OSError("simulated flush failure")

    class _DummyFile:
        _file = _BrokenHandle()

    class _DummyAdata:
        isbacked = True
        file = _DummyFile()

    with pytest.raises(RuntimeError, match="failed to flush backed AnnData handle"):
        actionet_core._flush_backed_handle(_DummyAdata(), context="reduce_kernel")


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


@pytest.mark.parametrize("sparse_fmt", ["csr", "csc", "dense"])
def test_run_svd_backed_lazy_logcounts_matches_eager_normalized(tmp_path, sparse_fmt):
    """Lazy-backed run_svd should match eager-normalized reference SVD."""
    n_components = 10
    seed = 11

    adata_ref = make_test_adata(n_cells=96, n_genes=72, sparse_fmt=sparse_fmt, seed=321)
    an.normalize_anndata(
        adata_ref,
        target_sum=1e4,
        log_transform=True,
        log_base=2,
        inplace=True,
    )
    svd_ref = an.run_svd(
        adata_ref,
        n_components=n_components,
        algorithm="halko",
        seed=seed,
        verbose=False,
    )

    adata_backed = open_backed(
        tmp_path / f"lazy_svd_{sparse_fmt}",
        make_test_adata(n_cells=96, n_genes=72, sparse_fmt=sparse_fmt, seed=321),
    )
    lazy_transform = an.create_lazy_transform(adata_backed, target_sum=1e4, log_base=2.0)
    svd_lazy = an.run_svd(
        adata_backed,
        n_components=n_components,
        algorithm="halko",
        seed=seed,
        verbose=False,
        backed_chunk_size=24,
        lazy_transform=lazy_transform,
    )

    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    sigma_ref = np.asarray(svd_ref["d"], dtype=float).reshape(-1)
    sigma_lazy = np.asarray(svd_lazy["d"], dtype=float).reshape(-1)
    sigma_rel = np.linalg.norm(sigma_lazy - sigma_ref) / (np.linalg.norm(sigma_ref) + 1e-12)
    assert sigma_rel < 1e-4


@pytest.mark.parametrize("sparse_fmt", ["csr", "csc", "dense"])
def test_reduce_kernel_backed_lazy_logcounts_matches_eager_normalized(tmp_path, sparse_fmt):
    """Lazy-backed reduce_kernel should match eager-normalized reference results."""
    n_components = 12
    seed = 5

    adata_ref = make_test_adata(n_cells=96, n_genes=72, sparse_fmt=sparse_fmt, seed=456)
    an.normalize_anndata(
        adata_ref,
        target_sum=1e4,
        log_transform=True,
        log_base=2,
        inplace=True,
    )
    an.reduce_kernel(
        adata_ref,
        n_components=n_components,
        key_added="action_ref",
        svd_algorithm="halko",
        seed=seed,
        verbose=False,
        inplace=True,
    )
    action_ref = np.asarray(adata_ref.obsm["action_ref"], dtype=float)
    sigma_ref = np.asarray(adata_ref.uns["action_ref_params"]["sigma"], dtype=float).reshape(-1)

    adata_backed = open_backed(
        tmp_path / f"lazy_reduce_{sparse_fmt}",
        make_test_adata(n_cells=96, n_genes=72, sparse_fmt=sparse_fmt, seed=456),
    )
    lazy_transform = an.create_lazy_transform(adata_backed, target_sum=1e4, log_base=2.0)
    an.reduce_kernel(
        adata_backed,
        n_components=n_components,
        key_added="action_lazy",
        svd_algorithm="halko",
        seed=seed,
        verbose=False,
        backed_chunk_size=24,
        lazy_transform=lazy_transform,
        inplace=True,
    )
    action_lazy = np.asarray(adata_backed.obsm["action_lazy"], dtype=float)
    sigma_lazy = np.asarray(adata_backed.uns["action_lazy_params"]["sigma"], dtype=float).reshape(-1)
    params = adata_backed.uns["action_lazy_params"]

    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    sigma_rel = np.linalg.norm(sigma_lazy - sigma_ref) / (np.linalg.norm(sigma_ref) + 1e-12)
    action_rel = _procrustes_rel_error(action_ref, action_lazy)
    assert sigma_rel < 1e-4
    assert action_rel < 1e-3
    assert params["lazy_logcounts"] is True
    assert np.isclose(float(params["lazy_target_sum"]), 1e4)
    assert np.isclose(float(params["lazy_log_base"]), 2.0)
    assert np.isclose(float(params["lazy_pseudocount"]), 1.0)


def test_reduce_kernel_from_svd_backed_lazy_logcounts_matches_eager(tmp_path):
    """Lazy-backed reduce_kernel_from_svd matches eager-normalized reference."""
    n_components = 12

    adata_ref = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=999)
    an.normalize_anndata(
        adata_ref,
        target_sum=1e4,
        log_transform=True,
        log_base=2,
        inplace=True,
    )
    svd_ref = an.run_svd(
        adata_ref,
        n_components=n_components,
        algorithm="halko",
        seed=3,
        verbose=False,
    )
    an.reduce_kernel_from_svd(
        adata_ref,
        svd_result=svd_ref,
        key_added="action_ref",
        verbose=False,
        inplace=True,
    )
    action_ref = np.asarray(adata_ref.obsm["action_ref"], dtype=float)
    sigma_ref = np.asarray(adata_ref.uns["action_ref_params"]["sigma"], dtype=float).reshape(-1)

    adata_backed = open_backed(
        tmp_path / "lazy_reduce_from_svd",
        make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=999),
    )
    lazy_transform = an.create_lazy_transform(adata_backed, target_sum=1e4, log_base=2.0)
    an.reduce_kernel_from_svd(
        adata_backed,
        svd_result=svd_ref,
        key_added="action_lazy",
        verbose=False,
        backed_chunk_size=24,
        lazy_transform=lazy_transform,
        inplace=True,
    )
    action_lazy = np.asarray(adata_backed.obsm["action_lazy"], dtype=float)
    sigma_lazy = np.asarray(adata_backed.uns["action_lazy_params"]["sigma"], dtype=float).reshape(-1)

    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    sigma_rel = np.linalg.norm(sigma_lazy - sigma_ref) / (np.linalg.norm(sigma_ref) + 1e-12)
    action_rel = _procrustes_rel_error(action_ref, action_lazy)
    assert sigma_rel < 1e-4
    assert action_rel < 1e-3


def test_lazy_logcounts_validation_errors(tmp_path):
    """Invalid lazy settings should raise informative errors."""
    adata = make_test_adata(n_cells=24, n_genes=20, sparse_fmt="csr", seed=7)
    adata_backed = open_backed(tmp_path / "lazy_validation", adata.copy())

    with pytest.raises(ValueError, match="lazy_target_sum"):
        an.create_lazy_transform(adata_backed, target_sum=0)

    with pytest.raises(ValueError, match="lazy_log_base"):
        an.create_lazy_transform(adata_backed, log_base=1.0)

    with pytest.raises(ValueError, match="lazy_pseudocount"):
        an.create_lazy_transform(adata_backed, pseudocount=0.5)

    lazy_transform = an.create_lazy_transform(adata_backed, target_sum=1e4, log_base=2.0)

    with pytest.raises(ValueError, match="supported only for backed"):
        an.reduce_kernel(
            adata,
            n_components=5,
            lazy_transform=lazy_transform,
            inplace=False,
        )

    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()
