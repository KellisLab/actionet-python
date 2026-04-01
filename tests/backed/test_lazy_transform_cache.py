"""Stage-2A tests for reusable lazy backed transforms."""

import anndata as ad
import numpy as np

import actionet as an
from actionet._matrix_source import MatrixSource

from .conftest import make_test_adata, open_backed


def _close_backed(adata) -> None:
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


def _relative_error(x_ref: np.ndarray, x_test: np.ndarray) -> float:
    x_ref = np.asarray(x_ref, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    return float(np.linalg.norm(x_test - x_ref) / (np.linalg.norm(x_ref) + 1e-12))


def _procrustes_rel_error(x_ref: np.ndarray, x_test: np.ndarray) -> float:
    u, _, vt = np.linalg.svd(x_test.T @ x_ref, full_matrices=False)
    q = u @ vt
    return _relative_error(x_ref, x_test @ q)


def test_minimal_lazy_workflow_matches_eager_layer_reference(tmp_path):
    n_components = 10
    seed = 7
    archetypes = np.random.default_rng(17).random((72, 4))

    adata_ref = make_test_adata(n_cells=72, n_genes=54, sparse_fmt="csr", seed=44)
    an.normalize_anndata(
        adata_ref,
        target_sum=1e4,
        layer="logcounts",
        log_transform=True,
        log_base=2,
        inplace=True,
    )
    an.reduce_kernel(
        adata_ref,
        n_components=n_components,
        layer="logcounts",
        key_added="action",
        svd_algorithm="halko",
        seed=seed,
        verbose=False,
        inplace=True,
    )
    an.correct_batch_effect(
        adata_ref,
        batch_key="batch",
        reduction_key="action",
        layer="logcounts",
        backed_chunk_size=24,
        inplace=True,
    )
    adata_ref.obsm["archetype_footprint"] = archetypes
    an.compute_archetype_feature_specificity(
        adata_ref,
        archetype_key="archetype_footprint",
        layer="logcounts",
        key_added="arch_ref",
        n_threads=1,
        inplace=True,
    )

    adata_lazy = open_backed(
        tmp_path / "lazy_workflow",
        make_test_adata(n_cells=72, n_genes=54, sparse_fmt="csr", seed=44),
    )
    lazy_transform = an.create_lazy_transform(
        adata_lazy,
        target_sum=1e4,
        log_base=2.0,
        key="workflow_log2",
        backed_chunk_size=24,
    )
    an.reduce_kernel(
        adata_lazy,
        n_components=n_components,
        key_added="action",
        svd_algorithm="halko",
        seed=seed,
        backed_chunk_size=24,
        lazy_transform=lazy_transform,
        verbose=False,
        inplace=True,
    )
    an.correct_batch_effect(
        adata_lazy,
        batch_key="batch",
        reduction_key="action",
        backed_chunk_size=24,
        lazy_transform=lazy_transform,
        inplace=True,
    )
    adata_lazy.obsm["archetype_footprint"] = archetypes
    an.compute_archetype_feature_specificity(
        adata_lazy,
        archetype_key="archetype_footprint",
        key_added="arch_lazy",
        n_threads=1,
        backed_chunk_size=24,
        lazy_transform=lazy_transform,
        inplace=True,
    )

    sigma_ref = np.asarray(adata_ref.uns["action_corrected_params"]["sigma"], dtype=float).reshape(-1)
    sigma_lazy = np.asarray(adata_lazy.uns["action_corrected_params"]["sigma"], dtype=float).reshape(-1)
    sr_ref = np.asarray(adata_ref.obsm["action_corrected"], dtype=float)
    sr_lazy = np.asarray(adata_lazy.obsm["action_corrected"], dtype=float)

    np.testing.assert_allclose(
        np.sort(sigma_lazy)[::-1],
        np.sort(sigma_ref)[::-1],
        rtol=5e-4,
    )
    assert _procrustes_rel_error(sr_ref, sr_lazy) < 5e-3

    prof_ref = np.asarray(adata_ref.varm["arch_ref_feat_profile"], dtype=float)
    prof_lazy = np.asarray(adata_lazy.varm["arch_lazy_feat_profile"], dtype=float)
    upper_ref = np.asarray(adata_ref.varm["arch_ref_feat_specificity_upper"], dtype=float)
    upper_lazy = np.asarray(adata_lazy.varm["arch_lazy_feat_specificity_upper"], dtype=float)
    lower_ref = np.asarray(adata_ref.varm["arch_ref_feat_specificity_lower"], dtype=float)
    lower_lazy = np.asarray(adata_lazy.varm["arch_lazy_feat_specificity_lower"], dtype=float)

    assert _relative_error(prof_ref, prof_lazy) < 1e-2
    assert _relative_error(upper_ref, upper_lazy) < 1e-2
    assert _relative_error(lower_ref, lower_lazy) < 1e-2

    params = adata_lazy.uns["action_corrected_params"]
    assert params["lazy_logcounts"] is True
    assert params["lazy_transform_key"] == "workflow_log2"

    _close_backed(adata_lazy)


def test_lazy_transform_cache_reused_across_reduce_correct_and_run_actionet(tmp_path, monkeypatch):
    adata = open_backed(
        tmp_path / "cache_reuse",
        make_test_adata(n_cells=48, n_genes=36, sparse_fmt="csr", seed=55),
    )

    call_count = {"row_sums": 0}
    original_row_sums = MatrixSource.row_sums

    def _count_row_sums(self, *args, **kwargs):
        call_count["row_sums"] += 1
        return original_row_sums(self, *args, **kwargs)

    monkeypatch.setattr(MatrixSource, "row_sums", _count_row_sums)

    lazy_transform = an.create_lazy_transform(
        adata,
        target_sum=1e4,
        log_base=2.0,
        key="workflow_log2",
        backed_chunk_size=16,
    )
    an.reduce_kernel(
        adata,
        n_components=8,
        key_added="action",
        svd_algorithm="halko",
        seed=3,
        verbose=False,
        backed_chunk_size=16,
        lazy_transform=lazy_transform,
        inplace=True,
    )
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        reduction_key="action",
        backed_chunk_size=16,
        lazy_transform=lazy_transform,
        inplace=True,
    )
    an.run_actionet(
        adata,
        reduction_key="action_corrected",
        layout_3d=False,
        k_min=2,
        k_max=5,
        network_k=10,
        n_threads=1,
        seed=3,
        backed_chunk_size=16,
        lazy_transform=lazy_transform,
        inplace=True,
    )

    assert call_count["row_sums"] == 1
    _close_backed(adata)


def test_lazy_transform_cache_reused_after_reopen(tmp_path, monkeypatch):
    path = tmp_path / "persist_reopen.h5ad"
    make_test_adata(n_cells=60, n_genes=42, sparse_fmt="csr", seed=61).write_h5ad(path)

    adata = ad.read_h5ad(path, backed="r+")
    lazy_transform = an.create_lazy_transform(
        adata,
        target_sum=1e4,
        log_base=2.0,
        key="workflow_log2",
        backed_chunk_size=20,
    )
    an.reduce_kernel(
        adata,
        n_components=8,
        key_added="action",
        svd_algorithm="halko",
        seed=5,
        verbose=False,
        backed_chunk_size=20,
        lazy_transform=lazy_transform,
        inplace=True,
    )
    _close_backed(adata)

    adata_reopen = ad.read_h5ad(path, backed="r+")
    params = adata_reopen.uns["action_params"]
    assert params["lazy_transform_key"] == "workflow_log2"

    call_count = {"row_sums": 0}
    original_row_sums = MatrixSource.row_sums

    def _count_row_sums(self, *args, **kwargs):
        call_count["row_sums"] += 1
        return original_row_sums(self, *args, **kwargs)

    monkeypatch.setattr(MatrixSource, "row_sums", _count_row_sums)

    an.correct_batch_effect(
        adata_reopen,
        batch_key="batch",
        reduction_key="action",
        backed_chunk_size=20,
        lazy_transform=lazy_transform,
        inplace=True,
    )

    assert call_count["row_sums"] == 0
    _close_backed(adata_reopen)


def test_lazy_transform_cache_invalidates_when_source_matrix_changes(tmp_path, monkeypatch):
    adata = open_backed(
        tmp_path / "invalidate",
        make_test_adata(n_cells=56, n_genes=40, sparse_fmt="dense", seed=73),
    )
    lazy_transform = an.create_lazy_transform(
        adata,
        target_sum=1e4,
        log_base=2.0,
        key="workflow_log2",
        backed_chunk_size=20,
    )
    an.reduce_kernel(
        adata,
        n_components=8,
        key_added="action",
        svd_algorithm="halko",
        seed=4,
        verbose=False,
        backed_chunk_size=20,
        lazy_transform=lazy_transform,
        inplace=True,
    )

    # Trigger a true source-matrix change on backed .X.
    adata.X[0, 0] = float(adata.X[0, 0]) + 5.0

    call_count = {"row_sums": 0}
    original_row_sums = MatrixSource.row_sums

    def _count_row_sums(self, *args, **kwargs):
        call_count["row_sums"] += 1
        return original_row_sums(self, *args, **kwargs)

    monkeypatch.setattr(MatrixSource, "row_sums", _count_row_sums)

    with np.testing.assert_raises_regex(ValueError, "validation failed|fingerprint mismatch"):
        an.correct_batch_effect(
            adata,
            batch_key="batch",
            reduction_key="action",
            backed_chunk_size=20,
            lazy_transform=lazy_transform,
            inplace=True,
        )

    # No full row-sum recomputation should happen during operator-time validation.
    assert call_count["row_sums"] == 0
    _close_backed(adata)
