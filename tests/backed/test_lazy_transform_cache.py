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
    lt = an.create_lazy_transform(
        adata_lazy,
        key_added="workflow_log2",
        target_sum=1e4,
        log_base=2.0,
        backed_chunk_size=24,
    )
    an.reduce_kernel(
        adata_lazy,
        n_components=n_components,
        key_added="action",
        svd_algorithm="halko",
        seed=seed,
        backed_chunk_size=24,
        lazy_transform=lt,
        verbose=False,
        inplace=True,
    )
    an.correct_batch_effect(
        adata_lazy,
        batch_key="batch",
        reduction_key="action",
        backed_chunk_size=24,
        lazy_transform=lt,
        inplace=True,
    )
    adata_lazy.obsm["archetype_footprint"] = archetypes
    an.compute_archetype_feature_specificity(
        adata_lazy,
        archetype_key="archetype_footprint",
        key_added="arch_lazy",
        n_threads=1,
        backed_chunk_size=24,
        lazy_transform=lt,
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

    # The params dict in .uns should record the transform key for reproducibility.
    params = adata_lazy.uns["action_corrected_params"]
    assert params["lazy_logcounts"] is True
    assert params["lazy_transform_key"] == "workflow_log2"

    # The .uns entry written by create_lazy_transform should be a plain dict.
    uns_entry = adata_lazy.uns["workflow_log2"]
    assert isinstance(uns_entry, dict)
    assert uns_entry["target_sum"] == 1e4
    assert uns_entry["log_base"] == 2.0

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

    lt = an.create_lazy_transform(
        adata,
        target_sum=1e4,
        log_base=2.0,
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
        lazy_transform=lt,
        inplace=True,
    )
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        reduction_key="action",
        backed_chunk_size=16,
        lazy_transform=lt,
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
        lazy_transform=lt,
        inplace=True,
    )

    assert call_count["row_sums"] == 1
    _close_backed(adata)


def test_lazy_transform_cache_reused_after_reopen(tmp_path, monkeypatch):
    path = tmp_path / "persist_reopen.h5ad"
    make_test_adata(n_cells=60, n_genes=42, sparse_fmt="csr", seed=61).write_h5ad(path)

    adata = ad.read_h5ad(path, backed="r+")
    lt = an.create_lazy_transform(
        adata,
        key_added="workflow_log2",
        target_sum=1e4,
        log_base=2.0,
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
        lazy_transform=lt,
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

    # After reopen, recreate the LazyTransform from the persisted params dict.
    lt_reopen = an.create_lazy_transform(
        adata_reopen,
        key_added="workflow_log2",
        target_sum=1e4,
        log_base=2.0,
        backed_chunk_size=20,
    )
    # call_count["row_sums"] == 1 from the recreated transform init.
    # The subsequent correct_batch_effect must not trigger another row_sums call
    # because _validated starts False but only samples rows (not a full recompute).
    an.correct_batch_effect(
        adata_reopen,
        batch_key="batch",
        reduction_key="action",
        backed_chunk_size=20,
        lazy_transform=lt_reopen,
        inplace=True,
    )

    assert call_count["row_sums"] == 1
    _close_backed(adata_reopen)


def test_lazy_transform_invalidates_when_source_changes_before_first_use(tmp_path, monkeypatch):
    adata = open_backed(
        tmp_path / "invalidate",
        make_test_adata(n_cells=56, n_genes=40, sparse_fmt="dense", seed=73),
    )
    lt = an.create_lazy_transform(
        adata,
        target_sum=1e4,
        log_base=2.0,
        backed_chunk_size=20,
    )
    # Trigger a true source-matrix change on backed .X before first use.
    adata.X[0, 0] = float(adata.X[0, 0]) + 5.0

    call_count = {"row_sums": 0}
    original_row_sums = MatrixSource.row_sums

    def _count_row_sums(self, *args, **kwargs):
        call_count["row_sums"] += 1
        return original_row_sums(self, *args, **kwargs)

    monkeypatch.setattr(MatrixSource, "row_sums", _count_row_sums)

    with np.testing.assert_raises_regex(ValueError, "validation failed|fingerprint mismatch"):
        an.reduce_kernel(
            adata,
            n_components=8,
            key_added="action",
            svd_algorithm="halko",
            seed=4,
            verbose=False,
            backed_chunk_size=20,
            lazy_transform=lt,
            inplace=True,
        )

    # No full row-sum recomputation should happen during operator-time validation.
    assert call_count["row_sums"] == 0
    _close_backed(adata)


# ---------------------------------------------------------------------------
# find_markers: lazy_transform parity against explicit logcounts layer
# ---------------------------------------------------------------------------

def _top_n_genes(ranks_df: "pd.DataFrame", col: str, n: int) -> set:
    return set(ranks_df[col].sort_values().index[:n])


def test_find_markers_lazy_transform_matches_logcounts_reference(tmp_path):
    """find_markers with lazy_transform must match an explicit logcounts baseline.

    Both paths normalize with identical (target_sum=1e4, log_base=2) params;
    the top-N marker sets must overlap well across every cluster.
    """
    import pandas as pd
    from actionet.specificity import compute_feature_specificity  # noqa: F401 (smoke import)

    seed = 89
    n_cells, n_genes = 96, 72
    topn = 10

    # --- reference: explicit logcounts layer ---
    adata_ref = make_test_adata(n_cells=n_cells, n_genes=n_genes, sparse_fmt="csr", seed=seed)
    # normalize_anndata writes into an existing layer; seed it from .X first.
    adata_ref.layers["logcounts_ref"] = adata_ref.X.copy()
    an.normalize_anndata(
        adata_ref,
        target_sum=1e4,
        layer="logcounts_ref",
        log_transform=True,
        log_base=2,
        inplace=True,
    )
    ranks_ref = an.find_markers(
        adata_ref,
        labels="CellLabel",
        features_use="Gene",
        layer="logcounts_ref",
        result="ranks",
        return_type="dataframe",
    )

    # --- lazy: backed .X, no logcounts layer written ---
    adata_lazy = open_backed(
        tmp_path / "find_markers_lazy",
        make_test_adata(n_cells=n_cells, n_genes=n_genes, sparse_fmt="csr", seed=seed),
    )
    lt = an.create_lazy_transform(
        adata_lazy,
        target_sum=1e4,
        log_base=2.0,
        backed_chunk_size=32,
    )
    ranks_lazy = an.find_markers(
        adata_lazy,
        labels="CellLabel",
        features_use="Gene",
        layer=None,
        result="ranks",
        return_type="dataframe",
        backed_chunk_size=32,
        lazy_transform=lt,
    )
    _close_backed(adata_lazy)

    shared_cols = sorted(set(ranks_ref.columns) & set(ranks_lazy.columns))
    assert len(shared_cols) > 0, "No shared cluster columns between reference and lazy"

    overlaps = []
    for col in shared_cols:
        top_ref = _top_n_genes(ranks_ref, col, topn)
        top_lazy = _top_n_genes(ranks_lazy, col, topn)
        overlaps.append(len(top_ref & top_lazy) / float(topn))

    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap >= 0.80, (
        f"Lazy find_markers top-{topn} overlap vs logcounts reference: "
        f"{mean_overlap:.2f} < 0.80.  Per-cluster: {overlaps}"
    )


def test_find_markers_lazy_transform_differs_from_raw_counts(tmp_path):
    """Sanity check: lazy (normalized) specificity scores differ from raw-count scores.

    Log-normalization changes the absolute scale of counts, so the raw
    specificity score matrices must differ numerically even if the rank
    ordering of the top markers happens to be stable.
    """
    seed = 91
    n_cells, n_genes = 96, 72

    # --- raw: no normalization ---
    adata_raw = make_test_adata(n_cells=n_cells, n_genes=n_genes, sparse_fmt="csr", seed=seed)
    scores_raw = an.find_markers(
        adata_raw,
        labels="CellLabel",
        features_use="Gene",
        result="scores",
        return_type="dataframe",
    )

    # --- lazy: backed, with log2 normalization ---
    adata_lazy = open_backed(
        tmp_path / "find_markers_raw_vs_lazy",
        make_test_adata(n_cells=n_cells, n_genes=n_genes, sparse_fmt="csr", seed=seed),
    )
    lt = an.create_lazy_transform(
        adata_lazy,
        target_sum=1e4,
        log_base=2.0,
        backed_chunk_size=32,
    )
    scores_lazy = an.find_markers(
        adata_lazy,
        labels="CellLabel",
        features_use="Gene",
        result="scores",
        return_type="dataframe",
        backed_chunk_size=32,
        lazy_transform=lt,
    )
    _close_backed(adata_lazy)

    shared_cols = sorted(set(scores_raw.columns) & set(scores_lazy.columns))
    assert len(shared_cols) > 0

    # Log-normalization must change the numeric scores — compare L2 relative error.
    raw_mat = scores_raw[shared_cols].to_numpy(dtype=float)
    lazy_mat = scores_lazy[shared_cols].to_numpy(dtype=float)
    rel_err = float(np.linalg.norm(raw_mat - lazy_mat) / (np.linalg.norm(raw_mat) + 1e-12))
    assert rel_err > 0.05, (
        f"Lazy (normalized) and raw specificity scores are nearly identical "
        f"(relative error {rel_err:.4f}); the lazy_transform appears to have had no effect."
    )


def test_annotate_cells_lazy_transform_runs_without_error(tmp_path):
    """annotate_cells with method='vision' and lazy_transform must run end-to-end."""
    seed = 97
    n_cells, n_genes = 96, 72

    # Build in-memory reference to get a network and marker table.
    adata_ref = make_test_adata(n_cells=n_cells, n_genes=n_genes, sparse_fmt="csr", seed=seed)
    an.normalize_anndata(adata_ref, target_sum=1e4, layer="logcounts", log_transform=True, log_base=2, inplace=True)
    an.reduce_kernel(adata_ref, n_components=10, layer="logcounts", key_added="action", seed=seed, inplace=True)
    an.correct_batch_effect(adata_ref, batch_key="batch", reduction_key="action", layer="logcounts", inplace=True)
    an.run_actionet(adata_ref, layer="logcounts", reduction_key="action_corrected", k_min=2, k_max=8,
                    layout_3d=False, n_threads=1, seed=seed, inplace=True)
    markers = an.find_markers(adata_ref, labels="CellLabel", features_use="Gene",
                               layer="logcounts", top_genes=6, return_type="dataframe")

    # Now use backed lazy path for annotate_cells.
    adata_lazy = open_backed(
        tmp_path / "annotate_cells_lazy",
        make_test_adata(n_cells=n_cells, n_genes=n_genes, sparse_fmt="csr", seed=seed),
    )
    # Copy the network built on the reference into the backed object.
    adata_lazy.obsp["actionet"] = adata_ref.obsp["actionet"]

    lt = an.create_lazy_transform(adata_lazy, target_sum=1e4, log_base=2.0, backed_chunk_size=32)
    result = an.annotate_cells(
        adata_lazy,
        markers,
        method="vision",
        features_use="Gene",
        layer=None,
        n_threads=1,
        backed_chunk_size=32,
        lazy_transform=lt,
    )
    _close_backed(adata_lazy)

    assert "labels" in result
    assert "enrichment" in result
    assert len(result["labels"]) == n_cells
    assert result["enrichment"].shape[0] == n_cells
