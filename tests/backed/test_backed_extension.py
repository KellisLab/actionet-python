"""End-to-end tests for backed AnnData extension.

Tests cover:
  - Full pipeline (preprocessing -> reduce_kernel -> batch correction ->
    run_actionet -> markers -> annotation -> imputation) on backed .X and
    backed layers.
  - Persistence after close / reopen.
  - Preprocessing correctness (CSR and CSC).
  - Parity between in-memory and backed marker ranking / imputation.
  - Backed basal-expression correction.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pytest

import actionet as an
from actionet import _core

from .conftest import make_test_adata, open_backed, MatrixLike

_has_perturbed_svd_with_prior = hasattr(_core, "perturbed_svd_with_prior")
requires_rebuilt_ext = pytest.mark.skipif(
    not _has_perturbed_svd_with_prior,
    reason="C++ extension needs rebuild to expose perturbed_svd_with_prior",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_backed_workflow(
    adata: ad.AnnData, layer: str | None = None
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run the standard backed E2E workflow and return (markers, annot, imp)."""
    an.normalize_ace(adata, target_sum=1e4, layer=layer, backed_chunk_size=32, inplace=True)
    an.log1p_ace(adata, layer=layer, base=2, backed_chunk_size=32, inplace=True)

    an.reduce_kernel(
        adata,
        n_components=16,
        layer=layer,
        key_added="action",
        backed_chunk_size=32,
        inplace=True,
    )
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        reduction_key="action",
        layer=layer,
        corrected_suffix="corrected",
        backed_chunk_size=32,
        inplace=True,
    )
    an.run_actionet(
        adata,
        layer=layer,
        reduction_key="action_corrected",
        k_min=2,
        k_max=10,
        layout_3d=False,
        n_threads=1,
        seed=1,
        backed_chunk_size=32,
        inplace=True,
    )

    markers = an.find_markers(
        adata,
        labels="CellLabel",
        features_use="Gene",
        layer=layer,
        top_genes=6,
        return_type="dataframe",
        backed_chunk_size=32,
    )

    annot = an.annotate_cells(
        adata,
        markers,
        method="actionet",
        features_use="Gene",
        layer=layer,
        n_threads=1,
        backed_chunk_size=32,
    )

    use_feats = [f for f in markers.iloc[:, 0].dropna().tolist()[:4] if f in adata.var_names]
    imp = an.impute_features(
        adata,
        features=use_feats,
        features_use="Gene",
        layer=layer,
        reduction_key="action_corrected",
        n_threads=1,
        backed_chunk_size=32,
    )
    return markers, annot, imp


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

@requires_rebuilt_ext
def test_backed_e2e_x(tmp_path):
    """Full pipeline on backed .X (CSR)."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr"))
    markers, annot, imp = _run_backed_workflow(adata, layer=None)

    assert markers.shape[1] >= 2
    assert len(annot["labels"]) == adata.n_obs
    assert imp.shape[0] == adata.n_obs
    assert imp.shape[1] > 0


@requires_rebuilt_ext
def test_backed_e2e_layer(tmp_path):
    """Full pipeline on backed layer='logcounts' (CSR)."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr"))
    markers, annot, imp = _run_backed_workflow(adata, layer="logcounts")

    assert markers.shape[1] >= 2
    assert len(annot["labels"]) == adata.n_obs
    assert imp.shape[0] == adata.n_obs
    assert imp.shape[1] > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@requires_rebuilt_ext
def test_backed_persistence_after_reopen(tmp_path):
    """Keys written during backed workflow survive close+reopen."""
    path = tmp_path / "persist_backed.h5ad"
    adata0 = make_test_adata(sparse_fmt="csr")
    adata0.write_h5ad(path)

    adata = ad.read_h5ad(path, backed="r+")
    _run_backed_workflow(adata, layer="logcounts")
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    reopened = ad.read_h5ad(path, backed="r")

    assert "action" in reopened.obsm
    assert "action_B" in reopened.obsm
    assert "action_corrected" in reopened.obsm
    assert "action_corrected_B" in reopened.obsm

    assert "action_U" in reopened.varm
    assert "action_A" in reopened.varm
    assert "action_corrected_U" in reopened.varm
    assert "action_corrected_A" in reopened.varm

    assert "actionet" in reopened.obsp
    assert "action_corrected_params" in reopened.uns
    assert "logcounts" in reopened.layers


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_backed_filter_anndata_inplace(tmp_path):
    """filter_anndata with inplace=True on backed AnnData rewrites the file."""
    adata_mem = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=55)
    adata = open_backed(tmp_path, adata_mem)
    orig_shape = adata.shape

    an.filter_anndata(adata, min_cells_per_feat=3, inplace=True, backed_chunk_size=32)

    # Some genes should have been removed (very sparse Poisson data)
    assert adata.n_vars <= orig_shape[1]
    assert adata.n_obs <= orig_shape[0]
    # Object should still be backed
    assert getattr(adata, "isbacked", False)


def test_backed_filter_anndata_not_inplace(tmp_path):
    """filter_anndata with inplace=False on backed AnnData returns in-memory."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=56))
    result = an.filter_anndata(adata, min_cells_per_feat=3, inplace=False, backed_chunk_size=32)
    assert result is not None
    assert result.n_vars <= adata.n_vars


def test_backed_preprocessing_csr_and_csc(tmp_path):
    """normalize_ace + log1p_ace produce correct row sums for CSR and CSC.

    Note: CSC-backed X triggers h5py 'increasing order' errors on row-slice
    read, so we test CSC via a layer rather than .X directly.
    """
    from actionet._matrix_source import MatrixSource

    for fmt in ("csr", "csc"):
        adata_mem = make_test_adata(sparse_fmt=fmt, seed=17 if fmt == "csr" else 23)
        adata = open_backed(tmp_path / fmt, adata_mem)

        target_layer = "logcounts" if fmt == "csc" else None
        an.normalize_ace(adata, target_sum=1e4, layer=target_layer, backed_chunk_size=24, inplace=True)
        src = MatrixSource(adata, layer=target_layer)
        row_sums = src.row_sums(chunk_size=24)
        nonzero_rows = row_sums > 0
        assert np.allclose(row_sums[nonzero_rows], 1e4, rtol=1e-2, atol=1e-2)

        an.log1p_ace(adata, layer=target_layer, base=2, backed_chunk_size=24, inplace=True)


# ---------------------------------------------------------------------------
# Parity: backed vs in-memory
# ---------------------------------------------------------------------------

@requires_rebuilt_ext
def test_backed_parity_markers_and_imputation(tmp_path):
    """Marker ranking overlap and imputation correlation between modes."""
    adata_mem = make_test_adata(sparse_fmt="csr", seed=31)
    adata_backed = open_backed(tmp_path, adata_mem)

    for obj in (adata_mem, adata_backed):
        an.normalize_ace(obj, target_sum=1e4, layer="logcounts", backed_chunk_size=24, inplace=True)
        an.log1p_ace(obj, layer="logcounts", base=2, backed_chunk_size=24, inplace=True)

    an.reduce_kernel(adata_mem, n_components=14, layer="logcounts", key_added="action", seed=2, inplace=True)
    an.correct_batch_effect(adata_mem, batch_key="batch", reduction_key="action", layer="logcounts", backed_chunk_size=24, inplace=True)
    an.run_actionet(adata_mem, layer="logcounts", reduction_key="action_corrected", k_min=2, k_max=8, layout_3d=False, seed=2, n_threads=1, backed_chunk_size=24, inplace=True)

    an.reduce_kernel(adata_backed, n_components=14, layer="logcounts", key_added="action", seed=2, backed_chunk_size=24, inplace=True)
    an.correct_batch_effect(adata_backed, batch_key="batch", reduction_key="action", layer="logcounts", backed_chunk_size=24, inplace=True)
    an.run_actionet(adata_backed, layer="logcounts", reduction_key="action_corrected", k_min=2, k_max=8, layout_3d=False, seed=2, n_threads=1, backed_chunk_size=24, inplace=True)

    ranks_mem = an.find_markers(adata_mem, labels="CellLabel", features_use="Gene", layer="logcounts", result="ranks", return_type="dataframe", backed_chunk_size=24)
    ranks_backed = an.find_markers(adata_backed, labels="CellLabel", features_use="Gene", layer="logcounts", result="ranks", return_type="dataframe", backed_chunk_size=24)

    topn = 12
    overlaps = []
    for col in sorted(set(ranks_mem.columns) & set(ranks_backed.columns)):
        top_mem = set(ranks_mem[col].sort_values().index[:topn])
        top_backed = set(ranks_backed[col].sort_values().index[:topn])
        overlaps.append(len(top_mem & top_backed) / float(topn))

    assert len(overlaps) > 0
    assert np.mean(overlaps) >= 0.60

    feat_list = list(ranks_mem.index[:6])
    imp_mem = an.impute_features(adata_mem, features=feat_list, features_use="Gene", layer="logcounts", reduction_key="action_corrected", backed_chunk_size=24)
    imp_backed = an.impute_features(adata_backed, features=feat_list, features_use="Gene", layer="logcounts", reduction_key="action_corrected", backed_chunk_size=24)

    corrs = []
    for feat in feat_list:
        x = np.asarray(imp_mem[feat], dtype=float)
        y = np.asarray(imp_backed[feat], dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        corrs.append(np.corrcoef(x, y)[0, 1])

    assert len(corrs) > 0
    assert np.mean(corrs) >= 0.90


# ---------------------------------------------------------------------------
# Batch correction: basal expression (backed path)
# ---------------------------------------------------------------------------

@requires_rebuilt_ext
def test_backed_correct_basal_expression(tmp_path):
    """Basal-expression correction runs on backed data without error."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=42))

    an.normalize_ace(adata, target_sum=1e4, backed_chunk_size=32, inplace=True)
    an.log1p_ace(adata, base=2, backed_chunk_size=32, inplace=True)
    an.reduce_kernel(adata, n_components=12, key_added="action", backed_chunk_size=32, inplace=True)

    basal_genes = list(adata.var_names[:4])
    an.correct_basal_expression(
        adata,
        basal_genes=basal_genes,
        reduction_key="action",
        corrected_key="action_basal_corrected",
        backed_chunk_size=32,
        inplace=True,
    )

    assert "action_basal_corrected" in adata.obsm
    assert "action_basal_corrected_U" in adata.varm
    assert "action_basal_corrected_params" in adata.uns
    sr = adata.obsm["action_basal_corrected"]
    assert sr.shape == (adata.n_obs, 12)
    assert not np.any(np.isnan(sr))
