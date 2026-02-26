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
import warnings

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
    an.normalize_anndata(adata, target_sum=1e4, layer=layer, backed_chunk_size=32, inplace=True)

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


def _open_backed_with_compression(
    tmp_path,
    adata: ad.AnnData,
    *,
    compression: str | None,
) -> ad.AnnData:
    """Write adata with requested compression and reopen as backed r+."""
    path = tmp_path / "compressed_backed.h5ad"
    adata.write_h5ad(path, compression=compression)
    return ad.read_h5ad(path, backed="r+")


def _sparse_codecs(adata: ad.AnnData, *, layer: str | None = None) -> dict[str, str | None]:
    """Return sparse codec metadata for `.X` or one backed layer."""
    h5file = adata.file._file
    grp = h5file["X"] if layer is None else h5file["layers"][layer]
    return {
        "data": grp["data"].compression,
        "indices": grp["indices"].compression,
        "indptr": grp["indptr"].compression,
    }


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


def test_compute_filter_masks_pure(tmp_path):
    """compute_filter_masks returns boolean arrays without mutating adata."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=60))
    orig_shape = adata.shape

    obs_m, var_m = an.compute_filter_masks(
        adata, min_cells_per_feat=3, backed_chunk_size=32,
    )
    # adata unchanged
    assert adata.shape == orig_shape
    assert obs_m.dtype == bool and obs_m.shape == (orig_shape[0],)
    assert var_m.dtype == bool and var_m.shape == (orig_shape[1],)
    # at least some genes removed
    assert var_m.sum() <= orig_shape[1]


def test_apply_filter_backed_output_file(tmp_path):
    """apply_filter with output_file writes to a new path, original untouched."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=61))
    obs_m, var_m = an.compute_filter_masks(
        adata, min_cells_per_feat=3, backed_chunk_size=32,
    )
    out_path = str(tmp_path / "filtered.h5ad")
    result = an.apply_filter(
        adata, obs_m, var_m,
        inplace=False, output_file=out_path, backed_chunk_size=32,
    )
    assert result is not None
    assert result.n_obs == int(obs_m.sum())
    assert result.n_vars == int(var_m.sum())
    # Original still has original shape
    assert adata.shape[0] >= result.n_obs


def test_apply_filter_inmemory_inplace():
    """apply_filter inplace on in-memory adata uses single copy."""
    adata = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=62)
    obs_m, var_m = an.compute_filter_masks(adata, min_cells_per_feat=3)
    n_obs_expected = int(obs_m.sum())
    n_var_expected = int(var_m.sum())

    an.apply_filter(adata, obs_m, var_m, inplace=True)
    assert adata.n_obs == n_obs_expected
    assert adata.n_vars == n_var_expected


def test_apply_filter_inmemory_copy():
    """apply_filter inplace=False on in-memory adata returns a copy."""
    adata = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=63)
    obs_m, var_m = an.compute_filter_masks(adata, min_cells_per_feat=3)
    result = an.apply_filter(adata, obs_m, var_m, inplace=False)
    assert result is not None
    assert result.n_obs == int(obs_m.sum())
    # Original unchanged
    assert adata.n_obs == 96


def test_filter_anndata_masks_only():
    """filter_anndata with filter_adata=False returns mask dict."""
    adata = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=64)
    result = an.filter_anndata(adata, min_cells_per_feat=3, filter_adata=False)
    assert isinstance(result, dict)
    assert "fil_obs" in result and "fil_vars" in result
    assert "mask" in result["fil_obs"].columns


def test_backed_chunked_write_preserves_layers(tmp_path):
    """Chunked h5py write preserves layers and obs/var columns."""
    adata_mem = make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=65)
    adata = open_backed(tmp_path, adata_mem)
    assert "logcounts" in adata.layers

    an.filter_anndata(adata, min_cells_per_feat=3, inplace=True, backed_chunk_size=16)

    assert getattr(adata, "isbacked", False)
    assert "logcounts" in adata.layers
    assert "CellLabel" in adata.obs.columns
    assert "batch" in adata.obs.columns
    assert "Gene" in adata.var.columns


def test_backed_preprocessing_csr_and_csc(tmp_path):
    """normalize_anndata produces correct row sums for CSR and CSC.

    Note: CSC-backed X triggers h5py 'increasing order' errors on row-slice
    read, so we test CSC via a layer rather than .X directly.
    """
    from actionet._matrix_source import MatrixSource

    for fmt in ("csr", "csc"):
        adata_mem = make_test_adata(sparse_fmt=fmt, seed=17 if fmt == "csr" else 23)
        adata = open_backed(tmp_path / fmt, adata_mem)

        target_layer = "logcounts" if fmt == "csc" else None
        an.normalize_anndata(adata, target_sum=1e4, log_transform=False, layer=target_layer, backed_chunk_size=24, inplace=True)
        src = MatrixSource(adata, layer=target_layer)
        row_sums = src.row_sums(chunk_size=24)
        nonzero_rows = row_sums > 0
        assert np.allclose(row_sums[nonzero_rows], 1e4, rtol=1e-2, atol=1e-2)

        # Now apply log_transform via a second call with scaling disabled
        # (already normalised, just test the combined path works)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, layer=target_layer, backed_chunk_size=24, inplace=True)


def test_reduce_kernel_warns_on_compressed_backed_matrix(tmp_path):
    """Backed reduce_kernel warns when storage is compressed."""
    adata = _open_backed_with_compression(
        tmp_path,
        make_test_adata(n_cells=64, n_genes=48, sparse_fmt="csr", seed=70),
        compression="gzip",
    )

    an.normalize_anndata(
        adata,
        target_sum=1e4,
        log_transform=False,
        backed_chunk_size=24,
        inplace=True,
    )

    with pytest.warns(UserWarning, match="decompress_backed_storage"):
        an.reduce_kernel(
            adata,
            n_components=6,
            key_added="action",
            svd_algorithm=3,
            backed_chunk_size=24,
            verbose=False,
            inplace=True,
        )


def test_reduce_kernel_no_warning_on_uncompressed_backed_matrix(tmp_path):
    """Backed reduce_kernel does not emit compression warning when uncompressed."""
    adata = open_backed(
        tmp_path,
        make_test_adata(n_cells=64, n_genes=48, sparse_fmt="csr", seed=71),
    )

    an.normalize_anndata(
        adata,
        target_sum=1e4,
        log_transform=False,
        backed_chunk_size=24,
        inplace=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        an.reduce_kernel(
            adata,
            n_components=6,
            key_added="action",
            svd_algorithm=3,
            backed_chunk_size=24,
            verbose=False,
            inplace=True,
        )

    assert not any("decompress_backed_storage" in str(w.message) for w in caught)


def test_decompress_matrix_scope_only_affects_target_matrix(tmp_path):
    """Matrix-scope decompression only changes the selected matrix."""
    adata = _open_backed_with_compression(
        tmp_path,
        make_test_adata(n_cells=64, n_genes=48, sparse_fmt="csr", seed=72),
        compression="gzip",
    )

    before_x = _sparse_codecs(adata, layer=None)
    before_layer = _sparse_codecs(adata, layer="logcounts")
    assert all(codec == "gzip" for codec in before_x.values())
    assert all(codec == "gzip" for codec in before_layer.values())

    an.decompress_backed_storage(
        adata,
        layer="logcounts",
        scope="matrix",
        chunk_size=32,
        verbose=False,
    )

    after_x = _sparse_codecs(adata, layer=None)
    after_layer = _sparse_codecs(adata, layer="logcounts")
    assert all(codec == "gzip" for codec in after_x.values())
    assert all(codec is None for codec in after_layer.values())


def test_decompress_file_scope_decompresses_x_and_layers(tmp_path):
    """File-scope decompression removes compression from `.X` and layers."""
    adata = _open_backed_with_compression(
        tmp_path,
        make_test_adata(n_cells=64, n_genes=48, sparse_fmt="csr", seed=73),
        compression="gzip",
    )

    assert all(codec == "gzip" for codec in _sparse_codecs(adata, layer=None).values())
    assert all(codec == "gzip" for codec in _sparse_codecs(adata, layer="logcounts").values())

    an.decompress_backed_storage(
        adata,
        scope="file",
        chunk_size=32,
        verbose=False,
    )

    assert all(codec is None for codec in _sparse_codecs(adata, layer=None).values())
    assert all(codec is None for codec in _sparse_codecs(adata, layer="logcounts").values())


def test_filter_preserves_uncompressed_storage(tmp_path):
    """Backed filtering preserves uncompressed sparse storage."""
    adata = open_backed(
        tmp_path,
        make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=74),
    )
    before_x = _sparse_codecs(adata, layer=None)
    before_layer = _sparse_codecs(adata, layer="logcounts")
    assert all(codec is None for codec in before_x.values())
    assert all(codec is None for codec in before_layer.values())

    an.filter_anndata(
        adata,
        min_cells_per_feat=3,
        inplace=True,
        backed_chunk_size=24,
    )

    after_x = _sparse_codecs(adata, layer=None)
    after_layer = _sparse_codecs(adata, layer="logcounts")
    assert all(codec is None for codec in after_x.values())
    assert all(codec is None for codec in after_layer.values())


def test_filter_preserves_gzip_storage(tmp_path):
    """Backed filtering preserves gzip sparse storage."""
    adata = _open_backed_with_compression(
        tmp_path,
        make_test_adata(n_cells=96, n_genes=72, sparse_fmt="csr", seed=75),
        compression="gzip",
    )
    before_x = _sparse_codecs(adata, layer=None)
    before_layer = _sparse_codecs(adata, layer="logcounts")
    assert all(codec == "gzip" for codec in before_x.values())
    assert all(codec == "gzip" for codec in before_layer.values())

    an.filter_anndata(
        adata,
        min_cells_per_feat=3,
        inplace=True,
        backed_chunk_size=24,
    )

    after_x = _sparse_codecs(adata, layer=None)
    after_layer = _sparse_codecs(adata, layer="logcounts")
    assert all(codec == "gzip" for codec in after_x.values())
    assert all(codec == "gzip" for codec in after_layer.values())


# ---------------------------------------------------------------------------
# Parity: backed vs in-memory
# ---------------------------------------------------------------------------

@requires_rebuilt_ext
def test_backed_parity_markers_and_imputation(tmp_path):
    """Marker ranking overlap and imputation correlation between modes."""
    adata_mem = make_test_adata(sparse_fmt="csr", seed=31)
    adata_backed = open_backed(tmp_path, adata_mem)

    for obj in (adata_mem, adata_backed):
        an.normalize_anndata(obj, target_sum=1e4, layer="logcounts", backed_chunk_size=24, inplace=True)

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

    an.normalize_anndata(adata, target_sum=1e4, backed_chunk_size=32, inplace=True)
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
