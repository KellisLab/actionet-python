import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import actionet as an


def _make_adata(n_cells: int = 96, n_genes: int = 72, sparse_fmt: str = "csr", seed: int = 13) -> ad.AnnData:
    rng = np.random.default_rng(seed)

    labels = np.array([f"CT_{i}" for i in (np.arange(n_cells) % 3)])
    batches = np.array(["B0" if i % 2 == 0 else "B1" for i in range(n_cells)])

    X = rng.poisson(0.2, size=(n_cells, n_genes)).astype(np.float64)
    for ct in range(3):
        rows = np.where(labels == f"CT_{ct}")[0]
        cols = np.arange(ct * 6, (ct + 1) * 6)
        X[np.ix_(rows, cols)] += rng.poisson(3.0, size=(rows.size, cols.size))

    if sparse_fmt == "csr":
        X_mat = sp.csr_matrix(X)
    elif sparse_fmt == "csc":
        X_mat = sp.csc_matrix(X)
    else:
        X_mat = X

    obs = pd.DataFrame({"CellLabel": labels, "batch": batches})
    var_names = np.array([f"G{i}" for i in range(n_genes)], dtype=object)
    var = pd.DataFrame({"Gene": var_names})

    adata = ad.AnnData(X=X_mat, obs=obs, var=var)
    adata.obs_names = np.array([f"cell_{i}" for i in range(n_cells)], dtype=object)
    adata.var_names = var_names
    adata.layers["logcounts"] = X_mat.copy()
    return adata


def _open_backed(tmp_path, adata: ad.AnnData) -> ad.AnnData:
    path = tmp_path / "test_backed.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r+")


def _run_backed_workflow(adata: ad.AnnData, layer: str | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
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


def test_backed_e2e_x(tmp_path):
    adata = _open_backed(tmp_path, _make_adata(sparse_fmt="csr"))
    markers, annot, imp = _run_backed_workflow(adata, layer=None)

    assert markers.shape[1] >= 2
    assert len(annot["labels"]) == adata.n_obs
    assert imp.shape[0] == adata.n_obs
    assert imp.shape[1] > 0



def test_backed_e2e_layer(tmp_path):
    adata = _open_backed(tmp_path, _make_adata(sparse_fmt="csr"))
    markers, annot, imp = _run_backed_workflow(adata, layer="logcounts")

    assert markers.shape[1] >= 2
    assert len(annot["labels"]) == adata.n_obs
    assert imp.shape[0] == adata.n_obs
    assert imp.shape[1] > 0



def test_backed_persistence_after_reopen(tmp_path):
    path = tmp_path / "persist_backed.h5ad"
    adata0 = _make_adata(sparse_fmt="csr")
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



def test_backed_preprocessing_csr_and_csc(tmp_path):
    for fmt in ("csr", "csc"):
        adata = _open_backed(tmp_path / fmt, _make_adata(sparse_fmt=fmt, seed=17 if fmt == "csr" else 23))

        an.normalize_anndata(adata, target_sum=1e4, layer=None, backed_chunk_size=24, inplace=True)
        row_sums = np.asarray(MatrixLike(adata.X).sum(axis=1)).ravel()
        nonzero_rows = row_sums > 0
        assert np.allclose(row_sums[nonzero_rows], 1e4, rtol=1e-2, atol=1e-2)



def test_backed_parity_markers_and_imputation(tmp_path):
    adata_mem = _make_adata(sparse_fmt="csr", seed=31)
    adata_backed = _open_backed(tmp_path, adata_mem)

    # Shared preprocessing
    for obj in (adata_mem, adata_backed):
        an.normalize_anndata(obj, target_sum=1e4, layer="logcounts", backed_chunk_size=24, inplace=True)

    an.reduce_kernel(adata_mem, n_components=14, layer="logcounts", key_added="action", seed=2, inplace=True)
    an.correct_batch_effect(adata_mem, batch_key="batch", reduction_key="action", layer="logcounts", backed_chunk_size=24, inplace=True)
    an.run_actionet(adata_mem, layer="logcounts", reduction_key="action_corrected", k_min=2, k_max=8, layout_3d=False, seed=2, n_threads=1, backed_chunk_size=24, inplace=True)

    an.reduce_kernel(adata_backed, n_components=14, layer="logcounts", key_added="action", seed=2, backed_chunk_size=24, inplace=True)
    an.correct_batch_effect(adata_backed, batch_key="batch", reduction_key="action", layer="logcounts", backed_chunk_size=24, inplace=True)
    an.run_actionet(adata_backed, layer="logcounts", reduction_key="action_corrected", k_min=2, k_max=8, layout_3d=False, seed=2, n_threads=1, backed_chunk_size=24, inplace=True)

    ranks_mem = an.find_markers(
        adata_mem,
        labels="CellLabel",
        features_use="Gene",
        layer="logcounts",
        result="ranks",
        return_type="dataframe",
        backed_chunk_size=24,
    )
    ranks_backed = an.find_markers(
        adata_backed,
        labels="CellLabel",
        features_use="Gene",
        layer="logcounts",
        result="ranks",
        return_type="dataframe",
        backed_chunk_size=24,
    )

    topn = 12
    overlaps = []
    for col in sorted(set(ranks_mem.columns) & set(ranks_backed.columns)):
        top_mem = set(ranks_mem[col].sort_values().index[:topn])
        top_backed = set(ranks_backed[col].sort_values().index[:topn])
        overlaps.append(len(top_mem & top_backed) / float(topn))

    assert len(overlaps) > 0
    assert np.mean(overlaps) >= 0.60

    feat_list = list(ranks_mem.index[:6])
    imp_mem = an.impute_features(
        adata_mem,
        features=feat_list,
        features_use="Gene",
        layer="logcounts",
        reduction_key="action_corrected",
        backed_chunk_size=24,
    )
    imp_backed = an.impute_features(
        adata_backed,
        features=feat_list,
        features_use="Gene",
        layer="logcounts",
        reduction_key="action_corrected",
        backed_chunk_size=24,
    )

    corrs = []
    for feat in feat_list:
        x = np.asarray(imp_mem[feat], dtype=float)
        y = np.asarray(imp_backed[feat], dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        corrs.append(np.corrcoef(x, y)[0, 1])

    assert len(corrs) > 0
    assert np.mean(corrs) >= 0.90


class MatrixLike:
    """Small helper to normalize backed matrix sum() return types."""

    def __init__(self, X):
        self.X = X

    def sum(self, axis=0):
        out = self.X.sum(axis=axis)
        return out.A if hasattr(out, "A") else out
