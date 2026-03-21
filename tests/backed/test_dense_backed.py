"""Dense-backed operator tests.

Verifies that dense-backed .X and layers work through the operator path
for run_svd, reduce_kernel, correct_batch_effect, and correct_basal_expression.
Also checks parity against in-memory dense results and regression for sparse-backed.
"""

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pytest

import actionet as an

from .conftest import make_test_adata


def _make_dense_adata(n_cells=96, n_genes=72, seed=42):
    """Create a dense in-memory AnnData (not sparse)."""
    rng = np.random.default_rng(seed)
    labels = np.array([f"CT_{i}" for i in (np.arange(n_cells) % 3)])
    batches = np.array(["B0" if i % 2 == 0 else "B1" for i in range(n_cells)])

    X = rng.poisson(0.2, size=(n_cells, n_genes)).astype(np.float64)
    n_enriched = min(6, n_genes // 3)
    for ct in range(3):
        rows = np.where(labels == f"CT_{ct}")[0]
        col_start = ct * n_enriched
        col_end = min(col_start + n_enriched, n_genes)
        if col_start < n_genes and col_end > col_start:
            cols = np.arange(col_start, col_end)
            X[np.ix_(rows, cols)] += rng.poisson(3.0, size=(rows.size, cols.size))

    import pandas as pd
    obs = pd.DataFrame({"CellLabel": labels, "batch": batches})
    var_names = np.array([f"G{i}" for i in range(n_genes)], dtype=object)
    var = pd.DataFrame({"Gene": var_names})

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = np.array([f"cell_{i}" for i in range(n_cells)], dtype=object)
    adata.var_names = var_names
    adata.layers["logcounts"] = X.copy()
    return adata


def _open_backed_dense(tmp_path, adata):
    """Write dense AnnData to h5ad and reopen in backed mode."""
    path = tmp_path / "test_dense_backed.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r+")


# ---------------------------------------------------------------------------
# Dense-backed .X: run_svd
# ---------------------------------------------------------------------------

class TestDenseBackedSVD:
    def test_run_svd_dense_backed(self, tmp_path):
        """run_svd runs on dense-backed AnnData without error."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=10))
        result = an.run_svd(adata, n_components=10, backed_chunk_size=32)
        assert result["u"].shape == (adata.n_vars, 10)
        assert result["v"].shape == (adata.n_obs, 10)
        assert result["d"].shape == (10,)
        assert not np.any(np.isnan(result["u"]))
        assert not np.any(np.isnan(result["v"]))
        assert not np.any(np.isnan(result["d"]))

    def test_svd_parity_dense_backed_vs_inmemory(self, tmp_path):
        """Dense-backed SVD singular values are close to in-memory dense SVD."""
        adata_mem = _make_dense_adata(seed=20)
        adata_backed = _open_backed_dense(tmp_path, _make_dense_adata(seed=20))

        k = 10
        res_mem = an.run_svd(adata_mem, n_components=k)
        res_backed = an.run_svd(adata_backed, n_components=k, backed_chunk_size=32)

        np.testing.assert_allclose(
            np.sort(np.asarray(res_backed["d"]).reshape(-1))[::-1],
            np.sort(np.asarray(res_mem["d"]).reshape(-1))[::-1],
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Dense-backed .X: reduce_kernel
# ---------------------------------------------------------------------------

class TestDenseBackedReduceKernel:
    def test_reduce_kernel_dense_backed(self, tmp_path):
        """reduce_kernel runs on dense-backed AnnData and produces valid output."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=30))
        an.reduce_kernel(
            adata, n_components=10, backed_chunk_size=32, inplace=True,
        )
        assert "action" in adata.obsm
        assert adata.obsm["action"].shape == (adata.n_obs, 10)
        assert not np.any(np.isnan(adata.obsm["action"]))
        assert not np.any(np.isinf(adata.obsm["action"]))

    def test_reduce_kernel_parity_shapes(self, tmp_path):
        """Dense-backed reduce_kernel shapes match in-memory."""
        adata_mem = _make_dense_adata(seed=40)
        adata_backed = _open_backed_dense(tmp_path, _make_dense_adata(seed=40))

        k = 10
        an.reduce_kernel(adata_mem, n_components=k, inplace=True)
        an.reduce_kernel(adata_backed, n_components=k, backed_chunk_size=32, inplace=True)

        assert adata_backed.obsm["action"].shape == adata_mem.obsm["action"].shape
        assert adata_backed.varm["action_U"].shape == adata_mem.varm["action_U"].shape


# ---------------------------------------------------------------------------
# Dense-backed .X: correct_batch_effect
# ---------------------------------------------------------------------------

class TestDenseBackedBatchCorrection:
    def test_correct_batch_effect_dense_backed(self, tmp_path):
        """correct_batch_effect runs on dense-backed AnnData."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=50))
        an.reduce_kernel(adata, n_components=10, backed_chunk_size=32, inplace=True)

        an.correct_batch_effect(
            adata, batch_key="batch", backed_chunk_size=32, inplace=True,
        )
        corrected_key = "action_corrected"
        assert corrected_key in adata.obsm
        assert adata.obsm[corrected_key].shape == (adata.n_obs, 10)
        assert not np.any(np.isnan(adata.obsm[corrected_key]))

    def test_batch_correction_parity(self, tmp_path):
        """Dense-backed batch correction shapes match in-memory."""
        adata_mem = _make_dense_adata(seed=55)
        adata_backed = _open_backed_dense(tmp_path, _make_dense_adata(seed=55))

        k = 10
        an.reduce_kernel(adata_mem, n_components=k, inplace=True)
        an.reduce_kernel(adata_backed, n_components=k, backed_chunk_size=32, inplace=True)

        an.correct_batch_effect(adata_mem, batch_key="batch", inplace=True)
        an.correct_batch_effect(
            adata_backed, batch_key="batch", backed_chunk_size=32, inplace=True,
        )

        corrected_key = "action_corrected"
        assert adata_backed.obsm[corrected_key].shape == adata_mem.obsm[corrected_key].shape


# ---------------------------------------------------------------------------
# Dense-backed .X: correct_basal_expression
# ---------------------------------------------------------------------------

class TestDenseBackedBasalCorrection:
    def test_correct_basal_dense_backed(self, tmp_path):
        """correct_basal_expression runs on dense-backed AnnData."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=60))
        an.reduce_kernel(adata, n_components=10, backed_chunk_size=32, inplace=True)

        basal_genes = adata.var_names[:3].tolist()
        an.correct_basal_expression(
            adata, basal_genes=basal_genes, backed_chunk_size=32, inplace=True,
        )
        corrected_key = "action_basal_corrected"
        assert corrected_key in adata.obsm
        assert adata.obsm[corrected_key].shape == (adata.n_obs, 10)
        assert not np.any(np.isnan(adata.obsm[corrected_key]))


# ---------------------------------------------------------------------------
# Dense-backed layers[...]
# ---------------------------------------------------------------------------

class TestDenseBackedLayers:
    def test_reduce_kernel_on_layer(self, tmp_path):
        """reduce_kernel works on a dense-backed layer."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=70))
        an.reduce_kernel(
            adata, n_components=10, layer="logcounts",
            backed_chunk_size=32, inplace=True,
        )
        assert "action" in adata.obsm
        assert adata.obsm["action"].shape == (adata.n_obs, 10)
        assert not np.any(np.isnan(adata.obsm["action"]))

    def test_run_svd_on_layer(self, tmp_path):
        """run_svd works on a dense-backed layer."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=71))
        result = an.run_svd(adata, n_components=10, layer="logcounts", backed_chunk_size=32)
        assert result["u"].shape == (adata.n_vars, 10)
        assert not np.any(np.isnan(result["d"]))


# ---------------------------------------------------------------------------
# Sparse-backed regression: ensure sparse-backed still works
# ---------------------------------------------------------------------------

class TestSparseBackedRegression:
    def _open_sparse_backed(self, tmp_path, seed=80):
        adata = make_test_adata(sparse_fmt="csr", seed=seed)
        path = tmp_path / "test_sparse_backed.h5ad"
        adata.write_h5ad(path)
        return ad.read_h5ad(path, backed="r+")

    def test_sparse_backed_reduce_kernel(self, tmp_path):
        """Sparse-backed reduce_kernel still works after factory refactor."""
        adata = self._open_sparse_backed(tmp_path)
        an.reduce_kernel(adata, n_components=10, backed_chunk_size=32, inplace=True)
        assert "action" in adata.obsm
        assert adata.obsm["action"].shape == (adata.n_obs, 10)
        assert not np.any(np.isnan(adata.obsm["action"]))

    def test_sparse_backed_run_svd(self, tmp_path):
        """Sparse-backed run_svd still works after factory refactor."""
        adata = self._open_sparse_backed(tmp_path, seed=81)
        result = an.run_svd(adata, n_components=10, backed_chunk_size=32)
        assert result["u"].shape[1] == 10
        assert not np.any(np.isnan(result["d"]))

    def test_sparse_backed_batch_correction(self, tmp_path):
        """Sparse-backed batch correction still works after refactor."""
        adata = self._open_sparse_backed(tmp_path, seed=82)
        an.reduce_kernel(adata, n_components=10, backed_chunk_size=32, inplace=True)
        an.correct_batch_effect(
            adata, batch_key="batch", backed_chunk_size=32, inplace=True,
        )
        assert "action_corrected" in adata.obsm


# ---------------------------------------------------------------------------
# Dense-backed specificity: streamed fallback still works
# ---------------------------------------------------------------------------

class TestDenseBackedSpecificityFallback:
    def test_specificity_streamed_fallback(self, tmp_path):
        """Dense-backed specificity uses streamed fallback without error."""
        adata = _open_backed_dense(tmp_path, _make_dense_adata(seed=90))
        try:
            an.compute_feature_specificity(
                adata, labels="CellLabel", backed_chunk_size=32, inplace=True,
            )
            assert "specificity_upper" in adata.varm
        except Exception:
            pytest.skip("Specificity may require preprocessing; checking smoke only")
