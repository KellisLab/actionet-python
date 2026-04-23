"""End-to-end parity tests: backed vs in-memory for annotate_cells and impute_features."""

import numpy as np
import pytest
import scipy.sparse as sp

from .conftest import make_test_adata, open_backed


def _setup_full_pipeline(adata, backed_chunk_size=None):
    """Run normalize -> reduce_kernel -> run_actionet so annotate/impute can run."""
    import actionet as an

    kw = {}
    if backed_chunk_size is not None:
        kw["backed_chunk_size"] = backed_chunk_size

    an.normalize_anndata(adata, target_sum=1e4, inplace=True, **kw)
    an.reduce_kernel(adata, n_components=12, seed=1, inplace=True, **kw)
    an.run_actionet(
        adata,
        reduction_key="ACTION",
        k_min=2,
        k_max=6,
        layout_3d=False,
        seed=1,
        n_threads=1,
        inplace=True,
        **kw,
    )
    return adata


def _make_markers(n_genes):
    """Build a small marker dict using the first 18 genes (3 cell types x 6)."""
    markers = {}
    for ct in range(3):
        start = ct * 6
        end = min(start + 6, n_genes)
        markers[f"CT_{ct}"] = [f"G{i}" for i in range(start, end)]
    return markers


try:
    import actionet as _an
    _has_ext = hasattr(_an, "_core")
except Exception:
    _has_ext = False

requires_ext = pytest.mark.skipif(not _has_ext, reason="C extension not built")


@requires_ext
@pytest.fixture(params=["csr"])
def paired_adata(request, tmp_path):
    """Yield (in_memory_adata, backed_adata) both with full pipeline run."""
    fmt = request.param
    adata = make_test_adata(n_cells=96, n_genes=72, sparse_fmt=fmt, seed=42)
    adata_mem = _setup_full_pipeline(adata)

    backed = open_backed(tmp_path, adata_mem)
    return adata_mem, backed


@requires_ext
class TestAnnotateCellsParity:
    def test_vision_parity(self, paired_adata):
        import actionet as an

        adata_mem, adata_backed = paired_adata
        markers = _make_markers(72)

        result_mem = an.annotate_cells(adata_mem, markers, method="vision")
        result_backed = an.annotate_cells(adata_backed, markers, method="vision")

        np.testing.assert_allclose(
            result_backed["enrichment"],
            result_mem["enrichment"],
            atol=1e-6,
            err_msg="Vision enrichment mismatch between backed and in-memory",
        )
        np.testing.assert_array_equal(
            result_backed["labels"],
            result_mem["labels"],
        )

    def test_actionet_parity(self, paired_adata):
        import actionet as an

        adata_mem, adata_backed = paired_adata
        markers = _make_markers(72)

        result_mem = an.annotate_cells(adata_mem, markers, method="actionet")
        result_backed = an.annotate_cells(adata_backed, markers, method="actionet")

        np.testing.assert_allclose(
            result_backed["enrichment"],
            result_mem["enrichment"],
            atol=1e-6,
            err_msg="ACTIONet enrichment mismatch between backed and in-memory",
        )
        np.testing.assert_array_equal(
            result_backed["labels"],
            result_mem["labels"],
        )


@requires_ext
class TestImputeFeaturesParity:
    def test_parity(self, paired_adata):
        import actionet as an

        adata_mem, adata_backed = paired_adata
        features = [f"G{i}" for i in range(6)]

        result_mem = an.impute_features(adata_mem, features)
        result_backed = an.impute_features(adata_backed, features)

        # Compare column-by-column correlation (imputation results may differ
        # slightly due to floating point path differences).
        for feat in features:
            x = np.asarray(result_mem[feat], dtype=float)
            y = np.asarray(result_backed[feat], dtype=float)
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            corr = np.corrcoef(x, y)[0, 1]
            assert corr >= 0.95, f"impute_features parity failed for {feat}: corr={corr:.4f}"




@requires_ext
def test_annotate_cells_backed_handles_retryable_open_conflict(tmp_path, monkeypatch):
    import actionet as an
    import actionet.annotation as an_annotation
    import actionet.backed_io as backed_io

    adata = make_test_adata(n_cells=32, n_genes=36, sparse_fmt="csr", seed=123)
    adata.obsp["actionet"] = sp.eye(adata.n_obs, format="csr")
    adata_backed = open_backed(tmp_path, adata)
    markers = _make_markers(36)

    sentinel = object()
    create_calls = {"count": 0}

    def flaky_create(**kwargs):
        create_calls["count"] += 1
        if create_calls["count"] == 1:
            raise RuntimeError("createBackedOperator: failed to open h5ad file: simulated lock")
        return sentinel

    monkeypatch.setattr(backed_io, "_create_backed_operator", flaky_create)

    monkeypatch.setattr(
        an_annotation,
        "_resolve_lazy_backed_transform",
        lambda source, lazy_transform=None, backed_chunk_size=4096: (None, False, 1.0),
    )

    def fake_stats(*, op, G, X, norm_method, alpha, max_it, approx, thread_no):
        assert op is sentinel
        n_cells = G.shape[0]
        n_types = X.shape[1]
        return np.tile(np.arange(1, n_types + 1, dtype=np.float64), (n_cells, 1))

    monkeypatch.setattr(
        an_annotation._core,
        "compute_feature_stats_vision_backed_operator",
        fake_stats,
    )

    try:
        result = an.annotate_cells(
            adata_backed,
            markers,
            method="vision",
            use_enrichment=False,
            n_threads=1,
            backed_chunk_size=8,
        )
    finally:
        adata_backed.file.close()

    assert create_calls["count"] >= 2
    assert result["enrichment"].shape[0] == 32
    assert result["enrichment"].shape[1] == len(markers)
