"""Specificity regression tests for backed dispatch and label handling."""

import numpy as np
import pytest
import scipy.sparse as sp

import actionet as an

from .conftest import make_test_adata, open_backed


def test_backed_compute_feature_specificity_sparse_runs(tmp_path):
    """Backed sparse compute_feature_specificity should run and persist outputs."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=101))
    an.compute_feature_specificity(
        adata,
        labels="CellLabel",
        backed_chunk_size=32,
        inplace=True,
    )

    assert "specificity_profile" in adata.varm
    assert "specificity_upper" in adata.varm
    assert "specificity_lower" in adata.varm
    adata.file.close()


def test_backed_compute_feature_specificity_dense_runs(tmp_path):
    """Backed dense compute_feature_specificity should run and persist outputs."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="dense", seed=111))
    an.compute_feature_specificity(
        adata,
        labels="CellLabel",
        backed_chunk_size=32,
        inplace=True,
    )

    assert "specificity_profile" in adata.varm
    assert "specificity_upper" in adata.varm
    assert "specificity_lower" in adata.varm
    adata.file.close()


def test_compute_feature_specificity_rejects_mismatched_label_length():
    """A labels array with wrong length should fail fast in Python."""
    adata = make_test_adata(n_cells=12, n_genes=8, sparse_fmt="csr", seed=102)
    labels = np.array([0, 1, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="labels length"):
        an.compute_feature_specificity(adata, labels=labels, return_raw=True)


def test_compute_feature_specificity_integer_labels_are_compacted():
    """Sparse integer ids should be compacted to avoid huge membership matrices."""
    adata = make_test_adata(n_cells=8, n_genes=6, sparse_fmt="csr", seed=103)
    labels = np.array([0, 10, -1, 10, 0, -5, 10, 0], dtype=np.int64)

    raw = an.compute_feature_specificity(adata, labels=labels, return_raw=True)
    # Valid non-negative ids are {0, 10} -> 2 output columns.
    assert raw["average_profile"].shape[1] == 2
    assert raw["upper_significance"].shape[1] == 2
    assert raw["lower_significance"].shape[1] == 2


def test_compute_archetype_feature_specificity_validates_membership_shape():
    """Archetype matrix must match n_obs rows."""
    adata = make_test_adata(n_cells=10, n_genes=8, sparse_fmt="csr", seed=104)
    H_bad = np.ones((adata.n_obs - 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="row count"):
        an.compute_archetype_feature_specificity(adata, archetype_key=H_bad, inplace=True)


def test_backed_sparse_parallel_matches_single_thread(tmp_path):
    """Backed sparse specificity should be numerically stable across thread counts."""
    adata = open_backed(tmp_path, make_test_adata(n_cells=160, n_genes=96, sparse_fmt="csr", seed=105))
    raw_1 = an.compute_feature_specificity(
        adata,
        labels="CellLabel",
        n_threads=1,
        backed_chunk_size=32,
        return_raw=True,
    )
    raw_4 = an.compute_feature_specificity(
        adata,
        labels="CellLabel",
        n_threads=4,
        backed_chunk_size=32,
        return_raw=True,
    )

    for key in ("average_profile", "upper_significance", "lower_significance"):
        np.testing.assert_allclose(raw_4[key], raw_1[key], rtol=1e-10, atol=1e-12)

    adata.file.close()


def test_backed_sparse_parallel_matches_single_thread_with_negative_values(tmp_path):
    """Parallel backed sparse path should match single-thread when min-shift support is used."""
    adata_mem = make_test_adata(n_cells=160, n_genes=96, sparse_fmt="csr", seed=106)
    X = adata_mem.X
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    X = X.copy()
    X.data = X.data - 2.0  # force a negative minimum to trigger support-pass correction
    adata_mem.X = X

    adata = open_backed(tmp_path, adata_mem)
    raw_1 = an.compute_feature_specificity(
        adata,
        labels="CellLabel",
        n_threads=1,
        backed_chunk_size=32,
        return_raw=True,
    )
    raw_4 = an.compute_feature_specificity(
        adata,
        labels="CellLabel",
        n_threads=4,
        backed_chunk_size=32,
        return_raw=True,
    )

    for key in ("average_profile", "upper_significance", "lower_significance"):
        np.testing.assert_allclose(raw_4[key], raw_1[key], rtol=1e-10, atol=1e-12)

    adata.file.close()
