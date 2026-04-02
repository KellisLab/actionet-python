"""Specificity regression tests for backed dispatch and label handling."""

import numpy as np
import pytest

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
