"""Duplicate-feature regression tests for annotation and imputation."""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from actionet._feature_lookup import resolve_feature_space, resolve_requested_features
from actionet.annotation import _encode_markers


class TestDuplicateVarNames:
    """Regression: duplicate adata.var_names should use first occurrence."""

    @pytest.fixture
    def adata_dup(self):
        X = np.arange(12, dtype=np.float64).reshape(3, 4)
        adata = ad.AnnData(X=X)
        adata.var_names = pd.Index(["A", "B", "A", "C"])
        return adata

    def test_first_occurrence_lookup(self, adata_dup):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space = resolve_feature_space(adata_dup, None, context="test")
        assert space.has_duplicates
        assert space.lookup["A"] == 0  # first occurrence, not 2
        assert len(w) == 1

    def test_encode_markers_first_match(self, adata_dup):
        feature_set = adata_dup.var_names.to_numpy()
        markers = {"label1": ["A", "B"]}
        X_m, names = _encode_markers(markers, feature_set)
        assert sp.issparse(X_m)
        dense = X_m.toarray().ravel()
        assert dense[0] == 1.0  # first A (index 0)
        assert dense[1] == 1.0  # B (index 1)
        assert dense[2] == 0.0  # second A (index 2) should NOT be marked
        assert dense[3] == 0.0  # C

    def test_duplicate_requested_collapse(self, adata_dup):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            space = resolve_feature_space(adata_dup, None, context="test")
        res = resolve_requested_features(["A", "A", "B"], space, context="test")
        assert res.matched_names == ["A", "B"]
        np.testing.assert_array_equal(res.matched_indices, [0, 1])


class TestDuplicateFeaturesUseColumn:
    """Regression: duplicate values in adata.var[features_use]."""

    @pytest.fixture
    def adata_dup_col(self):
        X = np.arange(12, dtype=np.float64).reshape(3, 4)
        adata = ad.AnnData(X=X)
        adata.var_names = pd.Index(["v0", "v1", "v2", "v3"])
        adata.var["Gene"] = ["X", "Y", "X", "Z"]
        return adata

    def test_custom_col_first_match(self, adata_dup_col):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space = resolve_feature_space(adata_dup_col, "Gene", context="test")
        assert space.has_duplicates
        assert space.lookup["X"] == 0
        assert len(w) == 1

    def test_encode_markers_custom_col_first_match(self, adata_dup_col):
        feature_set = adata_dup_col.var["Gene"].to_numpy()
        markers = {"label1": ["X"]}
        X_m, names = _encode_markers(markers, feature_set)
        dense = X_m.toarray().ravel()
        assert dense[0] == 1.0  # first X
        assert dense[2] == 0.0  # second X not marked
