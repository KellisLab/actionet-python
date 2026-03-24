"""Tests for feature lookup helpers and duplicate-feature semantics."""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from actionet._feature_lookup import (
    resolve_feature_space,
    resolve_requested_features,
)


class TestResolveFeatureSpace:
    def test_unique_labels(self):
        adata = ad.AnnData(
            X=np.zeros((3, 4)),
            var=pd.DataFrame(index=["A", "B", "C", "D"]),
        )
        space = resolve_feature_space(adata, None, context="test")
        assert not space.has_duplicates
        assert space.lookup == {"A": 0, "B": 1, "C": 2, "D": 3}

    def test_duplicate_labels_first_match(self):
        adata = ad.AnnData(
            X=np.zeros((2, 4)),
            var=pd.DataFrame(index=["X", "Y", "X", "Z"]),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            space = resolve_feature_space(adata, None, context="test")
        assert space.has_duplicates
        assert space.lookup["X"] == 0  # first occurrence
        assert "X" in space.duplicated_labels

        # Warning only fires when requesting a duplicated feature
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_requested_features(["Y"], space, context="test")
        assert len(w) == 0

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            resolve_requested_features(["X"], space, context="test")
        assert len(w2) == 1
        assert "X" in str(w2[0].message)

    def test_custom_column(self):
        adata = ad.AnnData(
            X=np.zeros((2, 3)),
            var=pd.DataFrame({"Gene": ["foo", "bar", "baz"]}, index=["a", "b", "c"]),
        )
        space = resolve_feature_space(adata, "Gene", context="test")
        assert space.lookup == {"foo": 0, "bar": 1, "baz": 2}

    def test_missing_column_raises(self):
        adata = ad.AnnData(
            X=np.zeros((2, 3)),
            var=pd.DataFrame(index=["a", "b", "c"]),
        )
        with pytest.raises(ValueError, match="not found"):
            resolve_feature_space(adata, "no_such_col", context="test")


class TestResolveRequestedFeatures:
    def test_basic_matching(self):
        adata = ad.AnnData(
            X=np.zeros((2, 5)),
            var=pd.DataFrame(index=["A", "B", "C", "D", "E"]),
        )
        space = resolve_feature_space(adata, None, context="test")
        res = resolve_requested_features(["C", "A", "Z"], space, context="test")
        assert res.matched_names == ["C", "A"]
        np.testing.assert_array_equal(res.matched_indices, [2, 0])
        assert res.missing_names == ["Z"]

    def test_duplicate_request_collapsed(self):
        adata = ad.AnnData(
            X=np.zeros((2, 3)),
            var=pd.DataFrame(index=["A", "B", "C"]),
        )
        space = resolve_feature_space(adata, None, context="test")
        res = resolve_requested_features(["A", "B", "A", "B"], space, context="test")
        assert res.matched_names == ["A", "B"]
        np.testing.assert_array_equal(res.matched_indices, [0, 1])

    def test_all_missing(self):
        adata = ad.AnnData(
            X=np.zeros((2, 2)),
            var=pd.DataFrame(index=["A", "B"]),
        )
        space = resolve_feature_space(adata, None, context="test")
        res = resolve_requested_features(["X", "Y"], space, context="test")
        assert len(res.matched_names) == 0
        assert res.missing_names == ["X", "Y"]
