"""Regression tests for SPA (Successive Projection Algorithm).

Covers the early-break contract:  when SPA finds no candidate column above the
numerical tolerance, it must shrink its outputs (selected_cols, norms) to the
number of columns actually selected, rather than returning zero-padded arrays
that downstream consumers can silently index.
"""

import numpy as np
import pytest

from actionet.action.archetypes import run_spa


class TestRunSpaEarlyBreak:
    """SPA shrinks outputs on early break instead of returning zero-padded slots."""

    def test_full_rank_returns_k_selections(self):
        """Sanity: with well-conditioned inputs, run_spa returns exactly k selections."""
        rng = np.random.default_rng(0)
        # 4-dim ambient space, 8 columns spanning it robustly.
        A = rng.standard_normal((4, 8))
        k = 3

        res = run_spa(A, k)

        assert res["selected_cols"].shape == (k,)
        assert res["norms"].shape == (k,)
        # All indices must be valid column indices (0-based).
        assert np.all(res["selected_cols"] >= 0)
        assert np.all(res["selected_cols"] < A.shape[1])
        # And unique (no duplicate re-selection).
        assert len(np.unique(res["selected_cols"])) == k

    def test_zero_matrix_shrinks_outputs(self):
        """A zero matrix forces SPA to break on the first iteration; outputs shrink to size 0."""
        A = np.zeros((4, 6), dtype=np.float64)
        k = 3

        res = run_spa(A, k)

        # No column can be selected — outputs must be empty, not length-k of zeros.
        assert res["selected_cols"].shape == (0,)
        assert res["norms"].shape == (0,)

    def test_rank_deficient_shrinks_below_k(self):
        """When rank < k, SPA breaks early; outputs shrink to the actual selection count."""
        # Construct a matrix with only two linearly independent directions.
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        # 5 columns, all in span{e1, e2}; k > 2 forces early break after 2 selections.
        A = np.column_stack([e1, e2, 2.0 * e1, -e2, 0.5 * e1 + 0.5 * e2])
        k = 4

        res = run_spa(A, k)

        # SPA orthogonalises, so after 2 selections all remaining norms are ~0.
        assert res["selected_cols"].shape[0] < k
        assert res["selected_cols"].shape[0] >= 1
        # selected_cols and norms shrink in lock-step.
        assert res["selected_cols"].shape[0] == res["norms"].shape[0]
        # Every remaining index must be a valid column.
        assert np.all(res["selected_cols"] < A.shape[1])
