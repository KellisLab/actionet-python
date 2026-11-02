"""Parity of actionet.tools.scale with R's base::scale.

Expected values follow R: with scale=TRUE each column is divided by
sqrt(sum(x^2) / max(1, n - 1)) of the (possibly) centered column, which is
the sample standard deviation only when center=TRUE.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from actionet.tools.scale import scale

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0], [6.0, 30.0]])


def test_default_matches_sample_standardisation():
    expected = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    np.testing.assert_allclose(scale(X), expected)


def test_center_false_divides_by_root_mean_square():
    # R: scale(X, center = FALSE)
    expected = np.array(
        [
            [0.2449490, 0.2449490],
            [0.4898979, 0.4898979],
            [0.7348469, 1.4696938],
            [1.4696938, 0.7348469],
        ]
    )
    np.testing.assert_allclose(scale(X, center=False), expected, atol=1e-7)


def test_explicit_center_uses_root_mean_square_of_centered_columns():
    # R: scale(X, center = c(1, 10))
    expected = np.array(
        [
            [0.0, 0.0],
            [0.3162278, 0.3162278],
            [0.6324555, 1.5811388],
            [1.5811388, 0.6324555],
        ]
    )
    np.testing.assert_allclose(
        scale(X, center=np.array([1.0, 10.0])), expected, atol=1e-7
    )


def test_center_false_vector_and_sparse_inputs():
    v = X[:, 0]
    np.testing.assert_allclose(
        scale(v, center=False), [0.2449490, 0.4898979, 0.7348469, 1.4696938], atol=1e-7
    )
    np.testing.assert_allclose(
        scale(sp.csr_matrix(X), center=False), scale(X, center=False)
    )


def test_unchanged_paths():
    np.testing.assert_allclose(scale(X, center=True, scale=False), X - X.mean(axis=0))
    np.testing.assert_allclose(scale(X, center=False, scale=False), X)
    np.testing.assert_allclose(
        scale(X, center=False, scale=np.array([2.0, 5.0])), X / [2.0, 5.0]
    )


def test_single_row_default_preserves_nan_and_warning():
    with pytest.warns(RuntimeWarning):
        out = scale(np.array([[3.0, 4.0]]))
    assert np.isnan(out).all()


def test_single_row_without_centering_uses_rms():
    np.testing.assert_allclose(
        scale(np.array([[3.0, -4.0, 0.0]]), center=False),
        [[1.0, -1.0, np.nan]],
        equal_nan=True,
    )


def test_single_row_explicit_center_zero_scale_is_nan():
    out = scale(np.array([[3.0, 4.0]]), center=np.array([3.0, 4.0]))
    assert np.isnan(out).all()
