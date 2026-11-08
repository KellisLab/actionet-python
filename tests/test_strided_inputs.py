"""Strided NumPy inputs must give the same results as contiguous ones.

``py::array_t<T>`` without ``c_style`` keeps the caller's strides, so a view
such as ``x[::2]``, ``x[::-1]`` or ``M[:, 0]`` reaches the bindings with
non-unit strides.  Several helpers read ``request().ptr`` as packed memory,
which silently picked up the wrong elements (or read outside the array for
negative strides).  Each test passes a view whose gaps hold junk values and
checks the result against the same values passed contiguously.
"""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import actionet as an
from actionet import _core

JUNK_INT = 9999
JUNK_FLOAT = -777.0


def strided(values, junk):
    """Return a non-contiguous view of ``values`` with ``junk`` between elements."""
    values = np.asarray(values)
    full = np.full(2 * values.size, junk, dtype=values.dtype)
    full[::2] = values
    view = full[::2]
    assert not view.flags.c_contiguous
    np.testing.assert_array_equal(view, values)
    return view


@pytest.fixture
def counts():
    return sparse.random(30, 40, density=0.4, format="csc", random_state=1) * 10


@pytest.fixture
def graph():
    rng = np.random.default_rng(3)
    n = 30
    dense = rng.random((n, n)) * (rng.random((n, n)) < 0.3)
    dense = np.triu(dense, 1)
    return sparse.csc_matrix(dense + dense.T)


def assert_dicts_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(
            np.asarray(actual[key]), np.asarray(expected[key]), err_msg=key
        )


def test_specificity_labels_view(counts):
    labels = np.random.default_rng(0).integers(1, 4, counts.shape[0]).astype(np.int32)
    expected = _core.compute_feature_specificity_sparse(counts, labels, 1)
    actual = _core.compute_feature_specificity_sparse(counts, strided(labels, JUNK_INT), 1)
    assert_dicts_equal(expected, actual)


def test_specificity_reversed_labels(counts):
    labels = np.random.default_rng(1).integers(1, 4, counts.shape[0]).astype(np.int32)
    reversed_view = labels[::-1]
    packed = np.ascontiguousarray(reversed_view)
    expected = _core.compute_feature_specificity_sparse(counts, packed, 1)
    actual = _core.compute_feature_specificity_sparse(counts, reversed_view, 1)
    assert_dicts_equal(expected, actual)


def test_sparse_matrix_built_from_views(counts):
    """SciPy keeps constructor arrays without copying, so .data can be a view."""
    labels = np.random.default_rng(2).integers(1, 4, counts.shape[0]).astype(np.int32)
    viewed = sparse.csc_matrix(
        (strided(counts.data, JUNK_FLOAT), strided(counts.indices.astype(np.int64), JUNK_INT),
         counts.indptr.astype(np.int64)),
        shape=counts.shape,
    )
    assert not viewed.data.flags.c_contiguous
    expected = _core.compute_feature_specificity_sparse(counts, labels, 1)
    actual = _core.compute_feature_specificity_sparse(viewed, labels, 1)
    assert_dicts_equal(expected, actual)


def test_sparse_indices_and_indptr_views(counts):
    labels = np.random.default_rng(7).integers(1, 4, counts.shape[0]).astype(np.int32)
    expected = _core.compute_feature_specificity_sparse(counts, labels, 1)
    for field in ("indices", "indptr"):
        viewed = counts.copy()
        setattr(viewed, field, strided(getattr(viewed, field), JUNK_INT))
        actual = _core.compute_feature_specificity_sparse(viewed, labels, 1)
        assert_dicts_equal(expected, actual)


def test_backed_row_scale_factors_view(tmp_path):
    X = np.arange(1, 61, dtype=np.float64).reshape(12, 5)
    scale = np.linspace(0.5, 1.6, X.shape[0])
    path = tmp_path / "strided-row-scale.h5ad"
    AnnData(X).write_h5ad(path)

    outputs = []
    for factors in (scale, strided(scale, JUNK_FLOAT)):
        op = _core.create_backed_operator(
            file_path=str(path), group_path="/X", chunk_size=8,
            row_scale_factors=factors, n_threads=1,
        )
        try:
            outputs.append(np.asarray(_core.backed_take_columns(op, np.arange(X.shape[1]))))
        finally:
            _core.close_backed_operator(op)

    np.testing.assert_array_equal(outputs[0], X * scale[:, None])
    np.testing.assert_array_equal(outputs[1], outputs[0])


@pytest.mark.parametrize("make_view", [
    lambda s: s[::-1].copy()[::-1],         # same values, negative stride
    lambda s: strided(s, JUNK_FLOAT),       # every other element
], ids=["reversed", "strided"])
def test_reduce_kernel_precomputed_sigma_view(make_view):
    """Public API: ``s[::-1]`` is the usual way to sort eigh output descending."""
    rng = np.random.default_rng(4)
    n_cells, n_genes, k = 60, 50, 5
    X = sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=2) * 10
    u = np.linalg.qr(rng.standard_normal((n_cells, k)))[0]
    v = np.linalg.qr(rng.standard_normal((n_genes, k)))[0]
    sigma = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    d_view = make_view(sigma)
    assert not d_view.flags.c_contiguous
    np.testing.assert_array_equal(d_view, sigma)

    results = []
    for d in (sigma, d_view):
        adata = AnnData(X.copy())
        an.reduce_kernel(adata, n_components=k, verbose=False,
                         precomputed_svd={"u": u, "d": d, "v": v})
        results.append(adata)
    expected, actual = results
    np.testing.assert_array_equal(actual.obsm["action"], expected.obsm["action"])
    np.testing.assert_array_equal(np.asarray(actual.uns["action_params"]["sigma"]),
                                  np.asarray(expected.uns["action_params"]["sigma"]))


def test_archetype_centrality_assignments_view(graph):
    assignments = np.random.default_rng(5).integers(0, 3, graph.shape[0]).astype(np.int32)
    expected = _core.compute_archetype_centrality(graph, assignments)
    actual = _core.compute_archetype_centrality(graph, strided(assignments, JUNK_INT))
    np.testing.assert_array_equal(actual, expected)


def test_run_lpa_fixed_labels_view(graph):
    rng = np.random.default_rng(6)
    labels = rng.integers(1, 4, graph.shape[0]).astype(np.float64)
    fixed = np.array([0, 3, 7, 12], dtype=np.int32)
    expected = _core.run_lpa(graph, labels, 1.0, 3, 3.0, fixed, 1)
    actual = _core.run_lpa(graph, labels, 1.0, 3, 3.0, strided(fixed, JUNK_INT), 1)
    np.testing.assert_array_equal(actual, expected)


def test_guide_thresholds_status_view():
    weights = np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    means = np.array([[0.0, 3.0], [0.5, 4.0], [0.2, 2.5]])
    sigma = np.array([1.0, 0.8, 1.2])
    status = np.array([0, 1, 0], dtype=np.int32)
    expected = _core.derive_guide_thresholds_quantile(weights, means, sigma, status)
    status_view = strided(status, JUNK_INT)
    actual = _core.derive_guide_thresholds_quantile(weights, means, sigma, status_view)
    assert_dicts_equal(expected, actual)


def test_layout_ai_aj_views():
    n = 12
    rows = np.arange(n)
    cols = (rows + 1) % n
    graph = sparse.csr_matrix(
        (np.ones(n * 2), (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)
    )
    rng = np.random.default_rng(8)
    adata = AnnData(rng.normal(size=(n, 5)))
    adata.obsp["actionet"] = graph
    ai = np.linspace(0.1, 1.0, n)
    aj = np.linspace(0.2, 1.1, n)
    kwargs = {
        "initial_coords": rng.normal(size=(n, 3)), "method": "leopold2",
        "n_components": 2, "n_epochs": 20, "seed": 0, "n_threads": 1,
        "verbose": False, "key_added": "X_test", "inplace": False,
    }
    expected = an.layout_network(adata, ai=ai, aj=aj, **kwargs)
    actual = an.layout_network(
        adata, ai=strided(ai, JUNK_FLOAT), aj=strided(aj, JUNK_FLOAT), **kwargs
    )
    np.testing.assert_array_equal(actual.obsm["X_test"], expected.obsm["X_test"])
