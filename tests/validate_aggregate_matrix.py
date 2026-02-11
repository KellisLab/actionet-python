#!/usr/bin/env python3
"""Validate aggregate_matrix against a pure NumPy/SciPy baseline."""

import numpy as np
import scipy.sparse as sp

from actionet import aggregate_matrix


def assert_close(a, b, name, atol=1e-8, rtol=1e-6):
    if sp.issparse(a):
        a = a.toarray()
    if sp.issparse(b):
        b = b.toarray()
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.max(np.abs(a - b))
        raise AssertionError(f"{name}: mismatch (max abs diff={diff})")


def make_groups(n, n_groups, rng):
    labels = rng.integers(0, n_groups, size=n)
    return np.array([f"g{val}" for val in labels], dtype=object)


def _aggregate_dense_by_group(X, v, axis=0, func="sum"):
    X = np.asarray(X)
    v = np.asarray(v)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if func not in ("sum", "mean", "var"):
        raise ValueError("func must be 'sum', 'mean', or 'var'")

    unique_labels, inverse = np.unique(v, return_inverse=True)
    aggs = []
    for i in range(len(unique_labels)):
        idx = np.where(inverse == i)[0]
        group = X[idx, :] if axis == 0 else X[:, idx]
        if func == "sum":
            agg_result = group.sum(axis=axis, keepdims=True)
        elif func == "mean":
            agg_result = group.mean(axis=axis, keepdims=True)
        else:
            if idx.size <= 1:
                shape = (1, X.shape[1]) if axis == 0 else (X.shape[0], 1)
                agg_result = np.zeros(shape, dtype=X.dtype)
            else:
                agg_result = group.var(axis=axis, keepdims=True, ddof=1)
        aggs.append(agg_result)

    return np.concatenate(aggs, axis=0 if axis == 0 else 1)


def _aggregate_sparse_by_group(X, v, axis=0, func="sum"):
    from scipy.sparse import vstack, hstack, csr_matrix

    v = np.asarray(v)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if func not in ("sum", "mean", "var"):
        raise ValueError("func must be 'sum', 'mean', or 'var'")

    if axis == 0:
        slicer = lambda idx: X[idx]
        stack_fn = vstack
        reduce_axis = 0
        reshape_fn = lambda arr: arr.reshape(1, -1)
    else:
        slicer = lambda idx: X[:, idx]
        stack_fn = hstack
        reduce_axis = 1
        reshape_fn = lambda arr: arr.reshape(-1, 1)

    unique_labels, inverse = np.unique(v, return_inverse=True)
    aggs = []
    for i in range(len(unique_labels)):
        idx = np.where(inverse == i)[0]
        group = slicer(idx)
        if func == "sum":
            agg_result = group.sum(axis=reduce_axis)
        elif func == "mean":
            agg_result = group.mean(axis=reduce_axis)
        else:
            if idx.size <= 1:
                shape = (1, X.shape[1]) if axis == 0 else (X.shape[0], 1)
                agg_result = np.zeros(shape, dtype=np.float64)
            else:
                agg_result = group.toarray().var(axis=reduce_axis, ddof=1)
                agg_result = reshape_fn(agg_result)
        aggs.append(csr_matrix(agg_result))

    return stack_fn(aggs)


def aggregate_baseline(X, v, axis=0, func="sum"):
    if sp.issparse(X):
        return _aggregate_sparse_by_group(X, v, axis=axis, func=func)
    return _aggregate_dense_by_group(X, v, axis=axis, func=func)


def run_case(name, X, group_vec, dim, method, return_sparse):
    axis = 1 if dim == 1 else 0
    got = aggregate_matrix(X, group_vec, dim=dim, method=method, return_sparse=return_sparse)
    ref = aggregate_baseline(X, group_vec, axis=axis, func=method)
    assert_close(got, ref, name)


def main():
    rng = np.random.default_rng(42)
    dense = rng.random((200, 150), dtype=np.float64)
    sparse = sp.random(200, 150, density=0.05, format="csr", random_state=42, dtype=np.float64)

    for method in ("sum", "mean", "var"):
        for dim in (1, 2):
            group_vec = make_groups(dense.shape[1] if dim == 1 else dense.shape[0], 7, rng)
            run_case(f"dense dim={dim} method={method}", dense, group_vec, dim, method, return_sparse=False)
            run_case(f"sparse dim={dim} method={method} dense_out", sparse, group_vec, dim, method, return_sparse=False)
            run_case(f"sparse dim={dim} method={method} sparse_out", sparse, group_vec, dim, method, return_sparse=True)

    print("All aggregate_matrix validations passed.")


if __name__ == "__main__":
    main()
