"""Low-level parity tests for backed matrix operators.

Focused regressions for the paths the io/backed_h5ad refactor touches:
- BackedSparseMatrixOperator::takeColumnsDense / takeColumnsSparse
  (CSR + CSC) — including duplicate column indices, arbitrary row_indices
  reorderings, and combined transforms (row_scale + log1p + log_scale).
- BackedDenseMatrixOperator::takeColumnsDense / takeColumnsSparse
  (dense path via the same public binding) — same matrix.
- Matvec/rmatvec/matmat/rmatmat semantics — exercised indirectly by
  ``run_svd_backed_operator`` versus in-memory ``run_svd``.

The `apply_log1p` path uses the Paul Mineiro `fastlog()` approximation,
so transforms involving log1p are compared with a looser (`rtol=5e-3`)
tolerance per the operator header documentation.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import pandas as pd
import anndata as ad

import actionet._core as _core


# ---------------------------------------------------------------------------
# Small synthetic fixtures — kept intentionally lean so CI cost is minimal.
# ---------------------------------------------------------------------------

_SHAPES = [(64, 32), (256, 128)]
_DENSITIES = [0.15]
_LOG_TOL = dict(rtol=5e-3, atol=5e-6)     # fastlog approximation regime
_EXACT_TOL = dict(rtol=1e-10, atol=1e-12) # no-log1p paths


def _make_matrix(shape, density, fmt, seed):
    rng = np.random.default_rng(seed)
    if fmt == "dense":
        X = rng.random(shape) * 5.0
        X[rng.random(shape) > density] = 0.0  # sparsify but keep dense storage
        return X
    S = sp.random(*shape, density=density, random_state=rng,
                  format=fmt, dtype=np.float64)
    S.data = np.abs(S.data) * 5.0  # counts-like, positive
    S.eliminate_zeros()
    if fmt == "csr":
        return sp.csr_matrix(S)
    return sp.csc_matrix(S)


def _write_h5ad(path, X, fmt):
    n_obs, n_var = X.shape
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    if fmt == "dense":
        adata = ad.AnnData(X=np.asarray(X, dtype=np.float64), obs=obs, var=var)
    else:
        adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return str(path)


def _apply_transform_reference(X_orig, row_scale, apply_log1p, log_scale):
    """Reference transform mirroring the operator's lazy transform.

    NOTE: The C++ path uses ``fastlog(1 + x)`` (float precision).  Callers
    that pass ``apply_log1p=True`` must compare with the loose tolerance.
    """
    if sp.issparse(X_orig):
        X = X_orig.copy().astype(np.float64)
        if row_scale is not None:
            X = sp.diags(row_scale, format="csc") @ X
        if apply_log1p:
            data = np.log1p(X.data.astype(np.float32)).astype(np.float64)
            if log_scale != 1.0:
                data = data * log_scale
            X = X.copy()
            X.data = data
        return X
    else:
        X = np.asarray(X_orig, dtype=np.float64).copy()
        if row_scale is not None:
            X = X * row_scale[:, None]
        if apply_log1p:
            X = np.log1p(X.astype(np.float32)).astype(np.float64)
            if log_scale != 1.0:
                X = X * log_scale
        return X


def _open_op(path, *, chunk_size, row_scale, apply_log1p, log_scale, n_threads=1):
    kwargs = dict(
        file_path=path,
        group_path="/X",
        chunk_size=chunk_size,
        apply_log1p=apply_log1p,
        log_scale=log_scale,
        n_threads=n_threads,
    )
    if row_scale is not None:
        kwargs["row_scale_factors"] = np.asarray(row_scale, dtype=np.float64)
    return _core.create_backed_operator(**kwargs)


# ---------------------------------------------------------------------------
# Parametrised test matrix
# ---------------------------------------------------------------------------

_FMTS = ["csr", "csc", "dense"]

_TRANSFORMS = [
    # (id, row_scale_fn, apply_log1p, log_scale, tol)
    ("none",             None,                              False, 1.0, _EXACT_TOL),
    ("row_scale",        lambda n: 1.0 + np.arange(n) / n,  False, 1.0, _EXACT_TOL),
    ("log1p",            None,                              True,  1.0, _LOG_TOL),
    ("row_scale+log1p",  lambda n: 1.0 + np.arange(n) / n,  True,  1.0, _LOG_TOL),
    ("row_scale+log1p+logscale",
                         lambda n: 1.0 + np.arange(n) / n,  True,  1.0 / np.log(2.0), _LOG_TOL),
]


@pytest.fixture(scope="module")
def _matrices():
    """Materialised (X_orig, X_ref_by_transform) per (fmt, shape)."""
    cache = {}
    for fmt in _FMTS:
        for shape in _SHAPES:
            X_orig = _make_matrix(shape, _DENSITIES[0], fmt, seed=hash((fmt, shape)) & 0xFFFF)
            per_transform = {}
            for tid, rs_fn, apply_log1p, log_scale, _tol in _TRANSFORMS:
                rs = rs_fn(shape[0]) if rs_fn is not None else None
                per_transform[tid] = _apply_transform_reference(
                    X_orig, rs, apply_log1p, log_scale,
                )
            cache[(fmt, shape)] = (X_orig, per_transform)
    return cache


def _params_id(fmt, shape, tid):
    return f"{fmt}-{shape[0]}x{shape[1]}-{tid}"


_ALL_PARAMS = [
    (fmt, shape, tid, rs_fn, apply_log1p, log_scale, tol)
    for fmt in _FMTS
    for shape in _SHAPES
    for (tid, rs_fn, apply_log1p, log_scale, tol) in _TRANSFORMS
]


# ---------------------------------------------------------------------------
# takeColumnsDense parity (unique cols, all rows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt,shape,tid,rs_fn,apply_log1p,log_scale,tol",
    _ALL_PARAMS,
    ids=[_params_id(f, s, t) for (f, s, t, *_ ) in _ALL_PARAMS],
)
def test_take_columns_dense_unique(tmp_path, _matrices, fmt, shape, tid, rs_fn,
                                   apply_log1p, log_scale, tol):
    X_orig, refs = _matrices[(fmt, shape)]
    X_ref = refs[tid]
    if sp.issparse(X_ref):
        X_ref = X_ref.toarray()
    else:
        X_ref = np.asarray(X_ref)

    path = _write_h5ad(tmp_path / "m.h5ad", X_orig, fmt=fmt)
    op = _open_op(
        path, chunk_size=17,
        row_scale=(rs_fn(shape[0]) if rs_fn is not None else None),
        apply_log1p=apply_log1p, log_scale=log_scale,
    )
    cols = np.array([0, 5, 3, shape[1] - 1, shape[1] // 2], dtype=np.int64)
    cols = np.unique(cols)
    out = _core.backed_take_columns(op, cols, prefer_sparse=False)

    np.testing.assert_allclose(out, X_ref[:, cols], **tol)


# ---------------------------------------------------------------------------
# takeColumnsDense parity with duplicate columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt,shape,tid,rs_fn,apply_log1p,log_scale,tol",
    _ALL_PARAMS,
    ids=[_params_id(f, s, t) for (f, s, t, *_ ) in _ALL_PARAMS],
)
def test_take_columns_dense_duplicates(tmp_path, _matrices, fmt, shape, tid, rs_fn,
                                       apply_log1p, log_scale, tol):
    X_orig, refs = _matrices[(fmt, shape)]
    X_ref = refs[tid]
    if sp.issparse(X_ref):
        X_ref = X_ref.toarray()
    else:
        X_ref = np.asarray(X_ref)

    path = _write_h5ad(tmp_path / "m.h5ad", X_orig, fmt=fmt)
    op = _open_op(
        path, chunk_size=17,
        row_scale=(rs_fn(shape[0]) if rs_fn is not None else None),
        apply_log1p=apply_log1p, log_scale=log_scale,
    )
    cols = np.array([3, 3, 3, 7, 3, 7, 1], dtype=np.int64)
    out = _core.backed_take_columns(op, cols, prefer_sparse=False)

    np.testing.assert_allclose(out, X_ref[:, cols], **tol)


# ---------------------------------------------------------------------------
# takeColumnsDense parity with row_indices (subset and reorder)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt,shape,tid,rs_fn,apply_log1p,log_scale,tol",
    _ALL_PARAMS,
    ids=[_params_id(f, s, t) for (f, s, t, *_ ) in _ALL_PARAMS],
)
def test_take_columns_dense_row_indices(tmp_path, _matrices, fmt, shape, tid, rs_fn,
                                        apply_log1p, log_scale, tol):
    X_orig, refs = _matrices[(fmt, shape)]
    X_ref = refs[tid]
    if sp.issparse(X_ref):
        X_ref = X_ref.toarray()
    else:
        X_ref = np.asarray(X_ref)

    path = _write_h5ad(tmp_path / "m.h5ad", X_orig, fmt=fmt)
    op = _open_op(
        path, chunk_size=17,
        row_scale=(rs_fn(shape[0]) if rs_fn is not None else None),
        apply_log1p=apply_log1p, log_scale=log_scale,
    )
    cols = np.array([0, 5, 3], dtype=np.int64)
    rows = np.array([shape[0] - 1, 0, 4, shape[0] // 2, 2, 1], dtype=np.int64)
    out = _core.backed_take_columns(op, cols, row_indices=rows, prefer_sparse=False)

    np.testing.assert_allclose(out, X_ref[np.ix_(rows, cols)], **tol)


# ---------------------------------------------------------------------------
# takeColumnsSparse parity (unique + duplicate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt,shape,tid,rs_fn,apply_log1p,log_scale,tol",
    # dense-storage input still returns a sparse output through the binding.
    _ALL_PARAMS,
    ids=[_params_id(f, s, t) for (f, s, t, *_ ) in _ALL_PARAMS],
)
def test_take_columns_sparse_duplicates(tmp_path, _matrices, fmt, shape, tid, rs_fn,
                                        apply_log1p, log_scale, tol):
    X_orig, refs = _matrices[(fmt, shape)]
    X_ref_dense = refs[tid].toarray() if sp.issparse(refs[tid]) else np.asarray(refs[tid])

    path = _write_h5ad(tmp_path / "m.h5ad", X_orig, fmt=fmt)
    op = _open_op(
        path, chunk_size=17,
        row_scale=(rs_fn(shape[0]) if rs_fn is not None else None),
        apply_log1p=apply_log1p, log_scale=log_scale,
    )
    cols = np.array([4, 4, 9, 4, 9, 0], dtype=np.int64)
    out = _core.backed_take_columns(op, cols, prefer_sparse=True)
    assert sp.issparse(out)
    np.testing.assert_allclose(out.toarray(), X_ref_dense[:, cols], **tol)


# ---------------------------------------------------------------------------
# SVD parity — exercises matvec/rmatvec/matmat/rmatmat end-to-end via PRIMME.
# Compares singular values (sign/rotation invariant, so only sigma is checked).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_backed_svd_sigma_parity(tmp_path, fmt):
    shape = (128, 64)
    X_orig = _make_matrix(shape, 0.20, fmt, seed=13)
    path = _write_h5ad(tmp_path / "svd.h5ad", X_orig, fmt=fmt)

    op = _core.create_backed_operator(
        file_path=path, group_path="/X", chunk_size=32, n_threads=1,
    )
    k = 10
    res_backed = _core.run_svd_backed_operator(
        op, k=k, max_it=200, seed=1, algorithm=1, verbose=False,  # 1 = Halko
    )

    if fmt == "dense":
        X_ref = np.asarray(X_orig)
    else:
        X_ref = X_orig.toarray()
    # Reference sigma from full SVD (only sigma is compared).
    u_ref, s_ref, vt_ref = np.linalg.svd(X_ref, full_matrices=False)

    s_backed = np.sort(np.asarray(res_backed["d"]).reshape(-1))[::-1]
    s_ref_top = np.sort(s_ref)[::-1][:k]
    np.testing.assert_allclose(s_backed[:k], s_ref_top, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# SVD parity with lazy transform (row_scale + log1p) — matvec path exercised
# with the operator-side transform applied at read time.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_backed_svd_sigma_parity_with_transform(tmp_path, fmt):
    shape = (128, 64)
    X_orig = _make_matrix(shape, 0.20, fmt, seed=17)
    path = _write_h5ad(tmp_path / "svd_t.h5ad", X_orig, fmt=fmt)

    row_scale = 1.0 + np.arange(shape[0]) / shape[0]
    log_scale = 1.0 / np.log(2.0)

    op = _core.create_backed_operator(
        file_path=path, group_path="/X", chunk_size=32,
        row_scale_factors=row_scale, apply_log1p=True, log_scale=log_scale,
        n_threads=1,
    )
    k = 10
    res_backed = _core.run_svd_backed_operator(
        op, k=k, max_it=200, seed=1, algorithm=1, verbose=False,  # 1 = Halko
    )

    X_ref = _apply_transform_reference(
        X_orig, row_scale, apply_log1p=True, log_scale=log_scale,
    )
    X_ref = X_ref.toarray() if sp.issparse(X_ref) else np.asarray(X_ref)
    _, s_ref, _ = np.linalg.svd(X_ref, full_matrices=False)

    s_backed = np.sort(np.asarray(res_backed["d"]).reshape(-1))[::-1]
    s_ref_top = np.sort(s_ref)[::-1][:k]
    # fastlog approximation: loosen sigma tolerance
    np.testing.assert_allclose(s_backed[:k], s_ref_top, rtol=1e-2, atol=1e-4)
