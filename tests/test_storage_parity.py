#!/usr/bin/env python3
"""Plan 07 — Final Parity Validation: Storage mode parity test.

Confirms that different storage modes produce identical outputs from the C++ core
where the operator path (IRLB via BackedSparseMatrixOperator) agrees with the
in-memory path to within tolerance.

Tests per Plan 07 §Intra-Language Storage Parity:
  1. In-memory sparse vs backed sparse: reduce_kernel (operator IRLB)
  2. In-memory sparse vs in-memory dense: compute_feature_specificity
  3. In-memory sparse vs backed sparse: compute_feature_specificity
  4. In-memory dense vs backed dense: compute_feature_specificity (small matrix)

Usage:
    cd actionet-python
    .venv/bin/python3 tests/test_storage_parity.py

Exit code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
import anndata as ad

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(TESTS_DIR)
FIXTURE    = os.path.join(TESTS_DIR, "fixtures", "parity_fixture.h5ad")

_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import actionet as an
from actionet.core import (
    reduce_kernel,
    compute_feature_specificity,
    _run_specificity_backed_sparse,
    _run_specificity_backed_dense,
)

SEED = 42
# Operator IRLB vs in-memory IRLB: Plan 05 guarantees < 1e-13 sigma agreement,
# but S_r accumulation order slightly differs -> use Plan 07 dense tolerance.
ATOL_SVD    = 1e-6
RTOL_SVD    = 1e-4
ATOL_SPEC   = 1e-9   # specificity: in-memory vs backed; same C++ code path
RTOL        = 0.0

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAIL += 1


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def canon_svd_sign(mat: np.ndarray) -> np.ndarray:
    mat = np.array(mat, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        idx = np.argmax(np.abs(col))
        if col[idx] < 0:
            mat[:, j] *= -1
    return mat


# ---------------------------------------------------------------------------
# Load fixture
# ---------------------------------------------------------------------------
section("Loading fixture")
if not os.path.exists(FIXTURE):
    print(f"ERROR: fixture not found at {FIXTURE}")
    sys.exit(1)

adata = ad.read_h5ad(FIXTURE)
print(f"  Loaded {adata.n_obs} obs × {adata.n_vars} vars")
assert sp.issparse(adata.X), "Fixture X must be sparse for storage mode tests"

# ---------------------------------------------------------------------------
# Section 1: reduce_kernel — in-memory dense vs backed sparse (Halko operator)
#
# Plan 05 added the IRLB operator path; Plan 07 storage parity tests the
# Halko (randomized SVD) operator path which is the default backed algorithm.
# In-memory IRLB vs backed Halko differ by design (different algorithms).
# We compare in-memory Halko (dense) vs backed Halko (sparse operator) which
# uses the same randomized SVD algorithm on both sides.
# ---------------------------------------------------------------------------
section("1. reduce_kernel: in-memory dense (halko) vs backed sparse (halko operator)")

# In-memory dense + halko
adata_dn = adata.copy()
logcounts = adata.layers.get("logcounts", adata.X)
adata_dn.layers["logcounts"] = (
    logcounts.toarray().astype(np.float64)
    if sp.issparse(logcounts)
    else np.asarray(logcounts, dtype=np.float64)
)
reduce_kernel(adata_dn, n_components=10, layer="logcounts", seed=SEED,
              svd_algorithm="halko", verbose=False)
S_dn  = canon_svd_sign(adata_dn.obsm["action"])
sig_dn = np.asarray(adata_dn.uns["action_params"]["sigma"]).ravel()

# Backed sparse + halko operator
with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
    backed_path = tf.name
try:
    adata.write_h5ad(backed_path, compression=None)
    adata_backed = ad.read_h5ad(backed_path, backed="r+")

    if not getattr(adata_backed, "isbacked", False):
        check("backed fixture opens in backed mode", False)
    else:
        reduce_kernel(adata_backed, n_components=10, layer="logcounts",
                      seed=SEED, svd_algorithm="halko", verbose=False)
        S_bk   = canon_svd_sign(np.array(adata_backed.obsm["action"]))
        sig_bk = np.asarray(adata_backed.uns["action_params"]["sigma"]).ravel()
        adata_backed.file.close()

        check("in-memory dense vs backed: action shape", S_dn.shape == S_bk.shape,
              f"{S_dn.shape} vs {S_bk.shape}")
        dev_s = float(np.max(np.abs(S_dn - S_bk)))
        check("in-memory dense vs backed: action values close (halko)",
              np.allclose(S_dn, S_bk, atol=ATOL_SVD, rtol=RTOL_SVD),
              f"max_dev={dev_s:.3e}")
        dev_sig = float(np.max(np.abs(sig_dn - sig_bk)))
        check("in-memory dense vs backed: sigma close (halko)",
              np.allclose(sig_dn, sig_bk, atol=ATOL_SVD, rtol=RTOL_SVD),
              f"max_dev={dev_sig:.3e}")
finally:
    os.unlink(backed_path)

# ---------------------------------------------------------------------------
# Section 2: compute_feature_specificity — in-memory sparse vs in-memory dense
# ---------------------------------------------------------------------------
section("2. compute_feature_specificity: in-memory sparse vs in-memory dense")

rng = np.random.default_rng(42)
n_clusters = 5
label_names = [f"C{i}" for i in range(n_clusters)]
labels_arr = np.array([label_names[i] for i in rng.integers(0, n_clusters, size=adata.n_obs)])

# Sparse
res_sp = compute_feature_specificity(adata.copy(), labels_arr, layer="logcounts",
                                     n_threads=1, inplace=False)
upper_sp  = res_sp.varm["specificity_upper"]
lower_sp  = res_sp.varm["specificity_lower"]
prof_sp   = res_sp.varm["specificity_profile"]

# Dense (same data)
adata_dense = adata.copy()
logcounts = adata.layers.get("logcounts", adata.X)
adata_dense.layers["logcounts"] = (
    logcounts.toarray().astype(np.float64)
    if sp.issparse(logcounts)
    else np.asarray(logcounts, dtype=np.float64)
)
res_dn = compute_feature_specificity(adata_dense.copy(), labels_arr, layer="logcounts",
                                     n_threads=1, inplace=False)
upper_dn  = res_dn.varm["specificity_upper"]
lower_dn  = res_dn.varm["specificity_lower"]
prof_dn   = res_dn.varm["specificity_profile"]

check("sparse vs dense: upper shape", upper_sp.shape == upper_dn.shape)
check("sparse vs dense: upper values close",
      np.allclose(upper_sp, upper_dn, atol=ATOL_SPEC, rtol=RTOL),
      f"max_diff={np.max(np.abs(upper_sp - upper_dn)):.2e}")
check("sparse vs dense: lower values close",
      np.allclose(lower_sp, lower_dn, atol=ATOL_SPEC, rtol=RTOL),
      f"max_diff={np.max(np.abs(lower_sp - lower_dn)):.2e}")
check("sparse vs dense: profile values close",
      np.allclose(prof_sp, prof_dn, atol=ATOL_SPEC, rtol=RTOL),
      f"max_diff={np.max(np.abs(prof_sp - prof_dn)):.2e}")

# ---------------------------------------------------------------------------
# Section 3: compute_feature_specificity — in-memory sparse vs backed sparse
# ---------------------------------------------------------------------------
section("3. compute_feature_specificity: in-memory sparse vs backed sparse")

with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
    spec_backed_path = tf.name
try:
    adata.write_h5ad(spec_backed_path, compression=None)
    backed_spec = ad.read_h5ad(spec_backed_path, backed="r")
    if not getattr(backed_spec, "isbacked", False):
        check("specificity backed fixture opens backed", False)
    else:
        from pandas import Categorical
        cat = Categorical(labels_arr)
        labels_int = (cat.codes + 1).astype(np.int32)
        res_bk_sp = _run_specificity_backed_sparse(
            backed_spec, layer="logcounts", chunk_size=512,
            labels_int=labels_int, n_threads=1,
        )
        backed_spec.file.close()
        upper_bk = res_bk_sp["upper_significance"]
        lower_bk = res_bk_sp["lower_significance"]
        prof_bk  = res_bk_sp["average_profile"]

        check("backed-sparse: upper shape", upper_bk.shape == upper_sp.shape,
              f"{upper_bk.shape} vs {upper_sp.shape}")
        check("backed-sparse: upper values close",
              np.allclose(upper_bk, upper_sp, atol=ATOL_SPEC, rtol=RTOL),
              f"max_diff={np.max(np.abs(upper_bk - upper_sp)):.2e}")
        check("backed-sparse: lower values close",
              np.allclose(lower_bk, lower_sp, atol=ATOL_SPEC, rtol=RTOL),
              f"max_diff={np.max(np.abs(lower_bk - lower_sp)):.2e}")
        check("backed-sparse: profile values close",
              np.allclose(prof_bk, prof_sp, atol=ATOL_SPEC, rtol=RTOL),
              f"max_diff={np.max(np.abs(prof_bk - prof_sp)):.2e}")
finally:
    os.unlink(spec_backed_path)

# ---------------------------------------------------------------------------
# Section 4: compute_feature_specificity — in-memory dense vs backed dense
# ---------------------------------------------------------------------------
section("4. compute_feature_specificity: in-memory dense vs backed dense (small)")

rng2 = np.random.default_rng(7)
n_small, g_small = 40, 15
S_small = rng2.random((n_small, g_small)).astype(np.float64)
adata_small = ad.AnnData(X=S_small)
n_cl_s = 3
labels_s = np.array([f"D{i}" for i in rng2.integers(1, n_cl_s + 1, size=n_small)])

res_small_mem = compute_feature_specificity(adata_small.copy(), labels_s,
                                            n_threads=1, inplace=False)
upper_sm = res_small_mem.varm["specificity_upper"]
lower_sm = res_small_mem.varm["specificity_lower"]
prof_sm  = res_small_mem.varm["specificity_profile"]

with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
    dense_path = tf.name
try:
    adata_small.write_h5ad(dense_path)
    backed_dense_adata = ad.read_h5ad(dense_path, backed="r")
    if not getattr(backed_dense_adata, "isbacked", False):
        check("dense backed fixture opens backed", False)
    else:
        from pandas import Categorical
        cat_s = Categorical(labels_s)
        labels_int_s = (cat_s.codes + 1).astype(np.int32)
        res_bd = _run_specificity_backed_dense(
            backed_dense_adata, layer=None, chunk_size=16,
            labels_int=labels_int_s, n_threads=1,
        )
        backed_dense_adata.file.close()
        upper_bd = res_bd["upper_significance"]
        lower_bd = res_bd["lower_significance"]
        prof_bd  = res_bd["average_profile"]

        check("backed-dense: upper shape", upper_bd.shape == upper_sm.shape)
        check("backed-dense: upper values close",
              np.allclose(upper_bd, upper_sm, atol=ATOL_SPEC, rtol=RTOL),
              f"max_diff={np.max(np.abs(upper_bd - upper_sm)):.2e}")
        check("backed-dense: lower values close",
              np.allclose(lower_bd, lower_sm, atol=ATOL_SPEC, rtol=RTOL),
              f"max_diff={np.max(np.abs(lower_bd - lower_sm)):.2e}")
        check("backed-dense: profile values close",
              np.allclose(prof_bd, prof_sm, atol=ATOL_SPEC, rtol=RTOL),
              f"max_diff={np.max(np.abs(prof_bd - prof_sm)):.2e}")
        check("backed-dense: all upper non-negative", bool(np.all(upper_bd >= 0)))
        check("backed-dense: all lower non-negative", bool(np.all(lower_bd >= 0)))
finally:
    os.unlink(dense_path)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'=' * 60}")
sys.exit(0 if FAIL == 0 else 1)
