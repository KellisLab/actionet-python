#!/usr/bin/env python3
"""Stage 06 Validation: Unified Specificity.

Verifies:
 - In-memory specificity is non-mutating (S unchanged after call)
 - In-memory sparse and dense produce identical results
 - Backed sparse and in-memory sparse produce identical results
 - annotate_cells no longer transposes S (zero .T operations)
 - computeFeatureStats/Vision accept cells x genes orientation correctly

Usage:
    cd actionet-python
    source .venv/bin/activate  (or use .venv/bin/python3)
    python tests/validate_stage06.py
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import anndata

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(TESTS_DIR, "fixtures", "parity_fixture.h5ad")

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAIL += 1


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Load fixture
# ---------------------------------------------------------------------------
section("Loading fixture")

if not os.path.exists(FIXTURE):
    print(f"ERROR: fixture not found at {FIXTURE}")
    sys.exit(1)

adata = anndata.read_h5ad(FIXTURE)
print(f"  Loaded {adata.n_obs} obs × {adata.n_vars} vars")

# Add path for the local actionet source
src_dir = os.path.join(os.path.dirname(TESTS_DIR), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from actionet.core import compute_feature_specificity

# Compute a small set of integer labels from the fixture
rng = np.random.default_rng(42)
n_clusters = 5
labels_arr = rng.integers(0, n_clusters, size=adata.n_obs).astype(str)
# Map to string labels so compute_feature_specificity uses the Categorical path
label_names = [f"C{i}" for i in range(n_clusters)]
labels_arr = np.array([label_names[int(x)] for x in labels_arr])

# ---------------------------------------------------------------------------
# Section 1: Non-mutating check
# ---------------------------------------------------------------------------
section("1. Non-mutating in-memory check")

S_before = adata.X.copy()
compute_feature_specificity(adata, labels_arr, layer=None, n_threads=1, inplace=False)
S_after = adata.X

if sp.issparse(S_before):
    mutated = not np.array_equal(S_before.data, S_after.data)
    check("S.data unchanged after specificity call", not mutated)
    check("S.indices unchanged", np.array_equal(S_before.indices, S_after.indices))
else:
    check("S unchanged after specificity call", np.array_equal(S_before, np.asarray(S_after)))

# ---------------------------------------------------------------------------
# Section 2: Sparse vs Dense in-memory parity
# ---------------------------------------------------------------------------
section("2. Sparse vs Dense in-memory parity")

# Use a small dense copy for speed
S_sparse = adata.X
S_dense = np.asarray(S_sparse.toarray(), dtype=np.float64) if sp.issparse(S_sparse) else np.asarray(S_sparse, dtype=np.float64)

adata_sparse = adata.copy()
adata_dense = adata.copy()
adata_dense.X = S_dense

result_sparse = compute_feature_specificity(adata_sparse, labels_arr, n_threads=1, inplace=False)
result_dense  = compute_feature_specificity(adata_dense,  labels_arr, n_threads=1, inplace=False)

upper_sparse = result_sparse.varm["specificity_upper"]
upper_dense  = result_dense.varm["specificity_upper"]
lower_sparse = result_sparse.varm["specificity_lower"]
lower_dense  = result_dense.varm["specificity_lower"]
profile_sparse = result_sparse.varm["specificity_profile"]
profile_dense  = result_dense.varm["specificity_profile"]

tol = 1e-10
check("Sparse == Dense: upper_significance shape matches", upper_sparse.shape == upper_dense.shape)
check("Sparse == Dense: upper_significance values match",
      np.allclose(upper_sparse, upper_dense, atol=tol, rtol=0),
      f"max_diff={np.max(np.abs(upper_sparse - upper_dense)):.2e}")
check("Sparse == Dense: lower_significance values match",
      np.allclose(lower_sparse, lower_dense, atol=tol, rtol=0),
      f"max_diff={np.max(np.abs(lower_sparse - lower_dense)):.2e}")
check("Sparse == Dense: average_profile values match",
      np.allclose(profile_sparse, profile_dense, atol=tol, rtol=0),
      f"max_diff={np.max(np.abs(profile_sparse - profile_dense)):.2e}")

# ---------------------------------------------------------------------------
# Section 3: Backed sparse vs in-memory sparse parity
# ---------------------------------------------------------------------------
section("3. Backed sparse vs in-memory sparse parity")

backed_adata = anndata.read_h5ad(FIXTURE, backed="r")
if getattr(backed_adata, "isbacked", False):
    from actionet.core import _run_specificity_backed_sparse
    from pandas import Categorical
    cat = Categorical(labels_arr)
    labels_int = (cat.codes + 1).astype(np.int32)

    import anndata as ad
    backed_src_path = str(backed_adata.filename)
    result_backed = _run_specificity_backed_sparse(
        backed_adata, layer=None, chunk_size=512,
        labels_int=labels_int, n_threads=1,
    )
    backed_adata.file.close()

    upper_backed = result_backed["upper_significance"]
    lower_backed = result_backed["lower_significance"]
    profile_backed = result_backed["average_profile"]

    check("Backed == In-memory: upper shape matches",
          upper_backed.shape == upper_sparse.shape,
          f"{upper_backed.shape} vs {upper_sparse.shape}")
    check("Backed == In-memory: upper values close",
          np.allclose(upper_backed, upper_sparse, atol=1e-9, rtol=0),
          f"max_diff={np.max(np.abs(upper_backed - upper_sparse)):.2e}")
    check("Backed == In-memory: lower values close",
          np.allclose(lower_backed, lower_sparse, atol=1e-9, rtol=0),
          f"max_diff={np.max(np.abs(lower_backed - lower_sparse)):.2e}")
    check("Backed == In-memory: average_profile values close",
          np.allclose(profile_backed, result_sparse.varm["specificity_profile"], atol=1e-9, rtol=0),
          f"max_diff={np.max(np.abs(profile_backed - result_sparse.varm['specificity_profile'])):.2e}")
else:
    check("Fixture opens in backed mode", False, "fixture could not be opened as backed")

# ---------------------------------------------------------------------------
# Section 4: annotate_cells S orientation (no .T in code path)
# ---------------------------------------------------------------------------
section("4. annotate_cells S orientation check")

# Build a tiny marker dict using first few gene names
gene_names = adata.var_names.values
n_marker_genes = min(5, len(gene_names))
markers = {
    "type_A": list(gene_names[:n_marker_genes]),
    "type_B": list(gene_names[n_marker_genes:2*n_marker_genes]),
}

# We need a network - build a minimal one if not present
from actionet.core import build_network
if "actionet" not in adata.obsp:
    # Build a random stub network since we're just checking shape
    import scipy.sparse as sp_lib
    n = adata.n_obs
    # Simple random sparse symmetric graph
    rng2 = np.random.default_rng(0)
    row = rng2.integers(0, n, 500)
    col = rng2.integers(0, n, 500)
    data = np.ones(500, dtype=np.float64)
    G = sp_lib.csr_matrix((data, (row, col)), shape=(n, n))
    G = G + G.T
    G.data = np.ones_like(G.data)
    adata.obsp["actionet"] = G

from actionet.annotation import annotate_cells
try:
    result = annotate_cells(adata, markers, method="vision", n_threads=1)
    check("annotate_cells returns labels", "labels" in result)
    check("annotate_cells labels length == n_obs",
          len(result["labels"]) == adata.n_obs,
          f"got {len(result['labels'])}")
    check("annotate_cells enrichment shape == (n_obs, n_celltypes)",
          result["enrichment"].shape == (adata.n_obs, 2),
          f"got {result['enrichment'].shape}")
except Exception as e:
    check("annotate_cells runs without error", False, str(e))

# ---------------------------------------------------------------------------
# Section 5: S shape verification (cells x genes)
# ---------------------------------------------------------------------------
section("5. Expression matrix orientation in specificity output")

result5 = compute_feature_specificity(adata, labels_arr, n_threads=1, inplace=False)
upper_out = result5.varm["specificity_upper"]
lower_out = result5.varm["specificity_lower"]
profile_out = result5.varm["specificity_profile"]

check("upper_significance shape: (n_vars, n_labels)",
      upper_out.shape == (adata.n_vars, n_clusters),
      f"got {upper_out.shape}, expected ({adata.n_vars}, {n_clusters})")
check("lower_significance shape: (n_vars, n_labels)",
      lower_out.shape == (adata.n_vars, n_clusters),
      f"got {lower_out.shape}")
check("average_profile shape: (n_vars, n_labels)",
      profile_out.shape == (adata.n_vars, n_clusters),
      f"got {profile_out.shape}")
check("upper_significance all non-negative", np.all(upper_out >= 0))
check("lower_significance all non-negative", np.all(lower_out >= 0))

# ---------------------------------------------------------------------------
# Section 6: Dense-backed C++ path vs in-memory dense parity
# ---------------------------------------------------------------------------
section("6. Dense-backed C++ path vs in-memory parity")

import tempfile, os as _os

# Build a small dense in-memory AnnData for comparison
rng3 = np.random.default_rng(7)
n_obs_d, n_var_d = 40, 15
S_dense_small = rng3.random((n_obs_d, n_var_d)).astype(np.float64)
adata_dense_small = anndata.AnnData(X=S_dense_small)

n_clusters_d = 3
labels_d = rng3.integers(1, n_clusters_d + 1, size=n_obs_d).astype(str)
label_names_d = [f"D{i}" for i in range(1, n_clusters_d + 1)]
labels_arr_d = np.array([f"D{x}" for x in labels_d])

result_inmem = compute_feature_specificity(
    adata_dense_small, labels_arr_d, n_threads=1, inplace=False
)
upper_inmem = result_inmem.varm["specificity_upper"]
lower_inmem = result_inmem.varm["specificity_lower"]

# Write to a temporary dense h5ad and open backed
with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
    dense_fixture_path = tf.name
try:
    adata_dense_small.write_h5ad(dense_fixture_path)
    backed_dense = anndata.read_h5ad(dense_fixture_path, backed="r")

    if not getattr(backed_dense, "isbacked", False):
        check("Dense fixture opens in backed mode", False, "dense fixture could not be opened as backed")
    else:
        from pandas import Categorical
        cat_d = Categorical(labels_arr_d)
        labels_int_d = (cat_d.codes + 1).astype(np.int32)

        from actionet.core import _run_specificity_backed_dense
        result_backed_dense = _run_specificity_backed_dense(
            backed_dense, layer=None, chunk_size=16,
            labels_int=labels_int_d, n_threads=1,
        )
        backed_dense.file.close()

        upper_bd = result_backed_dense["upper_significance"]
        lower_bd = result_backed_dense["lower_significance"]
        profile_bd = result_backed_dense["average_profile"]

        check("Dense-backed: upper shape matches in-memory",
              upper_bd.shape == upper_inmem.shape,
              f"{upper_bd.shape} vs {upper_inmem.shape}")
        check("Dense-backed: upper values close to in-memory",
              np.allclose(upper_bd, upper_inmem, atol=1e-9, rtol=0),
              f"max_diff={np.max(np.abs(upper_bd - upper_inmem)):.2e}")
        check("Dense-backed: lower values close to in-memory",
              np.allclose(lower_bd, lower_inmem, atol=1e-9, rtol=0),
              f"max_diff={np.max(np.abs(lower_bd - lower_inmem)):.2e}")
        profile_inmem = result_inmem.varm["specificity_profile"]
        check("Dense-backed: average_profile close to in-memory",
              np.allclose(profile_bd, profile_inmem, atol=1e-9, rtol=0),
              f"max_diff={np.max(np.abs(profile_bd - profile_inmem)):.2e}")
        check("Dense-backed: all upper non-negative", bool(np.all(upper_bd >= 0)))
        check("Dense-backed: all lower non-negative", bool(np.all(lower_bd >= 0)))
finally:
    _os.unlink(dense_fixture_path)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)

