#!/usr/bin/env python3
"""Plan 07 — Final Cross-Language Parity Validation: Python end-to-end test.

Runs the full canonical ACTIONet pipeline on the parity fixture, then
compares every parity-critical output slot against:
  1. The Plan 00 Python baseline (intra-language regression)
  2. The R baseline (cross-language parity)

Usage:
    cd actionet-python
    .venv/bin/python3 tests/test_parity.py [--skip-r]

Exit code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

TESTS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(TESTS_DIR)
LIBACTIONET_TEST = os.path.join(REPO_ROOT, "..", "libactionet", "test")
FIXTURE    = os.path.join(TESTS_DIR, "fixtures", "parity_fixture.h5ad")
BASELINE_PY = os.path.join(TESTS_DIR, "fixtures", "baseline_python.npz")
BASELINE_R_H5AD = os.path.join(TESTS_DIR, "..", "..", "actionet-r",
                               "tests", "fixtures", "baseline_r.h5ad")
BASELINE_R_NPZ  = os.path.join(TESTS_DIR, "fixtures", "baseline_r_for_parity.npz")

# Add src to path for non-installed use
_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import actionet as an

SEED = 42

# ---------------------------------------------------------------------------
# Tolerances (from Plan 07)
# ---------------------------------------------------------------------------
ATOL_DENSE  = 1e-6
RTOL_DENSE  = 1e-4
ATOL_SIGMA  = 1e-8
RTOL_SIGMA  = 1e-6
ATOL_SPARSE = 1e-6
RTOL_SPARSE = 1e-4
ATOL_SPEC   = 1e-4
RTOL_SPEC   = 1e-3

# ---------------------------------------------------------------------------
# Test state
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0
_DETAILED: list[dict] = []


def check(name: str, cond: bool, detail: str = "", warn: bool = False) -> None:
    global PASS, FAIL
    status = "PASS" if cond else ("WARN" if warn else "FAIL")
    _DETAILED.append({"name": name, "status": status, "detail": detail})
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    elif warn:
        print(f"  WARN  {name}" + (f": {detail}" if detail else ""))
        # Warnings don't count as failures
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAIL += 1


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canon_svd_sign(mat: np.ndarray) -> np.ndarray:
    mat = np.array(mat, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        idx = np.argmax(np.abs(col))
        if col[idx] < 0:
            mat[:, j] *= -1
    return mat


def canon_archetype_order(*mats: np.ndarray, ref: int = 0) -> list[np.ndarray]:
    norms = np.linalg.norm(np.array(mats[ref], dtype=float), axis=0)
    order = np.argsort(-norms)
    return [np.array(m, dtype=float)[:, order] for m in mats]


def _to_csr(m):
    if sp.issparse(m):
        csr = m.tocsr()
    else:
        csr = sp.csr_matrix(np.asarray(m))
    csr.sort_indices()
    return csr


# ---------------------------------------------------------------------------
# Slot-level comparisons
# ---------------------------------------------------------------------------

def cmp_dense(slot: str, a: np.ndarray, b: np.ndarray,
              atol: float = ATOL_DENSE, rtol: float = RTOL_DENSE) -> None:
    if a.shape != b.shape:
        check(slot, False, f"shape mismatch: {a.shape} vs {b.shape}")
        return
    dev = float(np.max(np.abs(a - b)))
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    check(slot, ok, f"max_dev={dev:.3e}")


def cmp_sparse(slot: str, a, b,
               atol: float = ATOL_SPARSE, rtol: float = RTOL_SPARSE) -> None:
    ca = _to_csr(a)
    cb = _to_csr(b)
    if ca.shape != cb.shape:
        check(slot, False, f"shape mismatch: {ca.shape} vs {cb.shape}")
        return
    if ca.nnz != cb.nnz:
        dev = float(np.max(np.abs(ca.toarray() - cb.toarray())))
        ok = bool(np.allclose(ca.toarray(), cb.toarray(), atol=atol, rtol=rtol))
        check(slot, ok, f"nnz Python={ca.nnz} vs other={cb.nnz}, max_dev={dev:.3e}")
        return
    dev = float(np.max(np.abs(ca.data - cb.data))) if ca.nnz > 0 else 0.0
    ok = bool(np.allclose(ca.data, cb.data, atol=atol, rtol=rtol))
    check(slot, ok, f"max_dev={dev:.3e}")


def cmp_archetype_slots(prefix: str, py_arr, ref_arr) -> None:
    """Compare two archetype matrices after column-norm canonicalization."""
    if py_arr is None or ref_arr is None:
        missing = "Python" if py_arr is None else "reference"
        check(f"{prefix} present", False, f"MISSING in {missing}")
        return
    py_c = canon_archetype_order(py_arr)[0]
    ref_c = canon_archetype_order(ref_arr)[0]
    if py_c.shape != ref_c.shape:
        check(prefix, False,
              f"shape mismatch after canon: {py_c.shape} vs {ref_c.shape}",
              warn=True)
        return
    dev = float(np.max(np.abs(py_c - ref_c)))
    ok = bool(np.allclose(py_c, ref_c, atol=ATOL_DENSE, rtol=RTOL_DENSE))
    check(prefix, ok, f"max_dev={dev:.3e}")


# ---------------------------------------------------------------------------
# Pipeline (mirrors generate_baseline.py)
# ---------------------------------------------------------------------------

def run_pipeline(adata: ad.AnnData) -> ad.AnnData:
    an.reduce_kernel(adata, n_components=20, layer="logcounts", seed=SEED, verbose=False)
    an.run_action(adata, k_min=2, k_max=20, reduction_key="action")
    an.build_network(adata)
    labels = adata.obs["assigned_archetype"].astype(int).values
    an.compute_feature_specificity(adata, labels=labels, layer="logcounts", key_added="specificity")
    an.compute_archetype_feature_specificity(adata, archetype_key="H_merged",
                                              layer="logcounts", key_added="archetype")
    an.correct_batch_effect(adata, batch_key="batch", reduction_key="action",
                             corrected_suffix="corrected", layer="logcounts")
    an.layout_network(adata, seed=SEED, verbose=False, key_added="actionet_2d")
    return adata


# ---------------------------------------------------------------------------
# Shape verification checklist (Plan 07, §Shape Verification)
# ---------------------------------------------------------------------------

def verify_shapes(adata: ad.AnnData) -> None:
    section("Shape verification (cells x genes AnnData-native contract)")
    n_obs, n_var = adata.n_obs, adata.n_vars
    k = adata.obsm["action"].shape[1] if "action" in adata.obsm else None

    check(f"X shape ({n_obs}, {n_var})", adata.X.shape == (n_obs, n_var))
    if "action" in adata.obsm:
        check(f"obsm/action ({n_obs}, k)", adata.obsm["action"].shape[0] == n_obs)
    if "action_B" in adata.obsm:
        check(f"obsm/action_B ({n_obs}, p)", adata.obsm["action_B"].shape[0] == n_obs)
    if "action_U" in adata.varm:
        check(f"varm/action_U ({n_var}, k)", adata.varm["action_U"].shape[0] == n_var)
    if "action_A" in adata.varm:
        check(f"varm/action_A ({n_var}, p)", adata.varm["action_A"].shape[0] == n_var)
    if "H_stacked" in adata.obsm:
        check(f"obsm/H_stacked ({n_obs}, *)", adata.obsm["H_stacked"].shape[0] == n_obs)
    if "H_merged" in adata.obsm:
        check(f"obsm/H_merged ({n_obs}, *)", adata.obsm["H_merged"].shape[0] == n_obs)
    if "C_stacked" in adata.obsm:
        check(f"obsm/C_stacked ({n_obs}, *)", adata.obsm["C_stacked"].shape[0] == n_obs)
    if "C_merged" in adata.obsm:
        check(f"obsm/C_merged ({n_obs}, *)", adata.obsm["C_merged"].shape[0] == n_obs)
    if "actionet" in adata.obsp:
        G = adata.obsp["actionet"]
        check(f"obsp/actionet ({n_obs}, {n_obs})", G.shape == (n_obs, n_obs))
    for slot in ["specificity_upper", "specificity_lower", "specificity_profile"]:
        if slot in adata.varm:
            check(f"varm/{slot} ({n_var}, *)", adata.varm[slot].shape[0] == n_var)
    for slot in ["archetype_feat_profile", "archetype_feat_specificity_upper",
                 "archetype_feat_specificity_lower"]:
        if slot in adata.varm:
            check(f"varm/{slot} ({n_var}, *)", adata.varm[slot].shape[0] == n_var)


# ---------------------------------------------------------------------------
# Intra-language regression (new vs Plan 00 baseline)
# ---------------------------------------------------------------------------

def check_intra_regression(adata: ad.AnnData, baseline: np.lib.npyio.NpzFile) -> None:
    section("Intra-language regression (new vs Plan 00 Python baseline)")

    def g(key: str):
        return baseline[key] if key in baseline.files else None

    def _recon_sparse(prefix: str):
        keys = [f"{prefix}_data", f"{prefix}_indices",
                f"{prefix}_indptr", f"{prefix}_shape"]
        if all(k in baseline.files for k in keys):
            shape = tuple(int(x) for x in baseline[f"{prefix}_shape"])
            return sp.csr_matrix(
                (baseline[f"{prefix}_data"],
                 baseline[f"{prefix}_indices"],
                 baseline[f"{prefix}_indptr"]),
                shape=shape,
            )
        return None

    # Reduction — SVD sign canon
    for key_obsm, bk, slot in [
        ("action",   "obsm_action",   "action regression"),
        ("action_B", "obsm_action_B", "action_B regression"),
    ]:
        if key_obsm in adata.obsm and g(bk) is not None:
            cmp_dense(slot,
                      canon_svd_sign(adata.obsm[key_obsm]),
                      canon_svd_sign(g(bk)))

    for key_varm, bk, slot in [
        ("action_U", "varm_action_U", "action_U regression"),
        ("action_A", "varm_action_A", "action_A regression"),
    ]:
        if key_varm in adata.varm and g(bk) is not None:
            cmp_dense(slot,
                      canon_svd_sign(adata.varm[key_varm]),
                      canon_svd_sign(g(bk)))

    if "action_params" in adata.uns and g("uns_action_sigma") is not None:
        a = np.asarray(adata.uns["action_params"]["sigma"]).ravel()
        b = np.asarray(g("uns_action_sigma")).ravel()
        cmp_dense("sigma regression", a, b, atol=ATOL_SIGMA, rtol=RTOL_SIGMA)

    # Network
    if "actionet" in adata.obsp:
        G_ref = _recon_sparse("obsp_actionet")
        if G_ref is not None:
            cmp_sparse("network regression", adata.obsp["actionet"], G_ref)

    # Specificity — use looser tolerance
    for key_varm, bk, slot in [
        ("specificity_upper", "varm_specificity_upper", "specificity_upper regression"),
        ("specificity_lower", "varm_specificity_lower", "specificity_lower regression"),
    ]:
        if key_varm in adata.varm and g(bk) is not None:
            cmp_dense(slot,
                      np.asarray(adata.varm[key_varm]),
                      np.asarray(g(bk)),
                      atol=ATOL_SPEC, rtol=RTOL_SPEC)

    # Batch correction
    for key_obsm, bk, slot in [
        ("action_corrected",   "obsm_action_corrected",   "action_corrected regression"),
    ]:
        if key_obsm in adata.obsm and g(bk) is not None:
            cmp_dense(slot,
                      canon_svd_sign(adata.obsm[key_obsm]),
                      canon_svd_sign(g(bk)))


# ---------------------------------------------------------------------------
# Cross-language parity (Python new vs R baseline)
# ---------------------------------------------------------------------------

def check_cross_language(py_adata: ad.AnnData, r_h5ad_path: str) -> None:
    section("Cross-language parity (Python vs R)")

    if not os.path.exists(r_h5ad_path):
        check("R baseline exists", False, f"not found: {r_h5ad_path}")
        return
    check("R baseline exists", True)

    r = ad.read_h5ad(r_h5ad_path)
    print(f"  R baseline: {r.n_obs} obs × {r.n_vars} vars")

    # Reduction — exact match expected (same C++ core, same seed)
    for py_key, r_key, slot in [
        ("action",   "action",   "cross-lang action"),
        ("action_B", "action_B", "cross-lang action_B"),
    ]:
        if py_key in py_adata.obsm and r_key in r.obsm:
            py_m = py_adata.obsm[py_key]
            r_m  = r.obsm[r_key]
            pa = canon_svd_sign(py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m))
            ra = canon_svd_sign(r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m))
            cmp_dense(slot, pa, ra)
        else:
            missing = "Python" if py_key not in py_adata.obsm else "R"
            check(slot, False, f"MISSING in {missing}")

    for py_key, r_key, slot in [
        ("action_U", "action_U", "cross-lang action_U"),
        ("action_A", "action_A", "cross-lang action_A"),
    ]:
        if py_key in py_adata.varm and r_key in r.varm:
            py_m = py_adata.varm[py_key]
            r_m  = r.varm[r_key]
            pa = canon_svd_sign(py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m))
            ra = canon_svd_sign(r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m))
            cmp_dense(slot, pa, ra)
        else:
            missing = "Python" if py_key not in py_adata.varm else "R"
            check(slot, False, f"MISSING in {missing}")

    if "action_params" in py_adata.uns and "action_params" in r.uns:
        py_s = np.asarray(py_adata.uns["action_params"]["sigma"]).ravel()
        r_s  = np.asarray(r.uns["action_params"]["sigma"]).ravel()
        cmp_dense("cross-lang sigma", py_s, r_s, atol=ATOL_SIGMA, rtol=RTOL_SIGMA)

    # ACTION archetypes — exact parity is expected across front-ends.
    for py_key, r_key, slot in [
        ("H_stacked", "H_stacked", "cross-lang H_stacked"),
        ("H_merged",  "H_merged",  "cross-lang H_merged"),
        ("C_stacked", "C_stacked", "cross-lang C_stacked"),
        ("C_merged",  "C_merged",  "cross-lang C_merged"),
    ]:
        py_m = py_adata.obsm.get(py_key)
        r_m  = r.obsm.get(r_key)
        if py_m is None or r_m is None:
            missing = "Python" if py_m is None else "R"
            check(slot, False, f"MISSING in {missing}")
            continue
        # Handle sparse matrices from R h5ad
        py_a = py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m)
        r_a  = r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m)
        if py_a.ndim < 2 or r_a.ndim < 2:
            check(slot, False, f"unexpected ndim: py={py_a.ndim} r={r_a.ndim}")
            continue
        if py_a.shape[1] != r_a.shape[1]:
            check(slot, False,
                  f"archetype count differs: Python={py_a.shape[1]} "
                  f"R={r_a.shape[1]}")
        else:
            cmp_archetype_slots(slot, py_a, r_a)

    # Network — exact parity is expected once archetypes align.
    if "actionet" in py_adata.obsp and "actionet" in r.obsp:
        py_G = _to_csr(py_adata.obsp["actionet"])
        r_G_raw = r.obsp["actionet"]
        r_G = _to_csr(r_G_raw.toarray() if sp.issparse(r_G_raw) else r_G_raw)
        if py_G.nnz != r_G.nnz:
            check("cross-lang network", False,
                  f"nnz differs: Python={py_G.nnz} R={r_G.nnz}")
        else:
            cmp_sparse("cross-lang network", py_G, r_G)

    # Specificity — exact parity is expected once archetypes align.
    for py_key, r_key, slot in [
        ("specificity_upper", "cluster_upper", "cross-lang specificity_upper"),
        ("specificity_lower", "cluster_lower", "cross-lang specificity_lower"),
    ]:
        py_m = py_adata.varm.get(py_key)
        r_m  = r.varm.get(r_key)
        if py_m is None or r_m is None:
            missing = "Python" if py_m is None else "R"
            check(slot, False, f"MISSING in {missing}")
            continue
        py_a = py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m)
        r_a  = r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m)
        if py_a.shape != r_a.shape:
            check(slot, False, f"shape mismatch: {py_a.shape} vs {r_a.shape}")
        else:
            cmp_dense(slot, py_a, r_a, atol=ATOL_SPEC, rtol=RTOL_SPEC)

    # Archetype feature specificity
    for py_key, r_key, slot in [
        ("archetype_feat_profile",            "archetype_feat_profile",            "cross-lang arch_feat_profile"),
        ("archetype_feat_specificity_upper",  "archetype_feat_specificity_upper",  "cross-lang arch_feat_upper"),
        ("archetype_feat_specificity_lower",  "archetype_feat_specificity_lower",  "cross-lang arch_feat_lower"),
    ]:
        py_m = py_adata.varm.get(py_key)
        r_m  = r.varm.get(r_key)
        if py_m is None or r_m is None:
            missing = "Python" if py_m is None else "R"
            check(slot, False, f"MISSING in {missing}")
            continue
        py_a = py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m)
        r_a  = r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m)
        if py_a.shape != r_a.shape:
            check(slot, False, f"shape mismatch: {py_a.shape} vs {r_a.shape}")
        else:
            cmp_dense(slot, py_a, r_a, atol=ATOL_SPEC, rtol=RTOL_SPEC)

    # Batch correction
    for py_key, r_key, slot in [
        ("action_corrected",   "action_orth",   "cross-lang action_corrected"),
    ]:
        py_m = py_adata.obsm.get(py_key)
        r_m  = r.obsm.get(r_key)
        if py_m is None or r_m is None:
            missing = "Python" if py_m is None else "R"
            check(slot, False, f"MISSING in {missing}")
        else:
            py_a = py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m)
            r_a  = r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m)
            cmp_dense(slot, canon_svd_sign(py_a), canon_svd_sign(r_a))

    for py_key, r_key, slot in [
        ("action_corrected_U", "action_U_orth", "cross-lang action_corrected_U"),
        ("action_corrected_A", "action_A_orth", "cross-lang action_corrected_A"),
    ]:
        py_m = py_adata.varm.get(py_key)
        r_m  = r.varm.get(r_key)
        if py_m is None or r_m is None:
            missing = "Python" if py_m is None else "R"
            check(slot, False, f"MISSING in {missing}")
        else:
            py_a = py_m.toarray() if sp.issparse(py_m) else np.asarray(py_m)
            r_a  = r_m.toarray()  if sp.issparse(r_m)  else np.asarray(r_m)
            cmp_dense(slot, canon_svd_sign(py_a), canon_svd_sign(r_a))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    global PASS, FAIL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-r", action="store_true",
                        help="Skip cross-language R comparison.")
    args = parser.parse_args(argv)

    # -- Load fixture and run pipeline
    section("Loading fixture and running pipeline")
    if not os.path.exists(FIXTURE):
        print(f"ERROR: fixture not found at {FIXTURE}")
        sys.exit(1)
    adata = ad.read_h5ad(FIXTURE)
    print(f"  Loaded: {adata.n_obs} obs × {adata.n_vars} vars")
    adata = run_pipeline(adata)
    print("  Pipeline complete.")

    # -- Shape verification
    verify_shapes(adata)

    # -- Intra-language regression
    if os.path.exists(BASELINE_PY):
        baseline = np.load(BASELINE_PY, allow_pickle=False)
        check_intra_regression(adata, baseline)
    else:
        section("Intra-language regression")
        check("Python baseline exists", False, f"not found: {BASELINE_PY}")

    # -- Cross-language parity
    if not args.skip_r:
        check_cross_language(adata, BASELINE_R_H5AD)
    else:
        section("Cross-language parity")
        print("  SKIPPED (--skip-r)")

    # -- Final summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    # Report any warnings
    warns = [r for r in _DETAILED if r["status"] == "WARN"]
    if warns:
        print("\nKnown differences (documented, not counted as failures):")
        for w in warns:
            print(f"  WARN  {w['name']}: {w['detail']}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
