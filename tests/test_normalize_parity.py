"""Parity test: normalize_anndata (in-memory vs backed vs scanpy).

Generates a 10k x 20k sparse matrix, runs normalization through all three
code paths, and asserts numerical identity. Times each variant.
"""

import os
import tempfile
import time

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

ad.settings.allow_write_nullable_strings = True

from actionet.preprocessing import normalize_anndata

# ── Parameters ──────────────────────────────────────────────────────────────
N_OBS = 10_000
N_VARS = 20_000
DENSITY = 0.05
TARGET_SUM = 1e4
LOG_BASE = 2.0
SEED = 42
ATOL = 1e-12
RTOL = 1e-12
CHUNK_SIZE = 4096


def _make_adata(seed=SEED):
    """Create a reproducible sparse count matrix."""
    rng = np.random.default_rng(seed)
    X = sp.random(N_OBS, N_VARS, density=DENSITY, format="csr",
                  dtype=np.float64, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(N_OBS)]
    adata.var_names = [f"gene_{i}" for i in range(N_VARS)]
    return adata


def _scanpy_reference(adata):
    """Canonical scanpy normalize_total + log1p (base-2)."""
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata, base=LOG_BASE)


def _assert_matrices_equal(label_a, a, label_b, b):
    """Check two sparse/dense matrices are numerically identical."""
    if sp.issparse(a):
        a = a.toarray()
    if sp.issparse(b):
        b = b.toarray()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=ATOL, rtol=RTOL):
        diff = np.abs(a - b)
        max_diff = diff.max()
        mean_diff = diff.mean()
        n_mismatch = (diff > ATOL).sum()
        raise AssertionError(
            f"Mismatch between {label_a} and {label_b}: "
            f"max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}, "
            f"n_mismatch={n_mismatch}/{a.size}"
        )
    print(f"  PASS  {label_a} == {label_b}")


def _time(label, fn, *args, **kwargs):
    """Run fn, print elapsed time, return result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {label:.<50s} {elapsed:>7.3f}s")
    return result


def main():
    print(f"\n{'='*65}")
    print(f"  normalize_anndata parity test")
    print(f"  {N_OBS:,} obs × {N_VARS:,} vars, density={DENSITY}")
    print(f"  target_sum={TARGET_SUM}, log_base={LOG_BASE}")
    print(f"{'='*65}\n")

    # ── 1. Build data ───────────────────────────────────────────────────
    print("[1/7] Generating test data ...")
    adata_base = _time("  generate", _make_adata)
    print(f"       nnz = {adata_base.X.nnz:,}\n")

    # ── 2. Scanpy reference (inplace) ──────────────────────────────────
    print("[2/7] scanpy  (inplace) ...")
    adata_sc = adata_base.copy()
    _time("scanpy normalize_total + log1p", _scanpy_reference, adata_sc)
    X_scanpy = adata_sc.X
    print()

    # ── 3. In-memory inplace ───────────────────────────────────────────
    print("[3/7] actionet in-memory (inplace) ...")
    adata_mem_ip = adata_base.copy()
    _time("normalize_anndata inplace",
          normalize_anndata, adata_mem_ip,
          target_sum=TARGET_SUM, log_base=LOG_BASE, inplace=True)
    _assert_matrices_equal("scanpy", X_scanpy, "in-memory inplace", adata_mem_ip.X)
    print()

    # ── 4. In-memory copy ──────────────────────────────────────────────
    print("[4/7] actionet in-memory (copy) ...")
    adata_mem_src = adata_base.copy()
    adata_mem_cp = _time("normalize_anndata copy",
                         normalize_anndata, adata_mem_src,
                         target_sum=TARGET_SUM, log_base=LOG_BASE, inplace=False)
    _assert_matrices_equal("scanpy", X_scanpy, "in-memory copy", adata_mem_cp.X)
    # source must be unmodified
    _assert_matrices_equal("original", adata_base.X, "in-memory copy source", adata_mem_src.X)
    print()

    # ── 5. Backed inplace ─────────────────────────────────────────────
    print("[5/7] actionet backed (inplace) ...")
    tmp_ip = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    tmp_ip.close()
    try:
        adata_base.write_h5ad(tmp_ip.name)
        adata_bk_ip = ad.read_h5ad(tmp_ip.name, backed="r+")
        _time("normalize_anndata backed inplace",
              normalize_anndata, adata_bk_ip,
              target_sum=TARGET_SUM, log_base=LOG_BASE,
              backed_chunk_size=CHUNK_SIZE, inplace=True)
        # Read back fully to compare
        adata_bk_ip_mem = ad.read_h5ad(tmp_ip.name)
        _assert_matrices_equal("scanpy", X_scanpy, "backed inplace", adata_bk_ip_mem.X)
    finally:
        try:
            if hasattr(adata_bk_ip, 'file') and adata_bk_ip.file is not None:
                adata_bk_ip.file.close()
        except Exception:
            pass
        os.unlink(tmp_ip.name)
    print()

    # ── 6. Backed copy ────────────────────────────────────────────────
    print("[6/7] actionet backed (copy) ...")
    tmp_cp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    tmp_cp.close()
    try:
        adata_base.write_h5ad(tmp_cp.name)
        adata_bk_src = ad.read_h5ad(tmp_cp.name, backed="r+")
        adata_bk_cp = _time("normalize_anndata backed copy",
                            normalize_anndata, adata_bk_src,
                            target_sum=TARGET_SUM, log_base=LOG_BASE,
                            backed_chunk_size=CHUNK_SIZE, inplace=False)
        _assert_matrices_equal("scanpy", X_scanpy, "backed copy", adata_bk_cp.X)
    finally:
        try:
            if hasattr(adata_bk_src, 'file') and adata_bk_src.file is not None:
                adata_bk_src.file.close()
        except Exception:
            pass
        os.unlink(tmp_cp.name)
    print()

    # ── 7. Cross-check in-memory inplace vs copy ─────────────────────
    print("[7/7] Cross-consistency ...")
    _assert_matrices_equal("in-memory inplace", adata_mem_ip.X,
                           "in-memory copy", adata_mem_cp.X)
    print()

    print(f"{'='*65}")
    print("  ALL PARITY CHECKS PASSED")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
