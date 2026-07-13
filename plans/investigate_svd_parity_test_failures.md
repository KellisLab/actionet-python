# Handoff: investigate `test_irlb_svd_parity.py` failures

Status: Draft handoff, 2026-07-13. Author: agent that completed the PRIMME/Feng
removal (`plans/remove_primme_and_feng_svd_...`). All three failures were
observed on the post-removal build; the pre-removal build could not even be
imported (`undefined symbol: ctrsm` from PRIMME's missing runtime), so we cannot
directly compare against the baseline. All three failures are independent of
the PRIMME/Feng deletion — the PRIMME/Feng cleanup only made the module
loadable, which is why these failures are visible now.

Do **not** loosen tolerances or delete tests as a first response. Each of the
three failures is a distinct signal worth understanding before any patch.

---

## Environment for repro

```bash
source /afs/csail.mit.edu/u/s/spineda/miniforge3/etc/profile.d/conda.sh
conda activate actionet-dev
cd /afs/csail.mit.edu/u/s/spineda/data/git_projects/actionet-python

# rebuild after any C++ edit
cmake --build build/cp313-cp313-linux_x86_64 -j
cp build/cp313-cp313-linux_x86_64/_core.cpython-313-x86_64-linux-gnu.so \
   /afs/csail.mit.edu/u/s/spineda/miniforge3/envs/actionet-dev/lib/python3.13/site-packages/actionet/

pytest tests/test_irlb_svd_parity.py -v --tb=short
```

Post-removal result on `dev-gpu` after the PRIMME/Feng cleanup: **27 pass, 3
fail** (see below).

---

## Failure 1 (highest signal): `test_inmemory_sparse_parity_irlb_vs_halko`

### Symptom

```
Mismatched elements: 1 / 15 (6.67%)
Mismatch at index:
 [14]: 458.8043236050096 (ACTUAL, IRLB), 435.0270216685793 (DESIRED, Halko)
Max relative difference among violations: 0.05465707
```

`SIGMA_RTOL = 0.05` in the test. IRLB and Halko differ by 5.47% at index 14
(smallest of 15 requested singular values).

### What is actually happening

This is **not** a randomized-algorithm tolerance flake. Ground-truth comparison
against `scipy.sparse.linalg.svds` on the same 300x150, density=0.1,
random_state=123 matrix (nnz=4500):

```
idx  scipy_gt      IRLB          Halko         IRLB_rel   Halko_rel
  0   1152.273984   1152.273984   1152.273983   0.0000%    0.0000%
  1    528.797413    528.797413    525.426679   0.0000%    0.6374%
  2    509.888643    509.888643    507.417016   0.0000%    0.4847%
  ...
 13    461.940183    461.940183    444.617255   0.0000%    3.7500%
 14    458.804324    458.804324    435.027022   0.0000%    5.1824%
```

Findings:

- IRLB agrees with the ARPACK ground truth to 0.0000% at every index. IRLB is
  the reference; the parity test is really asking "does Halko match ARPACK?".
- Halko's error grows monotonically with index, reaching 5.18% at the tail.
  This is textbook Halko-under-convergence on a matrix with a small spectral
  gap: singular values 1..14 are all in the 435..528 band, so the randomized
  power iteration cannot cleanly separate them at the default `max_it=5`.
- The matrix is not pathological — it is a well-defined sparse random matrix
  with a known singular value distribution, and IRLB nails it. Halko simply
  hasn't converged.

### Why now

The Halko default `max_it` for `runSVD` (in-memory) is `5`
(`svd_main.cpp::default_max_it` → `case ALG_HALKO: return 5;`). That number has
been in place since the SVD switch was consolidated. The parity test was likely
authored with a matrix whose spectral gap was wide enough that 5 iterations
sufficed; the current test matrix (density=0.1, random_state=123) is tighter.

Nothing in the PRIMME/Feng deletion touched Halko's iteration count, matvec
loop, or `svdHalko` internals.

### Actionable next steps (pick one, do not do multiple simultaneously)

1. **Preferred: raise Halko's default `max_it` for in-memory SVD.** The backed
   default is `max_it=5` because I/O passes dominate cost, but in-memory Halko
   has no I/O and 5 iterations is arbitrary. Try `max_it=10` and re-run the
   test; also re-run `docs/svd_algorithm_benchmark.md` reproduction on the
   in-memory harness to confirm wall-time regression is small. Edit is one
   line in
   `src/libactionet/src/decomposition/svd_main.cpp::default_max_it`. If wall
   time inflates unacceptably, try `max_it=7`.

2. **Alternative: pass `max_it` explicitly from the parity test.** Add
   `max_it=15` (or similar) to both `run_svd` calls in
   `test_inmemory_sparse_parity_irlb_vs_halko` and document that this test
   requires a tight-spectrum-friendly Halko iteration budget. This is the
   surgical fix, but it papers over a real user-facing default-quality issue.

3. **Do not** just bump `SIGMA_RTOL` to 0.06. The 5% cross-algorithm tolerance
   is a defensible contract; loosening it hides regression signal.

Verification for option 1: after the change, all three of these must hold:
- `pytest tests/test_irlb_svd_parity.py::test_inmemory_sparse_parity_irlb_vs_halko -v` passes.
- `pytest tests/test_irlb_svd_parity.py -v` shows the same 27/30 → 28/30 or better.
- Median wall time from `benchmark_svd_inmemory_defaults.py --tiers 25k --trials 2`
  for Halko sparse increases by less than the ~5% band already recorded in
  `docs/svd_algorithm_benchmark.md`.

### Cross-check

Also run this manual repro to confirm the fix reaches ARPACK ground truth:

```python
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, actionet as an
X = sp.random(300, 150, density=0.1, random_state=123, format='csr'); X.data *= 100
u, d_gt, vt = spla.svds(X.astype(float), k=15); d_gt = np.sort(d_gt)[::-1]
r = an.run_svd(X, n_components=15, algorithm='halko', seed=42, verbose=False, max_it=<N>)
d = np.asarray(r['d']).ravel()
print(np.max(np.abs(d - d_gt) / d_gt))  # want << 0.05
```

---

## Failure 2 & 3: `test_core_backed_operator_rejects_retired_algorithm_ids[2|3]`

### Symptom

```
RuntimeError: createBackedOperator: failed to open h5ad file:
    /tmp/pytest-of-spineda/pytest-XX/test_core_backed_operator_reje1/core_retired_3.h5ad
```

The test fails at line 620:

```python
op = _core.create_backed_operator(str(h5ad_path), "/X", 16)
```

before it ever reaches the `pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id")`
block it is trying to exercise.

### Root cause

Test-hygiene bug in `test_core_backed_operator_rejects_retired_algorithm_ids`
(lines 610-628 of `tests/test_irlb_svd_parity.py`):

1. `_create_backed_anndata()` calls `ad.read_h5ad(h5ad_path, backed="r+")`,
   which opens an HDF5 handle in **read-write** mode and keeps it open.
2. The test then calls `_core.create_backed_operator(str(h5ad_path), ...)` —
   the C++ side tries to open the **same file** for reading, but the anndata
   `r+` handle plus default HDF5 file locking on network filesystems (AFS/
   NFS) blocks the second open.
3. The test never sets `HDF5_USE_FILE_LOCKING=FALSE`, unlike every other
   backed test in the file (`test_backed_sparse_parity_irlb_vs_halko` at
   line 311, `test_backed_dense_parity_irlb_vs_halko` at line 343, etc.).
4. The `finally` block only closes the anndata handle *after* the C++ call,
   so the C++ call never gets a chance.

Confirmed with a minimal repro on this same workstation:

```python
os.environ.pop('HDF5_USE_FILE_LOCKING', None)  # default behavior
adata_backed = ad.read_h5ad(p, backed='r+')    # takes r+ handle
op = _core.create_backed_operator(str(p), '/X', 16)  # RuntimeError
```

Same test with `HDF5_USE_FILE_LOCKING=FALSE` set: works.
Same test with the anndata `r+` handle closed first: works.

Nothing in the PRIMME/Feng deletion changed backed I/O; this failure is a
pre-existing test-hygiene issue that only became visible because the module
now actually loads.

### Actionable fix (small, safe)

Edit `tests/test_irlb_svd_parity.py`, lines 610-628. Add the
`HDF5_USE_FILE_LOCKING` env var at the top of the test **and** close the
anndata handle before the `_core.create_backed_operator` call:

```python
@pytest.mark.parametrize("algorithm_id", [2, 3])
def test_core_backed_operator_rejects_retired_algorithm_ids(tmp_path, algorithm_id):
    """Backed `_core` SVD entry points expose only IRLB/Halko to Python."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    X_sparse = _create_test_matrix(
        n_obs=32, n_vars=24, density=0.3, as_sparse=True, random_state=4
    ).tocsr()
    adata_backed, h5ad_path = _create_backed_anndata(
        X_sparse, tmp_path, prefix=f"core_retired_{algorithm_id}"
    )
    # Close the r+ handle before opening the file from C++ to avoid HDF5
    # lock contention on network filesystems.
    adata_backed.file.close()

    try:
        op = _core.create_backed_operator(str(h5ad_path), "/X", 16)

        with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
            _core.run_svd_backed_operator(op, 4, 0, 0, algorithm_id, False)

        with pytest.raises(RuntimeError, match=r"unsupported SVD algorithm id"):
            _core.reduce_kernel_backed_operator(op, 4, algorithm_id, 0, 0, False)
    finally:
        # adata_backed handle already closed above; nothing else to release.
        pass
```

Notes:
- The test name says "retired" — that wording is historical (from the pre-
  deletion quarantine phase). The behavior it exercises is still valid:
  private `_core` entry points reject algorithm IDs outside `{0, 1}`. Keep
  the test; just fix the plumbing.
- The two other private-`_core` rejection tests
  (`test_core_run_svd_rejects_retired_algorithm_ids` line 579,
  `test_core_reduce_kernel_rejects_retired_algorithm_ids` line 594)
  do not touch backed operators, so they already pass.

### Verification

```bash
pytest tests/test_irlb_svd_parity.py::test_core_backed_operator_rejects_retired_algorithm_ids -v
```

Both parametrizations should pass. Then run the full file; expect 29/30 (with
Failure 1 still open) or 30/30 (once Failure 1 is fixed as well).

---

## Where the ground-truth ARPACK comparison lives

Not currently in-tree. Only `scipy.sparse.linalg.svds` from stdlib scipy is
used ad-hoc during debugging. Consider adding an ARPACK ground-truth reference
to `_compare_svd_results` when investigating Failure 1 — the current pairwise
IRLB-vs-Halko comparison hides which algorithm is wrong.

---

## Do not chase these red herrings

- **The PRIMME/Feng deletion.** Neither algorithm was touched in IRLB or Halko
  paths. `svd_main.cpp` still contains identical calls to `svdIRLB` and
  `svdHalko` as before. The only related change was removing the `ALG_FENG`
  and `ALG_PRIMME` switch arms from `runSVD`/`runSVD_Operator`, which cannot
  affect the numerics of the remaining two algorithms.
- **Compiler warnings.** The post-removal build produces zero PRIMME-related
  warnings. The old `USE_DOUBLE` redefinition warning from
  `src/extern/primme/include/template_types.h` is gone.
- **The `_core.so` copy step.** The `pip`-installed copy in
  `.../site-packages/actionet/` is what Python actually loads.
  `src/actionet/_core.cpython-...so` is unused by imports; do not be fooled
  by editing it and observing no change. Always copy to
  `.../miniforge3/envs/actionet-dev/lib/python3.13/site-packages/actionet/`
  after rebuilding.
- **The `test_inmemory_sparse_parity` seed.** Changing `random_state=123` to
  another value may "fix" the failure by picking a matrix Halko happens to
  converge on. This is not a fix — it's evidence-hiding.

---

## Reference: files that matter

- `src/libactionet/src/decomposition/svd_main.cpp` — `default_max_it`, `runSVD`, `runSVD_Operator`.
- `src/libactionet/src/decomposition/svd_halko.cpp` — Halko subspace iteration.
- `src/libactionet/src/decomposition/svd_irbla.cpp` — IRLB Lanczos loop.
- `tests/test_irlb_svd_parity.py` lines 34-37 (tolerances), 207-223 (Failure 1), 609-628 (Failures 2/3).
- `docs/svd_algorithm_benchmark.md` — Halko-vs-IRLB accuracy/wall-time trade-off record.
- `context/DECISIONS.md` "SVD algorithm strategy" — post-removal state.
