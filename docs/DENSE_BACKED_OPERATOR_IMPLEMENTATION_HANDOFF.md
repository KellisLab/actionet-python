# Dense-Backed Operator Consolidation: Implementation Handoff

## Metadata

- Prepared: 2026-03-20 19:31:28 EDT
- Primary repo: `actionet-python`
- Core repo: `libactionet`
- Predecessor assessment: `docs/DENSE_BACKED_OPERATOR_ARCHITECTURE_HANDOFF.md`
- Synced working branches:
  - `actionet-python`: `codex/dense-backed-operator-v1` @ `9788c499b7d2e12884b926220847be18f435fe96`
  - `actionet-python/src/libactionet`: `codex/dense-backed-operator-v1` @ `cfe43dd3596076bdb55d0542b6cfad9868582e55`
  - sibling `libactionet`: `codex/dense-backed-operator-v1` @ `cfe43dd3596076bdb55d0542b6cfad9868582e55`

## Executive Summary

The earlier assessment is directionally correct, but two findings are sharper after code review and live repro:

1. Dense-backed raw expression is currently a hard operator-path hole, not a slower alternate path.
   - `run_svd()` and `reduce_kernel()` treat any backed matrix as operator-capable.
   - `_core.create_backed_operator(...)` still instantiates only `BackedSparseMatrixOperator`.
   - For dense-backed `.X` or `layers[...]`, both functions fail immediately because the C++ path opens the target as a sparse HDF5 group.

2. Backed batch correction is duplicated above the core.
   - Dense-backed `correct_batch_effect()` and `correct_basal_expression()` do work today, but only through Python-streamed `MatrixSource.xt_dot()` / `x_dot()` plus Python-side `_deflate_terms()` and `_perturbed_with_prior()`.
   - The backed path therefore duplicates perturbation assembly and keeps reduction bookkeeping outside `libactionet`.

The best remedy remains:

- add a real `BackedDenseMatrixOperator` in `libactionet`
- replace the sparse-only backed factory with a generic `MatrixOperator` factory
- add operator-based orthogonalization APIs in `libactionet`
- move backed dense and backed sparse decomposition / batch correction onto one core path
- leave specificity, marker detection, preprocessing, and backed `obsm` work on the row-reader/scan abstraction

That gives the cleanest architecture while respecting the actual algorithm split in the codebase.

## Supplemental Findings

### 1. Dense-backed `run_svd()` and `reduce_kernel()` currently fail

Current code path:

- `src/actionet/core.py` uses the operator branch whenever `source.is_backed` or `_is_backed_matrix(...)` is true.
- `src/actionet/wp_io.cpp` exposes `_core.create_backed_operator(...)`.
- That binding currently returns only `std::shared_ptr<actionet::BackedSparseMatrixOperator>`.

Verified behavior:

- Dense-backed `.X` opens as `h5py.Dataset`.
- A live repro on a small dense-backed `.h5ad` produced:
  - `run_svd_err RuntimeError Failed to open sparse matrix group path`
  - `reduce_kernel_err RuntimeError Failed to open sparse matrix group path`

This is the single most important architectural fact for the implementation plan: dense-backed operator support is not an optimization pass over a working feature; it is a missing capability that currently breaks operator-shaped dense-backed workflows.

### 2. Dense-backed batch correction works, but the hot path is still Python-streamed

Current backed batch correction flow:

- `_streamed_batch_terms()` computes `X.T @ design` and `X @ Z` through Python chunk iteration.
- `_streamed_basal_terms()` computes `X @ Z` the same way.
- `_deflate_terms()` and `_perturbed_with_prior()` duplicate deflation and `perturbedSVD` assembly in Python.

That means the current backed implementation:

- keeps the same one-pass / two-pass matrix scan count as the desired core implementation
- pays Python loop / h5py slicing / NumPy temporary overhead on every chunk
- retains duplicate perturbation plumbing outside `libactionet`

Dense-backed batch correction is therefore a real consolidation target, not just a future nice-to-have.

### 3. Specificity bindings are intentionally sparse-backed only

The backed specificity ABI in `src/actionet/wp_annotation.cpp` is typed to `BackedSparseMatrixOperator`, not `MatrixOperator`.

That is correct for now. The backed specificity algorithm needs stored-support semantics and scan statistics that do not naturally collapse into `matvec` / `matmat`. Dense-backed specificity should remain on the existing streamed fallback in this pass.

### 4. A dense operator needs an internal byte cap, not a naive row count

The current public `backed_chunk_size` default is `4096`. For dense slabs that translates into:

- `4096 x 20,000 x 8 bytes = 655,360,000 bytes` = `0.61 GiB`
- `4096 x 28,692 x 8 bytes = 940,179,456 bytes` = `0.88 GiB`

That is too large for a default dense-backed slab. The dense operator should treat `backed_chunk_size` as an upper bound, then clamp internally to a byte budget. Recommended default internal cap: `256 MiB`.

At that cap:

- `20,000` genes -> `1677` rows per slab
- `28,692` genes -> `1169` rows per slab

This keeps the dense operator memory-bounded without exposing a new public API.

## Performance And Memory Estimates

### Assumptions

These are analytical estimates, not measured benchmarks.

- dtype: `float64`
- backed SVD algorithm: default backed Halko (`iters = 5`)
- dense operator internal slab cap: `256 MiB`
- batch correction representative orthogonalized rank: `q = 8`
- basal correction representative orthogonalized rank: `q = 1`
- storage bandwidth envelope: `2-4 GiB/s` sequential effective read rate
- tall single-cell regime: `n_obs >> n_vars`, so operator Halko stores its large basis on the cell axis

Representative scenarios:

1. Moderate dense layer: `250,000 cells x 20,000 genes`
2. Atlas dense layer: `1,792,201 cells x 28,692 genes`

### Scan Counts By Path

Assuming backed Halko with `iters = 5`:

- dense-backed `run_svd()`: `12` full matrix passes
  - initial projection: `1`
  - power iterations: `2 * 5 = 10`
  - final projection: `1`
- dense-backed `reduce_kernel()`: `14` full matrix passes
  - backed Halko SVD: `12`
  - perturbation `matvec` + `rmatvec`: `2`
- dense-backed `correct_batch_effect()`: `2` full matrix passes
  - `X.T @ design`
  - `X @ Z`
- dense-backed `correct_basal_expression()`: `1` full matrix pass
  - `X @ Z`

### Memory Envelope Estimates

| Path | Current state | Moderate dense | Atlas dense | Improvement |
|---|---|---:|---:|---:|
| `run_svd()` | unsupported; practical workaround is full dense materialization | `37.25 GiB -> 0.45 GiB` | `383.12 GiB -> 1.62 GiB` | `83x` to `236x` lower peak memory |
| `reduce_kernel()` | unsupported; practical workaround is full dense materialization | `37.25 GiB -> 0.54 GiB` | `383.12 GiB -> 2.26 GiB` | `69x` to `169x` lower peak memory |
| `correct_batch_effect()` | Python-streamed backed path | `0.84 GiB -> 0.37 GiB` | `2.48 GiB -> 1.06 GiB` | about `2.3x` lower peak memory |
| `correct_basal_expression()` | Python-streamed backed path | `0.81 GiB -> 0.36 GiB` | `2.29 GiB -> 0.97 GiB` | about `2.3x` lower peak memory |

Interpretation:

- For SVD and kernel reduction, the benefit is primarily enablement plus dramatic RAM reduction. The new path avoids materializing the dense matrix in memory and instead pays only for:
  - operator slab buffer
  - Halko basis / output matrices of size `O((n_obs + n_vars) * k)`
- For batch correction, the benefit is smaller but still meaningful because the current path already avoids materializing the whole dense matrix. The new operator path mainly removes:
  - large Python chunk temporaries
  - Python-side `old_V` reconstruction
  - Python-side deflation copies

### Runtime Envelope Estimates

#### `run_svd()` / `reduce_kernel()`

There is no honest "speedup" claim against the current dense-backed code because the current dense-backed operator path fails outright.

The right runtime comparison is:

- current state: unsupported unless the caller materializes the dense matrix into RAM
- proposed state: bounded-memory out-of-core operator path

Halko lower-bound read times from raw pass counts:

| Path | Moderate dense | Atlas dense |
|---|---:|---:|
| `run_svd()` backed Halko, 12 passes | `1.9-3.7 min` | `19-38 min` |
| `reduce_kernel()` backed Halko, 14 passes | `2.2-4.3 min` | `22-45 min` |

Implication:

- The dense-backed operator path is a memory-efficiency and scalability win, not a wall-clock win over an already in-memory dense matrix.
- That tradeoff is still favorable because the current alternative is either:
  - failure
  - manual dense materialization with `37-383+ GiB` RAM pressure

#### `correct_batch_effect()` / `correct_basal_expression()`

These paths already stream today, so the runtime benefit is more conventional.

Raw scan lower bounds:

| Path | Moderate dense | Atlas dense |
|---|---:|---:|
| `correct_batch_effect()` two passes | `18.6-37.3 s` | `3.2-6.4 min` |
| `correct_basal_expression()` one pass | `9.3-18.6 s` | `1.6-3.2 min` |

Expected runtime improvement after consolidation:

- `correct_batch_effect()`: roughly `1.5x-2.5x` faster
- `correct_basal_expression()`: roughly `1.3x-2.0x` faster

Why the gain is moderate rather than massive:

- both old and new implementations read the same matrix one or two times
- the arithmetic is already delegated to BLAS in large chunks
- the improvement comes from removing Python chunk orchestration, h5py slicing overhead, redundant temporary arrays, and Python-side perturbation bookkeeping

### Bottom-Line Estimate

If the proposed operator consolidation is implemented:

- dense-backed decomposition and kernel reduction become usable with `~70x-236x` lower peak memory than the full-dense workaround
- dense-backed batch correction gets a real, but smaller, improvement: about `2.3x` lower peak memory and `1.3x-2.5x` faster runtime
- the only area where wall-clock time does not improve is dense-backed SVD / kernel reduction versus a matrix that already fits in RAM; those paths are fundamentally paying for repeated out-of-core passes

Given the repo goals, that is still the correct trade:

- runtime performance improves where Python overhead is the bottleneck
- memory usage improves everywhere
- previously broken dense-backed operator workflows become available

## Recommended Implementation Scope

### `libactionet`

1. Add `BackedDenseMatrixOperator` under `include/io/backed_h5ad/` and `src/io/backed_h5ad/`.
   - Read dense HDF5 datasets via hyperslabs.
   - Expose logical shape `n_var x n_obs`, matching `BackedSparseMatrixOperator`.
   - Implement `matvec`, `rmatvec`, `matmat`, `rmatmat`.
   - Support `row_scale_factors` and `apply_log1p`.
   - Internally clamp dense row slabs to a `256 MiB` byte budget.

2. Add a generic backed factory returning `std::shared_ptr<MatrixOperator>`.
   - Detect whether `group_path` points to:
     - a sparse CSR / CSC group, or
     - a dense HDF5 dataset
   - Return the matching operator implementation.

3. Add operator orthogonalization APIs in `decomposition/orthogonalization.hpp`.
   - `orthogonalizeBatchEffect_Operator(...)`
   - `orthogonalizeBasal_Operator(...)`
   - Return `PerturbedSVDResult`
   - Reuse the same deflation and perturbation math as the in-memory path

4. Refactor existing in-memory orthogonalization helpers to share the core deflation code.

### `actionet-python`

1. Update `src/actionet/wp_io.cpp`.
   - Make `_core.create_backed_operator(...)` return a generic `MatrixOperator`.
   - Bind `BackedDenseMatrixOperator` for debugging / tests.

2. Update `src/actionet/core.py`.
   - `run_svd()` and `reduce_kernel()` should continue calling `_core.create_backed_operator(...)`, but that factory will now support dense-backed datasets correctly.

3. Update `src/actionet/wp_decomposition.cpp`.
   - Add:
     - `orthogonalize_batch_effect_backed_operator(...)`
     - `orthogonalize_basal_backed_operator(...)`

4. Update `src/actionet/batch_correction.py`.
   - Replace the backed branches with the new operator bindings.
   - Remove Python-side perturbation assembly from the backed hot path:
     - `_streamed_batch_terms`
     - `_streamed_basal_terms`
     - `_deflate_terms`
     - `_perturbed_with_prior`
   - These helpers can be deleted or left only for test/oracle purposes if useful during bring-up.

5. Leave scan-shaped dense-backed workflows unchanged.
   - `compute_feature_specificity()`
   - `compute_archetype_feature_specificity()`
   - `find_markers()`
   - normalization / filtering / imputation / backed `obsm`

## Test And Validation Plan

Add explicit dense-backed tests under `tests/backed/`.

Required coverage:

1. Dense-backed `.X`:
   - `run_svd()`
   - `reduce_kernel()`
   - `correct_batch_effect()`
   - `correct_basal_expression()`

2. Dense-backed `layers[...]`:
   - same four workflows

3. Parity checks against in-memory dense:
   - reduction shapes
   - no NaNs / infs
   - singular values within tolerance
   - corrected embeddings highly correlated up to sign / rotation tolerance

4. Regression protection for sparse-backed operator mode:
   - CSR and CSC still pass existing backed tests unchanged

5. Dense-backed fallback preservation:
   - backed dense specificity still uses the streamed fallback and still works

## Branch Handling

The synced development branches are already created:

- `/Users/sebastian/Documents/git_projects/actionet-python`
- `/Users/sebastian/Documents/git_projects/actionet-python/src/libactionet`
- `/Users/sebastian/Documents/git_projects/libactionet`

All three are on:

- branch: `codex/dense-backed-operator-v1`

Implementation order should be:

1. `libactionet` sibling repo
2. `actionet-python/src/libactionet` submodule worktree aligned to the same commit
3. `actionet-python` bindings and Python call sites
4. update the submodule pointer in `actionet-python`

## Final Recommendation

Proceed with dense-backed operator consolidation, but keep the scope exact:

- yes for decomposition and batch orthogonalization
- no for scan-based algorithms in this pass

That is the highest-leverage fix for the current fragmentation:

- it repairs the broken dense-backed operator boundary
- it moves performance-critical batch correction back into the core
- it preserves the correct abstraction split between operator-shaped and scan-shaped workflows
