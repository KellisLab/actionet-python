# Lazy `logcounts` Feasibility Pass

## Executive Summary

Feasibility is **high** for backed mode and **moderate-to-high** for in-memory mode.

The backed C++ operator stack already supports on-read transforms:

- per-row scaling (`row_scale_factors`)
- element-wise `log1p` (`apply_log1p`)

These hooks are wired through pybind but currently disabled at key Python call sites (`row_scale_factors=None`, `apply_log1p=False`).

To reach reproducible front-end parity, the operator transform contract must be extended beyond the current boolean `apply_log1p` to include:

- `scale_param` semantics (after library-size normalization)
- `log_base`
- `pseudocount`

This can remove the requirement to persist `.layers["logcounts"]` for heavy workflows, materially reducing RAM pressure and backed-mode memory spikes.

## Why This Is Promising

### 1. Backed operator transform hooks already exist

- Python binding accepts `row_scale_factors` and `apply_log1p` in `_core.create_backed_operator` (`src/actionet/wp_io.cpp`).
- Backed factory forwards these to sparse/dense operators (`src/libactionet/src/io/backed_h5ad/create_backed_operator.cpp`).
- Sparse operator applies transform in matvec/matmat and column extraction (`transform_value_`) (`src/libactionet/src/io/backed_h5ad/backed_sparse_matrix_operator.cpp`).
- Dense operator applies transform in slab reads (`apply_transforms_`) and thus all matvec/matmat + take-columns (`src/libactionet/src/io/backed_h5ad/backed_dense_matrix_operator.cpp`).

### 2. Downstream backed algorithms already consume operator math

Backed paths for the main hotspots call operator-based C++:

- `run_svd` and `reduce_kernel` (`src/actionet/core.py`)
- backed specificity for marker workflows (`src/actionet/core.py`, `src/libactionet/src/annotation/specificity.cpp`)
- backed VISION marker stats (`src/actionet/annotation.py`, `src/libactionet/src/annotation/marker_stats.cpp`)

So lazy transforms can propagate without rewriting algorithm internals.

### 3. `normalize_anndata` already computes needed row-scale factors

Normalization logic already computes:

- `row_sums`
- per-row scaling to `target_sum`
- optional `log1p`

(`src/actionet/preprocessing.py`)

This code can be reused or mirrored to produce transform metadata without materializing a dense/sparse normalized layer.

## R Normalization Semantics (Validated)

The current R frontend path is split between R and C++:

- R computes library sizes and normalization factors in `R/normalization.R`.
- C++ scales dense/sparse matrices via `C_scaleMatrixDense/Sparse` (`src/libactionet/wrappers_r/wr_tools.cpp` → `scaleMatrix` in `src/libactionet/src/tools/matrix_transform.cpp`).
- R applies the transform function and pseudocount on matrix values.

For `normalize.matrix(S, dim=1, scale_param=s, trans_func=log_base, pseudocount=p)` this is:

1. `lib_i = rowSum(abs(S_i))`
2. `norm_i = 1 / max(lib_i, 1)`
3. `scale_i = norm_i * scale_param_i` (scalar or row-wise vector)
4. transformed value: `f(scale_i * x + p)` where `f` is chosen by `trans_func`

For sparse matrices, transform is applied on stored entries (`S@x`) only. This is critical when `pseudocount != 1` because implicit zeros remain zero.

## Current Gaps

### 1. Transform hooks are not used by high-level backed entry points

Current backed call sites explicitly disable transforms:

- `reduce_kernel` and `run_svd` (`src/actionet/core.py`)
- backed specificity dispatch (`src/actionet/core.py`)

Other backed operator call sites omit transform args and therefore use defaults:

- batch correction operators (`src/actionet/batch_correction.py`)
- `annotate_cells` VISION backed path (`src/actionet/annotation.py`)
- backed column extraction via `MatrixSource.feature_subset` (`src/actionet/_matrix_source.py`)

### 2. Plotting paths still assume persisted `logcounts`

Notable eager/read-heavy paths:

- default layer is `"logcounts"` in feature expression plotting (`src/actionet/plotting/feature_expression.py`)
- `_extract_expression` uses `anndata_to_matrix(..., transpose=True)` which can force materialization for large objects (`src/actionet/plotting/feature_expression.py`)
- UMAP color helpers read directly from `adata.layers[layer]`/`adata.X` (`src/actionet/plotting/utils.py`, `src/actionet/plotting/umap.py`)

### 3. In-memory paths still rely on full matrix objects

In-memory branches often call `anndata_to_matrix(...)` and pass full matrices into C++ dense/sparse entrypoints (`src/actionet/core.py`, `src/actionet/imputation.py`, etc.). This is fine for compute speed but does not inherently provide lazy transform semantics.

## Operation-by-Operation Feasibility

### A. `run_svd` / `reduce_kernel`

**Feasibility: High**

- Backed operator already supports exactly the transform pattern needed.
- Expected overhead: one extra pass to compute row sums/scales; then transform is fused into operator passes.
- Should cover the largest OOM pain point first.

### B. `find_markers` / `compute_feature_specificity`

**Feasibility: High**

- Backed specificity C++ path already uses transformed operator values.
- Can avoid creating/storing a normalized layer entirely.

### C. `annotate_cells`

**Feasibility: High (VISION), Medium-High (ACTIONet method)**

- VISION backed path already operator-based.
- ACTIONet path uses backed column extraction; needs transform plumbing through `MatrixSource.feature_subset`.

### D. `impute_features` (selected genes)

**Feasibility: High**

- Backed path already uses `feature_subset`; same transform plumbing applies.

### E. Plot gene expression

**Feasibility: Medium**

- Needs refactor away from transpose/materialize flow to operator-backed selected-column reads.
- This is straightforward but touches user-facing plotting defaults and should be validated carefully.

### F. Batch correction (`correct_batch_effect`, `correct_basal_expression`)

**Feasibility: High**

- Backed branch already uses operator overloads; same transform wiring can be added.
- Important for parity if reduction was computed in lazy-logcounts mode.

## Important Design Decisions To Make

### 1. API surface for lazy transforms

Recommend adding an explicit transform spec (internal dataclass + public kwargs), e.g.:

- `lazy_logcounts=True`
- `logcounts_scale_param=1e4` (or explicit vector)
- `logcounts_log_base=None`
- `logcounts_pseudocount=1.0`
- optional `source_layer` (default `.X`)

This is preferable to implicit behavior tied only to `layer` because it avoids ambiguity and preserves backwards compatibility.

### 2. Caching row scale factors

Computing row sums repeatedly is expensive. Cache options:

- in memory only (fastest to ship)
- persist to `obs`/`uns` for backed re-use across calls/sessions

Start with in-memory cache keyed by `(source_layer, target_sum, n_obs, matrix fingerprint if available)`, then optionally persist.

### 3. Exact transform semantics

Do not lock to one log base. Operator-side lazy transforms should support:

- `scale_param`
- `log_base`
- `pseudocount`

to match front-end normalization behavior exactly.

Recommended operator transform equation:

- `y = log_base(scale * x + pseudocount)` where `scale` is per-row and `log_base(z) = ln(z) / ln(base)` for custom base

Sparse behavior should keep zero-preserving semantics aligned with existing front-end expectations for sparse inputs.

## Consolidation Recommendation (R + Operator)

Best medium-term direction: centralize value-transform math in C++ and reuse it across:

- backed operators (`BackedSparseMatrixOperator`, `BackedDenseMatrixOperator`)
- R wrappers (normalization hot path)
- future Python in-memory fast path (optional)

Pragmatic rollout:

1. Add an internal C++ transform spec/helper used by backed operators first (SVD/reduce_kernel scope).
2. Optionally wire R normalization to the same helper when `trans_func` corresponds to log-family transforms.
3. Keep current fully-generic R `trans_func` fallback for non-log custom functions.

## Performance and Risk Notes

### Expected wins

- Remove persistent 2x footprint from storing `logcounts` layers.
- Avoid backed `.layers` behavior that can trigger large memory use.
- Keep transforms fused with streaming operator passes.

### SVD / reduce_kernel runtime model (backed, Halko default)

With operator Halko (`max_it=5` default), matrix-pass counts are:

- `run_svd`: `2 * (iters + 1) = 12` passes
- `reduce_kernel` (without precomputed SVD): `12 + 2 = 14` passes  
  (`+2` from perturbation terms: one `rmatvec`, one `matmat`)
- `reduce_kernel_from_svd`: `2` passes

Adding lazy normalization introduces:

- one extra row-sum scan pass (if row scales are not cached)
- per-value transform work during each existing operator pass

Estimated I/O-pass overhead from row-sum scan alone:

- `run_svd`: `+1/12` → about **+8%**
- `reduce_kernel`: `+1/14` → about **+7%**
- `reduce_kernel_from_svd`: `+1/2` → about **+50%** (small absolute runtime path)

CPU overhead estimate from per-value transforms:

- scale-only (`value *= row_scale`): usually low, often I/O-hidden
- scale + log + base conversion + pseudocount: moderate-to-high, dataset-dependent  
  (typically modest on sparse I/O-bound backed runs; higher impact on dense or cache-hot runs)

Memory overhead estimate:

- row-scale cache vector: `8 * n_obs` bytes (float64)  
  (for 1M cells: ~8 MB; 5M cells: ~40 MB)

### Primary risks

- Dense-backed selected-column extraction can be less efficient than sparse-backed paths in current implementation; benchmark before broad defaults.
- Repeated row-sum scans without caching can erode gains.
- Behavior drift if existing pipelines rely on non-default `log_base`.

## Proposed Rollout

1. **Phase 1 (highest impact):**
   Add lazy transform plumbing to backed `run_svd` and `reduce_kernel`.

2. **Phase 2:**
   Extend to backed specificity (`compute_feature_specificity`, `find_markers`) and `annotate_cells` VISION path.

3. **Phase 3:**
   Plumb transform through `MatrixSource.feature_subset`, then enable in `annotate_cells` ACTIONet path and `impute_features`.

4. **Phase 4:**
   Refactor plotting read paths to selected-column lazy reads and de-emphasize default `"logcounts"` layer dependence.

5. **Phase 5:**
   Add caching/persistence for row-scale factors and optional exact `log_base` support.

## Test Strategy (minimum)

- Parity tests: lazy-logcounts vs explicit normalized layer for
  - SVD outputs (`u`, `d`, `v`)
  - reduced kernel outputs
  - marker ranks/scores
  - annotation outputs
  - selected-feature imputation
- Coverage across backed sparse CSR/CSC and backed dense.
- Memory regression tests focused on peak RSS for atlas-like dimensions.

## Bottom Line

This is a strong candidate project and aligns with the current architecture: most of the hard low-level pieces are already present. A phased rollout starting with SVD/reduction should deliver the largest memory benefit quickly while containing correctness and performance risk.
