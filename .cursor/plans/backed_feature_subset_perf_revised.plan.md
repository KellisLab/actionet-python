---
name: Backed Feature Subset Perf
overview: Eliminate Python-side row-chunked feature_subset bottleneck for annotate_cells and impute_features by routing backed column extraction through C++ operators and adding a vision-method backed overload.
todos:
  - id: ws1-feature-lookup
    content: "Workstream 1: Add _feature_lookup.py with first-match semantics; rewire imputation.py, annotation.py, _encode_markers()"
    status: pending
  - id: ws2-sparse-take-columns
    content: "Workstream 2a: Implement takeColumnsDense/takeColumnsSparse on BackedSparseMatrixOperator (CSR + CSC)"
    status: pending
  - id: ws2-dense-take-columns
    content: "Workstream 2b: Implement dense-backed takeColumns via matmat with selector matrix"
    status: pending
  - id: ws2-pybind
    content: "Workstream 2c: Add pybind11 backed_take_columns binding in wp_io.cpp; update MatrixSource.feature_subset()"
    status: pending
  - id: ws3a-vision-backed
    content: "Workstream 3a: Add computeFeatureStatsVision backed overloads (sparse + dense) + pybind wrapper"
    status: pending
  - id: ws3a-annotate-vision
    content: "Workstream 3a: Update annotate_cells(method='vision') backed path to use operator-native binding"
    status: pending
  - id: ws3b-annotate-actionet
    content: "Workstream 3b: Update annotate_cells(method='actionet') backed path to use C++ column-extract + existing in-memory binding"
    status: pending
  - id: ws4-impute
    content: "Workstream 4: Update impute_features backed path (now C++ via MatrixSource.feature_subset -> backed_take_columns)"
    status: pending
  - id: tests
    content: Add ABI tests for backed_take_columns, end-to-end parity tests, duplicate-feature regression tests
    status: pending
  - id: benchmark
    content: Add focused benchmark script for annotate_cells + impute_features across storage modes
    status: pending
isProject: false
---

# Handoff Plan: Backed Feature Extraction and Annotation C++ Path

## Summary

- Remove the Python chunked backed `feature_subset` path from the hot paths used by `impute_features` and `annotate_cells`.
- Add CSR-backed, CSC-backed, and dense-backed C++ ABI coverage for backed column extraction.
- Add a C++ `computeFeatureStatsVision` backed overload for `annotate_cells(method="vision")`.
- For `annotate_cells(method="actionet")`, use C++column extraction + existing in-memory `computeFeatureStats` binding (document gap for future full-C++ overload).
- Normalize Python duplicate-feature handling to match R's first-match behavior.
- Keep the R front-end API stable. All `libactionet` changes are additive.

## Design Decisions

- **Workstream 3 scope**: Only the `vision` method gets a full backed C++overload. The `actionet` method uses C++ column-extract (Workstream 2) + existing in-memory `computeFeatureStats` binding. This gives ~95% of the speedup with significantly less C++ work. The gap is documented for future implementation.
- `**row_indices` support**: Included in `takeColumns` for future-proofing (e.g. backed label-filtered annotation).
- **Dense-backed column extraction**: Uses `matmat` with a sparse selector matrix rather than custom HDF5 hyperslab reads. Dense-backed datasets are practically capped at ~50K cells, so the extra FLOP cost of the selector multiply is negligible, and this avoids a new HDF5 read path entirely.

## R Compatibility Decisions

- Keep the existing `computeFeatureStats(arma::sp_mat&, arma::sp_mat&, arma::sp_mat&, ...)` and `computeFeatureStatsVision(...)` signatures untouched. Add backed overloads beside them; do not change or remove the symbols used by `actionet-r` Rcpp exports.
- Do not change any `actionet-r` R function signatures, `RcppExports`, or wrapper call shapes. `actionet-r` currently does not support backed AnnData, and that remains unchanged in this pass.
- Mirror any `libactionet` header/source additions into the sibling `libactionet` repo and the vendored `actionet-r/src/libactionet` copy before R validation so the R build remains in sync.
- When constructing new Armadillo sparse outputs in C++, emit locations in column-major order, or use Armadillo construction paths that sort locations, so behavior stays compatible with the column-major CSC assumptions used by both Python and R builds.
- Python duplicate resolution must mimic R's first-match behavior (`unique(...)` + `match(...)`), not Python's current dict-overwrite behavior.

## Workstream 1: Python Feature Lookup Normalization

**Independent of Workstreams 2-4; can be implemented in parallel.**

- Add a new private helper module [src/actionet/_feature_lookup.py](src/actionet/_feature_lookup.py) with two stable helpers:
  - `resolve_feature_space(adata, features_use, *, context)` returns the raw length-`n_vars` feature-label vector, a first-occurrence lookup map, and duplicate metadata.
  - `resolve_requested_features(requested, lookup, *, context)` stable-deduplicates requested feature names, returns matched names in first-request order, matched indices, and missing names.
- Warning policy:
  - if the feature-space vector contains duplicated labels, emit one `UserWarning` per public call saying duplicates are invalid and the first occurrence will be used
  - duplicate requested feature names are collapsed silently to first occurrence to match R
- Replace direct `feature_to_idx = {feat: idx ...}` logic in [src/actionet/imputation.py](src/actionet/imputation.py), [src/actionet/annotation.py](src/actionet/annotation.py), and any other user-facing hot path that resolves feature labels by name.
- Rewrite `_encode_markers()` so it builds a full-width `(n_vars x n_labels)` marker matrix by lookup, not with `np.isin(feature_set, values)`. Only the first matching feature row is marked when duplicates exist.
- Keep the full-width marker matrix sparse on the Python side. For list/data-frame marker input, return CSR immediately instead of dense NumPy.

## Workstream 2: Backed Column Extraction C++ ABI

### 2a: Sparse-backed `takeColumns` (new C++ methods)

Add public methods to `BackedSparseMatrixOperator`:

```cpp
arma::mat takeColumnsDense(const arma::uvec& col_indices,
                           const arma::uvec& row_indices = {}) const;
arma::sp_mat takeColumnsSparse(const arma::uvec& col_indices,
                               const arma::uvec& row_indices = {}) const;
```

- CSR implementation: single-pass scan over NNZ chunks (same loop structure as `rmatmat_csr_`). For each row, filter stored elements to only those whose column index appears in the requested set. Use a lookup vector (length `n_var`, mapped col -> output position, -1 for unselected) for O(1) filtering per element. When `row_indices` is non-empty, skip rows not in the set using the eager `indptr_` array to jump over them.
- CSC implementation: iterate only over the requested columns' `indptr` ranges (direct seek). When `row_indices` is non-empty, filter rows during the scan. This is strictly more efficient than a full scan for small column subsets.
- `takeColumnsDense` returns `arma::mat(n_selected_rows, n_selected_cols)` in request order.
- `takeColumnsSparse` builds column-major-sorted triplets then constructs `arma::sp_mat`.
- Preserve request order and duplicate column/row indices.

Files: [backed_sparse_matrix_operator.hpp](src/libactionet/include/io/backed_h5ad/backed_sparse_matrix_operator.hpp), [backed_sparse_matrix_operator.cpp](src/libactionet/src/io/backed_h5ad/backed_sparse_matrix_operator.cpp)

### 2b: Dense-backed `takeColumns` (via matmat)

Add public methods to `BackedDenseMatrixOperator`:

```cpp
arma::mat takeColumnsDense(const arma::uvec& col_indices,
                           const arma::uvec& row_indices = {}) const;
```

Implementation: construct a sparse selector matrix `E` of shape `(n_var, n_selected)` with `E(col_indices[j], j) = 1.0`, then call `this->matmat(E, Y)`. For `row_indices`, extract the needed rows from `Y` after the matmat.

Rationale: dense-backed datasets are practically capped at ~~50K cells. At that scale, the full-width slab reads performed by `matmat` (~~12 GB total I/O for 30K genes) complete in a few seconds. A custom column-selective HDF5 read path would save I/O but adds significant complexity for a storage mode that rarely appears at scale. The `matmat` selector approach requires zero new HDF5 code.

`takeColumnsSparse` for dense-backed: call `takeColumnsDense` then convert to `arma::sp_mat` (acceptable given the small scale).

Files: [backed_dense_matrix_operator.hpp](src/libactionet/include/io/backed_h5ad/backed_dense_matrix_operator.hpp), [backed_dense_matrix_operator.cpp](src/libactionet/src/io/backed_h5ad/backed_dense_matrix_operator.cpp)

### 2c: pybind11 binding and Python integration

- Expose one internal pybind entry point in [src/actionet/wp_io.cpp](src/actionet/wp_io.cpp):
  - `_core.backed_take_columns(op, col_indices, row_indices=None, prefer_sparse=False)`
  - Accept `std::shared_ptr<MatrixOperator>`, dispatch via `dynamic_cast` to `BackedSparseMatrixOperator` or `BackedDenseMatrixOperator`
  - Release the GIL around the C++ extraction
  - Return dense numpy or scipy CSR based on `prefer_sparse`
- Update [src/actionet/_matrix_source.py](src/actionet/_matrix_source.py):
  - In `feature_subset()`, detect backed mode and route through `_core.backed_take_columns` instead of the Python chunk loop
  - In-memory behavior stays unchanged

## Workstream 3: Backed `annotate_cells`

### 3a: Vision method -- full C++ backed overload

Add additive backed overloads in [marker_stats.hpp](src/libactionet/include/annotation/marker_stats.hpp) and [marker_stats.cpp](src/libactionet/src/annotation/marker_stats.cpp):

```cpp
arma::mat computeFeatureStatsVision(BackedSparseMatrixOperator& op,
                                    arma::sp_mat& G, arma::sp_mat& X, ...);
arma::mat computeFeatureStatsVision(BackedDenseMatrixOperator& op,
                                    arma::sp_mat& G, arma::sp_mat& X, ...);
```

Design:

- Convert `X` to a dense column-major `arma::mat` once inside C++ (n_labels is small and `matmat` expects dense RHS)
- Compute `stats = S @ X_dense` via `op.matmat` -- single-pass streaming read
- Compute per-cell `mu`, `nnz`, and `sigma_sq` using dedicated operator scans:
  - Sparse-backed: accumulate during a single NNZ scan (similar to backed specificity)
  - Dense-backed: accumulate during slab loop (reuse `readSlab`)
- Derive sampling statistics analytically from `row_sum`, `row_sum_sq`, `nnz` without re-reading element-by-element
- Apply standardization and optional network diffusion smoothing (calls existing `computeNetworkDiffusion` on the in-memory marker stats)
- Output semantics identical to the current in-memory `computeFeatureStatsVision`

Add pybind wrapper in [wp_annotation.cpp](src/actionet/wp_annotation.cpp):

- `_core.compute_feature_stats_vision_backed_operator(op, G, X, ...)`
- Accept `std::shared_ptr<MatrixOperator>`, dispatch to sparse/dense backed overloads

Update `annotate_cells()` vision backed mode in [annotation.py](src/actionet/annotation.py):

- Resolve features through the shared lookup helper
- Build the full-width sparse marker matrix
- Create the backed operator once
- Call the new vision backed binding directly
- Skip `source.feature_subset()` entirely

### 3b: ACTIONet method -- column-extract + in-memory binding

For `annotate_cells(method="actionet")` in backed mode:

- Use `MatrixSource.feature_subset()` (now C++ via Workstream 2) to extract marker columns
- Pass the result through the existing in-memory `_core.compute_feature_stats(G, S, X, ...)` binding
- This eliminates the Python chunk-loop overhead (the main bottleneck) while reusing the existing C++ algorithm

**Documented gap**: a full `computeFeatureStats` backed overload (streaming per-label column extraction + diffusion inside C++) would eliminate the remaining `scipy_to_arma_sparse(S)` conversion cost. This is deferred because:

- The actionet method's per-label loop requires column extraction + network diffusion for each label, making a streaming overload architecturally complex
- The column-extract approach already removes ~90-95% of the backed overhead
- The conversion cost is proportional to `n_cells x n_markers` (small) not `n_cells x n_genes`

## Workstream 4: Backed `impute_features` and Sibling Callers

- `impute_features()` continues using `MatrixSource.feature_subset()`, which after Workstream 2 is entirely C++ for backed data.
- Update `impute_features()` and `impute_from_archetypes()` to use the shared feature-lookup helper so matched-feature order and first-match semantics stay coherent with R.
- Update `_encode_markers()` call sites in `annotation.py` so non-`annotate_cells` annotation flows share the same duplicate handling.

## Test Plan

- Add direct ABI tests for `_core.backed_take_columns` covering:
  - sparse CSR-backed
  - sparse CSC-backed
  - dense-backed
  - `prefer_sparse=False` dense output
  - `prefer_sparse=True` CSR output
  - `row_indices` filtering
  - empty selections
  - duplicate row/column index preservation
  - exact output equality against current in-memory slice semantics
- Add end-to-end parity tests for backed vs in-memory on CSR, CSC, and dense-backed fixtures:
  - `impute_features()` output `allclose`
  - `annotate_cells(method="vision")` enrichment `allclose` and labels exact
  - `annotate_cells(method="actionet")` enrichment `allclose` and labels exact
- Add duplicate-feature regression tests:
  - duplicate `adata.var_names`
  - duplicate `adata.var[features_use]`
  - warning emitted once
  - first occurrence selected in Python
  - duplicate requested feature names collapse to the first request occurrence
- Extend existing backed extension tests rather than replacing them; add focused new pytest modules for ABI and duplicate semantics.

## Benchmark and Acceptance Plan

- Add a focused benchmark script under `tests/` that runs only `annotate_cells` and `impute_features` on matched synthetic datasets in three storage modes: sparse CSR-backed, sparse CSC-backed, and dense-backed.
- Benchmark metrics: wall time, peak RSS, bytes read, bytes written.
- Reuse the existing benchmark harness/report format so results land in the current benchmark results directory.
- Post-implementation acceptance run:
  - run the new parity pytest modules in `actionet-python`
  - run the focused backed annotation/imputation benchmark
  - run the existing branch-compare benchmark against the baseline branch for the two targeted stages
  - sync the `libactionet` changes into sibling `libactionet` and `actionet-r/src/libactionet`, rebuild `actionet-r`, and run in-memory `annotateCells()` / `imputeFeatures()` smoke tests to confirm no R API or wrapper breakage

## Assumptions

- `actionet-r` backed AnnData support remains out of scope; the plan only guarantees that additive C++ changes do not break the current in-memory R API.
- No existing Python or R public function signatures change.
- Any new `libactionet` declarations added to public headers are additive only and must not alter overload resolution for existing Rcpp wrapper calls.
- Dense-backed datasets are practically capped at ~50K cells; full-width slab reads are acceptable for this storage mode.
