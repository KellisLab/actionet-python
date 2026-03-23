# Dense-Backed Expression Operator: Architecture Findings and Handoff

## Metadata
- Prepared: March 20, 2026
- Scope repo: `/home/sebastian/data/git_projects/actionet-python`
- Related core repo: `libactionet`
- Topic: whether adding dense-backed support for raw expression matrices would simplify the backed architecture

## Question

Investigate where the current package uses the strategy:

- sparse-backed inputs go through a C++ backed path
- dense-backed inputs stay on a Python fallback or are otherwise unsupported

and determine whether adding dense-backed support for backed operations would materially simplify the package architecture.

## Executive Summary

- Yes, but only for the subset of the package that is naturally expressed as a `MatrixOperator` problem over the raw expression matrix (`.X` or `layers[...]`).
- A dense-backed expression operator would cleanly simplify:
  - `run_svd`
  - `reduce_kernel`
  - future backed batch-correction operator work (`correct_batch_effect`, `correct_basal_expression`)
- A dense-backed expression operator would not, by itself, simplify the row-scan algorithms:
  - `compute_feature_specificity`
  - `compute_archetype_feature_specificity`
  - `find_markers`
  - normalization, filtering, and other `MatrixSource` row-chunk workflows
- Recommendation: add dense-backed support for raw expression matrices only, through a real backed operator factory that can return sparse or dense implementations behind `std::shared_ptr<MatrixOperator>`. Do not try to force specificity or network construction through that same abstraction.

## Current Backed Strategy by Area

### 1. Operator-shaped expression algorithms

These are the cleanest targets for dense-backed operator support.

#### `run_svd`

- Backed inputs are treated as operator-capable in general.
- The high-level code checks only whether the input is backed, then unconditionally calls `_core.create_backed_operator(...)`.
- The current factory creates `BackedSparseMatrixOperator` only.
- Result: dense-backed raw expression is an architectural hole, not an intentional alternate implementation.

Relevant code:

- `src/actionet/core.py`
  - `run_svd(...)`
  - backed branch at operator creation
- `src/actionet/wp_io.cpp`
  - `create_backed_operator(...)`
- `src/libactionet/src/io/backed_h5ad/backed_sparse_matrix_operator.cpp`
  - rejects non-CSR/CSC storage

#### `reduce_kernel`

- Same pattern as `run_svd`.
- Backed mode is treated as operator mode without checking whether the storage is sparse.
- The implementation therefore assumes a backed operator exists for any backed raw expression matrix, but the only concrete backed operator is sparse.

Relevant code:

- `src/actionet/core.py`
  - `reduce_kernel(...)`
- `src/actionet/wp_io.cpp`
  - `create_backed_operator(...)`

#### `correct_batch_effect` and `correct_basal_expression`

- Today these still split:
  - backed uses Python streamed helpers
  - in-memory uses C++ dense/sparse bindings
- A dense-backed expression operator would let future backed operator overloads for orthogonalization share one C++ path across:
  - in-memory dense
  - in-memory sparse
  - backed dense
  - backed sparse

Relevant code:

- `src/actionet/batch_correction.py`
  - `_streamed_batch_terms(...)`
  - `_streamed_basal_terms(...)`
  - `correct_batch_effect(...)`
  - `correct_basal_expression(...)`

### 2. Scan-shaped expression algorithms

These do not naturally collapse into `MatrixOperator`.

#### `compute_feature_specificity`

- Sparse-backed uses a dedicated C++ backed scan path.
- Dense-backed falls back to `_compute_specificity_streamed(...)`.
- The dense fallback is not just missing operator support; it is a different algorithmic shape.
- The implementation needs:
  - `global_min`
  - row-wise nonzero counting
  - chunked dense block reads
  - `arr.T @ h_block`
- A dense `MatrixOperator` would help only with the matrix multiply portion. It would not replace the row-counting and block-scan logic.

Relevant code:

- `src/actionet/core.py`
  - `_run_specificity_backed_sparse(...)`
  - `_compute_specificity_streamed(...)`
  - `compute_feature_specificity(...)`

#### `compute_archetype_feature_specificity`

- Same split and same limitation as `compute_feature_specificity`.
- Sparse-backed has a dedicated C++ backed path.
- Dense-backed stays on the Python streamed scan.

Relevant code:

- `src/actionet/core.py`
  - `compute_archetype_feature_specificity(...)`

#### `find_markers`

- Backed sparse delegates to the shared backed specificity dispatcher.
- Backed dense still uses the Python streamed specificity path indirectly.
- A dense expression operator alone would not remove this split.

Relevant code:

- `src/actionet/annotation.py`
  - backed sparse branch
  - backed dense fallback branch

### 3. Row-chunk utilities and preprocessing

These are already organized around row access, not operator algebra.

#### `imputation`

- Backed mode extracts selected features through `MatrixSource.feature_subset(...)`.
- This is a row-chunk materialization workflow, not a matvec/matmat workflow.
- A dense expression operator would not materially simplify this path.

Relevant code:

- `src/actionet/imputation.py`
  - `impute_features(...)`

#### Normalization and filtering

- These use `MatrixSource` row reads and, for backed sparse writes, direct HDF5 dataset updates.
- A dense `MatrixOperator` is the wrong abstraction here.

Relevant code:

- `src/actionet/preprocessing.py`
  - normalization branches
  - filtering stats and chunk-size logic

## Architectural Conclusion

There are really two different backed abstractions in the package:

### A. Operator abstraction

Use this when the algorithm is fundamentally built from:

- `matvec`
- `rmatvec`
- `matmat`
- `rmatmat`

This is the right abstraction for:

- `run_svd`
- `reduce_kernel`
- batch orthogonalization / basal orthogonalization

For this family, adding dense-backed support would make the architecture cleaner and more complete.

### B. Row-reader / scan abstraction

Use this when the algorithm fundamentally needs:

- full row blocks
- chunked scans
- rowwise statistics
- direct block transforms
- optional writes

This is the right abstraction for:

- specificity
- marker detection
- normalization
- filtering
- feature subsetting / imputation inputs

For this family, a dense-backed `MatrixOperator` does not solve the architectural split.

## Recommended Direction

### Near-term recommendation

Add dense-backed support for raw expression matrices only.

Specifically:

- Introduce a `BackedDenseMatrixOperator` in `libactionet` for `.X` / `layers[...]`.
- Make the backed operator factory return `std::shared_ptr<MatrixOperator>` and choose sparse or dense based on the storage encoding.
- Keep the same transpose convention as the sparse-backed operator:
  - logical operator shape should remain `n_var x n_obs`
  - this keeps compatibility with existing decomposition and kernel-reduction expectations

### What this should be used for

- `run_svd`
- `reduce_kernel`
- future backed batch correction

### What this should not be forced onto

- dense-backed specificity
- backed marker detection
- network construction over `obsm`
- normalization and filtering

Those need either:

- the existing `MatrixSource` row-chunk path, or
- a future dedicated dense-backed row-reader in C++

## Why This Simplifies the Architecture

If dense-backed expression operator support is added:

- `run_svd` no longer has an implicit sparse-only limitation hidden behind a generic backed branch.
- `reduce_kernel` can truthfully treat backed raw expression as operator-capable regardless of dense/sparse storage.
- batch correction can be refactored around one operator-based orthogonalization path instead of keeping a dense-backed Python fallback.
- the package architecture becomes cleaner at the boundary:
  - operator-based algorithms use backed operators
  - scan-based algorithms use row readers / `MatrixSource`

That is a meaningful simplification.

## Why This Does Not Solve Everything

Dense-backed specificity is still a separate problem.

The current streamed implementation needs more than matrix products:

- global minimum scan
- dense positivity / support counts
- row chunk iteration
- dense block transforms before accumulation

That means a dense-backed operator would not eliminate the dense-backed specificity split unless the specificity algorithm itself is reimplemented in C++ around a dense row-reader or dense scan interface.

The same applies to network construction over backed dense `obsm`, which previous investigation already identified as a row-reader problem rather than a `MatrixOperator` problem.

## Current Test Coverage Implication

- Backed end-to-end workflow coverage is overwhelmingly sparse-backed.
- Dense-backed cases exist mainly in infrastructure-level tests, not in full backed expression workflows.
- If dense-backed expression operator support is added, it should come with explicit end-to-end tests for:
  - dense-backed `run_svd`
  - dense-backed `reduce_kernel`
  - dense-backed `correct_batch_effect`
  - dense-backed `correct_basal_expression`

Otherwise the new architecture will exist without meaningful workflow coverage.

## Recommended Handoff Scope

If this topic is advanced into implementation, the clean next work item is:

1. Add `BackedDenseMatrixOperator` in `libactionet`.
2. Replace the sparse-only operator factory with a generic backed matrix operator factory.
3. Update `run_svd` and `reduce_kernel` to use that generic factory.
4. Refactor batch orthogonalization around `MatrixOperator` so dense-backed and sparse-backed can share the same C++ path.
5. Leave specificity and marker detection unchanged for now.
6. Leave the door open for a later dense-backed row-reader project for scan-based algorithms and backed `obsm`.

## Bottom Line

Adding dense-backed support is worthwhile if scoped to raw expression operator workflows.

It would make the package architecture cleaner for decomposition and batch-correction code paths.

It would not remove the dense-backed fallback architecture for specificity, marker detection, preprocessing, or network construction, because those problems are not naturally `MatrixOperator` problems.
