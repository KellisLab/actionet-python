---
name: ""
overview: ""
todos: []
isProject: false
---

# Revised Plan: Batch-Correction Operator Consolidation

## Summary

- Implement batch and basal orthogonalization once against `MatrixOperator` in `libactionet`, then have in-memory dense, in-memory sparse, and backed sparse all use that same core logic.
- Add new internal `_core` backed-operator bindings and route sparse-backed `correct_batch_effect` and `correct_basal_expression` through them.
- Keep backed dense on the existing Python streamed fallback.
- Preserve the public Python API and persisted AnnData schema.

## Interface Changes

- Add two public `libactionet` overloads:
  - `orthogonalizeBatchEffect(const MatrixOperator&, arma::field<arma::mat>&, const arma::mat&)`
  - `orthogonalizeBasal(const MatrixOperator&, arma::field<arma::mat>&, const arma::mat&)`
- Add two internal pybind `_core` bindings:
  - `orthogonalize_batch_effect_backed_operator(op, old_S_r, old_U, old_A, old_B, old_sigma, design)`
  - `orthogonalize_basal_backed_operator(op, old_S_r, old_U, old_A, old_B, old_sigma, basal)`
- Do not change the signatures of `correct_batch_effect` or `correct_basal_expression`.
- Keep the result contract unchanged: `_core` returns `U`, `sigma`, `S_r`, `A`, `B`, and AnnData persistence stays exactly as today.

## Implementation Changes

- `libactionet`
  - Add one internal operator-based helper per algorithm and make the existing dense/sparse template implementations forward into it through `DenseMatrixOperator` and `SparseMatrixOperator`. Do not keep separate dense, sparse, and backed bodies.
  - Use the correct operator orientation everywhere: `Z = S * design` via `op.matmat(design, Z)`, then `B = -(S' * Z)` via `op.rmatmat(Z, B)`. `orthogonalizeBasal` uses the same `op.rmatmat(Z, B)` step.
  - Keep `deflateReduction` unchanged so deflation and perturbed-SVD behavior remain identical across modes.
- C++ bindings
  - In the decomposition binding, factor the current “old reduction state -> SVD field” packing and “orthogonalization result -> dict” packing into shared helpers, and make the existing dense/sparse bindings use them too.
  - Accept `std::shared_ptr<actionet::MatrixOperator>` in the new backed bindings. Do not couple this binding file to the concrete backed H5AD type.
  - Release the GIL only around the long-running C++ orthogonalization call.
- Python dispatch
  - Add one private backed-sparse dispatcher in `batch_correction.py` that creates the operator in `try/finally`, calls the new `_core` binding, and clears the operator handle afterward.
  - Route `source.is_backed and source.is_sparse` through the new binding for both batch and basal correction.
  - Keep backed dense on `_streamed_batch_terms` and `_streamed_basal_terms`. Keep in-memory dense/sparse dispatch unchanged.
  - Do not add compression flags or auto-decompression in this pass.
- Cross-repo
  - Land the `libactionet` change first, then update the submodule pointer in `actionet-python`.
  - No `actionet-r` wrapper changes in this pass.
  - Update the batch-orthogonalization doc to state that sparse-backed batch/basal correction now uses the same C++ orthogonalization path as in-memory, while dense-backed remains the fallback.

## Test Plan

- Rebuild with `./install_optimized.sh`.
- Extend backed pytest coverage for:
  - sparse-backed CSR `correct_batch_effect` parity vs in-memory sparse
  - sparse-backed CSC `correct_batch_effect` parity vs in-memory sparse
  - sparse-backed `correct_basal_expression` parity vs in-memory sparse
  - backed dense fallback smoke for both functions
- In each parity case, assert near-equality for `S_r`, `U`, `sigma`, `A`, and `B`, not only the reduced embedding.
- Keep the existing backed pipeline parity smoke and basal smoke passing.
- Add a focused `scripts/bench_batch_correction_parity.py` that compares the old helper path (`_streamed_*_terms` + `_perturbed_with_prior`) against the new backed-operator path on the same sparse-backed input, reports wall time and peak RSS, and enforces `max abs diff <= 1e-6` for batch-correction outputs.

## Expected Impact

- Sparse-backed `correct_batch_effect`: expect about `1.2x` to `1.6x` lower wall time and `0%` to `15%` lower peak RSS. The algorithm still does the same two matrix passes and the same dense `Z` / `B` intermediates, so this is a moderate improvement, not a step-function change.
- On the current repo large benchmark baseline (`3.80s`, `28 MB` backed peak RSS), the expected post-change range is roughly `2.4s` to `3.2s` and about `24` to `28 MB`.
- Sparse-backed `correct_basal_expression`: expect a smaller `1.1x` to `1.4x` speedup and essentially flat memory.
- In-memory dense/sparse behavior should be unchanged. Backed dense behavior should also be unchanged because it stays on the fallback path.

## Assumptions And Defaults

- Public Python APIs stay unchanged in this pass.
- Only sparse-backed inputs move to the C++ ABI. Dense-backed remains the current Python fallback permanently for now.
- Sparse-backed semantics follow the shared C++ orthogonalization path after this refactor. Do not add extra rank-deficiency handling beyond what the shared path already does.
- Compression handling is intentionally unchanged for this work item.
