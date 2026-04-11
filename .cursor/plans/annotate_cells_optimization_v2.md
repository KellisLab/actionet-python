# Corrected Plan: `annotate_cells` Scalability + Parity Safety

## Summary
- Reduce runtime and peak memory for atlas-scale annotation (including backed + lazy transform) without changing public behavior.
- Address the confirmed bottlenecks: enrichment temporaries, diffusion graph mutation/rework, backed dense Pass 3, and repeated pybind graph serialization.
- Success criteria: numerical parity with baseline plus measurable runtime/memory improvements.

## Implementation Changes
- **Diffusion correctness and reuse (L2 + L3 together):** introduce an internal non-mutating diffusion path over a prepared graph; keep external entry points compatible; prepare normalized/scaled diffusion graph once per call and reuse across ACTIONet label iterations.
- **Enrichment kernel rewrite (L1):** replace `sum(square(G), 1)` materialization with one NNZ pass computing `row_sum`, `row_sum_sq`, and `row_max`; compute Bennett terms in-place from `Obs = G @ scores` to remove extra dense temporaries.
- **Backed VISION row stats rewrite (L4):** add fused `rowStats(row_sum, row_sum_sq, nnz)` for backed sparse operator using NNZ-only streaming with lazy-transform applied before accumulation; replace current backed sparse Pass 2 + Pass 3 with this call.
- **In-memory sparse row-sum-squared fix (P1):** replace `S.power(2).sum(axis=1)` with safe NNZ-based formulas (CSR via cumulative-sum + `indptr` diffs, CSC via `bincount` on row indices), preserving `float64` and current `sigma_sq` math.
- **Pybind fused VISION path (F1) with semantics guardrail:** add fused internal binding so `G` crosses pybind once; ensure enrichment never uses a diffusion-mutated graph state; remove any temporary in-process graph clone only after L2/L3 guarantees are in place.
- **Corrected sizing model:** update planning/benchmark notes to reflect that `(1,000,000 x 1024)` dense slab is ~8 GB, not ~8 MB.

## API / Interface Changes
- Public Python API remains unchanged (`annotate_cells` signature and return keys unchanged).
- Add internal interfaces only: prepared/non-mutating diffusion helper, backed sparse `rowStats(...)`, and internal `_core.annotate_cells_vision_fused(...)`.
- Keep existing bindings available during rollout for backward compatibility.

## Test Plan
- Parity tests for VISION and ACTIONet paths with and without enrichment.
- Coverage across in-memory CSR/CSC and backed CSR/CSC/dense, plus lazy-transform on/off.
- Mutation tests asserting diffusion paths that should be non-mutating do not alter caller graph state.
- Row-stat edge-case tests: empty rows, all-zero matrices, mixed sparsity, and large-index cases.
- Fused-vs-legacy pipeline tests comparing `marker_stats`, `log_pvals`, labels, and confidence.
- Acceptance tolerances: `rtol=1e-8`, `atol=1e-10` for continuous outputs; exact label match (with deterministic tie-break behavior).
- Performance regression checks confirming removal of dense Pass 3 behavior for backed sparse and significant row-stat-stage runtime reduction.

## Execution Order
- Step 1: P1 + tests.
- Step 2: L1 + parity tests.
- Step 3: L2 and L3 together + mutation/determinism tests.
- Step 4: L4 + lazy-transform parity + performance benchmark.
- Step 5: F1 with guardrail; optional clone removal after L2/L3 is fully validated.

## Assumptions and Defaults
- No public API breaking changes are desired.
- Output semantics must remain parity-compatible with current implementation.
- `lazy_transform` preserves zero-to-zero behavior for unstored entries, so NNZ-only accumulation is valid.
- Existing `thread_no` semantics and OpenMP behavior remain unchanged.
