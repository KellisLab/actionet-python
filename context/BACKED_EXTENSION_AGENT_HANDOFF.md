# Agent Handoff: Backed AnnData Extension

## Metadata
- Date: 2026-02-15
- Primary repo: `/Users/sebastian/Documents/git_projects/actionet-python`
- Companion repos/branches created:
  - `/Users/sebastian/Documents/git_projects/actionet-python` -> `codex/oom-backed-extension`
  - `/Users/sebastian/Documents/git_projects/actionet-python/src/libactionet` -> `codex/oom-backed-extension`
  - `/Users/sebastian/Documents/git_projects/libactionet` -> `codex/oom-backed-extension`
- Base branch used: `codex/oom-backed-svd-kernel`

## Problem Statement
Extend out-of-memory (OOM) functionality so Python ACTIONet workflows can run on backed AnnData (`backed='r+'`) for both `.X` and `.layers[layer]`, not only kernel reduction.

Required target workflows/functions:
- `reduce_kernel`
- `run_actionet`
- `compute_feature_specificity`
- `compute_archetype_feature_specificity`
- `find_markers`
- `annotate_cells`
- `impute_features`
- `correct_batch_effect`
- `correct_basal_expression`
- `filter_anndata`
- normalization/log1p preprocessing

Constraints/assumptions retained:
- Backed metadata/object writes use the experimental `_anndata_io` path.
- Native detach stays `AnnData.to_memory()` (no custom detach logic).
- C++ scope minimized to pybind surface extension only.

## Planned Implementation (Agreed Spec)
1. Add matrix abstraction for chunked `.X` and `.layers` reads/writes.
2. Add centralized backed persistence wrapper using `_anndata_io.append_to_anndata`.
3. Refactor mutating APIs to update in-memory object + persist when backed.
4. Add chunked preprocessing (`normalize_ace`, `log1p_ace`) and backed-safe `filter_anndata`.
5. Add streamed batch/basal correction path (no full transpose materialization).
6. Add pybind helper for prior-aware perturbed SVD accumulation.
7. Expand tests for backed E2E, layer mode, persistence, preprocessing formats, and parity.

## Execution Summary

### 1) Infrastructure Added
- Added backed persistence wrapper:
  - `src/actionet/_backed_persist.py`
- Added matrix source abstraction:
  - `src/actionet/_matrix_source.py`
- Moved experimental IO module into package path:
  - `src/actionet/experimental/_anndata_io.py`
  - `src/actionet/experimental/__init__.py`

### 2) Core + Pipeline Refactor
- `src/actionet/core.py`
  - switched result writes to centralized `persist_updates(...)`
  - extended operator/backed detection to generic backed matrices
  - added streamed specificity helper (`_compute_specificity_streamed`) for backed mode
  - `compute_feature_specificity` and `compute_archetype_feature_specificity` accept `backed_chunk_size`
  - backed paths avoid full matrix materialization
- `src/actionet/pipeline.py`
  - `run_actionet(...)` now accepts `backed_chunk_size` and forwards it to specificity stage
- `src/actionet/visualization.py`
  - `compute_node_colors` writes via persistence wrapper for backed AnnData

### 3) Batch/Basal Correction (New Required Scope)
- `src/actionet/batch_correction.py`
  - added streamed backed path for batch/basal correction:
    - compute `Z` from streamed products (`X.T @ design` or basal state)
    - compute `B = -(X @ Z)` in row chunks
    - apply deflation terms and prior accumulation without full transposed matrix
  - new helper flow uses prior-aware perturbed SVD call
  - both `correct_batch_effect` and `correct_basal_expression` accept `backed_chunk_size`
  - writes persisted via `persist_updates`

### 4) Pybind Extension
- `src/actionet/wp_decomposition.cpp`
  - added binding:
    - `perturbed_svd_with_prior(u, d, v, old_A, old_B, A_new, B_new)`
  - returns updated `u,d,v,A,B`

### 5) Annotation + Imputation Backed Paths
- `src/actionet/annotation.py`
  - `find_markers` now supports backed without forced `.copy()` materialization path
  - in backed mode, uses streamed specificity computation directly
  - added `backed_chunk_size` to `find_markers` and `annotate_cells`
  - `annotate_cells` backed path subsets only required marker features before C++ stats
- `src/actionet/imputation.py`
  - `impute_features` backed path uses feature-column extraction from matrix source
  - avoids transpose-whole-matrix path in backed mode
  - added `backed_chunk_size`
  - fixed `smooth_kernel(return_raw=True)` bug (`"U": U_left`)

### 6) Preprocessing
- `src/actionet/preprocessing.py`
  - added `normalize_ace(...)` (chunked scaling, backed-safe, `.X` + `layer`)
  - added `log1p_ace(...)` (chunked transform, backed-safe, `.X` + `layer`)
  - rewrote `filter_anndata(...)` to avoid unsupported backed operations like direct `(X > 0)` and full direct reductions
  - kept iterative R-style semantics by recomputing masks until shape convergence

### 7) Public API Exports
- `src/actionet/__init__.py`
  - exports `normalize_ace` and `log1p_ace`
  - added preprocessing exports to `__all__`

### 8) Tests
- Updated essential usage script:
  - `tests/test_essential.py`
  - now includes `correct_batch_effect` after `reduce_kernel`
- Added new test module:
  - `tests/test_backed_extension.py`
  - includes:
    - backed E2E (`.X`)
    - backed E2E (`layer='logcounts'`)
    - persistence verification after close/reopen
    - preprocessing checks for CSR and CSC backed inputs
    - parity checks for marker ranking overlap and imputation correlation

## Files Changed
- `src/actionet/__init__.py`
- `src/actionet/annotation.py`
- `src/actionet/batch_correction.py`
- `src/actionet/core.py`
- `src/actionet/imputation.py`
- `src/actionet/pipeline.py`
- `src/actionet/preprocessing.py`
- `src/actionet/visualization.py`
- `src/actionet/wp_decomposition.cpp`
- `src/actionet/_backed_persist.py` (new)
- `src/actionet/_matrix_source.py` (new)
- `src/actionet/experimental/_anndata_io.py` (moved from top-level `experimental/`)
- `src/actionet/experimental/__init__.py` (new)
- `tests/test_essential.py`
- `tests/test_backed_extension.py` (new)

## Validation Performed
- Syntax compilation succeeded for all modified Python files via `python3 -m py_compile`.

## Validation Not Performed
- Full runtime pytest and extension rebuild were not run in this session due environment/toolchain availability limits.
- No C++ compile/test cycle was executed after adding `perturbed_svd_with_prior` binding.

## Important Notes / Risks
1. Backed writes are centralized through `_backed_persist` and `_anndata_io`; behavior depends on AnnData/HDF5 backend capabilities and file mode (`r+` required).
2. The new streamed specificity implementation in Python mirrors existing C++ formulas; numerical parity should be validated in CI on real datasets.
3. Backed layer row-write fallback may rewrite full layer when direct slice assignment is unsupported.
4. Existing pre-existing unrelated workspace changes remain (e.g., notebook/test artifacts).

## Recommended Next Steps
1. Rebuild extension and run tests:
   - targeted: `tests/test_backed_extension.py`
   - existing regression suites including SVD/backed tests.
2. Run numeric parity benchmarks against in-memory baselines on representative production-like datasets.
3. If any parity drift is detected, tighten tolerances and/or port additional streamed math into C++.
4. Optionally add docs/examples for backed workflows using `.layers` and `backed_chunk_size` guidance.
5. After validation, stage/commit with focused commits (infra, core refactor, tests).

## Quick Continuation Checklist (for next agent)
1. `git checkout codex/oom-backed-extension` in all three repos.
2. Rebuild Python extension so `_core.perturbed_svd_with_prior` is available.
3. Run new backed tests and inspect failures by functional block:
   - preprocessing -> reduce_kernel -> batch correction -> pipeline -> markers/annotation -> imputation.
4. Confirm reopen persistence keys in `obsm/varm/obsp/uns/layers`.
5. Prepare final cleanup and PR notes.
