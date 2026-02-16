# Agent Handoff: Backed AnnData Extension

## Metadata
- Date: 2026-02-16
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
- `normalize_anndata` (total-count scaling + log1p, replacing separate `normalize_ace` + `log1p_ace`)

Constraints/assumptions retained:
- Backed metadata/object writes use the experimental `_anndata_io` path.
- Native detach stays `AnnData.to_memory()` (no custom detach logic).
- C++ scope minimized to pybind surface extension only.

## Planned Implementation (Agreed Spec)
1. Add matrix abstraction for chunked `.X` and `.layers` reads/writes.
2. Add centralized backed persistence wrapper using `_anndata_io.append_to_anndata`.
3. Refactor mutating APIs to update in-memory object + persist when backed.
4. Add chunked preprocessing (`normalize_anndata`) and backed-safe `filter_anndata`.
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
  - added `normalize_anndata(...)` — combined total-count scaling + log1p in a single
    chunked, backed-safe function (mirrors R `normalize.ace`)
    - `target_sum`: target total count per cell (default: 1e4)
    - `log_transform`: enable/disable log1p step (default: True)
    - `log_base`: custom log base (default: None = natural log)
    - `layer`: operate on `.X` or a named layer
    - `backed_chunk_size`: rows per streaming chunk
    - For backed sparse: scaling + log1p fused into a single `apply_rowwise` pass
      (one read + one write per chunk, vs two passes with separate functions)
  - deprecated aliases `normalize_ace(...)` and `log1p_ace(...)` are retained
    for backward compatibility — they emit `DeprecationWarning` and delegate
    to `normalize_anndata`
  - rewrote `filter_anndata(...)` with three-part architecture:
    - `_compute_filter_stats`: single-pass row_sums + row_nnz + col_nnz accumulation
    - `compute_filter_masks`: pure mask computation without mutation
    - `apply_filter`: backed-native subsetting via chunked h5py write, with `output_file` parameter
  - iterative R-style convergence semantics preserved

### 7) Direct h5py Write Path (anndata `__setitem__` deprecation fix)
- `src/actionet/_matrix_source.py` — `set_rows()` rewritten:
  - **Backed sparse**: writes directly to the h5py `data` array via `_h5py_set_sparse_rows`,
    bypassing anndata's deprecated `CSRDataset.__setitem__` entirely
    - Reads `indptr[start:end+1]` to find the flat data-array slice
    - Validates sparsity structure parity (same nnz per row)
    - Handles dtype promotion (e.g. int64 → float64) via `_h5py_cast_data_dataset`:
      copies the `data` dataset in 10M-element chunks to a temp dataset with new dtype, then swaps.
      Peak RAM: ~80 MB regardless of dataset size (safe for 5B+ nnz datasets).
  - **In-memory sparse**: promotes matrix dtype when values have a wider dtype
    (avoids silent truncation by scipy's `__setitem__`)
  - **Removed**: full-materialization fallback and `_rewrite_backed_X` method
  - Eliminates both:
    - `FutureWarning: __setitem__ for backed sparse will be removed in the next anndata release`
    - `UserWarning: Direct slice assignment failed for backed .X; falling back to full rewrite`

### 8) Public API Exports
- `src/actionet/__init__.py`
  - exports `normalize_anndata` as primary function
  - exports `normalize_ace` and `log1p_ace` as deprecated aliases
  - exports `compute_filter_masks` and `apply_filter`
  - added preprocessing exports to `__all__`

### 9) Tests
- Updated essential usage script:
  - `tests/test_essential.py`
  - now includes `correct_batch_effect` after `reduce_kernel`
- Manual integration test:
  - `tests/test_backed.py` — full backed pipeline using `normalize_anndata`
- Added new test modules:
  - `tests/test_backed_extension.py` — integration tests using new API
  - `tests/backed/test_infrastructure.py` — 55 unit tests for MatrixSource + BackedPersist
  - `tests/backed/test_backed_extension.py` — E2E, persistence, filtering, preprocessing, parity
  - `tests/backed/test_backed_reduce_kernel.py` — backed reduce_kernel smoke test

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
- `tests/test_backed.py`
- `tests/test_backed_extension.py` (new)
- `tests/backed/__init__.py` (new)
- `tests/backed/conftest.py` (new)
- `tests/backed/test_infrastructure.py` (new)
- `tests/backed/test_backed_extension.py` (new)
- `tests/backed/test_backed_reduce_kernel.py` (new)

## Validation Performed
- Full backed pipeline validated end-to-end: `filter_anndata` → `normalize_anndata` → `reduce_kernel` → `run_actionet` → `find_markers` → `annotate_cells` → `impute_features`
- In-memory pipeline validated end-to-end with the same sequence
- API variants tested: `log_transform=False` (scaling only), `log_base=None` (natural log), `log_base=2`, `inplace=False`
- Verified zero `FutureWarning` and zero `UserWarning` (run with `-W error::FutureWarning -W error::UserWarning`)
- dtype promotion verified: int64 sparse → float64 after normalization, both backed and in-memory
- Backed state preserved throughout the entire pipeline (`adata.isbacked == True`)

## Important Notes / Risks
1. Backed writes are centralized through `_backed_persist` and `_anndata_io`; behavior depends on AnnData/HDF5 backend capabilities and file mode (`r+` required).
2. The new streamed specificity implementation in Python mirrors existing C++ formulas; numerical parity should be validated in CI on real datasets.
3. `_h5py_set_sparse_rows` requires that the sparsity structure is preserved (same nnz per row). Operations that change sparsity patterns (e.g., thresholding to zero) would need a different write strategy.
4. `_h5py_cast_data_dataset` is a one-time O(nnz) operation when dtype changes. For very large datasets (>10B nnz), this takes a few minutes but stays within ~80 MB RAM.
5. The deprecated `normalize_ace` and `log1p_ace` aliases will emit `DeprecationWarning` — these should be removed in a future major version.
6. anndata 0.13 will remove `CSRDataset.__setitem__` entirely — our direct h5py path is already the required replacement.

## Recommended Next Steps
1. Rebuild extension and run tests:
   - targeted: `tests/backed/`, `tests/test_backed_extension.py`
   - existing regression suites including SVD/backed tests.
2. Run numeric parity benchmarks against in-memory baselines on representative production-like datasets.
3. If any parity drift is detected, tighten tolerances and/or port additional streamed math into C++.
4. Optionally add docs/examples for backed workflows using `.layers` and `backed_chunk_size` guidance.
5. After validation, stage/commit with focused commits (infra, core refactor, tests).
6. Plan removal of deprecated `normalize_ace`/`log1p_ace` aliases in next major version.

## Quick Continuation Checklist (for next agent)
1. `git checkout codex/oom-backed-extension` in all three repos.
2. Rebuild Python extension so `_core.perturbed_svd_with_prior` is available.
3. Run new backed tests and inspect failures by functional block:
   - preprocessing -> reduce_kernel -> batch correction -> pipeline -> markers/annotation -> imputation.
4. Confirm reopen persistence keys in `obsm/varm/obsp/uns/layers`.
5. Prepare final cleanup and PR notes.
