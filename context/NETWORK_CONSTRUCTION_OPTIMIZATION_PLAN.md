## Cross-Repo Network Construction Optimization With R Compatibility

### Summary

Key correction to the prior plan: no row-major dense or CSR-only result should become the shared replacement API for `libactionet`. The R side currently consumes column-major Armadillo types via the legacy `buildNetwork(const arma::mat&) -> arma::sp_mat` contract, so that contract must remain intact.

The design should therefore split into:
- a new internal/core fast builder that is layout-neutral and row-oriented for performance
- a legacy compatibility wrapper in `libactionet` that still accepts column-major `arma::mat` and returns `arma::sp_mat` for R
- a Python-only fast/based path that can use row-major `float32` input and direct CSR/HDF5 output without changing the R API

### Ordered Implementation Plan

| Order | Change | Repo(s) | Difficulty | Impact | Runtime Effect | R/API Impact |
|---|---|---|---|---|---|---|
| 1 | Large-N guardrails and pipeline knobs | `actionet-python` | Low | High | Strong positive by preventing catastrophic paths | No R code change. Update large-scale guidance in docs only. |
| 2 | Expose network input choice in pipeline | `actionet-python` | Low | High | Strong positive when moving from `H_stacked` to `H_merged`/`action` | No R API change. If desired later, mirror only as an R-facing pipeline enhancement. |
| 3 | New core internal `knn` builder with CSR scratch/output, but keep legacy Armadillo API | `libactionet` | Medium-High | Very High | Strong positive | Preserve `buildNetwork(const arma::mat&) -> arma::sp_mat` unchanged. R keeps using this wrapper path. |
| 4 | Python-only `float32` / row-major fast binding | `actionet-python` + additive `libactionet` API | Medium | High | Strong positive | No change to R path. Row-major handling stays Python-specific. |
| 5 | Direct CSR conversion for Python without COO roundtrip [completed 2026-03-07] | `actionet-python` | Medium | Medium-High | Positive | No R impact. |
| 6 | Backed dense `obsm` reader and direct `obsp` writer | `libactionet` + `actionet-python` | High | Very High | Slightly slower on small data, essential at scale | No R API change. Python/HDF5-only path. |
| 7 | Optional R adoption of new core fast path | `actionet-r` | Medium | Medium | Positive if adopted | Separate follow-up. Not required for initial rollout. |
| 8 | Adaptive `k*nn` rewrite for large N | `libactionet` + wrappers | Very High | Conditional | Poor at scale even if memory-safe | Defer unless scientifically required. |

### Implementation Details

#### 1. Immediate operator-facing fixes
- In `actionet-python`, add a Python-side `k*nn` preflight that estimates dense scratch and raises before entering C++ when the path is obviously unsafe.
- In `run_actionet(...)`, add `network_obsm_key` so large runs can select `H_merged` or `action` instead of hard-coded `H_stacked`.
- Document the recommended large-N bundle for both Python and R users:
  - `algorithm="knn"`
  - `k=10..15`
  - `mutual_edges_only=True`
  - moderate thread counts

#### 2. Core refactor with R-safe compatibility layer
- In `libactionet`, implement a new internal builder around a row-oriented reader abstraction and a compact sparse representation such as `CSRGraph32`.
- Do not replace the existing public header contract in [`build_network.hpp`](/Users/sebastian/Documents/git_projects/actionet-python/src/libactionet/include/network/build_network.hpp#L23).
- Instead, make the legacy `buildNetwork(const arma::mat&, ...)` an adapter:
  - accepts column-major Armadillo input for R and existing callers
  - calls the new internal builder
  - converts the internal sparse result back to `arma::sp_mat`
- This preserves R semantics and column-major expectations while still moving the heavy work onto a more scalable internal representation.

#### 3. Python fast path without changing the R ABI
- Add an additive fast API in `libactionet` specifically for Python/HPC use. It can accept row-major `float32` points or a reader object, but it must be additive, not a replacement for the legacy Armadillo API.
- In `actionet-python`, update pybind to call this fast path directly from NumPy without:
  - Python transpose-copy
  - pybind `arma::mat` copy
  - by-value `arma::mat` copy
  - full `arma::fmat` staging copy
- Keep this path Python-only. R continues using the legacy wrapper unless `actionet-r` explicitly opts into the new fast API later.

#### 4. Python sparse result path
- In `actionet-python`, convert the new sparse result directly to SciPy CSR from the core sparse buffers.
- Remove the current `arma::sp_mat -> COO vectors -> NumPy arrays -> coo_matrix -> .tocsr()` roundtrip for the Python fast path.
- Leave the legacy `arma::sp_mat` conversion path available for compatibility and debugging.

Status: Completed in `actionet-python` on 2026-03-07.

- [`src/actionet/wp_utils.cpp`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/wp_utils.cpp) now converts `arma::sp_mat` to SciPy CSR directly from Armadillo's CSC buffers using a row-count, prefix-sum, and scatter pass, which removes the Python COO materialization and `.tocsr()` roundtrip.
- The new converter chooses `int32` SciPy sparse indices when `n_rows`, `n_cols`, and `nnz` fit, and falls back to `int64` otherwise. This reduces Python-side sparse index memory for the common large-graph case without changing the R-facing `libactionet` API.
- The previous COO-based converter remains in the wrapper as an internal fallback/debug path so the Python binding can fall back safely if the direct CSR builder raises.
- Added [`tests/test_sparse_conversion.py`](/Users/sebastian/Documents/git_projects/actionet-python/tests/test_sparse_conversion.py) to cover both an exact sparse aggregation result and the `build_network(...)` sparse return path.
- Validation was benchmarked in `.venv` during implementation and the temporary debug hooks were removed from `src/` afterward so no extra non-public binding surface remains. Median direct-vs-legacy conversion timings were:
  - ring graph `n=10,000`, `nnz=200,000`: `0.544 ms` vs `1.360 ms` (`2.50x` faster)
  - ring graph `n=50,000`, `nnz=1,000,000`: `3.084 ms` vs `9.722 ms` (`3.15x` faster)
  - ring graph `n=100,000`, `nnz=2,000,000`: `6.962 ms` vs `17.675 ms` (`2.54x` faster)
- Output parity was confirmed in `.venv` for both synthetic sparse matrices and actual `build_network()` results. The checked `build_network()` cases were:
  - `n=2,000`, `d=32`, `k=10`: exact parity, conversion `0.130 ms` vs `0.219 ms`
  - `n=10,000`, `d=32`, `k=10`: exact parity, conversion `0.598 ms` vs `0.957 ms`

#### 5. Backed network construction
- In `libactionet`, add a dense HDF5 reader for `/obsm/<key>` and a streaming sparse writer for `/obsp/<key>`.
- Implement a new additive core entry point for backed `knn` only:
  - read `obsm` in row batches
  - build/query HNSW in batches
  - store fixed-width directed neighbor scratch
  - mutualize/symmetrize in a second pass
  - write final CSR directly to HDF5
- In `actionet-python`, dispatch to this path only for backed AnnData.
- Default behavior for the public Python API stays backward compatible:
  - `persist_only=False` by default
  - for large runs, users can opt into `persist_only=True` to avoid reloading the graph into memory
- `run_actionet()` should not switch to `persist_only=True` until downstream graph consumers are made backed-aware.

#### 6. R-specific handling
- `actionet-r` should not be forced to adopt any new data layout in the first rollout.
- Mandatory R work for every `libactionet` phase:
  - rebuild `actionet-r` against the new core
  - verify `C_buildNetwork` still compiles and returns the same `arma::sp_mat`/Matrix semantics
  - run one regression/parity test for graph construction behavior
- Optional later optimization:
  - expose the new core fast builder in `actionet-r`
  - convert its sparse output to `arma::sp_mat` or `dgCMatrix`
  - only do this once Python/HPC behavior is stable

### Test and Acceptance Plan

- `libactionet`
  - Add parity tests between legacy `buildNetwork` and the new internal `knn` builder for `jsd`, `l2`, `ip`, and both symmetrization modes.
  - Add tests for conversion from internal sparse buffers back to `arma::sp_mat`.
  - Add backed HDF5 tests for `/obsm` input and `/obsp` output.

- `actionet-python`
  - Add tests for `network_obsm_key` in `run_actionet`.
  - Add tests for the `k*nn` preflight refusal.
  - Add parity tests between legacy Python build path and the new fast path.
  - Add backed tests verifying that direct-to-disk network writes reopen correctly.

- `actionet-r`
  - Rebuild against the updated `libactionet`.
  - Add or run an integration test confirming `buildNetwork` output remains compatible with the existing R API and sparse matrix expectations.

### Assumptions and Defaults

- Preserve the existing R-facing `libactionet` network API in the initial rollout.
- Treat row-major dense input as a Python-only optimization detail, not a shared replacement ABI.
- Treat CSRGraph32 or similar as an internal/additive core representation, not the only public result type.
- Optimize `knn` first; keep `k*nn` guarded for large runs rather than rewriting it early.
- `actionet-r` adoption of the new fast path is optional and separate from the initial performance program.
