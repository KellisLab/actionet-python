## Cross-Repo Network Construction Optimization With R Compatibility

### Status Snapshot (2026-03-13, updated post-bugfix)

Completed in this pass:

- `libactionet`
  - new internal `buildNetworkCore(const float*, n, dim, params) → CSRGraph` entry point
  - **critical bugfix**: `symmetrize_to_csr` now sorts directed edges by unordered pair key
    `(min(u,v), max(u,v))` rather than `(src, dst)`; the prior sort interleaved forward and
    reverse edges for any node with k > 1 neighbours, so the accumulation loop never saw both
    directions together — with `mutual_edges_only=true` (the default) every edge was discarded
    and all networks were empty sparse matrices
  - fixed lossy `float` storage of HNSW neighbor labels in `k*nn`
  - `CSRGraph` uses `uint32_t` vertex IDs and `uint64_t` indptr offsets; the split keeps
    the dominant `indices` array at 4 bytes/entry while safely tracking nnz > 2 billion
  - HNSW space and index objects are now owned by an `HnswIndex` RAII struct — the prior
    per-call `SpaceInterface*` memory leak has been fixed
  - legacy `buildNetwork(const arma::mat&) → arma::sp_mat` contract is unchanged; it is
    now a thin shim that does one col-major-double → row-major-float32 pass then calls the core
- `actionet-python`
  - pybind binding calls `buildNetworkCore` directly from a NumPy float32 buffer (zero copy)
  - `CSRGraph` is converted directly to SciPy CSR; `int32` indices when safe, `int64` otherwise
  - `scipy_to_arma_sparse` now uses `forcecast` on the data array — float32 and integer
    SciPy matrices no longer throw at the binding boundary
  - `run_lpa` fixed-labels 1→0-index conversion now validates each value is ≥ 1 before
    subtracting; bad input raises a clear `RuntimeError` instead of silently underflowing
  - `build_network()` in `core.py` validates `np.isfinite(H)` before calling C++ — NaN/Inf
    values in the obsm matrix now raise a `ValueError` instead of producing undefined HNSW behavior
  - `src/libactionet` submodule synced to all `libactionet` changes above
- `actionet-r`
  - orphaned `C_buildNetworkFast` Roxygen block removed from `src/wr_network.cpp`; the stale
    block had been mis-attached to `C_runLPA` in the generated `RcppExports.R` — fixed by
    re-running `Rcpp::compileAttributes()`
  - `C_buildNetwork` wrapper now accepts `Rcpp::NumericMatrix` and calls `buildNetworkCore`
    directly; the `arma::mat` intermediate and `arma::fmat` staging copy are eliminated;
    the col-major double→row-major float32 conversion is inlined in the wrapper
  - `ef` default corrected from 50 to 200 to match all other callers and the `BuildNetworkParams` default
  - `Rcpp::compileAttributes()` re-run; `RcppExports.cpp` and `RcppExports.R` regenerated
  - `src/libactionet` submodule synced to all `libactionet` changes above
- validation-only test scaffolding used during implementation has been removed

Still pending:

- Python-side large-`k*nn` preflight / operator guardrails
- pipeline-level `network_obsm_key` support in `run_actionet()`
- backed `/obsm` reader + direct `/obsp` writer path
- any true large-N rewrite of adaptive `k*nn`

### Summary

The shared design remains:

- keep the legacy `libactionet` API used by R intact
- use an additive internal core path for performance work
- let Python call the core path directly with row-major `float32`
- treat large-scale `knn` as the primary optimization target
- keep `k*nn` behaviorally correct, but do not treat it as the large-N path

### Ordered Plan

| Order | Change | Repo(s) | Status | Notes |
| ----- | ------ | ------- | ------ | ----- |
| 1 | Large-N guardrails and pipeline knobs | `actionet-python` | Pending | Still needed for user-facing safety around `k*nn`. |
| 2 | Expose network input choice in pipeline | `actionet-python` | Pending | `build_network(..., obsm_key=...)` already exists, but `run_actionet(..., network_obsm_key=...)` is still not done. |
| 3 | New core internal builder with R-safe compatibility layer | `libactionet` | Completed 2026-03-13 | Correctness fixes, RAII HNSW ownership, 64-bit-safe indptr offsets, 32-bit vertex IDs. Critical empty-graph sort bug fixed 2026-03-13 (sort by unordered pair key, not directed src/dst). See type layout note below. |
| 4 | Python-only `float32` / row-major fast binding | `actionet-python` | Completed 2026-03-13 | Pybind calls `buildNetworkCore` directly; no Python transpose copy. |
| 5 | Direct sparse conversion for Python | `actionet-python` | Completed 2026-03-13 | Network path converts `CSRGraph` directly to SciPy CSR; `forcecast` on sparse input data. |
| 6 | Backed dense `obsm` reader and direct `obsp` writer | `libactionet` + `actionet-python` | Pending | Still the main missing piece for truly out-of-core network construction. |
| 7 | R handling of new core path | `actionet-r` | Completed 2026-03-13 | Completed 2026-03-13. C_buildNetwork now accepts Rcpp::NumericMatrix, inlines the col-major double→row-major float32 conversion, and calls buildNetworkCore + armaSpMatFromCSR directly. The arma::mat intermediate copy is eliminated. RcppExports regenerated. ef default corrected from 50 to 200. |
| 8 | Adaptive `k*nn` rewrite for large N | `libactionet` + wrappers | Deferred | Correctness is fixed; scalability rewrite is still intentionally out of scope. |

### Implemented Details

#### 1. `libactionet` core

Added / updated:

- `include/network/build_network_core.hpp`
- `include/network/hnsw_imp.hpp`
- `src/network/build_network.cpp`

Key behavior changes:

- Internal CSR representation is `CSRGraph`:
  - `CSRVertexIndex` = `uint32_t` for column indices — supports up to ~4.3 billion vertices,
    far beyond any foreseeable single-cell dataset; halves per-edge memory vs `arma::sp_mat`
  - `CSROffset` = `uint64_t` for `n`, `indptr`, and nnz counts — safely tracks cumulative
    edge offsets that can exceed 2 billion on very large dense graphs
  - `float32` edge weights throughout the hot path
- `CSRGraph32` alias retained for transitional callers
- HNSW index and space objects are owned by `HnswIndex` (RAII struct in `hnsw_imp.hpp`);
  the prior raw-pointer leak of `SpaceInterface*` per network construction call is fixed
- `k*nn` neighbor labels stored as `hnswlib::labeltype` (not `float`); prevents silent
  label truncation for vertex indices above 16,777,216
- non-mutual symmetrization matches legacy semantics exactly:
  - both directions present: `0.5 * (w_uv + w_vu)`
  - one direction present: `0.5 * w`
- mutual-only symmetrization still uses geometric mean and drops non-mutual edges
- **sort-bug fix**: edges in `symmetrize_to_csr` are sorted by unordered pair key
  `(min(u,v), max(u,v), src, dst)` rather than `(src, dst)`; the prior sort interleaved
  forward and reverse edges whenever a node had more than one neighbour, causing the
  accumulation loop to see each direction in isolation — with `mutual_edges_only=true`
  (the default) all edges were discarded, producing an empty sparse matrix for every call
- diagonal entries are always removed
- explicit range checks guard internal allocation sizes and `arma::sp_mat` conversion
- dead `AddWorker` struct (stale pre-refactor artifact, referenced `arma::fmat`) removed
  from `hnsw_imp.hpp`

Legacy API status:

- `actionet::buildNetwork(const arma::mat&)` signature is unchanged
- it now delegates to `buildNetworkCore` via a single col-major-double → row-major-float32
  conversion pass, eliminating the previous by-value `arma::mat` copy and `arma::fmat`
  staging copy
- R continues to receive `arma::sp_mat`

#### 2. `actionet-python`

Updated:

- `src/actionet/wp_network.cpp`
- `src/actionet/wp_utils.h`
- `src/actionet/wp_utils.cpp`
- `src/actionet/core.py`
- `src/libactionet/*` synced to the same `libactionet` changes

Current Python network path:

- Python passes `adata.obsm[obsm_key]` as row-major `float32` (via `np.ascontiguousarray`)
- NaN/Inf check (`np.isfinite`) runs before the C++ call; raises `ValueError` on bad input
- pybind calls `actionet::buildNetworkCore(ptr, n, dim, params)` directly — no `numpy_to_arma_mat` copy
- the returned `CSRGraph` is converted directly to SciPy CSR via `csr_graph_to_scipy`
- SciPy sparse indices use `int32` when n and nnz fit; `int64` otherwise
- `scipy_to_arma_sparse` (used by `run_lpa`, `compute_network_diffusion`, etc.) now accepts
  float32 and integer SciPy matrices via `forcecast`
- `run_lpa` fixed-labels 1→0 conversion validates each value is ≥ 1 before subtracting
- public `actionet.build_network(...)` signature is unchanged

#### 3. `actionet-r`

Updated:

- `src/wr_network.cpp`
- `R/RcppExports.R`
- `src/RcppExports.cpp`
- `src/libactionet/wrappers_r/wr_network.cpp` (mirror of `libactionet`)
- `src/libactionet/*` synced to the same `libactionet` changes

Current R network path:

- `C_buildNetwork` accepts `Rcpp::NumericMatrix H` (col-major double, k × n_cells)
- The wrapper inlines the col-major double → row-major float32 conversion that was
  previously buried inside the `actionet::buildNetwork` shim
- Calls `actionet::buildNetworkCore(ptr, n_points, dim, params)` directly
- Calls `actionet::armaSpMatFromCSR(g)` to produce the `arma::sp_mat` return value
- Eliminates: the `arma::mat` intermediate copy (k × n doubles) and the `std::vector<float>`
  owned by the legacy shim — the only allocation on the hot path is the single float32 buffer
  written in the wrapper's conversion loop
- The `ef` default was corrected from 50 to 200, matching `BuildNetworkParams` and all other callers
- Public R `buildNetwork()` in `network_tools.R` is unchanged
- `C_buildNetworkFast` export was added and then removed in the same pass; the final
  `src/wr_network.cpp` contains only `C_buildNetwork` as the network export
- The generated `RcppExports.R` and `RcppExports.cpp` are clean (no orphaned Roxygen blocks)

### CSR Type Layout and Graph Size Limits

The `CSRGraph` design uses a deliberate mixed-width layout:

| Field | Type | Width | Max value |
| ----- | ---- | ----- | --------- |
| vertex IDs in `indices` | `uint32_t` | 32-bit | ~4.3 billion cells |
| row pointers in `indptr` | `uint64_t` | 64-bit | ~1.8 × 10¹⁹ |
| edge weights in `data` | `float` | 32-bit | — |

The 32-bit vertex ID is the binding constraint. It covers all published single-cell datasets by
a factor of ~100 (the largest current references are in the tens of millions of cells). Upgrading
to 64-bit vertex IDs would double the size of the dominant `indices` array (4 bytes → 8 bytes per
edge) with no practical benefit at foreseeable dataset scales.

The 64-bit `indptr` is justified independently: for N = 50M cells and k = 15, nnz ≈ 1.5 billion,
which fits in `uint32_t` but leaves little headroom. Higher k or denser symmetrization can push
beyond `uint32_t` range, making 64-bit offsets the safe choice for indptr regardless of vertex ID
width.

### Validation Performed (most recent pass)

`libactionet`

- rebuilt in `cmake-build-llvm-arm64` with ninja; all translation units compiled cleanly
- sort-bug fix confirmed: `build_network` on 200-cell, 10-dim random input now produces
  a non-empty graph (nnz = 10 210, density ≈ 25.5 %, weights in [0.22, 0.96], |A−Aᵀ| = 0)
- parity test (12/12 cases across k\*nn, knn, all three metrics, both symmetrization modes)
  confirmed output identical between legacy `buildNetwork` and `buildNetworkCore` paths

`actionet-python`

- `./.venv/bin/python -m pip install --no-build-isolation -e .` — built cleanly
- `./.venv/bin/python -m pytest tests/test_sparse_conversion.py -v` — 4/4 passed
- runtime smoke tests confirmed:
  - NaN/Inf preflight raises `ValueError`
  - float32 SciPy sparse input accepted by `run_lpa`
  - `fixed_labels=0` raises `RuntimeError`; `fixed_labels=[1,2]` accepted

`actionet-r`

- `Rcpp::compileAttributes()` re-run; `C_buildNetwork` declaration in `RcppExports.cpp`
  updated from `const arma::mat&` to `Rcpp::NumericMatrix`; `ef` default corrected to 200
- `C_runLPA` no longer carries the orphaned `C_buildNetworkFast` Roxygen block

### Remaining Work

- add Python preflight refusal / warning for obviously unsafe `k*nn` sizes (plan item 1)
- expose `network_obsm_key` through `run_actionet()` (plan item 2)
- implement the backed `/obsm` → `/obsp` large-scale path (plan item 6)
- decide whether Python should eventually keep graph weights as `float32` end-to-end or
  intentionally upcast at the SciPy boundary (currently: float32 → float64 in `csr_graph_to_scipy`)
- if multi-million-cell adaptive graphs are still scientifically required, redesign `k*nn`
  rather than patching around its current scratch-space behavior (plan item 8)

### Working Assumptions

- preserve the R-facing `libactionet` network API
- optimize `knn` first; `k*nn` is correct but not the scalable path
- treat embedded `src/libactionet` checkouts in `actionet-python` and `actionet-r` as mirrors
  that must be kept in sync with master `libactionet` after each change
