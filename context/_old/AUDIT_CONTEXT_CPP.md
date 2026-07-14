# Codebase Cleanup Audit — C++ Core (`libactionet`)

Use this document as standing context when auditing modules of the
`libactionet` C++ core. Pass it to an agent alongside the specific
module(s) being audited.

The companion document `context/AUDIT_CONTEXT.md` covers the Python
front-end. Audit the two sides separately using their respective context
files. Neither document should reason about the other side unless the
finding is a signature or contract mismatch that spans the boundary.

---

## Project Overview

ACTIONet is a multi-language computational biology toolkit for single-cell
multi-resolution analysis. `libactionet` is the C++ core backend, exposed
to users via two language bindings:

- **Python front-end** (`actionet-python`) via pybind11 — performance-first
- **R front-end** (`actionet-r`) via Rcpp — feature-complete reference

This audit targets the C++ core as vendored into the Python front-end at
`src/libactionet/` (git submodule, branch `dev-gpu`).

### Build Stack

- **CMake ≥ 3.19**, C++17
- **Runtime deps:** BLAS/LAPACK, HDF5 (C library), OpenMP
- **Vendored:** Armadillo (linear algebra), hnswlib (ANN), uwot (UMAP),
  StatsLib+GCEM (distributions), aarand (RNG), PCG (RNG), fastapprox
  (fast math), colorspace (color conversion). **PRIMME has been deleted
  from the vendored tree.**
- **Threading:** OpenMP only (hard requirement, no OFF option)
- **Platform:** Linux x86_64 (primary), macOS arm64/x86_64, no Windows
  native

---

## Motivation for This Audit

`dev-gpu` accumulated four overlapping change streams since diverging from
`dev`. Before GPU work re-lands, the C++ core needs a targeted cleanup
pass to remove residue from those streams and flag inconsistencies they
introduced.

### Change streams landed on `dev-gpu` (C++ scope)

1. **SVD path removal.** `svd_feng.{cpp,hpp}`, `svd_primme.{cpp,hpp}`,
   the vendored `src/extern/primme/` tree (~53k lines), and
   `cmake/ConfigurePRIMME.cmake` are all deleted. `ALG_FENG`, `ALG_PRIMME`,
   the Feng/PRIMME switch cases in `runSVD` / `runSVD_Operator`, and the
   `MatrixOperator::prefer_block_solver_for_irlb()` virtual hint (plus
   both backed overrides) are gone. `CMakeLists.txt` no longer includes
   `ConfigurePRIMME` or filters `svd_primme.cpp` from R builds. Two
   remaining SVD algorithm codes: `ALG_IRLB = 0`, `ALG_HALKO = 1`.

2. **64-bit extensions to surviving SVD paths.** `svdIRLB` (all three
   overloads — sparse, dense, operator) now calls
   `check_irlb_axis_dimensions` and threads dimensions through as `int`
   with explicit casts. `svdHalko` (both templated and operator overload)
   calls `check_halko_axis_dimensions`. Halko's verbose banner switched
   from `%d` to `%llu` for the dimension print. `svd_irbla.cpp` gained
   an explicit "64-bit contract" comment on `svdIRLB_core` documenting
   the sparse-nnz-64-bit-clean / per-axis-INT_MAX contract.

3. **Threading changes for cross-BLAS harmony.** Three pieces:
   - New inline kernel helper header
     `include/utils_internal/utils_small_dense.hpp` (~209 lines) with an
     `INLINE_DIMENSION_LIMIT = 128` policy and inline SIMD/OpenMP kernels
     for `copy`, `dot`, `scale`, `axpy`, `gemv`, `symv_upper`,
     `rank_one_update`, `gram`, `residual_product`, `frobenius_norm`,
     and `clamp_and_normalize_columns`.
   - `aa.cpp`, `simplex_regression.cpp`, and
     `utils_internal/utils_active_set.cpp` migrated from raw `cblas_*`
     calls to `small_dense::*` dispatchers.
   - `include/utils_internal/utils_parallel.hpp` gained
     `OuterParallelRegionScope` (thread-local depth marker) and rewired
     `get_num_threads_nested_safe` to use `omp_get_level()` +
     the new scope, replacing the old `omp_in_parallel()` check that
     Intel OpenMP reports incorrectly for one-thread serialized teams.
   - `action_decomp.cpp` opens an `OuterParallelRegionScope` inside its
     `#pragma omp parallel for` over `k`.
   - A new numerical policy header
     `include/utils_internal/utils_action_numeric_policy.hpp` centralizes
     all ACTION discrete-decision constants (simplex support tolerance,
     SPA tie tolerance, AA singular threshold, active-set tolerances,
     landmark proximity). `aa.cpp`, `spa.cpp`, `action_post.cpp`,
     `simplex_regression.cpp`, and `utils_active_set.cpp` were updated
     to reference the shared constants and the header is included from
     `include/utils_internal/utils_active_set.hpp`.

4. **AnnData >= 0.13 compatibility patches.** No direct C++ changes
   landed for this stream — the fix is entirely Python-side. Mention
   only if a finding needs to compare wrapper behavior against the C++
   backed-operator contract.

### Audit goals

1. Remove residue from stream (1): stale includes, comments, TODO items,
   or dispatch defaults that still name Feng / PRIMME. Confirm the R
   wrappers in `wrappers_r/wr_decomposition.cpp` are noted as needing an
   out-of-tree patch and no in-tree code depends on them.
2. Verify stream (2): the `INT_MAX` guards are consistent across IRLB
   and Halko (both the templated and operator overloads), their error
   messages agree in shape, and the empty-matrix short-circuit (`m<2 ||
   n<2`) still fires before the guard.
3. Verify stream (3): all four sub-changes are internally consistent and
   the new inline-kernel and threading primitives have call sites that
   actually exercise them. Confirm no cblas call site was left behind
   in a hot path that should have been migrated. Confirm the numerical
   policy constants are the single source of truth (no duplicated
   literal `1e-6`, `1e-5`, `1e-16`, `1e-3`, `1e-7`, `1e-10` in ACTION
   sources).
4. Flag coupling hotspots and API drift that will block the GPU
   `ExecutionPolicy` / `ComputeBackend` insertion described in
   `plans/GPU_BACKEND_PLAN.md`. Especially in `decomposition/` and the
   `MatrixOperator` interface after the block-solver-hint removal.

### Known Debt (from `TODO.md`, still open)

- **Legacy `arma::field` vs typed structs** — dual return system across
  frontends, inconsistent return types. `svd_main.hpp` currently
  exposes `SVDResult` and `arma::field<arma::mat>` via the one-way
  `svdResultFromField` bridge. The reverse-direction
  `svdFieldFromResult` helper was deleted by the 2026-07-14 (part 2)
  cleanup pass (see below).
- **Undocumented C++ interface** — many public headers still lack usage
  documentation.
- **No formal test infrastructure** — no `test/` directory, no unit
  tests. Testing lives in `actionet-python/tests/`.
- **`actionet-r` Feng/PRIMME cleanup patch is not yet applied.** The
  submodule's `wrappers_r/wr_decomposition.cpp` still has Feng/PRIMME
  references at lines 126–144 and 162–164. These are documented as
  reference-only copies in the submodule's own TODO; do **not**
  recommend editing them here.

---

## Cleanup Pass Status (2026-07-14, part 2 — libactionet + pybind11 audit)

A follow-up audit driven by
`.cursor/plans/libactionet_cleanup_plan_fd9dcfd0.plan.md` covered the
C++ core and the pybind11 wrappers. Its recommendations were
implemented on `dev-gpu` on 2026-07-14 (uncommitted working tree,
staged to ship alongside the earlier Python-side pass and the
`0.3.0 -> 0.4.0` version bump). All 547 Python tests pass after the
edits (1 skipped). Subsequent audit runs should treat the items below
as **already resolved** and focus on new drift.

Landed cleanup items in `libactionet` (C++ core):

1. **[C-1] Halko operator-overload banner** in
   `src/libactionet/src/decomposition/svd_halko.cpp:151` now prints
   `%llu` with `unsigned long long` casts (was `%d`), matching the
   templated overload's Stream-2 format parity.
2. **[T-2a] Shared SVD axis-dimension guard.** New
   `check_svd_axis_dimensions(rows, cols, label)` in
   `include/utils_internal/utils_decomp.hpp` (+ `.cpp`) throws
   `std::overflow_error` with a unified message shape. Called from
   both `svdIRLB` (all three overloads — labels `"svdIRLB (sparse)"`,
   `"svdIRLB (dense)"`, `"svdIRLB (operator)"`) and `svdHalko` (both
   templated and operator overloads). Retires the previous
   `check_irlb_axis_dimensions` and `check_halko_axis_dimensions`
   local helpers.
3. **[T-2b] Shared Halko `dim` clamp.** New `clamp_halko_dim(rows,
   cols, dim&)` helper in `utils_decomp` enforces `dim + 2 <=
   min(rows, cols)` and `dim >= 1`; called from both Halko overloads.
   The inline clamp bodies (previously at
   `svd_halko.cpp:37-41` and `svd_halko.cpp:140-143`) are gone.
4. **[C-3] Dead `svdFieldFromResult` deleted** from
   `include/decomposition/svd_main.hpp:52-59`. The header comment
   updated from "field ↔ struct" to "field → struct".
5. **[C-5] `reduce_kernel.hpp:132`** now uses the `ALG_IRLB` symbolic
   constant (was bare literal `svd_alg = 0`), matching the sibling
   default at line 95.
6. **[T-3a] Unknown SVD algorithm codes now throw.** New helper
   `throw_unknown_svd_algorithm(algorithm, label)` in
   `src/libactionet/src/decomposition/svd_main.cpp` throws
   `std::invalid_argument`. Applied to both `runSVD` (in-memory) and
   `runSVD_Operator`; the `default:` fallthrough that silently
   executed IRLB is gone. The `default_max_it` switch was also
   restructured to remove its `IRLB default:` fallthrough. This is a
   **breaking change** for direct C++/R callers passing `algorithm=2`
   (retired Feng) or `algorithm=3` (retired PRIMME); Python callers
   are unaffected because `validate_python_svd_algorithm` already
   rejects those codes at the binding boundary. Coordinate with the
   `actionet-r` out-of-tree Feng/PRIMME patch (still tracked in
   `src/libactionet/TODO.md`).
7. **[T-3c] Nested-safe thread migration.** The following sites moved
   from `get_num_threads` to `get_num_threads_nested_safe`:
   - `src/io/backed_h5ad/backed_sparse_matrix_operator.cpp` (5 sites:
     lines 515, 550, 660, 696, 858)
   - `src/io/backed_h5ad/backed_dense_matrix_operator.cpp:188`
     (`log1p` threads)
   - `src/utils_internal/utils_stats.cpp`
   - `src/utils_internal/utils_matrix.cpp`

   The remaining `get_num_threads` sites in `annotation/specificity`,
   `tools/{autocorrelation,enrichment,xicor,guide_calling}`,
   `network/network_diffusion`, and `visualization/*` were **left
   as-is**: they are entry points invoked outside the ACTION outer
   parallel loop (which uses `OuterParallelRegionScope` at
   `action_decomp.cpp:41`) and do not need the nested guard. The
   ACTION inner path (`spa`, `aa`, `simplex_regression`) already uses
   `get_num_threads_nested_safe`.

Landed cleanup items in pybind11 wrappers (`src/actionet/bindings/`):

8. **[C-7] GIL release on backed orthogonalization** in
   `wp_decomposition.cpp`: `orthogonalizeBatchEffect_Operator` and
   `orthogonalizeBasal_Operator` now wrap their C++ calls in
   `py::gil_scoped_release` scoped blocks, matching the six
   sparse/dense siblings.
9. **[T-2c] Standardized C-order returns.** `run_svd_sparse`,
   `run_svd_dense`, and `perturbed_svd` (all in
   `wp_decomposition.cpp`) now return `u` and `v` via
   `arma_mat_to_numpy_c` (was `arma_mat_to_numpy`, Fortran-order).
   This matches `svd_to_dict` in `wp_io.cpp` and aligns all pybind
   SVD entry points on the C-order convention documented in
   `wp_utils.cpp:244-254`.
10. **[T-2d] Scoped-block GIL release** across four sites in
    `wp_annotation.cpp` (previously used bare `release; ...; acquire;`
    at lines ~252-254, 277-279, 368-370, 399-401).
11. **[T-2e] Shared `int_array_to_uvec<T>` template** in
    `bindings/wp_utils.h`: promoted from the private
    `int64_array_to_uvec` helper in `wp_io.cpp:43-60`, parameterized on
    element type, enforces the non-negative invariant, and now used
    across `wp_action.cpp`, `wp_annotation.cpp` (4 label-unpacking
    sites), and `wp_network.cpp` (`fixed_labels_vec`). `wp_io.cpp`'s
    `int64_array_to_uvec` is now a thin delegation to the shared
    template.
12. **[T-3b] `run_action` `tol` divergence documented.** A
    cross-referencing comment block was added above `decomp_action` in
    `wp_action.cpp` explaining the intentional 3-way default
    divergence (`C++ 1e-6`, pybind `1e-16`, Python wrapper `1e-100`).
    The pybind default is effectively unused because the Python
    wrapper (`run_action.py`) always passes an explicit `tolerance`;
    the C++ default is what the R wrapper consumes. Defaults were
    **not** aligned — instead the audit added the comment so future
    readers do not treat the divergence as a bug.

Items intentionally **not** changed during the pass:

- **[C-4] `#include <stdexcept>` in `svd_main.cpp`** was kept because
  T-3a's new `throw_unknown_svd_algorithm` helper needs it.
- **The `arma::field` dual return system** — full retirement is
  tracked as a larger multi-repo effort in `TODO.md`.
- **Halko `default_max_it=5`** — correct for real single-cell data;
  test-side fix in the 2026-07-14 (part 1) pass covers the synthetic
  parity case.
- **`wrappers_r/wr_decomposition.cpp` Feng/PRIMME residue** —
  out-of-tree, tracked separately.
- **`get_num_threads` sites in non-nested entry points** — left as-is
  per the T-3c call-graph analysis (see item 7 above).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Python API  (actionet-python: src/actionet/*.py)                     │
├──────────────────────────────────────────────────────────────────────┤
│  pybind11 Wrappers  (actionet-python: src/actionet/bindings/*.cpp)    │
│  Audited by context/AUDIT_CONTEXT.md, not this document               │
├──────────────────────────────────────────────────────────────────────┤
│  libactionet Public API  (include/*.hpp, include/<module>/*.hpp)      │
│  Contract consumed by both Python and R bindings                      │
├──────────────────────────────────────────────────────────────────────┤
│  libactionet Implementation  (src/<module>/*.cpp)                     │
├──────────────────────────────────────────────────────────────────────┤
│  Vendored / External  (include/extern/, src/extern/)                  │
│  DROP-IN. Must not be modified. (PRIMME deleted this branch.)         │
└──────────────────────────────────────────────────────────────────────┘
```

### Umbrella Header

`include/libactionet.hpp` defines the public API surface. All symbols in
headers included from this file are accessible through the `actionet`
namespace and visible to language bindings.

### Configuration Header

`include/libactionet_config.hpp` configures:
- Armadillo build mode (direct BLAS linking, `ARMA_64BIT_WORD` force-defined)
- StatsLib integration
- R vs standalone build dispatch (`LIBACTIONET_BUILD_R`)
- Printf/flush macros per build mode

---

## Module Map — Source & Headers

Line counts are approximate and predate this branch's changes; use them as
scope hints, not exact figures.

### Core Modules

| Module | Notable source files (`src/`) | Notable headers (`include/`) | Role | Change streams |
|--------|-------------------------------|------------------------------|------|----------------|
| **decomposition/** | `svd_main`, `svd_halko`, `svd_irbla`, `orthogonalization` | `svd_main`, `svd_halko`, `svd_irbla`, `orthogonalization`, `matrix_operator` | SVD algorithms, matrix operators, batch orthogonalization | (1), (2) |
| **action/** | `action_main`, `action_decomp`, `action_post`, `aa`, `spa`, `reduce_kernel`, `simplex_regression` | (matching) | Archetypal Analysis, ACTION decomposition, kernel reduction | (3) |
| **network/** | `build_network`, `network_diffusion`, `network_measures`, `label_propagation` | `build_network`, `build_network_core`, `hnsw_imp`, `hnsw_jensen_shannon`, `network_diffusion`, `network_measures`, `label_propagation` | kNN graph (HNSW), diffusion, centrality, clustering | none |
| **annotation/** | `specificity`, `marker_stats` | (matching) | Gene specificity scores, marker statistics | none |
| **io/** | `create_backed_operator`, `backed_dense_matrix_operator`, `backed_sparse_matrix_operator` | (matching) | HDF5-backed chunked matrix I/O | (1) via `matrix_operator.hpp` hint removal |
| **tools/** | `matrix_transform`, `matrix_aggregate`, `autocorrelation`, `enrichment`, `mwm`, `xicor`, `guide_calling` | (matching) | Matrix ops, stats, guide calling | none |
| **visualization/** | `layout_network`, `uwot_actionet`, `color_map` | `layout_network`, `uwot_actionet`, `color_map`, `UmapFactory`, `UwotArgs`, `OptimizerArgs`, `find_ab` | UMAP layout, node coloring | none |
| **utils_internal/** | `utils_matrix`, `utils_stats`, `utils_decomp`, `utils_active_set` | `utils_matrix`, `utils_stats`, `utils_decomp`, `utils_active_set`, `utils_parallel`, **`utils_small_dense`** *(new)*, **`utils_action_numeric_policy`** *(new)* | Internal numeric/threading helpers | (3) |

### Deleted this branch (do not re-audit)

- `src/decomposition/svd_feng.cpp` and `include/decomposition/svd_feng.hpp`
- `src/decomposition/svd_primme.cpp` and `include/decomposition/svd_primme.hpp`
- `src/extern/primme/` (entire vendored tree, ~53k LOC)
- `cmake/ConfigurePRIMME.cmake`

### Vendored External Code (not auditable, listed for context)

| Component | Location | Role |
|-----------|----------|------|
| Armadillo | `include/extern/armadillo/` | Linear algebra (header-only) |
| hnswlib | `include/extern/hnswlib/` | Approximate nearest neighbors |
| uwot | `include/extern/uwot/` | UMAP optimization |
| StatsLib + GCEM | `include/extern/StatsLib/`, `include/extern/gcem/` | Statistical distributions |
| colorspace | `src/extern/colorspace.cpp` + header | Color conversion |
| aarand / PCG | `include/extern/aarand/`, `include/extern/pcg/` | Random number generation |
| fastapprox | `include/extern/fastapprox/` | Fast approximate math |
| RcppPerpendicular | `include/extern/RcppPerpendicular.h` | R build helper |
| convert_seed | `include/extern/convert_seed.h` | Seed conversion utility |

### R Wrappers (`wrappers_r/`)

Reference Rcpp wrappers. Primary R package lives in `actionet-r`. These
are kept as reference and for R-mode builds but are **not compiled** in
the Python build path.

- `wr_action.cpp`, `wr_annotation.cpp`, `wr_decomposition.cpp`,
  `wr_experimental.cpp`, `wr_network.cpp`, `wr_tools.cpp`,
  `actionet_r_config.h`.
- `wr_decomposition.cpp` still contains stale references to Feng
  (`algorithm=2`) and PRIMME (`algorithm=3`) at lines 126–144 and 162–164.
  These are documented as reference-only copies in `TODO.md`. Do not
  recommend editing them; the fix belongs in the out-of-tree
  `actionet-r` package.

### Excluded Code (`_EXCLUDE/`)

Not compiled. Contains prior-cleanup residue (`_old_extern/`,
`experimental/umappp+subpar`, `extern/HDBSCAN`, `extra/`, unused CMake
modules). Do not recommend deleting without concrete evidence of
obsolescence.

---

## CMake Build System

### Root `CMakeLists.txt`

- Static library target (`actionet`)
- GLOB_RECURSE per module (no manual file listing)
- `include(ConfigurePRIMME)` and `CONFIGURE_PRIMME(actionet)` have been
  removed on this branch. Verify no remnants.
- The R-build filter that excluded `svd_primme.cpp` is also gone.
- Links: BLAS/LAPACK, HDF5, OpenMP
- C++17, PIC, export compile commands
- Thread count: auto-detect minus 2 (≥6 cores)

### CMake Modules (`cmake/`)

| Module | Role |
|--------|------|
| `ConfigureApple.cmake` | Apple-specific arch flags |
| `ConfigureBLAS.cmake` | BLAS/LAPACK vendor detection |
| `ConfigureOpenMP.cmake` | OpenMP runtime discovery |
| `ConfigureR.cmake` | R-mode build (Rcpp, RcppArmadillo) |

`ConfigurePRIMME.cmake` was deleted this branch.

---

## Binding Interface (pybind11 side, in `actionet-python`)

Audited separately by `context/AUDIT_CONTEXT.md`. Reference only:

- Each C++ module has a corresponding `wp_<module>.cpp`.
- The Python entry points call `validate_python_svd_algorithm` inline in
  `wp_utils.h` and reject any algorithm id outside `{ALG_IRLB=0,
  ALG_HALKO=1}`. When auditing `decomposition/` in C++, treat this as
  the observed public policy for the Python build.

---

## Key Architectural Decisions (Non-Negotiable)

Recorded in `context/DECISIONS.md` (Python-side) and the submodule's own
`src/libactionet/context/DECISIONS.md`. Must not be re-litigated during
audit.

1. **OpenMP is a hard requirement** — no OFF option; used pervasively.
2. **Multi-repo structure** — `libactionet` is standalone; bindings live
   in front-end repos.
3. **Python is performance-first** — R is feature-complete reference.
4. **Public SVD surface: IRLB and Halko.** Feng and PRIMME are deleted.
   `runSVD_Operator`'s default is `ALG_HALKO`; `runSVD` (in-memory)
   defaults to `ALG_IRLB`.
5. **`MatrixOperator::prefer_block_solver_for_irlb()` is gone.** Backed
   operators requesting `ALG_IRLB` unconditionally use the honest
   `svdIRLB(MatrixOperator&, ...)` overload. Do not reintroduce a
   fast-path hint.
6. **BLAS policy for ACTION.** Shape-specialized inline kernels below
   `INLINE_DIMENSION_LIMIT = 128` (min dimension); larger matrices route
   through the configured BLAS. No public API change, no BLAS-vendor
   detection, no runtime BLAS-thread guard, no import warnings, no
   diagnostic Python API. C/H comparisons use `rtol=1e-8`, `atol=1e-10`
   across backends and thread counts.
7. **ACTION numerical decision stability.** Simplex coefficient support
   is defined as strictly greater than `1e-6` (see
   `utils_action_numeric_policy.hpp`). C/H matrices are compared
   numerically, not bitwise. Seeded `reduce_kernel` output is not
   promised bitwise-identical across BLAS builds.
8. **Breaking changes allowed** if they substantially improve
   performance/usability/reproducibility.
9. **Exported APIs are contracts** — wrappers in both R and Python
   consume them.
10. **`extern/` is drop-in** — must not be modified.
11. **CUDA 12.2+ floor, Ampere+ only, Linux x86_64** for future GPU work.
    R-facing GPU API is deferred.
12. **Operator-backed SVD stays CPU-only** in the near-term GPU plan —
    callback overhead dominates; GPU SVD needs its own product backend.

---

## GPU Support Context (Upcoming)

From `plans/GPU_BACKEND_PLAN.md`:

| Phase | Scope | GPU Targets |
|-------|-------|-------------|
| 1 | SVD (Halko-style randomized SVD, in-memory dense/sparse) | `decomposition/` |
| 1-hardening | Error taxonomy, benchmarks, CI | Cross-cutting |
| 1b | Disk-backed GPU SVD (first-class v1 requirement) | `decomposition/`, `io/` |
| 2 | Network construction (cuVS + JSD rerank) | `network/` |
| 3 | ACTION (AA + SPA on GPU dense BLAS) | `action/` |

Integration will require:

- `ExecutionPolicy` struct threaded through entry points.
- `ComputeBackend` enum for dispatch (do **not** reuse `ALG_*` codes
  for backend selection — algorithm and backend are orthogonal).
- Clean module boundaries for conditional compilation.
- No global state in hot paths.
- Explicit device-ownership boundary; do not treat `MatrixOperator` as
  the device-memory contract.

---

## Audit Execution Instructions

When auditing a module, apply the `/codebase-cleanup-audit` skill with
these repo-specific instructions.

### Scope

- Audit the specified C++ module(s): both `src/libactionet/src/<module>/`
  and `src/libactionet/include/<module>/` together.
- Consider interactions with:
  - The umbrella header (`include/libactionet.hpp`)
  - The config header (`include/libactionet_config.hpp`)
  - `utils_internal/` (shared helpers, including new
    `utils_small_dense.hpp` and `utils_action_numeric_policy.hpp`)
  - The Python binding wrapper (only for boundary contract checks — the
    wrapper itself is audited by `context/AUDIT_CONTEXT.md`)
  - The corresponding R wrapper (`wrappers_r/wr_<module>.cpp`) only to
    confirm it is out-of-scope for edits, not as an audit target.
- Do NOT audit vendored `extern/` code.
- Do NOT audit deleted files (see "Deleted this branch"); confirm only
  that no references remain.

### Priorities (ordered)

1. **Confirmed residue from the four change streams**:
   - **Stream (1) — SVD removal.** Any lingering include of
     `svd_primme.hpp` or `svd_feng.hpp`, any `ALG_FENG` / `ALG_PRIMME`
     / `LIBACTIONET_BUILD_R`-guarded PRIMME block, any residual
     `prefer_block_solver_for_irlb` reference, any dead switch case,
     dead comment (`// PRIMME is preferred for large sparse matrices`),
     or dead CMake logic. Also verify `svd_main.hpp` is the single
     source of truth for `ALG_IRLB` / `ALG_HALKO` (no duplicate
     literals in other headers).
   - **Stream (2) — 64-bit SVD guards.** Verify guard placement is
     symmetric between IRLB and Halko across all overloads (sparse,
     dense, operator). Verify error message shape is consistent. Verify
     the `m<2 || n<2` short-circuit runs before the guard where the
     historical code returned early on tiny inputs.
   - **Stream (3) — threading + numerical policy.** Confirm the new
     inline kernels have call sites that actually reach them (i.e. no
     leftover `cblas_dgemv` / `cblas_dsymv` / `cblas_ddot` /
     `cblas_dcopy` / `cblas_daxpy` / `cblas_dscal` / `cblas_dger` in
     `aa.cpp`, `simplex_regression.cpp`, `utils_active_set.cpp`).
     Confirm no ACTION numerical constant is duplicated as a literal
     outside `utils_action_numeric_policy.hpp` (grep for `1e-6`,
     `1e-5`, `1e-3`, `1e-7`, `1e-10`, `1e-16` in `src/action/`,
     `src/utils_internal/`). Confirm `OuterParallelRegionScope` is
     opened at every coarse-grained ACTION parallel region that can
     invoke `get_num_threads_nested_safe`, and confirm all remaining
     `get_num_threads(...)` sites in nested-eligible functions have
     been evaluated (this is likely the largest open finding).
2. **API and contract drift**: `arma::field` vs typed struct returns
   (`SVDResult`, `PerturbedSVDResult`). Header signatures vs
   implementation signatures. Docstring `@param` order vs actual
   parameter order.
3. **Coupling hotspots that will block GPU dispatch insertion.** Global
   state in hot paths, hidden cross-module dependencies, non-obvious
   threading assumptions, `MatrixOperator` subclass expectations that
   would leak into a device backend.
4. **Documentation gaps** on public headers: undocumented public APIs,
   missing parameter semantics, unclear lifetime/ownership contracts.
5. **Consolidation opportunities** (duplicate logic, repeated patterns
   that could be factored). Keep this bounded.
6. **Easy efficiency wins** (unnecessary copies, redundant allocations,
   cache-unfriendly access patterns). Keep this bounded.

### Report Format

Use the findings-first report template from the skill. Include:

- File paths and line numbers (relative to `src/libactionet/`).
- Severity classification:
  - **Confirmed issue** — direct evidence of bug, mismatch, or dead code.
  - **Likely risk** — strong signal, needs one more verification step.
  - **Open question** — ambiguous ownership or intent, needs clarification.
- Which change stream (1–3) the finding relates to, or "pre-existing"
  if it predates this branch.
- Concrete evidence for each finding.
- Quick wins vs. larger follow-ups.

### Guardrails

- Audit only. Do not edit code.
- Do not re-litigate decisions in `context/DECISIONS.md`.
- Do not recommend modifying vendored `extern/` code.
- Do not recommend editing `wrappers_r/` files. The `wr_decomposition.cpp`
  Feng/PRIMME residue is a known out-of-tree issue tracked by
  `TODO.md`.
- Do not recommend deleting `_EXCLUDE/` or `wrappers_r/` without
  concrete evidence of obsolescence.
- Treat `include/libactionet.hpp` exports as the public API contract.
- Both R and Python bindings consume the public headers; when
  considering signature or default-value changes, mention both.

---

## Suggested Audit Order

Audit in dependency order (leaves first, roots last). Change-stream
column indicates which streams touched the module.

| Phase | C++ Module | Change streams | Rationale |
|-------|-----------|----------------|-----------|
| 1 | `utils_internal/` + `libactionet_config.hpp` + `blas_deps.hpp` | (3) | Foundational; new `utils_small_dense.hpp`, new `utils_action_numeric_policy.hpp`, updated `utils_parallel.hpp` |
| 2 | `decomposition/` | (1), (2) | Core SVD; largest deletion surface (~2500 lines removed from first-party + ~53k lines removed from vendored PRIMME); new 64-bit guards; first GPU target |
| 3 | `io/backed_h5ad/` | (1) | Backed operator stack; `prefer_block_solver_for_irlb` removed from both dense/sparse backed operators |
| 4 | `action/` | (3) | ACTION algorithm; largest set of small-dense kernel call-site migrations; numerical policy centralization |
| 5 | `network/` | none | Regression sweep only; verify no nested-parallel oversubscription regression |
| 6 | `annotation/` | none | Regression sweep only |
| 7 | `tools/` | none | Regression sweep only; `guide_calling.cpp` is 970 lines but unchanged |
| 8 | `visualization/` | none | Regression sweep only |
| 9 | Build system (`CMakeLists.txt` + `cmake/`) | (1) | Cross-cutting; verify PRIMME removal is complete and R-build filter cleanup is consistent |

---

## Phase-Specific Focus Areas

### Phase 1: `utils_internal/` + config

- **`utils_small_dense.hpp` (new, ~209 lines):** verify the inline
  kernels match the semantics of the cblas calls they replace, including
  transpose vs no-transpose, `beta=0` vs `beta!=0` handling in
  `prepare_output`, symmetric-upper access pattern in `symv_upper`, and
  the `gram(...)` symmetric-fill loop. Verify `use_inline_kernel` is
  applied consistently — `residual_product` requires *both* factors to
  be non-small before falling back, which is correct but non-obvious;
  document or note.
- **`utils_action_numeric_policy.hpp` (new, ~26 lines):** verify every
  constant has a call site. Confirm the constants are `inline constexpr
  double` (already are) and that the header is included via
  `utils_active_set.hpp` so downstream code does not need to include
  both.
- **`utils_parallel.hpp` (updated):** verify
  `OuterParallelRegionScope`'s thread-local depth counter is correctly
  incremented/decremented in the destructor even under exceptions;
  confirm the fallback branch when `_OPENMP` is not defined still
  respects the depth counter (`omp_get_level()` is only in the
  `_OPENMP` branch — do not remove).
  Check whether `get_num_threads_nested_safe` has more call sites that
  should be migrated from `get_num_threads` (see "Priorities" — this is
  the highest-yield open finding).
- **`utils_active_set.cpp` (updated):** all cblas calls migrated;
  confirm no cblas symbol remains (grep for `cblas_` in
  `src/utils_internal/utils_active_set.cpp`).
- **`utils_active_set.hpp` (updated):** default values now reference
  `numeric_policy` constants; verify no other header re-declares the
  defaults (`activeSet_arma`, `activeSetS_arma`) with literal values.
- **`libactionet_config.hpp`:** confirm `ARMA_64BIT_WORD` is
  force-defined and no build mode disables it — the entire 64-bit SVD
  contract depends on this.
- **`blas_deps.hpp`:** verify no PRIMME-related includes remain.

### Phase 2: `decomposition/`

- **`svd_main.{cpp,hpp}`:** the `ALG_*` enum and the `default_max_it`
  switch now cover only `ALG_HALKO` and `ALG_IRLB`. Confirm no dead
  `#if !defined(LIBACTIONET_BUILD_R)` blocks remain. Confirm the
  operator-overload default is `ALG_HALKO` per the header comment.
- **`svd_halko.cpp` (updated):** verify `check_halko_axis_dimensions`
  runs after the `m<2 || n<2` short-circuit for the operator overload
  (both are present) but before any allocation. Verify the printf
  format change (`%d` → `%llu` with `unsigned long long` cast) is
  correctly matched.
- **`svd_irbla.cpp` (updated):** verify the "64-bit contract" comment
  on `svdIRLB_core` matches the actual behavior — specifically that the
  `int` narrowing at the public overloads is guarded by
  `check_irlb_axis_dimensions` and that the sparse matvec path
  actually routes through `arma::sp_mat::operator*(vec)` (which uses
  64-bit uword under `ARMA_64BIT_WORD`). Verify the header comment `//
  IRLB implementation - Note: PRIMME is preferred for large sparse
  matrices` was correctly updated to remove the PRIMME reference.
- **`matrix_operator.hpp` (updated):** confirm
  `prefer_block_solver_for_irlb()` is gone from the base class *and*
  from both backed subclasses in `io/backed_h5ad/`. Confirm the header
  comment now describes the CPU host-matrix contract without referring
  to PRIMME.
- **`orthogonalization.cpp` (unchanged this branch, ~325 lines):**
  regression sweep only. Confirm no dead include of removed headers.
- **`svd_halko.hpp`, `svd_irbla.hpp`:** signature parity with
  implementation.
- **Known open bug** (from
  `plans/investigate_svd_parity_test_failures.md`): Halko's in-memory
  `default_max_it` is `5`, which is insufficient for the test matrix
  in `test_inmemory_sparse_parity_irlb_vs_halko`. This is not directly
  a residue of the four streams but became visible after the removal.
  Flag as a `Confirmed issue` and reference the plan for the
  recommended `max_it` bump (5 → 10, verify against
  `docs/svd_algorithm_benchmark.md`).

### Phase 3: `io/backed_h5ad/`

- **`backed_dense_matrix_operator.{cpp,hpp}` and
  `backed_sparse_matrix_operator.{cpp,hpp}`:** confirm
  `prefer_block_solver_for_irlb()` override has been removed from both
  subclasses (the diff shows one-line removals in each header). Confirm
  no other virtual on `MatrixOperator` was silently removed.
- **`create_backed_operator.{cpp,hpp}`:** unchanged this branch;
  regression sweep only. Confirm the factory does not depend on any
  removed hint.
- **Chunk-size and thread-count handling** in
  `backed_sparse_matrix_operator.cpp` (~1048 lines) is unchanged but
  large; audit for `get_num_threads` sites that could benefit from
  `get_num_threads_nested_safe` when the backed operator is invoked
  from within an outer ACTION parallel loop. This is a likely-risk
  category.

### Phase 4: `action/`

- **`aa.cpp` (updated):** confirm the migration to `small_dense::*` is
  complete and correct. Original code used `arma::dot(h, h)` for a
  vector dot; the migration uses `small_dense::dot(...)` with
  `inline_kernel=true` unconditionally at that site — verify this is
  intentional (a scalar dot product on a small vector is a natural
  inline case).
- **`spa.cpp` (updated):** confirm the SPA tie tolerance constant
  (`eps` → `numeric_policy::spa_relative_tie_tolerance`) is the only
  numerical change. No BLAS migration was needed for SPA.
- **`action_post.cpp` (updated):** confirm three constants were
  centralized: landmark proximity (`epsilon = 1e-3` →
  `numeric_policy::landmark_proximity_tolerance`), C-landmark support
  (`c > 0` → `c > numeric_policy::simplex_coefficient_support_tolerance`),
  and membership-count threshold (`C_stacked > 1e-6` →
  `C_stacked > numeric_policy::simplex_coefficient_support_tolerance`).
  Confirm no other predicate in this file uses a bare `0` or `1e-6` for
  simplex support.
- **`action_decomp.cpp` (updated):** verify
  `OuterParallelRegionScope outer_parallel_scope;` is declared inside
  the `#pragma omp parallel for` loop body (not outside it) so each
  thread has its own scope. Also confirm the scope destructor runs
  before the loop iteration ends.
- **`simplex_regression.cpp` (updated):** confirm `nthreads` now uses
  `get_num_threads_nested_safe(ncols)`. Confirm both
  `lambda2`/`epsilon` constants come from `numeric_policy` and are
  `constexpr`.
- **`reduce_kernel.cpp` (unchanged this branch, ~244 lines):**
  regression sweep only; watch for stale references to removed
  algorithms.
- **`action_main.cpp`, `action_decomp.cpp` (only latter touched):**
  verify no Python-side thread management state has leaked.

### Phase 5–8: regression sweep only

Streams (1)–(3) did not directly touch these modules. Audit only for:

- **Confirmed issues** from streams (1)–(3) (should be rare).
- Pre-existing `TODO.md` items whose fix is a quick win.
- Places where `get_num_threads(...)` is called inside a function that
  can be reached from within an outer ACTION parallel region and
  should probably use `get_num_threads_nested_safe(...)` instead.
  Notable candidates from the current tree:
  `annotation/specificity.cpp`, `io/backed_h5ad/*.cpp`,
  `tools/{enrichment,xicor,autocorrelation,guide_calling}.cpp`,
  `network/network_diffusion.cpp`, `visualization/*.cpp`,
  `utils_internal/utils_stats.cpp`, `utils_internal/utils_matrix.cpp`.
  Flag as `Likely risk`, not `Confirmed issue`, until each is verified
  by call-graph inspection.

Default to `No material cleanup or quality findings were confirmed in
the audited scope.` when nothing has drifted.

### Phase 9: Build system

- Confirm `CMakeLists.txt` no longer references `ConfigurePRIMME`,
  `CONFIGURE_PRIMME`, or the R-build `svd_primme.cpp` filter.
- Confirm no `cmake/*.cmake` file references PRIMME.
- Confirm `include/blas_deps.hpp` and any BLAS/LAPACK link line does
  not carry PRIMME-only symbols.
- Verify HDF5 linking (`_HDF5_TARGET`) still resolves cleanly on both
  Linux and macOS.
- The existing "SKETCHY DEFENSIVE FIX" comment for HDF5 linking (if
  still present) is pre-existing; flag as a `TODO.md` note, not new
  work.

---

## Example Agent Prompt

```
Read context/AUDIT_CONTEXT_CPP.md for project context and audit instructions.

Audit Phase [N]: [module name]

Apply the /codebase-cleanup-audit skill to the following files:
- src/libactionet/src/<module>/*.cpp
- src/libactionet/include/<module>/*.hpp

Focus on: [phase-specific focus areas from the document]

Produce a findings-first report. Do not edit code.
```

---

## Cross-References

- `context/AUDIT_CONTEXT.md` — Python front-end audit context; use for
  anything under `src/actionet/`.
- `context/DECISIONS.md` — Python-side architectural decisions.
- `src/libactionet/context/DECISIONS.md` — submodule-side decisions;
  contains the C++-scope statements of SVD strategy, BLAS policy for
  ACTION, and ACTION numerical decision stability.
- `src/libactionet/context/AGENT_PLAYBOOK.md` — submodule agent
  guardrails.
- `context/_old/AUDIT_CONTEXT.md` and `context/_old/AUDIT_CONTEXT_CPP.md`
  — prior cleanup cycle; kept for reference only.
- `src/libactionet/TODO.md` — submodule TODO, tracks the pending
  out-of-tree `actionet-r` Feng/PRIMME patch.
- `TODO.md` — Python-side TODO.
- `plans/openblas_threading_fix_handoff.md` — final BLAS policy handoff.
- `plans/openblas_threading_and_odr_findings.md` — full diagnosis and
  the small-dense kernel remedy.
- `plans/investigate_svd_parity_test_failures.md` — three known open
  test failures and recommended fixes; Failure 1 is a Halko `max_it`
  issue that surfaces as a C++-side finding.
- `src/libactionet/plans/GPU_BACKEND_PLAN.md` — GPU roadmap and
  post-mortem.
- `docs/svd_algorithm_benchmark.md` — benchmark evidence for retired
  Feng and current auto-selection defaults.
