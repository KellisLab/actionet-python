# Codebase Cleanup Audit — C++ Core (`libactionet`)

Use this document as standing context when auditing modules of the
`libactionet` C++ core. Pass it to an agent alongside the specific
module(s) being audited.

---

## Project Overview

ACTIONet is a multi-language computational biology toolkit for single-cell
multi-resolution analysis. `libactionet` is the C++ core backend, exposed to
users via two language bindings:

- **Python front-end** (`actionet-python`) via pybind11 — performance-first
- **R front-end** (`actionet-r`) via Rcpp — feature-complete reference

This audit targets the C++ core as vendored into the Python front-end at
`src/libactionet/` (git submodule).

### Build Stack

- **CMake ≥ 3.19**, C++17
- **Runtime deps:** BLAS/LAPACK, HDF5 (C library), OpenMP
- **Vendored:** Armadillo (linear algebra), PRIMME (eigensolver),
  hnswlib (ANN), uwot (UMAP), StatsLib+GCEM (distributions),
  aarand (RNG), PCG (RNG), fastapprox (fast math), colorspace (color
  conversion)
- **Threading:** OpenMP only (hard requirement, no OFF option)
- **Platform:** Linux x86_64 (primary), macOS arm64/x86_64, no Windows native

---

## Motivation for Audit

GPU support is the next major feature (see `plans/GPU_BACKEND_PLAN.md`). A
prior attempt (`dev-gpu-v2`) was scrapped due to structural issues (see
post-mortem in that document). Before the re-attempt, the C++ core needs
systematic review to:

1. Remove dead/redundant code that will complicate GPU dispatch insertion.
2. Identify API surface issues (return type inconsistencies, `arma::field`
   legacy patterns) that would propagate into GPU code paths.
3. Clarify module boundaries so GPU dispatch has clean integration seams.
4. Fix documentation/contract drift before adding new entry points.
5. Identify coupling hotspots that will block modular GPU extensions.
6. Flag known TODO items (`TODO.md`) that should be resolved first.

### Known Debt (from `TODO.md`)

- **Legacy `arma::field` vs typed structs** — dual return system across
  frontends, inconsistent return types
- **Undocumented C++ interface** — public headers lack usage documentation
- **No formal test infrastructure** — no `test/` directory, no unit tests

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Python API  (actionet-python: src/actionet/*.py)                     │
├──────────────────────────────────────────────────────────────────────┤
│  pybind11 Wrappers  (actionet-python: src/actionet/bindings/*.cpp)    │
│  ~2,900 lines; thin C++ → Python bridge                              │
├──────────────────────────────────────────────────────────────────────┤
│  libactionet Public API  (include/*.hpp, include/<module>/*.hpp)      │
│  ~3,500 lines; contract consumed by both Python and R bindings        │
├──────────────────────────────────────────────────────────────────────┤
│  libactionet Implementation  (src/<module>/*.cpp)                     │
│  ~9,100 lines of first-party code                                    │
├──────────────────────────────────────────────────────────────────────┤
│  Vendored / External  (include/extern/, src/extern/)                  │
│  ~270K lines (Armadillo, PRIMME, hnswlib, uwot, StatsLib, etc.)       │
│  DROP-IN. Must not be modified.                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Umbrella Header

`include/libactionet.hpp` defines the public API surface. All symbols in
headers included from this file are accessible through the `actionet`
namespace and visible to language bindings.

### Configuration Header

`include/libactionet_config.hpp` configures:
- Armadillo build mode (direct BLAS linking, 64-bit words)
- StatsLib integration
- R vs standalone build dispatch (`LIBACTIONET_BUILD_R`)
- Printf/flush macros per build mode

---

## Module Map — Source & Headers

All line counts are first-party code only (excludes `extern/`).

### Core Modules

| Module | Source (`src/`) | Headers (`include/`) | Total Lines | Role |
|--------|----------------|---------------------|-------------|------|
| **decomposition/** | `svd_main` (284), `svd_halko` (205), `svd_irbla` (299), `svd_feng` (244), `svd_primme` (262), `orthogonalization` (325) | `svd_main` (171), `svd_halko` (48), `svd_irbla` (51), `svd_feng` (48), `svd_primme` (42), `orthogonalization` (118), `matrix_operator` (129) | ~2,226 | SVD algorithms, matrix operators, batch orthogonalization |
| **action/** | `action_main` (57), `action_decomp` (61), `action_post` (152), `aa` (76), `spa` (69), `reduce_kernel` (244), `simplex_regression` (39) | `action_main` (42), `action_decomp` (39), `action_post` (63), `aa` (22), `spa` (28), `reduce_kernel` (147), `simplex_regression` (21) | ~1,060 | Archetypal Analysis, ACTION decomposition, kernel reduction |
| **network/** | `build_network` (809), `network_diffusion` (167), `network_measures` (110), `label_propagation` (47) | `build_network` (28), `build_network_core` (79), `hnsw_imp` (74), `hnsw_jensen_shannon` (79), `network_diffusion` (30), `network_measures` (23), `label_propagation` (23) | ~1,469 | kNN graph (HNSW), diffusion, centrality, clustering |
| **annotation/** | `specificity` (758), `marker_stats` (246) | `specificity` (122), `marker_stats` (92) | ~1,218 | Gene specificity scores, marker statistics |
| **io/** | `create_backed_operator` (77), `backed_dense_matrix_operator` (333), `backed_sparse_matrix_operator` (1048) | `create_backed_operator` (56), `backed_dense_matrix_operator` (123), `backed_sparse_matrix_operator` (268) | ~1,905 | HDF5-backed chunked matrix I/O |
| **tools/** | `matrix_transform` (191), `matrix_aggregate` (257), `autocorrelation` (214), `enrichment` (161), `mwm` (209), `xicor` (93), `guide_calling` (970) | `matrix_transform` (43), `matrix_aggregate` (40), `autocorrelation` (47), `enrichment` (26), `mwm` (25), `xicor` (29), `guide_calling` (153) | ~2,458 | Matrix ops, stats, guide calling |
| **visualization/** | `layout_network` (39), `uwot_actionet` (258), `color_map` (44) | `layout_network` (57), `uwot_actionet` (24), `color_map` (15), `UmapFactory` (197), `UwotArgs` (224), `OptimizerArgs` (36), `find_ab` (169) | ~1,063 | UMAP layout, node coloring |
| **utils_internal/** | `utils_matrix` (35), `utils_stats` (62), `utils_decomp` (118), `utils_misc` (63), `utils_active_set` (491) | `utils_matrix` (29), `utils_stats` (32), `utils_decomp` (41), `utils_misc` (24), `utils_active_set` (29), `utils_parallel` (119) | ~1,043 | Internal numeric/threading helpers |

### Vendored External Code (not auditable, listed for context)

| Component | Location | Approx Lines | Role |
|-----------|----------|-------------|------|
| Armadillo | `include/extern/armadillo/` | ~230K | Linear algebra (header-only) |
| PRIMME | `src/extern/primme/` | ~28K | Large sparse eigensolver |
| hnswlib | `include/extern/hnswlib/` | ~2.7K | Approximate nearest neighbors |
| uwot | `include/extern/uwot/` | ~2.0K | UMAP optimization |
| StatsLib + GCEM | `include/extern/StatsLib/`, `include/extern/gcem/` | ~8K | Statistical distributions |
| colorspace | `src/extern/colorspace.cpp` + header | ~1.1K | Color conversion |
| aarand / PCG | `include/extern/aarand/`, `include/extern/pcg/` | ~0.5K | Random number generation |
| fastapprox | `include/extern/fastapprox/` | ~0.3K | Fast approximate math |
| RcppPerpendicular | `include/extern/RcppPerpendicular.h` | 176 | R build helper (unused in Python build) |
| convert_seed | `include/extern/convert_seed.h` | 120 | Seed conversion utility |

### Excluded Code (`_EXCLUDE/`)

Not compiled. Contains:
- `_old_extern/` — superseded libraries (old hnswlib, mini_thread, n2, s_gd2, cblas.h)
- `experimental/` — UMAP++ wrapper (umappp, subpar)
- `extern/HDBSCAN/` — HDBSCAN implementation (removed from active build)
- `extra/` — Extended function variants, old R wrappers, deprecated algorithms
- Unused CMake modules (`ConfigureCHOLMOD.cmake.unused`, `FindSuiteSparse.cmake.unused`)

### R Wrappers (`wrappers_r/`)

Reference Rcpp wrappers (~903 lines total). Primary R package lives in
`actionet-r`. These are kept as reference and for R-mode builds but are
**not compiled** in the Python build path.

| File | Lines | Module |
|------|-------|--------|
| `wr_action.cpp` | 191 | ACTION / AA |
| `wr_annotation.cpp` | 110 | Specificity / markers |
| `wr_decomposition.cpp` | 196 | SVD algorithms |
| `wr_experimental.cpp` | 6 | Placeholder |
| `wr_network.cpp` | 141 | Network build / diffusion |
| `wr_tools.cpp` | 240 | Matrix tools |
| `actionet_r_config.h` | 19 | R build config |

---

## CMake Build System

### Root `CMakeLists.txt`

- Static library target (`actionet`)
- GLOB_RECURSE per module (no manual file listing)
- Conditional exclusion of `svd_primme.cpp` for R builds
- Links: BLAS/LAPACK, HDF5, OpenMP, PRIMME
- C++17, PIC, export compile commands
- Thread count: auto-detect minus 2 (≥6 cores)

### CMake Modules (`cmake/`)

| Module | Role |
|--------|------|
| `ConfigureApple.cmake` | Apple-specific arch flags |
| `ConfigureBLAS.cmake` | BLAS/LAPACK vendor detection |
| `ConfigureOpenMP.cmake` | OpenMP runtime discovery |
| `ConfigurePRIMME.cmake` | PRIMME library config |
| `ConfigureR.cmake` | R-mode build (Rcpp, RcppArmadillo) |

---

## Binding Interface (pybind11 side, in `actionet-python`)

Each C++ module has a corresponding pybind11 wrapper in
`src/actionet/bindings/`:

| Wrapper | Lines | C++ Module |
|---------|-------|-----------|
| `wp_action.cpp` | 292 | action |
| `wp_annotation.cpp` | 484 | annotation |
| `wp_decomposition.cpp` | 328 | decomposition |
| `wp_io.cpp` | 239 | io |
| `wp_network.cpp` | 162 | network |
| `wp_tools.cpp` | 747 | tools |
| `wp_visualization.cpp` | 112 | visualization |
| `wp_utils.{cpp,h}` | 501 | Shared arma↔numpy/scipy conversion |
| `_core.cpp` | 28 | Module entry point |

---

## Key Architectural Decisions (Non-Negotiable)

These are recorded in `context/DECISIONS.md` (within `libactionet`) and must
not be re-litigated during audit:

1. **OpenMP is a hard requirement** — no OFF option; used pervasively.
2. **Multi-repo structure** — `libactionet` is standalone; bindings live in
   front-end repos.
3. **Python is performance-first** — R is feature-complete reference.
4. **Breaking changes allowed** if they improve performance/usability.
5. **Exported APIs are contracts** — wrappers in both R and Python consume them.
6. **`extern/` is drop-in** — must not be modified.
7. **CUDA 12.2+ floor, Ampere+ only** — for future GPU work.
8. **Operator-backed SVD stays CPU-only** — callback overhead dominates.

---

## GPU Support Context (Upcoming)

From `plans/GPU_BACKEND_PLAN.md`:

| Phase | Scope | GPU Targets |
|-------|-------|-------------|
| 1 | SVD (in-memory PRIMME via cuBLAS) | `decomposition/` |
| 1-hardening | Error taxonomy, benchmarks, CI | Cross-cutting |
| 2 | Network construction (cuVS + JSD rerank) | `network/` |
| 3 | ACTION (AA + SPA on GPU dense BLAS) | `action/` |

Integration will require:
- `ExecutionPolicy` struct threaded through entry points
- `ComputeBackend` enum for dispatch
- Clean module boundaries for conditional compilation
- No global state in hot paths
- Minimal coupling between modules

---

## Audit Execution Instructions

When auditing a module, apply the `/codebase-cleanup-audit` skill with these
repo-specific instructions:

### Scope

- Audit the specified C++ module(s): both `src/<module>/` and
  `include/<module>/` together.
- Consider interactions with:
  - The umbrella header (`include/libactionet.hpp`)
  - The config header (`include/libactionet_config.hpp`)
  - `utils_internal/` (shared helpers)
  - The corresponding pybind11 wrapper (`src/actionet/bindings/wp_<module>.cpp`)
  - The corresponding R wrapper (`wrappers_r/wr_<module>.cpp`) if relevant
- Do NOT audit vendored `extern/` code.

### Priorities (ordered)

1. Dead code, unused helpers, stale compatibility shims, unreachable branches
2. Bugs, broken call paths, signature mismatches between header and implementation
3. API inconsistencies: `arma::field` returns vs typed structs, inconsistent
   naming, parameter ordering drift between C++ header and binding wrapper
4. Documentation gaps: undocumented public APIs, missing parameter semantics,
   unclear lifetime/ownership contracts
5. Coupling hotspots that will block GPU dispatch insertion (global state,
   hidden cross-module dependencies, non-obvious threading assumptions)
6. Consolidation opportunities (duplicate logic, repeated patterns that could
   be factored)
7. Easy efficiency wins (unnecessary copies, redundant allocations, cache-
   unfriendly access patterns)

### Report Format

Use the findings-first report template from the skill. Include:

- File paths and line numbers (relative to `src/libactionet/`)
- Severity classification:
  - **Confirmed issue** — direct evidence of bug, mismatch, or dead code
  - **Likely risk** — strong signal, needs one more verification step
  - **Open question** — ambiguous ownership or intent, needs clarification
- Concrete evidence for each finding
- Quick wins vs. larger follow-ups

### Guardrails

- Do NOT edit code — audit only.
- Do NOT re-litigate decisions in `context/DECISIONS.md` (either the
  `actionet-python` or `libactionet` copy).
- Do NOT recommend modifying vendored `extern/` code.
- Do NOT recommend deleting `_EXCLUDE/` or `wrappers_r/` without concrete
  evidence that they are obsolete and have no downstream consumers.
- Treat `include/libactionet.hpp` exports as the public API contract.
- Consider that both R and Python bindings consume the public headers.

---

## Suggested Audit Order

Audit in dependency order (leaves first, roots last). Each phase audits the
C++ module together with its binding interfaces.

| Phase | C++ Module | Binding Interface | Rationale |
|-------|-----------|-------------------|-----------|
| 1 | `utils_internal/` + `libactionet_config.hpp` | `wp_utils.{cpp,h}` | Foundational; used by all modules |
| 2 | `decomposition/` | `wp_decomposition.cpp` | Core SVD; first GPU target; largest algorithm surface |
| 3 | `io/backed_h5ad/` | `wp_io.cpp` | Backed operator stack; complex chunked I/O |
| 4 | `action/` | `wp_action.cpp` | ACTION algorithm; depends on decomposition; second GPU target |
| 5 | `network/` | `wp_network.cpp` | Graph construction; depends on utils; third GPU target |
| 6 | `annotation/` | `wp_annotation.cpp` | Depends on network + io; large specificity implementation |
| 7 | `tools/` | `wp_tools.cpp` | Heterogeneous utilities; guide_calling is 970 lines |
| 8 | `visualization/` | `wp_visualization.cpp` | UMAP integration; smallest binding surface |
| 9 | Build system | `CMakeLists.txt` + `cmake/` | Cross-cutting; after understanding all modules |

---

## Phase-Specific Focus Areas

### Phase 1: `utils_internal/`
- Threading model (`utils_parallel.hpp`): OpenMP patterns, thread safety
- `utils_active_set.cpp` (491 lines): largest utility file — consolidation?
- Config header: dead macros, commented-out code, build-mode clarity

### Phase 2: `decomposition/`
- 5 SVD algorithm implementations: overlap, dead paths, algorithm selection logic
- `matrix_operator.hpp` (129 lines): operator abstraction for backed SVD
- `orthogonalization.cpp` (325 lines): batch correction algorithm correctness
- GPU readiness: which entry points need `ExecutionPolicy` threading

### Phase 3: `io/backed_h5ad/`
- Sparse operator (1048 lines): chunking logic, memory management, thread safety
- Dense operator (333 lines): simpler but same patterns
- Factory pattern (`create_backed_operator`): type dispatch

### Phase 4: `action/`
- `reduce_kernel.cpp` (244 lines): kernel reduction, complex control flow
- `action_post.cpp` (152 lines): post-processing, archetype merging
- Return types: `arma::field` usage vs typed structs
- Memory: stacking/moving patterns documented in TODO as "Done"

### Phase 5: `network/`
- `build_network.cpp` (809 lines): largest single source file; HNSW integration
- Jensen-Shannon divergence: custom distance metric in `hnsw_jensen_shannon.hpp`
- Graph builder patterns: memory allocation, thread safety

### Phase 6: `annotation/`
- `specificity.cpp` (758 lines): large, performance-critical
- Backed operator overloads: parallel backed paths
- `marker_stats.cpp` (246 lines): precomputed stats optimization

### Phase 7: `tools/`
- `guide_calling.cpp` (970 lines): largest tool; GMM fitting, complex
- Heterogeneous collection: autocorrelation, enrichment, MWM, xicor

### Phase 8: `visualization/`
- uwot integration: `UmapFactory`, `UwotArgs`, `OptimizerArgs` (header-heavy)
- `find_ab.hpp` (169 lines): header-only UMAP parameter estimation
- `layout_network.cpp` (39 lines): thin dispatcher

### Phase 9: Build system
- GLOB_RECURSE for sources: risk of unintended file inclusion
- HDF5 linking strategy (the "SKETCHY DEFENSIVE FIX" comment)
- PRIMME configuration complexity
- R-mode vs standalone divergence

---

## Example Agent Prompt

```
Read context/AUDIT_CONTEXT_CPP.md for project context and audit instructions.

Audit Phase [N]: [module name]

Apply the /codebase-cleanup-audit skill to the following files:
- src/libactionet/src/<module>/*.cpp
- src/libactionet/include/<module>/*.hpp
- src/actionet/bindings/wp_<module>.cpp  (binding interface)

Focus on: [phase-specific focus areas from the document]

Produce a findings-first report. Do not edit code.
```

---

## Cross-References

- `context/AUDIT_CONTEXT.md` — original Python + overview audit context (Python audit complete)
- `src/libactionet/context/DECISIONS.md` — libactionet-specific decisions
- `src/libactionet/context/AGENT_PLAYBOOK.md` — libactionet agent guardrails
- `src/libactionet/plans/GPU_BACKEND_PLAN.md` — GPU roadmap and post-mortem
- `src/libactionet/TODO.md` — known remaining work items
- `src/libactionet/docs/` — algorithm documentation (batch orthogonalization, backed I/O chunking, guide calling)
