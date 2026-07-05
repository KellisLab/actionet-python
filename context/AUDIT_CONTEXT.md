# Codebase Cleanup Audit — Context Document

Use this document as standing context when auditing individual modules of
`actionet-python` and `libactionet`. Pass it to an agent alongside the specific
module(s) being audited.

---

## Project Overview

ACTIONet is a multi-language computational biology toolkit for single-cell
multi-resolution analysis. This repository (`actionet-python`) provides the
Python front-end via pybind11 bindings to the C++ core (`libactionet`, included
as a git submodule).

- **Repository root:** `actionet-python`
- **Python package:** `src/actionet/`
- **C++ core (submodule):** `src/libactionet/`
- **Build:** CMake + scikit-build-core + pybind11 (C++17, OpenMP required)
- **Runtime deps:** BLAS/LAPACK, HDF5, OpenMP
- **Python version:** ≥3.12

---

## Motivation for Audit

GPU support is the next major feature. Before implementation, the codebase needs
systematic cleanup to:

1. Remove dead/redundant code that will complicate extensions to the codebase.
2. Clarify module boundaries so new features have clean integration seams.
3. Fix documentation/contract drift before the API surface grows.
4. Identify coupling hotspots that will block modular extensions.

---

## Architecture (Layer Diagram)

```
┌──────────────────────────────────────────────────────────────┐
│  Python API  (src/actionet/*.py)                             │
│  User-facing functions, AnnData integration, orchestration   │
├──────────────────────────────────────────────────────────────┤
│  pybind11 Wrappers  (src/actionet/wp_*.cpp + _core.cpp)      │
│  Thin C++ → Python bridge; data conversion, lifetime mgmt    │
├──────────────────────────────────────────────────────────────┤
│  libactionet C++ Core  (src/libactionet/)                    │
│  Armadillo-based numerics, OpenMP parallel, HDF5 backed I/O  │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Map — Python Package (`src/actionet/`)

| Module | Lines | Role |
|--------|------:|------|
| `__init__.py` | 195 | Public API surface; re-exports from all modules |
| `core.py` | 731 | Primary entry points: `run_action`, `build_network`, layout, centrality, diffusion |
| `reduction.py` | 581 | SVD, kernel reduction, smoothing |
| `advanced.py` | 430 | Archetypal analysis, simplex regression, SPA, label propagation |
| `annotation.py` | 969 | Marker finding, cell/cluster annotation |
| `specificity.py` | 443 | Feature specificity scores |
| `clustering.py` | 232 | Network-based clustering |
| `preprocessing.py` | 1582 | Import, filter, normalize, backed decompression |
| `_backed_persist.py` | 1446 | Backed HDF5 checkpointing, subsetting, materialization |
| `backed_io.py` | 336 | Backed I/O helpers |
| `_backed_compression.py` | 165 | Compression utilities for backed storage |
| `_matrix_source.py` | 774 | Unified matrix access abstraction (dense/sparse/backed) |
| `lazy_transform.py` | 444 | Lazy matrix transform pipeline (deferred normalization, etc.) |
| `batch_correction.py` | 319 | Batch effect and basal expression correction |
| `imputation.py` | 231 | Network-based feature imputation |
| `anndata_utils.py` | 334 | AnnData ↔ matrix conversions, result injection |
| `tools.py` | 294 | Scale, aggregate, matrix sums |
| `guide_calling.py` | 765 | Perturb-seq guide calling (GMM fitting, thresholds) |
| `pipeline.py` | 285 | End-to-end `run_actionet` convenience pipeline |
| `visualization.py` | 96 | Node color computation |
| `_feature_lookup.py` | 162 | Feature name resolution |
| **plotting/** | 3595 | Plotting subpackage (UMAP, QC, feature expression) |
| `experimental/` | — | Experimental/unstable features (`_anndata_io.py`) |
| `_data/` | — | Bundled data/resources |

---

## Module Map — pybind11 Wrappers (`src/actionet/wp_*.cpp`)

| Wrapper File | Lines | C++ Module Bound |
|--------------|------:|------------------|
| `_core.cpp` | 28 | Module entry; dispatches to init_* functions |
| `wp_action.cpp` | 367 | action (AA, ACTION decomp, reduce_kernel, SPA, simplex) |
| `wp_decomposition.cpp` | 446 | decomposition (SVD algorithms, orthogonalization) |
| `wp_network.cpp` | 169 | network (build, diffusion, measures, label propagation) |
| `wp_annotation.cpp` | 497 | annotation (specificity, marker_stats) |
| `wp_io.cpp` | 255 | io (backed HDF5 operators) |
| `wp_tools.cpp` | 788 | tools (autocorrelation, enrichment, aggregation, xicor, MWM, guide calling) |
| `wp_visualization.cpp` | 112 | visualization (layout, color map) |
| `wp_utils.cpp` | 414 | Shared conversion utilities (arma ↔ numpy/scipy) |

---

## Module Map — libactionet C++ Core (`src/libactionet/`)

### Core Modules (in `src/` and `include/`)

| Module | Source Files | Headers | Role |
|--------|-------------|---------|------|
| **decomposition/** | `svd_main`, `svd_halko`, `svd_irbla`, `svd_feng`, `svd_primme`, `orthogonalization` | `svd_main`, `svd_halko`, `svd_irbla`, `svd_feng`, `svd_primme`, `orthogonalization`, `matrix_operator` | SVD algorithms, matrix operators, orthogonalization |
| **action/** | `action_main`, `action_decomp`, `action_post`, `aa`, `spa`, `reduce_kernel`, `simplex_regression` | (matching headers) | Archetypal Analysis, ACTION decomposition, kernel reduction |
| **network/** | `build_network` (in header-only `hnsw_imp.hpp`), `network_diffusion`, `network_measures`, `label_propagation` | `hnsw_imp`, `hnsw_jensen_shannon`, `network_diffusion`, `network_measures`, `label_propagation` | kNN graph construction (HNSW), diffusion, graph metrics |
| **annotation/** | `specificity`, `marker_stats` | (matching headers) | Gene specificity scores, marker statistics |
| **io/** | `create_backed_operator`, `backed_dense_matrix_operator`, `backed_sparse_matrix_operator` | (matching headers) | HDF5-backed chunked matrix I/O |
| **tools/** | `matrix_transform`, `matrix_aggregate`, `autocorrelation`, `enrichment`, `mwm`, `xicor`, `guide_calling` (970 lines) | (matching headers, note: `guide_calling.hpp` not in include/) | Matrix ops, statistical tools, guide calling |
| **visualization/** | `layout_network`, `uwot_actionet`, `color_map` | `layout_network`, `uwot_actionet`, `color_map`, `UmapFactory`, `UwotArgs`, `OptimizerArgs`, `find_ab` | UMAP layout, node coloring |
| **utils_internal/** | `utils_matrix`, `utils_stats`, `utils_decomp`, `utils_misc`, `utils_active_set` | (matching headers + `utils_parallel`) | Internal numeric helpers |

### External/Vendored (in `include/extern/`)

- **Armadillo** — Linear algebra (header-only, bundled)
- **StatsLib + GCEM** — Statistical distributions
- **aarand** — Random number generation
- **PRIMME** — Large sparse eigensolver (in `src/extern/primme/`)
- **colorspace.cpp** — Color conversion (1013 lines)

### Excluded Code (`_EXCLUDE/`)

- `experimental/` — UMAP++ wrapper (unused)
- `extra/` — Extended autocorrelation, old R wrappers

### R Wrappers (`wrappers_r/`)

- 7 Rcpp wrapper files (sibling front-end, not used by Python build)

---

## Key Architectural Decisions (Non-Negotiable)

1. **OpenMP is a hard requirement** — no OFF option; used pervasively.
2. **Halko is the default backed SVD** — predictable I/O cost model.
3. **Python is the performance-first front-end** — R is feature-complete reference.
4. **Breaking changes are allowed** if they improve performance/usability.
5. **libactionet APIs are contracts** — Python wrappers map to them cleanly.

---

## GPU Support Context (Upcoming)

The planned GPU support will primarily affect:

- **decomposition/** — SVD kernels (cuSOLVER/cuBLAS dispatch)
- **action/** — Dense matrix operations in AA and kernel reduction
- **network/** — kNN construction (FAISS or cuVS), diffusion
- **tools/** — Matrix transforms and aggregation
- **io/** — Device memory management for backed streaming

Clean module boundaries and minimal cross-module coupling are prerequisites.

---

## Audit Execution Instructions

When auditing a module, apply the `/codebase-cleanup-audit` skill with
these repo-specific instructions:

### Scope

- Audit the specified module(s) only.
- Consider interactions with adjacent modules (e.g., a Python module's
  corresponding `wp_*.cpp` wrapper and the C++ headers it calls).

### Priorities (ordered)

1. Dead code, unused helpers, stale compatibility shims
2. Bugs, broken call paths, signature mismatches between layers
3. Documentation/docstring drift from implementation
4. Consolidation opportunities (duplicate logic across modules)
5. Coupling hotspots that will block GPU dispatch insertion
6. Easy efficiency wins

### Report Format

Use the findings-first report template from the skill. Include:

- File paths and line numbers
- Severity classification (confirmed issue / likely risk / open question)
- Concrete evidence for each finding
- Quick wins vs. larger follow-ups

### Guardrails

- Do NOT edit code — audit only.
- Do NOT re-litigate decisions in `context/DECISIONS.md`.
- Do NOT recommend deleting `_EXCLUDE/` or `wrappers_r/` without evidence of obsolescence.
- Treat `__init__.py` exports as the public API contract.

---

## Suggested Audit Order

Audit in dependency order (leaves first, roots last):

| Phase | Modules | Rationale |
|-------|---------|-----------|
| 1 | `utils_internal/` (C++) + `wp_utils.cpp` | Foundational helpers; used everywhere |
| 2 | `decomposition/` (C++) + `wp_decomposition.cpp` + `reduction.py` | Core SVD; first GPU target |
| 3 | `io/` (C++) + `wp_io.cpp` + `backed_io.py` + `_backed_persist.py` + `_matrix_source.py` + `_backed_compression.py` | Backed I/O stack; complex, large |
| 4 | `action/` (C++) + `wp_action.cpp` + `core.py` + `advanced.py` | ACTION algorithm; second GPU target |
| 5 | `network/` (C++) + `wp_network.cpp` + `core.py` (network parts) | Graph construction; third GPU target |
| 6 | `annotation/` (C++) + `wp_annotation.cpp` + `annotation.py` + `specificity.py` | Annotation stack |
| 7 | `tools/` (C++) + `wp_tools.cpp` + `tools.py` + `guide_calling.py` | Utility tools |
| 8 | `visualization/` (C++) + `wp_visualization.cpp` + `visualization.py` + `plotting/` | Visualization + plotting |
| 9 | `preprocessing.py` + `anndata_utils.py` + `lazy_transform.py` + `batch_correction.py` + `imputation.py` | Python-only orchestration |
| 10 | `pipeline.py` + `clustering.py` + `__init__.py` | Top-level API surface |

---

## Example Agent Prompt

```
Read context/AUDIT_CONTEXT.md for project context and audit instructions.

Audit Phase [N]: [module list]

Apply the /codebase-cleanup-audit skill to the following files:
- [list specific file paths]

Focus on: [any phase-specific focus areas]

Produce a findings-first report. Do not edit code.
```
