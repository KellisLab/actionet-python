# Decisions (ADR-lite)

This document records **deliberate architectural and operational decisions** for the ACTIONet ecosystem. These decisions are considered settled unless explicitly revised.

---

## Software architecture

### Multi-repo structure

**Decision:** Maintain separate repositories for:

- `libactionet` (C++ core)
- `actionet-r` (R front-end)
- `actionet-python` (Python front-end)
- `ACTIONetExperiment` (Deprecated: AnnData symmetric data container for `actionet-r`)

**Rationale:**

- Clear separation of concerns
- Independent packaging and release cycles (C++ / CRAN-style / PyPI-style)
- Avoids monorepo friction while preserving coordination via shared specs

---

## Language bindings

### C++ core + wrappers

**Decision:**

- C++ core built with **CMake**
- R bindings via **Rcpp**
- Python bindings via **pybind11**

**Rationale:**

- Mature, stable tooling
- Explicit control over ABI and performance
- Good compatibility with HPC environments

---

## Front-end prioritization

### Python vs R

**Decision:**

- Python front-end is the **performance-first and pipeline-critical interface**
- R front-end remains supported and is more feature-complete

**Rationale:**

- R performance and ecosystem limitations at scale
- Python integration with pipeline and HPC workflows
- Preserve compatibility with R users

---

<!-- ## Reproducibility and stability

### Output contracts
**Decision:**
- Output formats, directory structures, and file naming are treated as **contracts**
- Changes require explicit documentation and migration plans

### Versioning
**Decision:**
- Critical dependencies (especially `actionet-python`) must be version-pinned and logged in pipeline runs

--- -->

## OpenMP as hard requirement

**Decision:**

- OpenMP is a **hard build requirement** for `libactionet` and all front-ends.
- The CMake build emits `FATAL_ERROR` if no OpenMP runtime is found.
- Valid runtime options are `AUTO`, `GNU`, `INTEL`, `LLVM`. There is no `OFF` option.

**Rationale:**

- OpenMP is used pervasively in C++ for parallel loops across decomposition, network, annotation, I/O, and tool code paths.
- Removing OpenMP would leave the library single-threaded with no practical benefit.
- HPC environments universally provide OpenMP via GCC (`libgomp`), Intel (`libiomp5`), or LLVM (`libomp`).

---

## SVD algorithm strategy

### Public SVD surface: IRLB, Halko, Feng (PRIMME removed)

**Decision:**

- The public Python SVD API exposes three algorithms: `"irlb"`, `"halko"`, and `"feng"`.
- `"auto"` selects IRLB for in-memory sparse inputs, Halko for in-memory dense inputs, and Halko for backed (HDF5-streamed) operator inputs.
- `"primme"` has been removed from the public Python API, from `_SVD_ALGORITHM_TO_ID`, and from every auto-selection heuristic.
- The C++ `ALG_PRIMME` enum, `svd_primme.{cpp,hpp}`, `runSVD_PRIMME_Operator`, and the vendored `src/libactionet/src/extern/primme/` tree remain compiled behind the existing R-build guard for one release cycle. No Python entry point can reach them. Deletion is tracked in `TODO.md`.
- The `MatrixOperator::prefer_block_solver_for_irlb()` hint and its two backed overrides have been removed. Backed operators requesting `svd_algorithm="irlb"` now unconditionally use the honest `svdIRLB(MatrixOperator&, ...)` overload; there is no hidden dispatch to PRIMME.

**Rationale:**

- Sparse `nnz > 2^31 - 1` no longer requires PRIMME. `libactionet` force-defines `ARMA_64BIT_WORD`, so `arma::sp_mat` handles 64-bit index arrays directly and IRLB's sparse product path goes through 64-bit-clean Armadillo operators.
- The backed `IRLB -> PRIMME` fast path was a design leak: users who explicitly requested `"irlb"` on backed inputs silently ran PRIMME's block Lanczos SVD, doubling the maintenance surface and violating the algorithm contract exposed to callers.
- PRIMME's cuBLAS path is a poor fit for the planned GPU work (confirmed by the scrapped July 2026 attempt documented in `plans/GPU_INTEGRATION.md`).
- The C++/R-side quarantine gives one release cycle of revert safety without complicating the Python surface.

**Related:**

- `plans/primme_removal_and_64bit_irlb_*.plan.md` for the implementation plan.
- `plans/SVD_STRATEGY_REDESIGN_v2.md` for the follow-up direction (shared product-backend abstraction, GPU strategy).

---

## Backed SVD algorithm default

### Backed operator path: Halko as default

**Decision:**

- For backed (HDF5-streamed) operator SVD, `auto` selects **Halko**.
- IRLB is available as an explicit backed option (pass `svd_algorithm="irlb"`) and now runs the honest `svdIRLB(MatrixOperator&, ...)` overload with no hidden PRIMME fallback.
- Feng is available as an explicit backed option (`svd_algorithm="feng"`); it is competitive with Halko and may become the auto-default in the future if the crossover observed at 200k cells widens with scale.

**Rationale:**

- Halko's matvec count is fixed at `2*(iters+1)` passes regardless of matrix conditioning, giving a predictable NNZ-proportional I/O cost model.
- IRLB's convergence-driven iteration count adds variance to I/O load that complicates scaling predictions for atlas-size datasets. Empirically, backed IRLB is 6.5-7.3x slower than Halko across every tier we benchmarked (25k-200k cells).
- Feng tracks Halko to within ~5% median wall time at tiers <=150k and beats Halko by ~13% at 200k, but the crossover point does not clear our 10% wall-time threshold for switching a documented default.
- All three algorithms are correctness-equivalent (`sigma_corr >= 0.999998` on the benchmark set).

**Benchmark reference:**

- `tests/benchmark_backed_svd_algorithm.py` — three-way Halko vs IRLB vs Feng benchmark on backed data across cell-count tiers.
- The benchmark measures wall time, peak RSS, I/O bytes read, singular value correlation (accuracy), and reconstruction error.
- Results and the full auto-selection table are in `docs/svd_algorithm_benchmark.md`.
- Reference run: 25k, 50k, 100k, 150k, 200k cell tiers, 2 trials per configuration, `n_components=30`.

**Status:** Confirmed. Halko is the auto-default for all backed inputs.

---

## In-memory SVD algorithm defaults

### Sparse in-memory: IRLB. Dense in-memory: Halko

**Decision:**

- For **sparse** in-memory inputs, `auto` selects **IRLB**.
- For **dense** in-memory inputs, `auto` selects **Halko**.
- Feng is available as an explicit choice for both storage forms but is never auto-selected.

**Rationale:**

- **Sparse:** IRLB is 3-6x faster than Halko or Feng across the 25k-200k cell tier range on real single-cell matrices. Sparse `nnz` is already 64-bit clean under `ARMA_64BIT_WORD`, so IRLB carries no residual size limitation vs the randomized methods.
- **Dense:** Halko narrowly beats Feng (~5% median wall time) and beats IRLB by roughly 2x at every tier. Feng is a viable second option but does not dislodge Halko as the default.
- All three algorithms produce singular values with `sigma_corr >= 0.999998` across the benchmark set.

**Benchmark reference:**

- `tests/benchmark_svd_inmemory_defaults.py` — three-way sparse and dense benchmark across `{25k, 50k, 100k, 150k, 200k}` (dense capped at 100k to fit typical single-node RAM).
- Full results, ratio tables, and reproduction commands: `docs/svd_algorithm_benchmark.md`.

**Status:** Confirmed. The auto-selection heuristics in `_select_svd_algorithm_inmemory` match the benchmark winners with no further changes required.

---

## Change management

### Backward compatibility

**Decision:**

- Breaking changes are allowed if justified.
- Such changes must substantially improve:
  - Performance
  - Resource usage
  - User ease-of-use
  - Reproducibility

### Agent behavior

**Decision:**

- LLM/coding agents should not re-litigate decisions recorded in this document
- Deviations require explicit human approval

---
