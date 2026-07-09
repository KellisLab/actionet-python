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

**Rationale:**

- Halko's matvec count is fixed at `2*(iters+1)` passes regardless of matrix conditioning, giving a predictable NNZ-proportional I/O cost model.
- IRLB's convergence-driven iteration count adds variance to I/O load that complicates scaling predictions for atlas-size datasets.
- Both algorithms share the same C++ `MatrixOperator` backend and are correctness-equivalent.

**Benchmark reference:**

- `tests/benchmark_backed_svd_algorithm.py` — focused Halko vs IRLB benchmark on backed data across cell-count tiers.
- The benchmark measures wall time, peak RSS, I/O bytes read, singular value correlation (accuracy), and reconstruction error.
- Run and update `docs/svd_algorithm_benchmark.md` with empirical results before re-litigating this decision.

**Status:** Pending empirical benchmark run. Default confirmed as Halko pending results.

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
