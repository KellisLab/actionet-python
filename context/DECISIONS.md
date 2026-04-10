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

## Backed SVD algorithm default

### Backed operator path: Halko as default

**Decision:**

- For backed (HDF5-streamed) operator SVD, `auto` selects **Halko**.
- IRLB is available as an explicit backed option (pass `svd_algorithm="irlb"`) but is not the auto-selected default.

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

- Avoid breaking changes are allowed if justifed.
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
