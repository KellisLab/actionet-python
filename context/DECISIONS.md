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

## GPU backend (Python surface)

See [`GPU_INTEGRATION.md`](GPU_INTEGRATION.md) for the full Python
roadmap and
[`../src/libactionet/context/DECISIONS.md`](../src/libactionet/context/DECISIONS.md)
for the C++/build-side GPU decisions. **No GPU code currently exists
on `dev`.** A prior attempt was scrapped after failing on hardware
sign-off; the decisions below survive the reset and constrain the
future re-attempt.

### Kwarg names

**Decision:** Three kwargs, exact spelling, on every entry point that
dispatches to a GPU-eligible libactionet routine:

- `compute_backend: Literal["auto", "cpu", "gpu"]`, default `"auto"`
- `device_id: int >= 0`, default `0`
- `allow_cpu_fallback: bool`, default `True`

**Rationale:** Stable Python surface across future phases; consistent
with the C++ `ExecutionPolicy` field names. No env-var override
(per-call only) because env vars hide policy decisions from notebooks
and complicate test isolation.

### Fallback semantics include runtime failure

**Decision:** With `compute_backend="gpu"` and
`allow_cpu_fallback=True`, the wrapper must fall back to CPU **both**
when build/runtime probes fail **and** when the GPU dispatch itself
fails at runtime (segfault-adjacent errors, OOM, cuBLAS/cuSOLVER
error, etc.). The scrapped implementation only checked probes and
therefore entered a broken dispatch on GPU-visible machines even with
`allow_cpu_fallback=True`.

**Rationale:** The whole point of the flag is to protect users from
GPU-side breakage. Probes are necessary but not sufficient. The
re-attempt should include a runtime canary (small dummy solve at
import time or first GPU call) so the fallback semantics hold end to
end.

### Backend identifier source of truth

**Decision:** The Python side re-exports `_core.BACKEND_CPU`,
`_core.BACKEND_GPU`, `_core.BACKEND_AUTO` **integer constants** from
the pybind11 module rather than redefining a parallel hardcoded
mapping.

**Rationale:** Prevents silent desynchronization with the C++ enum.

### Metadata symmetry

**Decision:** Both `reduce_kernel` and `run_svd` must record the
resolved policy on their result -- `reduce_kernel` via
`adata.uns[f"{key_added}_params"]`, `run_svd` via the returned dict
when `return_operator_compatible=False`. The recorded backend is
concrete (`"cpu"` or `"gpu"`, never `"auto"`).

**Rationale:** Reproducibility and post-hoc auditing; users must be
able to tell which backend actually ran without re-running.

### Error class re-export

**Decision:** `actionet.GpuError`, `actionet.GpuUnavailableError`,
`actionet.GpuRuntimeError` are re-exported from the C++ taxonomy via
`_core` and live on the top-level `actionet` namespace. They inherit
from `RuntimeError` to preserve backward compatibility for callers
that catch the base class.

### Test gating

**Decision:** Every GPU-touching test must be gated by
`@requires_gpu` (or the pytest `-m gpu` marker resolving to the same
predicate). "Fallback" tests are not exempt.

**Rationale:** The scrapped attempt had three ungated `_fallback`
tests in `tests/test_gpu_backend_policy.py`
(`test_run_svd_dense_gpu_fallback`, `test_run_svd_sparse_gpu_fallback`,
`test_reduce_kernel_gpu_fallback_records_gpu_policy`) that crashed on
any real GPU box because the probes succeeded and no fallback fired.
Gating strictly avoids repeating this class of bug.

---
