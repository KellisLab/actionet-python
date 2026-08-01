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

### Public SVD surface: IRLB, Halko (PRIMME and Feng removed)

**Decision:**

- The public Python SVD API exposes two algorithms: `"irlb"` and `"halko"`.
- `"auto"` selects IRLB for in-memory sparse inputs, Halko for in-memory dense inputs, and Halko for backed (HDF5-streamed) operator inputs.
- `"primme"` has been removed from the public Python API, from `_SVD_ALGORITHM_TO_ID`, and from every auto-selection heuristic. The C++ sources (`svd_primme.{cpp,hpp}`, `runSVD_PRIMME_Operator`, `ALG_PRIMME`), the vendored `src/libactionet/src/extern/primme/` tree, and the `cmake/ConfigurePRIMME.cmake` module have been deleted.
- `"feng"` has been removed from the public Python API and from `_SVD_ALGORITHM_TO_ID`. Requesting it raises `ValueError` from `_normalize_algorithm` listing the allowed set `{auto, halko, irlb}`. The C++ sources (`svd_feng.{cpp,hpp}`, `ALG_FENG`) and the Feng switch cases in `runSVD`/`runSVD_Operator` have been deleted.
- Python pybind `_core` SVD entry points validate raw algorithm IDs and reject anything other than IRLB (`0`) or Halko (`1`), so private `_core` calls cannot bypass the public wrapper policy.
- The `MatrixOperator::prefer_block_solver_for_irlb()` hint and its two backed overrides have been removed. Backed operators requesting `svd_algorithm="irlb"` now unconditionally use the honest `svdIRLB(MatrixOperator&, ...)` overload; there is no hidden dispatch to any other algorithm.
- The R-package (`actionet-r`, out-of-tree) still exposes `algorithm=2` (Feng) and `algorithm=3` (PRIMME) bindings and requires a matching cleanup patch. This is tracked in `src/libactionet/TODO.md`. The `wrappers_r/` files inside this repo's submodule are reference-only copies and were intentionally left untouched.

**Rationale:**

- Sparse `nnz > 2^31 - 1` no longer requires PRIMME. `libactionet` force-defines `ARMA_64BIT_WORD`, so `arma::sp_mat` handles 64-bit index arrays directly and IRLB's sparse product path goes through 64-bit-clean Armadillo operators.
- The backed `IRLB -> PRIMME` fast path was a design leak: users who explicitly requested `"irlb"` on backed inputs silently ran PRIMME's block Lanczos SVD, doubling the maintenance surface and violating the algorithm contract exposed to callers.
- PRIMME's cuBLAS path is a poor fit for the planned GPU work (confirmed by the scrapped July 2026 attempt documented in `plans/GPU_INTEGRATION.md`).
- PRIMME also introduced persistent ODR/LTO warnings against Armadillo's BLAS/LAPACK prototypes (documented in `plans/openblas_threading_and_odr_findings.md`), which the deletion obsoletes.
- Feng was never auto-selected and never wins a default on the benchmark set: 3.8-6.7x slower than IRLB on sparse in-memory, 5-9% slower than Halko on dense in-memory, and within ~5% of Halko on backed data at <=150k cells (13% faster only at 200k, below the repo's 10% wall-time threshold for switching a default). It is numerically indistinguishable from Halko (`sigma_corr >= 0.999998`) and belongs to the same randomized-power-iteration category. Halko is strictly the recommended randomized option; IRLB serves the deterministic/iterative need. Feng added no distinct capability while duplicating Halko's category. Retiring it aligns the public surface with the "one randomized SVD family" direction recorded in `plans/SVD_STRATEGY_REDESIGN_v2.md`.

**Related:**

- `plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md` for the active GPU-backed SVD implementation context.
- `plans/GPU_INTEGRATION.md` for the Python-facing GPU roadmap.
- `src/libactionet/plans/GPU_BACKEND_PLAN.md` for C++/build-side GPU constraints.
- `docs/svd_algorithm_benchmark.md` for the empirical evidence justifying the Feng retirement.

---

## Backed SVD algorithm default

### Backed operator path: Halko as default

**Decision:**

- For backed (HDF5-streamed) operator SVD, `auto` selects **Halko**.
- IRLB is available as an explicit backed option (pass `svd_algorithm="irlb"`) and now runs the honest `svdIRLB(MatrixOperator&, ...)` overload with no hidden fallback.
- Feng is no longer part of the public Python API (see "SVD algorithm strategy" above); the benchmark evidence that informed its retirement remains available.

**Rationale:**

- Halko's matvec count is fixed at `2*(iters+1)` passes regardless of matrix conditioning, giving a predictable NNZ-proportional I/O cost model.
- IRLB's convergence-driven iteration count adds variance to I/O load that complicates scaling predictions for atlas-size datasets. Empirically, backed IRLB is 6.5-7.3x slower than Halko across every tier we benchmarked (25k-200k cells).
- Feng tracked Halko to within ~5% median wall time at tiers <=150k and beat Halko by ~13% at 200k, but the crossover point did not clear our 10% wall-time threshold for switching a documented default and it belonged to the same randomized-power-iteration category as Halko without adding a distinct capability.
- All three algorithms produced correctness-equivalent results (`sigma_corr >= 0.999998`) on the backed benchmark set at the time of Feng's retirement.

**Benchmark reference:**

- `tests/benchmark_backed_svd_algorithm.py` — Halko vs IRLB benchmark on backed data across cell-count tiers.
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
- Feng is no longer part of the public Python API (see "SVD algorithm strategy" above); the benchmark evidence that informed its retirement remains available.

**Rationale:**

- **Sparse:** IRLB is 3-6x faster than Halko or Feng across the 25k-200k cell tier range on real single-cell matrices. Sparse `nnz` is already 64-bit clean under `ARMA_64BIT_WORD`, so IRLB carries no residual size limitation vs the randomized methods.
- **Dense:** Halko narrowly beats Feng (~5% median wall time) and beats IRLB by roughly 2x at every tier. Feng never dislodged Halko as a default; retirement follows.
- All three algorithms produced correctness-equivalent results (`sigma_corr >= 0.999998`) across the in-memory benchmark set at the time of Feng's retirement.

**Benchmark reference:**

- `tests/benchmark_svd_inmemory_defaults.py` — sparse and dense IRLB-vs-Halko benchmark across `{25k, 50k, 100k, 150k, 200k}` (dense capped at 100k to fit typical single-node RAM).
- Full results, ratio tables, and reproduction commands: `docs/svd_algorithm_benchmark.md`.

**Status:** Confirmed. The auto-selection heuristics in `_select_svd_algorithm_inmemory` match the benchmark winners with no further changes required.

---

## BLAS policy for ACTION

### Shape-specialized dense kernels under coarse OpenMP parallelism

**Decision:**

- ACTION's outer OpenMP decomposition across archetype count `k` owns
  parallelism for the default AA workload.
- Private column-major kernels handle copy, dot, scale, axpy, symmetric
  matvec, gemv, rank-one update, Gram, residual-product, norm, and
  normalization operations when the smaller matrix dimension is at most 128.
- Larger general-purpose matrices continue through the configured BLAS and
  Armadillo paths.
- The policy is based only on operation shape. There is no public API change,
  BLAS-vendor detection, new environment requirement, runtime BLAS-thread
  guard, import warning, or diagnostic Python API.
- Assignment results must remain identical across supported backends/thread
  counts; C and H comparisons use `rtol=1e-8`, `atol=1e-10`.

**Rationale:**

- Same-source measurement on the cached toy reduction was 9.40 s with MKL
  versus 79.36 s with OpenBLAS-OpenMP. SPA was backend-neutral; profiling
  isolated the gap to AA's high-frequency tiny/skinny BLAS dispatch.
- OpenBLAS runtime setters changed reported counts but did not reliably select
  its low-overhead serial path after initialization. The runtime-guard design
  on `bad-fix` is therefore rejected.
- Inline kernels keep the existing algorithm and coarse-grained OpenMP model
  while retaining vendor BLAS for shapes where its throughput is valuable.

**Deferred:**

- Near-MKL-parity redesign may evaluate batched active-set solves,
  blocked/fused AA updates, workspace reuse, and convergence policy. Those
  changes require separate numerical and performance evidence because they
  reorganize the algorithm or alter semantics.

**Related:**

- `plans/openblas_threading_and_odr_findings.md`
- `tests/benchmark_action_blas_backends.py`

---

## ACTION numerical decision stability

### Meaningful simplex support, not exact floating-point positivity

**Decision:**

- ACTION's private numerical constants are named in the shared internal
  `utils_action_numeric_policy.hpp` header so CPU and future accelerator
  implementations use the same decision policy.
- A simplex coefficient represents meaningful support only when it is
  strictly greater than `1e-6`. The same predicate is used for landmark
  reproducibility and trivial-membership counting.
- SPA tie handling, AA convergence and singularity handling, active-set
  regularization/optimality decisions, H-landmark proximity, specificity
  pruning, merge effective-rank rounding, and assignment argmax retain their
  existing values and behavior.
- Exact discrete ACTION decisions are the reproducibility contract for a
  controlled input. C/H matrices are compared numerically rather than
  bitwise. Seeded `reduce_kernel` output is a separate reproducibility
  boundary and is not promised bitwise-identical across BLAS builds.

**Rationale:**

- Recomputing the toy reduction under two MKL builds changed the normalized
  ACTION input by at most `7.22e-16`. One active-set coefficient consequently
  changed from exact zero to `1.73e-18`; the former `C > 0` landmark test
  treated that roundoff residue as biological support and retained one extra
  archetype.
- Any support cutoff from `1e-16` through `1e-6` removed the observed flip.
  `1e-6` is selected because simplex coefficients are dimensionless and the
  existing membership filter already uses that value.
- Controlled margin analysis found no crossing in SPA, specificity,
  H-landmark proximity, merge rank, or final assignment. Changing AA
  tolerances or iteration limits would alter the fitted model and is not a
  parity repair.
- Replacing one elementwise comparison has no meaningful runtime cost and may
  slightly reduce merge work for numerically unsupported archetypes.

**Related:**

- `plans/openblas_threading_fix_handoff.md`
- `tests/test_action_small_dense_kernels.py`

---

## GPU backend scope and platform

### NVIDIA CUDA backend: Python-first, SVD-first

**Decision:**

- GPU support is an optional execution backend, not a new public SVD algorithm.
- The first GPU target is Halko-style randomized SVD for dense/sparse and
  in-memory/backed inputs.
- Disk-backed GPU SVD is a first-class v1 requirement; do not implement GPU SVD
  by calling CPU `MatrixOperator::matmat` and copying the result to device.
- Native CUDA toolkit primitives are the default implementation direction.
  RAFT/RAPIDS may be evaluated only as an optional spike after the product and
  streaming boundaries exist.
- PRIMME and Feng have been deleted; they are not GPU routes.

**Platform contract:**

- Linux x86_64 with NVIDIA GPUs is the production/runtime target.
- Windows 11 + WSL2 with NVIDIA GPUs is the developer hardware validation
  target.
- CUDA 12.2 is the minimum toolkit target. CUDA 11.x is out of scope.
- Supported hardware starts at SM 8.0 / Ampere. Pre-Ampere GPUs are out of
  scope.
- macOS remains CPU-only. Native Windows outside WSL2 is out of scope.
- GPU support is disabled by default and must not change CPU-only builds.
- R-facing GPU API work is deferred; Python is the first supported front-end.

**Rationale:**

- The scrapped PRIMME/cuBLAS attempt showed that GPU support needs explicit
  host/device ownership, streaming boundaries, runtime canaries, and hardware
  validation.
- CUDA 12.2 and Ampere+ keep the target aligned with current HPC and WSL2
  development hardware without adding a legacy CUDA 11.x support burden.
- Keeping backend policy separate from algorithm choice preserves the simplified
  SVD public surface.

**Related:**

- `plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`
- `plans/GPU_INTEGRATION.md`
- `src/libactionet/plans/GPU_BACKEND_PLAN.md`

---

## Change management

### Native H5AD backed-I/O boundary

**Decision:**

- AnnData remains the sole public Python data model, but bulk backed dense,
  CSR, and CSC transfers are owned by a path-based `actionet::h5ad` C++ API.
- Native readers accept only explicitly supported H5AD matrix encoding
  versions and never depend on AnnData backed wrapper classes or SciPy
  indexing behavior.
- Transfer operations preserve exact source dtypes and values. Compute
  operators may continue to convert values to `double`.
- Python owns metadata serialization, AnnData-version adaptation, and one
  same-directory atomic rewrite transaction shared by subset, materialize,
  repack/decompress, normalization, checkpoint, and persistence paths.
- `backed_write_chunk_size` remains public as a maximum rows-per-batch. Native
  byte limits may lower the effective batch size.
- `ACTIONET_BACKED_IO_ENGINE=auto|native|python` is a private one-release
  rollback switch. Only pre-transfer capability rejection can fall back;
  runtime transfer failures abort the rewrite.
- Pre-transfer capability rejection raises a single canonical exception type,
  `actionet.io.native_h5ad.NativeCapabilityError` (a `RuntimeError` subclass),
  across the subset, normalization, and checkpoint/repack paths. Callers that
  fall back rely on the boolean return of the native-transaction helpers, not
  on catching a specific exception type.

**Rationale:**

- Historical AnnData and SciPy backed internals have changed independently of
  the stable HDF5 storage contract, causing version-sensitive behavior and
  position-dependent sparse-read performance.
- Workloads legitimately retain any fraction of either axis. A gap-aware
  selected-range/sequential-scan planner gives a bounded performance floor
  across that space while preserving arbitrary selector semantics.
- Keeping native code restricted to numeric matrix payloads avoids
  reimplementing AnnData metadata semantics and leaves a reusable C++ boundary
  for a future R binding.

**Related:**

- `plans/NATIVE_HDF5_BACKED_IO_IMPLEMENTATION_HANDOFF.md`
- `plans/NATIVE_HDF5_BACKED_IO_ROLLOUT.md`
- `src/libactionet/include/io/backed_h5ad/h5ad_matrix_io.hpp`
- `src/actionet/io/backed_adapter.py`
- `src/actionet/io/rewrite.py`

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
