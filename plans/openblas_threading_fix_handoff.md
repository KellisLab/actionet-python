# OpenBLAS ACTION performance fix handoff

Date: 2026-07-13

Status: implemented and locally validated on `dev-gpu`; changes are currently
uncommitted in both `actionet-python` and the `src/libactionet` submodule.

This document supersedes the handoff stored on `bad-fix`. That branch is a
failed reference and must remain unchanged. All work described here belongs on
`dev-gpu` in both repositories.

## Outcome

The OpenBLAS ACTION regression is fixed without BLAS-vendor detection, runtime
thread mutation, environment-variable requirements, or public API changes.

The original diagnosis was wrong: the slowdown was not specific to pthread
OpenBLAS oversubscription. Identical `dev-gpu` source measured 9.40 seconds for
`run_action` with MKL and 79.36 seconds with OpenBLAS-OpenMP. SPA was effectively
equal between backends; the gap began in archetypal analysis (AA), where the
active-set solver and AA update loop issue a very large number of tiny and
skinny BLAS calls.

The correction retains ACTION's existing coarse OpenMP parallelism across
archetype count `k`, but handles operations inline when the smaller matrix
dimension is at most 128. Larger general-purpose matrices still use the
configured BLAS or Armadillo implementation.

Final median-of-three results on the cached 6790-by-30 reduction from
`data/test_adata.h5ad` were:

| Backend | 1 thread | 2 | 4 | 8 | 16 | auto | Post-reduction pipeline, auto |
|---|---:|---:|---:|---:|---:|---:|---:|
| MKL | 53.72 s | 34.87 s | 21.19 s | 12.35 s | 10.01 s | 7.53 s | 13.73 s |
| OpenBLAS-pthread | 51.30 s | 35.01 s | 21.41 s | 12.10 s | 9.63 s | 7.00 s | 13.39 s |
| OpenBLAS-OpenMP | 53.65 s | 37.32 s | 22.35 s | 12.47 s | 9.48 s | 6.86 s | 13.56 s |

All assignments matched exactly. C and H outputs were numerically equivalent
at `rtol=1e-8`, `atol=1e-10` across thread counts, BLAS backends, and the
untouched pre-change MKL build.

## What was disproved

The `bad-fix` design used `BlasThreadScope`, `BlasGlobalThreadScope`, dynamic
symbol lookup, process-global thread setters, import warnings, and a public
`threading_info()` diagnostic.

OpenBLAS setters could change the value returned by
`openblas_get_num_threads()` without selecting the low-overhead execution path
after the library was initialized. A reported thread count of one therefore did
not demonstrate that skinny `DGEMV` or `DGER` calls used the serial path.

The following parts of `bad-fix` were intentionally not ported:

- per-call or per-region BLAS thread guards;
- `dlsym`-based BLAS detection and setter discovery;
- import-time OpenBLAS warnings;
- the Python `threading_info()` API and private probes;
- broad guards around unrelated OpenMP call sites.

Official OpenBLAS references are linked from
`plans/openblas_threading_and_odr_findings.md`.

## Implemented changes

### `libactionet`: private small-dense kernel layer

Added:

- `include/utils_internal/utils_small_dense.hpp`

This is a private, header-only, column-major utility containing deterministic,
vectorizable implementations of:

- copy;
- dot product;
- scale;
- axpy;
- symmetric matrix-vector multiplication;
- general matrix-vector multiplication;
- rank-one update;
- Gram calculation;
- residual product;
- Frobenius norm;
- clamp and column normalization.

The common dispatch policy is:

```text
min(rows, columns) <= 128  -> internal inline kernel
min(rows, columns) > 128   -> existing CBLAS/Armadillo path
```

This threshold covers default ACTION reductions, which are small in at least
one dimension, without replacing BLAS for genuinely large dense operations.
There is no vendor-specific behavior in the dispatch.

### `libactionet`: active-set solvers

Changed:

- `src/utils_internal/utils_active_set.cpp`

Both `activeSet_arma` and cached-Gram `activeSetS_arma` now route all former
CBLAS operations through the small-dense layer. This includes copies, dot
products, scaling, axpy, symmetric products, matrix-vector products, and
rank-one inverse updates.

The active-set algorithm itself was not redesigned. The following remain
unchanged:

- constraint and active-set logic;
- direct inverse update/downdate method;
- regularization;
- iteration limits;
- convergence decisions;
- input/output orientation;
- one solver invocation per right-hand side.

### `libactionet`: simplex regression

Changed:

- `src/action/simplex_regression.cpp`

The cached solver's Gram calculation and final clamp/normalization now use the
shared shape-aware utility. Cached and non-cached solvers retain their existing
interfaces and semantics.

### `libactionet`: AA hot path

Changed:

- `src/action/aa.cpp`

The shape-aware kernels now cover the complete default AA hot path:

- `A - W*H` residual construction;
- `R*h`;
- `A*C`;
- residual rank-one updates;
- repeated dot products and norms;
- archetype normalization.

The outer AA loop, reseeding behavior, tolerance handling, and returned C/H
matrices are unchanged.

### `libactionet`: nested OpenMP ownership

Changed:

- `include/utils_internal/utils_parallel.hpp`
- `src/action/action_decomp.cpp`

An internal thread-local `OuterParallelRegionScope` marks work owned by
ACTION's outer decomposition. The existing nested-safe thread helper now also
checks the OpenMP nesting level and this explicit marker.

This fixes an edge case observed with Intel OpenMP: a serialized
`num_threads(1)` region can report `omp_in_parallel() == false`, which allowed
a helper to create an unintended inner team. This is an ACTION/OpenMP ownership
fix, not a BLAS thread guard.

### Decision records and findings

Changed in both repositories:

- `context/DECISIONS.md`
- `src/libactionet/context/DECISIONS.md`

The records now establish shape-specialized kernels under coarse ACTION OpenMP
parallelism as the accepted policy. They explicitly reject runtime BLAS thread
mutation and defer algorithmic redesign.

Replaced/corrected:

- `plans/openblas_threading_and_odr_findings.md`

That report contains the controlled diagnosis, isolation measurements,
OpenBLAS setter evidence, final timing table, acceptance gates, and the updated
ODR record. PRIMME-specific ODR warnings are now historical because PRIMME was
deleted; they are not the cause of the ACTION regression.

## Tests, benchmark, and CI

### Deterministic correctness tests

Added:

- `tests/test_action_small_dense_kernels.py`

Coverage includes:

- cached and non-cached simplex solvers;
- simplex feasibility and finite outputs;
- duplicate and zero columns;
- a skinny inline case;
- the exact 128-dimension cutoff;
- a 129-by-130 case that forces the BLAS fallback;
- agreement between solver variants at `rtol=1e-8`, `atol=1e-10`;
- single- versus multithreaded `runAA`, `decompose_action`, and `run_action`;
- identical assignments and equivalent C/H matrices;
- an opt-in synthetic catastrophic-regression smoke test.

The timing smoke deliberately starts OpenBLAS with 32 threads in CI. It is
there to prove that correctness and completion no longer depend on the
benchmark's controlled one-thread vendor pool.

Added pytest markers in `pyproject.toml`:

- `benchmark`;
- `openblas_smoke`.

### Linux OpenBLAS CI

Added:

- `.github/workflows/openblas-action.yml`

The focused Ubuntu workflow builds and tests both apt OpenBLAS variants:

- pthread;
- OpenMP.

It verifies the resolved OpenBLAS path and GNU OpenMP linkage before running
the correctness and catastrophic-regression tests.

### Reproducible backend benchmark

Added:

- `tests/benchmark_action_blas_backends.py`

The benchmark:

- creates or reuses one cached reduction from `data/test_adata.h5ad`;
- verifies extension linkage with both `readelf` and `ldd`;
- resolves the OpenBLAS pthread/OpenMP variant rather than trusting a generic
  symlink;
- uses medians of three trials;
- sweeps ACTION thread counts `{1,2,4,8,16,auto}`;
- benchmarks `run_action` and optionally the post-reduction default pipeline;
- checks determinism within a backend;
- saves and compares cross-backend parity artifacts;
- enforces the documented timing and scaling gates.

The benchmark sets vendor BLAS thread pools to one before importing NumPy or
ACTIONet. This is experimental control: the thread sweep measures ACTION's
explicit OpenMP decomposition. It is not a package runtime requirement.

Example validation sequence:

```bash
python tests/benchmark_action_blas_backends.py \
  --mode both --expect-backend mkl --enforce-acceptance \
  --parity-output /tmp/action-mkl-parity.npz \
  --output /tmp/action-mkl.json

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openblas-pthread \
python tests/benchmark_action_blas_backends.py \
  --mode both --expect-backend openblas \
  --expect-openblas-threading pthread \
  --timing-reference /tmp/action-mkl.json \
  --parity-reference /tmp/action-mkl-parity.npz \
  --enforce-acceptance \
  --output /tmp/action-openblas-pthread.json

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openblas-openmp \
python tests/benchmark_action_blas_backends.py \
  --mode both --expect-backend openblas \
  --expect-openblas-threading openmp \
  --timing-reference /tmp/action-mkl.json \
  --parity-reference /tmp/action-mkl-parity.npz \
  --enforce-acceptance \
  --output /tmp/action-openblas-openmp.json
```

Acceptance gates are:

1. Both OpenBLAS variants complete `run_action` and the post-reduction default
   pipeline within 1.5 times the same-source MKL median.
2. MKL `run_action` is no more than 10% slower than the measured 9.40-second
   baseline.
3. Automatic ACTION threading is at least twice as fast as one ACTION thread
   and no more than 20% slower than the best explicit multithread count.
4. Assignments match exactly and C/H matrices meet the numerical tolerances.

All four gates passed on the validation machine.

## Public and behavioral contract

- No Python API changed.
- No public C++ API changed.
- No output schema or matrix orientation changed.
- No environment variable is required for correct package performance.
- OpenMP remains mandatory and owns coarse ACTION parallelism.
- The active-set algorithm and convergence policy remain unchanged.
- Larger matrices retain BLAS dispatch.

## Current workspace and commit order

Both repositories are currently on `dev-gpu`:

```text
actionet-python HEAD: e0021f5
libactionet HEAD:     10f180a
```

Current parent-repository changes:

```text
M  context/DECISIONS.md
M  plans/openblas_threading_and_odr_findings.md
M  pyproject.toml
m  src/libactionet
?? .github/workflows/openblas-action.yml
?? plans/openblas_threading_fix_handoff.md
?? tests/benchmark_action_blas_backends.py
?? tests/test_action_small_dense_kernels.py
```

Current `src/libactionet` changes:

```text
M  context/DECISIONS.md
M  include/utils_internal/utils_parallel.hpp
M  src/action/aa.cpp
M  src/action/action_decomp.cpp
M  src/action/simplex_regression.cpp
M  src/utils_internal/utils_active_set.cpp
?? include/utils_internal/utils_small_dense.hpp
```

Commit and push the submodule changes first. Then update and commit the
submodule pointer together with the parent-repository tests, workflow, reports,
and decision record. A parent commit pointing to an unpushed submodule commit
will make CI checkout fail.

Do not merge or cherry-pick the runtime-guard implementation from `bad-fix`.

## Interpretation of the current optimization

The current fix can look like an active-set refactor because most changed lines
are in the active-set implementation. More precisely, it is a dispatch
refactor: the same solver now calls an internal shape-aware kernel layer instead
of calling CBLAS directly for every primitive.

For the default skinny shapes, this intentionally leaves little work to a BLAS
library. The measured issue was the granularity of BLAS dispatch, not its
floating-point throughput. For matrices whose two dimensions exceed 128, BLAS
still owns the operation.

This is a good tactical CPU fix and a useful reference implementation, but its
fine-grained operation sequence should not be used as the abstraction for a
future GPU backend. A direct CUDA translation would risk thousands of tiny
cuBLAS launches, host/device synchronization for active-set decisions, and
repeated temporary allocation or data transfer.

## Recommended future ACTION/SPA refactor

The portability goal should be to define backend boundaries at batched or
fused algorithmic units, not to add virtual dispatch around `dot`, `axpy`, and
`gemv`.

| Present implementation unit | Candidate backend unit |
|---|---|
| One dot/axpy/gemv/rank-one call | One batched simplex solve |
| One active-set solve per right-hand side | A batch sharing design matrix and Gram data |
| Separate AA residual operations | Fused or blocked AA iteration/update |
| Per-call temporary Armadillo objects | Persistent backend-owned workspace |
| SPA projection/update primitives | One SPA projection/update operation |
| Host reduction followed by a branch | Batched/device reduction with an explicit synchronization boundary |

Recommended sequence:

1. **Land and freeze this fix as the CPU reference.** Preserve the current
   outputs, benchmark artifacts, and small-dense implementation as the golden
   fallback.
2. **Define coarse internal interfaces without changing the public API.**
   Candidate concepts are `solve_simplex_batch`, `aa_update_block`,
   `spa_projection_update`, and backend-owned workspace. Names are not settled.
3. **Implement the interfaces with the current CPU algorithm first.** Require
   assignment identity, C/H tolerances, and no meaningful CPU regression before
   adding accelerator code.
4. **Add a CUDA implementation around resident data and batching.** Keep AA
   matrices and workspaces on device across iterations where possible. Use
   batched GEMM/GEMV or custom fused kernels only where profiling supports
   them, and minimize host synchronization.
5. **Evaluate solver changes separately.** Do not combine a convergence-policy
   or simplex-algorithm change with the backend extraction.

### Active-set solver options

The current active-set method is the hardest component to map efficiently to a
GPU because each right-hand side can have a different active set, iteration
count, inverse update, and branch path.

Two distinct approaches should be evaluated:

- **Batched/masked active set:** preserve the current solver closely, batch
  right-hand sides sharing a design/Gram matrix, and tolerate or reduce branch
  divergence. This is the lower semantic-risk option.
- **Alternative simplex solver:** projected gradient, accelerated projected
  gradient/FISTA, ADMM, mirror descent, or another batch-friendly formulation.
  This may fit GPUs better, but it changes convergence and numerical behavior
  and must be treated as a separate algorithmic decision.

The near-term interface should therefore support a batch of right-hand sides
without making the active-set implementation itself part of the public
contract. Keep the current solver as the CPU/reference backend while any
alternative is prototyped and compared.

### AA-specific opportunities

- Cached H solves share the same design matrix and Gram matrix across many
  right-hand sides; this is the clearest batching opportunity.
- Residual formation, candidate archetype updates, norms, and normalization can
  be blocked or fused to reduce memory traffic.
- Work arrays and active-set buffers can be reused across iterations rather
  than repeatedly allocated.
- The outer decomposition over `k` is appropriate CPU work ownership but need
  not be the GPU kernel boundary. A GPU scheduler may use streams or schedule
  decompositions subject to device-memory limits.

### SPA-specific opportunities

SPA was not responsible for the OpenBLAS regression, so it did not need changes
for this fix. It should nevertheless use the same future backend/workspace
model. A GPU path should express projection and residual updates as coarse
operations rather than launch a separate vendor kernel for every vector
primitive.

### Refactor guardrails

- Keep execution backend separate from algorithm selection.
- Preserve CPU-only and macOS builds; CUDA remains optional.
- Make host/device ownership and synchronization explicit.
- Do not implement a GPU backend by running CPU products and copying each
  result to the device.
- Avoid per-primitive polymorphism in the hot path.
- Keep convergence-policy experiments out of a semantics-preserving backend
  refactor.
- Validate on real NVIDIA hardware; device discovery alone is not a sufficient
  canary.
- Continue to require cross-backend assignment identity and C/H tolerances
  unless a separately approved algorithm change establishes a new contract.

This direction is consistent with:

- `src/libactionet/plans/GPU_BACKEND_PLAN.md`;
- `plans/GPU_INTEGRATION.md`;
- `plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`.

Those documents already establish the broader principle that GPU support is an
execution backend and that CPU/GPU differences belong at meaningful product,
streaming, workspace, and data-residency boundaries.

## Suggested next task

Before implementing GPU AA, create a focused ACTION backend redesign plan and
prototype with these deliverables:

1. a profile quantifying solve counts, right-hand-side batch sizes,
   allocations, and synchronization points for AA and SPA;
2. internal batch/workspace interfaces with a CPU implementation;
3. parity and CPU-performance gates against this implementation;
4. a real-hardware CUDA spike comparing batched active set with at least one
   batch-friendly simplex method;
5. a decision record selecting the solver and GPU scheduling granularity only
   after numerical and performance evidence exists.

The existing OpenBLAS fix should not be removed during that exploration. It is
the performant CPU fallback, the correctness reference, and the baseline that
the refactor must beat or match.
