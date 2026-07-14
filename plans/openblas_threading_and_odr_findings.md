# OpenBLAS ACTION performance and Armadillo ODR findings

Status: confirmed and corrected, 2026-07-13. Applies to `actionet-python`
and the shared `libactionet` core.

## Executive summary

The original report attributed the ACTION slowdown to pthread-OpenBLAS
oversubscription and recommended runtime thread guards. Controlled builds
disproved that diagnosis:

- The exact `dev-gpu` source completed `run_action` in **9.40 s with MKL**
  and **79.36 s with OpenBLAS-OpenMP** on the same cached reduction and
  machine. The OpenMP build therefore retained an 8.4x regression without
  an independent pthread pool.
- SPA alone was backend-neutral (approximately 0.55-0.58 s). The regression
  started in archetypal analysis, specifically its high-frequency tiny and
  skinny BLAS operations.
- Runtime guards changed the thread count reported by OpenBLAS but did not
  select the low-overhead serial execution path after library initialization.
  They added global-state and portability complexity without fixing runtime.

The implemented correction keeps coarse parallelism in ACTION's existing
OpenMP loop over `k` and executes dense operations inline when the smaller
matrix dimension is at most 128. Larger, general-purpose matrices retain the
configured CBLAS/Armadillo path. No public API, environment variable, or
backend detection is involved.

## Reproducer and controlled baseline

The controlled input is one reduction cached from
`data/test_adata.h5ad` (6790 observations by 32236 features):

```text
reduction shape: 6790 x 30
ACTION defaults: k=2..30, max_iter=50, tolerance=1e-100
thread sweep:    1, 2, 4, 8, 16, auto
timing statistic: median of three trials
```

Every compared extension must come from the same source revision. ELF
dependencies are verified with both `readelf -d` and `ldd`; following a
generic `libopenblas.so.0` symlink without resolving it is insufficient to
distinguish the pthread and OpenMP packages.

Before the kernel change, the same-source measurements were:

| Build | `run_action` |
|---|---:|
| MKL + Intel OpenMP | 9.40 s |
| OpenBLAS-OpenMP + GNU OpenMP | 79.36 s |

The post-reduction default pipeline took 15.07 s in the MKL control. An older
installed MKL build took 8.47 s for `run_action`, which independently bounds
normal run-to-run/build variation.

## Isolation evidence

### SPA is not responsible

Running the selection stage with zero AA iterations was effectively equal:

| Backend | SPA / zero-AA runtime |
|---|---:|
| MKL | about 0.55 s |
| OpenBLAS | about 0.58 s |

The regression begins after `runAA` enters simplex regression and its active
set updates.

### Active-set dispatch is the hotspot

The default shapes are small in one dimension but invoked extremely often:

- cached `H` solves use a feature-by-archetype design (typically 30 by `k`),
- non-cached `C` solves use a 30 by 6790 design,
- each active-set iteration performs repeated copy, dot, scale, axpy, symv,
  gemv, and rank-one updates,
- AA additionally performs `A-WH`, `R*h`, `A*C`, residual rank-one updates,
  norms, and normalizations.

OpenBLAS cached-solver time became worse as independent outer work increased,
while MKL scaled in the expected direction:

| Concurrent outer workers | OpenBLAS cached solve | MKL cached solve |
|---:|---:|---:|
| 1 | 0.0378 s | 0.0690 s |
| 2 | 0.0658 s | 0.0440 s |
| 4 | 0.0947 s | 0.0349 s |
| 8 | 0.1418 s | 0.0363 s |
| 16 | 0.1456 s | 0.0352 s |
| 32 | 0.1787 s | 0.0432 s |

This is library-dispatch/coordination overhead on tiny products, not a lack
of arithmetic throughput. Large BLAS operations are not implicated by this
profile and should continue to use optimized vendor libraries.

## Why the thread guards failed

OpenBLAS documents its thread controls as configuration used when the library
is initialized. In the direct reproducer, calling a setter after initialization
changed `openblas_get_num_threads()` to one, yet repeated skinny `DGEMV` and
`DGER` calls remained around 8-9 ms. Starting the process with OpenBLAS already
configured for one thread made the same calls roughly 0.03-0.10 ms.

Consequently, a per-scope setter is not a reliable per-call kernel selector.
It may mutate process-global state while concurrent ACTION workers are active,
and its apparent diagnostic state does not prove that the low-overhead path is
being used. The failed `bad-fix` branch's `BlasThreadScope`,
`BlasGlobalThreadScope`, `dlsym` probes, import warning, and `threading_info()`
API are intentionally not carried forward.

Official references:

- [OpenBLAS usage and thread controls](https://github.com/OpenMathLib/OpenBLAS)
- [OpenBLAS threading FAQ](https://github.com/OpenMathLib/OpenBLAS/wiki/Faq/08848f2293927444abf06eec756f0fc17b33313f)

## Implemented kernel policy

`libactionet/include/utils_internal/utils_small_dense.hpp` is a private,
header-only column-major kernel layer. It supplies deterministic/vectorizable
copy, dot, scale, axpy, symmetric matvec, gemv, rank-one update, Gram,
residual-product, Frobenius norm, and column normalization operations.

The policy is shape-based and backend-independent:

```text
min(rows, columns) <= 128  -> internal inline kernel
min(rows, columns) > 128   -> existing BLAS/Armadillo implementation
```

The complete default AA hot path and both active-set solver variants use this
policy. Algorithm structure, active-set iteration limits, regularization,
convergence checks, matrix orientation, and the outer OpenMP decomposition
over `k` are unchanged.

The final median-of-three validation on the exact cached reduction was:

| Backend | 1 | 2 | 4 | 8 | 16 | auto | Pipeline auto |
|---|---:|---:|---:|---:|---:|---:|---:|
| MKL | 53.72 s | 34.87 s | 21.19 s | 12.35 s | 10.01 s | 7.53 s | 13.73 s |
| OpenBLAS-pthread | 51.30 s | 35.01 s | 21.41 s | 12.10 s | 9.63 s | 7.00 s | 13.39 s |
| OpenBLAS-OpenMP | 53.65 s | 37.32 s | 22.35 s | 12.47 s | 9.48 s | 6.86 s | 13.56 s |

The benchmark initializes vendor BLAS pools at one thread before loading the
extension so the sweep measures ACTION's explicit OpenMP count. A separate CI
smoke test initializes OpenBLAS at 32 threads to ensure the product no longer
depends on that benchmark control.

All assignments matched exactly. C and H outputs passed the stated numerical
tolerances across thread counts, both OpenBLAS variants, MKL, and the untouched
pre-change `dev-gpu` MKL binary. The one-thread Intel OpenMP run also exposed a
serialized-region edge case in the existing nested helper; an internal
thread-local outer-region marker now prevents inner OpenMP teams even when a
runtime reports a `num_threads(1)` region as inactive.

## Acceptance and reproducibility

`tests/benchmark_action_blas_backends.py`:

- creates or reuses exactly one cached reduction from the toy H5AD,
- verifies ELF linkage and the resolved OpenBLAS threading variant,
- runs three trials for `{1,2,4,8,16,auto}`,
- benchmarks `run_action` and optionally the full post-reduction pipeline,
- checks assignment identity and C/H equivalence (`rtol=1e-8`, `atol=1e-10`),
- can compare OpenBLAS reports and parity artifacts against an MKL reference.

Example full-gate sequence (run each command from its corresponding installed
wheel environment):

```bash
python tests/benchmark_action_blas_backends.py \
  --mode both --expect-backend mkl --enforce-acceptance \
  --parity-output /tmp/action-mkl-parity.npz --output /tmp/action-mkl.json

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openblas-pthread \
python tests/benchmark_action_blas_backends.py \
  --mode both --expect-backend openblas --expect-openblas-threading pthread \
  --timing-reference /tmp/action-mkl.json \
  --parity-reference /tmp/action-mkl-parity.npz --enforce-acceptance \
  --output /tmp/action-openblas-pthread.json

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openblas-openmp \
python tests/benchmark_action_blas_backends.py \
  --mode both --expect-backend openblas --expect-openblas-threading openmp \
  --timing-reference /tmp/action-mkl.json \
  --parity-reference /tmp/action-mkl-parity.npz --enforce-acceptance \
  --output /tmp/action-openblas-openmp.json
```

Local acceptance gates are:

1. pthread and OpenMP OpenBLAS medians are no more than 1.5x the same-source
   MKL median for `run_action` and the post-reduction default pipeline;
2. MKL `run_action` is no more than 10% slower than the 9.40 s baseline;
3. automatic threading is at least 2x faster than one ACTION thread and no
   more than 20% slower than the best tested explicit multithread count;
4. assignments match exactly and C/H matrices meet the tolerances above.

All four gates passed on the validation machine. The OpenBLAS automatic
`run_action` medians were 0.91-0.93x MKL, full-pipeline medians were
0.98-0.99x MKL, MKL improved by 19.9% from its 9.40 s baseline, and automatic
threading was 7.1-7.8x faster than one ACTION thread.

The focused Linux workflow builds both apt OpenBLAS variants and runs the
deterministic solver/thread tests plus a loose catastrophic-regression smoke
test.

## Deferred near-MKL-parity redesign

The current change deliberately preserves semantics. A subsequent measured
optimization project may pursue:

1. batched active-set solves so columns share control flow and data movement;
2. blocked/fused AA residual and archetype updates;
3. workspace reuse across iterations and archetypes;
4. convergence-policy evaluation, with explicit accuracy and iteration-count
   evidence before changing any stopping behavior.

Those items can improve cache locality and reduce allocations, but they alter
algorithm organization or policy and are not part of this fix.

## Armadillo / PRIMME ODR record

The earlier report also documented LTO type-mismatch warnings between PRIMME's
private BLAS/LAPACK declarations and Armadillo, plus inconsistent Armadillo
atomic state definitions across translation units. PRIMME and its vendored
sources have since been deleted, so the PRIMME-specific warning class is
obsolete and is not related to the OpenBLAS ACTION regression.

The general rule remains: every translation unit consuming the packaged
Armadillo headers must receive consistent compile definitions. Any future ODR
warning should be treated as a correctness issue, but no runtime BLAS-thread
API or ACTION behavior should be added on the basis of that historical defect.
