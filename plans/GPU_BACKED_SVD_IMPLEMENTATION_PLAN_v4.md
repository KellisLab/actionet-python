# GPU-backed SVD Implementation Plan (v4)

Date: 2026-07-12

Status: planning document. This is not an implementation-ready specification. It is intended to align the next implementation pass after PRIMME and Feng were removed from the Python SVD surface and IRLB/Halko were extended toward 64-bit matrix support.

## Executive Summary

GPU support should be added as a backend for the existing SVD strategy, not as a new public SVD algorithm. The public Python algorithm surface should remain:

- `auto`
- `irlb`
- `halko`

The first production GPU path should accelerate Halko-style randomized SVD for both in-memory and disk-backed inputs. Disk-backed support must be designed from the beginning, because that is where GPU acceleration is most likely to matter for real workloads.

Recommendation:

- Keep IRLB as the CPU sparse in-memory default.
- Keep Halko as the dense and backed default.
- Do not restore PRIMME or Feng to the Python API.
- Do not use PRIMME as the GPU path.
- Make native CUDA/cuBLAS/cuSPARSE/cuSOLVER the primary production direction.
- Treat RAFT/RAPIDS SVD as an optional reference or performance spike, not as a required dependency, unless a later bakeoff justifies the build and environment cost.

## Current Codebase Audit

### Python SVD Surface

Current public Python behavior is already aligned with the simplified CPU strategy:

- `src/actionet/decomposition/svd.py`
  - `_SVD_ALGORITHM_TO_ID = {"irlb": 0, "halko": 1}`
  - `_normalize_algorithm()` rejects anything outside `auto`, `irlb`, and `halko`.
  - In-memory `auto` selects IRLB for sparse inputs and Halko for dense inputs.
  - Backed `auto` selects Halko.
  - There are no GPU kwargs or execution-policy hooks yet.

- `src/actionet/decomposition/kernel.py`
  - Mirrors the same algorithm restrictions.
  - Records SVD algorithm metadata, but not a resolved execution backend.
  - `reduce_kernel` remains CPU-oriented beyond the SVD subcall.

This is a good public API baseline. GPU support should add an execution backend selector, not expand the algorithm set.

### C++ SVD State

The C++ layer still carries retired algorithms in isolation:

- `src/libactionet/include/decomposition/svd_main.hpp`
  - Still defines `ALG_FENG = 2` and `ALG_PRIMME = 3`.
  - Some comments still describe Feng/PRIMME as ordinary SVD algorithms.
  - Some comments refer to backed IRLB block-solver hooks that have been removed.

- `src/libactionet/src/decomposition/svd_main.cpp`
  - Still dispatches Feng and PRIMME if called directly with those integer IDs.
  - `runSVD_Operator(..., ALG_IRLB, ...)` now calls operator IRLB directly instead of routing through PRIMME.

- `src/actionet/bindings/wp_decomposition.cpp` and `src/actionet/bindings/wp_io.cpp`
  - Binding functions accept raw integer algorithm IDs.
  - The Python public wrapper prevents `2` and `3`, but direct `_core` callers could still reach quarantined C++ algorithms by integer ID.

Preflight cleanup should add binding-level validation so the Python extension cannot reach Feng/PRIMME accidentally even through private `_core` calls.

### 64-bit CPU Status

IRLB has been made substantially more 64-bit aware:

- `src/libactionet/src/decomposition/svd_irbla.cpp`
  - Sparse `nnz > 2^31 - 1` is treated as supported under `ARMA_64BIT_WORD`.
  - Explicit guards reject row or column dimensions above `INT_MAX`.
  - Dense, sparse, and operator overloads all perform axis checks.

Halko still needs a focused 64-bit audit before GPU work depends on it:

- `src/libactionet/src/decomposition/svd_halko.cpp`
  - The in-memory template uses `arma::uword` in places, but still casts dimensions to `int` for random sketch allocation and some output.
  - The operator overload narrows row and column counts to `int`.

This does not block GPU planning, but it should be fixed or explicitly guarded before claiming full 64-bit Halko support. The CPU and GPU paths should share the same dimension policy.

Recommended policy for v1:

- Support 64-bit `nnz` for sparse inputs where the storage and CUDA backend support it.
- Keep per-axis dimensions limited to `INT_MAX` unless a later project explicitly widens every downstream dense workspace and BLAS interaction.
- Make all axis-limit failures explicit and early.

### Backed Matrix State

Backed inputs currently reach SVD through CPU `MatrixOperator` implementations:

- `src/libactionet/include/decomposition/matrix_operator.hpp`
  - Provides CPU host `matmat` and `rmatmat` operations using Armadillo matrices.
  - No longer contains the backed IRLB block-solver preference hook.

- `BackedDenseMatrixOperator`
  - Has slab-reading functionality that can help seed a future chunk source.

- `BackedSparseMatrixOperator`
  - Encapsulates chunk details more tightly.
  - A GPU implementation should not depend on private implementation details leaking through the current `MatrixOperator` abstraction.

The current CPU `MatrixOperator` should not become the GPU boundary. It is too host-matrix centric and would recreate the architectural mismatch seen in the failed PRIMME GPU attempt.

### Current GPU Documents

There are stale documentation pointers:

- `plans/GPU_INTEGRATION.md` points to `../src/libactionet/context/GPU_BACKEND_PLAN.md`, but the live plan is under `src/libactionet/plans/GPU_BACKEND_PLAN.md`.
- `README.md` and `docs/architecture.md` also reference old GPU plan paths.

Preflight cleanup should update these links and reduce detail in older GPU docs that could mislead future agents.

## Review of the Attached v3 Plan

The attached plan is directionally strong in three important ways:

- It keeps GPU as a backend, not a new public algorithm.
- It makes disk-backed GPU support first-class.
- It delays Python API exposure until dense/sparse and in-memory/backed coverage is coherent.

Recommended revisions:

- Make native CUDA the primary production path.
- Move RAFT to an optional spike or reference implementation.
- Avoid relying on direct HDF5-to-device reads in v1. Plan for host chunk reads plus explicit host-to-device transfer first.
- Add preflight cleanup for stale comments, stale plan links, binding-level algorithm validation, and the remaining Halko 64-bit audit.
- Make `reduce_kernel` semantics explicit: v1 may pass GPU SVD options through, but only the SVD subcall is GPU-eligible unless later work accelerates the rest of kernel reduction.

## Design Principles

### Keep One SVD Strategy

The package should not grow a separate GPU SVD method. GPU is an execution backend for Halko:

```text
algorithm = auto | halko | irlb
backend   = auto | cpu | cuda
```

Proposed behavior:

- `algorithm="irlb", backend="cuda"` should either raise a clear unsupported error or fall back only if the user explicitly allows fallback.
- `algorithm="halko", backend="cuda"` should attempt the GPU Halko backend.
- `algorithm="auto", backend="cuda"` should resolve to Halko for v1.
- `backend="auto"` should preserve current CPU defaults unless GPU opt-in is added deliberately.

### Treat Disk-backed as a First-class Input Type

The architecture should be designed around four input forms from the start:

- dense in-memory
- sparse in-memory
- dense disk-backed
- sparse disk-backed

The implementation can still land in phases, but the abstraction boundary should be chosen so all four forms share the same Halko driver and differ only in product/chunk backends.

### Separate Product Semantics from Storage

Halko mainly needs repeated products with dense sketch blocks:

- `Y = A * Omega`
- optional power iterations involving `A.T * Q` and `A * Z`
- small dense QR/SVD steps

The GPU design should expose these products through a backend interface rather than exposing raw storage classes to the algorithm.

Proposed conceptual layering:

```text
Python API
  -> SVD dispatcher
    -> Halko randomized SVD driver
      -> SVD product backend
        -> CPU dense/sparse products
        -> CPU backed products
        -> CUDA dense/sparse products
        -> CUDA backed dense/sparse streaming products
```

The exact class names can change. The important point is that backed dense/sparse inputs and in-memory dense/sparse inputs should converge before the randomized SVD logic, not after it.

## Recommended Production Backend

### Primary: Native CUDA

Native CUDA should be the default production direction because it fits the current project constraints:

- The repository is C++17-oriented.
- The project avoids heavy or hard-to-build dependencies where possible.
- CUDA libraries can be optional and isolated behind CMake flags.
- cuSPARSE supports sparse matrix products with 32-bit and 64-bit sparse indices.
- cuBLAS/cuSOLVER cover dense products, QR/SVD building blocks, and small dense workspaces.

Native CUDA also gives direct control over disk-backed streaming, memory ownership, pinned host buffers, batching, precision, and fallback behavior.

Expected CUDA libraries:

- cuBLAS for dense matrix products and small dense operations.
- cuSPARSE for sparse-times-dense products.
- cuSOLVER for QR/SVD pieces where appropriate.
- CUDA runtime for streams, pinned buffers, memory accounting, and device selection.

`cusolverDnXgesvdr` may be useful as a dense in-memory experiment, but it should not be the central v1 design. It expects a dense device matrix and destroys `A` on exit, so it does not solve sparse or disk-backed streaming SVD by itself.

### Optional: RAFT/RAPIDS Spike

RAFT should be evaluated as an optional spike, not a default dependency:

- RAFT has a sparse randomized SVD implementation that resembles the desired Halko algorithm.
- Its operator interface is relevant as a design reference.
- Current RAFT downstream builds require CUDA language support and C++20/CUDA20.
- RAPIDS/RAFT introduces additional packaging and environment constraints that may conflict with the project's compatibility goals.

Suggested policy:

- Add a separate experimental flag only after the native product interface exists, for example `LIBACTIONET_ENABLE_RAFT_SVD_EXPERIMENT`.
- Do not let RAFT raise the default C++ standard for the package.
- Do not expose RAFT as a user-visible algorithm.
- Require explicit validation for 64-bit sparse `nnz` and backed sparse cases before considering RAFT production-worthy.

## Proposed Implementation Stages

### Stage 0: Preflight Cleanup and Contracts

Goal: make the CPU-only post-refactor state unambiguous before adding GPU complexity.

Tasks:

- Fix stale GPU plan links in `plans/GPU_INTEGRATION.md`, `README.md`, and `docs/architecture.md`.
- Update stale SVD comments in `src/libactionet/include/decomposition/svd_main.hpp`.
- Add binding-level validation in `wp_decomposition.cpp` and `wp_io.cpp` so Python `_core` entrypoints reject retired algorithm IDs.
- Finish or guard Halko 64-bit support consistently:
  - explicit axis checks;
  - no silent dimension narrowing;
  - documented `nnz` and per-axis limits.
- Add tests for direct `_core` rejection of retired algorithm IDs if private `_core` tests are acceptable.
- Add metadata vocabulary for resolved SVD backend, even before GPU is exposed.

Exit criteria:

- CPU tests still pass with only `auto`, `irlb`, and `halko`.
- Retired algorithms cannot be reached from the Python extension.
- 64-bit policy is documented and enforced consistently for IRLB and Halko.

### Stage 1: Execution Policy and Optional CUDA Build Skeleton

Goal: add the minimum backend-selection infrastructure without changing CPU behavior.

Conceptual API:

```text
backend = "auto" | "cpu" | "cuda"
gpu_options = {
  device: int | None,
  fallback: bool,
  memory_limit: int | None,
  stream_count: int | None,
}
```

The exact Python signature can be deferred, but the C++ and Python dispatchers should have a place to carry this policy.

Tasks:

- Add a small execution-policy struct in C++.
- Add a CUDA availability probe behind an optional CMake flag.
- Add clear runtime errors for unavailable CUDA, unsupported algorithm/backend combinations, and out-of-memory conditions.
- Keep CPU builds completely unaffected when GPU is disabled.
- Add a tiny CUDA canary target or test that initializes CUDA and runs a trivial cuBLAS/cuSPARSE operation.

Exit criteria:

- Default build remains CPU-only.
- GPU-disabled builds do not include CUDA headers.
- GPU-enabled builds can report device availability and fail cleanly.

### Stage 2: Refactor Halko Around Product Backends

Goal: make CPU Halko use the same product abstraction that GPU will use later.

Tasks:

- Extract the repeated `A * block` and `A.T * block` operations from `svd_halko.cpp`.
- Keep the randomized SVD driver responsible for:
  - random sketch generation;
  - orthogonalization;
  - power iterations;
  - small final dense SVD;
  - deterministic seeding behavior.
- Move storage-specific products into product backends:
  - dense in-memory CPU;
  - sparse in-memory CPU;
  - backed dense CPU;
  - backed sparse CPU.
- Preserve current numerical output and benchmark behavior.

Exit criteria:

- CPU Halko parity tests pass for dense, sparse, backed dense, and backed sparse inputs.
- No GPU code is required for this stage.
- Current `auto` behavior is unchanged.

### Stage 3: Backed Chunk Source Abstraction

Goal: make disk-backed data streamable for both CPU and GPU product backends.

Tasks:

- Introduce a chunk/slab source abstraction separate from `MatrixOperator`.
- Preserve all backed transforms currently applied by dense and sparse backed operators.
- Support row-major and column-major access patterns needed for `A * block` and `A.T * block`.
- Add chunk-size policy and memory accounting.
- For sparse backed inputs, expose chunked CSR/CSC-like views without forcing all data into memory.
- Keep v1 transfers simple: HDF5 chunk -> host buffer -> optional pinned host buffer -> device buffer.

Avoid in v1:

- Assuming HDF5 can read directly into device memory.
- Coupling CUDA code to private internals of current backed operators.
- Adding a separate backed-only SVD algorithm.

Exit criteria:

- CPU backed Halko can be driven through the new chunk source.
- Tests confirm backed dense and sparse products match existing `MatrixOperator` behavior.

### Stage 4: Native CUDA In-memory Halko

Goal: implement GPU Halko for in-memory dense and sparse matrices using the product backend interface.

Tasks:

- Add CUDA resource management:
  - device selection;
  - streams;
  - cuBLAS/cuSPARSE handles;
  - workspace allocation;
  - pinned host staging where useful.
- Implement dense in-memory products with cuBLAS.
- Implement sparse in-memory products with cuSPARSE SpMM.
- Preserve 64-bit sparse `nnz` where cuSPARSE descriptors support it.
- Keep final small dense SVD on GPU or CPU based on measured simplicity and transfer cost.
- Add parity tests against CPU Halko using fixed seeds and tolerance bands appropriate for randomized/GPU math.

Exit criteria:

- Dense and sparse in-memory GPU Halko work on representative small and medium matrices.
- CPU fallback behavior is explicit and tested.
- GPU failures do not silently change the selected algorithm.

### Stage 5: Native CUDA Backed Dense Halko

Goal: make dense disk-backed inputs use GPU products without materializing the full matrix.

Tasks:

- Stream dense slabs/chunks through the chunk source.
- Transfer chunks to device in batches.
- Keep sketch matrices and accumulators device-resident when practical.
- Overlap read, transfer, and compute only after a simple synchronous version is correct.
- Record memory and IO metrics for each SVD run.

Exit criteria:

- Backed dense GPU Halko matches CPU Halko within randomized tolerance.
- Peak host and device memory stay bounded by the configured chunk policy.
- Benchmarks show when GPU backed dense SVD is beneficial or not.

### Stage 6: Native CUDA Backed Sparse Halko

Goal: make sparse disk-backed inputs use GPU sparse-times-dense products without materializing the full matrix.

Tasks:

- Stream sparse chunks into host CSR/CSC-like buffers.
- Transfer sparse chunks and dense sketch blocks to device.
- Use cuSPARSE SpMM for chunk products.
- Support 64-bit sparse index/nnz paths where available and tested.
- Define a transpose-product strategy:
  - stream transposed chunks if the backed format supports it efficiently;
  - or accumulate using row/column chunk passes with bounded memory;
  - document any extra IO passes introduced by power iterations.

Exit criteria:

- Backed sparse GPU Halko matches CPU Halko within randomized tolerance.
- Large-`nnz` sparse tests exercise 64-bit index plumbing without requiring impossible CI memory.
- Benchmarks include IO time, transfer time, compute time, and total wall time.

### Stage 7: Public Python Exposure

Goal: expose GPU execution without fragmenting user-facing SVD choices.

Recommended Python surface:

```python
run_svd(
    X,
    n_components=50,
    algorithm="auto",
    backend="auto",
    gpu_device=None,
    gpu_fallback=False,
    random_state=None,
    ...
)
```

Open decisions:

- Whether `backend="auto"` should ever select GPU automatically.
- Whether fallback should default to `False` for reproducibility and transparency.
- Whether GPU options should be flat kwargs or a small options object.

Recommended defaults for first release:

- `backend="auto"` preserves current CPU behavior.
- `backend="cuda"` opts into GPU.
- `gpu_fallback=False` by default.
- Metadata records both requested and resolved backend.

`reduce_kernel`:

- May accept the same backend controls.
- In v1, only its SVD subcall should be GPU-eligible.
- Metadata should make clear that non-SVD kernel reduction steps ran on CPU unless separately accelerated.

Exit criteria:

- Public Python docs show one SVD algorithm strategy with selectable backend.
- Unsupported combinations fail with actionable errors.
- Existing user code sees no behavior change unless it opts into GPU.

### Stage 8: RAFT/RAPIDS Evaluation Gate

Goal: decide whether RAFT should remain a reference, become an optional backend, or be dropped.

Run this only after the product interface and native CUDA baseline exist.

Evaluate:

- Build compatibility with supported compilers, CUDA versions, and Python packaging.
- Whether C++20/CUDA20 can be isolated from the default build.
- Performance versus native CUDA on:
  - dense in-memory;
  - sparse in-memory;
  - dense backed via an operator or staged products;
  - sparse backed via an operator or staged products.
- Correctness on 64-bit sparse `nnz` and large backed cases.
- Dependency burden from RAFT, RMM, and RAPIDS packaging.

Decision rule:

- Keep RAFT only if it clearly reduces maintenance or improves performance without making ordinary builds harder.
- Otherwise, treat it as useful documentation and keep native CUDA as the production backend.

### Stage 9: Benchmarks, CI, and Release Hardening

Goal: make GPU SVD maintainable after the first implementation lands.

Benchmarks:

- CPU IRLB sparse in-memory baseline.
- CPU Halko dense/backed baseline.
- GPU Halko dense in-memory.
- GPU Halko sparse in-memory.
- GPU Halko dense backed.
- GPU Halko sparse backed.

Metrics:

- total wall time;
- HDF5 read time;
- host-to-device transfer time;
- GPU compute time;
- peak host memory;
- peak device memory;
- number of matrix passes;
- approximation quality against CPU reference.

CI:

- CPU-only CI remains mandatory.
- GPU CI can be optional or nightly if hardware is scarce.
- Add compile-only CUDA job if runtime hardware is not always available.
- Add small runtime smoke tests where GPU runners exist.

Documentation:

- Installation matrix for CPU-only and GPU-enabled builds.
- CUDA version and compiler expectations.
- Known limitations for axis dimensions, sparse indices, backed formats, and fallback behavior.
- Migration note: PRIMME and Feng are retired from Python and are not GPU routes.

## Key Risks

### Backed Transpose Products

Halko power iterations need both `A * block` and `A.T * block`. Disk-backed sparse transpose products can be IO-heavy if the on-disk layout is not friendly.

Mitigation:

- Make product pass counts visible in benchmarks.
- Start with correctness-first streaming.
- Add layout-aware optimizations only after measuring.

### Dependency Creep

RAPIDS/RAFT could solve part of the math but impose compiler, CUDA, and packaging constraints.

Mitigation:

- Keep RAFT behind an experimental flag.
- Do not let RAFT affect default C++ standard or CPU builds.

### Silent Fallbacks

GPU errors that silently fall back to CPU would make performance and reproducibility confusing.

Mitigation:

- Default fallback to off.
- Record requested and resolved backend in metadata.
- Require explicit user opt-in for fallback.

### 64-bit Overclaiming

The project can support 64-bit sparse `nnz` without supporting all possible 64-bit row and column dimensions.

Mitigation:

- Document the difference between 64-bit nonzero counts and per-axis dimension limits.
- Add early guards.
- Keep tests focused on index plumbing, not impossible matrix materialization.

## Source Notes

External documentation consulted while preparing this plan:

- NVIDIA cuSPARSE documentation: `cusparseSpMM` supports sparse-times-dense multiplication, multiple sparse formats, FP32/FP64, and 32-bit or 64-bit sparse indices. See https://docs.nvidia.com/cuda/cusparse/index.html
- NVIDIA cuSOLVER documentation: `cusolverDnXgesvdr` is an approximate randomized SVD for dense device matrices and is most useful when the target rank is small relative to the matrix dimensions. See https://docs.nvidia.com/cuda/cusolver/index.html
- RAPIDS RAFT sparse solver documentation: RAFT provides sparse randomized SVD and an operator-style interface, but current APIs use int row/column dimensions in the documented interface. See https://docs.rapids.ai/api/raft/stable/cpp_api/sparse_solver/
- RAPIDS RAFT build documentation: downstream GPU builds require CUDA language support and C++20/CUDA20, with CUDA toolkit and compiler requirements that are heavier than the current default project baseline. See https://docs.rapids.ai/api/raft/stable/build/

## Recommended Next Step

Before writing CUDA kernels or bindings, do Stage 0 and Stage 2. That will make the current CPU state honest, remove ambiguity around retired algorithms, and produce the product abstraction that both in-memory and disk-backed GPU paths need.
