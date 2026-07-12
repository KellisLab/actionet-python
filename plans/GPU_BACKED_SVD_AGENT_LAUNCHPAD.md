# GPU-backed SVD Agent Launchpad

Date: 2026-07-12

Status: active planning context for the next implementation pass. GPU SVD is
not implemented yet.

## Goal

Add NVIDIA GPU acceleration for SVD without fragmenting the SVD strategy again.
GPU support should be an execution backend for the existing Halko randomized SVD
path, not a new public SVD algorithm.

The production design must cover disk-backed data from the start. In-memory GPU
SVD is still useful, but it should not be planned separately in a way that
creates a second architecture.

## Current Public Contract

Python exposes only:

- `algorithm="auto"`
- `algorithm="irlb"`
- `algorithm="halko"`

Current auto-selection:

- sparse in-memory: IRLB
- dense in-memory: Halko
- backed dense/sparse: Halko

Do not restore `"primme"` or `"feng"` to the Python API.

## Current Code State

Important files:

- `src/actionet/decomposition/svd.py`
  - Public SVD wrapper and algorithm selection.
  - Adds CPU backend metadata when `return_operator_compatible=False`.

- `src/actionet/decomposition/kernel.py`
  - `reduce_kernel` SVD selection.
  - Persists `svd_algorithm`, `svd_algorithm_name`,
    `svd_backend_requested`, and `svd_backend_resolved`.

- `src/actionet/bindings/wp_decomposition.cpp`
  - `_core.run_svd_sparse` and `_core.run_svd_dense`.
  - Validates that Python callers can pass only `ALG_IRLB` or `ALG_HALKO`.

- `src/actionet/bindings/wp_action.cpp`
  - `_core.reduce_kernel_sparse` and `_core.reduce_kernel_dense`.
  - Validates that Python callers can pass only `ALG_IRLB` or `ALG_HALKO`.

- `src/actionet/bindings/wp_io.cpp`
  - Backed `_core.run_svd_backed_operator` and
    `_core.reduce_kernel_backed_operator`.
  - Validates that Python callers can pass only `ALG_IRLB` or `ALG_HALKO`.

- `src/libactionet/src/decomposition/svd_halko.cpp`
  - CPU Halko implementation.
  - Now explicitly rejects per-axis dimensions above `INT_MAX` before any
    `int` narrowing.

- `src/libactionet/include/decomposition/matrix_operator.hpp`
  - CPU host matrix-operator abstraction.
  - Do not treat this as the future GPU device-memory boundary.

- `src/libactionet/plans/GPU_BACKEND_PLAN.md`
  - Durable C++/build-side GPU roadmap.

- `plans/GPU_INTEGRATION.md`
  - Python-facing GPU roadmap.

## Phase 0 Cleanup Completed

The preflight cleanup has been implemented in this branch:

- Stale GPU plan links were updated from `context/` paths to `plans/` paths.
- Python `_core` SVD entry points now reject raw retired algorithm IDs.
- C++ SVD comments no longer describe PRIMME/Feng as normal Python-facing
  options.
- MatrixOperator comments now clarify that future GPU SVD needs a separate
  device-aware product backend.
- Halko now has explicit per-axis `INT_MAX` guards.
- CPU metadata vocabulary for requested/resolved SVD backend exists.
- Regression tests were added for private `_core` rejection of Feng/PRIMME IDs.

## Settled Decisions

### PRIMME

Do not use PRIMME for GPU SVD. The previous PRIMME GPU attempt failed for
structural reasons and does not fit the current architecture.

PRIMME C++ code remains temporarily quarantined for deletion later. It should
not influence the new GPU design.

### Feng

Do not use Feng as a public Python SVD option. It is retired from Python and
does not provide a distinct enough capability to justify another path.

### 64-bit CPU Support

The practical 64-bit CPU target is large sparse `nnz`, not arbitrary 64-bit
row/column dimensions.

Current policy:

- 64-bit sparse index plumbing is supported for realistic omics matrices.
- Per-axis row and column dimensions above `INT_MAX` are rejected in SVD paths.
- Do not silently narrow dimensions.

### GPU Backend Direction

Native CUDA is the production default direction:

- cuBLAS for dense products;
- cuSPARSE for sparse-times-dense products;
- cuSOLVER where useful for dense QR/SVD pieces;
- CUDA runtime for streams, device selection, pinned buffers, and memory
  accounting.

RAFT/RAPIDS may be evaluated only as an optional spike after the product
backend boundary exists. It must not raise the default C++ standard or become a
required package dependency without an explicit decision.

## Architecture Target

Halko should become a shared randomized SVD driver over product backends:

```text
Python API
  -> SVD dispatcher
    -> Halko randomized SVD driver
      -> product backend
        -> CPU dense in-memory
        -> CPU sparse in-memory
        -> CPU backed dense/sparse
        -> CUDA dense in-memory
        -> CUDA sparse in-memory
        -> CUDA backed dense/sparse streaming
```

The key boundary is product semantics, not storage type:

- `Y = A * X`
- `Y = A.T * X`
- bounded chunk iteration for backed inputs
- explicit host/device ownership for CUDA inputs

Do not build GPU SVD by wrapping CPU `MatrixOperator::matmat` calls and copying
their results to the GPU. That repeats the architecture problem exposed by the
failed PRIMME GPU attempt.

## Recommended Next Work

### Stage 1: Refactor CPU Halko Around Product Backends

Goal: no CUDA yet. Make the current CPU Halko implementation use a product
interface that later CUDA code can implement.

Tasks:

- Extract `A * block` and `A.T * block` from `svd_halko.cpp`.
- Preserve current numerical behavior and default selection.
- Add CPU product backends for dense, sparse, and current operator inputs.
- Keep the randomized SVD driver responsible for sketch generation,
  orthogonalization, power iterations, and final small SVD.

Exit criteria:

- Existing CPU SVD tests pass.
- Backed dense/sparse Halko parity remains unchanged.
- No GPU build dependency is introduced.

### Stage 2: Extract Backed Chunk Sources

Goal: give backed CPU and future backed GPU paths the same streaming substrate.

Tasks:

- Introduce a chunk/slab source abstraction outside `MatrixOperator`.
- Preserve lazy transforms, row scaling, log transform behavior, and backed I/O
  chunk policy.
- Support product passes for both `A * block` and `A.T * block`.
- Keep v1 data movement simple: HDF5 chunk -> host buffer -> optional pinned
  host buffer -> device buffer.

Exit criteria:

- CPU backed Halko can run through the new chunk source.
- Chunked products match current `MatrixOperator` products on test matrices.

### Stage 3: Add Optional CUDA Execution Skeleton

Goal: introduce optional build/runtime policy without changing CPU behavior.

Tasks:

- Add an execution policy type in C++.
- Add optional CUDA CMake wiring, disabled by default.
- Add runtime device/capability checks and a real computation canary.
- Add clear GPU unavailable/runtime error types.
- Keep GPU-disabled builds free of CUDA headers.

Exit criteria:

- CPU-only builds remain unchanged.
- GPU-enabled builds can compile a small CUDA canary target.
- Unsupported backend/algorithm combinations fail clearly.

### Stage 4: Native CUDA In-memory Halko

Goal: implement CUDA product backends for dense and sparse in-memory inputs.

Tasks:

- Use cuBLAS for dense products.
- Use cuSPARSE SpMM for sparse-times-dense products.
- Preserve 64-bit sparse index/`nnz` support where CUDA descriptors support it.
- Decide by measurement whether the final small SVD stays on GPU or transfers
  to CPU.

Exit criteria:

- Dense and sparse in-memory GPU Halko match CPU Halko within randomized
  tolerance.
- Requested/resolved backend metadata is correct.
- CPU fallback is explicit and tested.

### Stage 5: Native CUDA Backed Halko

Goal: implement GPU-backed dense and sparse streaming products.

Tasks:

- Stream backed dense slabs/chunks to device.
- Stream backed sparse chunks to device sparse descriptors.
- Keep sketch/work buffers device-resident where practical.
- Measure HDF5 read time, transfer time, GPU compute time, total wall time,
  host memory, and device memory.

Exit criteria:

- Backed dense and sparse GPU Halko match CPU Halko within randomized
  tolerance.
- Peak host/device memory is bounded by policy.
- Benchmarks show when backed GPU SVD is actually beneficial.

## Python API Guidance

Do not expose public GPU kwargs until the backend can give honest behavior for
the intended v1 matrix forms.

The eventual public surface should preserve:

```python
algorithm = "auto" | "irlb" | "halko"
```

Backend policy should be separate, for example:

```python
backend = "auto" | "cpu" | "cuda"
```

The exact spelling and defaults are still open. Recommended defaults for the
first public GPU release:

- CPU behavior remains the default unless GPU is explicitly requested.
- Forced CUDA either runs on CUDA or raises unless fallback is explicitly
  enabled.
- Metadata records requested and resolved backend.

`reduce_kernel` may accept the same backend policy once `run_svd` does, but in
v1 only the SVD subcall should be GPU-eligible unless the rest of kernel
reduction is explicitly accelerated.

## Validation Checklist

Before exposing GPU SVD publicly:

- CPU-only tests pass with GPU disabled.
- macOS remains CPU-only.
- Direct `_core` calls cannot reach PRIMME/Feng from Python.
- CPU Halko/IRLB retain current defaults and parity.
- GPU tests are gated by build + runtime + computation canary.
- Dense/sparse in-memory GPU Halko passes parity tests.
- Dense/sparse backed GPU Halko passes parity tests.
- Benchmarks report I/O, transfer, compute, total time, host memory, and device
  memory.
- Documentation states CUDA/compiler requirements and known limits.

## Reference Documentation

- Python GPU roadmap: `plans/GPU_INTEGRATION.md`
- C++ GPU roadmap: `src/libactionet/plans/GPU_BACKEND_PLAN.md`
- SVD decisions: `context/DECISIONS.md`
- libactionet decisions: `src/libactionet/context/DECISIONS.md`
- NVIDIA cuSPARSE: https://docs.nvidia.com/cuda/cusparse/index.html
- NVIDIA cuSOLVER: https://docs.nvidia.com/cuda/cusolver/index.html
- RAPIDS RAFT sparse solvers: https://docs.rapids.ai/api/raft/stable/cpp_api/sparse_solver/
- RAPIDS RAFT build requirements: https://docs.rapids.ai/api/raft/stable/build/
