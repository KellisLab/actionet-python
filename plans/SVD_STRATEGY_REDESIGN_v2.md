# SVD Strategy Redesign

Status: historical draft, 2026-07-09. Partially executed and partially
superseded by `GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`. Use the launchpad for new
GPU-backed SVD implementation work.

## Summary

This plan redesigns SVD around one scalable algorithm family rather than
separate paths that grew independently for dense, sparse, backed, and GPU
inputs. The target direction is:

- Halko-style randomized SVD is the default scalable path across CPU/GPU and
  in-memory/backed inputs.
- GPU support for disk-backed data is a first-class v1 requirement, not a
  follow-up after in-memory GPU support.
- PRIMME is removed from public strategy and eventually deleted.
- 64-bit CPU sparse support is retained without PRIMME for realistic omics
  matrices, especially matrices with more than 2^31 non-zero entries.
- The GPU implementation should prefer CUDA toolkit primitives over mandatory
  RAPIDS/RAFT dependencies if that preserves compatibility without excessive
  complexity.

The main architectural move is to separate the randomized SVD algorithm from
the matrix-product implementation. Dense, sparse, backed-dense, backed-sparse,
CPU, and GPU inputs should differ in product backends, not in the top-level SVD
algorithm.

## Current Problems

### Too many strategy paths

The package currently exposes four SVD algorithms:

- IRLB: default for sparse in-memory matrices.
- Halko: default for dense in-memory and backed matrices.
- Feng: explicit alternative randomized method.
- PRIMME: legacy large-sparse and operator-backed escape hatch.

This split has made the implementation harder to reason about. The backed CPU
path already exposed the failure mode: users can request one algorithm and get
another because backed `algorithm="irlb"` silently dispatches through PRIMME in
the C++ operator path.

### PRIMME no longer fits the architecture

PRIMME currently provides three forms of value:

- It is auto-selected for large in-memory sparse matrices based on a stale
  32-bit sparse-indexing assumption.
- It provides the current backed-IRLB block-solver fast path.
- It can provide high-accuracy sparse SVD for unusual precision-sensitive
  cases.

The costs now outweigh those benefits:

- Python builds force `ARMA_64BIT_WORD`, so sparse `nnz > INT32_MAX` is not by
  itself a reason to require PRIMME.
- The backed IRLB to PRIMME dispatch is surprising and expands maintenance
  surface.
- The previous GPU attempt showed that PRIMME's cuBLAS path is a poor fit for
  this package's host-oriented wrappers and callback architecture.
- PRIMME contributes build complexity and previously observed ODR/LTO warning
  risk.

### GPU disk-backed SVD must be designed from the start

In-memory GPU SVD is useful, but the highest-value target is disk-backed data.
Anything small enough to fit comfortably in memory may not need GPU
acceleration. Planning only in-memory GPU first risks recreating the current CPU
problem: separate implementations for each input type with inconsistent
semantics.

## Target Architecture

Introduce one internal product abstraction that the randomized SVD driver uses
for all storage/backend combinations.

Conceptual interface:

```cpp
struct SvdProductBackend {
    arma::uword rows() const;
    arma::uword cols() const;

    // CPU backends use host dense blocks. GPU backends use device blocks.
    void apply(const Block& X, Block& Y) const;           // Y = A * X
    void apply_transpose(const Block& X, Block& Y) const; // Y = A' * X

    MemoryLocation memory_location() const; // host or device
    StorageKind storage_kind() const;       // dense, sparse, backed_dense, backed_sparse
};
```

The concrete interface can differ, but it should preserve these ideas:

- The SVD driver owns the algorithmic sequence: random sketch, power
  iterations, orthogonalization, small dense SVD, orientation.
- Product backends own only `A * X` and `A' * X`.
- Backed GPU support is implemented by chunk streams feeding product kernels,
  not by copying CPU `MatrixOperator::matmat` results to the GPU.
- Public API exposes backend selection, not a matrix of algorithm variants.

## Objective 1: Remove PRIMME Without Losing 64-bit CPU Support

Plan:

- Stop auto-selecting PRIMME for `nnz > INT32_MAX`.
- Remove the backed `IRLB -> PRIMME` fast path.
- Keep PRIMME temporarily as hidden/quarantined legacy code only if a
  compatibility window is needed.
- Delete PRIMME after one stabilization window once CPU and GPU replacements
  have parity coverage.

64-bit CPU sparse support without PRIMME:

- Treat `nnz > INT32_MAX` as supported for Python builds when rows and columns
  fit the currently supported BLAS/LAPACK dimensions.
- Add explicit guards for `rows > INT_MAX` or `cols > INT_MAX` in IRLB, Halko,
  and Feng until every internal narrowing is audited or widened.
- Audit casts from `arma::uword` to `int` in the SVD implementations.
- Keep BLAS/LAPACK call boundaries conservative because common CBLAS APIs still
  take `int` dimensions.

Practical contract:

- Support realistic 64-bit sparse omics matrices where `nnz` can exceed
  2^31 - 1, but row and column counts remain within `INT_MAX`.
- Fail early with a clear error for matrices whose dimensions exceed current
  BLAS-backed implementation limits.

## Objective 2: One Randomized SVD Family Across CPU/GPU

Use Halko-style randomized SVD as the shared scalable algorithm.

Public algorithm surface:

- `algorithm="auto"`: recommended default.
- `algorithm="halko"`: explicit scalable randomized SVD.
- `algorithm="irlb"`: CPU legacy/precision option where supported.
- `algorithm="feng"`: explicit secondary randomized method.

GPU v1 should support only the Halko/randomized path. It should not expose a
separate public GPU algorithm knob until there are two mature GPU algorithms
with meaningful user-facing tradeoffs.

Internal dispatch:

- CPU + in-memory sparse: keep IRLB or move to Halko depending on benchmark and
  numerical results.
- CPU + dense in-memory: Halko default.
- CPU + backed: Halko default.
- GPU + dense/sparse/backed: Halko/randomized only.
- PRIMME: hidden or removed.

This reduces public choice while still allowing specialized CPU alternatives.

## Objective 3: GPU Support for Disk-backed Data

Do not implement disk-backed GPU support as a wrapper around existing CPU
`MatrixOperator::matmat` or `rmatmat`. That would preserve the wrong boundary:
CPU code would still perform the matrix product, and GPU work would only happen
after expensive host/device transfers.

Instead add GPU-aware backed readers and chunk streams.

### Backed sparse path

For HDF5-backed CSR/CSC:

- Stream contiguous sparse chunks from HDF5 into host buffers.
- Prefer pinned host buffers when beneficial on the target platform.
- Preserve current transform semantics: row scaling, log1p approximation, and
  log scaling.
- For v1, apply transforms on host for semantic safety and implementation
  simplicity.
- Upload chunk-local sparse buffers to device.
- Use cuSPARSE SpMM for chunk products.
- Accumulate sketch outputs on device where possible.
- Support 64-bit sparse indices where cuSPARSE allows it.

Options:

- Option A: Synchronous chunk loop. Read chunk, upload chunk, multiply,
  accumulate. This is simpler and should be the first correctness target.
- Option B: Double-buffered chunk loop. While the GPU processes chunk `i`, the
  CPU/HDF5 layer reads chunk `i + 1`. This is the expected performance target
  after synchronous correctness lands.
- Option C: Future GPU transform kernels. Move row scaling/log transforms to
  CUDA kernels once correctness and transfer costs are understood.

Recommendation: implement Option A first but design the `GpuChunkStream`
interface so Option B does not change the SVD driver.

### Backed dense path

For HDF5-backed dense arrays:

- Stream row slabs from HDF5 into host buffers.
- Apply current lazy transforms on host for v1.
- Upload slabs to device.
- Use cuBLAS GEMM for slab products.
- Accumulate sketch outputs on device.

Options:

- Option A: Host-side transforms and synchronous slab upload.
- Option B: Host-side transforms with double-buffered upload/compute.
- Option C: Device-side transform kernels for row scaling/log1p once the
  synchronous path is correct.

Recommendation: same as sparse: synchronous first, double-buffering next.

### Common backed GPU interface

Add a lower-level backed chunk interface distinct from `MatrixOperator`:

```cpp
struct BackedChunkView {
    arma::uword row_start;
    arma::uword row_count;
    StorageKind storage_kind;
    HostBufferView host_data;
    HostBufferView host_indices;
    HostBufferView host_indptr;
};

class BackedChunkStream {
public:
    arma::uword rows() const;
    arma::uword cols() const;
    bool next(BackedChunkView& chunk);
    void reset();
};
```

The exact shapes may change, but the important boundary is:

- HDF5 and lazy transforms feed chunks.
- CUDA product backends consume chunks.
- The randomized SVD driver never calls CPU `matmat` when running on GPU.

## Objective 4: GPU Implementation Library Options

### Option A: Native CUDA primary path

Use CUDA toolkit libraries directly:

- cuSPARSE SpMM for sparse matrix/block products.
- cuBLAS GEMM for dense products and small dense operations.
- cuSOLVER for QR and small dense SVD steps.
- cuRAND or a deterministic host-generated/random-upload path for sketch
  initialization.

Pros:

- Avoids making RAPIDS/RAFT a hard dependency.
- Can likely preserve the current C++17 baseline.
- Better compatibility with HPC conda environments that have CUDA but not full
  RAPIDS stacks.
- Lets disk-backed streaming be designed around this package's actual HDF5
  layout.

Cons:

- More implementation code.
- More numerical/detail ownership.
- Need careful testing around QR/SVD choices and determinism.

### Option B: RAFT primary path

Use RAPIDS RAFT `sparse_randomized_svd` where possible.

Pros:

- Existing sparse randomized SVD implementation.
- Algorithmically aligned with the desired Halko direction.
- Useful operator and CSR APIs.

Cons:

- RAFT downstream GPU use currently expects C++20/CUDA20 integration.
- Adds RAPIDS dependency and environment complexity.
- The RAFT operator interface is device-oriented; a host-backed adapter would
  not solve disk-backed GPU performance unless paired with a real GPU chunk
  pipeline.

### Option C: Hybrid path

Build the package-native product abstraction and streaming CUDA path first.
Use RAFT as an optional spike, benchmark/reference, or future backend.

Pros:

- Keeps core architecture independent.
- Allows direct comparison against RAFT.
- Avoids premature dependency commitment.

Cons:

- Requires maintaining an optional experimental integration during evaluation.

Recommendation: Option C, with native CUDA as the primary compatibility path
and RAFT as an optional reference/benchmark spike.

## Objective 5: Public API and Policy

Historical proposal. The exact Python-facing backend kwarg names and defaults
are no longer settled by this document. Use `GPU_INTEGRATION.md` and
`GPU_BACKED_SVD_AGENT_LAUNCHPAD.md` for the current policy.

The earlier proposed controls were:

- `compute_backend: "auto" | "cpu" | "gpu"`
- `device_id: int`
- `allow_cpu_fallback: bool`

Rules:

- Existing calls continue to work.
- `compute_backend="auto"` selects GPU only when the build, runtime, device,
  and runtime canary pass.
- `allow_cpu_fallback=True` includes GPU-path runtime failures, not just
  unavailable devices.
- Returned metadata records the resolved backend.
- No environment-variable override for backend choice in v1.

Additional user-facing knobs may be needed for backed GPU memory control:

- `gpu_workspace_bytes`
- `gpu_streaming_chunk_mb` or reuse/adapt existing backed chunk controls
- optional `gpu_pipeline_depth` once double-buffering exists

These should be added only when the implementation needs them and should have
CPU-safe defaults.

## Objective 6: Testing and Benchmarks

Correctness tests:

- CPU Halko parity across dense, sparse, backed-dense, backed-sparse.
- GPU Halko parity across dense, sparse, backed-dense, backed-sparse.
- Singular value agreement.
- Subspace agreement.
- Reconstruction error.
- Deterministic orientation/sign conventions.
- Lazy transform parity against existing CPU backed operators.
- 64-bit sparse `nnz` path coverage where feasible.
- Clear failure for unsupported `rows/cols > INT_MAX` cases.

Performance benchmarks:

- In-memory dense and sparse.
- Backed dense.
- Backed sparse.
- Compressed vs uncompressed backed storage where relevant.
- Wall time.
- Peak host RSS.
- Peak GPU memory.
- HDF5 pass count.
- Host-to-device and device-to-host transfer volume.
- Number of chunk uploads.
- CPU baseline comparison.

Acceptance gates:

- Correctness is mandatory before exposing GPU in public API.
- Backed GPU speedup is the primary performance gate.
- In-memory GPU speedup is secondary.
- CPU-only builds must remain unchanged by default.
- macOS CPU-only builds must remain green.
- GPU smoke-build must target the CUDA 12.2 floor.
- Manual WSL2 hardware validation is required before merging GPU work.

## Staged Implementation Sketch

### Stage 0: Strategy cleanup

- Remove PRIMME from Python auto-selection.
- Remove backed IRLB to PRIMME hidden dispatch.
- Add explicit dimension guards to non-PRIMME CPU SVD paths.
- Update docs and decisions to state the new public SVD strategy.

### Stage 1: Product-backend abstraction

- Introduce the internal SVD product backend interface.
- Refactor CPU Halko to use the product interface.
- Keep existing public behavior unchanged.
- Add parity tests proving old and refactored CPU Halko agree.

### Stage 2: Backed chunk stream

- Extract backed dense and sparse chunk iteration into reusable stream classes.
- Preserve existing CPU `MatrixOperator` behavior.
- Add tests that chunk streams reproduce existing `matmat` and `rmatmat`
  results.

### Stage 3: Native CUDA in-memory products

- Add optional CUDA build plumbing behind `LIBACTIONET_ENABLE_NVIDIA_GPU=ON`.
- Implement dense and sparse in-memory GPU product backends.
- Implement GPU Halko over the product interface.
- Add GPU canary and GPU-gated parity tests.

### Stage 4: Native CUDA backed products

- Implement backed dense and sparse GPU product backends using chunk streams.
- Start with synchronous read/upload/compute.
- Add backed GPU parity tests.
- Add backed GPU benchmark harness.

### Stage 5: Performance pipeline

- Add pinned-buffer strategy where it helps.
- Add double-buffered backed GPU pipeline.
- Move transforms to CUDA kernels only if profiling shows host transforms are a
  bottleneck.

### Stage 6: PRIMME deletion

- Delete PRIMME public mapping, C++ wrappers, vendored sources, and CMake
  wiring once the replacement paths are stable.
- Remove PRIMME-specific comments and stale 32-bit rationale.

## Open Questions

- Should CPU sparse in-memory `auto` remain IRLB or move to Halko for strategy
  consistency?
- Should host-side log1p approximation remain the required v1 behavior for GPU
  backed paths, or should CUDA kernels implement the same approximation from the
  start?
- Should RAFT be a short-lived spike only, or an optional backend kept behind a
  separate build flag?
- What minimum backed-GPU speedup should be required before public release?
- Should GPU backed SVD initially support both CSR and CSC, or normalize one
  format first and add the other immediately after?

## References

- RAPIDS RAFT sparse randomized SVD:
  https://docs.rapids.ai/api/raft/stable/cpp_api/sparse_solver/
- RAPIDS RAFT downstream build guidance:
  https://docs.rapids.ai/api/raft/stable/build/
- NVIDIA cuSPARSE SpMM:
  https://docs.nvidia.com/cuda/cusparse/index.html
- NVIDIA cuSOLVER randomized dense SVD:
  https://docs.nvidia.com/cuda/cusolver/index.html
