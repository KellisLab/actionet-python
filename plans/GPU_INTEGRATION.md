# GPU Integration Roadmap (Python Surface)

Status: planned, not implemented.

This document records the Python-facing GPU direction after the SVD strategy
cleanup. It intentionally avoids implementation details that are still unknown.
The launchpad for the next SVD implementation agents is
[`GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`](GPU_BACKED_SVD_AGENT_LAUNCHPAD.md).
The C++/build-side durable roadmap lives at
[`../src/libactionet/plans/GPU_BACKEND_PLAN.md`](../src/libactionet/plans/GPU_BACKEND_PLAN.md).

## Current State

- `pip install .` builds a CPU-only package.
- No production GPU SVD path exists.
- The public Python SVD algorithm surface is only `"auto"`, `"irlb"`, and
  `"halko"`.
- `"auto"` selects IRLB for sparse in-memory inputs and Halko for dense
  in-memory and all backed inputs.
- PRIMME and Feng are retired from the Python surface. Their C++ sources remain
  temporarily quarantined and should not be treated as GPU routes.

## Platform Scope

Python GPU API work must reflect the C++ platform contract:

- Linux x86_64 with NVIDIA GPUs is the runtime target.
- Windows 11 + WSL2 with NVIDIA GPUs is the developer test/sign-off target.
- CUDA 12.2 is the minimum toolkit target.
- Supported hardware is SM 8.0 / Ampere or newer.
- macOS stays CPU-only.
- Native Windows and R-facing GPU APIs are out of scope for v1.

The Python surface should not imply support for CUDA 11.x, pre-Ampere GPUs, or
GPU execution on macOS.

## Direction

GPU support should be exposed as an execution backend for existing algorithms,
not as a new SVD algorithm name.

The first GPU target is Halko-style randomized SVD across all four storage
forms:

- dense in-memory;
- sparse in-memory;
- dense disk-backed;
- sparse disk-backed.

Disk-backed support is first-class. Any public Python API should be delayed
until the implementation has a coherent story for backed and in-memory inputs,
even if those paths land internally in stages.

## Public API Contract To Preserve

The algorithm selector remains:

```python
algorithm = "auto" | "irlb" | "halko"
```

GPU work should add backend policy separately. Candidate policy concepts are:

- requested backend: automatic, CPU, or CUDA;
- CUDA device ordinal;
- explicit CPU fallback policy;
- resolved backend recorded in results/metadata.

The exact kwarg names and defaults are intentionally not settled here. They
should be finalized after the C++ product-backend abstraction exists and after
the first GPU-backed Halko path has been validated on real hardware.

## Metadata Contract

Today, CPU-only code records the SVD algorithm in `reduce_kernel` params and
the CPU backend vocabulary is available for future extension:

- `svd_algorithm`;
- `svd_algorithm_name`;
- `svd_backend_requested`;
- `svd_backend_resolved`.

Future GPU work should keep this shape and record concrete resolved backends
such as `"cpu"` or `"cuda"` rather than leaving `"auto"` in persisted metadata.

## Error And Fallback Expectations

GPU errors must be explicit and auditable.

- A forced GPU request should either run on GPU or raise a clear GPU error
  unless the caller explicitly allows CPU fallback.
- Fallback should not be silent in metadata: requested and resolved backends
  must differ when fallback happens.
- Runtime canaries must exercise real GPU computation, not only device
  discovery.
- Tests that touch GPU execution must be gated by a GPU capability predicate.

## reduce_kernel Scope

`reduce_kernel` may eventually accept the same backend policy as `run_svd`.
For the first GPU release, only the SVD subcall should be considered
GPU-eligible unless the non-SVD kernel-reduction steps are explicitly
accelerated later. Metadata must make that distinction clear.

## Non-goals

- Do not restore `"primme"` or `"feng"` as Python algorithm names.
- Do not route GPU work through PRIMME.
- Do not make GPU support mandatory for CPU-only builds.
- Do not make macOS a GPU target.
- Do not let a RAFT/RAPIDS experiment raise the default project C++ standard.

## Validation Themes

Before Python GPU API exposure, the implementation should demonstrate:

- unchanged CPU behavior with GPU disabled;
- clear errors for unsupported backend/algorithm combinations;
- parity with CPU Halko on dense/sparse and in-memory/backed inputs;
- bounded host and device memory for backed inputs;
- benchmarks that separate HDF5 read time, host-device transfer time, GPU
  compute time, and total wall time.
