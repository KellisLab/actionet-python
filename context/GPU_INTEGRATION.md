# GPU Integration (Python surface roadmap)

## Status

**Not implemented on `dev`.** A prior attempt on the `dev-gpu-v2`
branch (July 2026) added the Python-side plumbing for a GPU backend
(`compute_backend` / `device_id` / `allow_cpu_fallback` kwargs on
`run_svd` and `reduce_kernel`, `GpuError` taxonomy re-export, backend
policy tests, benchmark harness, GH Actions CI matrix) on top of a
`libactionet` cuBLAS-backed PRIMME dispatch. When exercised on real
hardware for the first time (WSL2 + RTX 4000 Ada + CUDA 13.2), the
C++ side segfaulted immediately on every `compute_backend="gpu"`
call. The failure was structural (see
[`../src/libactionet/context/GPU_BACKEND_PLAN.md`](../src/libactionet/context/GPU_BACKEND_PLAN.md)
section 6, "Post-mortem"), not a small fix. The entire branch was
scrapped so a clean re-attempt can happen on hardware.

This document is the durable Python-surface roadmap. It records the
settled Python-facing decisions that survive the reset and constrain
the re-attempt; it does not describe any code that currently exists in
this repo.

## Companion document

The C++ / build side (`ExecutionPolicy`, `ComputeBackend` enum, CMake
`LIBACTIONET_ENABLE_NVIDIA_GPU` option, cuBLAS PRIMME dispatch,
platform contract, CUDA / compute-capability floors, WSL2 quirks,
Phase 1 / 2 / 3 phase specifics, post-mortem of the scrapped attempt)
lives in
[`../src/libactionet/context/GPU_BACKEND_PLAN.md`](../src/libactionet/context/GPU_BACKEND_PLAN.md).
This document only pins the Python surface and the wrapper-level
decisions. Do not duplicate the C++/build content here.

## Target Python surface

When the re-attempt lands, GPU-eligible entry points (initially
`run_svd` and `reduce_kernel`) accept three kwargs:

| kwarg | type | default | semantics |
|---|---|---|---|
| `compute_backend` | `Literal["auto", "cpu", "gpu"]` | `"auto"` | Backend selector. `"auto"` = GPU if build + runtime probes pass, CPU otherwise. `"cpu"` = force CPU. `"gpu"` = force GPU (see `allow_cpu_fallback`). |
| `device_id` | `int >= 0` | `0` | CUDA device ordinal. |
| `allow_cpu_fallback` | `bool` | `True` | With `compute_backend="gpu"`, silently fall back to CPU if the GPU is unavailable **or if the GPU path itself fails at runtime**. `False` -> raise. |

The `allow_cpu_fallback` "or the GPU path itself fails" clause is
new. The scrapped implementation only checked build/runtime probes and
therefore entered a broken dispatch on GPU-visible machines even with
`allow_cpu_fallback=True`. The re-attempt must include a runtime
canary (a small dummy solve at startup) or a signal-safe wrapper so
the fallback semantics hold end to end.

## Target FFI conventions

- The wrapper module `_core` re-exports `BACKEND_CPU` / `BACKEND_GPU` / `BACKEND_AUTO` **integer constants** from the pybind11 module rather than the Python side redefining a parallel mapping. This prevents silent desynchronization with the C++ enum.
- The wrapper module re-exports `GpuError`, `GpuUnavailableError`, `GpuRuntimeError` from the C++ taxonomy at
  `actionet.GpuError`, `actionet.GpuUnavailableError`, `actionet.GpuRuntimeError`. They inherit from `RuntimeError` for backward compatibility with callers that catch the base class.
- The wrappers pass the three kwargs into a C++ `actionet::ExecutionPolicy` value; they do not synthesize policies on the Python side.

## Target metadata symmetry

Both `reduce_kernel` and `run_svd` record the resolved policy on their
result so that post-hoc auditing can tell which backend actually ran:

- `reduce_kernel`: writes into `adata.uns[f"{key_added}_params"]`.
- `run_svd`: includes the resolved policy in the returned dict when `return_operator_compatible=False`.

The recorded fields are the same three kwargs after resolution:
concrete backend (`"cpu"` or `"gpu"`, never `"auto"`), `device_id`,
`allow_cpu_fallback`. This is what makes benchmarks and reproducibility
reports meaningful.

## Target platform matrix

Matches the libactionet contract; consult
[`../src/libactionet/context/GPU_BACKEND_PLAN.md`](../src/libactionet/context/GPU_BACKEND_PLAN.md)
section 1 for the authoritative version.

| Platform | Python install | GPU eligible? |
|---|---|---|
| Linux x86_64 with NVIDIA GPU (Ampere+) | `LIBACTIONET_ENABLE_NVIDIA_GPU=ON pip install .` | Yes; production target. |
| Linux x86_64 without GPU | `pip install .` | No; identical to CPU baseline. `compute_backend="gpu"` raises unless `allow_cpu_fallback=True`. |
| Windows 11 + WSL2 with NVIDIA GPU | `LIBACTIONET_ENABLE_NVIDIA_GPU=ON pip install .` | Yes; primary dev/test environment. |
| macOS (any Apple Silicon or Intel) | `pip install .` | No, ever; the CMake option is silently ignored. Existing macOS CI must keep passing as the regression guard. |
| Windows native (no WSL2) | not supported | n/a |

## Target CI

Three required checks and one manual sign-off:

| Job | Trigger | What it builds |
|---|---|---|
| macOS no-GPU | every push to any dev-branch containing GPU work | default options; full test suite. Required. |
| Linux CPU-only | every push to any dev-branch containing GPU work | `LIBACTIONET_ENABLE_NVIDIA_GPU=OFF`; full test suite. Required. |
| Linux GPU smoke-build | every push to any dev-branch containing GPU work | `LIBACTIONET_ENABLE_NVIDIA_GPU=ON` against CUDA 12.2 stub libraries; link-only, no test execution. Catches include-path / link / API-surface regressions against the 12.2 floor. Required. |
| WSL2 manual | before each push to any dev-branch containing GPU work | full GPU build + GPU-marked tests + benchmark harness. |

## Test-file conventions (learned from the scrapped attempt)

- Every GPU-touching test **must** be gated by `@requires_gpu` (or the pytest `-m gpu` marker resolving to the same predicate). "Fallback" tests are not exempt: they must skip on non-GPU boxes _and_ skip on GPU boxes when the target algorithm is known-unusable. The scrapped attempt had three ungated `_fallback` tests that crashed on any real GPU box, not just under `-m gpu`.
- The GPU predicate must combine (a) build has GPU, (b) runtime probes pass, and (c) a small dummy solve completes without segfault. Probes-alone is not sufficient (learned the hard way).
- Numerical-parity tests (singular values, subspace agreement, reconstruction error) belong in a dedicated file (was `tests/test_gpu_backend_policy.py`) that only runs under `-m gpu`.

## Target Python-side error handling

```python
import actionet as an

try:
    an.reduce_kernel(adata, n_components=50, compute_backend="gpu", allow_cpu_fallback=False)
except an.GpuUnavailableError as e:
    ...  # rerun with compute_backend="cpu" will succeed
except an.GpuRuntimeError as e:
    ...  # something ran on the GPU and failed (OOM, cuBLAS error, kernel error, etc.)
except an.GpuError as e:
    ...  # base class catch-all
```

## Decisions log (Python surface)

Settled. Re-opening any of them requires updating
[`DECISIONS.md`](DECISIONS.md) with rationale.

- Three kwargs on every GPU-eligible entry point: `compute_backend`, `device_id`, `allow_cpu_fallback`. Exact spelling. No env-var override.
- `compute_backend` default is `"auto"`; `device_id` default is `0`; `allow_cpu_fallback` default is `True`.
- Backend integer constants live on `_core`, not in Python.
- Error classes live on the top-level `actionet` namespace and inherit from `RuntimeError`.
- Resolved policy is recorded on every GPU-eligible result (both `reduce_kernel` and `run_svd`) with concrete backend, never `"auto"`.
- `allow_cpu_fallback=True` must include "GPU path crashed at runtime" as a fallback trigger, not only "probes failed".
- All GPU tests are gated by `@requires_gpu`; the predicate includes a runtime canary.
- No env-var override for backend selection (per-call kwargs only) so notebook / test isolation is preserved.
