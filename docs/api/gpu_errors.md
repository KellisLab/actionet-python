# GPU error taxonomy

When `libactionet` is built with `LIBACTIONET_ENABLE_NVIDIA_GPU=ON`, selected
routines (currently SVD) can dispatch to CUDA. Failures raise a specific
exception subclass so application code can react without pattern-matching on
`RuntimeError` messages.

The base class `GpuError` inherits from `RuntimeError`, so existing
`except RuntimeError` sites continue to work.

## Class hierarchy

```text
RuntimeError
└── GpuError                 # base for all GPU-specific failures
    ├── GpuUnavailableError  # requested GPU path but none is usable
    └── GpuRuntimeError      # GPU call itself failed at runtime
```

All three classes are re-exported from the top-level `actionet` namespace and
are defined in the compiled extension `actionet._core`.

## `actionet.GpuError`

Base class for every GPU-related failure raised by ACTIONet. Inherits from
`RuntimeError`. Catch this to react to any GPU-side problem regardless of
whether it was an availability issue or a runtime crash.

```python
import actionet as an

try:
    an.run_svd(X, k=50, backend="gpu")
except an.GpuError as exc:
    # Fall back to CPU or surface the error.
    ...
```

## `actionet.GpuUnavailableError`

Raised when a routine was asked to run on the GPU but no usable GPU path is
available. Common causes:

- The package was built with `LIBACTIONET_ENABLE_NVIDIA_GPU=OFF`.
- No NVIDIA driver / no visible CUDA device at runtime.
- CUDA runtime version mismatch.

This is distinct from `GpuRuntimeError` because the request never made it to a
kernel — it's safe to fall back to the CPU path without worrying about partial
state on the device.

## `actionet.GpuRuntimeError`

Raised when a GPU call was attempted and the device or a CUDA API reported a
failure (out-of-memory, invalid launch config, cuBLAS/cuSOLVER error, etc.).
Application code should not assume the device is in a clean state after this
exception; if the process needs to continue on GPU, a reset or process restart
is usually the safest response.

## See also

- [`run_svd`](reduction.md#actionet.reduction.run_svd) — currently the primary
  GPU-dispatched routine.
- [Architecture § GPU](../architecture.md#gpu) for how GPU dispatch is wired.
