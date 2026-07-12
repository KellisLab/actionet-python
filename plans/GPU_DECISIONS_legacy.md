
## GPU backend (Python surface)

> Historical note. This file records decisions from the scrapped PRIMME-based
> GPU attempt and is not an implementation launchpad. Current GPU-backed SVD
> work should follow `GPU_INTEGRATION.md` and
> `GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`.

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
