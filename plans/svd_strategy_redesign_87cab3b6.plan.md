---
name: SVD strategy redesign
overview: Consolidate the SVD stack from four methods to three by hiding PRIMME from the public API without immediately deleting it, and stand up a GPU-backed Halko path by prototyping RAFT `sparse_randomized_svd` and a custom cuBLAS/cuSPARSE Halko in parallel before committing.
todos:
  - id: phase1_hide_primme
    content: "Phase 1: Remove PRIMME from public Python API and auto-selection, cut backed-IRLB PRIMME fast path. Keep C++ symbols compiled."
    status: pending
  - id: phase2_backed_irlb_bench
    content: "Phase 2: Benchmark honest backed svdIRLB(MatrixOperator&) vs Halko to decide whether backed IRLB is deprecated or kept."
    status: pending
  - id: phase3a_raft_prototype
    content: "Phase 3 Track A: RAFT prototype -- ConfigureRAFT.cmake, svd_halko_gpu_raft.cpp, MatrixOperator adapter."
    status: pending
  - id: phase3b_custom_prototype
    content: "Phase 3 Track B: Custom cuBLAS/cuSPARSE Halko prototype -- svd_halko_gpu_native.cu with cuRAND, cuBLAS QR, cuSOLVER Xgesvdj."
    status: pending
  - id: phase3c_bakeoff
    content: "Phase 3 bake-off: numerical parity, wallclock, memory, canary robustness on WSL2 + RTX 4000 Ada. Pick a winner."
    status: pending
  - id: phase4_expose_gpu_surface
    content: "Phase 4: Expose compute_backend/device_id/allow_cpu_fallback on run_svd and reduce_kernel per plans/GPU_INTEGRATION.md; add @requires_gpu, CI jobs."
    status: pending
  - id: phase5_delete_primme
    content: "Phase 5 (post-stabilization): delete svd_primme.*, vendored PRIMME tree, ConfigurePRIMME.cmake, R build guards; update DECISIONS.md."
    status: pending
isProject: false
---

## Answers to the three specific questions

**(1) Is PRIMME worth keeping?**

Not as a first-class option. Analysis:

- PRIMME's current public value is (a) the only auto-selected path for in-memory sparse matrices with `nnz > 2^31`, (b) the transparent block-matvec fast path for backed IRLB (`svd_main.cpp:87-95`), and (c) higher numerical precision than the randomized methods.
- (a) is genuinely rare — an in-memory sparse matrix with more than ~2 billion nonzeros implies ~16 GB of value storage alone, well past the point at which we route through backed mode. It's an unrealistic auto-trigger.
- (b) turns out to be a design leak, not a feature: backed users who ask for `"irlb"` silently run a completely different algorithm (PRIMME's block Lanczos SVD). This adds surprise and doubles the maintenance surface.
- (c) is real but is served better by keeping PRIMME available as an explicit `"primme"` option for the (rare) precision-sensitive user, not by having it in the auto-selection path.
- The scrapped July 2026 GPU attempt confirmed PRIMME is a poor GPU vehicle for this codebase — its cuBLAS/MAGMA path is fragile at the FFI boundary and re-attempting it is a bad ROI.

**Can in-memory 64-bit sparse SVD be achieved with the other 3 methods?**

Almost — with a caveat. Under `ARMA_64BIT_WORD` (force-defined for Python builds in `libactionet_config.hpp:35-36`) `arma::sp_mat` already uses 64-bit indices, so `nnz` can freely exceed 2^31 today. The only remaining 32-bit choke points are internal narrowings of the `m` and `n` axes to `int` in `svd_irbla.cpp`, `svd_halko.cpp`, and `svd_feng.cpp` (e.g. `svd_halko.cpp:109-110`, `svd_halko.cpp:30,34`). These are matrix-dimension limits (~2.1B rows or cols per axis), not `nnz` limits, and can be widened to `arma::uword` at low risk.

Practical answer: yes, for any realistic single-cell / omics matrix, IRLB / Halko / Feng cover the entire in-memory 64-bit sparse space today because the actual failure axis is `nnz` and that's already 64-bit clean. Widening `m,n` to `arma::uword` is a small future improvement, not a blocker.

**(3) What method/library for GPU-backed SVD as a first pass?**

Decision: prototype two candidates in parallel, commit after a numerical-parity + wallclock spike.

- **Candidate A: RAPIDS RAFT `raft::sparse::solver::sparse_randomized_svd`** — production Apache-2.0 Halko implementation with an operator interface (`matmul` / `rmatmul` on `raft::device_matrix_view`) that maps directly onto `MatrixOperator::matvec`/`rmatvec`. Ships as `libraft` on the rapidsai conda channel, active development. Companion `sparse_lanczos_svd` on the same operator interface as a higher-accuracy fallback.
- **Candidate B: custom Halko on cuBLAS + cuSPARSE + cuSOLVER** — same algorithm as our CPU Halko, no new heavy dependency beyond the CUDA toolkit. Reference implementations: `Michalos88/Randomized_SVD_in_CUDA`, arxiv 2403.06218. Estimated 500-1500 LOC of glue + tests.

Rejected: PRIMME's GPU path (matches the scrapped attempt's failure mode), MAGMA-sparse (officially "Legacy Support Mode" since Jan 2025), Ginkgo (no SVD; eigensolver on 1.10 roadmap), RandLAPACK (upstream: "no one use this as a dependency"), Trilinos/Anasazi (drags in Kokkos+Tpetra+Teuchos; far too heavy), `cusolverDnXgesvdr` alone (dense-only, no operator interface).

---

## Plan

### Phase 1 — Hide PRIMME from the public surface (low-risk cleanup, no C++ deletions)

Rationale: keep PRIMME compiled and reachable via `ALG_PRIMME` in C++ for one release cycle so we can delete it cleanly once GPU support lands and stabilizes. This isolates the SVD-strategy work from the GPU work.

- Remove `"primme"` from the public `_SVD_ALGORITHM_TO_ID` mapping in [src/actionet/decomposition/svd.py](src/actionet/decomposition/svd.py). Reject it in `_normalize_algorithm` with a clear "removed from public API" error.
- Drop the `total_elements > 2^31 -> primme` branch in `_select_svd_algorithm_inmemory` (`svd.py:63-73`). For in-memory sparse, keep IRLB as default; for dense, keep Halko. Document the effective m*n ~ 2^31 caveat in the docstring.
- Remove the "backed IRLB → PRIMME block fast path" in `runSVD_Operator` (`svd_main.cpp:87-95`) *and* in `prefer_block_solver_for_irlb()` in [matrix_operator.hpp](src/libactionet/include/decomposition/matrix_operator.hpp). Backed IRLB should call the honest `svdIRLB(MatrixOperator&, ...)` path (or, if we determine that path is slower than backed Halko in benchmarks, simply steer backed users to Halko in `_select_svd_algorithm_backed`, which is already the default).
- Leave `svd_primme.cpp`, `svd_primme.hpp`, `runSVD_PRIMME_Operator`, and the vendored `src/libactionet/src/extern/primme/` tree in place. Continue to build them for Python. This gives us a single-commit revert path if a downstream pipeline depended on PRIMME.
- Update [src/libactionet/context/DECISIONS.md](src/libactionet/context/DECISIONS.md) and [context/DECISIONS.md](context/DECISIONS.md) with the new SVD strategy (3-method public surface: IRLB default in-memory-sparse, Halko default in-memory-dense and all backed, Feng available; PRIMME quarantined for removal).
- Update [src/libactionet/TODO.md](src/libactionet/TODO.md) to explicitly track "delete PRIMME sources" as a follow-up after the GPU path is stable.
- Update the Halko-default `DECISIONS.md` entry to reflect that Halko is now the unified backed default with no PRIMME fast path underneath.

### Phase 2 — Backed IRLB honesty pass (unblocks Phase 1)

The moment we cut `prefer_block_solver_for_irlb()`, we need to know how the honest `svdIRLB(MatrixOperator&, ...)` path performs on backed data. Two possible outcomes:

- If the honest path is close enough to Halko for realistic backed data (within the bounds of the existing `benchmark_backed_svd_algorithm.py` decision), no further work — the backed default remains Halko and `"irlb"` is available as an explicit second option.
- If the honest path is noticeably slower or memory-heavier, either deprecate the backed `"irlb"` option (documented and error out with a helpful message) or add a small block-matvec loop inside `svdIRLB(MatrixOperator&, ...)` using the existing `MatrixOperator::matmat`/`rmatmat` methods that Halko already uses.

This gets settled empirically before Phase 1 ships. Reuse the existing benchmark at [tests/benchmark_backed_svd_algorithm.py](tests/benchmark_backed_svd_algorithm.py) with `"irlb"` added.

### Phase 3 — GPU Halko: parallel prototype (spike, no public API change)

Goal: pick between RAFT and custom cuBLAS/cuSPARSE Halko based on evidence, not vibes. Both prototypes live behind `LIBACTIONET_ENABLE_NVIDIA_GPU=ON` and produce results through the same `svdHalko(MatrixOperator&, ...)` entry point.

- **Track A — RAFT prototype**:
  - Add `find_package(raft)` to a new `cmake/ConfigureRAFT.cmake`; gate on `LIBACTIONET_ENABLE_NVIDIA_GPU`.
  - Add `src/libactionet/src/decomposition/svd_halko_gpu_raft.cpp` implementing a `MatrixOperator` -> `raft::sparse::solver::sparse_randomized_svd` adapter. Its `matmul`/`rmatmul` implementations copy result blocks between device and host via the existing `MatrixOperator::matmat`/`rmatmat`.
  - For pure in-memory `arma::sp_mat` and `arma::mat`, use RAFT's CSR/dense convenience overloads with a one-shot host-to-device copy.
  - Provide a conda env file (`environment-gpu.yml` already exists) that pulls `libraft`, `libraft-headers`, `rmm`, `cuda-version=12.2` from the rapidsai channel.
- **Track B — custom cuBLAS/cuSPARSE Halko prototype**:
  - Add `src/libactionet/src/decomposition/svd_halko_gpu_native.cu` implementing exactly the algorithm in `svd_halko.cpp:100-199` using cuBLAS (GEMM, QR via `cusolverDnXgeqrf`+`Xorgqr`), cuSPARSE (SpMM for sparse matvec), cuRAND (random sketch), and cuSOLVER `Xgesvdj` for the small k×k SVD.
  - Wire `MatrixOperator` to hand out device pointers directly (add `matmat_device`/`rmatmat_device` optional virtual overrides — CPU operators leave them unimplemented and we materialize on the fly).
  - Reference implementations: `Michalos88/Randomized_SVD_in_CUDA`, arxiv 2403.06218 (study; don't vendor).
- **Selection criteria** (measured at the end of the spike):
  - Numerical parity vs CPU Halko on a representative single-cell backed matrix — singular values, subspace agreement, reconstruction error.
  - Wallclock speedup vs CPU Halko on WSL2 + RTX 4000 Ada (our known-good dev environment).
  - Peak GPU memory footprint.
  - Build/install friction on the target HPC conda environment.
  - Runtime canary robustness (per `plans/GPU_INTEGRATION.md` — must survive the "small dummy solve at startup" test on the WSL2 box that killed the previous attempt).

### Phase 4 — Commit to one GPU path and expose it via the settled Python surface

- Delete the losing prototype. Keep the surviving `svd_halko_gpu*.cpp` behind `LIBACTIONET_ENABLE_NVIDIA_GPU=ON`.
- Add `compute_backend` / `device_id` / `allow_cpu_fallback` kwargs to `run_svd` and `reduce_kernel` exactly as pinned in [plans/GPU_INTEGRATION.md](plans/GPU_INTEGRATION.md) — no rediscussion.
- Add `@requires_gpu` predicate with the runtime canary (dummy Halko solve on device 0). Numerical parity tests live in `tests/test_gpu_svd_parity.py`, gated by `-m gpu`.
- Wire the three CI jobs (`macos_no_gpu`, `linux_cpu_only`, `linux_gpu_smoke_build`) per the roadmap.

### Phase 5 — Final PRIMME deletion (post-stabilization)

Once the GPU Halko path has been stable through one release cycle:

- Delete `svd_primme.cpp`, `svd_primme.hpp`, `runSVD_PRIMME_Operator`, `ALG_PRIMME`.
- Delete `src/libactionet/src/extern/primme/` and `src/libactionet/include/extern/primme/`.
- Delete `cmake/ConfigurePRIMME.cmake` and its CMake wiring.
- Remove the R-vs-Python compile guards around PRIMME in `svd_main.cpp`.
- Update `TODO.md`, `DECISIONS.md`, and the SVD-related docstrings.

---

## Files touched (Phase 1 + 2 explicit, later phases higher-level)

- [src/actionet/decomposition/svd.py](src/actionet/decomposition/svd.py) — remove `"primme"` from public API, drop auto-selection branches.
- [src/libactionet/src/decomposition/svd_main.cpp](src/libactionet/src/decomposition/svd_main.cpp) — remove `prefer_block_solver_for_irlb()` special-case for backed IRLB.
- [src/libactionet/include/decomposition/matrix_operator.hpp](src/libactionet/include/decomposition/matrix_operator.hpp) — remove `prefer_block_solver_for_irlb()` (or keep as no-op default with deprecation comment).
- [src/libactionet/context/DECISIONS.md](src/libactionet/context/DECISIONS.md) and [context/DECISIONS.md](context/DECISIONS.md) — record the new SVD strategy.
- [src/libactionet/TODO.md](src/libactionet/TODO.md) and [TODO.md](TODO.md) — track PRIMME deletion follow-up.
- [tests/benchmark_backed_svd_algorithm.py](tests/benchmark_backed_svd_algorithm.py) — extend to include honest backed IRLB.

Later (Phase 3+):

- `src/libactionet/cmake/ConfigureRAFT.cmake` (new).
- `src/libactionet/src/decomposition/svd_halko_gpu_raft.cpp` (new, Track A).
- `src/libactionet/src/decomposition/svd_halko_gpu_native.cu` (new, Track B).
- `src/actionet/decomposition/svd.py` / `kernel.py` — add the three GPU kwargs at end of Phase 3.
- `tests/test_gpu_svd_parity.py` (new).
