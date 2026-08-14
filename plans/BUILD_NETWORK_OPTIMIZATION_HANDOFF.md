# build_network Optimization Handoff (resume on x86)

Date: 2026-08-14
Status: implementation complete and validated on macOS/arm64; **x86_64
validation and benchmarking pending**
Repository: `actionet-python`
Branch: `dev-gpu` (both parent and `src/libactionet` submodule)

Parent HEAD at handoff: `9794779`
`libactionet` submodule HEAD at handoff: `880be748c47aad07d81dbe4159ab411f64a7e4de`

The work is currently **uncommitted** in the `src/libactionet` submodule working
tree, plus two new untracked test/benchmark files in the parent repo. Preserve
both. When publishing: commit `libactionet` first, then bump the submodule
pointer and commit the parent test/benchmark files.

## What changed

Two files in the `src/libactionet` submodule (the network construction hot
path used by both `algorithm="k*nn"` and `algorithm="knn"`):

- `src/network/_hnsw_jensen_shannon.hpp` — JSD distance kernel rewrite.
- `src/network/build_network.cpp` — parallel graph finalization, KNN merge fix,
  gated phase timers.

New in the parent repo (untracked, both lint-clean):

- `tests/benchmark_build_network.py` — subprocess-isolated old-vs-new benchmark
  harness with a `prep` phase (computes and caches `H_stacked`).
- `tests/test_build_network_invariants.py` — 15 correctness/invariant tests.

Nothing in the Python wrapper changed: the `_core.build_network` ABI is
unchanged, so no `wp_network.cpp` / `build.py` edits were needed.

### 1. JSD kernel (`_hnsw_jensen_shannon.hpp`)

- Removed the ~4 MB `fasterlog2` lookup table (`params[LOGLEN+2]`, `LOGLEN =
  1e6`). The old kernel did 3 data-dependent gathers into that table plus 3
  `floor()` calls and 3 branches per vector element per distance evaluation.
- Now computes `fasterlog2` directly per element via a `memcpy`-based bit
  reinterpret (`jsd_fasterlog2`). The `fastapprox` union type-pun blocked
  clang's loop vectorizer; `memcpy` does not.
- Branchless `x*log2(x)` (`jsd_xlog2x`) using an `(x>0)?v:0` select, plus a
  `#pragma clang loop vectorize(enable)` hint.
- `get_dist_func_param()` now returns `&dim_` instead of a pointer into the LUT.

Numerically this is *slightly more accurate* than the quantized LUT, so it is
**not bit-identical**. Parity is by tolerance (see Validation).

### 2. Finalization + KNN merge (`build_network.cpp`)

- Added `parallel_sort` (OpenMP merge sort: per-thread `std::sort` + pairwise
  `std::inplace_merge`; portable, no parallel-STL backend needed). Serial
  `std::sort` fallback below 65,536 elements or single thread.
- `symmetrize_to_csr` now takes `threads_use` and sorts by a packed 64-bit
  `(lo<<32)|hi` key (one integer compare vs four min/max ops). The old
  `a.dst < b.dst` final tie-break was dropped — it only ordered exact-duplicate
  directed edges whose weight sum is commutative, so the aggregate is unchanged.
- KNN edge merge converted from `#pragma omp critical` (serialized, copying) to
  the lock-free per-thread `AdaptiveScratch` accumulator pattern already used by
  `k*nn`.
- Added `PhaseTimer` / `network_timing_enabled()`: gated on
  `ACTIONET_NETWORK_TIMING=1`, zero overhead when off, prints per-phase seconds
  to stderr (`index_build`, `query`, `merge`, `sort`, `finalize`).

## Build type note (relevant to x86 validation)

- Default `pip install` uses scikit-build-core with `cmake.build-type =
  "Release"` (`-O3`), but **no `-march`**. On x86_64, `__SSE__`/`__SSE2__` are
  baseline so hnswlib L2/IP get SSE and the JSD loop auto-vectorizes to SSE.
  AVX/AVX2/AVX512 are **not** enabled in the default build.
- `install_optimized.sh` adds `-march=native -mtune=native -O3 -ffp-contract=fast
  -funroll-loops` and IPO on Linux x86_64. This is what unlocks AVX2/AVX512 for
  both the JSD loop and hnswlib's hand-written L2/IP SIMD (`space_l2.h`,
  `space_ip.h` dispatch on runtime CPUID: `AVXCapable()` / `AVX512Capable()`).
- The top-level `CMakeLists.txt` was intentionally **not** changed, to preserve
  manylinux portability. Native tuning stays opt-in via `install_optimized.sh`.

## Results so far (macOS/arm64, Apple Silicon, JSD, real cached `H_stacked`)

Baseline = submodule HEAD before these edits; current = with edits. Speedup =
baseline_median / current_median.

| Dataset | Mode | 1 thr | 4 thr | 8 thr |
| --- | --- | --- | --- | --- |
| 100k | knn (k=100) | 2.52x | 2.14x | 1.95x |
| 100k | k\*nn | (see note) | 2.09x | 1.94x |
| 6790 | knn | 3.41x | — | 3.32x |
| 6790 | k\*nn | 3.27x | — | 3.31x |

Note: the first 100k k\*nn single-thread run reported 1.39x but was on a
contended machine; the clean 4/8-thread runs show ~1.9-2.1x.

Phase profiling (`ACTIONET_NETWORK_TIMING=1`) at knn 100k / 8 threads: index
build ~50% and query ~49% of time (both JSD-bound), finalize ~1.6%. So the
distance kernel is the dominant lever; `parallel_sort` matters mainly for
`k*nn`'s ~47M-edge sort and at multi-million-cell scale.

## Validation status

- 15/15 `tests/test_build_network_invariants.py` pass; 38/38 across
  network+diffusion tests.
- JSD parity vs baseline (6790, k\*nn and knn): edge Jaccard >= 0.9996, weight
  correlation 1.000000, max relative weight diff 7.9e-4.
- L2/IP topology variation is HNSW multithread nondeterminism, not a regression:
  baseline-vs-baseline `knn ip` Jaccard was 0.888, *lower* than current-vs-
  baseline 0.954.

## How to resume on x86_64

### 0. Confirm state

```bash
cd actionet-python
git rev-parse --short HEAD                 # expect the parent commit you land on
cd src/libactionet && git status --short   # expect the two modified network files
grep -c "jsd_fasterlog2\|parallel_sort" \
  src/network/_hnsw_jensen_shannon.hpp src/network/build_network.cpp
```

If the submodule changes are absent (fresh clone), re-apply them from this
branch / a saved patch before proceeding.

### 1. Build two ways and compare

The benchmark compares a saved "baseline" `_core.so` (pre-edit) against the
"current" one. Recreate the baseline by stashing the two files, building, saving
the `.so`, then restoring:

```bash
# Portable (default) build first:
pip install -e .                           # or the repo's editable install path

# Baseline _core.so (pre-edit):
cd src/libactionet
git stash push src/network/_hnsw_jensen_shannon.hpp src/network/build_network.cpp
cd ../../build/<wheel_tag> && ninja _core
cp _core*.so ../../tests/_bench_cache/_core_baseline.so
cd ../../src/libactionet && git stash pop
cd ../../build/<wheel_tag> && ninja _core   # current, then reinstall the .so
```

**x86-specific:** also build an optimized variant to measure AVX2/AVX512 gains:

```bash
./install_optimized.sh                      # -march=native, unlocks AVX
```

Rebuild both baseline and current under the optimized flags for an apples-to-
apples optimized comparison (SIMD affects baseline L2/IP and the current JSD
loop differently).

### 2. Prep embeddings (backed mode for large data)

```bash
python tests/benchmark_build_network.py prep \
  --dataset data/actionet_benchmark/scale_subset_100k.h5ad \
  --out tests/_bench_cache/h_100k.npy --backed --chunk 4096
```

Repeat for 25k/50k/150k/200k and, since 300k runs on the dev machine,
`data/adata_agg_Scn4b_OX_fil.h5ad` (300,157 cells) to get a real large-scale
`knn` point. Prep runs the full ACTION pipeline once and caches the small
`H_stacked` array; benchmarks then reuse it.

### 3. Benchmark

```bash
python tests/benchmark_build_network.py bench \
  --embedding tests/_bench_cache/h_100k.npy \
  --algorithms "knn,k*nn" --metric jsd --k 100 \
  --threads 1,4,8,16,32 --trials 3 \
  --baseline-core tests/_bench_cache/_core_baseline.so \
  --out tests/_bench_cache/results_100k.jsonl
```

Each build runs in its own subprocess (the `_core` extension always exports
`PyInit__core`, so baseline and current cannot coexist in one process).

### 4. Phase profiling when a number looks off

```bash
ACTIONET_NETWORK_TIMING=1 python -c "
import numpy as np; from actionet import _core
H=np.load('tests/_bench_cache/h_100k.npy').astype('float32')
_core.build_network(H,'knn','jsd',1.0,8,16.,200.,200.,True,100)"
```

## x86 validation goals (what to actually check)

1. Reproduce the arm64 speedups on x86 default (`-O3`, SSE-only) build. Expect
   the JSD win to hold; confirm no regression on L2/IP.
2. Measure the *additional* gain from `install_optimized.sh` (AVX2/AVX512). The
   JSD loop should vectorize wider; quantify vs portable build.
3. Run a real large tier (>=300k) in backed mode for `knn` (the priority path)
   and record wall time + peak RSS. Compare against the projections in
   `tests/ACTIONet Scaling Benchmark Run 001 Report.md`.
4. Re-run the JSD parity check on x86 (numerics can differ slightly across ISA):
   edge Jaccard should stay >= ~0.999, weight correlation ~1.0.
5. Thread scaling to high core counts (32/44): confirm the KNN merge and
   `parallel_sort` changes hold up where the old `omp critical` and serial sort
   would bite.

## Reproducing the correctness / parity check

A quick worker used during development (compares graph edge sets + weights
between two `_core` builds); recreate or adapt as needed:

```bash
# writes G to npz for a given core (CUR = installed, or a path to a baseline .so)
python /tmp/corr_worker.py CUR  tests/_bench_cache/h_6790.npy knn jsd /tmp/cur.npz
python /tmp/corr_worker.py tests/_bench_cache/_core_baseline.so \
  tests/_bench_cache/h_6790.npy knn jsd /tmp/base.npz
# then compare edge Jaccard + weight correlation
```

Or just rely on `tests/test_build_network_invariants.py` for regression safety.

## Housekeeping

- `tests/_bench_cache/` is git-ignored (contains the baseline `.so`, cached
  `.npy` embeddings, `*.work.h5ad`, and result JSONL). It will not be committed;
  recreate it on the x86 machine via `prep`.
- The `PhaseTimer` instrumentation is intentionally retained (gated, zero
  overhead). Keep it unless there's a reason to remove.

## Open follow-ups (not done)

- Stage D (deferred): revisit `k*nn`'s `5*sqrt(N)` neighbor heuristic and `ef`
  floor as a benchmarked default change (subject to the DECISIONS backward-compat
  bar). `k*nn` remains algorithmically O(N^1.5); the report already recommends
  `knn` for large data.
- Backed streaming into HNSW: `build_network` still materializes the embedding
  in memory. A backed row-streaming reader is a larger, separate effort.
- GPU/cuVS: explicitly out of scope for this pass (CPU + disk I/O priority).
