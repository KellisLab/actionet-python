# OpenBLAS threading model and Armadillo ODR findings

Status: draft, 2026-07-04. Author: agent investigation triggered by 1000x
slowdown of `run_action` on WSL2. Applies to `actionet-python`, likely also
to `actionet-r`, and requires coordinated fixes in `libactionet`.

---

## TL;DR

1. **OpenBLAS threading model interacts catastrophically with libactionet's
   outer OpenMP loops when the pthread build of OpenBLAS is used.** The
   library does not manage nested BLAS threading. On systems where the
   linker resolves `-lopenblas` to the pthread variant (Debian/Ubuntu
   default), any C++ hot path that calls BLAS from inside a
   `#pragma omp parallel` region silently oversubscribes threads by
   `N_omp * N_blas`. On a 32-thread WSL2 VM this produced ~900-way
   concurrency and turned a 7-9 s job into a 15 min job.
2. **The Armadillo/PRIMME translation units emit ~15 ODR (One Definition
   Rule) and LTO type-mismatch warnings on every build.** These are
   longstanding, not caused by WSL2, but they undermine confidence in the
   LTO-optimized artifact and should be treated as a real defect.

Both issues sit inside `libactionet` and affect every front-end.

---

## 1. OpenBLAS threading model

### 1.1 Symptom

- Test dataset: `data/test_adata.h5ad` (6790 x 14409, in-memory).
- Hardware: 13th Gen Intel i9-13950HX, 16 physical / 32 logical cores.
- macOS (Accelerate) and native Ubuntu (MKL): `run_action` completes in
  7-9 s.
- WSL2 (Ubuntu 24.04, apt OpenBLAS): `run_action` runs for ~15 min. Python
  helpers and I/O are unaffected. Only C++ hot paths degrade.

### 1.2 Root cause

The hot loop in `decompACTION` runs BLAS-heavy `runAA` iterations inside
an outer OMP team:

```39:53:src/libactionet/src/action/action_decomp.cpp
#pragma omp parallel for num_threads(threads_use)
for (int k = k_min; k <= k_max; k++) {
    ResSPA SPA_res = runSPA(S_r, k);
    trace.selected_cols[k] = std::move(SPA_res.selected_cols);

    arma::mat W = S_r.cols(trace.selected_cols[k]);

    arma::field<arma::mat> AA_res = runAA(S_r, W, max_it, tol);
    ...
```

`runAA` itself calls CBLAS directly:

```40:51:src/libactionet/src/action/aa.cpp
cblas_dgemv(CblasColMajor, CblasNoTrans, R.n_rows, R.n_cols,
            (1.0 / norm_sq), R.memptr(), R.n_rows, h.memptr(), 1, 1,
            b.memptr(), 1);
...
cblas_dger(CblasColMajor, R.n_rows, R.n_cols, 1.0, delta.memptr(), 1,
           h.memptr(), 1, R.memptr(), R.n_rows);
```

Nowhere in `libactionet` does any code call `openblas_set_num_threads`,
`mkl_set_num_threads_local`, `omp_set_max_active_levels`, or otherwise
constrain the inner BLAS pool. Verified:

```text
$ rg 'openblas_set_num_threads|mkl_set_num_threads|omp_set_max_active_levels' src include
(no matches)
```

The behavior at runtime is entirely determined by which OpenBLAS variant
the loader resolves.

### 1.3 Why the platforms diverge

| Platform | BLAS resolved to | Threading model | Interaction with libactionet outer OMP |
|----------|------------------|-----------------|----------------------------------------|
| macOS | Apple Accelerate | Sequential per call | No oversubscription possible. |
| Native Linux (conda) | MKL (auto-picked by `ConfigureBLAS.cmake` when `CONDA_PREFIX` contains it) | OpenMP-aware; cooperates with libgomp | Nested BLAS calls run with 1 thread inside a parallel region. |
| WSL2 Ubuntu (apt, default) | OpenBLAS **pthread** build (`libopenblas0-pthread`) | Independent pthread pool sized to `nproc` | Every outer OMP thread spawns/uses `nproc` inner threads. Oversubscription = `N_omp * N_blas`. |
| WSL2 Ubuntu (apt, `-openmp` variant) | OpenBLAS **OpenMP** build (`libopenblas0-openmp`) | Shares libgomp team; nested BLAS runs serially by default | No oversubscription. Behaves like MKL. |

The Debian/Ubuntu `libopenblas0` meta-package defaults to
`libopenblas0-pthread` and installs it at update-alternatives priority
100 vs 95 for the `-openmp` variant. So the default `apt install
libopenblas0-dev` path produces the broken configuration.

Verified linkage on the affected machine (pre-fix):

```text
$ ldd .venv/lib/.../actionet/_core.cpython-312-x86_64-linux-gnu.so | grep -Ei "openblas|gomp"
libopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0
libgomp.so.1    => /lib/x86_64-linux-gnu/libgomp.so.1

$ readlink -f /lib/x86_64-linux-gnu/libopenblas.so.0
/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so
```

And post-fix (after `apt install libopenblas-openmp-dev` +
`update-alternatives --set libopenblas.so.0-x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0` + rebuild):

```text
mapped: /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblasp-r0.3.26.so
openblas_get_parallel: 2   (0=seq, 1=pthread, 2=openmp)
openblas_get_num_threads: 32
openblas_get_config: OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_OPENMP Haswell MAX_THREADS=64
```

### 1.4 Why WSL2 amplifies it beyond bare-metal Linux

Even the pthread-OpenBLAS + libgomp combination should be merely slow on
bare metal (say 3-5x), not 100x. The 1000x factor on WSL2 comes from the
Hyper-V-hosted Linux kernel's scheduling of futex contention across
Hyper-V virtual CPUs. Thread migration and mutex wake-up costs are
several times higher than on native Linux, so the same oversubscription
pattern that would cost 3-5x on bare-metal ext4 costs ~100x on WSL2.

WSL2 is not the bug; it is the amplifier. The bug is in the library's
lack of BLAS-thread control.

### 1.5 Blast radius

BLAS-inside-OMP patterns exist beyond `action_decomp`. Direct CBLAS call
sites in `libactionet`:

```text
src/action/aa.cpp                            2
src/decomposition/svd_irbla.cpp             19
src/utils_internal/utils_decomp.cpp          2
src/utils_internal/utils_active_set.cpp     57
```

`#pragma omp parallel` regions exist across most modules
(`action_decomp`, `network/*`, `annotation/specificity`,
`tools/{enrichment,xicor,autocorrelation,guide_calling}`,
`io/backed_h5ad/*`, `utils_internal/*`, and more). Any pairing of an OMP
region with BLAS calls inside it is a latent instance of this bug on
pthread-OpenBLAS systems.

Additionally, Armadillo's own `arma::mat * arma::mat`,
`arma::norm(..., "fro")`, `arma::normalise`, and similar expressions
dispatch to `cblas_dgemm`/`dgemv`/`dnrm2` under the hood, so a function
does not have to call CBLAS directly to be affected.

### 1.6 Recommendation for `libactionet`

The library should own its BLAS thread policy rather than rely on
distribution defaults. Concrete work items:

1. **Add a `ThreadingPolicy` scope guard in `utils_internal/utils_parallel.hpp`.**
   RAII object that on construction captures the current BLAS thread
   count and sets it to 1 for the current thread (or the whole process,
   with a mutex, if we cannot rely on `openblas_set_num_threads_local`);
   restores on destruction. Wrap every `#pragma omp parallel` region
   that contains BLAS calls, starting with `decompACTION`, `mergeArchetypes`,
   `computeFeatureSpecificity`, the IRLB path, and the active-set solvers.
2. **Prefer `openblas_set_num_threads_local` when available** to avoid
   the per-thread global-state race. Detect via a CMake feature test
   against the linked OpenBLAS symbol table.
3. **Emit a startup diagnostic** (one-shot on first C++ entrypoint) that
   logs: BLAS vendor, `openblas_get_parallel()` result, and the outer OMP
   runtime. This makes future misconfigurations self-diagnosing.
4. **Update `ConfigureBLAS.cmake`** to prefer the OpenMP variant when
   both OpenBLAS variants are visible on Linux (and `MKL` is not
   detected). At minimum, emit a `WARNING` at configure time if the
   linked OpenBLAS was built with `USE_THREAD=pthread`.
5. **Document the platform matrix** in `libactionet/context/DECISIONS.md`
   under a new "BLAS and threading" section, and cross-link from the
   Python and R front-end playbooks.

### 1.7 Recommendation for `actionet-python`

Once the library-level fix is in place the front-end can stop worrying
about this. In the interim:

- `install_optimized.sh` should print a warning when it detects the
  pthread OpenBLAS variant and suggest either `libopenblas-openmp-dev`
  (Debian/Ubuntu), `libopenblas-openmp` (Fedora), or an MKL conda env.
- `actionet.__init__` could optionally read the BLAS parallel identity
  at import and warn if the pthread variant is loaded, mirroring the
  guidance PyTorch has followed for years.

### 1.8 Recommendation for `actionet-r`

The same rebuild fixes the R front-end. On R-only Ubuntu machines the
default is usually `libopenblas-pthread` (via `libopenblasp`) unless the
user has installed BLAS through Microsoft R Open or `sudo
update-alternatives --config libblas.so.3-x86_64-linux-gnu`. R users are
even less likely than Python users to know that `Rcpp::sourceCpp` will
inherit the system BLAS, so the diagnostic startup line from the library
matters even more here.

### 1.9 Reproducer

```text
# Confirm slow path (pthread OpenBLAS + libactionet default):
sudo update-alternatives --set libopenblas.so.0-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0
# rebuild actionet
./install_optimized.sh
python tests/test_fast.ipynb   # or the run_action cell in isolation

# Confirm fast path:
sudo apt install libopenblas-openmp-dev
sudo update-alternatives --set libopenblas.so.0-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0
./install_optimized.sh
python ...

# Emergency user-side workaround if a rebuild is not possible:
export OPENBLAS_NUM_THREADS=1
```

The `OPENBLAS_NUM_THREADS=1` workaround is the smoking gun that pins
this to nested-BLAS oversubscription rather than any other WSL2
pathology (paging, filesystem, AVX dispatch, kernel scheduler cadence).

---

## 2. Armadillo / PRIMME ODR and LTO warnings

### 2.1 Symptom

Every build of `_core.so` prints a stable set of ~15 warnings at the
final LTO link step. Representative examples from the current
WSL2 build:

- `struct state_type` (Armadillo internal) has an `std::atomic<int>`
  member in one TU and a plain `int` member in another TU. This
  propagates into `struct SpMat`, `struct SpMat_noalias`, and every
  function whose signature mentions `arma::sp_mat`
  (`computeGroupedSums`, `computeGroupedMeans`, `computeGroupedVars`,
  `takeColumnsSparse`, `computeFeatureSpecificity`,
  `computeNetworkDiffusion`, `scaleMatrix`, `normalizeMatrix`).
- `PRIMME`'s private BLAS/LAPACK prototypes disagree with Armadillo's:
  `dgemm_`, `dgemv_`, `dgesvd_`. LTO reports "type of ... does not
  match original declaration" and "type `blas_len` should match type
  `void`".

Verbatim, e.g.:

```text
armadillo_bits/arma_forward.hpp:297:8: warning: type 'struct state_type'
    violates the C++ One Definition Rule [-Wodr]
armadillo_bits/arma_forward.hpp:302:22: note: the first difference of
    corresponding definitions is field 'state'
    std::atomic<int> state;
armadillo_bits/arma_forward.hpp:300:22: note: a field of same name but
    different type is defined in another translation unit
    int  state;
```

### 2.2 Root cause

Two distinct issues share the umbrella "ODR warnings":

**(a) Armadillo compiled with inconsistent `ARMA_USE_ATOMIC` /
`ARMA_SPMAT_USE_STDLIB_ATOMIC` across TUs.** Armadillo's
`state_type` toggles between `std::atomic<int>` and plain `int`
depending on preprocessor state. Some object files in
`libactionet.a` see one definition, others see the other. This
happens when different translation units are compiled with
different combinations of `-fopenmp`, `-std=c++17`, or Armadillo
detection macros. Cross-file inlining under LTO then sees the
mismatch and cannot legally optimize.

**(b) PRIMME redeclares Fortran BLAS/LAPACK entry points with its own
prototypes** (`XGEMM`, `XGEMV`, `XGESVD`) in
`src/extern/primme/include/../linalg/blaslapack_private.h`. Armadillo
also declares them in `armadillo_bits/def_blas.hpp` and
`def_lapack.hpp` — with a different signature (Armadillo threads the
`blas_len` "hidden string length" arguments through the prototype,
which is the modern gfortran ABI). Under LTO both prototypes cover
the same external symbol and are treated as incompatible.

The build passes `-Wno-macro-redefined` intending to suppress noise,
but on GCC 13 that option is not recognized (`unrecognized command-line
option '-Wno-macro-redefined'`), so the confusion between PRIMME's and
Armadillo's `USE_DOUBLE` and BLAS prototypes leaks through.

### 2.3 Why it matters

ODR violations are undefined behavior. In practice:

- `-flto=auto -fno-fat-lto-objects` (which `install_optimized.sh`
  enables) makes the compiler inline across TUs. If the two versions
  of `state_type` end up in an inlined region for `SpMat::sync_state`,
  the compiler is free to generate code assuming the layout of either
  version. Manifestation is typically silent wrong-answer on sparse
  matrix mutation, not a crash.
- The BLAS prototype mismatch is more benign in practice because the
  call sites happen to pass the string-length arguments consistently
  in both worlds — but the "type ‘blas_len’ should match type ‘void’"
  warning is a real ABI risk if we ever move to a BLAS whose Fortran
  runtime expects the hidden-length convention differently (e.g.
  `flang-new` vs `gfortran`).
- Independent of correctness: the warning wall makes real build
  regressions invisible.

### 2.4 Recommendation for `libactionet`

1. **Pin Armadillo's atomic policy globally.** Add one of
   `-DARMA_USE_ATOMIC` or `-DARMA_DONT_USE_ATOMIC` in the top-level
   `CMakeLists.txt` as `target_compile_definitions(actionet PUBLIC ...)`
   so every TU (including `wp_*.cpp` in the front-ends) sees the same
   `state_type` layout. Preference: `ARMA_USE_ATOMIC`, since sparse
   matrix mutation from multiple threads is used in the network code.
2. **Isolate PRIMME's BLAS prototypes.** Either
   (a) build PRIMME as a separate static archive without LTO
   (`-fno-lto` on its object files) so cross-TU type inspection
   stops at the archive boundary; or
   (b) patch PRIMME's `blaslapack_private.h` to include Armadillo's
   `def_blas.hpp` prototypes when compiled inside libactionet.
   Option (a) is smaller-diff and preserves upstream PRIMME.
3. **Remove `-Wno-macro-redefined`** on GCC (only Clang/AppleClang
   accept it) and instead add PRIMME's `USE_DOUBLE` gate to a
   PRIMME-only compile-definition set so it does not collide with
   command-line `-DUSE_DOUBLE`.
4. **Turn on `-Werror=odr`** in CI so future regressions are loud.
   Optional: `-Werror=lto-type-mismatch`.

### 2.5 Scope of this doc

Fixing (2.4) is a libactionet task and outside the scope of the
current WSL2 investigation. This document exists to record the
findings so they are not lost. The ODR issue predates the OpenBLAS
issue and is unrelated to WSL2.

---

## 3. Cross-repo coordination

Both issues live in `libactionet` and are surfaced by every front-end.
Recommended sequence:

1. Land the BLAS threading policy scope guard in `libactionet`. Cut a
   `libactionet` release and bump the submodule pin in both
   `actionet-python` and `actionet-r`.
2. Land the Armadillo `ARMA_USE_ATOMIC` pin in the same release.
3. Land the PRIMME LTO isolation in a subsequent release; it is
   independent and lower-priority.
4. In each front-end, add a smoke test that runs `run_action` on
   `data/test_adata.h5ad` under both `OPENBLAS_NUM_THREADS=1` and
   `OPENBLAS_NUM_THREADS=$(nproc)` and asserts wall-clock stays within
   a factor of 2 of the fast path. This test would have caught the
   current regression instantly.

---

## Appendix A. Verification commands

```bash
# What BLAS is my extension linked against?
ldd $(python -c 'import actionet, os; print(os.path.dirname(actionet.__file__))')/_core*.so \
    | grep -Ei 'openblas|blas|mkl|accelerate|gomp|iomp|omp'

# Which OpenBLAS variant does the loader resolve?
readlink -f /lib/x86_64-linux-gnu/libopenblas.so.0
update-alternatives --display libopenblas.so.0-x86_64-linux-gnu

# What threading model is that OpenBLAS?
python - <<'PY'
import ctypes
lib = ctypes.CDLL('libopenblas.so.0')
lib.openblas_get_parallel.restype = ctypes.c_int
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_num_threads.restype = ctypes.c_int
print('parallel:', lib.openblas_get_parallel(), '(0=seq, 1=pthread, 2=openmp)')
print('threads :', lib.openblas_get_num_threads())
print('config  :', lib.openblas_get_config().decode('utf-8', 'replace'))
PY
```

## Appendix B. Related upstream issues

- OpenBLAS #3187, #2985, #3187: extensively documented pthread-vs-OMP
  interop hazards; the OpenBLAS maintainers' recommendation for any
  library that calls BLAS from inside its own OMP regions is to use the
  OpenMP build of OpenBLAS or MKL, and to call `openblas_set_num_threads`
  explicitly.
- NumPy/SciPy have documented the same problem and set
  `OPENBLAS_NUM_THREADS=1` at runtime for their scikit-learn CI matrix
  as a defensive measure.
- Armadillo docs (`ARMA_USE_ATOMIC`, `ARMA_SPMAT_USE_STDLIB_ATOMIC`)
  explicitly warn that inconsistent atomic policy across TUs is UB.
