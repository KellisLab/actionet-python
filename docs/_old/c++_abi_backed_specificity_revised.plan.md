---
name: ""
overview: ""
todos: []
isProject: false
---

# Backed Sparse `computeFeatureSpecificity` via `libactionet` + Python Fallback Preservation

**Summary**

- Implement backed sparse feature-specificity natively in `libactionet` as new `BackedSparseMatrixOperator` overloads of `computeFeatureSpecificity`.
- Keep backed **dense** inputs on the existing Python streamed fallback. `_compute_specificity_streamed` exists specifically to handle dense-backed AnnData: there is no `BackedDenseMatrixOperator` in `libactionet`, the shift-and-count algorithm has no natural dense streaming primitive in the C++core, and dense-backed count matrices are uncommon enough that a C++ dense path is not worth the added complexity in this pass. Do not regress non-sparse backed AnnData.
- Route every backed specificity caller in `actionet-python` through one dispatcher so sparse-backed `compute_feature_specificity`, `compute_archetype_feature_specificity`, and marker detection all use the same C++ path.
- Land the C++ change in `libactionet` first, then advance the `actionet-python` submodule pointer and add the Python binding/caller work.

**Key Changes**

- `libactionet`
  - Add non-template overloads in `annotation/specificity.hpp` for:
    - `computeFeatureSpecificity(BackedSparseMatrixOperator&, arma::mat&, int)`
    - `computeFeatureSpecificity(BackedSparseMatrixOperator&, arma::uvec&, int)`
  - Forward-declare `BackedSparseMatrixOperator` in the specificity header instead of including the backed HDF5 header there. Keep the backed H5AD surface out of the umbrella include path.
  - Update the specificity docs so the new backed overload is documented as non-mutating. The existing in-memory template overloads still mutate `S` in place.
  - In `specificity.cpp`, implement the backed sparse overload as a single scan over stored entries that reuses the operator's existing chunk readers and `transform_value_()` logic. During that scan, accumulate:
    - `min_val` over logical values plus implicit zero
    - `row_count` / `col_count` from stored-entry counts
    - `row_factor_sum_orig`
    - `obs_orig`
    - `support_obs` (binary support weighted by `H_norm`)
  - After the scan, apply the shift analytically:
    - `shift = -min_val`
    - `row_factor_sum = row_factor_sum_orig + shift * row_count`
    - `Obs = obs_orig + shift * support_obs`
  - Reuse the existing Bernstein-tail math after those corrected quantities are formed.
  - Preserve sparse semantics exactly: for backed sparse matrices, nnz counts must follow stored entries, not post-shift `> 0` tests.
  - Support both CSR and CSC-backed storage.
  - Do not generalize this to `MatrixOperator`; the algorithm needs stored-support counts and implicit-zero semantics that generic matvec/rmatvec cannot provide.
  - Prefer friend access from the backed operator to the new overload rather than adding a public `streamStats()` API that would expose low-level backed internals without solving the shift problem.
- `actionet-python`
  - Add new `_core` bindings in `wp_annotation.cpp` following existing naming conventions:
    - `archetype_feature_specificity_backed_operator(op, H, thread_no)`
    - `compute_feature_specificity_backed_operator(op, labels, thread_no)`
  - Preserve current dict shapes/keys:
    - archetype path returns `"archetypes"`, `"upper_significance"`, `"lower_significance"`
    - label path returns `"average_profile"`, `"upper_significance"`, `"lower_significance"`
  - In `core.py`, add one private backed-specificity dispatcher:
    - if `source.is_backed and source.is_sparse`, create a `BackedSparseMatrixOperator` and call the new binding
    - if backed but **dense** (i.e. `source.is_backed and not source.is_sparse`), fall back to `_compute_specificity_streamed` — this is the intended and permanent role of that function for backed inputs
    - keep `_compute_specificity_streamed` as the dense-backed fallback and as a parity oracle
  - `compute_feature_specificity()` should use the labels-backed binding directly for sparse-backed inputs instead of building the membership matrix in Python.
  - `compute_archetype_feature_specificity()` should validate `H` as `(n_obs, k)`, transpose once to `k x n_obs`, and use the archetype-backed binding for sparse-backed inputs.
  - `annotation.py` should stop calling `_compute_specificity_streamed()` directly for backed marker detection. Route through `compute_feature_specificity(..., return_raw=True)` so backed sparse marker detection also takes the new C++ path and dense-backed marker detection still falls back correctly.
  - Update the docstrings/comments for both public specificity functions so backed sparse vs backed dense behavior is explicit.
- Cross-repo integration
  - Make the C++ change in the standalone `libactionet` repo first.
  - Update the `actionet-python` submodule pointer after the new `libactionet` commit exists.
  - No `actionet-r` wrapper work in this pass. The overload is valid C++ API surface, but backed H5AD specificity remains a Python-side workflow.

**Expected Performance and Memory Impact**

The estimates below use the production reference dataset (1,792,201 cells × 28,692 genes, nnz = 5.7 × 10⁹, ~11% density, stored as int64 CSR in a ~128 GB h5ad file) as the sizing basis, with `k = 50` archetypes / label groups as a representative problem size.

- **Wall-clock time.** The current Python streamed path iterates `n_obs / chunk_size` ≈ 438 chunks at the default `chunk_size = 4096`. For each chunk it executes Python dispatch overhead, `tocsr()`, NumPy sparse-to-CSR conversion, and multiple intermediate array allocations. At realistic per-chunk overheads of 50–200 ms (dominated by Python dispatch and NumPy allocation, not IO), this gives 6–24 hours of wall time for the full dataset. The C++ path performs two sequential streaming passes over the stored data (one stat-accumulation scan + one matmat call), both at native memory bandwidth. On typical HPC storage (1–4 GB/s sequential read), reading the ~64 GB data+indices payload twice takes roughly 30–120 seconds. **Expected improvement: 200× to 3000× speedup**, bringing a multi-hour job to under 5 minutes.
- **Process-local peak memory.** The Python path allocates per-chunk temporaries (sparse block, shifted copy, intermediate CSR, dense matmul output) that sum to roughly `chunk_nnz × 3 × 8 bytes + chunk_size × k × 8 bytes` per chunk — but Python's allocator fragments and holds these across chunks. In practice, peak working set is several hundred MB to ~2 GB at default chunk size for this dataset. The C++ path holds exactly two streaming buffers of size `chunk_nnz × (8 + 8) bytes` (data + indices) plus the output accumulators `(n_var × k) × 8 bytes × 3 ≈ 3 × 28692 × 50 × 8 ≈ 34 MB`. **Expected improvement: peak process memory reduced from ~0.5–2 GB to ~100–200 MB**, well within a single compute node's L3+DRAM budget for the accumulator arrays.
- **Python GIL and threading.** The current path holds the GIL continuously throughout all chunk iterations. The C++ path releases the GIL for the duration of both streaming passes (standard pybind11 `py::call_guard<py::gil_scoped_release>`), allowing other Python threads to proceed and enabling future OpenMP parallelization of the stat-accumulation scan without API changes.
- **Dense-backed fallback.** `_compute_specificity_streamed` remains unchanged for dense-backed inputs. Dense-backed count matrices are rare in practice (all standard AnnData writers produce sparse CSR for count data), and the Python streamed path is adequate for the dense case given typical dense dataset sizes.

**Test Plan**

- Add focused `pytest` coverage in `actionet-python`; do not introduce a new C++ unit-test harness in `libactionet` for this change.
- Cover these cases directly:
  - backed sparse CSR `compute_feature_specificity` matches the in-memory C++ path
  - backed sparse CSC `compute_feature_specificity` matches the in-memory C++ path
  - backed sparse `compute_archetype_feature_specificity` matches the in-memory C++ path
  - sparse negative-valued input matches current sparse in-memory semantics, especially stored-nnz counting after shift
  - backed dense input still works by falling back to `_compute_specificity_streamed`
  - backed `find_markers` still matches in-memory rankings/order closely after routing through the new path
- Keep the existing end-to-end parity smoke for archetype specificity and extend backed tests under `tests/backed/` rather than relying only on the large pipeline parity script.

**Assumptions And Defaults**

- Scope is backed sparse AnnData matrices stored as CSR or CSC groups under `.X` or `layers[...]`. Dense-backed matrices stay on the Python streamed fallback permanently (not just in v1); this is the intended division of responsibility.
- No new `allow_compressed` or auto-decompression flag is added for specificity in this pass. Unlike operator SVD, specificity is a bounded sequential scan workload.
- `thread_no` stays in the API for parity, but the backed scan itself remains effectively single-threaded unless a separate parallelization change is made later.
- Success means semantic parity with the current in-memory sparse implementation and removal of the Python streamed path from primary sparse-backed specificity callers.
