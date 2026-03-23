---
name: Backed Feature Subset Perf
overview: Eliminate the Python-side row-chunked `feature_subset` bottleneck in backed mode for `annotate_cells` and `impute_features` by reformulating them as C++-level matmat operations on the `BackedSparseMatrixOperator`, avoiding per-chunk Python/HDF5/scipy overhead entirely.
todos:
  - id: cpp-extract-columns
    content: Implement `extract_columns(col_indices)` method on `BackedSparseMatrixOperator` in C++ (CSR + CSC variants)
    status: pending
  - id: pybind-binding
    content: Expose `backed_extract_columns` via pybind11 in wp_io.cpp
    status: pending
  - id: update-impute
    content: Replace `feature_subset` in `impute_features` backed path with C++ `backed_extract_columns`
    status: pending
  - id: update-annotate
    content: Replace `feature_subset` in `annotate_cells` backed path with C++ `backed_extract_columns`
    status: pending
  - id: tests
    content: Add correctness test for `backed_extract_columns` and benchmark before/after
    status: completed
isProject: false
---

# Backed Feature-Subset Performance Fix

## Root Cause Analysis

Two functions are severely affected in backed mode: `annotate_cells` and `impute_features`. Both call `MatrixSource.feature_subset()` which has compounding inefficiencies:

**Current `feature_subset` path (Python-side):**

```
for each row-chunk (4096 rows):
    1. h5py reads ALL columns for rows [start:end]  (full-width I/O)
    2. anndata wraps result in scipy sparse             (Python object overhead)
    3. Python slices to requested columns               (copy + discard)
    4. append to list of blocks                         (repeated allocation)
finally: scipy.sparse.vstack all blocks                 (final copy)
```

For a 500K-cell x 30K-gene matrix with 200 marker genes, this executes ~122 Python loop iterations, each doing a full-width HDF5 read, constructing a scipy sparse matrix, slicing it, and copying. The per-chunk Python/C++ boundary crossing (h5py -> numpy -> scipy -> slice -> list append) is the dominant cost, not the raw I/O.

**Contrast with the efficient path:** `compute_feature_specificity` in backed mode already avoids this entirely by dispatching to `BackedSparseMatrixOperator` in C++, which does a single-pass scan over the NNZ data with no Python round-trips.

## Key Insight: Both Functions Can Be Reformulated as matmat

Both downstream C++ consumers perform operations expressible as `S @ M` or `S[:, cols]` followed by a matmat:

1. `**annotate_cells`**: extracts `S_cells = S[:, marker_idx]` then calls `computeFeatureStats(G, S_cells, X_markers, ...)`. The C++ code internally does `S_cells * X_markers` (a matmat). This is equivalent to `S @ (E @ X_markers)` where `E` is a selector matrix -- or more directly, `S @ X_full` where `X_full` is the full-gene-width marker matrix (mostly zeros, already sparse).
2. `**impute_features`**: extracts `X0 = S[:, gene_idx]` then calls `compute_network_diffusion(G, X0, ...)`. The initial seed `X0` is just `S @ e_j` for each requested gene -- a matmat with a selector matrix.

Both can be served by the existing C++`BackedSparseMatrixOperator::matmat()` which already does efficient chunked HDF5 reads in a single C++ pass.

## Plan

### Strategy A: Pass full-width sparse markers to C++ (annotate_cells)

For `annotate_cells`, stop extracting columns. Instead, pass the **full-gene-width** `X_markers` matrix (shape `n_vars x n_celltypes`, very sparse) directly to `compute_feature_stats` / `compute_feature_stats_vision`. The C++ code already handles sparse `X` and only touches columns where `X` is nonzero. This eliminates `feature_subset` entirely.

- In `[annotation.py` lines 375-398](src/actionet/annotation.py), replace the backed branch: instead of calling `source.feature_subset(required_idx)` and then passing the subsetted `S_cells`, pass the full expression matrix (or, for backed mode, construct the backed operator and add a new C++ binding that accepts the operator).

### Strategy B: C++ matmat for seed extraction (impute_features)

For `impute_features`, replace `feature_subset` with a matmat through the backed operator:

- Construct a selector matrix `E` of shape `(n_vars, n_selected_features)` (identity columns at the gene indices).
- Call `BackedSparseMatrixOperator::matmat(E, Y)` via pybind11 to get `Y = S @ E` = the desired `(n_obs, n_selected_features)` dense result.
- This reuses the existing C++ chunked I/O path with zero Python per-chunk overhead.

### Strategy C (Alternative): Dedicated C++ column-extract method

If the matmat indirection via selector matrices is awkward or introduces unnecessary FLOPs (multiplying by identity columns), add a dedicated `extract_columns(col_indices)` method to `BackedSparseMatrixOperator` that:

- Reads NNZ chunks exactly as `matmat_csr`_ does
- Filters to only the requested column indices during the scan
- Returns a dense `(n_obs, n_selected)` matrix

This is the most I/O-efficient: same single-pass scan, but avoids even the selector-matrix multiply overhead.

### Implementation Steps

#### 1. Add `extract_columns` to `BackedSparseMatrixOperator` (C++)

Add a new public method in [backed_sparse_matrix_operator.hpp](src/libactionet/include/io/backed_h5ad/backed_sparse_matrix_operator.hpp):

```cpp
arma::mat extract_columns(const arma::uvec& col_indices) const;
```

Implementation in the `.cpp` file: single-pass CSR scan (same chunk loop as `rmatmat_csr_`), but instead of a matmat accumulation, directly scatter values into the output for matching column indices. Use a hash set or sorted-index lookup for the column filter.

#### 2. Expose via pybind11

In [wp_io.cpp](src/actionet/wp_io.cpp) or a new binding file, expose:

```cpp
m.def("backed_extract_columns", [](BackedSparseMatrixOperator& op, py::array_t<int64_t> col_indices) {
    arma::uvec cols = numpy_to_arma_uvec(col_indices);
    arma::mat result = op.extract_columns(cols);
    return arma_mat_to_numpy(result);
});
```

#### 3. Update `impute_features` backed path

In [imputation.py lines 97-104](src/actionet/imputation.py), replace the `feature_subset` call with:

```python
op = _core.create_backed_operator(file_path, group_path, backed_chunk_size)
X0 = _core.backed_extract_columns(op, feature_indices)
```

#### 4. Update `annotate_cells` backed path

In [annotation.py lines 375-398](src/actionet/annotation.py), two options (pick one):

**Option 4a (simplest):** Use `backed_extract_columns` as a drop-in replacement for `feature_subset`:

```python
op = _core.create_backed_operator(file_path, group_path, backed_chunk_size)
S_cells = _core.backed_extract_columns(op, required_idx)
S = csr_matrix(S_cells)
```

**Option 4b (best perf):** Pass full-width markers directly. This requires a new pybind11 binding `compute_feature_stats_backed(op, X_markers, G, ...)` that constructs `S` columns on-the-fly inside C++ from the backed operator. More invasive but eliminates both the column extraction AND the `scipy_to_arma_sparse(S)` copy.

#### 5. Update `MatrixSource.feature_subset` for non-backed fast path

Keep the current Python implementation for in-memory data (it is fine there). Only reroute backed calls to C++.

#### 6. Tests

- Add test in `tests/backed/` that validates `backed_extract_columns` against the Python `feature_subset` path for correctness.
- Benchmark: time `annotate_cells` and `impute_features` in backed mode before/after.

## Expected Speedup

The current path does `ceil(n_obs / chunk_size)` Python round-trips (~122 for 500K cells). Each involves h5py -> numpy -> scipy -> slice -> append. The C++path does the same I/O but in a single C++ loop with no Python boundary crossings. Based on the existing backed SVD performance (which uses the same C++ operator), the speedup should be **10-50x** for these functions.

## Files to Modify

- `[src/libactionet/include/io/backed_h5ad/backed_sparse_matrix_operator.hpp](src/libactionet/include/io/backed_h5ad/backed_sparse_matrix_operator.hpp)` -- add `extract_columns` declaration
- `[src/libactionet/src/io/backed_h5ad/backed_sparse_matrix_operator.cpp](src/libactionet/src/io/backed_h5ad/backed_sparse_matrix_operator.cpp)` -- implement `extract_columns` for CSR and CSC
- `[src/actionet/wp_io.cpp](src/actionet/wp_io.cpp)` -- pybind11 binding for `backed_extract_columns`
- `[src/actionet/imputation.py](src/actionet/imputation.py)` -- replace `feature_subset` with C++ path
- `[src/actionet/annotation.py](src/actionet/annotation.py)` -- replace `feature_subset` with C++ path
- `[src/actionet/core.py](src/actionet/core.py)` -- add helper to resolve backed file/group path (may already exist as `_backed_group_path`)
