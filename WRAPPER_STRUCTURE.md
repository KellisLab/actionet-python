# ACTIONet Python Wrapper Structure

## Overview

The pybind11 bindings for ACTIONet Python have been reorganized to match the modular structure of the R package's Rcpp wrappers. Each module is now in a separate file, following the naming convention `wp_*.cpp` (wrapper-python) to parallel the R package's `wr_*.cpp` (wrapper-R) structure.

## File Organization

### Core Structure

```
src/actionet/
├── _core_new.cpp          # Main module definition (replaces _core.cpp)
├── wp_utils.h             # Conversion utilities header
├── wp_utils.cpp           # Conversion utilities implementation
├── wp_action.cpp          # ACTION module bindings
├── wp_network.cpp         # Network module bindings
├── wp_annotation.cpp      # Annotation module bindings
├── wp_decomposition.cpp   # Decomposition/batch correction bindings
├── wp_tools.cpp           # Tools module bindings (SVD)
└── wp_visualization.cpp   # Visualization module bindings
```

### Comparison with R Package Structure

| R Package | Python Package | Description |
|-----------|----------------|-------------|
| `wr_action.cpp` | `wp_action.cpp` | ACTION decomposition and kernel reduction |
| `wr_network.cpp` | `wp_network.cpp` | Network construction and diffusion |
| `wr_annotation.cpp` | `wp_annotation.cpp` | Feature specificity and markers |
| `wr_decomposition.cpp` | `wp_decomposition.cpp` | Batch correction and orthogonalization |
| `wr_tools.cpp` | `wp_tools.cpp` | SVD and other utilities |
| `wr_visualization.cpp` | `wp_visualization.cpp` | Network layout (UMAP/t-SNE) |
| N/A | `wp_utils.cpp` | Python-specific conversion utilities |

## Module Details

### 1. `wp_utils.cpp` / `wp_utils.h`

**Purpose:** Conversion utilities between Python (NumPy/SciPy) and C++ (Armadillo) data structures.

**Functions:**
- `numpy_to_arma_mat()`: NumPy array → Armadillo dense matrix
- `scipy_to_arma_sparse()`: SciPy sparse matrix → Armadillo sparse matrix
- `arma_mat_to_numpy()`: Armadillo dense matrix → NumPy array
- `arma_sparse_to_scipy()`: Armadillo sparse matrix → SciPy sparse matrix
- `numpy_to_arma_vec()`: NumPy vector → Armadillo vector
- `arma_vec_to_numpy()`: Armadillo vector → NumPy array

These utilities handle the row-major (NumPy) to column-major (Armadillo) conversion and sparse matrix format translation.

---

### 2. `wp_action.cpp`

**Purpose:** ACTION decomposition and kernel reduction bindings.

**Functions:**
- `run_action()`: Multi-resolution archetypal analysis
  - Corresponds to `actionet::runACTION()`
  - Returns H matrices (cell assignments), C matrices (archetype profiles), and discrete assignments

- `reduce_kernel_sparse()`: Kernel reduction for sparse matrices
  - Corresponds to `actionet::reduceKernel()`
  - Returns reduced representation S_r, singular values, and SVD components

- `reduce_kernel_dense()`: Kernel reduction for dense matrices
  - Same as sparse version but for dense input

**Initialization:** `init_action(py::module_ &m)`

---

### 3. `wp_network.cpp`

**Purpose:** Network construction and network-based operations.

**Functions:**
- `build_network()`: Construct cell-cell similarity network
  - Corresponds to `actionet::buildNetwork()`
  - Supports multiple algorithms (k*nn, knn, etc.)
  - Returns sparse adjacency matrix

- `compute_network_diffusion()`: Smooth data over network topology
  - Corresponds to `actionet::computeNetworkDiffusion()`
  - Implements iterative diffusion process

**Initialization:** `init_network(py::module_ &m)`

---

### 4. `wp_annotation.cpp`

**Purpose:** Feature specificity and marker gene identification.

**Functions:**
- `compute_feature_specificity_sparse()`: Compute archetype/cluster-specific marker genes
  - Corresponds to `actionet::computeFeatureSpecificity()`
  - Returns average profiles and significance scores

**Initialization:** `init_annotation(py::module_ &m)`

---

### 5. `wp_decomposition.cpp`

**Purpose:** Batch correction and orthogonalization methods.

**Functions:**
- `orthogonalize_batch_effect_sparse()`: Remove batch effects from sparse data
  - Corresponds to `actionet::orthogonalizeBatchEffect()`
  - Takes design matrix encoding batch membership

- `orthogonalize_batch_effect_dense()`: Remove batch effects from dense data
  - Same as sparse version but for dense input

- `orthogonalize_basal_sparse()`: Remove basal/housekeeping gene effects
  - Corresponds to `actionet::orthogonalizeBasal()`
  - Takes indicator matrix for basal genes

**Initialization:** `init_decomposition(py::module_ &m)`

---

### 6. `wp_tools.cpp`

**Purpose:** General-purpose tools and utilities.

**Functions:**
- `run_svd_sparse()`: Truncated SVD decomposition
  - Corresponds to `actionet::runSVD()`
  - Returns U, D, and V matrices

**Initialization:** `init_tools(py::module_ &m)`

---

### 7. `wp_visualization.cpp`

**Purpose:** Network layout and embedding generation.

**Functions:**
- `layout_network()`: Generate 2D/3D layouts from networks
  - Corresponds to `actionet::layoutNetwork()`
  - Supports UMAP and t-SNE methods
  - Returns coordinate matrix

**Initialization:** `init_visualization(py::module_ &m)`

---

### 8. `_core_new.cpp`

**Purpose:** Main pybind11 module definition that orchestrates all submodules.

**Structure:**
```cpp
PYBIND11_MODULE(_core, m) {
    m.doc() = "ACTIONet C++ core bindings";

    init_action(m);
    init_network(m);
    init_annotation(m);
    init_decomposition(m);
    init_tools(m);
    init_visualization(m);
}
```

Each `init_*()` function adds its bindings to the main module.

## Build System Integration

### CMakeLists.txt

The module is compiled as a single extension with multiple source files:

```cmake
pybind11_add_module(_core MODULE
    src/actionet/_core_new.cpp
    src/actionet/wp_utils.cpp
    src/actionet/wp_action.cpp
    src/actionet/wp_network.cpp
    src/actionet/wp_annotation.cpp
    src/actionet/wp_decomposition.cpp
    src/actionet/wp_tools.cpp
    src/actionet/wp_visualization.cpp
)
```

All files are compiled together and linked into the `_core` Python extension module.

## Benefits of Modular Structure

1. **Improved Maintainability:** Each module is self-contained and easier to understand
2. **Parallel with R Package:** Developers familiar with the R package can easily navigate
3. **Easier Debugging:** Issues can be isolated to specific modules
4. **Better Organization:** Related functions are grouped together
5. **Simplified Updates:** Changes to one module don't require recompiling unrelated code
6. **Clear Documentation:** Each file documents its corresponding C++ backend functions

## Usage from Python

The modular C++ structure is transparent to Python users. All functions are still imported from the same `_core` module:

```python
import actionet._core as core

# All functions available at module level
result = core.run_action(...)
network = core.build_network(...)
specificity = core.compute_feature_specificity_sparse(...)
```

## Development Workflow

### Adding New Functions

1. Identify the appropriate module based on the C++ backend function
2. Add the pybind11 wrapper to the corresponding `wp_*.cpp` file
3. Add the function binding to the module's `init_*()` function
4. No changes needed to CMakeLists.txt or other files

### Adding New Modules

1. Create new `wp_<module>.cpp` file with bindings
2. Add `void init_<module>(py::module_ &m);` forward declaration to `_core_new.cpp`
3. Call `init_<module>(m);` in `PYBIND11_MODULE()`
4. Add `src/actionet/wp_<module>.cpp` to CMakeLists.txt

## Testing

The modular structure does not affect testing. All existing tests continue to work because the Python API remains unchanged.

## Migration Notes

### Old Structure
- Single file: `_core.cpp` (~434 lines)
- All bindings in one monolithic file
- Conversion utilities inline

### New Structure
- Main file: `_core_new.cpp` (~25 lines)
- 6 module files (~30-140 lines each)
- Shared utilities in separate header/source
- Total: ~800 lines across 9 files

The old `_core.cpp` file is preserved for reference but is no longer used in the build.
