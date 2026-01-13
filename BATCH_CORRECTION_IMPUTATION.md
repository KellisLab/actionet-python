# Batch Correction and Imputation Implementation

This document describes the batch correction and imputation features that have been added to ACTIONet Python.

## Overview

The following functionality has been implemented:

### Batch Correction
- `correct_batch_effect()`: Remove batch effects from reduced representations using orthogonalization
- `correct_basal_expression()`: Remove effects of basal/housekeeping genes

### Imputation
- `impute_features()`: Impute gene expression using network diffusion
- `impute_from_archetypes()`: Impute expression from archetype profiles
- `smooth_kernel()`: Smooth reduced representations using network diffusion

## Implementation Details

### C++ Bindings

The following C++ functions from libactionet have been wrapped in `src/actionet/_core.cpp`:

- `orthogonalize_batch_effect_sparse()`: Batch correction for sparse matrices
- `orthogonalize_batch_effect_dense()`: Batch correction for dense matrices
- `orthogonalize_basal_sparse()`: Basal expression correction for sparse matrices

These bindings handle the conversion between Python NumPy/SciPy arrays and C++ Armadillo matrices.

### Python Modules

#### `src/actionet/batch_correction.py`

Contains high-level functions that wrap the C++ bindings:

**`correct_batch_effect(adata, batch_key, reduction_key='action', layer=None, corrected_key='action_corrected')`**
- Removes batch effects from a reduced representation
- Creates a one-hot encoded design matrix from batch labels
- Calls the appropriate C++ orthogonalization function
- Stores corrected reduction in `adata.obsm[corrected_key]`
- Stores parameters in `adata.uns[f"{corrected_key}_params"]`

**`correct_basal_expression(adata, basal_genes, reduction_key='action', layer=None, corrected_key='action_basal_corrected')`**
- Removes the effect of specified basal/housekeeping genes
- Creates an indicator matrix for basal genes
- Calls the C++ orthogonalization function
- Stores corrected reduction and parameters in AnnData

#### `src/actionet/imputation.py`

Contains imputation functions:

**`impute_features(adata, features, network_key='actionet', alpha=0.85, max_iter=100, layer=None, rescale=True)`**
- Imputes gene expression using network diffusion
- Extracts expression matrix for specified features
- Runs network diffusion via C++ core
- Optionally rescales imputed values to match original expression ranges
- Returns imputed expression matrix (cells × features)

**`impute_from_archetypes(adata, features, H_key='H_merged', layer=None, rescale=True)`**
- Imputes expression from archetype profiles
- Computes archetype-specific expression (C matrix)
- Reconstructs cell expression using H matrix (cell archetype weights)
- Optionally rescales to match original ranges
- Returns imputed expression matrix

**`smooth_kernel(adata, reduction_key='action', smoothed_key='action_smoothed', alpha=0.85, max_iter=100, network_key='actionet')`**
- Smooths a reduced representation using network diffusion
- Applies diffusion to each component independently
- Stores smoothed reduction in `adata.obsm[smoothed_key]`
- Useful for denoising reduced representations

## Usage Examples

### Batch Correction

```python
import actionet as an

# Run reduction
an.reduce_kernel(adata, k=50, reduction_key='action')

# Correct batch effects
an.correct_batch_effect(
    adata,
    batch_key='batch',
    reduction_key='action',
    corrected_key='action_corrected'
)

# Build network on corrected reduction
an.build_network(adata, reduction_key='action_corrected', network_key='actionet_corrected')

# Run ACTION on corrected data
an.run_action(adata, k_min=3, k_max=10, reduction_key='action_corrected')
```

### Basal Expression Correction

```python
# Define housekeeping genes
basal_genes = ['ACTB', 'GAPDH', 'B2M', 'PPIA']

# Correct for basal expression
an.correct_basal_expression(
    adata,
    basal_genes=basal_genes,
    reduction_key='action',
    corrected_key='action_basal_corrected'
)
```

### Feature Imputation

```python
# Build network first
an.build_network(adata, reduction_key='action', network_key='actionet')

# Impute expression for specific genes
genes_to_impute = ['GENE1', 'GENE2', 'GENE3']
X_imputed = an.impute_features(
    adata,
    features=genes_to_impute,
    network_key='actionet',
    alpha=0.85,
    rescale=True
)

# X_imputed is a (n_cells × n_genes) array
```

### Archetype-Based Imputation

```python
# Run ACTION first
an.run_action(adata, k_min=3, k_max=10, reduction_key='action')

# Impute from archetypes
genes_to_impute = ['GENE1', 'GENE2']
X_imputed = an.impute_from_archetypes(
    adata,
    features=genes_to_impute,
    H_key='H_merged',
    rescale=True
)
```

### Kernel Smoothing

```python
# Smooth reduced representation
an.smooth_kernel(
    adata,
    reduction_key='action',
    smoothed_key='action_smoothed',
    alpha=0.85,
    max_iter=10
)

# Use smoothed reduction for downstream analysis
an.build_network(adata, reduction_key='action_smoothed', network_key='actionet_smoothed')
```

## Testing

Comprehensive tests have been created:

### `tests/test_batch_correction.py`
- Test batch correction with 2 and 3 batches
- Test basal expression correction
- Test error handling for missing keys and genes
- Test that corrections produce different results from originals

### `tests/test_imputation.py`
- Test network diffusion-based imputation
- Test archetype-based imputation
- Test kernel smoothing
- Test with and without rescaling
- Test layer support
- Test error handling for missing networks and invalid genes

### `tests/test_integration.py`
- Test complete pipelines combining multiple features
- Test batch correction in full workflow
- Test imputation in full workflow
- Test multiple reductions and networks coexist properly

### `tests/test_import.py`
- Smoke tests for package import
- Verify all new functions are available in namespace
- Verify C++ core bindings are accessible

## AnnData Storage Conventions

### Batch Correction Results

**After `correct_batch_effect()`:**
- `adata.obsm[corrected_key]`: Corrected reduction matrix (cells × components)
- `adata.uns[f"{corrected_key}_params"]`: Dictionary containing:
  - `V`: Left singular vectors
  - `sigma`: Singular values
  - `A`, `B`: Perturbation matrices
  - `batch_key`: Name of batch column used
  - `original_reduction`: Original reduction key

**After `correct_basal_expression()`:**
- `adata.obsm[corrected_key]`: Corrected reduction matrix
- `adata.uns[f"{corrected_key}_params"]`: Dictionary containing:
  - `V`, `sigma`, `A`, `B`: SVD components
  - `basal_genes`: List of basal genes that were found in data
  - `original_reduction`: Original reduction key

### Imputation Results

Imputation functions return NumPy arrays and do not modify the AnnData object:
- `impute_features()` returns: `(n_cells × n_features)` array
- `impute_from_archetypes()` returns: `(n_cells × n_features)` array

Users can store these results as needed:
```python
X_imputed = an.impute_features(adata, features=['GENE1', 'GENE2'])
adata.layers['imputed_subset'] = X_imputed
```

### Kernel Smoothing Results

**After `smooth_kernel()`:**
- `adata.obsm[smoothed_key]`: Smoothed reduction matrix (cells × components)

## Differences from R Implementation

1. **Design Matrix Construction**: Python implementation creates one-hot encoded design matrices from categorical batch labels, similar to R but using pandas Categorical dtype.

2. **Rescaling**: Python imputation functions include an optional `rescale` parameter to preserve expression ranges, which helps maintain biological interpretability.

3. **Error Handling**: Python implementation includes comprehensive input validation with clear error messages.

4. **Return Values**: Imputation functions return NumPy arrays rather than modifying the input object, giving users more control over storage.

5. **Layer Support**: All functions support reading from specific layers in the AnnData object.

## Performance Considerations

- **Sparse vs Dense**: Batch correction automatically detects sparse/dense input and calls the appropriate C++ function
- **Network Diffusion**: Imputation via network diffusion has complexity O(n_cells² × max_iter) due to sparse matrix multiplication
- **Archetype Imputation**: Archetype-based imputation is faster as it only requires matrix multiplication with the archetype matrices

## Examples

A complete example demonstrating all features is available in:
- `examples/04_batch_correction_imputation.py`

This example shows:
1. Creating synthetic data with batch effects
2. Running ACTION reduction
3. Correcting batch effects
4. Building networks on corrected data
5. Imputing gene expression
6. Correcting for basal expression
7. Smoothing reduced representations
8. Running ACTION on corrected data
9. Imputing from archetypes

## R → Python Function Mapping

| R Function | Python Function | Module |
|------------|----------------|--------|
| `orthogonalizeBatchEffect()` | `an.correct_batch_effect()` | `batch_correction` |
| `orthogonalizeBasal()` | `an.correct_basal_expression()` | `batch_correction` |
| `imputeFeatures()` | `an.impute_features()` | `imputation` |
| `imputeFromArchetypes()` | `an.impute_from_archetypes()` | `imputation` |
| `smoothKernel()` | `an.smooth_kernel()` | `imputation` |
