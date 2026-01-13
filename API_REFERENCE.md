# ACTIONet Python API Reference

## Core Functions

### Dimensionality Reduction

#### `reduce_kernel()`
```python
actionet.reduce_kernel(
    adata,
    k=50,
    layer=None,
    reduction_key='action',
    svd_algorithm=0,
    max_iter=0,
    seed=0,
    verbose=True
)
```
Compute reduced kernel matrix using randomized SVD.

**Parameters:**
- `adata` (AnnData): Annotated data matrix
- `k` (int): Number of singular vectors to compute
- `layer` (str, optional): Layer to use (None uses .X)
- `reduction_key` (str): Key to store reduction in adata.obsm
- `svd_algorithm` (int): SVD algorithm choice (0=auto)
- `max_iter` (int): Maximum iterations for iterative methods
- `seed` (int): Random seed
- `verbose` (bool): Print status messages

**Stores:**
- `adata.obsm[reduction_key]`: Reduced kernel matrix (cells × k)
- `adata.uns[f"{reduction_key}_params"]`: SVD parameters (V, sigma, A, B)

---

### ACTION Decomposition

#### `run_action()`
```python
actionet.run_action(
    adata,
    k_min=2,
    k_max=30,
    reduction_key='action',
    max_iter=100,
    tol=1e-16,
    spec_threshold=-3,
    min_observations=3,
    n_threads=0
)
```
Multi-resolution archetypal analysis.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with reduction
- `k_min` (int): Minimum number of archetypes
- `k_max` (int): Maximum number of archetypes
- `reduction_key` (str): Key in adata.obsm for reduction
- `max_iter` (int): Maximum AA iterations
- `tol` (float): Convergence tolerance
- `spec_threshold` (float): Specificity threshold (z-score)
- `min_observations` (int): Minimum cells per archetype
- `n_threads` (int): Number of threads (0=auto)

**Stores:**
- `adata.obsm['H_stacked']`: All archetype weights (cells × n_archetypes)
- `adata.obsm['H_merged']`: Merged archetype weights
- `adata.varm['C_stacked']`: All archetype profiles (genes × n_archetypes)
- `adata.varm['C_merged']`: Merged archetype profiles
- `adata.obs['assigned_archetypes']`: Discrete assignments

---

### Network Construction

#### `build_network()`
```python
actionet.build_network(
    adata,
    reduction_key='action',
    network_key='actionet',
    k=30,
    alpha=0.0,
    symmetric=True,
    normalize=True
)
```
Build cell-cell similarity network.

**Parameters:**
- `adata` (AnnData): Annotated data matrix
- `reduction_key` (str): Key in adata.obsm for reduction
- `network_key` (str): Key to store network in adata.obsp
- `k` (int): Number of nearest neighbors
- `alpha` (float): Jaccard weighting parameter
- `symmetric` (bool): Symmetrize network
- `normalize` (bool): Normalize edge weights

**Stores:**
- `adata.obsp[network_key]`: Sparse adjacency matrix (cells × cells)

---

### Network Diffusion

#### `compute_network_diffusion()`
```python
actionet.compute_network_diffusion(
    adata,
    scores_key='H_merged',
    network_key='actionet',
    alpha=0.85,
    max_iter=100,
    tol=1e-6,
    diffused_key='H_diffused'
)
```
Smooth scores over network topology.

**Parameters:**
- `adata` (AnnData): Annotated data matrix
- `scores_key` (str): Key in adata.obsm for scores to diffuse
- `network_key` (str): Key in adata.obsp for network
- `alpha` (float): Diffusion coefficient (0-1)
- `max_iter` (int): Maximum iterations
- `tol` (float): Convergence tolerance
- `diffused_key` (str): Key to store diffused scores

**Stores:**
- `adata.obsm[diffused_key]`: Diffused scores matrix

---

### Feature Specificity

#### `compute_feature_specificity()`
```python
actionet.compute_feature_specificity(
    adata,
    H_key='H_merged',
    layer=None,
    specificity_key='feature_specificity'
)
```
Compute archetype/cluster-specific marker genes.

**Parameters:**
- `adata` (AnnData): Annotated data matrix
- `H_key` (str): Key in adata.obsm for archetype weights
- `layer` (str, optional): Layer to use for expression
- `specificity_key` (str): Key to store specificity scores

**Stores:**
- `adata.varm[specificity_key]`: Specificity scores (genes × archetypes)

---

### Visualization

#### `layout_network()`
```python
actionet.layout_network(
    adata,
    network_key='actionet',
    method='umap',
    n_components=2,
    layout_key='X_umap',
    seed=0
)
```
Compute 2D/3D layout using UMAP or t-SNE.

**Parameters:**
- `adata` (AnnData): Annotated data matrix
- `network_key` (str): Key in adata.obsp for network
- `method` (str): Layout method ('umap' or 'tsne')
- `n_components` (int): Number of dimensions (2 or 3)
- `layout_key` (str): Key to store layout coordinates
- `seed` (int): Random seed

**Stores:**
- `adata.obsm[layout_key]`: Layout coordinates (cells × n_components)

---

## Batch Correction

### `correct_batch_effect()`
```python
actionet.correct_batch_effect(
    adata,
    batch_key,
    reduction_key='action',
    layer=None,
    corrected_key='action_corrected'
)
```
Remove batch effects from reduced representation using orthogonalization.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with reduction
- `batch_key` (str): Key in adata.obs containing batch labels
- `reduction_key` (str): Key in adata.obsm for reduction to correct
- `layer` (str, optional): Layer to use for correction
- `corrected_key` (str): Key to store corrected reduction

**Stores:**
- `adata.obsm[corrected_key]`: Batch-corrected reduction (cells × k)
- `adata.uns[f"{corrected_key}_params"]`: Correction parameters

**Example:**
```python
import actionet as an

# Add batch labels
adata.obs['batch'] = ['batch1'] * 100 + ['batch2'] * 100

# Run reduction
an.reduce_kernel(adata, k=50)

# Correct batch effects
an.correct_batch_effect(adata, batch_key='batch')

# Continue with corrected data
an.build_network(adata, reduction_key='action_corrected')
```

---

### `correct_basal_expression()`
```python
actionet.correct_basal_expression(
    adata,
    basal_genes,
    reduction_key='action',
    layer=None,
    corrected_key='action_basal_corrected'
)
```
Correct for basal/housekeeping gene expression.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with reduction
- `basal_genes` (list): List of basal/housekeeping gene names
- `reduction_key` (str): Key in adata.obsm for reduction to correct
- `layer` (str, optional): Layer to use for correction
- `corrected_key` (str): Key to store corrected reduction

**Stores:**
- `adata.obsm[corrected_key]`: Basal-corrected reduction
- `adata.uns[f"{corrected_key}_params"]`: Correction parameters including basal_genes list

**Example:**
```python
# Define housekeeping genes
basal_genes = ['ACTB', 'GAPDH', 'B2M', 'PPIA']

# Correct for basal expression
an.correct_basal_expression(adata, basal_genes=basal_genes)
```

---

## Imputation

### `impute_features()`
```python
actionet.impute_features(
    adata,
    features,
    network_key='actionet',
    alpha=0.85,
    max_iter=100,
    tol=1e-6,
    layer=None,
    rescale=True
)
```
Impute gene expression using network diffusion.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with network
- `features` (list): List of gene names to impute
- `network_key` (str): Key in adata.obsp for network
- `alpha` (float): Diffusion coefficient (0-1)
- `max_iter` (int): Maximum iterations
- `tol` (float): Convergence tolerance
- `layer` (str, optional): Layer to use for expression
- `rescale` (bool): Rescale imputed values to match original range

**Returns:**
- `ndarray`: Imputed expression matrix (cells × features)

**Example:**
```python
# Build network first
an.build_network(adata)

# Impute specific genes
genes = ['GENE1', 'GENE2', 'GENE3']
X_imputed = an.impute_features(adata, features=genes)

# Store in layer
adata.layers['imputed'] = X_imputed
```

---

### `impute_from_archetypes()`
```python
actionet.impute_from_archetypes(
    adata,
    features,
    H_key='H_merged',
    layer=None,
    rescale=True
)
```
Impute expression from archetype profiles.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with ACTION results
- `features` (list): List of gene names to impute
- `H_key` (str): Key in adata.obsm for archetype weights
- `layer` (str, optional): Layer to use for expression
- `rescale` (bool): Rescale imputed values to match original range

**Returns:**
- `ndarray`: Imputed expression matrix (cells × features)

**Example:**
```python
# Run ACTION first
an.run_action(adata, k_min=3, k_max=10)

# Impute from archetypes
genes = ['GENE1', 'GENE2']
X_imputed = an.impute_from_archetypes(adata, features=genes)
```

---

### `smooth_kernel()`
```python
actionet.smooth_kernel(
    adata,
    reduction_key='action',
    smoothed_key='action_smoothed',
    alpha=0.85,
    max_iter=100,
    tol=1e-6,
    network_key='actionet'
)
```
Smooth reduced representation using network diffusion.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with reduction and network
- `reduction_key` (str): Key in adata.obsm for reduction to smooth
- `smoothed_key` (str): Key to store smoothed reduction
- `alpha` (float): Diffusion coefficient (0-1)
- `max_iter` (int): Maximum iterations
- `tol` (float): Convergence tolerance
- `network_key` (str): Key in adata.obsp for network

**Stores:**
- `adata.obsm[smoothed_key]`: Smoothed reduction matrix

**Example:**
```python
# Smooth the reduction
an.smooth_kernel(adata, alpha=0.85)

# Use smoothed reduction
an.build_network(adata, reduction_key='action_smoothed')
```

---

## Utilities

### `run_svd()`
```python
actionet.run_svd(
    adata,
    k=50,
    layer=None,
    svd_key='X_svd',
    algorithm=0,
    max_iter=0,
    seed=0
)
```
Run truncated SVD decomposition.

**Parameters:**
- `adata` (AnnData): Annotated data matrix
- `k` (int): Number of singular vectors
- `layer` (str, optional): Layer to use
- `svd_key` (str): Key to store results
- `algorithm` (int): Algorithm choice (0=auto)
- `max_iter` (int): Maximum iterations
- `seed` (int): Random seed

**Stores:**
- `adata.obsm[svd_key]`: Right singular vectors (cells × k)
- `adata.uns[f"{svd_key}_params"]`: SVD parameters

---

## AnnData Conventions

### Storage Locations

| Slot | Content | Example Keys |
|------|---------|-------------|
| `adata.X` | Raw or normalized counts | Main expression matrix |
| `adata.layers` | Alternative expression matrices | 'counts', 'logcounts', 'imputed' |
| `adata.obs` | Cell metadata | 'assigned_archetypes', 'batch' |
| `adata.var` | Gene metadata | Gene names, biotypes |
| `adata.obsm` | Cell embeddings/matrices | 'action', 'H_merged', 'X_umap' |
| `adata.varm` | Gene embeddings/matrices | 'C_merged', 'feature_specificity' |
| `adata.obsp` | Cell-cell networks | 'actionet' |
| `adata.uns` | Unstructured metadata | 'action_params', 'action_results' |

### Standard Key Names

**Reductions:**
- `'action'`: Reduced kernel representation
- `'action_corrected'`: Batch-corrected reduction
- `'action_basal_corrected'`: Basal-corrected reduction
- `'action_smoothed'`: Smoothed reduction

**Archetype Matrices:**
- `'H_stacked'`: All multi-resolution archetypes
- `'H_merged'`: Unified archetypes (cells × archetypes)
- `'C_stacked'`: All archetype gene profiles
- `'C_merged'`: Unified archetype profiles (genes × archetypes)

**Networks:**
- `'actionet'`: Cell-cell similarity network

**Layouts:**
- `'X_umap'`: UMAP coordinates
- `'X_tsne'`: t-SNE coordinates

**Feature Scores:**
- `'feature_specificity'`: Archetype-specific marker scores

**Cell Metadata:**
- `'assigned_archetypes'`: Discrete archetype assignments
