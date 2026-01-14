# Advanced ACTIONet Functions

This document describes the advanced and lower-level functions that have been added to ACTIONet Python, providing finer control over the ACTION decomposition pipeline and network analysis.

## Overview

The `actionet.advanced` module provides access to the underlying algorithms and intermediate steps of the ACTION pipeline. These functions are useful for:

1. Custom workflows requiring manual control over each step
2. Research and algorithm development
3. Fine-tuning parameters at each stage
4. Understanding the internal workings of ACTION

## Functions

### Archetypal Analysis

#### `run_archetypal_analysis(data, W0, max_iter=100, tolerance=1e-6)`

Run archetypal analysis (AA) on a data matrix given initial archetypes.

**Parameters:**
- `data` (ndarray): Input data matrix (features × observations)
- `W0` (ndarray): Initial archetype matrix (features × k)
- `max_iter` (int): Maximum iterations
- `tolerance` (float): Convergence tolerance

**Returns:** dict with keys:
- `C`: Archetype compositions (observations × k)
- `H`: Archetype weights (k × observations)
- `W`: Archetypes in original space

**Example:**
```python
import actionet as an
import numpy as np

# Run SPA to find initial archetypes
data = np.random.randn(50, 100)
spa_result = an.run_spa(data, k=5)
W0 = data[:, spa_result['selected_cols'] - 1]

# Run archetypal analysis
result = an.run_archetypal_analysis(data, W0, max_iter=100)
C = result['C']
H = result['H']
```

---

### ACTION Decomposition Pipeline

#### `decompose_action(S_r, k_min=2, k_max=30, max_iter=100, tolerance=1e-16, n_threads=0)`

Run ACTION decomposition and return full trace of C and H matrices for all k values.

This is the lower-level version of `run_action()` that doesn't perform filtering or merging.

**Parameters:**
- `S_r` (ndarray): Reduced kernel matrix (components × cells)
- `k_min` (int): Minimum number of archetypes
- `k_max` (int): Maximum number of archetypes
- `max_iter` (int): Maximum iterations
- `tolerance` (float): Convergence tolerance
- `n_threads` (int): Number of threads (0=auto)

**Returns:** dict with keys:
- `C`: List of C matrices for k=k_min to k_max
- `H`: List of H matrices for k=k_min to k_max

**Example:**
```python
# Get reduced representation
S_r = adata.obsm['action'].T

# Run decomposition for multiple k
trace = an.decompose_action(S_r, k_min=5, k_max=20)

# Access specific k
C_10 = trace['C'][5]  # k=10 (0-indexed from k_min)
H_10 = trace['H'][5]
```

#### `collect_archetypes(C_trace, H_trace, specificity_threshold=-3.0, min_observations=3)`

Filter and aggregate multi-level archetypes based on specificity and observation thresholds.

**Parameters:**
- `C_trace` (list): List of C matrices from ACTION decomposition
- `H_trace` (list): List of H matrices from ACTION decomposition
- `specificity_threshold` (float): Minimum z-score threshold to filter archetypes
- `min_observations` (int): Minimum observations per archetype

**Returns:** dict with keys:
- `selected_archs`: Indices of selected archetypes (1-indexed)
- `C_stacked`: Horizontally stacked C matrix
- `H_stacked`: Vertically stacked H matrix

**Example:**
```python
# Collect and filter archetypes
collected = an.collect_archetypes(
    trace['C'],
    trace['H'],
    specificity_threshold=-3.0,
    min_observations=5
)

C_stacked = collected['C_stacked']
H_stacked = collected['H_stacked']
```

#### `merge_archetypes(S_r, C_stacked, H_stacked, n_threads=0)`

Identify and merge redundant archetypes into a representative subset.

**Parameters:**
- `S_r` (ndarray): Reduced kernel matrix
- `C_stacked` (ndarray): Stacked C matrix from `collect_archetypes`
- `H_stacked` (ndarray): Stacked H matrix from `collect_archetypes`
- `n_threads` (int): Number of threads (0=auto)

**Returns:** dict with keys:
- `selected_archetypes`: Indices of representative archetypes (1-indexed)
- `C_merged`: Merged C matrix
- `H_merged`: Merged H matrix
- `assigned_archetypes`: Cell assignments to merged archetypes (1-indexed)

**Example:**
```python
# Merge redundant archetypes
merged = an.merge_archetypes(S_r, C_stacked, H_stacked)

# Get final assignments
assignments = merged['assigned_archetypes']
```

---

### Optimization Algorithms

#### `run_simplex_regression(A, B, compute_XtX=False)`

Solve simplex-constrained regression: min ||AX - B|| subject to simplex constraint.

Each column of X sums to 1 and all entries are non-negative.

**Parameters:**
- `A` (ndarray): Input matrix A in AX - B
- `B` (ndarray): Input matrix B in AX - B
- `compute_XtX` (bool): Whether to return X^T X

**Returns:** X (ndarray): Solution matrix

**Example:**
```python
# Solve for X such that AX ≈ B with simplex constraints
X = an.run_simplex_regression(A, B)

# Each column of X sums to 1
print(X.sum(axis=0))  # Should be all ones
```

#### `run_spa(data, k)`

Run successive projections algorithm (SPA) for separable non-negative matrix factorization.

SPA finds columns of the data matrix that can serve as candidate extreme points (archetypes).

**Parameters:**
- `data` (ndarray): Input data matrix
- `k` (int): Number of candidate vertices to find

**Returns:** dict with keys:
- `selected_cols`: Selected column indices (1-indexed)
- `norms`: Column norms

**Example:**
```python
# Find 10 candidate archetypes
result = an.run_spa(data, k=10)
selected_indices = result['selected_cols'] - 1  # Convert to 0-indexed

# Extract candidate archetypes
W0 = data[:, selected_indices]
```

---

### Network Analysis

#### `run_label_propagation(adata, initial_labels, network_key='actionet', lambda_param=1.0, iterations=3, sig_threshold=3.0, fixed_labels=None, n_threads=0, key_added='propagated_labels')`

Run label propagation algorithm to smooth labels over the network.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with network
- `initial_labels` (ndarray): Initial label assignments
- `network_key` (str): Key in adata.obsp containing network
- `lambda_param` (float): Propagation strength (0-1)
- `iterations` (int): Number of propagation iterations
- `sig_threshold` (float): Significance threshold
- `fixed_labels` (ndarray, optional): Indices of labels to keep fixed (1-indexed)
- `n_threads` (int): Number of threads
- `key_added` (str): Key to store propagated labels in adata.obs

**Returns:** Updates adata.obs[key_added] with propagated labels

**Example:**
```python
# Semi-supervised label propagation
# Fix labels for some known cells
initial_labels = np.zeros(adata.n_obs)
initial_labels[:100] = 1  # Type 1
initial_labels[100:200] = 2  # Type 2

# Fix the known labels
fixed = np.arange(1, 201, dtype=np.int32)  # First 200 cells (1-indexed)

an.run_label_propagation(
    adata,
    initial_labels,
    fixed_labels=fixed,
    lambda_param=0.8,
    iterations=5
)
```

#### `compute_coreness(adata, network_key='actionet', key_added='coreness')`

Compute k-shell decomposition (coreness) of graph vertices.

Coreness is a measure of centrality based on iteratively removing nodes with degree < k.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with network
- `network_key` (str): Key in adata.obsp containing network
- `key_added` (str): Key to store coreness in adata.obs

**Returns:** Updates adata.obs[key_added] with coreness values

**Example:**
```python
# Compute coreness
an.compute_coreness(adata)

# Find highly central cells
highly_central = adata.obs['coreness'] > adata.obs['coreness'].quantile(0.9)
```

#### `compute_archetype_centrality(adata, assignments, network_key='actionet', key_added='centrality')`

Compute centrality of vertices within archetype-induced subgraphs.

For each cell, computes its centrality within the subgraph formed by cells assigned to the same archetype.

**Parameters:**
- `adata` (AnnData): Annotated data matrix with network
- `assignments` (ndarray): Archetype assignments for each cell
- `network_key` (str): Key in adata.obsp containing network
- `key_added` (str): Key to store centrality in adata.obs

**Returns:** Updates adata.obs[key_added] with centrality values

**Example:**
```python
# Get archetype assignments
assignments = adata.obs['assigned_archetypes'].values

# Compute within-archetype centrality
an.compute_archetype_centrality(adata, assignments)

# Find central cells within each archetype
central_per_archetype = adata.obs.groupby('assigned_archetypes')['centrality'].nlargest(10)
```

---

## Complete Manual Pipeline Example

Here's an example showing how to manually control the entire ACTION pipeline:

```python
import actionet as an
import numpy as np

# 1. Load and preprocess data
adata = an.read_h5ad('data.h5ad')

# 2. Reduce to kernel space
an.reduce_kernel(adata, n_components=50, key_added='action')
S_r = adata.obsm['action'].T

# 3. Run ACTION decomposition (manual)
trace = an.decompose_action(S_r, k_min=5, k_max=30, max_iter=100)

# 4. Collect and filter archetypes
collected = an.collect_archetypes(
    trace['C'],
    trace['H'],
    specificity_threshold=-3.0,
    min_observations=5
)

# 5. Merge redundant archetypes
merged = an.merge_archetypes(
    S_r,
    collected['C_stacked'],
    collected['H_stacked']
)

# 6. Store results
adata.obsm['H_merged'] = merged['H_merged'].T
adata.obs['assigned_archetypes'] = merged['assigned_archetypes']

# 7. Build network
an.build_network(adata, archetype_key='H_merged')

# 8. Compute network properties
an.compute_coreness(adata)
an.compute_archetype_centrality(
    adata,
    adata.obs['assigned_archetypes'].values
)

# 9. Label propagation for refinement
an.run_label_propagation(
    adata,
    adata.obs['assigned_archetypes'].values.astype(float),
    lambda_param=0.5,
    iterations=3,
    key_added='refined_archetypes'
)
```

## Comparison: High-Level vs. Manual Control

### High-Level API (Recommended for most users)

```python
import actionet as an

an.reduce_kernel(adata, n_components=50)
an.run_action(adata, k_min=5, k_max=30)  # Does collect + merge automatically
an.build_network(adata)
```

### Manual Control (Advanced users)

```python
import actionet as an

an.reduce_kernel(adata, n_components=50)
S_r = adata.obsm['action'].T

# Full control over each step
trace = an.decompose_action(S_r, k_min=5, k_max=30)
collected = an.collect_archetypes(trace['C'], trace['H'],
                                  specificity_threshold=-2.5)
merged = an.merge_archetypes(S_r, collected['C_stacked'],
                             collected['H_stacked'])
```

## When to Use Advanced Functions

**Use high-level API (`run_action`) when:**
- Standard analysis workflow is sufficient
- You want automatic filtering and merging
- Quick exploratory analysis

**Use advanced functions when:**
- Custom filtering criteria needed
- Want to inspect intermediate results
- Need to modify algorithms
- Benchmarking different parameters
- Research and method development

## Function Mapping to R Package

| Python Function | R Function | Module |
|----------------|------------|--------|
| `run_archetypal_analysis()` | `runAA()` | action |
| `decompose_action()` | `decompACTION()` | action |
| `collect_archetypes()` | `collectArchetypes()` | action |
| `merge_archetypes()` | `mergeArchetypes()` | action |
| `run_simplex_regression()` | `runSimplexRegression()` | action |
| `run_spa()` | `runSPA()` | action |
| `run_label_propagation()` | `runLPA()` | network |
| `compute_coreness()` | `computeCoreness()` | network |
| `compute_archetype_centrality()` | `computeArchetypeCentrality()` | network |
