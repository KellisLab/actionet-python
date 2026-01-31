# ACTIONet Python

Python bindings for ACTIONet (Action-based Cell-Type Identification and Organism Niche Extraction Tool), a single-cell multi-resolution data analysis toolkit.

This package wraps the C++ backend `libactionet` without modifications, providing a Python interface with AnnData as the core data container, designed to integrate seamlessly with the scanpy ecosystem.

## Features

- **Full C++ backend**: Leverages the high-performance `libactionet` C++ library
- **AnnData integration**: Native support for AnnData objects used throughout the Python single-cell ecosystem
- **Scanpy compatibility**: Works alongside standard scanpy workflows
- **Multi-resolution analysis**: ACTION decomposition for multi-scale archetype discovery
- **Network-based analysis**: Build and analyze cell-cell interaction networks
- **Cross-platform**: Supports macOS (Intel & Apple Silicon) and Linux (manylinux2014)

## Installation

### Prerequisites

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake openblas lapack
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libsuitesparse-dev
```

**Conda (rootless / HPC):**
```bash
conda install -c conda-forge cmake openblas lapack suitesparse
```

libactionet now searches for CHOLMOD/SuiteSparse in conda prefixes (e.g. `CONDA_PREFIX`) in addition to system paths.

### Install from source

```bash
# Clone repository
git clone https://github.com/KellisLab/actionet-python.git
cd actionet-python

# Initialize libactionet submodule
git submodule update --init --recursive

# Install in development mode
pip install -e .

# Or build and install
pip install .
```

## Quick Start

```python
import scanpy as sc
import actionet as an

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Preprocess
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ACTIONet pipeline
an.reduce_kernel(adata, n_components=50)  # Kernel reduction
an.run_action(adata, k_min=2, k_max=30)   # ACTION decomposition
an.build_network(adata)                    # Build cell network
an.layout_network(adata)                   # UMAP layout

# Feature specificity
an.compute_feature_specificity(adata, labels='assigned_archetype')

# Visualize
sc.pl.embedding(adata, basis='X_umap', color='assigned_archetype')
```

## Core Functions

### Dimensionality Reduction

```python
an.reduce_kernel(adata, n_components=50, layer=None, key_added='action')
```
Compute reduced kernel matrix using randomized SVD.

### ACTION Decomposition

```python
an.run_action(adata, k_min=2, k_max=30, reduction_key='action')
```
Multi-resolution archetypal analysis to identify cell states.

### Network Construction

```python
an.build_network(adata, archetype_key='H_stacked', 
                 algorithm='k*nn', distance_metric='jsd')
```
Build cell-cell interaction network from archetype footprints.

### Network Diffusion

```python
an.compute_network_diffusion(adata, scores='H_merged', 
                              network_key='actionet', alpha=0.85)
```
Smooth scores over the network topology.

### Feature Specificity

```python
an.compute_feature_specificity(adata, labels='assigned_archetype')
```
Compute archetype/cluster-specific marker genes.

### Layout Visualization

```python
an.layout_network(adata, network_key='actionet',
                  method='umap', n_components=2)
```
Compute 2D/3D layout using UMAP or t-SNE.

### Batch Correction

```python
an.correct_batch_effect(adata, batch_key='batch',
                        reduction_key='action',
                        corrected_key='action_corrected')
```
Remove batch effects from reduced representation using orthogonalization.

```python
an.correct_basal_expression(adata, basal_genes=['ACTB', 'GAPDH'],
                            reduction_key='action',
                            corrected_key='action_basal_corrected')
```
Correct for basal/housekeeping gene expression.

### Imputation

```python
an.impute_features(adata, features=['GENE1', 'GENE2'],
                   network_key='actionet', alpha=0.85)
```
Impute gene expression using network diffusion.

```python
an.impute_from_archetypes(adata, features=['GENE1', 'GENE2'],
                          H_key='H_merged')
```
Impute expression from archetype profiles.

```python
an.smooth_kernel(adata, reduction_key='action',
                 smoothed_key='action_smoothed', alpha=0.85)
```
Smooth reduced representation using network diffusion.

## AnnData Structure

ACTIONet stores results in standard AnnData slots:

- **adata.obsm**: Dimensionality reductions and archetype matrices
  - `action`: Reduced kernel representation
  - `H_stacked`: Stacked archetype matrix (all scales)
  - `H_merged`: Merged archetype matrix (unified)
  - `X_umap`: UMAP coordinates
  
- **adata.obsp**: Cell-cell networks
  - `actionet`: ACTIONet graph adjacency matrix
  
- **adata.obs**: Cell metadata
  - `assigned_archetype`: Discrete archetype assignments
  
- **adata.varm**: Gene/feature metadata
  - `specificity_upper`: Marker gene scores (upper-tail)
  - `specificity_lower`: Marker gene scores (lower-tail)
  
- **adata.uns**: Parameters and auxiliary data
  - `action_params`: Kernel reduction parameters
  - `action_results`: ACTION decomposition results (C matrices)

## R → Python API Mapping

| R Function | Python Function | Notes |
|------------|----------------|-------|
| `reduce()` | `an.reduce_kernel()` | Kernel reduction |
| `runACTION()` | `an.run_action()` | ACTION decomposition |
| `buildNetwork()` | `an.build_network()` | Network construction |
| `computeNetworkDiffusion()` | `an.compute_network_diffusion()` | Network smoothing |
| `compute_archetype_feature_specificity()` | `an.compute_feature_specificity()` | Marker genes |
| `layoutNetwork()` | `an.layout_network()` | UMAP/t-SNE layout |
| `runSVD()` | `an.run_svd()` | SVD decomposition |
| `orthogonalizeBatchEffect()` | `an.correct_batch_effect()` | Batch correction |
| `orthogonalizeBasal()` | `an.correct_basal_expression()` | Basal correction |
| `imputeFeatures()` | `an.impute_features()` | Network diffusion imputation |
| `imputeFromArchetypes()` | `an.impute_from_archetypes()` | Archetype-based imputation |
| `smoothKernel()` | `an.smooth_kernel()` | Kernel smoothing |
| `colMaps(ace)` | `adata.obsm` | Cell-level embeddings |
| `colNets(ace)` | `adata.obsp` | Cell-level networks |
| `metadata(ace)` | `adata.obs` | Cell annotations |
| `rowMaps(ace)` | `adata.varm` | Gene-level annotations |

### Data Container Translation

| R (ACTIONetExperiment) | Python (AnnData) |
|------------------------|------------------|
| `assays(ace)$counts` | `adata.X` or `adata.layers['counts']` |
| `assays(ace)$logcounts` | `adata.layers['logcounts']` |
| `colMaps(ace)$ACTION` | `adata.obsm['action']` |
| `colMaps(ace)$H_stacked` | `adata.obsm['H_stacked']` |
| `colNets(ace)$ACTIONet` | `adata.obsp['actionet']` |
| `metadata(ace)$assigned_archetype` | `adata.obs['assigned_archetype']` |
| `rowMaps(ace)$specificity` | `adata.varm['specificity_upper']` |

## Examples

See `examples/` directory for complete workflows:

- `01_basic_pipeline.py`: End-to-end ACTIONet analysis
- `02_graph_building.py`: Network construction strategies
- `03_integration_with_scanpy.py`: Integration with scanpy workflows
- `04_batch_correction_imputation.py`: Batch correction and imputation workflows

## Building From Source

### Build Configuration

The build system uses `scikit-build-core` with CMake. Key options:

```bash
# Standard build
pip install .

# Enable architecture-specific optimizations (Linux only)
pip install . -C cmake.define.ACTIONET_ENABLE_OPTIMIZED=ON

# Verbose build output
pip install . -v
```

### Platform-Specific Notes

**macOS:**
- Default deployment target: macOS 11.0
- Builds native architecture (x86_64 or arm64)
- Uses Accelerate framework for BLAS/LAPACK
- Set `CMAKE_OSX_ARCHITECTURES` for cross-compilation

**Linux:**
- Targets manylinux2014 (glibc ≥ 2.17)
- Requires `libcholmod` from SuiteSparse
- OpenMP runtime defaults to `AUTO` (compiler-selected); override with `-C cmake.define.LIBACTIONET_OPENMP_RUNTIME=GNU|INTEL|LLVM|OFF`

### Troubleshooting

**Submodule not initialized:**
```bash
git submodule update --init --recursive
```

**Missing Armadillo:**
Armadillo headers are bundled in `libactionet/include/extern`. If CMake can't find them, check submodule status.

**OpenMP warnings:**
OpenMP is optional. If unavailable, the package builds with single-threaded C++ code (you can still use `n_threads` parameter via Python's multiprocessing).
If using MKL (e.g., conda numpy), avoid mixed OpenMP runtimes by setting `MKL_THREADING_LAYER=GNU` when using GNU OpenMP, or by selecting Intel OpenMP with an Intel toolchain.

Examples:
```bash
# Force GNU OpenMP and align MKL
MKL_THREADING_LAYER=GNU pip install . -C cmake.define.LIBACTIONET_OPENMP_RUNTIME=GNU

# Use Intel OpenMP (Intel/IntelLLVM toolchains)
pip install . -C cmake.define.LIBACTIONET_OPENMP_RUNTIME=INTEL
```

**Linking errors on macOS:**
Ensure Xcode Command Line Tools are installed and up to date.

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Code Formatting

```bash
black src/actionet examples tests
ruff check src/actionet examples tests
```

## Omissions Report

The following R package components are **not implemented** in this Python translation:

### Omitted Components

1. **R-specific visualization helpers** (`plots.R`, `r_visualization.R`, `utils_plotting_*.R`)
   - **Reason**: Python ecosystem has mature alternatives.
   - **Alternative**: Use `scanpy.pl.*` functions for visualization.

2. **R-specific data I/O** (`data.R`)
   - **Reason**: AnnData provides native I/O; scanpy handles format conversion.
   - **Alternative**: `scanpy.read_*()` and `adata.write_h5ad()`.

3. **Parallel backend utilities** (`utils_parallel.R`)
   - **Reason**: Different parallelization approach. C++ threading via OpenMP is preserved.
   - **Alternative**: Use `n_threads` parameter in functions.

4. **Enrichment database utilities** (`enrichment.R`)
   - **Reason**: Python has dedicated packages.
   - **Alternative**: Use `gseapy`, `gprofiler-official`, or `decoupler`.

5. **Projection** (`projection.R`)
   - **Reason**: Requires R-specific reference dataset handling.
   - **Alternative**: Future work or use `scvi-tools` for reference mapping.

6. **Pseudobulk DGE** (`pseudobulk_DGE.R`)
   - **Reason**: Wrapper around R statistics packages.
   - **Alternative**: Use `scanpy.tl.rank_genes_groups()`, `pydeseq2`, or call R via `rpy2`.

7. **Marker detection helpers** (`marker_detection.R`)
   - **Reason**: High-level R wrappers.
   - **Alternative**: `an.compute_feature_specificity()` + post-processing.

8. **Alignment** (`alignment.R`)
   - **Reason**: Multi-dataset alignment utilities specific to R workflows.
   - **Alternative**: Use scanpy integration tools or scvi-tools.

9. **Filter ACE** (`filter_ace.R`)
   - **Reason**: ACTIONetExperiment-specific filtering.
   - **Alternative**: Standard AnnData filtering: `adata[adata.obs['column'] > threshold, :]`.

10. **Autocorrelation statistics** (Moran's I, Geary's C)
    - **Reason**: Low usage; available via squidpy for spatial data.
    - **Alternative**: `squidpy.gr.spatial_autocorr()`.

13. **Maximum-weight matching (MWM)**
    - **Reason**: Utility for batch alignment (omitted feature).
    - **Alternative**: `scipy.optimize.linear_sum_assignment()`.

14. **XICOR correlation**
    - **Reason**: Specialized rank-based correlation; niche use case.
    - **Alternative**: `scipy.stats.spearmanr()` or pandas correlation methods.

### Implemented Core Modules

✅ **Action decomposition**: All archetypal analysis functions  
✅ **Network construction**: Full graph building pipeline  
✅ **Network diffusion**: Smoothing and propagation  
✅ **Feature specificity**: Marker gene identification  
✅ **SVD/Kernel reduction**: Dimensionality reduction  
✅ **Visualization layouts**: UMAP/t-SNE via C++ backend  
✅ **Matrix operations**: Aggregation, normalization, transforms  

## License

GPL-3.0 (same as R package)

## Links

- **C++ Backend**: https://github.com/KellisLab/libactionet
- **R Package**: https://github.com/KellisLab/actionet-r
- **R Data Container**: https://github.com/shmohammadi86/ACTIONetExperiment
- **AnnData**: https://anndata.readthedocs.io
- **scanpy**: https://scanpy.readthedocs.io
