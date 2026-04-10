# ACTIONet Python

Python bindings for ACTIONet (Action-based Cell-Type Identification and Organism Niche Extraction Tool), a single-cell multi-resolution data analysis toolkit.

This package wraps the C++ backend `libactionet` without modifications, providing a Python interface with AnnData as the core data container, designed to integrate seamlessly with the scanpy ecosystem.

## Features

- **Full C++ backend**: Leverages the high-performance `libactionet` C++ library
- **OpenMP parallelism**: Multi-threaded C++ execution via OpenMP for all compute-intensive operations
- **AnnData integration**: Native support for AnnData objects used throughout the Python single-cell ecosystem
- **Scanpy compatibility**: Works alongside standard scanpy workflows
- **Multi-resolution analysis**: ACTION decomposition for multi-scale archetype discovery
- **Network-based analysis**: Build and analyze cell-cell interaction networks
- **Out-of-core computation**: Backed (HDF5-streamed) operator paths for datasets larger than RAM
- **Cross-platform**: Supports macOS (Intel & Apple Silicon) and Linux

## Installation

### Prerequisites

A C++17 compiler, CMake (≥ 3.19), BLAS/LAPACK, HDF5 (C library), and **OpenMP** are required. OpenMP is a hard build requirement — the build will fail without it.

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake openblas lapack libomp hdf5
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libhdf5-dev libgomp1
```

**Conda (rootless / HPC):**
```bash
conda install -c conda-forge cmake compilers openblas lapack hdf5
```

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

### Building with Intel MKL (Recommended for Best Performance)

Intel MKL provides highly optimized BLAS/LAPACK implementations and can significantly improve performance, especially for large matrix operations. ACTIONet will automatically detect and use MKL if available.

#### Option 1: Using Conda (Easiest)

```bash
# Create a new environment with MKL
conda create -n actionet-mkl python=3.12
conda activate actionet-mkl

# Install Intel MKL and build dependencies
conda install -c conda-forge cmake compilers numpy scipy mkl mkl-include

# Clone and build ACTIONet
git clone https://github.com/KellisLab/actionet-python.git
cd actionet-python
git submodule update --init --recursive

# Build with MKL (automatically detected)
pip install -e .
```

#### Option 2: Using Intel oneAPI (Most Optimized)

For maximum performance, use Intel's oneAPI toolkit with the Intel C++ compiler:

```bash
# Download and install Intel oneAPI Base Toolkit
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

# Source the Intel environment
source /opt/intel/oneapi/setvars.sh

# Set environment variables for Intel MKL and compiler
export CC=icx
export CXX=icpx
export MKLROOT=/opt/intel/oneapi/mkl/latest

# Clone and build
git clone https://github.com/KellisLab/actionet-python.git
cd actionet-python
git submodule update --init --recursive

# Build with Intel compiler and MKL
pip install -e .
```

#### Option 3: System MKL (Linux)

```bash
# Install Intel MKL from package manager (Ubuntu/Debian)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-mkl intel-oneapi-mkl-devel

# Source MKL environment
source /opt/intel/oneapi/mkl/latest/env/vars.sh

# Build ACTIONet
cd actionet-python
pip install -e .
```

#### Verifying MKL Usage

After installation, verify that MKL is being used:

```python
import numpy as np
print(np.__config__.show())  # Should show MKL in BLAS/LAPACK info
```

#### Performance Tuning with MKL

For optimal performance, set these environment variables:

```bash
# Use all available cores
export MKL_NUM_THREADS=$(nproc)

# For GNU OpenMP runtime (conda default)
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=$(nproc)

# For Intel OpenMP runtime (Intel compiler)
export MKL_THREADING_LAYER=INTEL
export OMP_NUM_THREADS=$(nproc)

# Disable MKL's internal threading (if using external parallelization)
export MKL_NUM_THREADS=1
```

#### Expected Performance Improvements

With Intel MKL, you should see:
- **2-4x faster** matrix operations (SVD, matrix multiplication)
- **Especially beneficial** for large datasets (>100K cells)
- **Better multi-threading** efficiency
- **Lower memory usage** for some operations

#### Troubleshooting MKL Builds

**MKL not detected:**
- Ensure `MKLROOT` environment variable is set
- Verify `mkl-include` is installed (conda) or headers are in `/opt/intel/oneapi/mkl/latest/include`

**Mixed OpenMP runtime warnings:**
- Set `MKL_THREADING_LAYER=GNU` for conda environments
- Set `MKL_THREADING_LAYER=INTEL` for Intel oneAPI builds
- OpenMP is required and cannot be disabled; ensure a compatible runtime is installed

**Link errors with Intel compiler:**
- Ensure `setvars.sh` is sourced before building
- Try: `export LDFLAGS="-L${MKLROOT}/lib/intel64"`

## Quick Start

```python
import actionet as an
import anndata as ad

# Load data
adata = ad.read_h5ad("your_data.h5ad")

# Preprocess (if not already done)
an.normalize_anndata(adata)

# ACTIONet pipeline
an.reduce_kernel(adata, n_components=50)  # Kernel reduction
an.run_action(adata, k_min=2, k_max=30)   # ACTION decomposition
an.build_network(adata)                    # Build cell network
an.layout_network(adata)                   # UMAP layout

# Feature specificity
an.compute_feature_specificity(adata, labels='assigned_archetype')

# Or run the full pipeline in one call
an.run_actionet(adata, k_min=2, k_max=30)
```

## QC Plotting

ACTIONet includes a lets-plot based QC violin helper for per-cell metrics stored in ``adata.obs``.

```python
import actionet as an

an.plot_qc_violin(
    adata,
    keys=["n_counts", "n_genes"],
    groupby="CellLabel",
    log_trans="log10",
    title="QC metrics",
)
```

## Core Functions

### Preprocessing

```python
an.normalize_anndata(adata, target_sum=1e4, pseudocount=0.5, log_base=None)
```
Normalize and log-transform count data. Stores normalized counts in `adata.layers['logcounts']`.

```python
an.filter_anndata(adata, min_genes=200, min_cells=3)
```
Filter cells and genes by minimum thresholds.

### Dimensionality Reduction

```python
an.reduce_kernel(adata, n_components=50, layer=None, key_added='action')
```
Compute reduced kernel matrix using SVD. **Automatically selects the optimal SVD algorithm** based on matrix properties (sparse vs dense, size, sparsity) with negligible overhead (~1-2 microseconds).

New in OOM v1:
- Backed sparse AnnData `.X` is executed through an out-of-memory operator path (Halko by default; IRLB, PRIMME, and Feng also supported).
- You can reuse an external SVD via `precomputed_svd`:
  `an.reduce_kernel(adata, precomputed_svd=an.run_svd(adata.X, n_components=50))`
- Explicit helper for this workflow: `an.reduce_kernel_from_svd(...)`.
- Backed SVD/reduction paths expose `backed_n_threads`:
  `an.run_svd(adata_backed, n_components=50, backed_n_threads=8)`
  `an.reduce_kernel(adata_backed, n_components=50, backed_n_threads=8)`
  (`0` = auto, `1` = serial debug path).

Available algorithms:
- **IRLB** (default for in-memory sparse): Implicitly Restarted Lanczos Bidiagonalization
- **Halko** (default for dense and backed): Randomized SVD (fastest for dense; predictable I/O cost for backed)
- **PRIMME** (auto-selected for large in-memory sparse): Memory-efficient for huge sparse matrices
- **Feng**: Alternative randomized method

### ACTION Decomposition

```python
an.run_action(adata, k_min=2, k_max=30, reduction_key='action')
```
Multi-resolution archetypal analysis to identify cell states.

### Network Construction

```python
an.build_network(adata, obsm_key='H_stacked', 
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
an.impute_features_from_archetypes(adata, features=['GENE1', 'GENE2'],
                          H_key='H_merged')
```
Impute expression from archetype profiles.

```python
an.smooth_kernel(adata, reduction_key='action',
                 smoothed_key='action_smoothed', alpha=0.85)
```
Smooth reduced representation using network diffusion.

### Annotation

```python
an.find_markers(adata, labels='assigned_archetype', layer='logcounts')
```
Identify marker genes for each group.

```python
an.annotate_cells(adata, marker_dict={'CellType': ['GENE1', 'GENE2']})
```
Annotate cells using marker gene dictionaries.

### Clustering

```python
an.cluster_network(adata, network_key='actionet', resolution=1.0)
```
Leiden clustering on the ACTIONet graph.

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
| `compute_archetype_feature_specificity()` | `an.compute_archetype_feature_specificity()` | Per-archetype specificity |
| `layoutNetwork()` | `an.layout_network()` | UMAP/t-SNE layout |
| `runSVD()` | `an.run_svd()` | SVD decomposition |
| `orthogonalizeBatchEffect()` | `an.correct_batch_effect()` | Batch correction |
| `orthogonalizeBasal()` | `an.correct_basal_expression()` | Basal correction |
| `imputeFeatures()` | `an.impute_features()` | Network diffusion imputation |
| `imputeFromArchetypes()` | `an.impute_features_from_archetypes()` | Archetype-based imputation |
| `smoothKernel()` | `an.smooth_kernel()` | Kernel smoothing |
| `findMarkers()` | `an.find_markers()` | Marker detection |
| `annotateCells()` | `an.annotate_cells()` | Cell annotation |
| `annotateClusters()` | `an.annotate_clusters()` | Cluster annotation |
| `clusterNetwork()` | `an.cluster_network()` | Leiden clustering |
| `run_SPA()` | `an.run_spa()` | Sparse archetypal analysis |
| `run_simplex_regression()` | `an.run_simplex_regression()` | Simplex regression |
| `run_label_propagation()` | `an.run_label_propagation()` | Label propagation |
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
- OpenMP via Homebrew `libomp` (required: `brew install libomp`)
- Set `CMAKE_OSX_ARCHITECTURES` for cross-compilation

**Linux:**
- Targets glibc ≥ 2.17
- OpenMP via `libgomp` (GCC) by default; override with `-C cmake.define.LIBACTIONET_OPENMP_RUNTIME=GNU|INTEL|LLVM`
- **For best performance**, consider building with Intel MKL (see installation section above)

### Troubleshooting

**Submodule not initialized:**
```bash
git submodule update --init --recursive
```

**Missing Armadillo:**
Armadillo headers are bundled in `libactionet/include/extern`. If CMake can't find them, check submodule status.

**OpenMP errors:**
OpenMP is a hard requirement. The build will fail if no OpenMP runtime is found.
- **Linux:** Install `libgomp` (`apt install libgomp1` or `yum install libgomp`).
- **macOS:** Install `libomp` via Homebrew (`brew install libomp`).
- **Conda:** Install compilers (`conda install -c conda-forge compilers`).

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
✅ **SVD/Kernel reduction**: Dimensionality reduction (in-memory and backed/out-of-core)  
✅ **Visualization layouts**: UMAP/t-SNE via C++ backend  
✅ **Matrix operations**: Aggregation, normalization, transforms  
✅ **Batch correction**: Orthogonalization-based batch effect and basal expression correction  
✅ **Imputation**: Network diffusion and archetype-based gene expression imputation  
✅ **Clustering**: Leiden clustering via igraph  
✅ **Annotation**: Marker detection and cell/cluster annotation  
✅ **Preprocessing**: Filtering, normalization, backed decompression  
✅ **Plotting**: UMAP, feature expression overlays, QC violin plots (lets-plot, matplotlib, Plotly)  
✅ **Pipeline**: End-to-end `run_actionet()` orchestration

## License

GPL-3.0 (same as R package)

## Links

- **C++ Backend**: https://github.com/KellisLab/libactionet
- **R Package**: https://github.com/KellisLab/actionet-r
- **R Data Container**: https://github.com/shmohammadi86/ACTIONetExperiment
- **AnnData**: https://anndata.readthedocs.io
- **scanpy**: https://scanpy.readthedocs.io

## Plotting (preview)

UMAP plots use `lets-plot` for static rendering and Plotly WebGL for interactive views. Install plotting extras first.

```bash
pip install -e ".[plotting]"
```

```python
import actionet as act
from lets_plot import LetsPlot

# Prefer static notebook rendering for lets-plot UMAPs.
LetsPlot.setup_html(no_js=True)

# adata.obsm["X_umap"] must exist
p_static = act.plot_umap(adata, color="cluster")
p_png = p_static.to_png()
fig_raster = act.plot_umap_raster(adata, color="cluster")
fig = act.plot_umap_interactive(adata, color="gene1")

# Feature expression overlays
plots = act.plot_feature_expression(
    adata,
    features=["GeneA", "GeneB"],
    features_use=None,
    alpha=0,
    layer="logcounts",
)

plots_raster = act.plot_feature_expression_raster(
    adata,
    features=["GeneA", "GeneB"],
    alpha=0,
    layer="logcounts",
)
```

`plot_umap()` now disables lets-plot sampling and tooltips by default for fidelity-first static rendering. For very large notebook outputs, prefer `PlotSpec.to_png()` / `to_pdf()` first; move to `plot_umap_raster()` when full lets-plot specs become too heavy.
