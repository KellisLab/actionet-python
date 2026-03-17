# ACTIONet Scaling Benchmark Suite — Runbook

This document is the runbook for the ACTIONet Scaling Benchmark Suite.
See `ACTIONet Scaling Benchmark Suite.md` for the full specification.

## Files

| File | Role |
|---|---|
| `benchmark_backed_extension.py` | CLI orchestrator |
| `benchmark_support.py` | Helper module (dataset manifests, metrics, reporting) |

## Prerequisites

1. **Environment**: `.venv` at the repo root (`.venv/bin/python`).
2. **Dependencies**: `psutil`, `matplotlib`, `tabulate` must be installed in `.venv`.
3. **Data**: All source datasets in `data/`:
   - `test_adata.h5ad` (small_full, 6,790 cells)
   - `adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad` (sparse_medium, 41,007 cells)
   - `adata_agg_Scn4b_OX_fil.h5ad` (scale_full, 300,157 cells)
4. **Disk**: ≥ 500 GB free on `/data/actionet_benchmark/` for backed copies + subsets.
5. **`LD_LIBRARY_PATH`**: `_core.so` was built against HDF5 1.14.5 (installed at
   `/HDF_Group/HDF5/1.14.5`). Must be on `LD_LIBRARY_PATH` at runtime. See **Build Notes**.

## One-Time Setup

```bash
# Install benchmark dependencies
.venv/bin/pip install psutil matplotlib tabulate

# Set LD_LIBRARY_PATH (add to ~/.bashrc or equivalent for persistence)
export LD_LIBRARY_PATH=/HDF_Group/HDF5/1.14.5/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH

# Convert sparse_medium to CSR format (one-time, detects and skips if already sparse)
cd tests
.venv/bin/python -c "import sys; sys.path.insert(0,'tests'); import benchmark_support as bs; bs.ensure_sparse_medium()"
```

## Build Notes

The C++ backed SVD operator (`_core.so`) must be built against HDF5 ≥ 1.14 and run with
that library on `LD_LIBRARY_PATH`. The root cause of previous failures was a **bug in the
C++ code** in `src/libactionet/src/io/backed_h5ad/backed_sparse_matrix_operator.cpp`:
`H5Aread` was called with a memory type that had `H5T_CSET_ASCII` (the default for a copied
`H5T_C_S1`), but modern h5py writes variable-length string attributes with `H5T_CSET_UTF8`.
HDF5 ≥ 1.14 enforces strict charset matching during type conversion. The fix: call
`H5Tset_cset(mem_type, H5T_CSET_UTF8)` before `H5Aread`. This has been applied.

HDF5 files are fully portable across HDF5 versions. The original files created on different
machines open without issue in Python (h5py) and R (rhdf5/hdf5r) — the problem was solely
in the C++ reading code.

HDF5 1.14.5 was installed from the official [GitHub release `.deb`](https://github.com/HDFGroup/hdf5/releases)
to `/HDF_Group/HDF5/1.14.5/`. To rebuild `_core.so` against it:

```bash
cd /data/git_projects/actionet-python
PYBIND11_DIR=$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')
HDF5_ROOT=/HDF_Group/HDF5/1.14.5 \
CMAKE_ARGS="-DHDF5_ROOT=/HDF_Group/HDF5/1.14.5 -Dpybind11_DIR=${PYBIND11_DIR} -DCMAKE_CXX_FLAGS='-march=native -O3'" \
.venv/bin/pip install . --no-build-isolation --no-cache-dir
```

## Running the Benchmark

### Smoke test (validates harness, ~5 min on small_full)

```bash
cd /home/sebastian/data/git_projects/actionet-python/tests
LD_LIBRARY_PATH=/HDF_Group/HDF5/1.14.5/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH \
/home/sebastian/data/git_projects/actionet-python/.venv/bin/python benchmark_backed_extension.py \
  --suite smoke \
  --output-dir /data/actionet_benchmark/run_smoke \
  --skip-subsets \
  --threads 44
```

### Full benchmark (all suites, all tiers, both profiles)

```bash
cd /home/sebastian/data/git_projects/actionet-python/tests
LD_LIBRARY_PATH=/HDF_Group/HDF5/1.14.5/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH \
/home/sebastian/data/git_projects/actionet-python/.venv/bin/python benchmark_backed_extension.py \
  --suite all \
  --output-dir /data/actionet_benchmark/run_001 \
  --threads 44 \
  --resume
```

### Workflow only (no focused sweeps)

```bash
LD_LIBRARY_PATH=/HDF_Group/HDF5/1.14.5/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH \
/home/sebastian/data/git_projects/actionet-python/.venv/bin/python benchmark_backed_extension.py \
  --suite workflow \
  --datasets small_full sparse_medium scale_subset_25k scale_subset_50k scale_subset_100k \
  --profiles default knn_ceiling \
  --modes backed_decompressed in_memory \
  --output-dir /data/actionet_benchmark/run_workflow \
  --threads 44 \
  --resume
```

### Specific tiers only

```bash
LD_LIBRARY_PATH=/HDF_Group/HDF5/1.14.5/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH \
/home/sebastian/data/git_projects/actionet-python/.venv/bin/python benchmark_backed_extension.py \
  --suite workflow \
  --max-tier 100k \
  --output-dir /data/actionet_benchmark/run_to_100k \
  --threads 44
```

### Regenerate report from existing results

```bash
LD_LIBRARY_PATH=/HDF_Group/HDF5/1.14.5/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH \
/home/sebastian/data/git_projects/actionet-python/.venv/bin/python benchmark_backed_extension.py \
  --output-dir /data/actionet_benchmark/run_001 \
  --report-only
```

## CLI Reference

```
python benchmark_backed_extension.py \
  --suite {all,workflow,network,reduction,batch,thread,ef,kmax,smoke}
  --datasets HANDLE [HANDLE ...]   # dataset handles (see DATASET_MANIFEST)
  --profiles {default,knn_ceiling} [...]
  --modes {in_memory,backed_decompressed,backed_compressed} [...]
  --output-dir PATH                # default: /data/actionet_benchmark/run_<timestamp>
  --resume                         # skip cases with existing status=ok total row
  --max-tier 200k                  # stop after this tier (e.g. 100k, 200k, 300k)
  --trials N                       # override trial count per case
  --threads N                      # thread count (default: 44)
  --skip-subsets                   # skip stratified subset generation
  --report-only                    # regenerate report without running new cases
```

## Output Structure

```
/data/actionet_benchmark/<run_id>/
  raw/         one JSONL file per case (one row per stage)
  summary.csv  aggregated table
  report.md    Markdown summary with tables and plot references
  plots/       PNG scaling plots
  work/        temporary backed h5ad copies (auto-cleaned after each case)
```

## Dataset Handles

| Handle | Description |
|---|---|
| `small_full` | test_adata.h5ad (6,790 cells) |
| `sparse_medium` | sparse_medium h5ad (41,007 cells, 25 batches) |
| `scale_full` | scale_full h5ad (300,157 cells, 26 batches) |
| `scale_subset_25k` | 25,000-cell stratified subset of scale_full |
| `scale_subset_50k` | 50,000-cell stratified subset of scale_full |
| `scale_subset_100k` | 100,000-cell stratified subset |
| `scale_subset_150k` | 150,000-cell stratified subset |
| `scale_subset_200k` | 200,000-cell stratified subset |
| `scale_subset_250k` | 250,000-cell stratified subset |
| `scale_subset_300k` | 300,000-cell (full scale_full) |

Subsets are generated automatically from `scale_full` (stratified by UID, seed=42, min 100 cells/batch)
and written to `/data/actionet_benchmark/scale_subset_<tier>.h5ad`.

## Notes on Backed Mode

- `_core.create_backed_operator` opens the h5ad file with its own HDF5 handle.
- The benchmark closes the h5py (anndata) handle before calling `reduce_kernel`, then
  re-opens it after. `HDF5_USE_FILE_LOCKING=FALSE` is set in the benchmark script as an
  additional safeguard against transient lock contention.
- All 13 stages work correctly in backed-decompressed mode after the C++ UTF-8 string fix
  and the HDF5 1.14.5 rebuild.
- **`LD_LIBRARY_PATH` must be set** (see Build Notes above) or `_core.so` will fail to load.

