# SVD Algorithm Benchmark Suite

This directory contains a comprehensive benchmark test for comparing the 4 SVD algorithms implemented in ACTIONet.

## Benchmark Script

**File:** `benchmark_svd_algorithms.py`

### Features

The benchmark evaluates all 4 SVD methods (IRLB, Halko, Feng, PRIMME) with:

1. **Both Input Types:**
   - Sparse matrices (CSR format)
   - Dense matrices (NumPy arrays)

2. **Performance Metrics:**
   - Execution time (mean, std, min, max)
   - Peak memory usage (requires `psutil`)
   - Multiple runs for statistical reliability

3. **Comprehensive Visualization:**
   - Execution time comparison (bar plot with error bars)
   - Memory usage comparison
   - Speedup factors (sparse vs dense)
   - Timing distribution (box plots)

4. **Detailed Output:**
   - Console summary table
   - CSV export for further analysis
   - PNG visualization

### Requirements

**Required:**
- numpy
- scipy
- matplotlib
- pandas
- anndata
- scanpy
- actionet

**Optional (for memory tracking):**
- psutil (recommended, install with `pip install psutil`)
- memory_profiler (for detailed profiling, install with `pip install memory-profiler`)

### Usage

#### Basic Usage
```bash
python benchmark_svd_algorithms.py
```

This runs with default settings:
- 30 components
- 3 runs per test
- Standard memory tracking (if psutil is available)

#### Custom Parameters
```bash
# Specify number of components
python benchmark_svd_algorithms.py --components 50

# Specify number of runs per test
python benchmark_svd_algorithms.py --runs 5

# Enable detailed memory profiling (requires memory_profiler)
python benchmark_svd_algorithms.py --detailed-memory
```

#### Combined Options
```bash
python benchmark_svd_algorithms.py --components 30 --runs 3
```

### Output Files

The benchmark generates three output files in the `tests/` directory:

1. **benchmark_svd_results.csv** - Detailed results in CSV format
2. **benchmark_svd_results.png** - Comprehensive visualization with 4 subplots
3. **benchmark_output.txt** - Full console output (if redirected)

### Interpreting Results

#### Execution Time
- Lower is better
- Compare mean times across methods
- Error bars show standard deviation across runs

#### Memory Usage
- Lower is better
- Shows peak memory increase during computation
- Note: First run may show higher memory due to initial allocations

#### Speedup Factor
- Values > 1.0 mean dense is slower (sparse is faster)
- Values < 1.0 mean sparse is slower (dense is faster)
- Green bars indicate sparse advantage, red bars indicate dense advantage

#### Best Practices
- Run with `--runs 5` or more for production benchmarks
- Close other applications to reduce memory noise
- Use consistent hardware/environment for comparisons
- Check that data characteristics (size, sparsity) match your use case

### Example Output

```
======================================================================
BENCHMARK SUMMARY
======================================================================

Method Input Type  Mean Time (s)  Std Time (s)  Min Time (s)  Max Time (s)  Memory (MB)
  IRLB     sparse       2.365         0.008         2.357         2.372        1304.7
  IRLB      dense       1.354         0.072         1.282         1.426          19.5
 Halko     sparse       4.560         0.063         4.496         4.623          44.3
 Halko      dense       0.896         0.001         0.896         0.897          19.6
  Feng     sparse       5.091         0.003         5.087         5.094          10.5
  Feng      dense       0.915         0.000         0.915         0.915          21.9
PRIMME     sparse       3.994         0.027         3.967         4.021           3.1
PRIMME      dense       2.496         0.014         2.481         2.510           0.0

----------------------------------------------------------------------
FASTEST METHODS:
----------------------------------------------------------------------
Sparse input:  IRLB       - 2.365 s
Dense input:   Halko      - 0.896 s

----------------------------------------------------------------------
MEMORY EFFICIENCY:
----------------------------------------------------------------------
Sparse input:  PRIMME     - 3.1 MB
Dense input:   PRIMME     - 0.0 MB
```

### Key Findings (Example Dataset)

Based on test data (6790 cells × 14445 genes, 81% sparse):

1. **For Sparse Input:**
   - Fastest: IRLB (~2.4s)
   - Most memory efficient: PRIMME (~3MB)
   
2. **For Dense Input:**
   - Fastest: Halko (~0.9s)
   - Most memory efficient: PRIMME (~0MB)

3. **Sparse vs Dense Performance:**
   - Dense input is generally faster for all methods
   - Memory advantage of sparse depends on method and sparsity

### Customizing for Your Data

To benchmark with your own data:

1. Modify the `load_and_prepare_data()` function
2. Update the data path to your AnnData file
3. Adjust preprocessing steps as needed
4. Ensure data is stored in a layer (default: 'logcounts')

### Troubleshooting

**Memory tracking not working:**
- Install psutil: `pip install psutil`
- Memory values may be 0.0 on some runs due to GC timing

**Benchmark runs too slowly:**
- Reduce `--runs` parameter
- Reduce `--components` parameter
- Use a smaller dataset

**Out of memory errors:**
- Reduce number of components
- Use a smaller dataset
- Close other applications
- Try sparse input only

**Import errors:**
- Ensure actionet is built and installed
- Check that all dependencies are installed
- Verify Python path includes the src directory

## Related Test Files

- `test_svd_methods.py` - Basic validation of all SVD methods
- `test_svd_sparse_vs_dense.py` - Consistency testing between sparse/dense
- `test_svd_comprehensive.py` - (if exists) Additional comprehensive tests

## Contributing

To add new benchmark metrics or visualizations:

1. Add metric collection in `benchmark_svd_method()`
2. Update `create_summary_table()` to include new columns
3. Add visualization in `visualize_benchmark_results()`
4. Update this README with usage notes

## License

Same as the main ACTIONet project.
