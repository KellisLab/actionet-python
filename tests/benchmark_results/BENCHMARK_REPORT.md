# ACTIONet Backed Extension Benchmark Report

Date: 2026-02-27 01:56:04
Trials per mode: 1
SVD algorithm: PRIMME (id=3)
Backed chunk size: 4096
Backed decompression: scope='file' before every backed trial

## Dataset: large (adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad)
Shape: 41,007 cells x 36,377 genes
Batch key: UID (25 batches)
Label column: CellType

### Runtime & Peak Memory Comparison

| Step                 | Mem Time (s)    | Bck Time (s)    | Speedup (Mem/Bck)   | Mem Peak RSS (MB)   | Bck Peak RSS (MB)   |
|:---------------------|:----------------|:----------------|:--------------------|:--------------------|:--------------------|
| action_decomposition | 28.16 +/- 0.00  | 29.94 +/- 0.00  | 0.94x               | 2361 +/- 0          | 1030 +/- 0          |
| annotate_cells       | 4.73 +/- 0.00   | 1.57 +/- 0.00   | 3.01x               | 6651 +/- 0          | 276 +/- 0           |
| archetype_diffusion  | 0.43 +/- 0.00   | 0.60 +/- 0.00   | 0.72x               | 16 +/- 0            | 0 +/- 0             |
| batch_correction     | 9.18 +/- 0.00   | 3.80 +/- 0.00   | 2.42x               | 8466 +/- 0          | 28 +/- 0            |
| color_computation    | 0.00 +/- 0.00   | 0.02 +/- 0.00   | 0.21x               | 3 +/- 0             | 0 +/- 0             |
| feature_specificity  | 7.38 +/- 0.00   | 2.60 +/- 0.00   | 2.84x               | 7064 +/- 0          | 0 +/- 0             |
| filter               | 4.67 +/- 0.00   | 6.05 +/- 0.00   | 0.77x               | 5406 +/- 0          | 450 +/- 0           |
| find_markers         | 7.56 +/- 0.00   | 1.40 +/- 0.00   | 5.41x               | 12324 +/- 0         | 0 +/- 0             |
| impute_features      | 1.44 +/- 0.00   | 0.75 +/- 0.00   | 1.93x               | 3624 +/- 0          | 0 +/- 0             |
| network_construction | 28.20 +/- 0.00  | 30.92 +/- 0.00  | 0.91x               | 3874 +/- 0          | 1976 +/- 0          |
| normalize            | 1.39 +/- 0.00   | 2.52 +/- 0.00   | 0.55x               | 1812 +/- 0          | 1811 +/- 0          |
| reduce_kernel        | 39.91 +/- 0.00  | 182.84 +/- 0.00 | 0.22x               | 6732 +/- 0          | 12 +/- 0            |
| umap_2d              | 17.48 +/- 0.00  | 29.96 +/- 0.00  | 0.58x               | 1931 +/- 0          | 382 +/- 0           |
| umap_3d              | 18.45 +/- 0.00  | 30.84 +/- 0.00  | 0.60x               | 1786 +/- 0          | 0 +/- 0             |
| TOTAL                | 168.99 +/- 0.00 | 323.82 +/- 0.00 | 0.52x               | 10402 +/- 0         | 5873 +/- 0          |

### Result Parity (In-Memory vs Backed)

- **reduction_mean_abs_corr**: 1.0000
- **marker_overlap**: 1.0000
- **annotation_agreement**: 1.0000
- **imputation_mean_corr**: 0.9981

![Benchmark large](benchmark_large.png)

### Scaling Projections (Cell Count)

| Mode      | Cells      | Genes   |   Est. Time (min) |   Est. Peak RSS (GB) |
|:----------|:-----------|:--------|------------------:|---------------------:|
| in-memory | 100,000    | 10,000  |               2.8 |                  7   |
| in-memory | 500,000    | 10,000  |              14.9 |                 34.9 |
| in-memory | 1,000,000  | 10,000  |              30.6 |                 69.7 |
| in-memory | 5,000,000  | 10,000  |             161.9 |                348.7 |
| in-memory | 10,000,000 | 10,000  |             331.2 |                697.3 |
| backed    | 100,000    | 10,000  |               4.6 |                  3.9 |
| backed    | 500,000    | 10,000  |              24.1 |                 19.7 |
| backed    | 1,000,000  | 10,000  |              49.1 |                 39.4 |
| backed    | 5,000,000  | 10,000  |             254.9 |                196.9 |
| backed    | 10,000,000 | 10,000  |             518   |                393.7 |

### Scaling Projections (Batch Count)

| Mode      |   Batches |   Est. Batch Corr Time (s) |   Est. Batch Corr Peak RSS (MB) |
|:----------|----------:|---------------------------:|--------------------------------:|
| in-memory |         1 |                        0.4 |                             339 |
| in-memory |         5 |                        1.8 |                            1693 |
| in-memory |        25 |                        9.2 |                            8466 |
| in-memory |        50 |                       18.4 |                           16932 |
| in-memory |       100 |                       36.7 |                           33864 |
| backed    |         1 |                        0.2 |                               1 |
| backed    |         5 |                        0.8 |                               6 |
| backed    |        25 |                        3.8 |                              28 |
| backed    |        50 |                        7.6 |                              55 |
| backed    |       100 |                       15.2 |                             110 |
