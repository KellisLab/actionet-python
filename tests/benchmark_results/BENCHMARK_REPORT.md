# ACTIONet Backed Extension Benchmark Report

Date: 2026-02-26 04:15:50
Trials per mode: 3
SVD algorithm: PRIMME (id=3)
Backed chunk size: 4096
Backed decompression: scope='file' before every backed trial

## Dataset: small (test_adata.h5ad)
Shape: 6,790 cells x 32,236 genes
Label column: CellLabel

### Runtime & Peak Memory Comparison

| Step                 | Mem Time (s)   | Bck Time (s)   | Speedup (Mem/Bck)   | Mem Peak RSS (MB)   | Bck Peak RSS (MB)   |
|:---------------------|:---------------|:---------------|:--------------------|:--------------------|:--------------------|
| action_decomposition | 4.96 +/- 0.25  | 5.06 +/- 0.02  | 0.98x               | 206 +/- 255         | 38 +/- 15           |
| annotate_cells       | 0.40 +/- 0.00  | 0.17 +/- 0.00  | 2.28x               | 14 +/- 11           | 17 +/- 13           |
| archetype_diffusion  | 0.04 +/- 0.00  | 0.06 +/- 0.00  | 0.64x               | 0 +/- 0             | 0 +/- 0             |
| color_computation    | 0.00 +/- 0.00  | 0.00 +/- 0.00  | 0.42x               | 0 +/- 0             | 0 +/- 0             |
| feature_specificity  | 0.41 +/- 0.01  | 0.28 +/- 0.00  | 1.45x               | 0 +/- 0             | 1 +/- 1             |
| filter               | 0.26 +/- 0.01  | 0.57 +/- 0.01  | 0.45x               | 219 +/- 309         | 28 +/- 23           |
| find_markers         | 0.43 +/- 0.01  | 0.17 +/- 0.00  | 2.59x               | 54 +/- 74           | 0 +/- 0             |
| impute_features      | 0.12 +/- 0.00  | 0.06 +/- 0.00  | 1.97x               | 0 +/- 0             | 0 +/- 0             |
| network_construction | 1.36 +/- 0.01  | 1.54 +/- 0.01  | 0.88x               | 79 +/- 111          | 5 +/- 6             |
| normalize            | 0.18 +/- 0.01  | 0.23 +/- 0.00  | 0.80x               | 0 +/- 0             | 0 +/- 0             |
| reduce_kernel        | 3.68 +/- 0.08  | 15.69 +/- 0.29 | 0.23x               | 336 +/- 473         | 0 +/- 0             |
| umap_2d              | 3.07 +/- 0.02  | 3.68 +/- 0.02  | 0.83x               | 26 +/- 16           | 11 +/- 13           |
| umap_3d              | 3.43 +/- 0.03  | 4.06 +/- 0.01  | 0.84x               | 4 +/- 3             | 0 +/- 0             |
| TOTAL                | 18.33 +/- 0.39 | 31.58 +/- 0.36 | 0.58x               | 937 +/- 1246        | 98 +/- 22           |

### Result Parity (In-Memory vs Backed)

- **reduction_mean_abs_corr**: 1.0000
- **marker_overlap**: 1.0000
- **annotation_agreement**: 0.9996
- **imputation_mean_corr**: 1.0000

![Benchmark small](benchmark_small.png)

### Scaling Projections (Cell Count)

| Mode      | Cells      | Genes   |   Est. Time (min) |   Est. Peak RSS (GB) |
|:----------|:-----------|:--------|------------------:|---------------------:|
| in-memory | 100,000    | 10,000  |               1.7 |                  4.3 |
| in-memory | 500,000    | 10,000  |               8.9 |                 21.4 |
| in-memory | 1,000,000  | 10,000  |              18.1 |                 42.8 |
| in-memory | 5,000,000  | 10,000  |              93.7 |                214   |
| in-memory | 10,000,000 | 10,000  |             190.1 |                428   |
| backed    | 100,000    | 10,000  |               2.8 |                  0.4 |
| backed    | 500,000    | 10,000  |              14.3 |                  2.2 |
| backed    | 1,000,000  | 10,000  |              28.8 |                  4.5 |
| backed    | 5,000,000  | 10,000  |             147.5 |                 22.4 |
| backed    | 10,000,000 | 10,000  |             297.9 |                 44.8 |

## Dataset: large (adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad)
Shape: 41,007 cells x 36,377 genes
Batch key: UID (25 batches)
Label column: CellType

### Runtime & Peak Memory Comparison

| Step                 | Mem Time (s)    | Bck Time (s)    | Speedup (Mem/Bck)   | Mem Peak RSS (MB)   | Bck Peak RSS (MB)   |
|:---------------------|:----------------|:----------------|:--------------------|:--------------------|:--------------------|
| action_decomposition | 27.23 +/- 0.48  | 27.85 +/- 0.22  | 0.98x               | 2748 +/- 60         | 752 +/- 84          |
| annotate_cells       | 4.40 +/- 0.04   | 1.49 +/- 0.00   | 2.95x               | 4143 +/- 40         | 401 +/- 48          |
| archetype_diffusion  | 0.41 +/- 0.00   | 0.57 +/- 0.01   | 0.71x               | 9 +/- 5             | 0 +/- 0             |
| batch_correction     | 8.17 +/- 0.03   | 3.66 +/- 0.02   | 2.23x               | 8033 +/- 177        | 18 +/- 20           |
| color_computation    | 0.00 +/- 0.00   | 0.01 +/- 0.00   | 0.25x               | 0 +/- 0             | 0 +/- 0             |
| feature_specificity  | 6.10 +/- 0.25   | 2.53 +/- 0.02   | 2.41x               | 6973 +/- 61         | 97 +/- 69           |
| filter               | 3.21 +/- 0.14   | 6.83 +/- 0.15   | 0.47x               | 5683 +/- 472        | 6289 +/- 166        |
| find_markers         | 5.70 +/- 0.08   | 1.33 +/- 0.01   | 4.30x               | 9988 +/- 168        | 0 +/- 0             |
| impute_features      | 1.43 +/- 0.02   | 0.70 +/- 0.00   | 2.04x               | 3624 +/- 0          | 0 +/- 0             |
| network_construction | 26.21 +/- 0.27  | 28.15 +/- 0.09  | 0.93x               | 3197 +/- 371        | 1035 +/- 67         |
| normalize            | 1.39 +/- 0.04   | 2.48 +/- 0.10   | 0.56x               | 1812 +/- 0          | 67 +/- 5            |
| reduce_kernel        | 38.95 +/- 0.22  | 172.19 +/- 1.21 | 0.23x               | 6352 +/- 238        | 16 +/- 6            |
| umap_2d              | 16.26 +/- 0.04  | 27.17 +/- 0.06  | 0.60x               | 1530 +/- 91         | 374 +/- 14          |
| umap_3d              | 17.26 +/- 0.13  | 28.38 +/- 0.03  | 0.61x               | 1548 +/- 129        | 0 +/- 0             |
| TOTAL                | 156.70 +/- 0.92 | 303.35 +/- 1.63 | 0.52x               | 7856 +/- 2088       | 6312 +/- 134        |

### Result Parity (In-Memory vs Backed)

- **reduction_mean_abs_corr**: 1.0000
- **marker_overlap**: 1.0000
- **annotation_agreement**: 1.0000
- **imputation_mean_corr**: 0.9981

![Benchmark large](benchmark_large.png)

### Scaling Projections (Cell Count)

| Mode      | Cells      | Genes   |   Est. Time (min) |   Est. Peak RSS (GB) |
|:----------|:-----------|:--------|------------------:|---------------------:|
| in-memory | 100,000    | 10,000  |               2.6 |                  5.3 |
| in-memory | 500,000    | 10,000  |              13.9 |                 26.3 |
| in-memory | 1,000,000  | 10,000  |              28.4 |                 52.7 |
| in-memory | 5,000,000  | 10,000  |             150.2 |                263.3 |
| in-memory | 10,000,000 | 10,000  |             307.4 |                526.6 |
| backed    | 100,000    | 10,000  |               4.3 |                  4.2 |
| backed    | 500,000    | 10,000  |              22.4 |                 21.2 |
| backed    | 1,000,000  | 10,000  |              45.6 |                 42.3 |
| backed    | 5,000,000  | 10,000  |             236.8 |                211.6 |
| backed    | 10,000,000 | 10,000  |             481.1 |                423.1 |

### Scaling Projections (Batch Count)

| Mode      |   Batches |   Est. Batch Corr Time (s) |   Est. Batch Corr Peak RSS (MB) |
|:----------|----------:|---------------------------:|--------------------------------:|
| in-memory |         1 |                        0.3 |                             321 |
| in-memory |         5 |                        1.6 |                            1607 |
| in-memory |        25 |                        8.2 |                            8033 |
| in-memory |        50 |                       16.3 |                           16066 |
| in-memory |       100 |                       32.7 |                           32133 |
| backed    |         1 |                        0.1 |                               1 |
| backed    |         5 |                        0.7 |                               4 |
| backed    |        25 |                        3.7 |                              18 |
| backed    |        50 |                        7.3 |                              37 |
| backed    |       100 |                       14.7 |                              74 |
