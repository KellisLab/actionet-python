# Branch Benchmark Report

Generated: 2026-03-23 22:44:40

Baseline branch: `dev-backed`
Candidate branch: `feature/orientation-unification`

## Thresholds

- Repeated cases: wall `>20%`, RSS `>15%`
- Single-trial large backed cases: wall `>25%`, RSS `>20%`

## Datasets


| Dataset           | Tier   | Path                                                                                |
| ----------------- | ------ | ----------------------------------------------------------------------------------- |
| sparse_medium     | 41007  | /data/actionet_benchmark/branch_compare_20260323_185615/data/sparse_medium.h5ad     |
| scale_subset_50k  | 50000  | /data/actionet_benchmark/branch_compare_20260323_185615/data/scale_subset_50k.h5ad  |
| scale_subset_100k | 100000 | /data/actionet_benchmark/branch_compare_20260323_185615/data/scale_subset_100k.h5ad |
| scale_subset_200k | 200000 | /data/actionet_benchmark/branch_compare_20260323_185615/data/scale_subset_200k.h5ad |
| scale_subset_250k | 250000 | /data/actionet_benchmark/branch_compare_20260323_185615/data/scale_subset_250k.h5ad |
| scale_full        | 300157 | /data/git_projects/actionet-python/data/adata_agg_Scn4b_OX_fil.h5ad                 |


## Failures

*No failures recorded.*

## Findings

Obvious regressions:

- `scale_subset_200k` `backed_decompressed` `compute_network_diffusion`: wall +95.5%, RSS +246.3%
- `scale_subset_100k` `backed_decompressed` `compute_network_diffusion`: wall +81.3%, RSS +219.5%
- `scale_subset_200k` `backed_decompressed` `correct_batch_effect`: wall +182.7%, RSS +38.4%
- `scale_subset_100k` `backed_decompressed` `correct_batch_effect`: wall +173.8%, RSS +73.0%
- `scale_full` `backed_decompressed` `correct_batch_effect`: wall +165.0%, RSS -0.3%
- `sparse_medium` `backed_decompressed` `correct_batch_effect`: wall +135.5%, RSS +5.1%
- `scale_subset_200k` `backed_decompressed` `total`: wall +1.1%, RSS +47.6%
- `sparse_medium` `backed_decompressed` `build_network`: wall -9.5%, RSS +32.6%
- `sparse_medium` `backed_decompressed` `layout_network_2d`: wall +11.3%, RSS +22.8%
- `scale_subset_100k` `backed_decompressed` `total`: wall -0.1%, RSS +19.6%
- `sparse_medium` `backed_decompressed` `compute_network_diffusion`: wall +0.9%, RSS +16.8%

## Comparable Stage Deltas


| Dataset           | Mode                | Stage                                 | dev-backed wall (s) | feature/orientation-unification wall (s) | Wall delta | dev-backed RSS (MB) | feature/orientation-unification RSS (MB) | RSS delta | Flag |
| ----------------- | ------------------- | ------------------------------------- | ------------------- | ---------------------------------------- | ---------- | ------------------- | ---------------------------------------- | --------- | ---- |
| sparse_medium     | backed_decompressed | reduce_kernel                         | 147.38              | 148.00                                   | +0.4%      | 287.60              | 281.42                                   | -2.2%     |      |
| sparse_medium     | backed_decompressed | correct_batch_effect                  | 8.80                | 20.74                                    | +135.5%    | 462.56              | 486.12                                   | +5.1%     | yes  |
| sparse_medium     | backed_decompressed | run_action                            | 36.76               | 35.75                                    | -2.8%      | 1740.13             | 1485.54                                  | -14.6%    |      |
| sparse_medium     | backed_decompressed | build_network                         | 53.43               | 48.37                                    | -9.5%      | 298.25              | 395.38                                   | +32.6%    | yes  |
| sparse_medium     | backed_decompressed | compute_network_diffusion             | 1.22                | 1.23                                     | +0.9%      | 145.12              | 169.49                                   | +16.8%    | yes  |
| sparse_medium     | backed_decompressed | layout_network_2d                     | 5.39                | 5.99                                     | +11.3%     | 137.96              | 169.42                                   | +22.8%    | yes  |
| sparse_medium     | backed_decompressed | compute_archetype_feature_specificity | 16.52               | 17.23                                    | +4.3%      | 1100.10             | 1085.04                                  | -1.4%     |      |
| sparse_medium     | backed_decompressed | total                                 | 269.86              | 277.68                                   | +2.9%      | 3111.13             | 2831.38                                  | -9.0%     |      |
| sparse_medium     | in_memory           | reduce_kernel                         | 66.85               | 56.53                                    | -15.4%     | 7149.34             | 4393.02                                  | -38.6%    |      |
| sparse_medium     | in_memory           | correct_batch_effect                  | 28.44               | 19.51                                    | -31.4%     | 7087.00             | 4395.21                                  | -38.0%    |      |
| sparse_medium     | in_memory           | run_action                            | 34.60               | 33.28                                    | -3.8%      | 1595.73             | 1329.76                                  | -16.7%    |      |
| sparse_medium     | in_memory           | build_network                         | 45.50               | 43.79                                    | -3.8%      | 321.77              | 339.68                                   | +5.6%     |      |
| sparse_medium     | in_memory           | compute_network_diffusion             | 1.05                | 0.94                                     | -10.6%     | 168.24              | 126.27                                   | -24.9%    |      |
| sparse_medium     | in_memory           | layout_network_2d                     | 5.96                | 5.87                                     | -1.6%      | 168.24              | 126.27                                   | -24.9%    |      |
| sparse_medium     | in_memory           | compute_archetype_feature_specificity | 26.02               | 26.39                                    | +1.4%      | 7170.16             | 4417.05                                  | -38.4%    |      |
| sparse_medium     | in_memory           | total                                 | 208.93              | 186.82                                   | -10.6%     | 9201.04             | 6271.93                                  | -31.8%    |      |
| scale_subset_100k | backed_decompressed | reduce_kernel                         | 202.64              | 202.10                                   | -0.3%      | 249.08              | 285.14                                   | +14.5%    |      |
| scale_subset_100k | backed_decompressed | correct_batch_effect                  | 9.99                | 27.34                                    | +173.8%    | 277.40              | 479.85                                   | +73.0%    | yes  |
| scale_subset_100k | backed_decompressed | run_action                            | 103.38              | 101.72                                   | -1.6%      | 3469.02             | 2728.25                                  | -21.4%    |      |
| scale_subset_100k | backed_decompressed | build_network                         | 185.82              | 170.85                                   | -8.1%      | 1402.45             | 1355.35                                  | -3.4%     |      |
| scale_subset_100k | backed_decompressed | compute_network_diffusion             | 4.97                | 9.01                                     | +81.3%     | 797.09              | 2547.00                                  | +219.5%   | yes  |
| scale_subset_100k | backed_decompressed | layout_network_2d                     | 27.83               | 23.22                                    | -16.6%     | 815.90              | 0.30                                     | -100.0%   |      |
| scale_subset_100k | backed_decompressed | compute_archetype_feature_specificity | 21.95               | 21.67                                    | -1.3%      | 509.59              | 0.00                                     | -100.0%   |      |
| scale_subset_100k | backed_decompressed | total                                 | 556.93              | 556.26                                   | -0.1%      | 5510.03             | 6588.11                                  | +19.6%    | yes  |
| scale_subset_100k | in_memory           | reduce_kernel                         | 75.94               | 61.21                                    | -19.4%     | 8091.77             | 5118.80                                  | -36.7%    |      |
| scale_subset_100k | in_memory           | correct_batch_effect                  | 33.20               | 26.97                                    | -18.8%     | 8225.55             | 5085.16                                  | -38.2%    |      |
| scale_subset_100k | in_memory           | run_action                            | 96.83               | 97.42                                    | +0.6%      | 3180.75             | 2410.23                                  | -24.2%    |      |
| scale_subset_100k | in_memory           | build_network                         | 152.76              | 142.38                                   | -6.8%      | 1354.59             | 1539.14                                  | +13.6%    |      |
| scale_subset_100k | in_memory           | compute_network_diffusion             | 4.34                | 3.97                                     | -8.7%      | 713.28              | 730.61                                   | +2.4%     |      |
| scale_subset_100k | in_memory           | layout_network_2d                     | 26.68               | 26.21                                    | -1.8%      | 765.70              | 764.16                                   | -0.2%     |      |
| scale_subset_100k | in_memory           | compute_archetype_feature_specificity | 29.32               | 27.79                                    | -5.2%      | 8156.21             | 5106.97                                  | -37.4%    |      |
| scale_subset_100k | in_memory           | total                                 | 419.60              | 386.38                                   | -7.9%      | 12743.15            | 9032.79                                  | -29.1%    |      |
| scale_subset_200k | backed_decompressed | reduce_kernel                         | 406.02              | 406.56                                   | +0.1%      | 148.03              | 148.09                                   | +0.0%     |      |
| scale_subset_200k | backed_decompressed | correct_batch_effect                  | 19.29               | 54.52                                    | +182.7%    | 297.48              | 411.73                                   | +38.4%    | yes  |
| scale_subset_200k | backed_decompressed | run_action                            | 256.32              | 261.66                                   | +2.1%      | 4161.52             | 3853.66                                  | -7.4%     |      |
| scale_subset_200k | backed_decompressed | build_network                         | 544.16              | 513.29                                   | -5.7%      | 3593.06             | 3381.68                                  | -5.9%     |      |
| scale_subset_200k | backed_decompressed | compute_network_diffusion             | 11.85               | 23.16                                    | +95.5%     | 1864.88             | 6457.79                                  | +246.3%   | yes  |
| scale_subset_200k | backed_decompressed | layout_network_2d                     | 65.17               | 59.53                                    | -8.7%      | 1907.38             | 0.59                                     | -100.0%   |      |
| scale_subset_200k | backed_decompressed | compute_archetype_feature_specificity | 43.55               | 42.70                                    | -1.9%      | 622.37              | 0.00                                     | -100.0%   |      |
| scale_subset_200k | backed_decompressed | total                                 | 1346.77             | 1361.84                                  | +1.1%      | 8237.76             | 12159.34                                 | +47.6%    | yes  |
| scale_full        | backed_decompressed | reduce_kernel                         | 607.88              | 610.54                                   | +0.4%      | 447.98              | 451.95                                   | +0.9%     |      |
| scale_full        | backed_decompressed | correct_batch_effect                  | 31.05               | 82.28                                    | +165.0%    | 538.91              | 537.34                                   | -0.3%     | yes  |
| scale_full        | backed_decompressed | run_action                            | 445.23              | 440.29                                   | -1.1%      | 4435.44             | 4036.46                                  | -9.0%     |      |
| scale_full        | backed_decompressed | build_network                         | 953.80              | 856.90                                   | -10.2%     | 5159.90             | 3499.93                                  | -32.2%    |      |
| scale_full        | backed_decompressed | compute_network_diffusion             | 37.72               | 42.78                                    | +13.4%     | 11043.68            | 11620.15                                 | +5.2%     |      |
| scale_full        | backed_decompressed | layout_network_2d                     | 108.27              | 111.61                                   | +3.1%      | 0.53                | 0.59                                     | +11.5%    |      |
| scale_full        | backed_decompressed | compute_archetype_feature_specificity | 55.40               | 62.86                                    | +13.5%     | 0.00                | 0.00                                     | NA        |      |
| scale_full        | backed_decompressed | total                                 | 2239.73             | 2207.69                                  | -1.4%      | 18280.61            | 18250.83                                 | -0.2%     |      |


## Backed Growth Summary


| Dataset           | dev-backed total (s) | feature/orientation-unification total (s) | Wall delta | dev-backed RSS (MB) | feature/orientation-unification RSS (MB) | RSS delta |
| ----------------- | -------------------- | ----------------------------------------- | ---------- | ------------------- | ---------------------------------------- | --------- |
| sparse_medium     | 269.86               | 277.68                                    | +2.9%      | 3111.13             | 2831.38                                  | -9.0%     |
| scale_subset_100k | 556.93               | 556.26                                    | -0.1%      | 5510.03             | 6588.11                                  | +19.6%    |
| scale_subset_200k | 1346.77              | 1361.84                                   | +1.1%      | 8237.76             | 12159.34                                 | +47.6%    |
| scale_full        | 2239.73              | 2207.69                                   | -1.4%      | 18280.61            | 18250.83                                 | -0.2%     |


## Outputs

- Raw rows: `/data/actionet_benchmark/branch_compare_20260323_185615/raw`
- Summary CSV: `/data/actionet_benchmark/branch_compare_20260323_185615/summary.csv`
- Comparison CSV: `/data/actionet_benchmark/branch_compare_20260323_185615/comparison.csv`

