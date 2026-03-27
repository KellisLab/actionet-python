# Branch Benchmark Report

Generated: 2026-03-27 03:16:54

Baseline branch: `dev-backed`
Candidate branch: `feature/orientation-unification`

## Thresholds

- Repeated cases: wall `>20%`, RSS `>15%`
- Single-trial large backed cases: wall `>25%`, RSS `>20%`

## Datasets


| Dataset           | Tier   | Path                                                                                |
| ----------------- | ------ | ----------------------------------------------------------------------------------- |
| sparse_medium     | 41007  | /data/actionet_benchmark/branch_compare_20260326_232908/data/sparse_medium.h5ad     |
| scale_subset_50k  | 50000  | /data/actionet_benchmark/branch_compare_20260326_232908/data/scale_subset_50k.h5ad  |
| scale_subset_100k | 100000 | /data/actionet_benchmark/branch_compare_20260326_232908/data/scale_subset_100k.h5ad |
| scale_subset_200k | 200000 | /data/actionet_benchmark/branch_compare_20260326_232908/data/scale_subset_200k.h5ad |
| scale_subset_250k | 250000 | /data/actionet_benchmark/branch_compare_20260326_232908/data/scale_subset_250k.h5ad |
| scale_full        | 300157 | /data/git_projects/actionet-python/data/adata_agg_Scn4b_OX_fil.h5ad                 |


## Failures

*No failures recorded.*

## Findings

Obvious regressions:

- `scale_full` `backed_decompressed` `compute_archetype_feature_specificity`: wall +12.4%, RSS +1600.0%
- `scale_subset_200k` `backed_decompressed` `compute_network_diffusion`: wall +92.6%, RSS +227.5%
- `scale_subset_100k` `backed_decompressed` `compute_network_diffusion`: wall +81.4%, RSS +193.2%
- `scale_subset_200k` `backed_decompressed` `reduce_kernel`: wall +0.8%, RSS +95.6%
- `scale_subset_200k` `backed_decompressed` `total`: wall -0.8%, RSS +48.8%
- `scale_full` `backed_decompressed` `build_network`: wall -2.5%, RSS +42.2%
- `sparse_medium` `backed_decompressed` `compute_network_diffusion`: wall -2.0%, RSS +40.2%
- `sparse_medium` `backed_decompressed` `layout_network_2d`: wall +12.2%, RSS +34.3%
- `scale_subset_200k` `backed_decompressed` `build_network`: wall -1.1%, RSS +32.2%
- `scale_full` `backed_decompressed` `layout_network_2d`: wall +4.4%, RSS +23.8%
- `scale_subset_100k` `backed_decompressed` `total`: wall -0.7%, RSS +16.6%

## Comparable Stage Deltas


| Dataset           | Mode                | Stage                                 | dev-backed wall (s) | feature/orientation-unification wall (s) | Wall delta | dev-backed RSS (MB) | feature/orientation-unification RSS (MB) | RSS delta | Flag |
| ----------------- | ------------------- | ------------------------------------- | ------------------- | ---------------------------------------- | ---------- | ------------------- | ---------------------------------------- | --------- | ---- |
| sparse_medium     | backed_decompressed | reduce_kernel                         | 147.56              | 147.45                                   | -0.1%      | 263.42              | 299.34                                   | +13.6%    |      |
| sparse_medium     | backed_decompressed | correct_batch_effect                  | 20.63               | 20.60                                    | -0.1%      | 547.57              | 521.08                                   | -4.8%     |      |
| sparse_medium     | backed_decompressed | run_action                            | 35.30               | 36.18                                    | +2.5%      | 1740.67             | 1354.14                                  | -22.2%    |      |
| sparse_medium     | backed_decompressed | build_network                         | 52.14               | 48.41                                    | -7.1%      | 385.23              | 353.29                                   | -8.3%     |      |
| sparse_medium     | backed_decompressed | compute_network_diffusion             | 1.24                | 1.22                                     | -2.0%      | 118.38              | 165.97                                   | +40.2%    | yes  |
| sparse_medium     | backed_decompressed | layout_network_2d                     | 5.31                | 5.96                                     | +12.2%     | 78.84               | 105.89                                   | +34.3%    | yes  |
| sparse_medium     | backed_decompressed | compute_archetype_feature_specificity | 16.55               | 17.05                                    | +3.0%      | 1085.04             | 1095.23                                  | +0.9%     |      |
| sparse_medium     | backed_decompressed | total                                 | 279.19              | 277.25                                   | -0.7%      | 3164.86             | 2872.59                                  | -9.2%     |      |
| sparse_medium     | in_memory           | reduce_kernel                         | 66.63               | 56.37                                    | -15.4%     | 7163.89             | 4390.85                                  | -38.7%    |      |
| sparse_medium     | in_memory           | correct_batch_effect                  | 27.92               | 19.68                                    | -29.5%     | 7186.51             | 4447.54                                  | -38.1%    |      |
| sparse_medium     | in_memory           | run_action                            | 33.14               | 34.28                                    | +3.5%      | 1617.46             | 1329.77                                  | -17.8%    |      |
| sparse_medium     | in_memory           | build_network                         | 43.69               | 43.85                                    | +0.4%      | 297.85              | 339.68                                   | +14.0%    |      |
| sparse_medium     | in_memory           | compute_network_diffusion             | 1.05                | 0.94                                     | -11.1%     | 167.84              | 84.18                                    | -49.8%    |      |
| sparse_medium     | in_memory           | layout_network_2d                     | 5.95                | 5.81                                     | -2.4%      | 129.74              | 84.18                                    | -35.1%    |      |
| sparse_medium     | in_memory           | compute_archetype_feature_specificity | 25.73               | 26.35                                    | +2.4%      | 7076.84             | 4452.70                                  | -37.1%    |      |
| sparse_medium     | in_memory           | total                                 | 204.62              | 187.81                                   | -8.2%      | 9221.14             | 6247.60                                  | -32.2%    |      |
| scale_subset_100k | backed_decompressed | reduce_kernel                         | 200.96              | 201.64                                   | +0.3%      | 201.61              | 201.14                                   | -0.2%     |      |
| scale_subset_100k | backed_decompressed | correct_batch_effect                  | 27.21               | 27.19                                    | -0.1%      | 442.71              | 389.38                                   | -12.0%    |      |
| scale_subset_100k | backed_decompressed | run_action                            | 101.84              | 100.41                                   | -1.4%      | 3474.34             | 2461.72                                  | -29.1%    |      |
| scale_subset_100k | backed_decompressed | build_network                         | 173.82              | 171.28                                   | -1.5%      | 1424.81             | 1350.67                                  | -5.2%     |      |
| scale_subset_100k | backed_decompressed | compute_network_diffusion             | 4.96                | 9.00                                     | +81.4%     | 889.36              | 2607.74                                  | +193.2%   | yes  |
| scale_subset_100k | backed_decompressed | layout_network_2d                     | 27.78               | 23.31                                    | -16.1%     | 805.35              | 0.63                                     | -99.9%    |      |
| scale_subset_100k | backed_decompressed | compute_archetype_feature_specificity | 21.83               | 21.64                                    | -0.9%      | 510.10              | 0.07                                     | -100.0%   |      |
| scale_subset_100k | backed_decompressed | total                                 | 558.80              | 554.80                                   | -0.7%      | 5614.62             | 6548.68                                  | +16.6%    | yes  |
| scale_subset_100k | in_memory           | reduce_kernel                         | 75.48               | 60.65                                    | -19.6%     | 8175.53             | 5079.93                                  | -37.9%    |      |
| scale_subset_100k | in_memory           | correct_batch_effect                  | 33.10               | 27.02                                    | -18.4%     | 8248.03             | 5095.68                                  | -38.2%    |      |
| scale_subset_100k | in_memory           | run_action                            | 96.49               | 95.80                                    | -0.7%      | 3150.52             | 2412.64                                  | -23.4%    |      |
| scale_subset_100k | in_memory           | build_network                         | 141.26              | 142.17                                   | +0.6%      | 1479.07             | 1537.10                                  | +3.9%     |      |
| scale_subset_100k | in_memory           | compute_network_diffusion             | 4.24                | 3.91                                     | -7.7%      | 798.58              | 759.06                                   | -4.9%     |      |
| scale_subset_100k | in_memory           | layout_network_2d                     | 26.68               | 26.27                                    | -1.5%      | 737.76              | 788.42                                   | +6.9%     |      |
| scale_subset_100k | in_memory           | compute_archetype_feature_specificity | 27.60               | 27.72                                    | +0.5%      | 8075.48             | 5002.11                                  | -38.1%    |      |
| scale_subset_100k | in_memory           | total                                 | 405.39              | 384.10                                   | -5.3%      | 12847.68            | 9094.16                                  | -29.2%    |      |
| scale_subset_200k | backed_decompressed | reduce_kernel                         | 400.01              | 403.26                                   | +0.8%      | 140.19              | 274.15                                   | +95.6%    | yes  |
| scale_subset_200k | backed_decompressed | correct_batch_effect                  | 53.51               | 54.25                                    | +1.4%      | 408.21              | 363.63                                   | -10.9%    |      |
| scale_subset_200k | backed_decompressed | run_action                            | 246.93              | 240.03                                   | -2.8%      | 4223.59             | 3201.17                                  | -24.2%    |      |
| scale_subset_200k | backed_decompressed | build_network                         | 513.22              | 507.43                                   | -1.1%      | 2575.59             | 3405.91                                  | +32.2%    | yes  |
| scale_subset_200k | backed_decompressed | compute_network_diffusion             | 11.71               | 22.54                                    | +92.6%     | 1965.78             | 6438.06                                  | +227.5%   | yes  |
| scale_subset_200k | backed_decompressed | layout_network_2d                     | 67.65               | 55.81                                    | -17.5%     | 1910.67             | 0.60                                     | -100.0%   |      |
| scale_subset_200k | backed_decompressed | compute_archetype_feature_specificity | 42.88               | 41.56                                    | -3.1%      | 622.38              | 0.07                                     | -100.0%   |      |
| scale_subset_200k | backed_decompressed | total                                 | 1336.33             | 1325.26                                  | -0.8%      | 8121.22             | 12083.33                                 | +48.8%    | yes  |
| scale_full        | backed_decompressed | reduce_kernel                         | 606.52              | 606.37                                   | -0.0%      | 396.34              | 409.11                                   | +3.2%     |      |
| scale_full        | backed_decompressed | correct_batch_effect                  | 81.86               | 81.55                                    | -0.4%      | 537.19              | 465.15                                   | -13.4%    |      |
| scale_full        | backed_decompressed | run_action                            | 450.06              | 465.51                                   | +3.4%      | 4346.97             | 3203.78                                  | -26.3%    |      |
| scale_full        | backed_decompressed | build_network                         | 883.59              | 861.18                                   | -2.5%      | 3506.15             | 4984.44                                  | +42.2%    | yes  |
| scale_full        | backed_decompressed | compute_network_diffusion             | 39.76               | 40.16                                    | +1.0%      | 11085.50            | 11555.10                                 | +4.2%     |      |
| scale_full        | backed_decompressed | layout_network_2d                     | 107.09              | 111.82                                   | +4.4%      | 0.53                | 0.66                                     | +23.8%    | yes  |
| scale_full        | backed_decompressed | compute_archetype_feature_specificity | 56.09               | 63.08                                    | +12.4%     | 0.00                | 0.07                                     | +1600.0%  | yes  |
| scale_full        | backed_decompressed | total                                 | 2225.44             | 2230.13                                  | +0.2%      | 17992.91            | 18180.50                                 | +1.0%     |      |


## Backed Growth Summary


| Dataset           | dev-backed total (s) | feature/orientation-unification total (s) | Wall delta | dev-backed RSS (MB) | feature/orientation-unification RSS (MB) | RSS delta |
| ----------------- | -------------------- | ----------------------------------------- | ---------- | ------------------- | ---------------------------------------- | --------- |
| sparse_medium     | 279.19               | 277.25                                    | -0.7%      | 3164.86             | 2872.59                                  | -9.2%     |
| scale_subset_100k | 558.80               | 554.80                                    | -0.7%      | 5614.62             | 6548.68                                  | +16.6%    |
| scale_subset_200k | 1336.33              | 1325.26                                   | -0.8%      | 8121.22             | 12083.33                                 | +48.8%    |
| scale_full        | 2225.44              | 2230.13                                   | +0.2%      | 17992.91            | 18180.50                                 | +1.0%     |


## Outputs

- Raw rows: `/data/actionet_benchmark/branch_compare_20260326_232908/raw`
- Summary CSV: `/data/actionet_benchmark/branch_compare_20260326_232908/summary.csv`
- Comparison CSV: `/data/actionet_benchmark/branch_compare_20260326_232908/comparison.csv`

