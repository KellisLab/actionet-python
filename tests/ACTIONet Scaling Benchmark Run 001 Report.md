# ACTIONet Scaling Benchmark Run 001 Report

Source benchmark run: `/home/sebastian/data/actionet_benchmark/run_001/`  
Benchmark spec: [ACTIONet Scaling Benchmark Suite.md](/home/sebastian/data/git_projects/actionet-python/tests/ACTIONet%20Scaling%20Benchmark%20Suite.md)

## Summary

The run confirms the central conclusion from the design discussion: the current
default `k*nn` path is not a realistic route to 10M+ cells, even with large
RAM, while the `knn_ceiling` path is materially better but still leaves
`reduce_kernel`, `run_action`, `layout_2d`, and `color_computation` as the next
major scale limits.

At the largest measured real dataset (`scale_full`, 300,157 cells):


| Profile                                            | Mode                | Total wall time | Peak RSS proxy |
| -------------------------------------------------- | ------------------- | --------------- | -------------- |
| `default` (`k*nn`, `H_stacked`)                    | in-memory           | 43.1 min        | 37.0 GB        |
| `default` (`k*nn`, `H_stacked`)                    | backed-decompressed | 49.4 min        | 4.0 GB         |
| `knn_ceiling` (`knn`, `k=100`, `action_corrected`) | in-memory           | 23.4 min        | 34.6 GB        |
| `knn_ceiling` (`knn`, `k=100`, `action_corrected`) | backed-decompressed | 27.7 min        | 4.0 GB         |


`knn_ceiling` roughly halves total runtime at 300k while preserving a much
lower memory envelope in backed mode. The default `k*nn` profile still runs at
300k, but the network stage is already the dominant term and becomes the main
reason the default path fails the 10M target.

## Method

- Projections below were fit from the scale-family tiers only:
`25k`, `50k`, `100k`, `150k`, `200k`, `250k`, `300k`, and `scale_full`
(`300,157` cells).
- Workflow wall time was recomputed as the sum of stage means.
- Workflow peak RSS was recomputed as the maximum stage peak RSS, which is a
better proxy than the workflow-level `total` rows in the raw summary.
- Practical ceiling uses the suite's runtime budget of `12h`, but replaces the
original 150 GB memory budget with the requested HPC budget of `500 GB`.
- Extrapolations beyond 10M are planning estimates, not confidence-grade
forecasts. The 50M rows should be read as order-of-magnitude guidance.

## Main Findings

- `default` backed is runtime-limited, not RAM-limited, under a 500 GB budget.
The fit projects `10M` at `51.2h / 79.9 GB`, so RAM is not the blocker.
`k*nn` simply grows too fast.
- `default` in-memory is not competitive beyond ~1M. It projects `5M` at
`22.3h / 660.6 GB`, which is already beyond the assumed RAM budget.
- `knn_ceiling` backed is the only currently benchmarked path with a plausible
10M story: `10M` projects to `15.9h / 22.4 GB`. That is still above the 12h
target, but within reach if the runtime target is relaxed or the next
bottlenecks are optimized.
- `knn_ceiling` in-memory still becomes RAM-limited before 5M:
`5M -> 7.9h / 594.8 GB`.
- `k*nn` representation matters a lot. In the network-only sweep,
`H_stacked` is far worse than `H_merged` or `action_corrected`.

## Projections

These are the fitted end-to-end workflow projections under the current
implementation.


| Profile       | Mode                | Cells      | Est. wall time (h) | Est. peak RSS (GB) |
| ------------- | ------------------- | ---------- | ------------------ | ------------------ |
| `default`     | backed-decompressed | 1,000,000  | 3.36               | 16.0               |
| `default`     | backed-decompressed | 5,000,000  | 22.55              | 49.2               |
| `default`     | backed-decompressed | 10,000,000 | 51.20              | 79.9               |
| `default`     | backed-decompressed | 50,000,000 | 343.53             | 246.0              |
| `default`     | in-memory           | 1,000,000  | 3.13               | 126.5              |
| `default`     | in-memory           | 5,000,000  | 22.26              | 660.6              |
| `default`     | in-memory           | 10,000,000 | 51.82              | 1346.1             |
| `default`     | in-memory           | 50,000,000 | 368.56             | 7027.1             |
| `knn_ceiling` | backed-decompressed | 1,000,000  | 1.51               | 7.7                |
| `knn_ceiling` | backed-decompressed | 5,000,000  | 7.83               | 16.3               |
| `knn_ceiling` | backed-decompressed | 10,000,000 | 15.92              | 22.4               |
| `knn_ceiling` | backed-decompressed | 50,000,000 | 82.67              | 47.2               |
| `knn_ceiling` | in-memory           | 1,000,000  | 1.41               | 117.1              |
| `knn_ceiling` | in-memory           | 5,000,000  | 7.87               | 594.8              |
| `knn_ceiling` | in-memory           | 10,000,000 | 16.51              | 1197.7             |
| `knn_ceiling` | in-memory           | 50,000,000 | 92.20              | 6083.6             |


## Practical Ceilings Under 500GB

### Full workflow


| Runtime budget | Profile       | Mode                | Runtime ceiling (M cells) | Memory ceiling (M cells) | Practical ceiling (M cells) |
| -------------- | ------------- | ------------------- | ------------------------- | ------------------------ | --------------------------- |
| 12h            | `default`     | backed-decompressed | 2.93                      | 137.92                   | 2.93                        |
| 12h            | `default`     | in-memory           | 3.01                      | 3.81                     | 3.01                        |
| 12h            | `knn_ceiling` | backed-decompressed | 7.59                      | 8218.52                  | 7.59                        |
| 12h            | `knn_ceiling` | in-memory           | 7.42                      | 4.21                     | 4.21                        |
| 24h            | `default`     | backed-decompressed | 5.27                      | 137.92                   | 5.27                        |
| 24h            | `default`     | in-memory           | 5.32                      | 3.81                     | 3.81                        |
| 24h            | `knn_ceiling` | backed-decompressed | 14.93                     | 8218.52                  | 14.93                       |
| 24h            | `knn_ceiling` | in-memory           | 14.19                     | 4.21                     | 4.21                        |
| 36h            | `default`     | backed-decompressed | 7.43                      | 137.92                   | 7.43                        |
| 36h            | `default`     | in-memory           | 7.42                      | 3.81                     | 3.81                        |
| 36h            | `knn_ceiling` | backed-decompressed | 22.19                     | 8218.52                  | 22.19                       |
| 36h            | `knn_ceiling` | in-memory           | 20.74                     | 4.21                     | 4.21                        |
| 48h            | `default`     | backed-decompressed | 9.47                      | 137.92                   | 9.47                        |
| 48h            | `default`     | in-memory           | 9.39                      | 3.81                     | 3.81                        |
| 48h            | `knn_ceiling` | backed-decompressed | 29.40                     | 8218.52                  | 29.40                       |
| 48h            | `knn_ceiling` | in-memory           | 27.14                     | 4.21                     | 4.21                        |


### `k*nn` network stage only

This is the useful way to think about the `k*nn` ceiling. `build_network()`
still materializes the representation in memory, so these numbers are
effectively in-memory-like even in backed workflows.


| Runtime budget | Representation     | Runtime ceiling (M cells) | Memory ceiling (M cells) | Practical ceiling (M cells) |
| -------------- | ------------------ | ------------------------- | ------------------------ | --------------------------- |
| 12h            | `H_stacked`        | 4.60                      | 9.41                     | 4.60                        |
| 12h            | `H_merged`         | 12.04                     | 11.93                    | 11.93                       |
| 12h            | `action_corrected` | 13.82                     | 14.53                    | 13.82                       |
| 24h            | `H_stacked`        | 7.32                      | 9.41                     | 7.32                        |
| 24h            | `H_merged`         | 18.96                     | 11.93                    | 11.93                       |
| 24h            | `action_corrected` | 22.45                     | 14.53                    | 14.53                       |
| 36h            | `H_stacked`        | 9.62                      | 9.41                     | 9.41                        |
| 36h            | `H_merged`         | 24.73                     | 11.93                    | 11.93                       |
| 36h            | `action_corrected` | 29.82                     | 14.53                    | 14.53                       |
| 48h            | `H_stacked`        | 11.67                     | 9.41                     | 9.41                        |
| 48h            | `H_merged`         | 29.87                     | 11.93                    | 11.93                       |
| 48h            | `action_corrected` | 36.47                     | 14.53                    | 14.53                       |


## What The Ceilings Mean

- **Current default `k*nn` workflow**:
backed mode rises from **2.93M** cells at `12h` to **5.27M** at `24h`,
**7.43M** at `36h`, and **9.47M** at `48h`. In-memory rises only to about
**3.81M** before the 500 GB memory ceiling takes over.
- **Current `knn_ceiling` workflow**:
backed mode rises from **7.59M** cells at `12h` to **14.93M** at `24h`,
**22.19M** at `36h`, and **29.40M** at `48h`. In-memory plateaus at about
**4.21M** cells because memory, not time, becomes the limit.
- **Practical ceiling for the `k*nn` network stage itself**:
`H_stacked` reaches **4.60M**, **7.32M**, then plateaus at about **9.41M**
cells by `36h+`; `H_merged` is already memory-limited at about **11.93M**;
`action_corrected` rises from **13.82M** at `12h` and then plateaus at about
**14.53M**.
- That still does **not** make `k*nn` a good 10M+ default. Even the compact
representations project extremely poorly at 50M, and the rest of the
pipeline still has to run.
- In other words: **500 GB does not rescue `k*nn` as the main large-scale
strategy**. It only delays the point where runtime becomes unacceptable.

## Stage-Level Interpretation

### Default profile (`k*nn`, backed) at 10M

The default profile projects to about `51.2h`. The dominant contributors are:


| Stage                  | Projected time at 10M |
| ---------------------- | --------------------- |
| `network_construction` | 40.1 h                |
| `action_decomposition` | 8.6 h                 |
| `color_computation`    | 7.7 h                 |
| `layout_2d`            | 6.4 h                 |
| `reduce_kernel`        | 5.5 h                 |


The network stage is still the overwhelming reason the default path misses the
10M target.

### `knn_ceiling` profile (backed) at 10M

The `knn` profile projects to about `15.9h`. The dominant contributors are:


| Stage                  | Projected time at 10M |
| ---------------------- | --------------------- |
| `action_decomposition` | 7.8 h                 |
| `reduce_kernel`        | 5.6 h                 |
| `color_computation`    | 1.5 h                 |
| `layout_2d`            | 1.0 h                 |
| `network_construction` | 0.5 h                 |


Once `k*nn` is removed from the critical path, the next scale blockers are
`run_action`, `reduce_kernel`, then layout/color generation.

## Focused Sweep Takeaways

- **Compression vs decompression**:
after correcting `sparse_medium` to sparse storage, the
`reduce_kernel` sweep showed almost no runtime difference between compressed
and decompressed runs on the tested sparse datasets. The earlier concern
about decompression remains valid for dense/compressed inputs, but it was not
the dominant effect in this run.
- **Chunk size**:
`16384` increased RSS substantially without improving runtime.
`1024` to `4096` is the safe range for the current backed reduction path.
- **Thread sweep**:
at `100k`, `build_network_knn` improved from `1324s` at 1 thread to `50.9s`
at 44 threads. `build_network_k*nn` also scaled with threads, but still
remained ~3x slower than `knn`.
- **Batch sweep**:
batch correction scaled close to linearly with batch count and was not the
dominant large-scale blocker. On the `scale_full` fit, even `100` batches
projected to only about `316s` and `82.8 GB` in-memory.
- **Legacy stages**:
`marker_detection`, `annotation`, and `imputation` are still legacy
implementations. They are not the main reason the backed profiles miss 10M,
but they remain significant in-memory RSS consumers and will matter again
once `run_action` and `reduce_kernel` are optimized further.

## Bottom Line

- If the target is **10M cells**, the current default `k*nn` workflow is not a
viable production path even on a 500 GB node.
- The current `knn_ceiling` profile is the only tested route that remains
plausibly usable at multi-million scale. It still misses the suite's `12h`
target at 10M, but only by a factor of about `1.3x`, not `3x+`.
- If the target is **50M cells**, neither current profile is operationally
acceptable. `k*nn` is completely out of scope, and even `knn_ceiling`
requires major additional work in `run_action`, `reduce_kernel`, layout, and
probably downstream graph consumers.

