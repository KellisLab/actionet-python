# ACTIONet Scaling Benchmark Suite

## Purpose

Identify and prioritize performance and memory bottlenecks across the full
ACTIONet workflow at massive data scale. The primary goal is to find the
practical ceiling of each workflow stage — especially `k*nn` network
construction, `reduce_kernel`, and the three legacy-unoptimized stages
(`find_markers`, `annotate_cells`, `impute_features`) — and to produce
stage-level scaling models that can guide future optimization work.

---

## Datasets

### Source files (in `data/`)

| Handle | File | Shape | Storage | Batches | Labels |
|---|---|---|---|---|---|
| `small_full` | `test_adata.h5ad` | 6,790 × 32,236 | sparse | none | `.obs['CellLabel']` |
| `sparse_medium` | `adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad` | 41,007 × 36,377 | **see note** | 25 `UID` | `.obs['CellType']` |
| `scale_full` | `adata_agg_Scn4b_OX_fil.h5ad` | 300,157 × 31,728 | sparse uncompressed | 26 `UID` | `.obs['CellType']` |

> **Dataset correction — `sparse_medium`**: This file was saved with a dense
> backing store, which is incorrect. Before any benchmark trial, run the
> one-time conversion below. Do not repeat this per trial.
>
> ```python
> import anndata, scipy.sparse
> adata = anndata.read_h5ad("data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad")
> if not scipy.sparse.issparse(adata.X):
>     adata.X = scipy.sparse.csr_matrix(adata.X)
> adata.write_h5ad("data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad")
> ```
>
> The handle is renamed `sparse_medium` (was `dense_full`) throughout this
> document to reflect the corrected format.

### Stratified subsets from `scale_full`

Build deterministic subsets at the following cell counts. Stratify by `UID`,
use fixed seed `42`, and drop any sample that would contribute fewer than `100`
cells at that tier.

Tiers: `25k`, `50k`, `100k`, `150k`, `200k`, `250k`, `300k` (full)

Handle naming: `scale_subset_<tier>` (e.g., `scale_subset_100k`).

No derived subsets are committed to the repository; they are generated into the
benchmark output directory or temp space at run time.

### Dataset metadata used per dataset

| Handle | batch_key | label_key | features_key | batch_correction |
|---|---|---|---|---|
| `small_full` | — | `CellLabel` | `Gene` | no |
| `sparse_medium` | `UID` | `CellType` | `Gene` | yes |
| `scale_subset_*` / `scale_full` | `UID` | `CellType` | `Gene` | yes |

---

## Workflow

The full benchmark workflow runs the following stages **in order**. Both
`wall_s` and `peak_rss_mb` are recorded individually for every stage and for
the total. The result schema requires a separate row per stage per trial.

| # | Stage handle | Function | Notes |
|---|---|---|---|
| 1 | `filter` | `an.filter_anndata` | `min_cells_per_feat=0.01`, inplace |
| 2 | `normalize` | `an.normalize_anndata` | `target_sum=1e4`, `log_transform=True`, `log_base=2` |
| 3 | `reduce_kernel` | `an.reduce_kernel` | `svd_algorithm="halko"`, `n_components=30` |
| 4 | `batch_correction` | `an.correct_batch_effect` | skipped for `small_full` |
| 5 | `action_decomposition` | `an.run_action` | `k_min=2`, `k_max=30`; always operates on obsm — in-memory regardless of backing mode |
| 6 | `network_construction` | `an.build_network` | profile-dependent (see below) |
| 7 | `archetype_diffusion` | `an.compute_network_diffusion` | `scores="H_merged"` |
| 8 | `layout_2d` | `an.layout_network` | `n_components=2`, `key_added="umap_2d_actionet"` |
| 9 | `color_computation` | `an.compute_node_colors` | `embedding_key="umap_2d_actionet"` |
| 10 | `feature_specificity` | `an.compute_archetype_feature_specificity` | `archetype_key="archetype_footprint"` |
| 11 | `marker_detection` | `an.find_markers` | `top_genes=30`, `features_use="Gene"` — **legacy unoptimized** |
| 12 | `annotation` | `an.annotate_cells` | `method="vision"`, `features_use="Gene"` — **legacy unoptimized** |
| 13 | `imputation` | `an.impute_features` | 10 arbitrary features, `features_use="Gene"` — **legacy unoptimized** |

> **3D UMAP is excluded.** `layout_network(n_components=3)` does not provide
> additional diagnostic value for scaling analysis and is omitted from all
> workflow runs and result schemas.

> **Legacy-unoptimized stages**: `marker_detection`, `annotation`, and
> `imputation` have had no algorithmic optimizations since the original
> sub-million-cell implementation. Python-level backed streaming paths exist
> (and give speedups in backed mode), but the underlying C++ core
> implementations are unchanged. Expect disproportionately high wall time and
> RSS at large scale. Results for these stages are reported in a dedicated
> sub-section of the scaling analysis (see Reporting).

### Marker/annotation minimum-label guard

On derived subsets, use only labels with at least `50` cells at that tier. If
fewer than `2` labels remain, mark `marker_detection`, `annotation`, and
`imputation` as `skipped` for that trial.

---

## Profiles

Two full-workflow profiles are evaluated. They differ only in the
`build_network` call.

### `default` — `k*nn` (primary scaling target)

```python
an.build_network(
    algorithm="k*nn",
    obsm_key="H_stacked",
    mutual_edges_only=True,
)
```

`k*nn` uses an adaptive neighbor count `kNN = 5·√N` per query. Total query
cost is **O(N^1.5 · log N)**. This is the algorithm with the hard practical
ceiling at large N and is therefore the **primary target** of the scaling
analysis. Run tiers in ascending order; stop this profile's frontier after the
first hard failure and report all larger tiers as infeasible.

`ef` and `ef_construction` for `k*nn` are **not manually set**. The C++ core
auto-floors both at `kNN = 5·√N`, which grows with dataset size. No override
is needed.

### `knn_ceiling` — `knn` at k=100 (large-scale production path)

```python
an.build_network(
    algorithm="knn",
    k=100,
    ef=500,
    ef_construction=400,
    mutual_edges_only=True,
    obsm_key="action_corrected",   # batch-corrected datasets
    # obsm_key="action",           # small_full (no batch correction)
)
```

`knn` uses a fixed `k=100` with HNSW. Total query cost is **O(N · k · log
ef) ≈ O(N log N)**. This profile is expected to complete at all tiers. Its
purpose is to establish the absolute cost floor for a quality-appropriate
`knn` graph and to confirm there is no practical ceiling within the tested
range.

> **Why k=100?** Default `k=10` gives poor recall on high-dimensional
> single-cell data at scale. k=100 is the appropriate working value.
>
> **Why ef=500, ef_construction=400?** HNSW defaults (ef=200,
> ef_construction=200) are calibrated for small k. At k=100, recall degrades
> without raising these values. These are the baseline values for all
> `knn_ceiling` trials.

---

## Execution Parameters

### Storage mode

Primary mode for all trials: **fully decompressed backed**
(`decompress_backed_storage(scope="file")`). A compressed vs. decompressed
comparison is run on representative tiers as a focused sweep (see below).

### Thread count

Full-workflow runs: `44` threads (set via env vars and `n_threads` argument in
all supporting functions).

### Trial repetition

| Tier | Trials |
|---|---|
| ≤ 50k | 3 |
| 100k – 200k | 2 |
| ≥ 250k | 1 |

### Stopping rules

- Run tiers in ascending order per profile.
- Stop a profile's frontier after the first hard failure (wall time > 12h or
  peak RSS ≥ 150 GB). Report all larger tiers as infeasible.
- **Per-stage failure**: if a single stage exceeds 150 GB RSS or 4h wall time
  alone, mark that stage as failed. Continue remaining stages only if they do
  not depend on the failed stage's output. Record `status=stage_failed` and
  `failure_reason` in the result row for that stage.
- `knn_ceiling` profile: run all tiers regardless of `default` frontier
  results.

### Process isolation

Run every trial in a fresh subprocess. The parent process enforces:
- 12h timeout
- 150 GB RSS kill limit
- Resume-safe persistence after every completed case (write result row to JSONL
  before starting the next case)

---

## Focused Sweeps

### 1. `build_network` — size frontier

Run both `k*nn` and `knn` independently across all `scale_subset_*` tiers and
`scale_full`. Record `wall_s` and `peak_rss_mb` for each tier. Report as
independent frontier curves, not a side-by-side comparison.

Representations to test:

| `obsm_key` | Available when |
|---|---|
| `H_stacked` | all datasets |
| `H_merged` | all datasets |
| `action_corrected` | batch-corrected datasets |
| `action` | `small_full` and as fallback |

For `knn`: `k=100`, `ef=500`, `ef_construction=400` throughout.

For `k*nn`: no ef override (auto-floored by C++).

### 2. `build_network` — thread sweep

Run on `scale_subset_100k`. Thread counts: `1`, `8`, `16`, `44`.

Cover all three parallelized stages:
- `build_network` with `k*nn` (primary production case)
- `build_network` with `knn` at `k=100`
- `run_action`
- `layout_network` (2D UMAP)

### 3. `build_network` — `knn` ef sweep

Run on `scale_subset_100k` with `k=100`. Test `ef` in `{200, 350, 500, 750}`.
`ef_construction` fixed at `400`. Records recall (via ground-truth comparison
against `k*nn` on the same data) and runtime to characterize the recall vs.
cost trade-off and validate the `ef=500` baseline.

### 4. `reduce_kernel` — storage and chunk-size sweep

- Halko (5 power iterations), `n_components=30`. Fixed pass count ensures
  I/O cost is proportional to NNZ, giving a clean signal for the backed vs.
  in-memory comparison and a smooth power-law scaling model. PRIMME's
  adaptive convergence would introduce iteration-count variance that
  contaminates the storage-layer overhead measurement.
- In-memory vs. backed-decompressed on all tiers.
- Compressed vs. decompressed on `sparse_medium` and `scale_subset_100k`.
  Note: after conversion of `sparse_medium` to sparse format, the
  compressed-vs-decompressed comparison must be re-validated; compressed sparse
  and compressed dense have different read patterns and the old result does not
  transfer.
- Chunk-size sweep `{1024, 4096, 16384}` on `sparse_medium` and
  `scale_subset_100k` in backed-decompressed mode.

### 5. `correct_batch_effect` — batch-count sweep

Run on `sparse_medium`, `scale_subset_100k`, and `scale_full`. Evaluate
merged batch labels at `{1, 5, 10, 20, all}` observed batches. Fit a
batch-augmented scaling model. Extrapolate to `25`, `50`, and `100` batches.

### 6. `run_action` — k_max sweep

`run_action` operates entirely on `adata.obsm` and is always in-memory
regardless of backing mode. Run on `scale_subset_50k`. Sweep `k_max` in
`{15, 30, 50}` at fixed cell count to isolate the k_max contribution from the
cell-count contribution in the scaling model.

---

## Result Schema

Every completed trial (one stage of one case) produces one JSON row with the
following fields:

```
case_id            str    unique identifier for this trial
dataset            str    handle (e.g., "sparse_medium", "scale_subset_100k")
tier               int    cell count (e.g., 100000; null for non-subset datasets)
profile            str    "default" | "knn_ceiling"
mode               str    "in_memory" | "backed_decompressed" | "backed_compressed"
stage              str    stage handle (see Workflow table; "total" for full run)
params             dict   stage-specific params (k, ef, k_max, chunk_size, ...)
n_obs              int
n_vars             int
nnz                int
n_batches          int    0 for datasets without batch correction
representation_dim int    n_components used for reduce_kernel
execution_kind     str    "in_memory" | "backed_streamed" | "backed_materialized"
wall_s             float  wall-clock seconds for this stage
peak_rss_mb        float  peak RSS increase during this stage (psutil sampling)
io_read_mb         float
io_write_mb        float
status             str    "ok" | "oom" | "timeout" | "stage_failed" | "skipped"
failure_reason     str    null if status=="ok"
```

`wall_s` and `peak_rss_mb` are **mandatory** for every individual stage row,
not only the `total` row.

---

## Outputs

Write all outputs under `tests/benchmark_results/<run_id>/`:

```
tests/benchmark_results/<run_id>/
  raw/          one JSONL file per case
  summary.csv   aggregated table (one row per stage × trial)
  report.md     Markdown summary with tables and embedded plot references
  plots/        PNG files
```

---

## Implementation

### Files

| File | Role |
|---|---|
| `tests/benchmark_backed_extension.py` | CLI orchestrator (refactored) |
| `tests/benchmark_support.py` | helper module (new) |
| `tests/BENCHMARK_README_DRAFT.md` | runbook (to be finalized) |

### CLI (orchestrator)

```
python benchmark_backed_extension.py \
  --suite {all,workflow,network,reduction,batch} \
  --datasets small_full sparse_medium scale_full scale_subset_100k ... \
  --profiles default knn_ceiling \
  --modes backed_decompressed in_memory \
  --output-dir tests/benchmark_results/run_001 \
  --resume \
  --max-tier 200k
```

### Helper module responsibilities

- Dataset manifests and subset generation (stratified by `UID`, seed `42`,
  min-cells-per-sample `100`)
- Child-process case dispatch
- Parent monitoring (timeout, RSS kill)
- Per-stage metrics collection (`psutil` sampling thread)
- Scaling-model fitting
- Report and plot generation

### Dev dependencies (benchmark-only)

`psutil`, `matplotlib`, `tabulate`

---

## Test Plan

### Subset generation

- Fixed seed `42` reproduces identical subsets across runs.
- Samples below `100` cells are dropped per tier.
- Retained cell counts sum to the requested tier size (within dropped-sample
  tolerance).

### Orchestration

- Manifest expansion enumerates the correct set of cases.
- `--resume` skips all cases with an existing `status=ok` row in the output
  JSONL.
- Timeout and RSS-limit failures are classified and recorded with correct
  `status` and `failure_reason`.

### Reporting

- Report generation works from canned raw JSONL inputs.
- Partial frontiers and failed stages do not break aggregation or plotting.

### Smoke path

Run only `small_full` with `1` trial in in-memory and backed-decompressed
modes. This validates the harness end-to-end in under a few minutes.

---

## Assumptions

- Python frontend only; R is out of scope.
- Benchmark execution uses
  `/home/sebastian/data/git_projects/actionet-python/.venv/bin/python`.
- No large derived subsets or temporary copies are committed to the repository.
- `sparse_medium` one-time sparse conversion is performed before the first run
  and does not need to be repeated.

---

## Scaling Analysis and Reporting

### Per-stage scaling models

Fit a power-law model `T(N) = a · N^b` to wall time and peak RSS for each
stage across tiers. Report fit coefficients and predicted values at `500k`,
`1M`, `5M`, and `10M` cells.

Exceptions:
- **`correct_batch_effect`**: use a batch-augmented model `T(N, B) = a · N^b
  · B^c` fit from the batch-count sweep.
- **`build_network`**: fit separate models per algorithm (`k*nn`, `knn`) and
  per `obsm_key`. Report `k*nn` model as `O(N^1.5)` reference and annotate
  deviations.
- **`archetype_diffusion`**: note that cost scales with edge count, which is a
  function of both `N` and the network algorithm. Report alongside
  `network_construction`.

### Scaling ceiling definitions

The report defines two ceilings:

| Ceiling | Definition |
|---|---|
| `default_ceiling` | Largest tier where `default` profile completes within 12h and < 150 GB RSS |
| `knn_ceiling` | Confirmed to have no ceiling within the tested range (or largest tier if unexpectedly limited) |

Extrapolate `k*nn` beyond the last successful tier using the analytical HNSW
memory lower bound for O(N · kNN) = O(N^1.5) edge scratch.

### Unoptimized stage reporting

`marker_detection`, `annotation`, and `imputation` are reported in a dedicated
sub-section with the following structure:

- Per-tier table: wall time and peak RSS at each tested tier.
- Power-law fit with explicit exponent (expected: close to O(N · n_markers) or
  O(N · n_genes) — flag deviations).
- Projected bottleneck tier: the predicted N at which the stage alone exceeds
  the 4h or 150 GB limit.
- Note on optimization status: Python-level backed streaming exists (and gives
  speedup over naive in-memory loading), but C++ core algorithmic complexity is
  unchanged from the original implementation.
