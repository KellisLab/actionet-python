# Backed HDF5 Subsetting Performance Handoff

Date: 2026-07-24  
Repository: `actionet-python`  
Branch/HEAD at handoff: `dev-gpu` / `3cb7d38`  
Relevant fix: `f6ebad5` (`Fix backed subset and write slowdown`)

## Purpose

This handoff records the completed investigation and the remaining optimization
for repeated curation-time subsetting of atlas-scale backed AnnData objects.
The source dataset used for measurement is:

`/data/tau_project/tmp/adata_agg_ALL_pass132_post.h5ad`

Always open this dataset read-only during investigation. Reproduction outputs
must be uniquely named temporary files on `/data`, validated before removal,
and must never replace the source.

## Current state

Commit `f6ebad5` fixed the original approximately 16.5-minute tail in
`obsp/actionet`. That graph was loaded as a SciPy CSR matrix with
`indices=int32` and `indptr=int64`. SciPy fancy row indexing promoted the
entire 758-million-element indices array to int64 on every chunk.

The implemented fix:

- normalizes SciPy sparse index dtypes once without mutating the source;
- converts in-memory CSC inputs to CSR once;
- skips identity column selections;
- writes coherent `indices`/`indptr` dtypes when safe;
- adds private per-chunk, per-component, flush, and close profiling.

The full repository suite passed after that change. A same-filesystem
near-identity rewrite produced an 88.05 GB output with these measurements:

| Stage | Time |
|---|---:|
| Entire filtered write | 157.28 s |
| `X` total | 123.92 s |
| `X` source reads | 87.50 s |
| `X` destination writes | 35.94 s |
| `obsp/actionet` total | 23.86 s |
| `obsp/actionet` source reads | 1.44 s |
| `obsp/actionet` column selection | 15.27 s |
| `obsp/actionet` destination writes | 7.02 s |
| HDF5 close | 1.26 s |

`obsp/actionet` therefore improved from roughly 9–10 MB/s to approximately
380 MB/s. The output reopened correctly, the temporary file was deleted, and
the source inode/size/mtime remained unchanged.

At the time of this handoff, after a user-driven two-observation curation
rewrite, the source snapshot was:

- shape: `(1,575,069, 28,856)`;
- file size: `88,072,042,312` bytes;
- `X.nnz`: `5,627,020,482`;
- `obsp/actionet.nnz`: `758,206,624`;
- `obsp/actionet/indices`: int32;
- `obsp/actionet/indptr`: int32.

The dataset may continue to change through user curation, so re-inventory it
before future performance runs.

## Why destination size gives misleading progress

The current `X` storage is uncompressed contiguous CSR:

| Dataset | Logical bytes | Dtype | File offset |
|---|---:|---|---:|
| `X/data` | 45,016,163,856 | int64 | 8,192 |
| `X/indices` | 22,508,081,928 | int32 | 45,016,172,048 |
| `X/indptr` | 12,600,560 | int64 | 67,524,253,976 |

The apparently instantaneous first 45 GB is exactly the `X/data` extent.
HDF5 extends the logical file when a contiguous dataset is first touched;
logical size can therefore jump long before those bytes are physically
written. A local HDF5 check showed that writing 8 KB into a newly created
256 MB contiguous dataset immediately increased `st_size` to 256 MB while
only about 70 KB of filesystem blocks had been allocated.

The host has about 181 GB RAM and Linux dirty-page throttling at 20%
(approximately 36 GB). Page-cache buffering and subsequent writeback further
explain the burst/slow/steady pattern. Use writer profiling, `du`, and process
I/O counters rather than apparent HDF5 file size to identify the active stage.

## Remaining source-read bottleneck

The current sparse writer reads each block with:

```python
rows = obs_idx[pos:end]
block = source_mat[rows, :]
```

For an AnnData backed CSR dataset, an integer row array routes through
AnnData's `get_compressed_vectors()`. That function performs one HDF5 slice
of `data` and one of `indices` for every selected row. On this object, a
near-identity `X` rewrite therefore performs roughly 3.15 million small HDF5
reads.

AnnData's contiguous-slice path instead uses
`_get_contiguous_compressed_slice()` and performs one bulk `data` read plus
one bulk `indices` read per block.

Read-only measurements using 362-row blocks at six positions across `X`:

| Selector | Typical time |
|---|---:|
| Integer row array | 11–14 ms (21 ms on the first cold sample) |
| Equivalent row slice | 5–7 ms |

The observed steady-state improvement was about 2.3x. This suggests that a
simple contiguous fast path could save approximately 45–55 seconds from the
recorded 87.5-second `X` source-read stage.

## Curation workload changes the design

The real curation workflow repeatedly removes observations whose positions
and quantity are arbitrary and approximately uniformly distributed. Retained
rows remain in source order when selection is expressed as a boolean mask or
sorted integer indices.

A fast path that converts an entire output chunk to one slice is sufficient
only when very few observations are removed. For the observed adaptive sparse
chunk size of 362 retained rows, the probability that a chunk contains no
randomly distributed deletion is approximately:

`P(contiguous chunk) = (1 - p) ** 361`

| Removal fraction per pass | Fully contiguous chunks |
|---:|---:|
| 0.01% | 96% |
| 0.1% | 70% |
| 0.5% | 16% |
| 1% | 2.7% |
| 2% | 0.07% |
| 5% | effectively none |

Therefore, the previously proposed single-slice-per-chunk optimization is not
adequate by itself for general curation.

Repeated physical compaction is also fundamentally expensive: every pass
rewrites nearly the entire surviving file. Writer optimization can reduce
CPU and HDF5 call overhead, but it cannot eliminate the aggregate read/write
volume across repeated 88 GB rewrites.

## Recommended implementation

Implement the following in order.

### 1. Prefer cumulative logical filtering in curation workflows

Where pipeline semantics permit, maintain a cumulative observation mask and
apply physical `subset_anndata()` only after QC filtering converges.

- Preserve source row order in the mask.
- Recompute iterative QC metrics against the logical selection rather than
  compacting after every rule.
- Perform one atomic backed rewrite at the end.
- Treat this as a workflow/API follow-up; do not block the writer improvement
  below on it.

### 2. Add run-aware backed row reading

Replace the all-or-nothing contiguous-chunk fast path with maximal contiguous
run detection.

- Partition each normalized row-index chunk wherever `diff(rows) != 1`.
- Read every strictly ascending run with `slice(first, last + 1)`.
- Preserve run order exactly.
- Never sort or deduplicate selectors.
- Retain the existing fancy-index fallback for reordered, descending, or
  duplicate selectors.
- Bound work by the existing adaptive output chunk size; never materialize an
  entire atlas-scale run at once.

For a uniform removal fraction `p`, run-aware reading reduces source calls
from approximately one pair per retained row to one pair per retained run.
At 1% removal this is roughly 15,600 runs instead of 1.56 million rows; at
5%, roughly 75,000 runs instead of 1.50 million rows.

### 3. Use direct CSR streaming when columns are unchanged

For a backed CSR source with `var_idx is None`, bypass AnnData and SciPy row
indexing entirely.

1. Read/cache the source `indptr` once (about 12.6 MB for this object).
2. For each retained row run within the bounded output chunk:
   - translate the row run to `[indptr[row_start], indptr[row_stop])`;
   - read the corresponding contiguous `data` and `indices` ranges directly
     from the source HDF5 group;
   - write those ranges directly into the destination datasets;
   - append output row counts from `diff(source_indptr[row_start:row_stop+1])`.
3. Build the destination `indptr` cumulatively as the current writer does.
4. Preserve the existing safe output index-dtype selection and compression
   policy.

This direct path should handle `X`, identity-column layers, and other backed
CSR components. It avoids temporary SciPy matrices and is the preferred path
for the dominant 67.5 GB `X` component.

When columns are subsetted, read each row run as a bounded CSR block, apply
the existing column selection, and feed the result to the existing writer.
Keep the current normalized in-memory SciPy path for `obsp`/`varp`; it is no
longer the dominant bottleneck.

Apply the same maximal-run selector concept to backed dense matrices, using
slice reads and bounded concatenation/writes. Do not change public chunk-size
defaults or combine this work with the existing read/write chunk-decoupling
TODO.

## Profiling changes

Extend the private profiling schema to make dispatch visible:

- chunk fields:
  - `row_selector_kind`: `direct_csr`, `slice_runs`, or `indices`;
  - `row_run_count`;
  - existing source-read, selection, conversion, and destination-write times;
- component totals:
  - counts of direct-CSR, slice-run, and fancy-index chunks;
  - total row runs;
  - source and destination bytes where practical.

Profiling must remain opt-in. Normal writes must retain their current HDF5
flush behavior.

## Correctness tests

Add focused tests in `tests/test_subset_anndata.py` covering:

- contiguous, singleton, gapped, and multiple-run selectors;
- random sorted masks at approximately 0.1%, 1%, 5%, and 20% removal;
- gaps at chunk boundaries and inside chunks;
- reordered, descending, and duplicate selectors using the unchanged
  fallback;
- empty rows at the private-helper level, while preserving the public
  rejection of empty AnnData outputs;
- backed CSR with identity columns through the direct path;
- backed CSR with column subsetting through run-block conversion;
- backed dense matrices;
- data dtype, sparse orientation, coherent index dtype, row/column order,
  duplicate semantics, metadata, raw, layers, and pairwise shapes;
- profiling dispatch fields and run counts;
- failure cleanup and atomic replacement invariants.

Compare every optimized output against ordinary SciPy/NumPy selection on
small deterministic fixtures. Run focused subset tests, all backed tests,
Ruff, and the complete repository test suite.

## Performance validation

Use a temporary output on the same `/data` filesystem and keep the atlas
source read-only.

1. Re-inventory the source because curation may have changed it.
2. Benchmark deterministic uniformly distributed masks at several removal
   fractions, including two rows, 0.1%, 1%, and 5%.
3. Record per-component profiling rather than inferring progress from file
   size.
4. Reopen and validate each output, then delete it.
5. Verify source inode, size, and mtime remain unchanged.

Acceptance targets:

- exact output parity for every selector class;
- no regression in the fixed `obsp/actionet` path;
- direct/slice block reads at least 1.5x faster than integer-array reads on
  the atlas;
- at least 25% reduction from the recorded 87.5-second `X` source-read stage
  for near-identity selection;
- material reduction in HDF5 read-call count for 1–5% uniformly distributed
  removal, even when total disk throughput is cache-sensitive.

Timing thresholds are benchmark gates, not CI tests.

## Explicit non-goals

- Do not tune kernel dirty-page settings.
- Do not infer progress from apparent destination size.
- Do not add compression or change HDF5 layout as part of this optimization.
- Do not change public chunk defaults.
- Do not combine this change with read/write chunk-parameter decoupling.
- Do not weaken atomic replacement or temporary-file cleanup.
- Do not modify the source dataset during development or benchmarking.

