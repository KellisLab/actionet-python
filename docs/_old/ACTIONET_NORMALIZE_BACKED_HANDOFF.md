# Handoff: Avoid OOM in Backed `normalize_anndata(layer_added=...)`

## Summary

This note is for implementation in the `actionet-python` repo at:

- `/data/git_projects/actionet-python`

Observed from the calling pipeline in this repo:

- [`preprocessing/actionet_init_pass.py`](/home/sebastian/data/tau_project/preprocessing/actionet_init_pass.py)

The backed `normalize_anndata()` path behaves as expected at the chunk level, but the current `layer_added=...` implementation still triggers very large whole-file I/O and memory pressure. On the production dataset, that pressure is enough to cause OOM during normalization.

The fix is to replace the current backed `layer_added` copy-and-rewrite path with a direct streamed writer that creates the destination layer in final form and writes normalized chunks into it.

## Observed Failure

In the current pipeline, this line is the failing step:

- [`preprocessing/actionet_init_pass.py:141`](/home/sebastian/data/tau_project/preprocessing/actionet_init_pass.py#L141)

```python
actionet.normalize_anndata(
    adata,
    target_sum=TARGET_SUM,
    log_transform=True,
    log_base=2,
    layer_added=LAYER_USE,
    inplace=True,
    backed_chunk_size=CHUNK_SIZE,
)
```

Runtime behavior reported during the failing run:

- RAM sits around `~80 GB` before normalization
- during `normalize_anndata()` it periodically spikes to `~120 GB`
- the job then OOMs

The important point is that this is inconsistent with the expected chunk-local working set of the normalization math itself.

## Production Dataset Facts

Inspected file:

- `tmp/adata_agg_init/adata_agg_CTX_init.h5ad`

Actual matrix shape:

- `1,792,201 cells x 28,692 genes`

Actual sparse payload:

- `nnz = 5,724,934,742`
- density `= 0.11133276704513663`
- sparsity `= 88.87%`

This is much sparser than the initial `~70% sparsity` assumption.

Current file state after the destination layer exists:

- `.h5ad` size: `128.12 GB`
- `X/data`: `42.65 GB` (`int64`)
- `X/indices`: `21.33 GB` (`int32`)
- `layers/logcounts/data`: `42.65 GB` (`int64`)
- `layers/logcounts/indices`: `21.33 GB` (`int32`)

## Root Cause

The chunked normalization code is not the main problem. The problem is the backed `layer_added` path in `actionet-python`:

- `src/actionet/preprocessing.py`

Current behavior:

1. `normalize_anndata(..., layer_added=...)` copies the full source matrix into `layers[layer_added]`
2. the file handle is refreshed
3. normalization then runs chunk-by-chunk on that copied layer
4. the first write sees an on-disk dtype mismatch (`int64` source copy vs `float32` normalized output)
5. ACTIONet recasts the entire copied `data` dataset before writing the first normalized chunk

Relevant locations in `actionet-python`:

- `src/actionet/preprocessing.py:146`
- `src/actionet/preprocessing.py:153`
- `src/actionet/preprocessing.py:160`
- `src/actionet/preprocessing.py:937`
- `src/actionet/_matrix_source.py:326`
- `src/actionet/_matrix_source.py:329`

So, even though the normalization transform itself is chunked, the full workflow still includes:

- one whole sparse-group copy
- one full row-sum scan
- one full chunked read of the copied layer
- one full `int64 -> float32` rewrite of the copied `data` dataset

This explains why the code path looks memory-efficient in isolation but still behaves like a large-memory event under job accounting.

## Why The OOM Happens

The likely OOM mechanism is:

- very large HDF5 reads and writes
- dirty page accumulation during whole-file copy / rewrite
- page cache and writeback being charged to the job or cgroup memory limit

This is consistent with:

- the estimated Python working set staying sub-1-GB
- the observed machine/job memory jumping by tens of GB

There is no evidence that `adata.X[start:end, :]` is materializing the full matrix for the row-slice pattern used here. AnnData's backed CSR row-slice path appears to use the sparse override as intended.

## Proposed Change

Replace the backed `layer_added` copy-and-rewrite path with a direct streamed destination-layer writer.

### Desired behavior

When `source.is_backed` and `layer_added is not None`:

- do **not** call `h5file.copy()` on the source sparse group
- do **not** normalize by mutating the copied layer through `MatrixSource.set_rows()`
- instead, create `layers[layer_added]` directly in the final output layout
- compute row sums from the source matrix
- stream source rows in chunks
- normalize each chunk in memory
- write the normalized chunk into the destination layer directly

### Sparse-backed destination

For sparse CSR-backed input:

- create `layers/<layer_added>/data` directly as `dtype_out`
- reuse the source sparse structure
- hard-link `indices` and `indptr` from the source sparse group instead of copying them
- copy sparse-group attrs such as:
  - `shape`
  - `encoding-type`
  - `encoding-version`

This is valid because total-count normalization plus `log1p` preserves the sparse pattern for non-negative count data.

### Dense-backed destination

For dense-backed input:

- create a dense dataset of shape `(n_obs, n_vars)` and dtype `dtype_out`
- stream normalized blocks into it

## Why Hard-Linking Is Viable

A small local proof-of-concept was tested in `.venv_tau`:

- an HDF5 layer with its own `data` dataset and hard-linked `indices`/`indptr`
- AnnData could read that layer normally
- a subsequent `write_h5ad()` to a new file produced a valid standalone layer

So the hard-link approach is compatible with:

- backed reads
- in-memory reads after reopen
- rewrite to a new `.h5ad`

## Expected Memory And Performance Impact

### Current path on the production file

Approximate total I/O touched during `normalize_anndata()`:

- `~341 GiB`

Approximate writes during the step:

- `~107 GiB`

Approximate chunk-local process working set:

- typical: `~0.52 GiB`
- worst observed chunk upper bound: `~0.88 GiB`

### Proposed streamed `layer_added` path

Approximate total I/O:

- `~149 GiB`

Approximate writes:

- `~21.3 GiB`

Approximate process-local peak:

- `~0.30 to 0.55 GiB`

### Practical impact

Expected improvement on this workload:

- total I/O reduced by about `56%`
- writes reduced by about `80%`
- file high-water reduced from roughly `128-149 GB` to roughly `85 GB`
- chunk-local process memory reduced by about `30-45%`
- wall time for this step likely improves by about `1.8x to 2.3x` if storage is the bottleneck

The most important outcome is not the small process-memory reduction. It is the removal of the large whole-file write amplification that likely causes the job-level OOM.

## Suggested Implementation Outline

Target file:

- `src/actionet/preprocessing.py`

Suggested changes:

1. Keep the in-memory `layer_added` path unchanged.
2. Keep the backed in-place overwrite path (`layer_added is None`) unchanged.
3. Replace the backed `layer_added` path with a new streamed destination writer.
4. Add a helper that:
   - resolves the source HDF5 sparse group
   - creates `layers/<layer_added>` from scratch
   - for sparse input, hard-links `indices` and `indptr`
   - creates `data` as `dtype_out`
5. Add a normalization helper that:
   - computes `row_sums`
   - iterates source chunks
   - normalizes each chunk
   - writes only the corresponding `data` slice to the destination
6. Avoid `MatrixSource.set_rows()` for this path so `_h5py_cast_data_dataset()` is never needed.

## Test Plan

Existing backed tests already cover the main semantics and should be kept:

- `tests/backed/test_backed_extension.py`

Add or update tests for:

1. backed `.X -> layer_added` normalization preserves `.X`
2. backed `layer -> layer_added` normalization preserves the source layer
3. output matches the in-memory normalized result
4. existing destination layer is replaced correctly
5. result persists after close/reopen
6. file can still be re-written via `write_h5ad()` and reopened successfully
7. HDF5-level regression for sparse-backed output:
   - `layers/<layer_added>/data.dtype == dtype_out`
   - `indices` and `indptr` are shared with the source sparse group if the hard-link implementation is used

## Assumptions

- the failing workload uses CSR-backed sparse counts
- counts are non-negative, so normalization does not change sparsity structure
- downstream code mutates sparse values, not sparse structure
- reducing `backed_chunk_size` is only a secondary mitigation; it does not address the main whole-file copy/recast problem

## Bottom Line

The OOM is not evidence that chunked normalization is fundamentally broken. It is evidence that the current backed `layer_added` implementation performs two large whole-file operations that defeat the intended memory profile.

The implementation change should be:

- from: copy full sparse matrix, then mutate copied layer chunk-by-chunk
- to: create destination layer directly and stream normalized output into it

That should preserve existing API behavior while removing the whole-file copy/recast path that is most likely responsible for the OOM.
