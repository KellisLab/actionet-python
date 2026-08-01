# Native HDF5 Backed Column-Read Investigation

Date: 2026-07-26  
Status: first conservative optimization implemented; production validation required  
Scope: backed compute reads, especially `impute_features()`

## Executive finding

The write/transfer refactor does not cover the slow path reported here.
`impute_features()` follows:

`impute_features -> _core.backed_take_columns -> BackedSparseMatrixOperator::takeColumnsDense`

For CSR `/X`, the old implementation read and converted both `indices` and
`data` for every stored value, transformed every value, then discarded values
whose columns were not requested. Runtime was therefore invariant to the
number of requested features.

The production-sized atlas makes the cost explicit:

- shape: `(1,575,069, 28,856)`;
- `/X`: uncompressed contiguous CSR;
- NNZ: `5,627,020,482`;
- `data`: int64, 45.016 GB;
- `indices`: int32, 22.508 GB;
- `indptr`: int64, 12.601 MB.

Extracting ten random columns read the complete 67.54 GB sparse payload.

CSR cannot provide fast arbitrary column access without either scanning its
column-index stream or maintaining a feature-major index. Tuning CSR gathers
can improve constants and bound memory, but the route to interactive
atlas-scale feature lookup is CSC storage (primary analysis layout or a
validated mirror).

## Source safety

All benchmarks opened the sources read-only and verified inode, size, and
mtime after each completed run.

| Source | Fingerprint `(inode, size, mtime_ns)` |
|---|---|
| `/data/adatas/adata_agg_ALL_pass132_post.h5ad` | `(108003329, 88072042312, 1785040579810668286)` |
| `/data/tau_project/tmp/adata_agg_ALL_pass132_post.h5ad` | `(80412761, 88072042312, 1785012869170401100)` |

One deliberately scheduled cold-cache prototype used read-only
`POSIX_FADV_DONTNEED`. It was stopped when 4 KiB point-read latency proved
non-competitive. No H5AD content or metadata was modified.

## Atlas measurements

All extraction comparisons used columns:

`[2479, 22327, 2574, 18883, 12661, 12492, 20121, 2717, 5813, 24772]`

The exact dense-output checksum was `8,178,247`.

| Implementation / cache state | Take wall | Logical read | Physical read | `syscr` | Peak RSS | User CPU |
|---|---:|---:|---:|---:|---:|---:|
| Old full scan, cold | 182.10 s | 67.54 GB | 66.75 GB | 86,098 | 1.270 GB | 1,661.5 s |
| Old full scan, warm | 65.41 s | 67.54 GB | 0.021 GB | 86,098 | 1.269 GB | 1,645.6 s |
| 4 KiB point prototype, warm | 88.06 s | 32.35 GB | 0 | 2,398,145 | 0.448 GB | 83.8 s |
| Conservative sequential planner, partially cold | 147.54 s | 67.55 GB | 10.06 GB | 43,499 | 0.445 GB | 103.5 s |
| Conservative planner + 8-thread index match, warm | 70.15 s | 67.55 GB | 0 | 43,510 | 0.453 GB | 527.2 s |

The 4 KiB prototype demonstrated why byte-count-only planning is unsafe:
HDF5 point selection reduced logical bytes by 52%, but generated 2.4 million
reads. A deliberately cold run was slower than the sequential baseline and
was stopped. The implemented cost policy therefore permits point reads only
when selected values touch at most 5% of the source data blocks.

Additional implemented-path probes:

- one requested column: 28.43 s warm, 24.16 GB logical read, 311,806 read
  calls, 322 MB peak RSS, checksum `1,331,068`;
- seven arbitrary/duplicate rows by three columns: 6.6 ms after operator open,
  25.38 MB total logical read including `indptr`/inspection, exact duplicate
  row parity.

The ten-column warm-cache path is 7% slower than the old highly parallel full
scan but uses 2.8x less peak memory and about 3.1x less aggregate CPU. The
partially cold observation was 19% faster. A fully cold sequential run should
be treated as storage-bound until it is repeated on deployed filesystems.

## Implemented changes

`BackedSparseMatrixOperator::takeColumnsDense` now:

1. validates row and column bounds before indexing native buffers;
2. reads CSR indices in their compact 32- or 64-bit signed/unsigned width;
3. caps parallel in-memory index matching at eight threads;
4. transforms only selected values;
5. estimates touched HDF5 data blocks;
6. uses point reads only below a conservative 5% touched-block threshold;
7. otherwise performs one sequential value read per bounded block;
8. applies the existing NNZ-targeted byte cap instead of the former raw
   rows-per-chunk cap;
9. reads small explicit row selections directly from their `indptr` ranges;
10. preserves reordered and duplicate rows and columns.

Dense and sparse backed column APIs now reject out-of-range selectors rather
than risking native out-of-bounds access. CSC dense extraction now restores
duplicate row selections correctly.

Reusable read-only benchmark:

`tests/benchmark_backed_take_columns.py`

## Other bottlenecks on the same workflow

### P0: feature-major expression layout

A ten-feature CSR query must still inspect 22.5 GB of column indices on this
atlas. At 10M observations with similar density, this remains a minutes-scale
operation even after constant-factor optimization.

Add a native atomic orientation conversion/repack operation:

- CSR for row-centric ingestion, QC, and repeated observation filtering;
- CSC for analysis/visualization workloads dominated by feature lookup;
- optionally retain both only when storage budget justifies it;
- if a CSC mirror is used, store source identity/shape/NNZ/version metadata
  and invalidate or update it in every structural rewrite/transform path.

The converter must be external-memory/bounded and must not materialize the
atlas in SciPy. A native temporary output followed by the existing durable
publication transaction is the correct boundary.

### P0: graph loading and diffusion copies

AnnData 0.12 backed mode backs only `X`; `obsp` is eagerly materialized by
`read_h5ad_backed`. The atlas `obsp/actionet` graph is already:

- NNZ: `758,206,624`;
- data: float64, 6.066 GB;
- indices: int32, 3.033 GB;
- total sparse payload: about 9.1 GB.

`compute_network_diffusion` then:

1. calls SciPy `tocsc()` on the CSR graph;
2. force-casts indices to platform-width integers;
3. copies values and 64-bit indices into Armadillo CSC;
4. copies the graph again into `PreparedGraph::Gn`.

This can exceed 30 GB of live graph storage on the current atlas and scales
beyond a 172 GiB node near 10M observations.

Implement a native CSR diffusion kernel that borrows validated SciPy CSR
buffers or streams a backed HDF5 CSR graph. Normalize with row/column scale
vectors and apply scaling during CSR SpMM instead of copying a normalized
graph. Preserve a prepared graph object across feature batches/calls.

The network builder already produces float32 `CSRGraph` weights, but the
Python conversion widens them to float64. Keeping graph weights float32 after
numerical validation would save another 3.0 GB on the current atlas.

### P1: checked dtype compaction

`/X/data` is int64. If a sequential validation pass proves values fit int32,
an opt-in exact downcast would reduce the current expression payload by
22.5 GB. This is more portable than adding a mandatory HDF5 compression
plugin and preserves fast contiguous reads.

### P1: selected aggregations and dense storage

`MatrixSource.col_sums`, `nnz_col_counts`, and row aggregations restricted to
selected columns still slice row chunks through AnnData/SciPy and can rescan
the full CSR payload. Reuse the native selected-column scanner for aggregation
without materializing `n_obs x n_selected_features`.

Dense `takeColumnsDense` still reads full-width row slabs. Add a layout-aware
selected-hyperslab planner, but require chunk-aligned evidence: contiguous
row-major dense data is intrinsically hostile to arbitrary column lookup.

### P1: bounded imputation output

At 10M observations, each float64 feature is 80 MB. Ten features are 0.8 GB
per dense buffer, and diffusion currently creates several copies. Add feature
batching around a reusable prepared graph and optionally validate a float32
result mode.

### P2: compute-read profiling

The transfer writer has `TransferStats`; compute reads do not expose equivalent
evidence. Add profiled column-read results with:

- compact index bytes read;
- selected values and touched storage blocks;
- logical/estimated physical data bytes;
- HDF5 call counts;
- planning/index scan/data read/transform/scatter time;
- peak native buffers;
- selected planner (`points`, `sequential`, `CSC-native`).

This is required to tune deployed NVMe, Lustre, and network storage without
using wall time alone.

## Validation still required

1. Repeat 1/10/100/1000-column tests with controlled cold and warm cache on
   each deployed storage class.
2. Test common, rare, contiguous, random, duplicate, and reordered features.
3. Benchmark lazy row scaling/log transforms.
4. Run the full Python and native suites under AnnData 0.12 and 0.13.
5. Add performance gates for selector parity, peak RSS, and warm/cold
   non-regression.
6. Implement and benchmark the native CSR diffusion path before claiming
   10M-observation `impute_features()` readiness.
