# Lazy HDF5 Feature Cache: Implementation Draft

Date: 2026-07-26  
Status: proposed; revised to preserve the AnnData-only public model  
Scope: backed CSR feature extraction only  
Public data model: AnnData remains the sole Python data model

## Decision

Do not introduce an ACTIONet-specific dataset container.

Keep the standard H5AD file and its AnnData object authoritative. Add a
transparent, optional, rebuildable cache for the narrow pathological
operation: extracting a small number of columns from a very large backed CSR
matrix.

The cache is:

- derived entirely from one H5AD matrix path;
- never authoritative;
- safe to delete at any time;
- not required to open or understand the dataset;
- not serialized through `adata.uns`;
- invisible to AnnData, Scanpy, and other ecosystem tools;
- shared by Python and, later, R through the native path-based implementation.

This preserves the interoperability decision behind the Python/R rebuild.
Losing or moving the cache changes performance only, never data or semantics.

## Is a lazy CSC index reasonable?

Yes, with one important qualification.

A CSC **index alone** is insufficient. If it stores only column pointers and
offsets into CSR `/X/data`, a feature query still has to issue many scattered
value reads against the CSR payload. The production point-read prototype
already demonstrated this failure mode:

- one ten-column query generated 2.4 million HDF5 reads;
- a one-column query still performed 311,806 reads and took 28.43 seconds
  warm;
- reducing logical bytes did not make the access interactive.

For a cache hit to be trivial, each cached feature must contain both:

- the observation row IDs containing stored entries;
- the corresponding raw values in row order.

That is a feature posting list, equivalent to one materialized CSC column.
The important optimization is to materialize only requested columns, not a
second copy of the entire matrix.

The proposed design is therefore a **lazy feature posting cache**, not a
universal dual-orientation store.

## Why this matches the actual workload

The current code has two different column-oriented workloads.

### Variable filtering

`compute_filter_masks()` streams CSR row chunks and computes row sums, row
NNZ, and column NNZ together in one pass per iteration. This is a deliberate
sequential analytics pass. It is expensive at 10M observations, but it is
bandwidth-efficient and normally occurs during preprocessing rather than
interactive exploration.

It does not require arbitrary CSC column lookup. The existing native transfer
path can continue to perform the eventual physical subset.

### Feature extraction

Both interactive plotting and expression-based imputation converge on:

```text
plot_feature_expression / impute_features
  -> open_backed_operator_for
  -> _core.backed_take_columns
  -> BackedSparseMatrixOperator::takeColumnsDense
```

For CSR, this path scans the global column-index stream to find a handful of
features. This is the pathological latency users experience.

The cache should be designed only around this call path first. A broad
storage abstraction is unnecessary.

## User-visible behavior

The public object remains an AnnData:

```python
adata = anndata.read_h5ad("atlas.h5ad", backed="r")

# First uncached call uses the existing CSR scan and retains the selected
# raw feature postings as a disposable cache.
actionet.plot_feature_expression(
    adata,
    ["MS4A1", "CD3D", "LYZ"],
    feature_cache="auto",
)

# Reuses cached row/value postings. No full CSR scan.
actionet.impute_features(
    adata,
    ["MS4A1", "CD3D"],
    feature_cache="auto",
)
```

Users may explicitly prewarm known panels:

```python
actionet.prepare_feature_cache(
    adata,
    features=marker_panel,
    layer=None,
    cache_dir="/scratch/me/actionet-cache",
)
```

And inspect or remove the cache:

```python
actionet.feature_cache_info(adata, layer=None)
actionet.clear_feature_cache(adata, layer=None)
```

No new object type is returned. No existing AnnData method changes.

## Cache policy

Add a common keyword to feature-extraction entry points:

```text
feature_cache = "auto" | "read" | "refresh" | "off"
```

- `auto`: use valid hits; on a miss, run the normal CSR extraction and
  persist the missing features during the same scan when possible;
- `read`: use valid hits but do not create new entries;
- `refresh`: rebuild requested entries even if a valid hit exists;
- `off`: bypass all cache discovery and writes.

`auto` never builds a full CSC mirror and never performs an additional source
scan solely to populate the cache. If cache publication fails, the requested
result still succeeds unless the caller invoked the explicit preparation API.

Cache location resolution:

1. explicit `cache_dir`;
2. configured ACTIONet feature-cache directory;
3. a writable sibling cache for a file-backed H5AD;
4. cache disabled with one actionable warning if no safe location exists.

A configurable byte cap prevents unbounded growth. Reaching the cap stops new
cache writes; automatic deletion is not required in version one.

## Cache artifact

An example cache is:

```text
atlas.h5ad.actionet-cache/
  manifest.json
  shards/
    X-4f8c...h5
    layers-logcounts-a891...h5
  staging/
  build.lock
```

This directory is not a data container:

- there is no API to open it as a dataset;
- it contains no `obs`, `var`, graph, embedding, analysis result, or
  authoritative metadata;
- it may contain only a few requested features;
- it cannot replace the H5AD;
- deletion is always semantically safe.

The published/shareable artifact remains the original `.h5ad`.

### Manifest

The manifest records:

- cache format version;
- source matrix identity;
- H5AD group path (`/X` or `/layers/<name>`);
- source encoding, shape, NNZ, and exact dtypes;
- each cached global feature position;
- shard path and local feature path;
- stored row-index and value dtypes;
- entry counts and payload bytes;
- build time and source-read statistics;
- optional integrity digest;
- package/native writer version.

The manifest is written last with same-directory atomic replacement.
Readers ignore staging files and unreferenced shards.

### HDF5 shard schema

Each requested feature is an independent posting list:

```text
/features/<global-feature-position>
  attrs:
    source-column
    source-shape
    source-data-dtype
    nnz
  /rows
  /data
```

`rows` contains source observation positions in increasing order. `data`
contains exact source values in matching order.

Independent feature groups are preferable to one monolithic CSC group for
the incremental cache:

- a batch can add only the missing features;
- queries do not depend on contiguous cached feature IDs;
- interrupted builds never modify an existing shard;
- no full-length CSC `indptr` or empty-column metadata is required;
- small feature panels remain small.

The schema is private and versioned because it is a cache, not an
interchange format. The native reader must reject unknown versions.

Hot cache datasets are contiguous and uncompressed by default. Compression
can be benchmarked later, but no mandatory HDF5 filter plugin is allowed.

## Storage cost

Storage is proportional to the features actually requested.

At the production density projected to 10M observations, an average feature
has approximately 1.24M stored entries.

Approximate per-feature cache payload:

| Physical values | Rows | Bytes per average feature |
|---|---|---:|
| `int64` | `uint32` | 14.8 MB |
| `int32` | `uint32` | 9.9 MB |
| `float32` | `uint32` | 9.9 MB |

Ten typical features therefore cost approximately 100-150 MB, rather than a
286-429 GB full CSC mirror. Expression skew means common features will be
larger and rare features smaller; preflight uses measured source counts when
available.

Values are stored in their exact source dtype. A later optional cache policy
may compact integral values only after exact range validation, but dtype
conversion is not part of version one.

## Lazy transforms

Cache the raw source posting list, not the transformed output.

For a `LazyTransform`, cache hits perform:

1. read cached row IDs and raw values;
2. gather the relevant row scale factors;
3. apply normalization/log transformation natively;
4. scatter to sparse or dense output.

The same raw cache therefore serves different compatible lazy transform
descriptors. Changing row scaling or log parameters does not duplicate the
feature cache.

A persisted layer has its own cache identity because its raw values may
differ from `/X`.

## Cache construction

### First miss

For a CSR matrix and a requested set of missing features:

1. normalize requested feature positions to unique sorted IDs and retain an
   inverse map for duplicates/order;
2. allocate a small lookup table over `n_vars`;
3. scan CSR `indptr`/`indices` in bounded row blocks;
4. identify matching entries and their source value positions;
5. use the existing block-touch cost model to choose coalesced value reads or
   sequential value reads;
6. append `(row, raw value)` pairs to bounded per-feature buffers;
7. produce the requested dense/sparse result during the same operation;
8. flush buffers to a unique staging shard;
9. validate the shard and re-check the source identity;
10. atomically publish the manifest entry.

The result path must not wait for a second source scan. Cache staging and
publication add only selected-output writes and bounded packing work to the
existing miss.

For a larger requested panel, split construction by a cache-byte budget, not
an arbitrary feature count. One source scan should populate all features in
the current request when the projected postings fit the configured staging
budget.

### Mixed hit and miss

If a request contains cached and uncached features:

- read cached features immediately;
- perform one CSR scan for the union of missing features;
- merge results into the original requested order;
- cache each unique missing feature once.

Duplicate requested features never create duplicate cache entries.

### Explicit prewarm

`prepare_feature_cache()` performs the same selective scan without requiring
plotting or imputation. It accepts names or positions and reports:

- cache hits/misses;
- source bytes expected/read;
- selected values;
- cache bytes written;
- HDF5 read/write calls;
- planning, scanning, packing, writing, validation, and fsync time;
- peak native buffers.

It requires an explicit feature list in version one. `features=None` must not
silently build a full CSC copy.

### Opportunistic fusion

Expose a native cache writer that other full CSR passes may feed when the
desired feature panel is already known. Examples include a pipeline-supplied
marker panel or a later GPU streaming pass.

Do not couple cache construction to SVD/filtering by default. A hidden full
transpose or an unknown feature policy would recreate the original scope
problem.

## Cache reads

For a cache hit:

1. validate the manifest/source identity;
2. read the feature's contiguous row and value arrays;
3. optionally intersect with a backed-view row selection;
4. apply any raw-value transform;
5. scatter into dense output or construct sparse output;
6. restore requested row and feature ordering/duplicates.

One feature query reads only that feature's posting payload. It never opens or
scans CSR `/X/indices`.

If the source H5AD is already CSC, use the source directly and do not create a
feature cache.

Dense HDF5 matrices are out of scope for version one. Their optimal cache
layout depends on source chunking and should be benchmarked separately.

## Invalidation and source identity

Cache correctness depends on conservative invalidation.

The cache key includes:

- canonical source filename;
- device, inode, size, and nanosecond mtime;
- HDF5 matrix group path;
- encoding type/version;
- shape and NNZ;
- `data`, `indices`, and `indptr` dtypes;
- dataset storage layout and sizes;
- ACTIONet matrix-generation metadata when available.

Rules:

1. fingerprint before and after a build; publish only if unchanged;
2. check the cheap fingerprint before every cache read;
3. if any component differs, treat all entries for that matrix identity as
   misses;
4. ACTIONet structural rewrites explicitly invalidate the old identity after
   successful atomic replacement;
5. an in-memory replacement of `adata.X` or a layer never uses a cache for the
   old backed group;
6. cache hits are disabled while an incompatible source write is in flight;
7. a full content verification mode may recompute a digest, but is never
   required for ordinary opens.

HDF5 does not provide a cheap per-dataset mutation timestamp. A file-level
mtime change caused only by metadata may therefore produce a false
invalidation. That costs performance but preserves correctness. Missing a
real matrix change is unacceptable.

Version one does not attempt to preserve cache entries across external H5AD
rewrites, copies, or moves. Reattachment by content digest can be added later.

## AnnData and ecosystem guarantees

- all public analysis functions continue to accept and mutate/return AnnData
  exactly as documented;
- the H5AD remains valid without the cache directory;
- `adata.write_h5ad()` and third-party rewrites need no cache awareness;
- moving or publishing only the H5AD remains supported;
- third-party tools neither see nor copy cached features;
- no private AnnData class or Dask object enters the numeric path;
- no cache pointer is embedded into H5AD metadata;
- cache invalidation is an ACTIONet performance concern, not an ecosystem
  data contract.

This is analogous to a database index or compiled cache, not a replacement
data model.

## Variable filtering and var-centric workflows

Do not use the feature cache to accelerate general filtering.

### QC statistics

Retain the current fused CSR scan for:

- row sums;
- row NNZ;
- column NNZ.

Add native profiling and ensure each filter iteration reads the sparse
payload only once. Small reusable statistics may be written to ordinary
`adata.obs`/`adata.var` columns when the caller requests it, which is fully
standard AnnData behavior.

### Physical filtering

Retain native bounded H5AD subsetting. A successful rewrite changes the
matrix identity and invalidates the feature cache.

A later optimization may transform cached postings through the same
observation/variable selection instead of discarding them, but invalidation
is the simpler and safer first implementation.

### Explicit CSC orientation

For the uncommon workflow dominated by variable-space operations, add or
retain an explicit orientation conversion that writes another standard H5AD:

```python
actionet.repack_h5ad(
    adata,
    output_file="atlas.csc.h5ad",
    sparse_orientation="csc",
)
```

The result remains ordinary AnnData/H5AD. Users knowingly trade row access
for column access rather than maintaining two authoritative orientations.

## Native implementation

### libactionet

Add a focused API beside the existing backed H5AD reader:

- `FeatureCacheDescriptor`;
- `FeatureCacheEntry`;
- `FeatureCacheReadRequest`;
- `FeatureCacheBuildRequest`;
- `FeatureCacheStats`;
- `read_cached_features()`;
- `extract_and_cache_features()`;
- `validate_feature_cache_shard()`.

The API remains path-based and independent of AnnData. It receives the source
H5AD path/group and a resolved cache shard path.

`BackedSparseMatrixOperator::takeColumnsDense/Sparse` gains an optional cache
descriptor. Its public matrix/operator semantics remain unchanged.

### Python

Proposed modules:

- `src/actionet/io/feature_cache.py`: policy, path resolution, fingerprint,
  manifest, locking, and inspection;
- `src/actionet/io/operator.py`: resolve cache hits before operator creation;
- `src/actionet/io/matrix_source.py`: `feature_subset(..., feature_cache=...)`;
- `src/actionet/network/imputation.py`: route through
  `MatrixSource.feature_subset`;
- `src/actionet/visualization/feature_expression.py`: route through the same
  method;
- `src/actionet/io/subset.py` and `rewrite.py`: invalidate after successful
  matrix replacement;
- `src/actionet/bindings/wp_io.cpp`: native cache bindings.

Consolidating plotting and imputation on `MatrixSource.feature_subset` is part
of the feature. They currently duplicate the backed operator preamble.

### R

The native cache schema and source fingerprint are language-neutral.
An R wrapper can later use the same path-based cache without reintroducing an
R-specific data container. R support is not required to land the first Python
implementation, but the C++ interface must not preclude it.

## Concurrency and transactions

- existing cache shards are immutable;
- one short-lived advisory build lock protects manifest publication;
- readers never wait for a build and may fall back to CSR;
- builders write a unique staging shard;
- source fingerprints are checked after extraction;
- shard validation and fsync complete before manifest publication;
- manifest replacement is atomic and parent-directory fsynced;
- abandoned staging files are ignored and removed by explicit cleanup;
- concurrent requests for the same missing feature may do duplicate work, but
  only one valid entry is published.

Cache write failures never modify the source H5AD.

## Implementation phases

### Phase 0: benchmark contract

- Extend `tests/benchmark_backed_take_columns.py` with cache miss/hit modes.
- Measure one, ten, and one hundred common/rare/random features.
- Separate sparse read, dense scatter, transform, and DataFrame construction.
- Run cold and warm on production NVMe/shared storage.
- Add a 10M fixture with production density and feature-frequency skew.

### Phase 1: read-only prototype

- Manually create posting-list shards for known features.
- Add native cache-hit reads for dense and sparse outputs.
- Validate row selections, duplicates, lazy transforms, and exact dtypes.

Gate: cached reads meet the latency/amplification targets below.

### Phase 2: fused miss/build path

- Add native selective extraction plus posting-list capture.
- Publish immutable shards and manifest entries.
- Ensure the first miss performs no additional source scan.
- Support mixed hit/miss requests.

Gate: miss performance is no worse than the current optimized CSR path beyond
the bounded selected-cache write.

### Phase 3: public policy and call-site routing

- Add `feature_cache` options and explicit management functions.
- Route plotting, imputation, and `MatrixSource.feature_subset` through one
  implementation.
- Add cache path diagnostics and byte-cap handling.

### Phase 4: invalidation and failure injection

- Integrate with every ACTIONet matrix rewrite/transform path.
- Add source-change-before/during/after-build tests.
- Add concurrent reader/builder tests and staging cleanup.

### Phase 5: production validation

- Prewarm realistic marker panels.
- Validate repeated notebook/visualization sessions.
- Measure cache hit rate and total cache size.
- Decide from evidence whether selective caching is sufficient.

Only after Phase 5 should a full CSC cache, Zarr cache, or TileDB-backed index
be considered.

## Correctness gates

- exact raw values and dtype on cache hit;
- dense and sparse parity with uncached extraction;
- row subset, reorder, boolean mask, and duplicate parity;
- feature reorder and duplicate parity;
- stored zeros and duplicate sparse coordinates;
- exact integers above `2^53`;
- empty requested feature and zero-NNZ feature behavior;
- lazy normalization/log parity;
- `/X` and named-layer isolation;
- backed view parity;
- source replacement and in-place value change invalidation;
- no cache use for an in-memory replacement matrix;
- malformed/truncated/unknown-version shard rejection;
- interrupted build leaves no published entry;
- source H5AD fingerprint/content unchanged after cache build.

## Performance and resource gates

### Cache hit

On the 10M reference NVMe fixture:

- one average sparse feature: <= 250 ms warm, <= 1 second cold;
- one dense float64 feature: <= 1 second warm, <= 3 seconds cold;
- ten dense float64 features: <= 5 seconds warm, <= 10 seconds cold;
- source CSR bytes read: zero after fingerprint/metadata inspection;
- cache payload amplification: <= 1.25x selected posting bytes;
- native scratch: <= 128 MiB excluding requested dense output.

Absolute shared-filesystem latency is reported separately from read
amplification and raw throughput.

### Cache miss

- source bytes read no more than the current planner;
- additional cache writes <= 1.1x the selected posting payload;
- no second source scan;
- wall time <= current uncached path + 10%;
- selected cache bytes within 1.1x `rows + exact data` logical bytes;
- peak native scratch <= configured cap;
- failure to publish does not fail the extraction result in `auto` mode.

### Open and interoperability

- cache discovery <= 100 ms warm for up to 10,000 cached features;
- importing ACTIONet adds no eager cache scan;
- `anndata.read_h5ad()` behavior is unchanged;
- deleting the complete cache passes all non-performance tests;
- moving/copying only the H5AD yields a correct cache miss;
- no new mandatory runtime dependency.

## When selective caching is insufficient

Escalate only with measured evidence.

### Full CSC cache

Build a complete rebuildable CSC mirror if users routinely touch a large
fraction of features and repeated selective misses cause multiple global CSR
scans. It remains an optional cache, not a public container.

The crossover should be based on:

- cumulative source bytes scanned because of cache misses;
- distinct feature-cache hit rate;
- cache bytes already materialized;
- estimated remaining full-CSC build/storage cost.

### Zarr

Zarr may be a better cache transport for object storage or compressed
per-feature chunks. It does not change the public AnnData model, but it adds a
dependency and should beat the existing HDF5 posting cache on deployed
storage before adoption.

### TileDB Embedded

TileDB can serve as a technically sophisticated secondary index without
becoming the public container. Its open-source licensing is not a blocker.
It should be tried if HDF5 cache publication, concurrent access, or selective
query latency fails the gates.

The comparison is then narrowly scoped to an optional feature index rather
than a wholesale replacement of H5AD/AnnData.

## Go/no-go

Proceed if:

- common interactive feature panels become cache hits after one existing CSR
  miss;
- cache storage remains proportional to actual interactive use;
- invalidation is conservative and reliable;
- the source H5AD and AnnData APIs remain untouched;
- first-miss overhead stays within the stated bound.

Reconsider the strategy if:

- exploration routinely requests thousands of distinct features;
- users cannot tolerate the first CSR scan and cannot prewarm panels;
- source matrices are frequently mutated in place;
- cache invalidation creates repeated false misses;
- a tuned TileDB/Zarr feature index materially outperforms the HDF5 cache
  without increasing deployment risk.

The expected outcome is that the common case pays the unavoidable CSR scan
once per requested feature panel, then obtains near-CSC feature reads without
duplicating the full matrix or introducing a new scientific data container.
