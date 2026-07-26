# Native HDF5 Backed-I/O Implementation Handoff

Date: 2026-07-26  
Status: implementation complete; production-validation period active  
Repository: `actionet-python`  
Branch: `dev-gpu`  
Parent baseline: `f697fcc09ed9dd258f7ce2b092c8fd5c226b6c10`  
`libactionet` baseline: `2a10cf8b1e2f75c7ad4aec9cf85096c8c944ae9c`

The implementation is currently an uncommitted parent worktree plus an
uncommitted `src/libactionet` submodule worktree. Preserve both sets of changes.
When publishing, commit `libactionet` first, then commit the Python bindings,
integration, tests, documentation, and updated submodule pointer in the parent
repository.

## Purpose

This handoff supersedes the removed
`plans/BACKED_SUBSET_ROW_SELECTION_HANDOFF.md`. The earlier work focused on a
single near-identity CSR row-selection optimization. The completed refactor
instead provides a format-aware native HDF5 data plane with bounded behavior
across the full selector space: any retained fraction, dense/CSR/CSC storage,
ordered or general selectors, and observation-only or joint-axis subsets.

AnnData remains the only public Python container. Native code owns versioned
numeric H5AD matrix I/O; Python owns metadata semantics, compatibility with
AnnData releases, and crash-safe file publication.

## Non-negotiable safety boundary

The production atlas used during development is:

`/data/tau_project/tmp/adata_agg_ALL_pass132_post.h5ad`

It must remain read-only. Every performance run must:

1. capture the source inode, size, and mtime fingerprint;
2. create a uniquely named temporary output on the same filesystem;
3. validate the completed output;
4. remove that output; and
5. confirm the source fingerprint is unchanged.

The last verified source state was:

- inode: `80412761`;
- size: `88,072,042,312` bytes;
- shape: `(1,575,069, 28,856)`;
- `/X.nnz`: `5,627,020,482`.

## Implemented architecture

### Native numeric data plane

The public C++ contract is:

`src/libactionet/include/io/backed_h5ad/h5ad_matrix_io.hpp`

Its implementation is:

`src/libactionet/src/io/backed_h5ad/h5ad_matrix_io.cpp`

The path-based `actionet::h5ad` API exposes:

- `inspect_matrix(file, path)`;
- `validate_matrix(file, path, level)`;
- `subset_matrix(source, destination, row_selection, column_selection, options)`;
- `copy_matrix(...)`; and
- `transform_matrix(...)`.

No public API exposes `hid_t`. Files, groups, datasets, spaces, property
lists, and HDF5 types are RAII-owned. Shared HDF5 utilities live in:

`src/libactionet/src/io/backed_h5ad/_h5_utils.hpp`

The existing dense and sparse backed compute operators now share native
encoding inspection, shape validation, signed/unsigned index conversion,
pointer validation, and RAII handle management. Compute readers may convert
values to `double`; transfer readers preserve the source dtype and exact
values, including integers above `2^53`.

Supported H5AD encodings are deliberately version checked:

- dense `array`, encoding version `0.2.0`;
- sparse `csr_matrix`, encoding version `0.1.0`;
- sparse `csc_matrix`, encoding version `0.1.0`.

Future encoding versions must be enabled intentionally after compatibility
review.

### Selection and transfer behavior

`AxisSelection.indices == nullopt` means the whole axis. A present empty
vector means an empty axis. The native API supports empty axes; the existing
public Python AnnData subset APIs continue to reject empty AnnData outputs.

The default transfer limits are:

- maximum native buffer: 128 MiB;
- gap-merge allowance: 64 KiB;
- maximum rows per batch: the public `backed_write_chunk_size`, default
  `16384`;
- layout policy: preserve source layout.

Byte limits may reduce the effective row batch. Peak native matrix buffers do
not include the loaded sparse `indptr`, selectors, metadata objects, or HDF5
library overhead.

Ordered unique selectors receive the sequential-scan performance guarantee:

- CSR translates selected row runs through `indptr`;
- CSC scans selected columns and filters/remaps rows without converting
  orientation;
- dense data uses bounded row slabs and the same range planner;
- gaps of at most 64 KiB are merged;
- ranges touching the same compressed HDF5 chunk are merged;
- spans are capped by the 128 MiB buffer;
- the selected-span cost is compared with a bounded full scan and the cheaper
  plan is used.

Reordered, descending, and duplicate selectors retain exact semantics through
bounded output batches. Source reads are sorted and deduplicated inside a
batch, then scattered or repeated into output order. Duplicate column
selections are expanded correctly. General selectors have bounded memory but
do not claim the ordered-selector sequential-scan guarantee.

Sparse output orientation is preserved. Output `indices` use signed `int32`
when the indexed axis fits and signed `int64` otherwise. Output `indptr` uses
signed `int32` when output NNZ fits and signed `int64` otherwise.

### Layout, filters, and transformations

Inspection reports, for every physical matrix dataset:

- logical shape and dtype;
- logical and stored bytes;
- contiguous/chunked layout;
- chunk shape;
- filter IDs, names, flags, and client data;
- filter decoder and encoder availability.

Transfers preserve source chunks, filters, shuffle, and checksums when the
corresponding encoders are available. Unsupported encodings and unavailable
filter encoders fail during preflight, before destination matrix mutation.
`Uncompressed` is an explicit copy policy used by decompression.

Native persistent transforms support row scaling and optional logarithms for
dense, CSR, and CSC inputs. Sparse orientation and structure are preserved.
When a destination layer is created from an exact structure copy in the same
transaction, immutable sparse `indices` may be hard-linked. `indptr` remains
independent so later structural rewrites cannot alias it.

### Profiling

`TransferStats` reports:

- selected logical source bytes;
- actual source bytes read and gap bytes;
- destination bytes written;
- HDF5 read/write calls and span count;
- planning, source-read, packing, destination-write, and flush time;
- peak native buffer bytes;
- optional span-level read and packing timings.

The Python transaction additionally reports temporary-file `fsync`,
fingerprint, source-close, replacement, and parent-directory `fsync` timing.
Do not infer write progress from HDF5 destination file size: contiguous
dataset extent allocation and page-cache writeback can make logical size jump
far ahead of durable data.

## Python control plane

### AnnData compatibility adapter

`src/actionet/io/backed_adapter.py` is the only backed-I/O module that should
know about:

- AnnData private file-manager state;
- filename and open-mode recovery;
- close, reopen, and existing-object refresh;
- backed view parent and axis selectors;
- the AnnData 0.13 `layers[None]` alias;
- whether a logical matrix value is genuinely backed at a specific HDF5
  file/path.

An in-memory replacement remains authoritative even if the source H5AD still
contains an object at the same logical path. Such values are serialized by
the public AnnData codec rather than copied from the stale backing dataset.

Do not reintroduce AnnData private-handle or private-view knowledge elsewhere.

### Native engine control and rollback

`src/actionet/io/native_h5ad.py` owns native preflight and execution.

The private temporary rollback switch is:

`ACTIONET_BACKED_IO_ENGINE=auto|native|python`

- `auto`: use native I/O for supported genuinely backed matrices; capability
  rejection may fall back to the Python writer before native transfer starts;
- `native`: capability rejection is an error;
- `python`: use the legacy Python matrix writer;
- any native runtime error after writing starts aborts the transaction and
  never silently retries through Python.

Keep this switch for one release of production validation. Remove the legacy
engine only after the performance grid and deployed-storage non-regression
gates pass.

### Rewrite transaction

`src/actionet/io/rewrite.py` owns same-directory temporary files and durable
publication.

The shared rewrite sequence is:

1. flush ACTIONet-tracked pending changes;
2. capture the source inode/size/mtime fingerprint;
3. create a unique temporary H5AD in the destination directory;
4. serialize current metadata and in-memory values with public
   `anndata.io.write_elem`;
5. copy unknown HDF5 objects with HDF5 object-copy semantics;
6. close Python destination handles;
7. run native transfers for genuinely backed numeric matrices;
8. structurally validate the completed H5AD by reopening it backed;
9. close HDF5 handles, then flush and `fsync` the temporary file;
10. for in-place publication, verify the original fingerprint and close the
    source only at commit;
11. atomically replace the destination and best-effort `fsync` its parent
    directory;
12. refresh the existing AnnData object without changing Python object
    identity.

Before replacement, any failure deletes only the temporary file. Close or
replace failures attempt to restore the original AnnData handle. A fingerprint
change refuses in-place publication.

## Migrated call paths

The native transaction substrate is active in:

| Area | Main implementation |
|---|---|
| Backed subset/materialize/in-place filtering | `src/actionet/io/subset.py` |
| QC filtering wrappers | `src/actionet/preprocessing/filter.py` |
| Repack and checkpoint | `src/actionet/io/checkpoint.py` |
| Atomic matrix/file decompression | `src/actionet/preprocessing/io.py` |
| Persistent backed scaling/log transform | `src/actionet/preprocessing/normalize.py` |
| Annotation persistence | `src/actionet/io/anndata_io.py` |
| Pending write/handle lifecycle | `src/actionet/io/persist.py`, `operator.py`, `matrix_source.py` |
| Private pybind interface | `src/actionet/bindings/wp_io.cpp` |

The private `_core` bindings release the GIL around native inspection,
validation, selection, copying, and transforms. HDF5 calls remain serial.
OpenMP is not used to issue concurrent HDF5 calls.

Public Python container types, function signatures, selector ordering and
duplicate semantics, `backed_write_chunk_size`, and atomic replacement
behavior remain backward compatible. `backed_write_chunk_size` is now a
maximum native row count; the rollback Python path retains its historical
coupled read/write-stride interpretation.

## Test coverage

Native tests:

`src/libactionet/tests/test_h5ad_matrix_io.cpp`

They cover:

- dense, CSR, and CSC;
- integer and floating payloads, including exact values above `2^53`;
- signed and unsigned sparse indices;
- contiguous, chunked, gzip, shuffle, and Fletcher32 layouts;
- empty selections, zero NNZ, malformed `indptr`, invalid indices, unknown
  encodings, and unavailable filters;
- all, random, reordered, descending, and duplicate selectors;
- randomized property comparisons against a small dense reference;
- uncompressed copy;
- persistent transforms and sparse structure hard-link behavior;
- preflight failures that leave existing destination sentinels untouched.

Python integration and failure-injection coverage is in:

- `tests/test_subset_anndata.py`;
- `tests/backed/test_native_h5ad_control.py`;
- `tests/backed/test_rewrite_transaction.py`;
- `tests/backed/test_checkpoint.py`;
- `tests/backed/test_infrastructure.py`;
- `tests/backed/test_backed_extension.py`.

The last full validation in Python 3.12.13 with AnnData 0.12.10 was:

`579 passed, 14 skipped, 386 warnings`

The native H5AD test executable passed. The normal extension build passed.
The `libactionet` core library also built successfully with
`LIBACTIONET_ENABLE_HDF5=OFF`. Ruff checks on all changed Python files and
both parent/submodule `git diff --check` passed.

Useful validation commands in the existing development environment:

```bash
cmake --build /tmp/actionet-native-build -j4
ctest --test-dir /tmp/actionet-native-build/src/libactionet --output-on-failure
cmake --build /tmp/libactionet-nohdf -j4
PYTHONPATH=/tmp/actionet-core-preload:src \
  /data/tau_project/.venv_tau/bin/python -m pytest -q
/data/tau_project/.venv_tau/bin/ruff check <changed-python-files>
git diff --check
git -C src/libactionet diff --check
```

The `/tmp` build directories are ephemeral; reconfigure them if they no
longer exist.

## Production-atlas evidence

Every atlas run used a unique same-filesystem output, validated it, deleted
it, and rechecked the source fingerprint.

| `/X` workload | Wall | Selected bytes | Actual read | Amplification | Peak buffer |
|---|---:|---:|---:|---:|---:|
| Raw sequential reference | 189.42 s | — | 67.54 GB | — | — |
| Ordered random 5% | 8.41 s | 3.373 GB | 3.467 GB | 1.028x | 67.3 MB |
| Ordered random 50% | 44.98 s | 33.77 GB | 38.32 GB | 1.135x | 68.5 MB |
| Ordered random 99.9% | 65.66 s | 67.46 GB | 67.48 GB | 1.0004x | 117.4 MB |

The 99.9% `/X` transfer is 1.89x faster than the previously profiled
123.92-second legacy `/X` stage. The tested low-, middle-, and
near-identity-fraction cases meet the relevant read-amplification,
throughput, and buffer gates.

An end-to-end subset removing two observations completed and validated in
393.37 seconds, producing an 88.19 GB output. The long tail was durable
writeback and `fsync` of a nearly complete 88 GB file, not selector position.
This is expected for physical compaction: the refactor removes pathological
selection overhead but cannot avoid writing the retained output.

For repeated curation, accumulating logical QC masks and compacting once is
still the most efficient workflow when semantics permit. That workflow
optimization is separate from the backed-I/O implementation and is not
required for exact arbitrary-fraction subsetting.

Detailed evidence is retained in:

`plans/NATIVE_HDF5_BACKED_IO_ROLLOUT.md`

The reusable performance harness is:

`tests/benchmark_native_h5ad_subsets.py`

## Remaining rollout work

The implementation is complete. The following are validation and retirement
tasks, not missing architecture:

1. Run the complete retained-fraction/pattern grid from
   `tests/benchmark_native_h5ad_subsets.py`, including 0.1%, 1%, 5%, 10%,
   25%, 50%, 75%, 90%, 99%, and 99.9%; contiguous, random, clustered,
   alternating, and position-shifted selections; and observation-only plus
   joint-axis subsets.
2. Run the integration suite in an AnnData 0.13 environment in addition to
   the verified 0.12.10 environment.
3. Collect production profiles on deployed storage for one release.
4. If non-regression gates hold, remove
   `ACTIONET_BACKED_IO_ENGINE=python` and the superseded Python bulk matrix
   writers.
5. Publish in dependency order: `libactionet` commit first, parent
   integration commit second.

Do not weaken atomic publication, dtype exactness, selector semantics,
filter preflight, orientation preservation, or the source read-only policy
while completing these gates.
