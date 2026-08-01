# Native HDF5 Backed-I/O Rollout

Date: 2026-07-25  
Status: implemented; production-validation period active

## Delivered

- Path-based, RAII-owned `actionet::h5ad` inspection, validation, subset,
  copy, and persistent-transform APIs for dense, CSR, and CSC H5AD matrices.
- Exact transfer dtypes, signed output sparse indices, selector
  order/duplicate parity, bounded general-selector batching, gap-aware
  ordered spans, scan/gather cost selection, layout/filter preservation, and
  uncompressed copy policy.
- Aggregate and optional span profiling, including layout inventory, logical
  and stored bytes, filter capability, read amplification, calls, packing,
  writing, flushing, and peak buffers.
- One AnnData compatibility adapter and one durable rewrite transaction used
  by subsetting, materialization, repack/decompression, backed
  normalization, checkpoint, and annotation persistence.
- Private `ACTIONET_BACKED_IO_ENGINE=auto|native|python` rollback switch.
- Public AnnData containers, function signatures, selector behavior, and
  atomic replacement semantics remain unchanged.

## Verification

- Native C++ tests cover dense/CSR/CSC, signed and unsigned sparse indices,
  exact integers above `2^53`, contiguous/chunked/gzip/shuffle/Fletcher32,
  empty outputs, zero NNZ, malformed pointers, invalid indices, unsupported
  encodings, unavailable filters, layout changes, transforms, duplicates,
  reorderings, and randomized dense-reference properties.
- Python integration and failure-injection tests cover metadata/container
  round trips, unknown top-level HDF5 objects, direct in-memory annotation
  authority, transaction cleanup and handle restoration, compression scope,
  sparse structure sharing, native rollback policy, and exact backed
  duplicate selectors.
- Current environment: 579 passed, 14 skipped.
- `libactionet` builds and tests with HDF5 enabled; the core library also
  builds with `LIBACTIONET_ENABLE_HDF5=OFF`.

## Atlas evidence

Read-only source:

`/data/tau_project/tmp/adata_agg_ALL_pass132_post.h5ad`

Source fingerprint was identical before and after all runs:

- inode: `80412761`
- size: `88,072,042,312` bytes
- shape: `(1,575,069, 28,856)`
- `/X.nnz`: `5,627,020,482`

Every benchmark used a unique temporary output on `/data`, validated it, and
removed it.

| `/X` workload | Wall | Selected bytes | Actual read | Read amplification | Peak native buffer |
|---|---:|---:|---:|---:|---:|
| Raw sequential reference (100%) | 189.42 s | — | 67.54 GB | — | — |
| Ordered random 5% | 8.41 s | 3.373 GB | 3.467 GB | 1.028x | 67.3 MB |
| Ordered random 50% | 44.98 s | 33.77 GB | 38.32 GB | 1.135x | 68.5 MB |
| Ordered random 99.9% | 65.66 s | 67.46 GB | 67.48 GB | 1.0004x | 117.4 MB |

The 99.9% native `/X` transfer is 1.89x faster than the previously profiled
123.92-second `/X` stage. The 5% selective-read case is below the 2x
amplification gate. The 50% and 99.9% transfers both exceed the required
fraction of raw sequential-reference throughput.

An end-to-end two-observation H5AD subset validated successfully in 393.37
seconds and produced an 88.19 GB output. The source stayed unchanged and the
output was removed. A separate full-H5AD 5% profile mapped the costs:

- serialization and validation: 7.56 s;
- `/X` native transfer within that run: 3.86 s;
- `obsp/actionet`: 0.40 s;
- durable temporary-file `fsync`: 2.64 s.

The near-identity end-to-end tail is therefore durable writeback of the full
88 GB output, not position-sensitive source selection. The transaction now
reports this explicitly as `transaction_commit.temp_fsync_s`.

## Rollout gates still requiring scheduled infrastructure

- Run the complete retained-fraction/pattern grid with
  `tests/benchmark_native_h5ad_subsets.py`; the targeted 5%, 50%, and 99.9%
  atlas probes above establish the critical low/mid/high-fraction bounds but
  are not the entire grid.
- Exercise the integration suite in an AnnData 0.13 environment in addition
  to the current AnnData 0.12.10 environment.
- Keep the private Python rollback engine for one release, collect production
  profiles, then remove the Python transfer engine only after the stated
  non-regression gates hold on deployed storage.
