# I/O and backed persistence

Streaming (HDF5-backed) `AnnData` support plus the auto-persistence,
checkpoint, and subset helpers used across the pipeline.

## Read/compute versus write chunks

The Python front-end exposes chunk-size parameters **granularly**: functions
supporting backed mode take a **read/compute** control (`backed_chunk_size`),
a **write** control (`backed_write_chunk_size`), or both, depending on what
the function does. The two directions have independent shared defaults:

| Direction | Parameter | Default |
| --- | --- | --- |
| Read / compute streaming | `backed_chunk_size` | `8192` |
| HDF5 write / rewrite | `backed_write_chunk_size` | `16384` |

`backed_write_chunk_size` remains a row-count ceiling for compatibility.
Native transfers may use fewer rows to stay within a 128 MiB buffer bound.
They inspect the physical HDF5 layout, merge nearby sparse ranges, and choose
between selected-range reads and a bounded sequential scan. Increasing the
row ceiling can still reduce overhead, but it no longer permits an
unbounded temporary matrix.

Write-only APIs expose their transfer size directly under the write name:

| Operation | Write control |
| --- | --- |
| `checkpoint_backed(..., compact=True)` | `backed_write_chunk_size` |
| `decompress_backed_storage(...)` | `backed_write_chunk_size` |
| `subset_anndata(...)`, `apply_filter(...)` | `backed_write_chunk_size` |
| `materialize_backed(...)`, `subset_backed_inplace(...)` | `backed_write_chunk_size` |

`checkpoint_backed(..., compact=False)` does not repack the file, so its
`backed_write_chunk_size` is unused.

Hybrid compute/write APIs expose an independent `backed_write_chunk_size`
alongside `backed_chunk_size`. Leaving `backed_write_chunk_size` as `None`
now falls back to the shared write default (`16384`); it no longer
implicitly inherits `backed_chunk_size`.

```python
an.run_svd(
    adata,
    backed_chunk_size=8192,
    backed_write_chunk_size=32768,
)

an.filter_anndata(
    adata,
    backed_chunk_size=8192,
    backed_write_chunk_size=32768,
)

an.checkpoint_backed(
    adata,
    compact=True,
    backed_write_chunk_size=32768,
)
```

For SVD and kernel reduction, the write control applies to automatic
decompression while `backed_chunk_size` continues to configure the C++
read/compute operator. For filtering it applies to the structural rewrite,
and for normalization it applies to the transform/write pass. The C++ backed
operators remain read/compute components and keep their independently tuned
`4096` default.

## Native backed rewrite engine

Backed dense, CSR, and CSC matrices are inspected, validated, selected, and
transferred by `libactionet` using version-checked H5AD encodings. AnnData
continues to own the Python container and metadata codec; it is not used as
the bulk matrix reader. Native transfers preserve matrix orientation, exact
data values and dtypes, ordered or duplicated selectors, and compatible HDF5
chunks and filters. Repacking can explicitly request uncompressed output.

Subsetting, materialization, repacking, decompression, backed normalization,
and persistence publish through one same-directory rewrite transaction. A
completed temporary H5AD is validated and synced before atomic replacement.
For in-place work, ACTIONet also verifies the source inode, size, and
modification time immediately before commit. Failures before commit remain
confined to the temporary file.

On very large outputs, the final `fsync` can take longer than serialization:
it is the point where the operating system must make buffered writes durable.
Private rewrite profiling reports this separately as `transaction_commit`
with `temp_fsync_s`; apparent HDF5 file size is not a reliable progress
indicator for contiguous datasets.

During the compatibility rollout, the private environment variable
`ACTIONET_BACKED_IO_ENGINE` accepts:

- `auto` (default): use native transfer for supported file-backed matrices and
  fall back before writing when capability inspection rejects a matrix;
- `native`: require native support for genuinely backed matrices;
- `python`: use the previous Python/SciPy transfer path.

An error after a native transfer begins aborts the transaction and is never
silently retried with the Python engine.

The focused benchmark compares different chunk sizes on the same filesystem
and records wall time, peak RSS, and process I/O for repacking, decompression,
subsetting, normalization, and SVD auto-decompression:

```bash
python tests/benchmark_backed_write_chunks.py data/example.h5ad \
  --chunk-sizes 4096 32768 \
  --min-repack-speedup 2.0
```

The speedup gate is opt-in and is intentionally excluded from CI because
shared and HPC filesystem performance is hardware-dependent.

## Lazy transforms

`LazyTransform` describes a deferred normalization / log-transform pipeline
that can be composed once and applied streaming (row-block) to backed
matrices, avoiding materialization of intermediate matrices.

::: actionet.io.lazy_transform
    options:
      members:
        - LazyTransform
        - create_lazy_transform

## Persistence and auto-write

::: actionet.io.persist
    options:
      members:
        - get_auto_persist
        - set_auto_persist

## Checkpoint

::: actionet.io.checkpoint
    options:
      members:
        - checkpoint_backed

## Subset

::: actionet.io.subset
    options:
      members:
        - materialize_backed
        - subset_backed_inplace
