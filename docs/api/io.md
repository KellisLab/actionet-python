# I/O and backed persistence

Streaming (HDF5-backed) `AnnData` support plus the auto-persistence,
checkpoint, and subset helpers used across the pipeline.

## Read/compute versus write chunks

The default backed chunk size remains 4,096 for compatibility and for the
read/compute paths tuned by the backed SVD benchmarks. HDF5 rewrites can
benefit substantially from larger transfers because a 4,096-element transfer
may be smaller than one physical HDF5 storage chunk. For atlas-scale sparse
files, 32,768 is a useful first value to test. Larger values increase the
amount of temporary data held for each transfer.

Write-only APIs already expose their transfer size directly:

| Operation | Write control |
| --- | --- |
| `checkpoint_backed(..., compact=True)` | `chunk_size` |
| `decompress_backed_storage(...)` | `chunk_size` |
| `subset_anndata(...)`, `apply_filter(...)` | `backed_chunk_size` |
| `materialize_backed(...)`, `subset_backed_inplace(...)` | `chunk_size` |

`checkpoint_backed(..., compact=False)` does not repack the file, so its
`chunk_size` is unused.

Hybrid compute/write APIs expose an independent
`backed_write_chunk_size`. Leaving it as `None` preserves historical behavior
by inheriting `backed_chunk_size`:

```python
an.run_svd(
    adata,
    backed_chunk_size=4096,
    backed_write_chunk_size=32768,
)

an.filter_anndata(
    adata,
    backed_chunk_size=4096,
    backed_write_chunk_size=32768,
)

an.checkpoint_backed(
    adata,
    compact=True,
    chunk_size=32768,
)
```

For SVD and kernel reduction, the write control applies to automatic
decompression while `backed_chunk_size` continues to configure the C++
read/compute operator. For filtering it applies to the structural rewrite,
and for normalization it applies to the transform/write pass. The C++ backed
operators remain read/compute components and keep their independently tuned
4,096 default.

The focused benchmark compares 4,096 and 32,768 on the same filesystem and
records wall time, peak RSS, and process I/O for repacking, decompression,
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
