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

HDF5 rewrites can benefit substantially from larger transfers because a
small transfer may be smaller than one physical HDF5 storage chunk. For
atlas-scale sparse files, `32768` is a useful first value to test. Larger
values increase the amount of temporary data held for each transfer.

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
