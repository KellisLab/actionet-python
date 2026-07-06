# I/O and backed persistence

Streaming (HDF5-backed) `AnnData` support plus the auto-persistence,
checkpoint, and subset helpers used across the pipeline.

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
