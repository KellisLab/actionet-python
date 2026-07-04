# Lazy transform

`LazyTransform` describes a deferred normalization / log-transform pipeline that
can be composed once and applied streaming (row-block) to backed matrices,
avoiding materialization of intermediate matrices.

::: actionet.lazy_transform
    options:
      members:
        - LazyTransform
        - create_lazy_transform
