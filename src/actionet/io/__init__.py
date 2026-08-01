"""Backed HDF5 I/O for AnnData.

This subpackage groups all code that manages HDF5-backed AnnData state:

- :mod:`compression` — compression policy dataclass + h5ad introspection.
- :mod:`matrix_source` — unified matrix access abstraction (dense/sparse/backed).
- :mod:`operator` — factory for C++ ``BackedDenseMatrixOperator`` /
  ``BackedSparseMatrixOperator`` bindings.
- :mod:`lazy_transform` — deferred normalization/scaling wrapper for backed matrices.
- :mod:`persist` — in-memory + eager disk write of annotation results.
- :mod:`checkpoint` — explicit flush + optional file compaction.
- :mod:`subset` — atomic row/column subsetting and view materialization.
- :mod:`anndata_io` — HDF5 primitives for writing annotation results.
"""

from .compression import (
    CompressionPolicy,
    format_compression_summary,
    get_matrix_compression_policy,
    get_storage_metadata_from_adata,
    get_storage_metadata_from_matrix,
    is_compressed_storage,
    sparse_group_format,
    write_sparse_csr_group_attrs,
)
from .matrix_source import MatrixChunk, MatrixSource
from .operator import open_backed_operator_for
from .lazy_transform import LazyTransform, create_lazy_transform
from .persist import (
    apply_inmemory_updates,
    coerce_nullable_strings_for_write,
    get_auto_persist,
    is_backed_adata,
    is_writable_backed,
    persist_updates,
    set_auto_persist,
)
from .checkpoint import checkpoint_backed, copy_h5_group
from .subset import materialize_backed, subset_backed_inplace

__all__ = [
    "CompressionPolicy",
    "LazyTransform",
    "MatrixChunk",
    "MatrixSource",
    "apply_inmemory_updates",
    "checkpoint_backed",
    "coerce_nullable_strings_for_write",
    "copy_h5_group",
    "create_lazy_transform",
    "format_compression_summary",
    "get_auto_persist",
    "get_matrix_compression_policy",
    "get_storage_metadata_from_adata",
    "get_storage_metadata_from_matrix",
    "is_backed_adata",
    "is_compressed_storage",
    "is_writable_backed",
    "materialize_backed",
    "open_backed_operator_for",
    "persist_updates",
    "set_auto_persist",
    "sparse_group_format",
    "subset_backed_inplace",
    "write_sparse_csr_group_attrs",
]
