"""Shared validation and defaults for backed read and write chunk controls.

The Python front-end exposes chunk-size parameters granularly. Backed
functions that stream data through the compute path use
``backed_chunk_size`` (i.e. the read/compute chunk size); functions that
write chunked HDF5 output additionally expose ``backed_write_chunk_size``.

Historically both directions shared a single default of ``4096`` and the
write chunk implicitly inherited the read chunk. The two are now
independent with distinct defaults tuned for the read- and write-heavy
paths respectively.
"""

from __future__ import annotations

from numbers import Integral

#: Default row/element chunk size for backed **read/compute** streaming.
DEFAULT_BACKED_READ_CHUNK_SIZE: int = 8192

#: Default row/element chunk size for backed **write** operations.
DEFAULT_BACKED_WRITE_CHUNK_SIZE: int = 16384


def validate_chunk_size(value: int, *, name: str) -> int:
    """Return *value* as ``int`` after validating a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be a positive integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"`{name}` must be > 0.")
    return resolved


def resolve_backed_write_chunk_size(
    backed_chunk_size: int,
    backed_write_chunk_size: int | None,
) -> tuple[int, int]:
    """Resolve independent read/compute and write chunk sizes.

    Both directions are validated as positive integers. When
    ``backed_write_chunk_size`` is ``None`` the shared write default
    (:data:`DEFAULT_BACKED_WRITE_CHUNK_SIZE`) is used; it no longer
    silently inherits from ``backed_chunk_size``.
    """
    compute_chunk_size = validate_chunk_size(
        backed_chunk_size,
        name="backed_chunk_size",
    )
    if backed_write_chunk_size is None:
        write_chunk_size = DEFAULT_BACKED_WRITE_CHUNK_SIZE
    else:
        write_chunk_size = validate_chunk_size(
            backed_write_chunk_size,
            name="backed_write_chunk_size",
        )
    return compute_chunk_size, write_chunk_size
