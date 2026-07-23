"""Shared validation for backed read and write chunk controls."""

from __future__ import annotations

from numbers import Integral


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

    ``None`` preserves the historical behavior by using the compute chunk
    size for write-heavy phases as well.
    """
    compute_chunk_size = validate_chunk_size(
        backed_chunk_size,
        name="backed_chunk_size",
    )
    if backed_write_chunk_size is None:
        return compute_chunk_size, compute_chunk_size
    write_chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )
    return compute_chunk_size, write_chunk_size
