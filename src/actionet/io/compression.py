"""Helpers for inspecting and handling backed HDF5 compression metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from anndata import AnnData


def _dataset_compression_info(dataset: Any) -> Dict[str, Any]:
    """Return compression metadata for an h5py-like dataset."""
    raw_codec = getattr(dataset, "compression", None)
    codec = raw_codec.decode("utf-8", errors="ignore") if isinstance(raw_codec, bytes) else raw_codec
    return {
        "compression": codec,
        "compression_opts": getattr(dataset, "compression_opts", None),
    }


def _is_sparse_group(node: Any) -> bool:
    """Return True when *node* looks like an on-disk sparse matrix group."""
    if not hasattr(node, "keys"):
        return False
    keys = set(node.keys())
    return {"data", "indices", "indptr"}.issubset(keys)


def sparse_group_format(group: Any) -> Optional[str]:
    """Return ``'csr'`` or ``'csc'`` when *group* has a recognizable encoding.

    Inspects the ``encoding-type`` attribute (h5py or zarr-style) and returns
    the matching format string, else ``None``. Passes a raw h5py group or any
    object exposing an ``attrs`` mapping.
    """
    attrs = getattr(group, "attrs", None)
    if attrs is None:
        return None
    enc = attrs.get("encoding-type", "")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8", errors="ignore")
    if not isinstance(enc, str):
        return None
    enc = enc.lower()
    if "csr" in enc:
        return "csr"
    if "csc" in enc:
        return "csc"
    return None


def _normalize_matrix_key(matrix_key: Optional[str], fallback: str = "X") -> str:
    if matrix_key:
        return matrix_key
    return fallback


def get_storage_metadata_from_matrix(
    matrix: Any,
    *,
    matrix_key: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Inspect backed storage metadata for a matrix-like object.

    Parameters
    ----------
    matrix
        Backed matrix object (e.g. h5py dataset or anndata sparse backed object).
    matrix_key
        Optional logical key such as ``"X"`` or ``"layers/logcounts"``.
    filename
        Optional backing file path override.
    """
    # Raw h5py sparse group (e.g. file["X"] or file["layers/<name>"]).
    if _is_sparse_group(matrix):
        key = _normalize_matrix_key(matrix_key, getattr(matrix, "name", "").lstrip("/") or "X")
        path = filename or getattr(getattr(matrix, "file", None), "filename", None)
        datasets = {
            name: _dataset_compression_info(matrix[name])
            for name in ("data", "indices", "indptr")
            if name in matrix
        }
        return {
            "filename": path,
            "matrix_key": key,
            "is_sparse": True,
            "datasets": datasets,
        }

    # Backed sparse objects in anndata expose a .group pointing to the HDF5 group.
    group = getattr(matrix, "group", None)
    if group is not None and _is_sparse_group(group):
        key = _normalize_matrix_key(matrix_key, group.name.lstrip("/") or "X")
        path = filename or getattr(getattr(group, "file", None), "filename", None)
        datasets = {
            name: _dataset_compression_info(group[name])
            for name in ("data", "indices", "indptr")
            if name in group
        }
        return {
            "filename": path,
            "matrix_key": key,
            "is_sparse": True,
            "datasets": datasets,
        }

    # Backed dense matrices are h5py datasets.
    if hasattr(matrix, "compression"):
        dataset_name = getattr(matrix, "name", None)
        key = _normalize_matrix_key(matrix_key, dataset_name.lstrip("/") if dataset_name else "X")
        path = filename or getattr(getattr(matrix, "file", None), "filename", None)
        return {
            "filename": path,
            "matrix_key": key,
            "is_sparse": False,
            "datasets": {key: _dataset_compression_info(matrix)},
        }

    return None


def get_storage_metadata_from_adata(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Inspect backed storage metadata for ``adata.X`` or a backed layer."""
    if not bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None)):
        return None

    matrix_key = "X" if layer is None else f"layers/{layer}"
    matrix = adata.X if layer is None else adata.layers[layer]
    return get_storage_metadata_from_matrix(
        matrix,
        matrix_key=matrix_key,
        filename=str(adata.filename),
    )


def is_compressed_storage(metadata: Optional[Dict[str, Any]]) -> bool:
    """Return True when any dataset in *metadata* uses compression."""
    if not metadata:
        return False
    datasets = metadata.get("datasets", {})
    return any(details.get("compression") is not None for details in datasets.values())


def format_compression_summary(metadata: Optional[Dict[str, Any]]) -> str:
    """Format dataset compression codecs for warnings and logs."""
    if not metadata:
        return "unknown"

    parts = []
    for dataset_name, details in metadata.get("datasets", {}).items():
        codec = details.get("compression")
        codec_str = "none" if codec is None else str(codec)
        parts.append(f"{dataset_name}={codec_str}")

    if not parts:
        return "none"
    return ", ".join(parts)


def get_matrix_compression_policy(matrix: Any) -> Optional[Dict[str, Any]]:
    """Return compression policy used by the backed matrix datasets.

    Returns ``None`` for in-memory matrices or when compression metadata is
    unavailable.
    """
    metadata = get_storage_metadata_from_matrix(matrix)
    if not metadata:
        return None

    return {
        "is_sparse": bool(metadata.get("is_sparse", False)),
        "datasets": {
            name: {
                "compression": details.get("compression"),
                "compression_opts": details.get("compression_opts"),
            }
            for name, details in metadata.get("datasets", {}).items()
        },
    }


def _spec_to_create_kwargs(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate a single dataset compression spec into h5py create kwargs."""
    if not spec:
        return {}
    kwargs: Dict[str, Any] = {}
    codec = spec.get("compression")
    if codec is not None:
        kwargs["compression"] = codec
        if spec.get("compression_opts") is not None:
            kwargs["compression_opts"] = spec["compression_opts"]
    return kwargs


def write_sparse_csr_group_attrs(
    group,
    *,
    shape: tuple[int, int] | Sequence[int] | np.ndarray,
    encoding: str = "csr_matrix",
    version: str = "0.1.0",
) -> None:
    """Set the standard AnnData sparse-CSR group attrs on *group*.

    Writes ``shape``, ``encoding-type`` and ``encoding-version`` in the exact
    layout that AnnData and this package expect. Callers still create the
    ``data``/``indices``/``indptr`` datasets themselves because their
    compression/allocation strategy varies.
    """
    group.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    group.attrs["encoding-type"] = encoding
    group.attrs["encoding-version"] = version


@dataclass(frozen=True)
class CompressionPolicy:
    """Typed view over compression metadata for a backed matrix.

    A ``CompressionPolicy`` wraps the per-dataset compression codec/opts that
    should be preserved when writing subsetted/re-encoded copies of a backed
    matrix. Use :meth:`dense_kwargs` for dense datasets and
    :meth:`sparse_kwargs` for the ``data``/``indices``/``indptr`` triplet of
    a sparse group.

    Empty policies (no metadata available, or in-memory matrices) act as
    no-ops and return empty kwarg dicts.
    """

    is_sparse: bool = False
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "CompressionPolicy":
        return cls(is_sparse=False, datasets={})

    @classmethod
    def from_matrix(cls, matrix: Any) -> "CompressionPolicy":
        """Build a policy by inspecting a backed matrix (returns empty for in-memory)."""
        raw = get_matrix_compression_policy(matrix)
        if raw is None:
            return cls.empty()
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "CompressionPolicy":
        """Build a policy from its dict representation.

        Accepts the ``{"is_sparse": bool, "datasets": {...}}`` form used
        by internal writer sites (see :func:`_as_compression_policy`).
        Passing ``None`` or an empty dict yields :meth:`empty`.
        """
        if not raw:
            return cls.empty()
        return cls(
            is_sparse=bool(raw.get("is_sparse", False)),
            datasets=dict(raw.get("datasets", {}) or {}),
        )

    def dense_kwargs(self) -> Dict[str, Any]:
        """Return h5py create-dataset kwargs for a dense write.

        The first dataset entry is used (matching prior behaviour for
        single-dataset dense policies).
        """
        if not self.datasets:
            return {}
        return _spec_to_create_kwargs(next(iter(self.datasets.values())))

    def sparse_kwargs(self, name: str) -> Dict[str, Any]:
        """Return h5py create-dataset kwargs for a sparse component dataset.

        ``name`` should be one of ``"data"``, ``"indices"``, ``"indptr"``.
        """
        return _spec_to_create_kwargs(self.datasets.get(name))
