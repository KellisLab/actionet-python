"""Helpers for inspecting and handling backed HDF5 compression metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional

from anndata import AnnData


def _decode_codec(value: Any) -> Any:
    """Decode HDF5 codec values to plain Python scalars."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _dataset_compression_info(dataset: Any) -> Dict[str, Any]:
    """Return compression metadata for an h5py-like dataset."""
    return {
        "compression": _decode_codec(getattr(dataset, "compression", None)),
        "compression_opts": getattr(dataset, "compression_opts", None),
    }


def _is_sparse_group(node: Any) -> bool:
    """Return True when *node* looks like an on-disk sparse matrix group."""
    if not hasattr(node, "keys"):
        return False
    keys = set(node.keys())
    return {"data", "indices", "indptr"}.issubset(keys)


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
