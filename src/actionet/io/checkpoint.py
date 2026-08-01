"""Checkpoint & compact backed AnnData files.

This is one third of the former ``_backed_persist.py``. It provides
:func:`checkpoint_backed` (flush + optional repack) and the HDF5
group/dataset copy primitives it uses.
"""

from __future__ import annotations

import os
from typing import Any

import anndata as ad
from anndata import AnnData

from . import anndata_io
from .chunking import (
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    validate_chunk_size,
)
from .backed_adapter import BackedAnnDataAdapter
from .native_h5ad import (
    NativeCapabilityError,
    backed_io_engine,
    native_copy_matrix,
    native_layout_capability,
)
from .rewrite import RewriteTransaction
from .persist import (
    _dirty_tracker,
    _ensure_backed_writable,
    _include_all_inmemory_annotations,
    _real_layer_keys,
    _refresh_backed_handle,
    is_backed_adata,
)


def _copy_h5_attrs(src, dst) -> None:
    """Copy all HDF5 attrs from *src* to *dst* (shared helper)."""
    anndata_io.copy_h5_attrs(src, dst)


def _copy_h5_dataset_chunked(src_ds, dst_ds, chunk_size: int) -> None:
    """Copy an h5py dataset's contents into an existing dataset in chunks."""
    if src_ds.shape is None:
        # HDF5 null dataspace: the destination was created with shape=None and
        # deliberately has no value to transfer.
        return
    if src_ds.shape == () or src_ds.ndim == 0:
        dst_ds[()] = src_ds[()]
        return

    n_rows = src_ds.shape[0]
    if n_rows == 0:
        return

    step = int(max(1, chunk_size))
    for start in range(0, n_rows, step):
        end = min(start + step, n_rows)
        dst_ds[start:end, ...] = src_ds[start:end, ...]


def _dataset_kwargs_mirror(src_ds, *, preserve_compression: bool) -> dict:
    """Build create_dataset kwargs mirroring *src_ds*'s layout.

    When ``preserve_compression`` is True, all codec settings
    (``compression``, ``compression_opts``, ``shuffle``, ``fletcher32``)
    are copied. When False, only shape/dtype/chunks/maxshape are kept.
    """
    kwargs: dict[str, Any] = {
        "shape": src_ds.shape,
        "dtype": src_ds.dtype,
    }
    if src_ds.chunks is not None:
        kwargs["chunks"] = src_ds.chunks
    if src_ds.maxshape is not None:
        kwargs["maxshape"] = src_ds.maxshape

    if preserve_compression:
        if src_ds.compression is not None:
            kwargs["compression"] = src_ds.compression
        if src_ds.compression_opts is not None:
            kwargs["compression_opts"] = src_ds.compression_opts
        if getattr(src_ds, "shuffle", False):
            kwargs["shuffle"] = True
        if getattr(src_ds, "fletcher32", False):
            kwargs["fletcher32"] = True
    return kwargs


def copy_h5_group(
    src_group,
    dst_group,
    *,
    chunk_size: int,
    preserve_compression: bool,
) -> None:
    """Compatibility wrapper for recursively copying a generic HDF5 group.

    ACTIONet's H5AD rewrite paths use :func:`rewrite_h5ad_payload`; this
    helper remains available for callers that imported the previous public
    utility.
    """
    import h5py

    _copy_h5_attrs(src_group, dst_group)
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            copy_h5_group(
                obj,
                child,
                chunk_size=chunk_size,
                preserve_compression=preserve_compression,
            )
        elif isinstance(obj, h5py.Dataset):
            kwargs = _dataset_kwargs_mirror(
                obj,
                preserve_compression=preserve_compression,
            )
            destination = dst_group.create_dataset(name, **kwargs)
            _copy_h5_dataset_chunked(
                obj,
                destination,
                chunk_size=chunk_size,
            )
            _copy_h5_attrs(obj, destination)
        else:
            raise TypeError(
                f"Unsupported HDF5 object type for key '{name}': {type(obj)}"
            )


def _h5ad_matrix_candidate(obj) -> bool:
    """Recognize only versioned numeric H5AD matrix encodings."""
    import h5py
    import numpy as np

    encoding = obj.attrs.get("encoding-type", "")
    version = obj.attrs.get("encoding-version", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8", errors="replace")
    if isinstance(version, bytes):
        version = version.decode("utf-8", errors="replace")

    if isinstance(obj, h5py.Dataset):
        return bool(
            encoding == "array"
            and version == "0.2.0"
            and obj.ndim == 2
            and np.issubdtype(obj.dtype, np.number)
        )
    return bool(
        isinstance(obj, h5py.Group)
        and encoding in {"csr_matrix", "csc_matrix"}
        and version == "0.1.0"
        and {"data", "indices", "indptr"}.issubset(obj.keys())
    )


def _collect_h5ad_matrix_paths(group, prefix: str = "") -> list[str]:
    import h5py

    paths: list[str] = []
    for name, obj in group.items():
        path = f"{prefix}/{name}" if prefix else f"/{name}"
        if _h5ad_matrix_candidate(obj):
            paths.append(path)
        elif isinstance(obj, h5py.Group):
            paths.extend(_collect_h5ad_matrix_paths(obj, path))
    return paths


def rewrite_h5ad_payload(
    source_path: str,
    destination_path: str,
    *,
    chunk_size: int,
    uncompressed_paths: set[str] | None = None,
    native_matrix_paths: set[str] | None = None,
    omit_paths: set[str] | None = None,
) -> None:
    """Copy one H5AD payload, deferring supported matrices to libactionet.

    ``uncompressed_paths=None`` preserves every dataset layout. A set
    decompresses only those matrix/group paths; ``{"/"}`` decompresses the
    complete file. ``native_matrix_paths=None`` allows every discovered matrix
    to use the native engine, while a set restricts native transfer to those
    paths.
    """
    import h5py

    source_path = os.path.realpath(os.fspath(source_path))
    destination_path = os.path.realpath(os.fspath(destination_path))
    requested_uncompressed = set(uncompressed_paths or ())
    omitted = set(omit_paths or ())
    decompress_all = "/" in requested_uncompressed

    def _is_uncompressed(path: str) -> bool:
        if decompress_all:
            return True
        return any(
            path == target or path.startswith(target.rstrip("/") + "/")
            for target in requested_uncompressed
        )

    with h5py.File(source_path, "r") as source:
        matrix_paths = _collect_h5ad_matrix_paths(source)

    allowed_native = (
        set(matrix_paths)
        if native_matrix_paths is None
        else set(matrix_paths).intersection(native_matrix_paths)
    )
    allowed_native.difference_update(omitted)
    native_paths: set[str] = set()
    engine = backed_io_engine()
    if engine != "python":
        from .. import _core

        for path in sorted(allowed_native):
            try:
                info = dict(_core.h5ad_inspect_matrix(source_path, path))
            except Exception:
                if engine == "native":
                    raise
            else:
                supported, reason = native_layout_capability(
                    info,
                    preserve_layout=not _is_uncompressed(path),
                )
                if supported:
                    native_paths.add(path)
                elif engine == "native":
                    raise NativeCapabilityError(
                        f"native H5AD transfer rejected {path}: {reason}"
                    )

    def _copy_group(source_group, destination_group, prefix: str = "") -> None:
        _copy_h5_attrs(source_group, destination_group)
        known_top_level = {
            "X",
            "obs",
            "var",
            "layers",
            "obsm",
            "varm",
            "obsp",
            "varp",
            "uns",
            "raw",
        }
        for name, obj in source_group.items():
            path = f"{prefix}/{name}" if prefix else f"/{name}"
            if path in omitted:
                continue
            if path in native_paths:
                continue
            if not prefix and name not in known_top_level:
                source_group.copy(name, destination_group, name=name)
                continue
            if isinstance(obj, h5py.Group):
                child = destination_group.create_group(name)
                _copy_group(obj, child, path)
                continue
            if not isinstance(obj, h5py.Dataset):
                raise TypeError(
                    f"Unsupported HDF5 object type for key '{path}': {type(obj)}"
                )
            kwargs = _dataset_kwargs_mirror(
                obj,
                preserve_compression=not _is_uncompressed(path),
            )
            destination_dataset = destination_group.create_dataset(name, **kwargs)
            _copy_h5_dataset_chunked(
                obj,
                destination_dataset,
                chunk_size=chunk_size,
            )
            _copy_h5_attrs(obj, destination_dataset)

    with h5py.File(source_path, "r") as source, h5py.File(
        destination_path, "w"
    ) as destination:
        _copy_group(source, destination)

    for path in sorted(native_paths):
        native_copy_matrix(
            source_path,
            path,
            destination_path,
            path,
            max_rows_per_batch=chunk_size,
            preserve_layout=not _is_uncompressed(path),
        )


def _repack_h5ad(
    adata: AnnData,
    *,
    chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    verbose: bool = False,
) -> None:
    """Repack a backed H5AD file to reclaim dead space, then refresh the handle.

    Performs an atomic copy to a temp file, replaces the original, and
    re-opens the AnnData handle so it points at the compacted file.
    """
    adapter = BackedAnnDataAdapter(adata)
    src_path = adapter.filename
    original_mode = adapter.mode
    with RewriteTransaction(src_path, src_path) as transaction:
        rewrite_h5ad_payload(
            src_path,
            transaction.temp_path,
            chunk_size=chunk_size,
        )
        validated = ad.read_h5ad(transaction.temp_path, backed="r")
        file_handle = getattr(validated, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass
        transaction.commit(
            close_source=adapter.close,
            restore_source=lambda: adapter.reopen(mode=original_mode),
        )
    adapter.reopen(mode=original_mode)

    if verbose:
        print(f"[INFO] Compacted {src_path}")


def _checkpoint_collect_args(adata: AnnData) -> dict:
    """Build collect_annotation_results kwargs using only dirty keys.

    If nothing is dirty (e.g. user calls checkpoint without prior persist_updates),
    falls back to collecting all annotation keys for a full flush.
    """
    dirty = _dirty_tracker.get_dirty(adata)

    if not dirty:
        return {
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "obsm_keys": list(adata.obsm.keys()),
            "varm_keys": list(adata.varm.keys()),
            "obsp_keys": list(adata.obsp.keys()),
            "varp_keys": list(adata.varp.keys()),
            "layers_keys": _real_layer_keys(adata),
            "uns_keys": list(adata.uns.keys()),
        }

    return {
        "obs_columns": sorted(dirty.get("obs_columns", set())),
        "var_columns": sorted(dirty.get("var_columns", set())),
        "obsm_keys": sorted(dirty.get("obsm_keys", set())),
        "varm_keys": sorted(dirty.get("varm_keys", set())),
        "obsp_keys": sorted(dirty.get("obsp_keys", set())),
        "varp_keys": sorted(dirty.get("varp_keys", set())),
        "layers_keys": sorted(dirty.get("layers_keys", set())),
        "uns_keys": sorted(dirty.get("uns_keys", set())),
    }


def checkpoint_backed(
    adata: AnnData,
    *,
    compact: bool = False,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """Flush all in-memory annotations to the backing HDF5 file.

    This is the recommended way to checkpoint a backed AnnData object.
    Unlike ``adata.write_h5ad()``, which rewrites the full object and
    nearly doubles file size due to HDF5 dead-space accumulation,
    ``checkpoint_backed`` writes only in-memory annotation slots and
    optionally repacks the file to reclaim any dead space.

    Parameters
    ----------
    adata : AnnData
        A backed AnnData object opened in ``r+`` mode.
    compact : bool, optional (default: False)
        If ``True``, repack the HDF5 file after writing to reclaim
        dead space from prior delete-then-create overwrites.  This
        requires a full file copy and is expensive for large files.
    backed_write_chunk_size : int, optional (default: 16384)
        Row/element chunk size for the full-file payload copy. It applies
        both to the annotation-append rewrite performed on every checkpoint
        and to the optional ``compact`` repack. Atlas-scale files may benefit
        from starting with ``32768``; larger values use proportionally more
        temporary memory.
    validate : bool, optional (default: False)
        Run ``anndata_io`` validation before writing.
    verbose : bool, optional (default: False)
        Print progress messages.

    Raises
    ------
    ValueError
        If *adata* is not backed or is opened read-only.
    RuntimeError
        If the annotation IO module is unavailable.
    """
    backed_write_chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )
    if not is_backed_adata(adata):
        raise ValueError(
            "checkpoint_backed requires a backed AnnData object. "
            "Open with ad.read_h5ad(path, backed='r+')."
        )

    _ensure_backed_writable(adata)

    results = anndata_io.collect_annotation_results(
        adata,
        **_checkpoint_collect_args(adata),
        verbose=verbose,
    )

    _include_all_inmemory_annotations(adata, results)

    has_data = any(len(v) > 0 for v in results.values())

    if has_data:
        if verbose:
            print(f"[INFO] Checkpointing to {adata.filename}")

        filepath = str(adata.filename)

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        anndata_io.append_to_anndata(
            filepath,
            results,
            verbose=verbose,
            validate=validate,
            chunk_size=backed_write_chunk_size,
        )

        _refresh_backed_handle(adata, filepath, mode="r+")

    _dirty_tracker.clear(adata)

    if compact:
        _repack_h5ad(adata, chunk_size=backed_write_chunk_size, verbose=verbose)
