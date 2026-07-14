"""Checkpoint & compact backed AnnData files.

This is one third of the former ``_backed_persist.py``. It provides
:func:`checkpoint_backed` (flush + optional repack) and the HDF5
group/dataset copy primitives it uses.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import anndata as ad
from anndata import AnnData

from . import anndata_io
from .persist import (
    _dirty_tracker,
    _ensure_backed_writable,
    _include_all_inmemory_annotations,
    _init_from_reopened,
    _real_layer_keys,
    _refresh_backed_handle,
    is_backed_adata,
)


def _copy_h5_attrs(src, dst) -> None:
    """Copy all HDF5 attrs from *src* to *dst*."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_h5_dataset_chunked(src_ds, dst_ds, chunk_size: int) -> None:
    """Copy an h5py dataset's contents into an existing dataset in chunks."""
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
    """Recursively copy an HDF5 group into *dst_group*.

    Parameters
    ----------
    src_group, dst_group
        h5py groups (or files) — source and destination.
    chunk_size
        Row chunk size for streaming dataset copies (axis-0).
    preserve_compression
        When True, faithfully preserves compression codec, opts, shuffle
        and fletcher32 filters. When False, destination datasets are
        written uncompressed but keep the original ``chunks``/``maxshape``.
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
                obj, preserve_compression=preserve_compression
            )
            dst_ds = dst_group.create_dataset(name, **kwargs)
            _copy_h5_dataset_chunked(obj, dst_ds, chunk_size=chunk_size)
            _copy_h5_attrs(obj, dst_ds)
        else:
            raise TypeError(
                f"Unsupported HDF5 object type for key '{name}': {type(obj)}"
            )


def _repack_h5ad(
    adata: AnnData,
    *,
    chunk_size: int = 4096,
    verbose: bool = False,
) -> None:
    """Repack a backed H5AD file to reclaim dead space, then refresh the handle.

    Performs an atomic copy to a temp file, replaces the original, and
    re-opens the AnnData handle so it points at the compacted file.
    """
    import h5py

    src_path = str(adata.filename)

    parent_dir = os.path.dirname(src_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".h5ad", dir=parent_dir, prefix=".compact_"
    )
    os.close(fd)

    try:
        with h5py.File(src_path, "r") as src_f, h5py.File(tmp_path, "w") as dst_f:
            copy_h5_group(
                src_f,
                dst_f,
                chunk_size=chunk_size,
                preserve_compression=True,
            )

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        os.replace(tmp_path, src_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    reopened = ad.read_h5ad(src_path, backed="r+")
    _init_from_reopened(adata, reopened)

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
    chunk_size: int = 4096,
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
    chunk_size : int, optional (default: 4096)
        Row-chunk size used during the compact file copy.
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
        )

        _refresh_backed_handle(adata, filepath, mode="r+")

    _dirty_tracker.clear(adata)

    if compact:
        _repack_h5ad(adata, chunk_size=chunk_size, verbose=verbose)
