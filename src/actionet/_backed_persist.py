"""Backed AnnData persistence helpers.

Centralises two concerns:

1. **In-memory updates** -- every API function stores its results on the
   AnnData object so they are immediately visible to the caller.
2. **Disk persistence** -- when the AnnData is in backed mode (HDF5), the
   same results are written to the underlying file via the experimental
   ``_anndata_io.append_to_anndata`` writer so that closing and re-opening
   the file preserves the computed annotations.

The typical call site is simply::

    persist_updates(adata, obsm={"key": array}, uns={"key": value})

which transparently does both steps.

For a full flush of all in-memory state, use :func:`checkpoint_backed`.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Mapping, MutableMapping

from anndata import AnnData

try:
    from .experimental import _anndata_io
except Exception:  # pragma: no cover - optional in some build contexts
    _anndata_io = None


def is_backed_adata(adata: AnnData) -> bool:
    """Return True when AnnData is backed and has a filename."""
    return bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None))


def _ensure_backed_writable(adata: AnnData) -> None:
    """Raise if backed AnnData appears to be read-only."""
    if not is_backed_adata(adata):
        return

    file_handle = getattr(getattr(adata, "file", None), "_file", None)
    mode = getattr(file_handle, "mode", None)
    if mode == "r":
        raise ValueError(
            "Backed AnnData was opened read-only (mode='r'). "
            "Re-open with backed='r+' to persist updates."
        )


def _as_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if values is None else dict(values)


def _assign_mapping(
    target: MutableMapping[str, Any],
    values: Mapping[str, Any],
    *,
    tolerate_errors: bool,
) -> None:
    for key, value in values.items():
        try:
            target[key] = value
        except Exception:
            if not tolerate_errors:
                raise


def apply_inmemory_updates(
    adata: AnnData,
    *,
    obs: Mapping[str, Any] | None = None,
    var: Mapping[str, Any] | None = None,
    obsm: Mapping[str, Any] | None = None,
    varm: Mapping[str, Any] | None = None,
    obsp: Mapping[str, Any] | None = None,
    varp: Mapping[str, Any] | None = None,
    layers: Mapping[str, Any] | None = None,
    uns: Mapping[str, Any] | None = None,
    tolerate_errors: bool | None = None,
) -> None:
    """Assign values to the in-memory AnnData object only (no disk write).

    This is the first half of the persist workflow.  Use
    :func:`persist_updates` instead when the caller also needs disk
    persistence for backed objects.

    Parameters
    ----------
    adata : AnnData
        Target object.
    obs, var : dict, optional
        Column name -> 1-D array-like mappings for ``adata.obs`` / ``adata.var``.
    obsm, varm, obsp, varp, layers, uns : dict, optional
        Key -> value mappings for the corresponding AnnData slots.
    tolerate_errors : bool or None
        If ``True``, silently skip assignments that raise.  ``None``
        (default) auto-enables tolerance for backed AnnData where some
        assignments may be unsupported.
    """
    if tolerate_errors is None:
        tolerate_errors = is_backed_adata(adata)

    obs = _as_mapping(obs)
    var = _as_mapping(var)
    obsm = _as_mapping(obsm)
    varm = _as_mapping(varm)
    obsp = _as_mapping(obsp)
    varp = _as_mapping(varp)
    layers = _as_mapping(layers)
    uns = _as_mapping(uns)

    for key, value in obs.items():
        try:
            adata.obs[key] = value
        except Exception:
            if not tolerate_errors:
                raise

    for key, value in var.items():
        try:
            adata.var[key] = value
        except Exception:
            if not tolerate_errors:
                raise

    _assign_mapping(adata.obsm, obsm, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.varm, varm, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.obsp, obsp, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.varp, varp, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.layers, layers, tolerate_errors=tolerate_errors)
    _assign_mapping(adata.uns, uns, tolerate_errors=tolerate_errors)


def persist_updates(
    adata: AnnData,
    *,
    obs: Mapping[str, Any] | None = None,
    var: Mapping[str, Any] | None = None,
    obsm: Mapping[str, Any] | None = None,
    varm: Mapping[str, Any] | None = None,
    obsp: Mapping[str, Any] | None = None,
    varp: Mapping[str, Any] | None = None,
    layers: Mapping[str, Any] | None = None,
    uns: Mapping[str, Any] | None = None,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """Apply updates in-memory and, for backed AnnData, write them to disk.

    This is the primary entry point used by all public API functions to
    store computed results.  For in-memory objects it is equivalent to
    :func:`apply_inmemory_updates`.  For backed objects it additionally
    calls ``_anndata_io.append_to_anndata`` to persist results in the
    underlying HDF5 file.

    Parameters
    ----------
    adata : AnnData
        Target object.
    obs, var, obsm, varm, obsp, varp, layers, uns : dict, optional
        Mappings of keys to values for each AnnData slot.
    validate : bool
        Run ``_anndata_io`` validation before writing (backed only).
    verbose : bool
        Print progress messages during disk writes (backed only).
    """
    obs = _as_mapping(obs)
    var = _as_mapping(var)
    obsm = _as_mapping(obsm)
    varm = _as_mapping(varm)
    obsp = _as_mapping(obsp)
    varp = _as_mapping(varp)
    layers = _as_mapping(layers)
    uns = _as_mapping(uns)

    apply_inmemory_updates(
        adata,
        obs=obs,
        var=var,
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        varp=varp,
        layers=layers,
        uns=uns,
        tolerate_errors=is_backed_adata(adata),
    )

    if not is_backed_adata(adata):
        return

    if _anndata_io is None:
        raise RuntimeError(
            "Backed persistence requested but actionet.experimental._anndata_io "
            "could not be imported."
        )

    _ensure_backed_writable(adata)

    results = {
        "obs_columns": obs,
        "var_columns": var,
        "obsm_keys": obsm,
        "varm_keys": varm,
        "obsp_keys": obsp,
        "varp_keys": varp,
        "layers_keys": layers,
        "uns_keys": uns,
    }

    if not any(len(v) > 0 for v in results.values()):
        return

    _anndata_io.append_to_anndata(
        str(adata.filename),
        results,
        verbose=verbose,
        validate=validate,
    )


def persist_layer(
    adata: AnnData,
    layer: str,
    matrix: Any,
    *,
    validate: bool = False,
    verbose: bool = False,
) -> None:
    """Convenience wrapper: persist a single layer matrix to a backed file."""
    persist_updates(
        adata,
        layers={layer: matrix},
        validate=validate,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Checkpoint / compact
# ---------------------------------------------------------------------------


def _copy_h5_group_faithful(src_group, dst_group, chunk_size: int) -> None:
    """Recursively copy an HDF5 group preserving compression settings."""
    import h5py

    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            _copy_h5_group_faithful(obj, child, chunk_size=chunk_size)
        elif isinstance(obj, h5py.Dataset):
            kwargs: dict[str, Any] = {
                "shape": obj.shape,
                "dtype": obj.dtype,
            }
            if obj.chunks is not None:
                kwargs["chunks"] = obj.chunks
            if obj.maxshape is not None:
                kwargs["maxshape"] = obj.maxshape
            if obj.compression is not None:
                kwargs["compression"] = obj.compression
            if obj.compression_opts is not None:
                kwargs["compression_opts"] = obj.compression_opts
            if obj.shuffle:
                kwargs["shuffle"] = True
            if obj.fletcher32:
                kwargs["fletcher32"] = True

            dst_ds = dst_group.create_dataset(name, **kwargs)

            # Chunked copy along axis-0.
            if obj.shape == () or obj.ndim == 0:
                dst_ds[()] = obj[()]
            else:
                n_rows = obj.shape[0]
                step = max(1, chunk_size)
                for start in range(0, n_rows, step):
                    end = min(start + step, n_rows)
                    dst_ds[start:end, ...] = obj[start:end, ...]

            for key, value in obj.attrs.items():
                dst_ds.attrs[key] = value
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
            _copy_h5_group_faithful(src_f, dst_f, chunk_size=chunk_size)

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        os.replace(tmp_path, src_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    import anndata as ad

    reopened = ad.read_h5ad(src_path, backed="r+")
    adata._init_as_actual(reopened)
    adata.file = reopened.file

    if verbose:
        print(f"[INFO] Compacted {src_path}")


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
        Run ``_anndata_io`` validation before writing.
    verbose : bool, optional (default: False)
        Print progress messages.

    Raises
    ------
    ValueError
        If *adata* is not backed or is opened read-only.
    RuntimeError
        If the experimental IO module is unavailable.
    """
    if not is_backed_adata(adata):
        raise ValueError(
            "checkpoint_backed requires a backed AnnData object. "
            "Open with ad.read_h5ad(path, backed='r+')."
        )

    _ensure_backed_writable(adata)

    if _anndata_io is None:
        raise RuntimeError(
            "Backed persistence requested but actionet.experimental._anndata_io "
            "could not be imported."
        )

    results = _anndata_io.collect_annotation_results(
        adata,
        obs_columns=list(adata.obs.columns),
        var_columns=list(adata.var.columns),
        obsm_keys=list(adata.obsm.keys()),
        varm_keys=list(adata.varm.keys()),
        obsp_keys=list(adata.obsp.keys()),
        varp_keys=list(adata.varp.keys()),
        layers_keys=[],
        uns_keys=list(adata.uns.keys()),
        verbose=verbose,
    )

    has_data = any(len(v) > 0 for v in results.values())

    if has_data:
        if verbose:
            print(f"[INFO] Checkpointing to {adata.filename}")

        _anndata_io.append_to_anndata(
            str(adata.filename),
            results,
            verbose=verbose,
            validate=validate,
        )

    if compact:
        _repack_h5ad(adata, chunk_size=chunk_size, verbose=verbose)
