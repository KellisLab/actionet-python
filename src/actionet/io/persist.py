"""In-memory + backed-disk annotation persistence.

This is one third of the former ``_backed_persist.py``. It covers:

1. **In-memory updates** — every API function stores its results on the
   AnnData object so they are immediately visible to the caller.
2. **Disk persistence** — when the AnnData is in backed mode (HDF5), the
   same results are written to the underlying file via the
   ``anndata_io.append_to_anndata`` writer so that closing and re-opening
   the file preserves the computed annotations.

The other two thirds live in :mod:`actionet.io.checkpoint` (checkpoint /
compact) and :mod:`actionet.io.subset` (structural rewrites).

The typical call site is simply::

    persist_updates(adata, obsm={"key": array}, uns={"key": value})

which transparently does both steps.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from typing import Any, Mapping, MutableMapping

import anndata as ad
from anndata import AnnData

from . import anndata_io
from .backed_adapter import BackedAnnDataAdapter, init_from_reopened


# ---------------------------------------------------------------------------
# Dirty-key tracking: records which annotation keys have been written since
# the last checkpoint, so checkpoint_backed only rewrites what changed.
# ---------------------------------------------------------------------------


class _DirtyTracker:
    """Per-object tracker of annotation keys modified since last flush."""

    def __init__(self):
        self._dirty: dict[int, dict[str, set[str]]] = {}

    def mark(self, adata: AnnData, slot: str, keys: set[str]) -> None:
        """Record that *keys* within *slot* have been modified."""
        obj_id = id(adata)
        if obj_id not in self._dirty:
            self._dirty[obj_id] = defaultdict(set)
            weakref.finalize(adata, self._dirty.pop, obj_id, None)
        self._dirty[obj_id][slot].update(keys)

    def get_dirty(self, adata: AnnData) -> dict[str, set[str]]:
        """Return dirty slots/keys for *adata*, or empty dict if clean."""
        return self._dirty.get(id(adata), {})

    def clear(self, adata: AnnData) -> None:
        """Mark *adata* as fully flushed."""
        self._dirty.pop(id(adata), None)


_dirty_tracker = _DirtyTracker()


def is_backed_adata(adata: AnnData) -> bool:
    """Return True when AnnData is backed and has a filename."""
    return bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None))


def _real_layer_keys(adata: AnnData) -> list:
    """Return real layer keys, filtering out the anndata >= 0.13 None alias for X."""
    if is_backed_adata(adata):
        return BackedAnnDataAdapter(adata).real_layer_keys()
    return [k for k in adata.layers.keys() if k is not None]


def _ensure_backed_open(adata: AnnData) -> None:
    """Reopen the backing HDF5 file if anndata silently closed it.

    Delegates to :func:`operator._flush_backed_handle` which performs
    the reopen-then-flush sequence.  Persist call-sites need only the
    reopen (the flush is a harmless no-op when nothing is dirty), so
    this thin wrapper keeps call-sites readable.
    """
    if not is_backed_adata(adata):
        return
    from .operator import _flush_backed_handle
    _flush_backed_handle(adata, context="persist")


def set_auto_persist(adata: AnnData, enabled: bool = True) -> None:
    """Control whether backed disk writes happen automatically.

    When *enabled* is ``False``, :func:`persist_updates` applies changes
    in-memory only and defers the HDF5 write until an explicit
    :func:`checkpoint_backed` call or a structural operation that requires
    on-disk consistency (subset, materialize, etc.).

    The default (``True``) preserves the existing behaviour: every
    ``persist_updates`` call writes to the backing file immediately.

    Parameters
    ----------
    adata : AnnData
        Target object (backed or in-memory -- the flag is simply stored).
    enabled : bool, optional (default: True)
        ``True`` for immediate writes, ``False`` for deferred mode.
    """
    adata.uns["_actionet_auto_persist"] = bool(enabled)


def get_auto_persist(adata: AnnData) -> bool:
    """Return the current auto-persist setting for *adata*.

    Returns ``True`` (immediate writes) unless the flag has been
    explicitly set to ``False`` via :func:`set_auto_persist`.
    """
    return bool(adata.uns.get("_actionet_auto_persist", True))


def is_writable_backed(adata: AnnData) -> bool:
    """Return True when *adata* is backed and its file allows writes.

    Detects files opened in a mode containing ``"+"`` (e.g. ``"r+"``), which
    is the mode AnnData uses for writable backed access. Returns ``False``
    for in-memory AnnData or for files opened read-only.
    """
    if not bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None)):
        return False
    try:
        return BackedAnnDataAdapter(adata).writable
    except Exception:
        return False


def _ensure_backed_writable(adata: AnnData) -> None:
    """Raise if backed AnnData appears to be read-only."""
    if not is_backed_adata(adata):
        return

    if not BackedAnnDataAdapter(adata).writable:
        raise ValueError(
            "Backed AnnData was opened read-only (mode='r'). "
            "Re-open with backed='r+' to persist updates."
        )


def _refresh_backed_handle(adata: AnnData, path: str, mode: str = "r+") -> None:
    """Close and re-open a backed AnnData handle in-place."""
    BackedAnnDataAdapter(adata).close()
    reopened = ad.read_h5ad(path, backed=mode)
    init_from_reopened(adata, reopened)


def _init_from_reopened(adata: AnnData, reopened: AnnData) -> None:
    """Reinitialize *adata* from *reopened*, handling backed-raw edge cases.

    Passing *reopened* directly as ``X`` to ``_init_as_actual`` triggers a
    ValueError under ``anndata >= 0.13`` when the source is backed:
    ``reopened.X`` and ``reopened.layers[None]`` are distinct wrapper
    instances returned freshly from the file each access, so anndata's
    ``X is layers[None]`` identity check inside ``_init_as_actual``
    always fails.

    Route around it by unpacking *reopened* into explicit kwargs, driving
    the "init from file" branch (so ``layers.isbacked`` becomes ``True``
    and ``X`` is served from the on-disk dataset), and then adopting the
    reopened file handle so we don't leak the auxiliary one that
    ``_init_as_actual`` opens.

    ``raw`` handling: passing a :class:`~anndata.Raw` instance alongside
    ``filename`` trips an anndata assertion, and passing ``None`` when
    the file has a raw group crashes on ``dict(X=None, **None)``. Pass a
    ``{"var": raw.var, "varm": raw.varm}`` mapping and let the file-init
    branch resolve ``raw.X`` from disk.
    """
    init_from_reopened(adata, reopened)


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


def _is_live_backed_wrapper(value: Any) -> bool:
    """Return True for a live HDF5-backed matrix wrapper (never a snapshot).

    AnnData exposes ``CSRDataset`` / ``CSCDataset`` (and, in some releases,
    experimental variants) that hold an open ``h5py`` handle rather than an
    in-memory array. Such a wrapper must never be captured into the results
    dict: ``persist_updates`` closes the source file before
    ``append_to_anndata`` re-serializes the results, at which point the
    wrapper's handle is orphaned. These matrices are already carried through
    the rewrite by the full-file ``rewrite_h5ad_payload`` copy, so they need
    no re-serialization here.
    """
    csr_type = getattr(getattr(ad, "abc", None), "CSRDataset", ())
    csc_type = getattr(getattr(ad, "abc", None), "CSCDataset", ())
    backed_sparse_types = tuple(
        cls for cls in (csr_type, csc_type) if isinstance(cls, type)
    )
    if backed_sparse_types and isinstance(value, backed_sparse_types):
        return True
    # Experimental / unversioned backed wrappers: identify by a live HDF5
    # group or dataset handle rather than a stable base class.
    if hasattr(value, "group") and getattr(value, "group", None) is not None:
        return True
    return False


def _include_all_inmemory_annotations(adata: AnnData, results: dict) -> None:
    """Augment *results* with all in-memory annotations not already present.

    Because ``append_to_anndata`` performs an atomic full-file rewrite (copy
    source + apply updates), any in-memory state not included in *results* is
    lost when the file is replaced and the handle refreshed.  This function
    ensures that user modifications made directly on the AnnData (bypassing
    ``persist_updates``) are preserved through the rewrite cycle.

    Keys already present in *results* (freshly computed by the calling ACTIONet
    function) take priority and are never overwritten.

    Live HDF5-backed matrix wrappers (``CSRDataset`` / ``CSCDataset`` and
    experimental variants) are skipped: the full-file rewrite already copies
    them, and snapshotting them here would hand an orphaned handle to
    ``ad.io.write_elem`` after the source file is closed.
    """
    for col in adata.obs.columns:
        if col not in results["obs_columns"]:
            results["obs_columns"][col] = adata.obs[col]

    for col in adata.var.columns:
        if col not in results["var_columns"]:
            results["var_columns"][col] = adata.var[col]

    for slot, key in [("obsm", "obsm_keys"), ("varm", "varm_keys"),
                      ("obsp", "obsp_keys"), ("varp", "varp_keys")]:
        container = getattr(adata, slot)
        for k in container.keys():
            if k in results[key]:
                continue
            value = container[k]
            if _is_live_backed_wrapper(value):
                continue
            results[key][k] = value

    for k in _real_layer_keys(adata):
        if k in results["layers_keys"]:
            continue
        value = adata.layers[k]
        if _is_live_backed_wrapper(value):
            continue
        results["layers_keys"][k] = value

    for k, v in adata.uns.items():
        if k not in results["uns_keys"]:
            results["uns_keys"][k] = v


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
    calls ``anndata_io.append_to_anndata`` to persist results in the
    underlying HDF5 file.

    Parameters
    ----------
    adata : AnnData
        Target object.
    obs, var, obsm, varm, obsp, varp, layers, uns : dict, optional
        Mappings of keys to values for each AnnData slot.
    validate : bool
        Run ``anndata_io`` validation before writing (backed only).
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

    if obs:
        _dirty_tracker.mark(adata, "obs_columns", set(obs.keys()))
    if var:
        _dirty_tracker.mark(adata, "var_columns", set(var.keys()))
    if obsm:
        _dirty_tracker.mark(adata, "obsm_keys", set(obsm.keys()))
    if varm:
        _dirty_tracker.mark(adata, "varm_keys", set(varm.keys()))
    if obsp:
        _dirty_tracker.mark(adata, "obsp_keys", set(obsp.keys()))
    if varp:
        _dirty_tracker.mark(adata, "varp_keys", set(varp.keys()))
    if layers:
        # Drop the anndata >= 0.13 ``layers[None]`` alias for ``.X`` so the
        # dirty tracker never asks callers to persist a phantom layer key.
        _dirty_tracker.mark(
            adata,
            "layers_keys",
            {key for key in layers.keys() if key is not None},
        )
    if uns:
        _dirty_tracker.mark(adata, "uns_keys", set(uns.keys()))

    if not get_auto_persist(adata):
        return

    filepath = str(adata.filename)

    _include_all_inmemory_annotations(adata, results)

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    anndata_io.append_to_anndata(
        filepath,
        results,
        verbose=verbose,
        validate=validate,
    )

    _refresh_backed_handle(adata, filepath, mode="r+")


def _flush_pending(adata: AnnData) -> None:
    """Flush deferred in-memory changes to the backing file.

    No-op when any of the following is true:
    - *adata* is not backed.
    - ``auto_persist`` is ``True`` (writes already happened eagerly).
    - No dirty keys have been recorded.

    Called automatically by structural operations that need the HDF5
    file to be up-to-date before they read or rewrite it.
    """
    if not is_backed_adata(adata):
        return
    if get_auto_persist(adata):
        return
    dirty = _dirty_tracker.get_dirty(adata)
    if not dirty:
        return

    _ensure_backed_open(adata)

    _ensure_backed_writable(adata)

    results = anndata_io.collect_annotation_results(
        adata,
        obs_columns=sorted(dirty.get("obs_columns", set())),
        var_columns=sorted(dirty.get("var_columns", set())),
        obsm_keys=sorted(dirty.get("obsm_keys", set())),
        varm_keys=sorted(dirty.get("varm_keys", set())),
        obsp_keys=sorted(dirty.get("obsp_keys", set())),
        varp_keys=sorted(dirty.get("varp_keys", set())),
        layers_keys=sorted(dirty.get("layers_keys", set())),
        uns_keys=sorted(dirty.get("uns_keys", set())),
        verbose=False,
    )

    has_data = any(len(v) > 0 for v in results.values())
    if not has_data:
        _dirty_tracker.clear(adata)
        return

    filepath = str(adata.filename)

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    anndata_io.append_to_anndata(filepath, results, verbose=False, validate=False)
    _refresh_backed_handle(adata, filepath, mode="r+")
    _dirty_tracker.clear(adata)
