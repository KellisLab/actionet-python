"""Version-isolated access to AnnData's backed HDF5 lifecycle.

Only this module should know about AnnData's private file-manager details.
The data plane receives ordinary filenames and logical H5AD paths instead
of AnnData or h5py handles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
from anndata import AnnData


def _normalized_h5_path(path: str) -> str:
    return "/" + str(path).strip("/")


@dataclass(frozen=True)
class BackedMatrixLocation:
    """The stable path identity of one genuinely file-backed matrix."""

    filename: str
    h5_path: str


class BackedAnnDataAdapter:
    """Small compatibility boundary for AnnData 0.12/0.13 backed objects."""

    def __init__(self, adata: AnnData):
        if not bool(getattr(adata, "isbacked", False)):
            raise ValueError("AnnData object is not file-backed")
        self.adata = adata

    @property
    def filename(self) -> str:
        if not getattr(self.adata, "filename", None):
            raise RuntimeError("Backed AnnData does not expose a filename")
        return os.path.realpath(os.fspath(self.adata.filename))

    @property
    def file_handle(self):
        """Return the open h5py file, reopening through AnnData if necessary."""
        manager = getattr(self.adata, "file", None)
        if manager is None:
            raise RuntimeError("AnnData does not expose a backed file manager")
        if not getattr(manager, "is_open", False):
            mode = getattr(manager, "_filemode", None) or "r+"
            manager.open(filemode=mode)
        handle = getattr(manager, "_file", None)
        if handle is None:
            raise RuntimeError("AnnData did not provide an open HDF5 file handle")
        return handle

    def flush(self, *, context: str = "backed I/O") -> None:
        try:
            self.file_handle.flush()
        except Exception as exc:
            raise RuntimeError(
                f"{context}: failed to flush backed AnnData handle "
                f"({type(exc).__name__}: {exc})"
            ) from exc

    @property
    def mode(self) -> str:
        handle = self.file_handle
        mode = getattr(handle, "mode", None)
        if mode:
            return str(mode)
        manager = getattr(self.adata, "file", None)
        return str(getattr(manager, "_filemode", "r"))

    @property
    def writable(self) -> bool:
        return "+" in self.mode or self.mode in {"a", "w", "w-", "x"}

    def close(self) -> None:
        manager = getattr(self.adata, "file", None)
        if manager is not None:
            manager.close()

    def reopen(self, *, mode: str | None = None) -> None:
        """Refresh the existing Python object while preserving its identity."""
        reopen_mode = mode or ("r+" if self.writable else "r")
        self.close()
        reopened = ad.read_h5ad(self.filename, backed=reopen_mode)
        init_from_reopened(self.adata, reopened)

    def real_layer_keys(self) -> list[str]:
        """Hide AnnData 0.13's ``layers[None]`` alias for ``X``."""
        return [key for key in self.adata.layers.keys() if key is not None]

    def matrix_location(
        self,
        matrix: Any,
        expected_h5_path: str,
    ) -> BackedMatrixLocation | None:
        """Return a location only when *matrix* is truly backed at that path.

        In-memory NumPy/SciPy replacements deliberately return ``None`` even
        when an object with the same logical path remains in the source file;
        the in-memory value is authoritative and must use AnnData's codec.
        """
        import h5py

        expected = _normalized_h5_path(expected_h5_path)

        if isinstance(matrix, h5py.Dataset):
            try:
                filename = os.path.realpath(os.fspath(matrix.file.filename))
                actual = _normalized_h5_path(matrix.name)
            except Exception:
                return None
            if filename == self.filename and actual == expected:
                return BackedMatrixLocation(filename, expected)
            return None

        csr_type = getattr(getattr(ad, "abc", None), "CSRDataset", ())
        csc_type = getattr(getattr(ad, "abc", None), "CSCDataset", ())
        backed_sparse_types = tuple(
            cls for cls in (csr_type, csc_type) if isinstance(cls, type)
        )
        if backed_sparse_types and isinstance(matrix, backed_sparse_types):
            group = getattr(matrix, "group", None)
            try:
                filename = os.path.realpath(os.fspath(group.file.filename))
                actual = _normalized_h5_path(group.name)
            except Exception:
                return None
            if filename == self.filename and actual == expected:
                return BackedMatrixLocation(filename, expected)
            return None

        # Some AnnData releases expose experimental backed wrappers without a
        # stable public base class. Accept them only when their HDF5 group
        # proves both file and path identity.
        group = getattr(matrix, "group", None)
        if group is not None:
            try:
                filename = os.path.realpath(os.fspath(group.file.filename))
                actual = _normalized_h5_path(group.name)
            except Exception:
                return None
            if filename == self.filename and actual == expected:
                return BackedMatrixLocation(filename, expected)
        return None


def backed_view_selection(
    adata: AnnData,
) -> tuple[AnnData, np.ndarray, np.ndarray]:
    """Resolve a backed view to its parent and concrete axis selectors."""
    if not bool(getattr(adata, "is_view", False)):
        raise ValueError("AnnData object is not a view")
    parent = getattr(adata, "_adata_ref", None)
    if parent is None:
        raise RuntimeError("AnnData backed view does not expose its parent")

    def _indices(selector, size: int) -> np.ndarray:
        if isinstance(selector, slice):
            return np.arange(*selector.indices(size), dtype=np.int64)
        values = np.asarray(selector)
        if values.dtype == bool:
            return np.flatnonzero(values).astype(np.int64, copy=False)
        return values.astype(np.int64, copy=False).ravel()

    return (
        parent,
        _indices(getattr(adata, "_oidx"), parent.n_obs),
        _indices(getattr(adata, "_vidx"), parent.n_vars),
    )


def init_from_inmemory(adata: AnnData, source: AnnData) -> None:
    """Replace an AnnData object's in-memory contents while keeping identity."""
    adata._init_as_actual(source)


def init_from_reopened(adata: AnnData, reopened: AnnData) -> None:
    """Reinitialize an existing AnnData across 0.12/0.13 backed differences."""
    reopened_raw = getattr(reopened, "raw", None)
    if reopened_raw is not None and getattr(reopened_raw, "_X", None) is None:
        try:
            reopened_raw._X = reopened_raw.X
        except Exception:
            pass
    if reopened_raw is None:
        raw_arg = None
    else:
        reopened_raw_varm = getattr(reopened_raw, "varm", None)
        raw_arg = {
            "var": reopened_raw.var,
            "varm": dict(reopened_raw_varm) if reopened_raw_varm else None,
        }
    real_layers = {
        key: value
        for key, value in reopened.layers.items()
        if key is not None
    }
    reopened_filemode = getattr(
        getattr(reopened, "file", None), "_filemode", None
    )
    adata._init_as_actual(
        None,
        obs=reopened.obs,
        var=reopened.var,
        uns=reopened.uns,
        obsm=reopened.obsm,
        varm=reopened.varm,
        obsp=reopened.obsp,
        varp=reopened.varp,
        layers=real_layers,
        raw=raw_arg,
        filename=reopened.filename,
        filemode=reopened_filemode,
    )
    try:
        adata.file.close()
    except Exception:
        pass
    adata.file = reopened.file
