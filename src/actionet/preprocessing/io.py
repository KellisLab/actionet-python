"""Data import and backed HDF5 decompression utilities for AnnData."""

import os
import pathlib
import shutil
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import csr_matrix

from ..io.persist import (
    is_writable_backed,
    _refresh_backed_handle,
)
from ..io.chunking import (
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    validate_chunk_size,
)
from ..io.checkpoint import copy_h5_group

from .filter import filter_anndata


def import_anndata_generic(
    input_path: str,
    mtx_file: str,
    gene_annotations: str,
    sample_annotations: str,
    gene_headers: list[str] | None = None,
    sample_headers: list[str] | None = None,
    sep: str = "\t",
    prefilter: bool = False,
    prefil_params: dict | None = None,
) -> AnnData:
    """Python implementation of import.se.generic from R, using AnnData."""
    gene_path = os.path.join(input_path, gene_annotations)
    gene_table = pd.read_csv(gene_path, sep=sep, header=None, dtype=str)
    if gene_headers is not None:
        gene_table.columns = gene_headers
    gene_table.index = gene_table.iloc[:, 0].astype(str)
    gene_table.index.name = None

    sample_path = os.path.join(input_path, sample_annotations)
    sample_annots = pd.read_csv(sample_path, sep=sep, header=None, dtype=str)
    if sample_headers is not None:
        sample_annots.columns = sample_headers
    sample_annots.index = sample_annots.iloc[:, 0].astype(str)
    sample_annots.index.name = None

    mtx_path = os.path.join(input_path, mtx_file)
    matrix = mmread(mtx_path, spmatrix=True)
    if sp.issparse(matrix):
        X = matrix.tocsc(copy=False).transpose(copy=False)
    else:
        X = csr_matrix(np.asarray(matrix).T)
    adata = AnnData(X=X, obs=sample_annots, var=gene_table)
    adata.obs_names_make_unique(join="_")
    adata.var_names_make_unique(join="_")

    if prefilter and prefil_params is not None:
        adata = filter_anndata(
            adata,
            layer_name=None,
            min_cells_per_feat=prefil_params.get("min_cells_per_feat", None),
            min_feats_per_cell=prefil_params.get("min_feats_per_cell", None),
            min_umis_per_cell=prefil_params.get("min_umis_per_cell", None),
            max_umis_per_cell=prefil_params.get("max_umis_per_cell", None),
            filter_adata=True,
            inplace=False,
        )

    return adata


# ---------------------------------------------------------------------------
# Backed decompression helpers + public utility
# ---------------------------------------------------------------------------


def _copy_h5_attrs(src, dst) -> None:
    """Copy all attributes from one h5py object to another."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_dataset_chunked(src_ds, dst_ds, chunk_size: int) -> None:
    """Copy dataset contents in chunks along axis-0."""
    if src_ds.shape == ():
        dst_ds[()] = src_ds[()]
        return

    if src_ds.ndim == 0:
        dst_ds[()] = src_ds[()]
        return

    n_rows = src_ds.shape[0]
    if n_rows == 0:
        return

    step = int(max(1, chunk_size))
    for start in range(0, n_rows, step):
        end = min(start + step, n_rows)
        dst_ds[start:end, ...] = src_ds[start:end, ...]


def _dataset_create_kwargs_uncompressed(src_ds) -> dict:
    """Build create_dataset kwargs that preserve shape/chunking but drop compression."""
    kwargs = {
        "shape": src_ds.shape,
        "dtype": src_ds.dtype,
    }
    if src_ds.chunks is not None:
        kwargs["chunks"] = src_ds.chunks
    if src_ds.maxshape is not None:
        kwargs["maxshape"] = src_ds.maxshape
    return kwargs


def _replace_dataset_with_uncompressed(parent, name: str, chunk_size: int) -> bool:
    """Replace one dataset with an uncompressed copy in-place."""
    src_ds = parent[name]
    if getattr(src_ds, "compression", None) is None:
        return False

    tmp_name = f"__tmp_uncompressed_{name}"
    if tmp_name in parent:
        del parent[tmp_name]

    dst_ds = parent.create_dataset(tmp_name, **_dataset_create_kwargs_uncompressed(src_ds))
    _copy_dataset_chunked(src_ds, dst_ds, chunk_size=chunk_size)
    _copy_h5_attrs(src_ds, dst_ds)

    del parent[name]
    parent.move(tmp_name, name)
    return True


def _decompress_sparse_group_inplace(group, chunk_size: int) -> bool:
    """Decompress sparse `data/indices/indptr` datasets in-place."""
    changed = False
    for dataset_name in ("data", "indices", "indptr"):
        if dataset_name in group:
            changed = _replace_dataset_with_uncompressed(
                group,
                dataset_name,
                chunk_size=chunk_size,
            ) or changed
    return changed


def _resolve_backed_matrix_node(adata: AnnData, layer: str | None):
    """Resolve the HDF5 node backing `.X` or one layer."""
    h5file = adata.file._file
    if layer is None:
        return h5file["X"], "X"

    if "layers" not in h5file or layer not in h5file["layers"]:
        raise KeyError(f"Layer '{layer}' not found in backed file")
    return h5file["layers"][layer], f"layers/{layer}"


def _decompress_matrix_in_adata(
    adata: AnnData,
    *,
    layer: str | None,
    chunk_size: int,
) -> tuple[bool, str]:
    """Decompress one backed matrix target (`.X` or one layer)."""
    node, matrix_key = _resolve_backed_matrix_node(adata, layer)
    if hasattr(node, "keys") and {"data", "indices", "indptr"}.issubset(set(node.keys())):
        changed = _decompress_sparse_group_inplace(node, chunk_size=chunk_size)
        return changed, matrix_key

    # Dense backed matrix (h5py Dataset).
    parent = node.parent
    ds_name = node.name.rsplit("/", 1)[-1]
    changed = _replace_dataset_with_uncompressed(parent, ds_name, chunk_size=chunk_size)
    return changed, matrix_key


def _copy_h5_group_uncompressed(src_group, dst_group, chunk_size: int) -> None:
    """Recursively copy an HDF5 group without compression.

    Thin wrapper around :func:`copy_h5_group` that drops codec settings while
    preserving shape/chunks/maxshape.
    """
    copy_h5_group(
        src_group,
        dst_group,
        chunk_size=chunk_size,
        preserve_compression=False,
    )


def _rewrite_h5ad_uncompressed(src_path: str, dest_path: str, chunk_size: int) -> None:
    """Rewrite a full .h5ad file with all datasets uncompressed."""
    import h5py

    with h5py.File(src_path, "r") as src_f, h5py.File(dest_path, "w") as dst_f:
        _copy_h5_group_uncompressed(src_f, dst_f, chunk_size=chunk_size)


def decompress_backed_storage(
    adata: AnnData,
    *,
    layer: str | None = None,
    scope: str = "matrix",
    output_file: str | None = None,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    verbose: bool = True,
) -> AnnData | None:
    """Decompress backed AnnData storage in-place or into a copy.

    Parameters
    ----------
    adata : AnnData
        Backed AnnData object.
    layer : str or None, optional (default: None)
        Layer to target when ``scope='matrix'``. ``None`` targets ``.X``.
    scope : {'matrix', 'file'}, optional (default: 'matrix')
        - ``'matrix'``: decompress only ``.X`` or one layer.
        - ``'file'``: rewrite the entire ``.h5ad`` uncompressed.
    output_file : str or None, optional
        If provided, write decompressed output to this path and return a new
        backed AnnData opened in ``r+`` mode. If ``None``, mutate in-place.
    backed_write_chunk_size : int, optional (default: 16384)
        Row/element chunk size used while copying dataset payloads.
        Atlas-scale rewrites may benefit from starting with ``32768``;
        larger values use proportionally more temporary memory.
    verbose : bool, optional (default: True)
        Print a brief status line when work is done.

    Returns
    -------
    AnnData or None
        ``None`` for in-place updates; backed AnnData for copy mode.
    """
    if scope not in {"matrix", "file"}:
        raise ValueError("scope must be either 'matrix' or 'file'")

    if not bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None)):
        raise ValueError("decompress_backed_storage requires a backed AnnData object")

    src_path = str(adata.filename)
    inplace = output_file is None or os.path.abspath(output_file) == os.path.abspath(src_path)
    dest_path = src_path if inplace else str(output_file)
    chunk_size = validate_chunk_size(
        backed_write_chunk_size,
        name="backed_write_chunk_size",
    )

    if inplace and not is_writable_backed(adata):
        raise ValueError(
            "In-place decompression requires backed mode 'r+'. "
            "Re-open with `ad.read_h5ad(path, backed=\"r+\")` or pass `output_file`."
        )

    if scope == "matrix":
        if not inplace:
            shutil.copy2(src_path, dest_path)
            target = ad.read_h5ad(dest_path, backed="r+")
            changed, matrix_key = _decompress_matrix_in_adata(
                target,
                layer=layer,
                chunk_size=chunk_size,
            )
            if verbose:
                status = "decompressed" if changed else "already uncompressed"
                print(f"[decompress_backed_storage] {status}: {matrix_key} -> {dest_path}")
            return target

        changed, matrix_key = _decompress_matrix_in_adata(
            adata,
            layer=layer,
            chunk_size=chunk_size,
        )
        # Dense-backed wrappers and sparse dataset handles are safest to refresh.
        _refresh_backed_handle(adata, src_path, mode="r+")
        if verbose:
            status = "decompressed" if changed else "already uncompressed"
            print(f"[decompress_backed_storage] {status}: {matrix_key}")
        return None

    # scope == "file"
    if inplace:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(pathlib.Path(src_path).parent),
            suffix=".h5ad",
        )
        os.close(tmp_fd)
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            _rewrite_h5ad_uncompressed(src_path, tmp_path, chunk_size=chunk_size)
            shutil.move(tmp_path, src_path)
            _refresh_backed_handle(adata, src_path, mode="r+")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        if verbose:
            print(f"[decompress_backed_storage] decompressed full file in place: {src_path}")
        return None

    _rewrite_h5ad_uncompressed(src_path, dest_path, chunk_size=chunk_size)
    if verbose:
        print(f"[decompress_backed_storage] decompressed full file copy: {dest_path}")
    return ad.read_h5ad(dest_path, backed="r+")
