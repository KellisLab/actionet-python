"""Data import and backed HDF5 decompression utilities for AnnData."""

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import csr_matrix

from ..io.persist import (
    _flush_pending,
)
from ..io.chunking import (
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    validate_chunk_size,
)
from ..io.backed_adapter import BackedAnnDataAdapter
from ..io.checkpoint import rewrite_h5ad_payload
from ..io.rewrite import RewriteTransaction

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


def _matrix_path_and_filter_state(
    adapter: BackedAnnDataAdapter,
    layer: str | None,
) -> tuple[str, bool]:
    """Resolve a matrix and report whether any of its datasets are filtered."""
    import h5py

    path = "/X" if layer is None else f"/layers/{layer}"
    handle = adapter.file_handle
    if path not in handle:
        if layer is None:
            raise KeyError("X not found in backed file")
        raise KeyError(f"Layer '{layer}' not found in backed file")
    node = handle[path]
    datasets = (
        [node]
        if isinstance(node, h5py.Dataset)
        else [node[name] for name in ("data", "indices", "indptr")]
    )
    filtered = False
    for dataset in datasets:
        creation = dataset.id.get_create_plist()
        filtered = filtered or creation.get_nfilters() > 0
    return path, filtered


def decompress_backed_storage(
    adata: AnnData,
    *,
    layer: str | None = None,
    scope: str = "matrix",
    output_file: str | None = None,
    backed_write_chunk_size: int = DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    chunk_size: int | None = None,
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

    _flush_pending(adata)
    adapter = BackedAnnDataAdapter(adata)
    src_path = adapter.filename
    inplace = output_file is None or os.path.abspath(output_file) == os.path.abspath(src_path)
    dest_path = src_path if inplace else os.path.realpath(os.fspath(output_file))
    effective_chunk_size = (
        backed_write_chunk_size if chunk_size is None else chunk_size
    )
    chunk_size = validate_chunk_size(
        effective_chunk_size,
        name="backed_write_chunk_size",
    )

    if inplace and not adapter.writable:
        raise ValueError(
            "In-place decompression requires backed mode 'r+'. "
            "Re-open with `ad.read_h5ad(path, backed=\"r+\")` or pass `output_file`."
        )

    original_mode = adapter.mode
    if scope == "matrix":
        matrix_path, changed = _matrix_path_and_filter_state(adapter, layer)
        uncompressed_paths = {matrix_path}
        native_paths = {matrix_path}
    else:
        matrix_path = "/"
        changed = True
        uncompressed_paths = {"/"}
        native_paths = None

    if inplace and scope == "matrix" and not changed:
        if verbose:
            print(
                f"[decompress_backed_storage] already uncompressed: "
                f"{matrix_path.lstrip('/')}"
            )
        return None

    with RewriteTransaction(src_path, dest_path) as transaction:
        rewrite_h5ad_payload(
            src_path,
            transaction.temp_path,
            chunk_size=chunk_size,
            uncompressed_paths=uncompressed_paths,
            native_matrix_paths=native_paths,
        )
        validated = ad.read_h5ad(transaction.temp_path, backed="r")
        file_handle = getattr(validated, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass
        transaction.commit(
            close_source=adapter.close if inplace else None,
            restore_source=(
                lambda: adapter.reopen(mode=original_mode)
                if inplace
                else None
            ),
        )

    if inplace:
        adapter.reopen(mode=original_mode)
        if verbose:
            label = "full file" if scope == "file" else matrix_path.lstrip("/")
            print(f"[decompress_backed_storage] decompressed: {label}")
        return None

    if verbose:
        label = "full file" if scope == "file" else matrix_path.lstrip("/")
        print(f"[decompress_backed_storage] decompressed: {label} -> {dest_path}")
    return ad.read_h5ad(dest_path, backed="r+")
