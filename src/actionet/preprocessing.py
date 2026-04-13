"""Data preprocessing functions."""

import os
import pathlib
import shutil
import tempfile
import warnings
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse

from ._backed_compression import get_matrix_compression_policy
from ._backed_persist import (
    is_backed_adata,
    _refresh_backed_handle,
    _adaptive_sparse_chunk_size,
    _write_filtered_backed,
    _normalize_index_array,
    _warn_if_duplicates,
    _view_idx_to_int,
    materialize_backed,
    subset_backed_inplace,
)
from ._matrix_source import MatrixSource


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

    sample_path = os.path.join(input_path, sample_annotations)
    sample_annots = pd.read_csv(sample_path, sep=sep, header=None, dtype=str)
    if sample_headers is not None:
        sample_annots.columns = sample_headers

    obs_names = sample_annots.iloc[:, 0].tolist()
    var_names = gene_table.iloc[:, 0].tolist()

    mtx_path = os.path.join(input_path, mtx_file)
    X = mmread(mtx_path).transpose()
    adata = AnnData(X=csr_matrix(X), obs=sample_annots, var=gene_table)
    adata.obs_names = obs_names
    adata.var_names = var_names
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


def normalize_anndata(
    adata: AnnData,
    target_sum: float = 1e4,
    log_transform: bool = True,
    log_base: Optional[float] = None,
    pseudocount: float = 1.0,
    layer: str | None = None,
    backed_chunk_size: int = 4096,
    dtype_out: str = "float32",
    inplace: bool = True,
    layer_added: str | None = None,
) -> Optional[AnnData]:
    """Total-count normalization with optional log transform.

    Mimics the R ``normalize.ace`` function: each cell's counts are scaled
    to *target_sum*, then (by default) a ``log(x + pseudocount)`` transform
    is applied.  The default normalization path works for both in-memory and
    HDF5-backed AnnData objects.  When ``layer_added`` is provided in backed
    mode, the normalized values are streamed directly into a new on-disk layer
    without copying the full source matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    target_sum : float, optional (default: 1e4)
        Target total count per cell after scaling.
    log_transform : bool, optional (default: True)
        If ``True``, apply ``log(x + pseudocount)`` (optionally with a custom
        base) after scaling.  Set to ``False`` for scaling only.
    log_base : float or None, optional (default: None)
        Logarithm base for the log step.  ``None`` uses the natural logarithm.
        Ignored when *log_transform* is ``False``.
    pseudocount : float, optional (default: 1.0)
        Value added to each scaled count before taking the log.  Must be
        positive when *log_transform* is ``True``.  The default of ``1.0``
        is identical to the classic ``log1p`` transform.  Ignored when
        *log_transform* is ``False``.
    layer : str or None, optional (default: None)
        Layer to normalize.  ``None`` uses ``adata.X``.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when operating on backed AnnData.
        Ignored for in-memory objects.
    dtype_out : str, optional (default: "float32")
        Output dtype for backed normalization blocks.
    inplace : bool, optional (default: True)
        If ``True``, normalize in place and return ``None``; otherwise
        return a modified copy.
    layer_added : str or None, optional (default: None)
        If provided, write normalized values to ``adata.layers[layer_added]``
        instead of overwriting the input matrix. In backed mode this requires
        a writable handle (``backed='r+'``), and any existing
        ``layers[layer_added]`` is overwritten.

    Returns
    -------
    AnnData or None
        Modified copy if ``inplace=False``, otherwise ``None``.

    Raises
    ------
    ValueError
        If ``target_sum`` or ``log_base`` is not positive, if ``pseudocount``
        is not positive when *log_transform* is ``True``, if ``layer_added``
        equals ``layer``, or if ``layer_added`` is requested on a read-only
        backed AnnData object.
    """
    if target_sum <= 0:
        raise ValueError("target_sum must be positive.")
    if log_base is not None and log_base <= 0:
        raise ValueError("log_base must be positive.")
    if log_transform and pseudocount <= 0:
        raise ValueError("`pseudocount` must be > 0 when `log_transform=True`.")
    if layer_added is not None and layer is not None and layer_added == layer:
        raise ValueError("`layer_added` must differ from `layer`.")

    out_dtype = np.dtype(dtype_out)

    if not inplace:
        if getattr(adata, "isbacked", False):
            adata = adata.to_memory()
        else:
            adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    if layer_added is not None:
        if source.is_backed:
            if not _is_writable_backed(adata):
                raise ValueError(
                    "`layer_added` with backed AnnData requires writable mode 'r+'. "
                    "Re-open with `ad.read_h5ad(path, backed=\"r+\")`."
                )
            if source.is_sparse and source.backed_sparse_format() == "csc":
                _normalize_backed_csc_via_csr_rewrite(
                    source,
                    target_sum=target_sum,
                    log_transform=log_transform,
                    log_base=log_base,
                    pseudocount=pseudocount,
                    chunk_size=backed_chunk_size,
                    dtype_out=out_dtype,
                    layer_added=layer_added,
                )
            else:
                _create_backed_normalized_layer(
                    adata,
                    source=source,
                    layer_added=layer_added,
                    dtype_out=out_dtype,
                )
                _normalize_backed_streamed(
                    source,
                    adata,
                    layer_added=layer_added,
                    target_sum=target_sum,
                    log_transform=log_transform,
                    log_base=log_base,
                    pseudocount=pseudocount,
                    chunk_size=backed_chunk_size,
                    dtype_out=out_dtype,
                )
                _refresh_backed_handle(adata, str(adata.filename), mode="r+")
        else:
            adata.layers[layer_added] = _normalize_matrix_in_memory(
                source.matrix,
                target_sum=target_sum,
                log_transform=log_transform,
                log_base=log_base,
                pseudocount=pseudocount,
            )
    elif not source.is_backed:
        normalized = _normalize_matrix_in_memory(
            source.matrix,
            target_sum=target_sum,
            log_transform=log_transform,
            log_base=log_base,
            pseudocount=pseudocount,
        )
        if layer is None:
            adata.X = normalized
        else:
            adata.layers[layer] = normalized
    else:
        _normalize_backed(
            source,
            target_sum,
            log_transform,
            log_base,
            pseudocount,
            backed_chunk_size,
            out_dtype,
        )

    if inplace:
        return None
    return adata


def _apply_log_transform(arr, pseudocount: float, log_scale: Optional[float]) -> None:
    """In-place log transform: ``log(arr + pseudocount) / log(base)``.

    When *pseudocount* is 1.0 (the default), the faster ``np.log1p`` kernel
    is used so existing behaviour is bit-for-bit identical.
    """
    if np.isclose(pseudocount, 1.0):
        np.log1p(arr, out=arr)
    else:
        arr += pseudocount
        np.log(arr, out=arr)
    if log_scale is not None and not np.isclose(log_scale, 1.0):
        arr *= log_scale


def _normalize_matrix_in_memory(
    matrix,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    pseudocount: float = 1.0,
):
    """Return a normalized in-memory copy of ``matrix``.

    The input is never modified. Sparse inputs are returned as CSR.
    """
    if issparse(matrix):
        X = matrix.tocsr(copy=True).astype(np.float64, copy=False)

        row_sums = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        scaling = np.divide(
            target_sum,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 0,
        )

        if X.nnz > 0:
            row_nnz = np.diff(X.indptr)
            X.data *= np.repeat(scaling, row_nnz)

            if log_transform:
                if X.data.min() < 0:
                    import warnings
                    warnings.warn(
                        f"Matrix contains negative values (min={X.data.min():.4g}). "
                        "log transform is only meaningful for non-negative data; "
                        "results may contain NaN.",
                        stacklevel=2,
                    )
                log_scale = 1.0 if log_base is None else 1.0 / np.log(log_base)
                _apply_log_transform(X.data, pseudocount, log_scale)
        return X

    arr = np.array(matrix, dtype=np.float64, copy=True)
    row_sums = arr.sum(axis=1)
    scaling = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )

    arr *= scaling[:, np.newaxis]

    if log_transform:
        log_scale = 1.0 if log_base is None else 1.0 / np.log(log_base)
        _apply_log_transform(arr, pseudocount, log_scale)
    return arr


def _normalize_sparse_block(
    block,
    scale: np.ndarray,
    *,
    log_transform: bool,
    log_scale: Optional[float],
    pseudocount: float = 1.0,
    dtype_out: np.dtype,
) -> sp.csr_matrix:
    """Normalize one sparse row block and return CSR output."""
    block = block.tocsr(copy=True)
    block = block.astype(dtype_out, copy=False)
    if block.nnz > 0:
        row_nnz = np.diff(block.indptr)
        block.data *= np.repeat(scale, row_nnz)
        if log_transform:
            _apply_log_transform(block.data, pseudocount, log_scale)
    return block


def _normalize_dense_block(
    block,
    scale: np.ndarray,
    *,
    log_transform: bool,
    log_scale: Optional[float],
    pseudocount: float = 1.0,
    dtype_out: np.dtype,
) -> np.ndarray:
    """Normalize one dense row block and return dense output."""
    arr = np.asarray(block, dtype=dtype_out)
    arr *= scale[:, np.newaxis]
    if log_transform:
        _apply_log_transform(arr, pseudocount, log_scale)
    return arr


# ---------------------------------------------------------------------------
# Backed (disk-backed) chunked path
# ---------------------------------------------------------------------------

def _normalize_backed(
    source: MatrixSource,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    pseudocount: float,
    chunk_size: int,
    dtype_out: np.dtype,
) -> None:
    """Normalize a backed AnnData matrix using chunked streaming I/O."""
    if source.is_sparse and source.backed_sparse_format() == "csc":
        _normalize_backed_csc_via_csr_rewrite(
            source,
            target_sum=target_sum,
            log_transform=log_transform,
            log_base=log_base,
            pseudocount=pseudocount,
            chunk_size=chunk_size,
            dtype_out=dtype_out,
            layer_added=None,
        )
        return

    row_sums = source.row_sums(chunk_size=chunk_size)
    scaling = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )
    log_scale = None if not log_transform else (1.0 if log_base is None else 1.0 / np.log(log_base))

    def _normalize_block(block, start: int, end: int):
        scale = scaling[start:end]
        if issparse(block):
            return _normalize_sparse_block(
                block,
                scale,
                log_transform=log_transform,
                log_scale=log_scale,
                pseudocount=pseudocount,
                dtype_out=dtype_out,
            )
        return _normalize_dense_block(
            block,
            scale,
            log_transform=log_transform,
            log_scale=log_scale,
            pseudocount=pseudocount,
            dtype_out=dtype_out,
        )

    source.apply_rowwise(_normalize_block, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Streamed backed layer_added path
# ---------------------------------------------------------------------------

def _create_backed_normalized_layer(
    adata: AnnData,
    *,
    source: MatrixSource,
    layer_added: str,
    dtype_out: np.dtype,
) -> None:
    """Create ``layers[layer_added]`` on disk ready for streamed writes.

    For sparse CSR input the destination group shares the ``indices``
    dataset with the source via HDF5 hard-link (zero-copy, ~21 GB saved
    on production data) and copies ``indptr`` independently (~14 MB).
    A fresh ``data`` dataset is created with *dtype_out*.

    For dense input a new dataset of the correct shape and dtype is
    created.
    """
    import h5py

    h5file = adata.file._file
    layers_group = h5file["layers"] if "layers" in h5file else h5file.create_group("layers")

    if layer_added in layers_group:
        del layers_group[layer_added]

    src_grp = source._resolve_h5_group() if source.is_sparse else None

    if source.is_sparse and src_grp is not None:
        dest_grp = layers_group.create_group(layer_added)

        total_nnz = int(src_grp["data"].shape[0])
        src_compression = src_grp["data"].compression
        src_compression_opts = src_grp["data"].compression_opts
        data_kwargs: dict = {}
        if src_compression is not None:
            data_kwargs["compression"] = src_compression
            if src_compression_opts is not None:
                data_kwargs["compression_opts"] = src_compression_opts

        dest_grp.create_dataset(
            "data", shape=(total_nnz,), dtype=dtype_out, **data_kwargs,
        )

        dest_grp["indices"] = src_grp["indices"]

        indptr_src = src_grp["indptr"]
        indptr_kwargs: dict = {}
        ipc = indptr_src.compression
        if ipc is not None:
            indptr_kwargs["compression"] = ipc
            ipo = indptr_src.compression_opts
            if ipo is not None:
                indptr_kwargs["compression_opts"] = ipo
        dest_grp.create_dataset(
            "indptr", data=indptr_src[...], **indptr_kwargs,
        )

        for attr_name in ("shape", "encoding-type", "encoding-version"):
            if attr_name in src_grp.attrs:
                dest_grp.attrs[attr_name] = src_grp.attrs[attr_name]

    else:
        n_obs, n_vars = source.n_obs, source.n_vars
        layers_group.create_dataset(
            layer_added,
            shape=(n_obs, n_vars),
            dtype=dtype_out,
        )
        ds = layers_group[layer_added]
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"

    h5file.flush()


def _normalize_backed_streamed(
    source: MatrixSource,
    adata: AnnData,
    *,
    layer_added: str,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    pseudocount: float = 1.0,
    chunk_size: int,
    dtype_out: np.dtype,
) -> None:
    """Streamed normalize: read from *source*, write to ``layers[layer_added]``.

    Two passes over the source matrix:
      1. Compute per-row sums.
      2. Read each chunk, normalize in-memory, write the ``data`` slice
         (sparse) or row slice (dense) directly to the destination.

    This avoids the full-file copy and dtype recast of the legacy path.
    """
    row_sums = source.row_sums(chunk_size=chunk_size)
    scaling = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )

    log_scale = None
    if log_transform:
        log_scale = 1.0 if log_base is None else 1.0 / np.log(log_base)

    h5file = adata.file._file
    dest_node = h5file["layers"][layer_added]

    is_sparse_dest = hasattr(dest_node, "keys") and "data" in dest_node

    if is_sparse_dest:
        dest_encoding = _backed_sparse_group_format(dest_node)
        if dest_encoding != "csr":
            raise ValueError(
                "Streamed sparse normalization destinations must use CSR-backed storage."
            )

        dest_data_ds = dest_node["data"]
        dest_indptr_ds = dest_node["indptr"]

        for chunk in source.iter_row_chunks(chunk_size=chunk_size):
            block = chunk.block
            scale = scaling[chunk.start:chunk.end]

            if issparse(block):
                block = _normalize_sparse_block(
                    block,
                    scale,
                    log_transform=log_transform,
                    log_scale=log_scale,
                    pseudocount=pseudocount,
                    dtype_out=dtype_out,
                )
            else:
                block = sp.csr_matrix(
                    _normalize_dense_block(
                        block,
                        scale,
                        log_transform=log_transform,
                        log_scale=log_scale,
                        pseudocount=pseudocount,
                        dtype_out=dtype_out,
                    )
                )

            ip_start = int(dest_indptr_ds[chunk.start])
            ip_end = int(dest_indptr_ds[chunk.end])
            if ip_end > ip_start:
                dest_data_ds[ip_start:ip_end] = block.data.astype(
                    dtype_out, copy=False,
                )
    else:
        for chunk in source.iter_row_chunks(chunk_size=chunk_size):
            block = chunk.block
            scale = scaling[chunk.start:chunk.end]

            if issparse(block):
                block = block.toarray()
            arr = _normalize_dense_block(
                block,
                scale,
                log_transform=log_transform,
                log_scale=log_scale,
                pseudocount=pseudocount,
                dtype_out=dtype_out,
            )

            dest_node[chunk.start:chunk.end, :] = arr

    h5file.flush()


def _backed_sparse_group_format(group) -> str | None:
    """Return ``'csr'`` or ``'csc'`` for one sparse HDF5 group when known."""
    enc = group.attrs.get("encoding-type", "")
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


def _dataset_create_kwargs_like(
    src_ds,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> dict:
    """Build create_dataset kwargs that preserve codec settings best-effort."""
    kwargs = {
        "shape": shape,
        "dtype": dtype,
    }
    if getattr(src_ds, "compression", None) is not None:
        kwargs["compression"] = src_ds.compression
        if src_ds.compression_opts is not None:
            kwargs["compression_opts"] = src_ds.compression_opts
    if getattr(src_ds, "chunks", None) is not None and all(dim > 0 for dim in shape):
        chunks = tuple(min(int(src_chunk), int(dim)) for src_chunk, dim in zip(src_ds.chunks, shape))
        if all(chunk > 0 for chunk in chunks):
            kwargs["chunks"] = chunks
    return kwargs


def _compute_backed_sparse_row_stats(
    source: MatrixSource,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row sums and row nnz counts for one backed sparse source."""
    row_sums = np.zeros(source.n_obs, dtype=np.float64)
    row_nnz = np.zeros(source.n_obs, dtype=np.int64)

    for chunk in source.iter_row_chunks(chunk_size=chunk_size):
        block = chunk.block
        if issparse(block):
            block_csr = block.tocsr(copy=False)
            row_sums[chunk.start:chunk.end] = np.asarray(block_csr.sum(axis=1)).ravel()
            row_nnz[chunk.start:chunk.end] = np.diff(block_csr.indptr).astype(np.int64, copy=False)
        else:
            arr = np.asarray(block, dtype=np.float64)
            row_sums[chunk.start:chunk.end] = arr.sum(axis=1)
            row_nnz[chunk.start:chunk.end] = np.count_nonzero(arr, axis=1)

    return row_sums, row_nnz


def _create_backed_csr_group_from_row_nnz(
    parent,
    *,
    name: str,
    source: MatrixSource,
    source_grp,
    row_nnz: np.ndarray,
    dtype_out: np.dtype,
):
    """Create one CSR sparse HDF5 group sized from row nnz counts."""
    total_nnz = int(row_nnz.sum())
    indptr_dtype = np.dtype(source_grp["indptr"].dtype)
    indices_dtype = np.dtype(source_grp["indices"].dtype)

    if np.issubdtype(indptr_dtype, np.integer):
        max_value = np.iinfo(indptr_dtype).max
        if total_nnz > max_value:
            raise OverflowError(
                f"Normalized sparse output needs {total_nnz} entries, which exceeds {indptr_dtype}."
            )

    if name in parent:
        del parent[name]
    dest_grp = parent.create_group(name)

    indptr = np.empty(source.n_obs + 1, dtype=indptr_dtype)
    indptr[0] = 0
    np.cumsum(row_nnz, dtype=indptr_dtype, out=indptr[1:])

    data_kwargs = _dataset_create_kwargs_like(
        source_grp["data"],
        shape=(total_nnz,),
        dtype=dtype_out,
    )
    indices_kwargs = _dataset_create_kwargs_like(
        source_grp["indices"],
        shape=(total_nnz,),
        dtype=indices_dtype,
    )
    indptr_kwargs = _dataset_create_kwargs_like(
        source_grp["indptr"],
        shape=(source.n_obs + 1,),
        dtype=indptr_dtype,
    )

    dest_grp.create_dataset("data", **data_kwargs)
    dest_grp.create_dataset("indices", **indices_kwargs)
    dest_grp.create_dataset("indptr", data=indptr, **indptr_kwargs)

    _copy_h5_attrs(source_grp, dest_grp)
    dest_grp.attrs["encoding-type"] = "csr_matrix"
    dest_grp.attrs["shape"] = np.asarray([source.n_obs, source.n_vars], dtype=np.int64)
    return dest_grp


def _write_normalized_chunks_to_csr_group(
    source: MatrixSource,
    *,
    dest_grp,
    scaling: np.ndarray,
    log_transform: bool,
    log_scale: Optional[float],
    pseudocount: float = 1.0,
    chunk_size: int,
    dtype_out: np.dtype,
) -> None:
    """Stream normalized row chunks into one CSR sparse destination group."""
    dest_indptr_ds = dest_grp["indptr"]
    dest_indices_ds = dest_grp["indices"]
    dest_data_ds = dest_grp["data"]

    for chunk in source.iter_row_chunks(chunk_size=chunk_size):
        scale = scaling[chunk.start:chunk.end]
        block_csr = _normalize_sparse_block(
            chunk.block,
            scale,
            log_transform=log_transform,
            log_scale=log_scale,
            pseudocount=pseudocount,
            dtype_out=dtype_out,
        )

        ip_start = int(dest_indptr_ds[chunk.start])
        ip_end = int(dest_indptr_ds[chunk.end])
        expected_nnz = ip_end - ip_start
        if block_csr.nnz != expected_nnz:
            raise ValueError(
                f"CSR destination nnz mismatch for rows [{chunk.start}, {chunk.end}): "
                f"expected {expected_nnz}, observed {block_csr.nnz}."
            )

        if expected_nnz == 0:
            continue

        dest_indices_ds[ip_start:ip_end] = block_csr.indices.astype(dest_indices_ds.dtype, copy=False)
        dest_data_ds[ip_start:ip_end] = block_csr.data.astype(dtype_out, copy=False)


def _normalize_backed_csc_via_csr_rewrite(
    source: MatrixSource,
    *,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    pseudocount: float = 1.0,
    chunk_size: int,
    dtype_out: np.dtype,
    layer_added: str | None,
) -> None:
    """Normalize one backed CSC source by rewriting the destination as CSR."""
    adata = source.adata
    h5file = adata.file._file
    source_grp = source._resolve_h5_group()

    if layer_added is None:
        parent = h5file if source.layer is None else h5file["layers"]
        target_name = "X" if source.layer is None else source.layer
    else:
        parent = h5file["layers"] if "layers" in h5file else h5file.create_group("layers")
        target_name = layer_added

    temp_name = f"__actionet_normalize_tmp_{target_name}"
    row_sums, row_nnz = _compute_backed_sparse_row_stats(source, chunk_size=chunk_size)
    scaling = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )
    log_scale = None if not log_transform else (1.0 if log_base is None else 1.0 / np.log(log_base))

    try:
        dest_grp = _create_backed_csr_group_from_row_nnz(
            parent,
            name=temp_name,
            source=source,
            source_grp=source_grp,
            row_nnz=row_nnz,
            dtype_out=dtype_out,
        )
        _write_normalized_chunks_to_csr_group(
            source,
            dest_grp=dest_grp,
            scaling=scaling,
            log_transform=log_transform,
            log_scale=log_scale,
            pseudocount=pseudocount,
            chunk_size=chunk_size,
            dtype_out=dtype_out,
        )
        h5file.flush()

        if target_name in parent:
            del parent[target_name]
        parent.move(temp_name, target_name)
        h5file.flush()
    except Exception:
        if temp_name in parent:
            del parent[temp_name]
            h5file.flush()
        raise

    _refresh_backed_handle(adata, str(adata.filename), mode="r+")


def _compute_filter_stats(
    source: MatrixSource,
    obs_idx: np.ndarray,
    var_idx: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass computation of row sums, row nnz, and column nnz.

    Reads each chunk once and accumulates all three statistics
    simultaneously, cutting I/O by 3x compared to three separate passes.

    Returns ``(row_sums, row_nnz, col_nnz)`` where ``row_sums`` and
    ``row_nnz`` have length ``obs_idx.size`` and ``col_nnz`` has length
    ``var_idx.size``.
    """
    row_sums = np.zeros(obs_idx.size, dtype=np.float64)
    row_nnz = np.zeros(obs_idx.size, dtype=np.int64)
    col_nnz = np.zeros(var_idx.size, dtype=np.int64)

    obs_is_full = (
        obs_idx.size == source.n_obs
        and np.array_equal(obs_idx, np.arange(source.n_obs, dtype=np.int64))
    )
    var_is_full = (
        var_idx.size == source.n_vars
        and np.array_equal(var_idx, np.arange(source.n_vars, dtype=np.int64))
    )
    col_indices = None if var_is_full else var_idx

    pos = 0
    if obs_is_full:
        for chunk in source.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices):
            block = chunk.block
            sz = chunk.end - chunk.start
            if sp.issparse(block):
                row_sums[pos:pos + sz] = np.asarray(block.sum(axis=1)).ravel()
                row_nnz[pos:pos + sz] = np.asarray(block.getnnz(axis=1)).ravel()
                col_nnz += np.asarray(block.getnnz(axis=0)).ravel().astype(np.int64, copy=False)
            else:
                arr = np.asarray(block, dtype=np.float64)
                row_sums[pos:pos + sz] = arr.sum(axis=1)
                row_nnz[pos:pos + sz] = np.count_nonzero(arr, axis=1)
                col_nnz += np.count_nonzero(arr, axis=0)
            pos += sz
    else:
        for _rows, block in source.iter_selected_row_chunks(
            obs_idx, chunk_size=chunk_size, col_indices=col_indices,
        ):
            sz = _rows.size
            if sp.issparse(block):
                row_sums[pos:pos + sz] = np.asarray(block.sum(axis=1)).ravel()
                row_nnz[pos:pos + sz] = np.asarray(block.getnnz(axis=1)).ravel()
                col_nnz += np.asarray(block.getnnz(axis=0)).ravel().astype(np.int64, copy=False)
            else:
                arr = np.asarray(block, dtype=np.float64)
                row_sums[pos:pos + sz] = arr.sum(axis=1)
                row_nnz[pos:pos + sz] = np.count_nonzero(arr, axis=1)
                col_nnz += np.count_nonzero(arr, axis=0)
            pos += sz

    return row_sums, row_nnz, col_nnz


def compute_filter_masks(
    adata: AnnData,
    layer_name: str | None = None,
    *,
    min_cells_per_feat: int | float | None = None,
    min_feats_per_cell: int | None = None,
    min_umis_per_cell: int | None = None,
    max_umis_per_cell: int | None = None,
    backed_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute iterative filtering masks without modifying *adata*.

    Alternately evaluates cell and feature QC thresholds until the set of
    passing cells/features stabilises (typically 1--3 iterations).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_name : str or None, optional (default: None)
        Layer to compute statistics from.  ``None`` uses ``adata.X``.
    min_cells_per_feat : int or float or None, optional
        Minimum cells expressing a feature.  A ``float`` in ``(0, 1)``
        is interpreted as a fraction of the current number of passing cells.
    min_feats_per_cell : int or None, optional
        Minimum features detected per cell.
    min_umis_per_cell : int or None, optional
        Minimum total UMI count per cell.
    max_umis_per_cell : int or None, optional
        Maximum total UMI count per cell.
    backed_chunk_size : int, optional (default: 4096)
        Rows per streaming chunk (backed mode only).

    Returns
    -------
    obs_mask : ndarray of bool, shape ``(n_obs,)``
    var_mask : ndarray of bool, shape ``(n_vars,)``
    """
    source = MatrixSource(adata, layer=layer_name)

    obs_idx = np.arange(source.n_obs, dtype=np.int64)
    var_idx = np.arange(source.n_vars, dtype=np.int64)
    prev_shape = None

    while True:
        row_mask = np.ones(obs_idx.size, dtype=bool)
        col_mask = np.ones(var_idx.size, dtype=bool)

        chunk_size = int(max(1, backed_chunk_size))
        if source.is_backed and source.is_sparse:
            chunk_size = _adaptive_sparse_chunk_size(
                source.matrix,
                obs_idx,
                var_idx,
                chunk_size,
                target_block_mb=128,
                overhead_factor=10.0,
            )

        row_sums, row_nnz, col_nnz = _compute_filter_stats(
            source, obs_idx, var_idx, chunk_size,
        )

        if min_umis_per_cell is not None:
            row_mask &= row_sums >= min_umis_per_cell
        if max_umis_per_cell is not None:
            row_mask &= row_sums <= max_umis_per_cell
        if min_feats_per_cell is not None:
            row_mask &= row_nnz >= min_feats_per_cell
        if min_cells_per_feat is not None:
            if isinstance(min_cells_per_feat, float) and 0 < min_cells_per_feat < 1:
                min_fc = int(np.ceil(min_cells_per_feat * obs_idx.size))
            else:
                min_fc = int(min_cells_per_feat)
            col_mask &= col_nnz >= min_fc

        new_shape = (int(row_mask.sum()), int(col_mask.sum()))
        if prev_shape == new_shape:
            break
        prev_shape = new_shape

        obs_idx = obs_idx[row_mask]
        var_idx = var_idx[col_mask]

    obs_mask = np.zeros(adata.n_obs, dtype=bool)
    var_mask = np.zeros(adata.n_vars, dtype=bool)
    obs_mask[obs_idx] = True
    var_mask[var_idx] = True
    return obs_mask, var_mask


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


def _copy_backed_matrix_to_layer(
    adata: AnnData,
    *,
    source_layer: str | None,
    layer_added: str,
) -> None:
    """Copy backed ``.X`` or one backed layer to ``layers[layer_added]``.

    Existing destination layers are overwritten.
    """
    if not bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None)):
        raise ValueError("_copy_backed_matrix_to_layer requires a backed AnnData object")
    if not _is_writable_backed(adata):
        raise ValueError(
            "Backed layer copy requires backed mode 'r+'. "
            "Re-open with `ad.read_h5ad(path, backed=\"r+\")`."
        )

    h5file = adata.file._file
    layers_group = h5file["layers"] if "layers" in h5file else h5file.create_group("layers")

    if layer_added in layers_group:
        del layers_group[layer_added]

    source_key = "X" if source_layer is None else f"layers/{source_layer}"
    h5file.copy(source_key, layers_group, name=layer_added)
    h5file.flush()


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
    """Recursively copy an HDF5 group without compression."""
    import h5py

    _copy_h5_attrs(src_group, dst_group)
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            _copy_h5_group_uncompressed(obj, child, chunk_size=chunk_size)
        elif isinstance(obj, h5py.Dataset):
            dst_ds = dst_group.create_dataset(
                name,
                **_dataset_create_kwargs_uncompressed(obj),
            )
            _copy_dataset_chunked(obj, dst_ds, chunk_size=chunk_size)
            _copy_h5_attrs(obj, dst_ds)
        else:
            raise TypeError(f"Unsupported HDF5 object type for key '{name}': {type(obj)}")


def _rewrite_h5ad_uncompressed(src_path: str, dest_path: str, chunk_size: int) -> None:
    """Rewrite a full .h5ad file with all datasets uncompressed."""
    import h5py

    with h5py.File(src_path, "r") as src_f, h5py.File(dest_path, "w") as dst_f:
        _copy_h5_group_uncompressed(src_f, dst_f, chunk_size=chunk_size)


def _is_writable_backed(adata: AnnData) -> bool:
    """Return True when backed file is opened in writable mode."""
    if not bool(getattr(adata, "isbacked", False) and getattr(adata, "filename", None)):
        return False
    mode = getattr(getattr(adata, "file", None), "_file", None)
    if mode is None:
        return False
    return "+" in getattr(mode, "mode", "")


def decompress_backed_storage(
    adata: AnnData,
    *,
    layer: str | None = None,
    scope: str = "matrix",
    output_file: str | None = None,
    chunk_size: int = 4096,
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
    chunk_size : int, optional (default: 4096)
        Chunk size used while copying dataset payloads.
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
    chunk_size = int(max(1, chunk_size))

    if inplace and not _is_writable_backed(adata):
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


# ---------------------------------------------------------------------------
# Generic backed-safe subsetting
# ---------------------------------------------------------------------------


def _coerce_to_int_idx(
    mask_or_idx: np.ndarray,
    axis_size: int,
    *,
    name: str,
) -> np.ndarray:
    """Normalise a boolean mask **or** integer index array to int64 indices.

    Parameters
    ----------
    mask_or_idx : ndarray
        Boolean mask of length *axis_size*, or integer index array.
    axis_size : int
        Length of the axis being subsetted (for validation).
    name : str
        Human-readable axis name for error messages.
    """
    arr = np.asarray(mask_or_idx).ravel()
    if arr.dtype != bool and arr.size > 0 and arr.astype(np.int64, copy=False).min() < 0:
        raise ValueError(f"{name} indices contain negative values")
    idx = _normalize_index_array(
        mask_or_idx,
        axis_size,
        name=name,
        allow_negative=False,
    )
    if arr.dtype != bool:
        _warn_if_duplicates(idx, name=name)
    return idx


def subset_anndata(
    adata: AnnData,
    obs_idx: np.ndarray | None = None,
    var_idx: np.ndarray | None = None,
    *,
    inplace: bool = True,
    output_file: str | None = None,
    backed_chunk_size: int = 4096,
) -> AnnData | None:
    """Subset an AnnData safely on both axes (backed and in-memory).

    This is the recommended way to shrink a backed AnnData object.
    For backed objects the HDF5 file is atomically rewritten so that the
    on-disk dimensions match the Python object.  For in-memory objects
    the subset is a standard copy.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_idx : ndarray or None, optional
        Cells to keep -- boolean mask *or* integer index array.
        ``None`` keeps all cells.
    var_idx : ndarray or None, optional
        Features to keep -- boolean mask *or* integer index array.
        ``None`` keeps all features.
    inplace : bool, optional (default: True)
        If ``True``, modify *adata* in place and return ``None``.
        If ``False``, return a new AnnData (see ``output_file`` for backed
        objects).
    output_file : str or None, optional
        Only used when *adata* is backed and ``inplace=False``.
        If ``None``, the filtered data is written to a temporary file,
        loaded into memory, and the temporary file is deleted — the
        returned AnnData is **in-memory**.
        If a path is given, the filtered data is written to that path and
        a new **backed** AnnData opened at that path is returned.  The
        returned handle is always opened in ``r+`` (read-write) mode,
        regardless of the mode of the input object.
        Ignored for in-memory objects and when ``inplace=True``.
    backed_chunk_size : int, optional (default: 4096)
        Rows per chunk during backed writes.

    Returns
    -------
    AnnData or None
        ``None`` when ``inplace=True``; modified copy otherwise.
    """
    obs_int = (
        _coerce_to_int_idx(obs_idx, adata.n_obs, name="obs")
        if obs_idx is not None
        else np.arange(adata.n_obs, dtype=np.int64)
    )
    var_int = (
        _coerce_to_int_idx(var_idx, adata.n_vars, name="var")
        if var_idx is not None
        else np.arange(adata.n_vars, dtype=np.int64)
    )

    if obs_int.size == 0 or var_int.size == 0:
        raise ValueError(
            "Subset selects zero observations or variables; empty AnnData is not supported"
        )

    backed = is_backed_adata(adata)

    if backed:
        is_view = getattr(adata, "is_view", False)

        if is_view:
            parent = adata._adata_ref
            view_obs = _view_idx_to_int(adata._oidx, parent.n_obs)
            view_var = _view_idx_to_int(adata._vidx, parent.n_vars)
            combined_obs = view_obs[obs_int]
            combined_var = view_var[var_int]
            source = parent
        else:
            combined_obs = obs_int
            combined_var = var_int
            source = adata

        if inplace:
            if is_view:
                materialize_backed(adata, chunk_size=backed_chunk_size)
                no_extra_obs = (obs_idx is None)
                no_extra_var = (var_idx is None)
                if not (no_extra_obs and no_extra_var):
                    subset_backed_inplace(
                        adata, obs_int, var_int, chunk_size=backed_chunk_size,
                    )
            else:
                subset_backed_inplace(
                    adata, obs_int, var_int, chunk_size=backed_chunk_size,
                )
            return None

        filepath = str(source.filename)

        if output_file is None:
            # No destination given: load subset into memory and clean up temp.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(pathlib.Path(filepath).parent), suffix=".h5ad",
            )
            os.close(tmp_fd)
            try:
                _write_filtered_backed(
                    source, combined_obs, combined_var, tmp_path, backed_chunk_size,
                )
                return ad.read_h5ad(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # output_file given: write to that path and return a backed handle.
        dest = str(output_file)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(pathlib.Path(dest).parent), suffix=".h5ad",
        )
        os.close(tmp_fd)
        try:
            _write_filtered_backed(
                source, combined_obs, combined_var, tmp_path, backed_chunk_size,
            )
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        shutil.move(tmp_path, dest)
        return ad.read_h5ad(dest, backed="r+")

    # In-memory path.
    if inplace:
        subset = adata[obs_int, var_int].copy()
        adata._init_as_actual(subset)
        return None

    return adata[obs_int, var_int].copy()


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------

def apply_filter(
    adata: AnnData,
    obs_mask: np.ndarray,
    var_mask: np.ndarray,
    *,
    inplace: bool = True,
    output_file: str | None = None,
    backed_chunk_size: int = 4096,
) -> AnnData | None:
    """Subset *adata* using precomputed boolean masks.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_mask : ndarray of bool, shape ``(n_obs,)``
        Cells to keep.
    var_mask : ndarray of bool, shape ``(n_vars,)``
        Features to keep.
    inplace : bool, optional (default: True)
        If True, modify *adata* in place (returns ``None``).
        If False, return a new (possibly in-memory) AnnData.
    output_file : str or None, optional
        For backed AnnData, write the filtered result to this path using
        chunked h5py I/O (constant-memory).  If ``None`` and ``inplace=True``,
        the backing file is overwritten in place.  If ``None`` and
        ``inplace=False``, an in-memory AnnData is returned and the source
        backing file is left unchanged.
        Ignored for in-memory objects.
    backed_chunk_size : int, optional (default: 4096)
        Rows per chunk during backed writes.
    """
    obs_idx = np.where(obs_mask)[0].astype(np.int64)
    var_idx = np.where(var_mask)[0].astype(np.int64)
    backed = is_backed_adata(adata)

    if backed:
        if inplace and output_file is None:
            subset_backed_inplace(adata, obs_idx, var_idx, chunk_size=backed_chunk_size)
            return None

        if not inplace and output_file is None:
            filepath = str(adata.filename)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(pathlib.Path(filepath).parent), suffix=".h5ad",
            )
            os.close(tmp_fd)
            try:
                _write_filtered_backed(adata, obs_idx, var_idx, tmp_path, backed_chunk_size)
                return ad.read_h5ad(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        dest = str(output_file)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(pathlib.Path(dest).parent), suffix=".h5ad",
        )
        os.close(tmp_fd)
        try:
            _write_filtered_backed(adata, obs_idx, var_idx, tmp_path, backed_chunk_size)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if inplace:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            shutil.move(tmp_path, dest)
            _refresh_backed_handle(adata, dest, mode="r+")
            return None

        shutil.move(tmp_path, dest)
        return ad.read_h5ad(dest, backed="r+")

    # In-memory path.
    if inplace:
        subset = adata[obs_mask, var_mask].copy()
        adata._init_as_actual(subset)
        return None

    return adata[obs_mask, var_mask].copy()


# ---------------------------------------------------------------------------
# filter_anndata (backward-compatible wrapper)
# ---------------------------------------------------------------------------

def filter_anndata(
    adata: AnnData,
    layer_name: str | None = None,
    min_cells_per_feat: int | float | None = None,
    min_feats_per_cell: int | None = None,
    min_umis_per_cell: int | None = None,
    max_umis_per_cell: int | None = None,
    inplace: bool = True,
    filter_adata: bool = True,
    backed_chunk_size: int = 4096,
) -> Union[AnnData, dict, None]:
    """Iterative QC filtering -- backed-safe, single-pass per iteration.

    Thin wrapper around :func:`compute_filter_masks` and
    :func:`apply_filter`.  Iteratively removes cells and features that
    fail specified thresholds until the dimensions stabilise.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_name : str or None, optional (default: None)
        Layer to compute statistics from.  ``None`` uses ``adata.X``.
    min_cells_per_feat : int or float or None, optional
        Minimum cells expressing a feature.  A ``float`` in ``(0, 1)``
        is interpreted as a fraction of ``n_obs``.
    min_feats_per_cell : int or None, optional
        Minimum features detected per cell.
    min_umis_per_cell : int or None, optional
        Minimum total UMI count per cell.
    max_umis_per_cell : int or None, optional
        Maximum total UMI count per cell.
    inplace : bool, optional (default: True)
        Subset ``adata`` in place.  When False, return a new object.
    filter_adata : bool, optional (default: True)
        If True, apply the filter.  If False, return a dict of masks.
    backed_chunk_size : int, optional (default: 4096)
        Rows per streaming chunk (backed mode only).
    """
    obs_mask, var_mask = compute_filter_masks(
        adata,
        layer_name=layer_name,
        min_cells_per_feat=min_cells_per_feat,
        min_feats_per_cell=min_feats_per_cell,
        min_umis_per_cell=min_umis_per_cell,
        max_umis_per_cell=max_umis_per_cell,
        backed_chunk_size=backed_chunk_size,
    )

    if filter_adata:
        return apply_filter(
            adata, obs_mask, var_mask,
            inplace=inplace,
            backed_chunk_size=backed_chunk_size,
        )

    obs_names = np.array(adata.obs_names)
    var_names = np.array(adata.var_names)
    filtered_obs = pd.DataFrame(
        {"name": obs_names, "idx": np.arange(adata.n_obs), "mask": obs_mask}
    )
    filtered_vars = pd.DataFrame(
        {"name": var_names, "idx": np.arange(adata.n_vars), "mask": var_mask}
    )
    return {"fil_vars": filtered_vars, "fil_obs": filtered_obs}
