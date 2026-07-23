"""Total-count normalization for AnnData (in-memory and backed)."""

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse import issparse

from ..io.compression import sparse_group_format
from ..io.chunking import (
    DEFAULT_BACKED_READ_CHUNK_SIZE,
    resolve_backed_write_chunk_size,
)
from ..io.persist import (
    is_writable_backed,
    _refresh_backed_handle,
)
from ..io.matrix_source import MatrixSource


def _copy_h5_attrs(src, dst) -> None:
    """Copy all attributes from one h5py object to another."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _safe_row_scale(target_sum: float, row_sums: np.ndarray) -> np.ndarray:
    """Return ``target_sum / row_sums`` with zero rows mapped to zero scale.

    All five normalize paths in this module share the same divide-with-guard
    pattern: rows whose sum is <= 0 get a scale of 0 (so their data survives
    unchanged, effectively skipping normalization for those rows). Factoring
    the idiom out keeps the arithmetic in one place.
    """
    return np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )


def normalize_anndata(
    adata: AnnData,
    target_sum: float = 1e4,
    log_transform: bool = True,
    log_base: Optional[float] = None,
    pseudocount: float = 1.0,
    layer: str | None = None,
    backed_chunk_size: int = DEFAULT_BACKED_READ_CHUNK_SIZE,
    dtype_out: str = "float32",
    inplace: bool = True,
    layer_added: str | None = None,
    backed_write_chunk_size: int | None = None,
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
    backed_chunk_size : int, optional (default: 8192)
        Number of rows per chunk for backed row-stat computation
        (read/compute path). Ignored for in-memory objects.
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
    backed_write_chunk_size : int or None, optional (default: None)
        Number of rows per transform/write chunk in backed mode. ``None``
        uses the shared write default of ``16384``. Atlas-scale HDF5 writes
        may benefit from starting with ``32768``; larger values use
        proportionally more temporary memory.

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
    backed_chunk_size, backed_write_chunk_size = resolve_backed_write_chunk_size(
        backed_chunk_size,
        backed_write_chunk_size,
    )

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
            if not is_writable_backed(adata):
                raise ValueError(
                    "`layer_added` with backed AnnData requires writable mode 'r+'. "
                    'Re-open with `ad.read_h5ad(path, backed="r+")`.'
                )
            if source.is_sparse and source.backed_sparse_format() == "csc":
                _normalize_backed_csc_via_csr_rewrite(
                    source,
                    target_sum=target_sum,
                    log_transform=log_transform,
                    log_base=log_base,
                    pseudocount=pseudocount,
                    read_chunk_size=backed_chunk_size,
                    write_chunk_size=backed_write_chunk_size,
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
                    read_chunk_size=backed_chunk_size,
                    write_chunk_size=backed_write_chunk_size,
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
            target_sum=target_sum,
            log_transform=log_transform,
            log_base=log_base,
            pseudocount=pseudocount,
            read_chunk_size=backed_chunk_size,
            write_chunk_size=backed_write_chunk_size,
            dtype_out=out_dtype,
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
        scaling = _safe_row_scale(target_sum, row_sums)

        if X.nnz > 0:
            row_nnz = np.diff(X.indptr)
            X.data *= np.repeat(scaling, row_nnz)

            if log_transform:
                if X.data.min() < 0:
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
    scaling = _safe_row_scale(target_sum, row_sums)

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
    read_chunk_size: int,
    write_chunk_size: int,
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
            read_chunk_size=read_chunk_size,
            write_chunk_size=write_chunk_size,
            dtype_out=dtype_out,
            layer_added=None,
        )
        return

    row_sums = source.row_sums(chunk_size=read_chunk_size)
    scaling = _safe_row_scale(target_sum, row_sums)
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

    source.apply_rowwise(_normalize_block, chunk_size=write_chunk_size)


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
            "data",
            shape=(total_nnz,),
            dtype=dtype_out,
            **data_kwargs,
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
            "indptr",
            data=indptr_src[...],
            **indptr_kwargs,
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
    read_chunk_size: int,
    write_chunk_size: int,
    dtype_out: np.dtype,
) -> None:
    """Streamed normalize: read from *source*, write to ``layers[layer_added]``.

    Two passes over the source matrix:
      1. Compute per-row sums.
      2. Read each chunk, normalize in-memory, write the ``data`` slice
         (sparse) or row slice (dense) directly to the destination.

    This avoids the full-file copy and dtype recast of the legacy path.
    """
    row_sums = source.row_sums(chunk_size=read_chunk_size)
    scaling = _safe_row_scale(target_sum, row_sums)

    log_scale = None
    if log_transform:
        log_scale = 1.0 if log_base is None else 1.0 / np.log(log_base)

    h5file = adata.file._file
    dest_node = h5file["layers"][layer_added]

    is_sparse_dest = hasattr(dest_node, "keys") and "data" in dest_node

    if is_sparse_dest:
        dest_encoding = sparse_group_format(dest_node)
        if dest_encoding != "csr":
            raise ValueError(
                "Streamed sparse normalization destinations must use CSR-backed storage."
            )

        dest_data_ds = dest_node["data"]
        dest_indptr_ds = dest_node["indptr"]

        for chunk in source.iter_row_chunks(chunk_size=write_chunk_size):
            block = chunk.block
            scale = scaling[chunk.start : chunk.end]

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
                    dtype_out,
                    copy=False,
                )
    else:
        for chunk in source.iter_row_chunks(chunk_size=write_chunk_size):
            block = chunk.block
            scale = scaling[chunk.start : chunk.end]

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

            dest_node[chunk.start : chunk.end, :] = arr

    h5file.flush()


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
        chunks = tuple(
            min(int(src_chunk), int(dim)) for src_chunk, dim in zip(src_ds.chunks, shape)
        )
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
            row_sums[chunk.start : chunk.end] = np.asarray(block_csr.sum(axis=1)).ravel()
            row_nnz[chunk.start : chunk.end] = np.diff(block_csr.indptr).astype(
                np.int64, copy=False
            )
        else:
            arr = np.asarray(block, dtype=np.float64)
            row_sums[chunk.start : chunk.end] = arr.sum(axis=1)
            row_nnz[chunk.start : chunk.end] = np.count_nonzero(arr, axis=1)

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
        scale = scaling[chunk.start : chunk.end]
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

        dest_indices_ds[ip_start:ip_end] = block_csr.indices.astype(
            dest_indices_ds.dtype, copy=False
        )
        dest_data_ds[ip_start:ip_end] = block_csr.data.astype(dtype_out, copy=False)


def _normalize_backed_csc_via_csr_rewrite(
    source: MatrixSource,
    *,
    target_sum: float,
    log_transform: bool,
    log_base: Optional[float],
    pseudocount: float = 1.0,
    read_chunk_size: int,
    write_chunk_size: int,
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
    row_sums, row_nnz = _compute_backed_sparse_row_stats(
        source,
        chunk_size=read_chunk_size,
    )
    scaling = _safe_row_scale(target_sum, row_sums)
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
            chunk_size=write_chunk_size,
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
