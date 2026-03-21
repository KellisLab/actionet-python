"""Chunked matrix access for AnnData ``.X`` and ``.layers``.

This module provides :class:`MatrixSource`, a unified abstraction that
transparently handles both in-memory (dense / sparse) and backed (on-disk
HDF5) AnnData matrices.  All public ACTIONet functions that need to read
or transform the expression matrix use ``MatrixSource`` under the hood so
that backed AnnData objects are streamed in constant-memory row chunks
rather than materialised wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ._backed_persist import persist_layer


def _is_sparse_matrix_like(X: object) -> bool:
    """Return ``True`` if *X* looks like a sparse matrix.

    Checks ``scipy.sparse.issparse``, a ``format`` attribute (backed sparse
    datasets expose this), and the HDF5 ``encoding-type`` group attribute.
    """
    if sp.issparse(X):
        return True

    fmt = getattr(X, "format", None)
    if isinstance(fmt, str) and fmt.lower() in {"csr", "csc", "coo", "bsr", "dia", "dok", "lil"}:
        return True

    group = getattr(X, "group", None)
    if group is not None and hasattr(group, "attrs"):
        enc = group.attrs.get("encoding-type", "")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8", errors="ignore")
        if isinstance(enc, str) and ("csr" in enc or "csc" in enc):
            return True

    return False


@dataclass(frozen=True)
class MatrixChunk:
    """A contiguous row slice ``[start, end)`` of a matrix.

    Attributes
    ----------
    start : int
        First row index (inclusive).
    end : int
        Last row index (exclusive).
    block : array-like
        The ``(end - start, n_vars)`` sub-matrix.
    """
    start: int
    end: int
    block: object


class MatrixSource:
    """Unified matrix accessor for in-memory and backed AnnData matrices.

    Wraps either ``adata.X`` or ``adata.layers[layer]`` and exposes chunked
    iteration, aggregation, and matrix--vector product helpers that work
    identically on in-memory and HDF5-backed sparse data.

    Parameters
    ----------
    adata : AnnData
        The annotated data object.
    layer : str or None
        Which layer to access.  ``None`` (default) uses ``adata.X``.

    Attributes
    ----------
    shape : tuple[int, int]
        ``(n_obs, n_vars)``
    n_obs : int
        Number of observations (rows / cells).
    n_vars : int
        Number of variables (columns / genes).
    is_backed : bool
        Whether the underlying AnnData is in backed mode.
    is_sparse : bool
        Whether the underlying matrix is sparse (or sparse-on-disk).
    """

    def __init__(self, adata: AnnData, layer: Optional[str] = None):
        self.adata = adata
        self.layer = layer
        self.is_backed = bool(getattr(adata, "isbacked", False))

        matrix = self.matrix
        self.shape = tuple(int(v) for v in matrix.shape)
        self.n_obs, self.n_vars = self.shape
        self.is_sparse = _is_sparse_matrix_like(matrix)

    @property
    def matrix(self):
        """Return the raw underlying matrix (may be backed / lazy)."""
        if self.layer is None:
            return self.adata.X
        if self.layer not in self.adata.layers:
            raise KeyError(f"Layer '{self.layer}' not found in adata.layers")
        return self.adata.layers[self.layer]

    def to_memory(self):
        """Materialise the full matrix into RAM.

        Returns a copy so that mutations do not affect the source.
        """
        X = self.matrix
        if hasattr(X, "to_memory"):
            return X.to_memory()
        if sp.issparse(X):
            return X.copy()
        return np.asarray(X).copy()

    # ------------------------------------------------------------------
    # Row access
    # ------------------------------------------------------------------

    def get_rows(
        self,
        start: int,
        end: int,
        col_indices: Optional[Sequence[int]] = None,
    ):
        """Return rows ``[start, end)`` and optionally subset columns.

        Parameters
        ----------
        start, end : int
            Row range (0-indexed, half-open).
        col_indices : array-like of int, optional
            If given, only these columns are returned.
        """
        if not (0 <= start <= end <= self.n_obs):
            raise IndexError(f"Invalid row bounds [{start}, {end}) for n_obs={self.n_obs}")

        block = self.matrix[start:end, :]
        if col_indices is not None:
            block = block[:, np.asarray(col_indices, dtype=np.int64)]
        return block

    def iter_row_chunks(
        self,
        chunk_size: int = 4096,
        col_indices: Optional[Sequence[int]] = None,
    ) -> Iterator[MatrixChunk]:
        """Iterate over contiguous row chunks of the matrix.

        Parameters
        ----------
        chunk_size : int
            Maximum rows per chunk.
        col_indices : array-like of int, optional
            Column subset applied to every chunk.

        Yields
        ------
        MatrixChunk
        """
        chunk_size = int(max(1, chunk_size))
        for start in range(0, self.n_obs, chunk_size):
            end = min(start + chunk_size, self.n_obs)
            yield MatrixChunk(start=start, end=end, block=self.get_rows(start, end, col_indices=col_indices))

    def iter_selected_row_chunks(
        self,
        row_indices: Sequence[int],
        chunk_size: int = 4096,
        col_indices: Optional[Sequence[int]] = None,
    ) -> Iterator[tuple[np.ndarray, object]]:
        """Iterate over chunks of explicitly selected rows.

        Parameters
        ----------
        row_indices : array-like of int
            Row indices to include.
        chunk_size : int
            Maximum indices per chunk.
        col_indices : array-like of int, optional
            Column subset applied to every chunk.

        Yields
        ------
        (row_idx_chunk, block)
            A pair of the index array and the corresponding sub-matrix.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        chunk_size = int(max(1, chunk_size))
        for pos in range(0, row_indices.size, chunk_size):
            rows = row_indices[pos:pos + chunk_size]
            block = self.matrix[rows, :]
            if col_indices is not None:
                block = block[:, np.asarray(col_indices, dtype=np.int64)]
            yield rows, block

    # ------------------------------------------------------------------
    # Row write
    # ------------------------------------------------------------------

    def set_rows(self, start: int, end: int, values) -> None:
        """Write a contiguous row block back to the source matrix.

        For backed sparse matrices this writes directly to the underlying
        h5py ``data`` array, bypassing anndata's deprecated ``__setitem__``
        entirely.  If the on-disk dtype differs from *values* (e.g. int64
        counts vs. float64 after normalisation), the ``data`` dataset is
        re-cast in constant-memory chunks before writing.

        For in-memory sparse matrices, the matrix is promoted to float64
        when the incoming values have a wider dtype (e.g. int64 -> float64)
        to avoid silent truncation by scipy's ``__setitem__``.
        """
        if not (0 <= start <= end <= self.n_obs):
            raise IndexError(f"Invalid row bounds [{start}, {end}) for n_obs={self.n_obs}")

        if self.is_backed and self.is_sparse:
            self._h5py_set_sparse_rows(start, end, values)
            return

        target = self.matrix

        # Promote in-memory sparse matrix dtype if needed to avoid
        # silent truncation (e.g. float64 values -> int64 matrix).
        new_dtype = values.dtype if hasattr(values, "dtype") else None
        if (
            not self.is_backed
            and sp.issparse(target)
            and new_dtype is not None
            and not np.can_cast(new_dtype, target.dtype, casting="same_kind")
        ):
            promoted = target.astype(new_dtype, copy=False)
            if self.layer is None:
                self.adata.X = promoted
            else:
                self.adata.layers[self.layer] = promoted
            target = promoted

        target[start:end, :] = values

    # ------------------------------------------------------------------
    # Direct h5py write helpers (backed sparse)
    # ------------------------------------------------------------------

    def _resolve_h5_group(self):
        """Return the raw h5py ``Group`` that stores the sparse matrix on disk.

        For ``.X`` this is ``file["X"]``; for a layer it is
        ``file["layers/<name>"]``.  Accesses the underlying ``h5py.File``
        directly, bypassing anndata's dataset wrappers.
        """
        h5file = self.adata.file._file
        if self.layer is None:
            return h5file["X"]
        return h5file["layers"][self.layer]

    def backed_sparse_format(self) -> Optional[str]:
        """Return ``'csr'`` or ``'csc'`` for backed sparse storage when known."""
        if not (self.is_backed and self.is_sparse):
            return None

        fmt = getattr(self.matrix, "format", None)
        if isinstance(fmt, str):
            fmt = fmt.lower()
            if fmt in {"csr", "csc"}:
                return fmt

        grp = self._resolve_h5_group()
        enc = grp.attrs.get("encoding-type", "")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8", errors="ignore")
        if isinstance(enc, str):
            enc = enc.lower()
            if "csr" in enc:
                return "csr"
            if "csc" in enc:
                return "csc"

        return None

    @staticmethod
    def _h5py_cast_data_dataset(
        grp,
        target_dtype: np.dtype,
        copy_chunk: int = 10_000_000,
    ) -> None:
        """Re-cast the ``data`` dataset inside *grp* to *target_dtype*.

        Operates in constant memory by copying in chunks of *copy_chunk*
        elements to a temporary dataset, then swapping.  Peak RAM is
        ``copy_chunk * itemsize`` (80 MB at the default 10 M elements for
        float64).
        """
        old_ds = grp["data"]
        total = old_ds.shape[0]
        compression = old_ds.compression
        compression_opts = old_ds.compression_opts

        tmp_name = "__data_cast_tmp"
        if tmp_name in grp:
            del grp[tmp_name]
        new_ds = grp.create_dataset(
            tmp_name,
            shape=(total,),
            dtype=target_dtype,
            compression=compression,
            compression_opts=compression_opts,
        )

        for pos in range(0, total, copy_chunk):
            end = min(pos + copy_chunk, total)
            new_ds[pos:end] = old_ds[pos:end].astype(target_dtype)

        del grp["data"]
        grp.move(tmp_name, "data")

    def _h5py_set_sparse_rows(self, start: int, end: int, values) -> None:
        """Write rows ``[start, end)`` directly to the h5py sparse group.

        Validates that the incoming block preserves the sparsity structure
        (same nnz per row) and handles dtype promotion when needed.
        """
        sparse_format = self.backed_sparse_format()
        if sparse_format == "csc":
            raise ValueError(
                "Direct row-wise writes to CSC-backed sparse matrices are not "
                "supported. Rewrite the destination through CSR-backed storage."
            )

        values_csr = sp.csr_matrix(values) if not sp.issparse(values) else values.tocsr()

        grp = self._resolve_h5_group()
        indptr_ds = grp["indptr"]
        data_ds = grp["data"]

        ip_start = int(indptr_ds[start])
        ip_end = int(indptr_ds[end])
        expected_nnz = ip_end - ip_start

        actual_nnz = values_csr.nnz
        if actual_nnz != expected_nnz:
            raise ValueError(
                f"Sparsity structure changed: on-disk rows [{start}, {end}) "
                f"have {expected_nnz} stored elements but the new block has "
                f"{actual_nnz}.  In-place backed writes require identical "
                f"sparsity patterns."
            )

        on_disk_dtype = data_ds.dtype
        new_dtype = values_csr.data.dtype
        if on_disk_dtype != new_dtype:
            self._h5py_cast_data_dataset(grp, new_dtype)

        grp["data"][ip_start:ip_end] = values_csr.data

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def row_sums(
        self,
        chunk_size: int = 4096,
        col_indices: Optional[Sequence[int]] = None,
        row_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return per-row sums as a 1-D float64 array.

        Parameters
        ----------
        chunk_size : int
            Rows per streaming chunk.
        col_indices : array-like of int, optional
            Restrict summation to these columns.
        row_indices : array-like of int, optional
            If given, only these rows are computed (returned array has the
            same length as *row_indices*).
        """
        if row_indices is None:
            out = np.zeros(self.n_obs, dtype=np.float64)
            for chunk in self.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices):
                block = chunk.block
                if sp.issparse(block):
                    out[chunk.start:chunk.end] = np.asarray(block.sum(axis=1)).ravel()
                else:
                    out[chunk.start:chunk.end] = np.asarray(block, dtype=np.float64).sum(axis=1)
            return out

        row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        out = np.zeros(row_indices.size, dtype=np.float64)
        pos = 0
        for rows, block in self.iter_selected_row_chunks(
            row_indices,
            chunk_size=chunk_size,
            col_indices=col_indices,
        ):
            size = rows.size
            if sp.issparse(block):
                out[pos:pos + size] = np.asarray(block.sum(axis=1)).ravel()
            else:
                out[pos:pos + size] = np.asarray(block, dtype=np.float64).sum(axis=1)
            pos += size
        return out

    def col_sums(
        self,
        chunk_size: int = 4096,
        col_indices: Optional[Sequence[int]] = None,
        row_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return per-column sums as a 1-D float64 array.

        Parameters
        ----------
        chunk_size : int
            Rows per streaming chunk.
        col_indices : array-like of int, optional
            Restrict to these columns (output length matches).
        row_indices : array-like of int, optional
            Restrict to these rows.
        """
        if col_indices is None:
            n_cols = self.n_vars
        else:
            n_cols = len(col_indices)

        out = np.zeros(n_cols, dtype=np.float64)
        if row_indices is None:
            iterator = ((None, c.block) for c in self.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices))
        else:
            iterator = self.iter_selected_row_chunks(
                row_indices,
                chunk_size=chunk_size,
                col_indices=col_indices,
            )

        for _, block in iterator:
            if sp.issparse(block):
                out += np.asarray(block.sum(axis=0)).ravel()
            else:
                out += np.asarray(block, dtype=np.float64).sum(axis=0)
        return out

    def nnz_row_counts(
        self,
        chunk_size: int = 4096,
        col_indices: Optional[Sequence[int]] = None,
        row_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Count non-zero entries per row (int64)."""
        if row_indices is None:
            out = np.zeros(self.n_obs, dtype=np.int64)
            for chunk in self.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices):
                block = chunk.block
                if sp.issparse(block):
                    out[chunk.start:chunk.end] = np.asarray(block.getnnz(axis=1)).ravel()
                else:
                    out[chunk.start:chunk.end] = np.count_nonzero(np.asarray(block), axis=1)
            return out

        row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        out = np.zeros(row_indices.size, dtype=np.int64)
        pos = 0
        for rows, block in self.iter_selected_row_chunks(
            row_indices,
            chunk_size=chunk_size,
            col_indices=col_indices,
        ):
            size = rows.size
            if sp.issparse(block):
                out[pos:pos + size] = np.asarray(block.getnnz(axis=1)).ravel()
            else:
                out[pos:pos + size] = np.count_nonzero(np.asarray(block), axis=1)
            pos += size
        return out

    def nnz_col_counts(
        self,
        chunk_size: int = 4096,
        col_indices: Optional[Sequence[int]] = None,
        row_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Count non-zero entries per column (int64)."""
        if col_indices is None:
            n_cols = self.n_vars
        else:
            n_cols = len(col_indices)

        out = np.zeros(n_cols, dtype=np.int64)
        if row_indices is None:
            iterator = ((None, c.block) for c in self.iter_row_chunks(chunk_size=chunk_size, col_indices=col_indices))
        else:
            iterator = self.iter_selected_row_chunks(
                row_indices,
                chunk_size=chunk_size,
                col_indices=col_indices,
            )

        for _, block in iterator:
            if sp.issparse(block):
                out += np.asarray(block.getnnz(axis=0)).ravel().astype(np.int64, copy=False)
            else:
                out += np.count_nonzero(np.asarray(block), axis=0)
        return out

    # ------------------------------------------------------------------
    # Feature / column extraction
    # ------------------------------------------------------------------

    def feature_subset(
        self,
        feature_indices: Sequence[int],
        *,
        chunk_size: int = 4096,
        prefer_sparse: bool | None = None,
        row_indices: Optional[Sequence[int]] = None,
    ):
        """Extract a subset of columns (features), returning a full matrix.

        The result is always in-memory; peak RAM is proportional to
        ``n_selected_rows * len(feature_indices)``.

        Parameters
        ----------
        feature_indices : array-like of int
            Column indices to extract.
        chunk_size : int
            Rows per streaming chunk.
        prefer_sparse : bool or None
            Force sparse (True) or dense (False) output.  ``None`` infers
            from the source data.
        row_indices : array-like of int, optional
            If given, only these rows are extracted.
        """
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D sequence")

        blocks = []
        sparse_seen = False
        if row_indices is None:
            iterator = (c.block for c in self.iter_row_chunks(chunk_size=chunk_size, col_indices=feature_indices))
        else:
            iterator = (
                block for _, block in self.iter_selected_row_chunks(
                    row_indices,
                    chunk_size=chunk_size,
                    col_indices=feature_indices,
                )
            )

        for block in iterator:
            sparse_seen = sparse_seen or sp.issparse(block)
            blocks.append(block)

        if len(blocks) == 0:
            if prefer_sparse:
                return sp.csr_matrix((0, feature_indices.size))
            return np.zeros((0, feature_indices.size), dtype=np.float64)

        if prefer_sparse is None:
            prefer_sparse = sparse_seen

        if prefer_sparse:
            blocks_sp = [b if sp.issparse(b) else sp.csr_matrix(np.asarray(b)) for b in blocks]
            return sp.vstack(blocks_sp, format="csr")

        blocks_dense = [b.toarray() if sp.issparse(b) else np.asarray(b) for b in blocks]
        return np.vstack(blocks_dense)

    # ------------------------------------------------------------------
    # Matrix--vector products
    # ------------------------------------------------------------------

    def xt_dot(self, right: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Compute ``X.T @ right`` in streamed row chunks.

        Parameters
        ----------
        right : ndarray, shape ``(n_obs, k)``
            Dense right-hand side.
        chunk_size : int
            Rows per chunk.

        Returns
        -------
        ndarray, shape ``(n_vars, k)``
        """
        right = np.asarray(right, dtype=np.float64)
        if right.ndim != 2 or right.shape[0] != self.n_obs:
            raise ValueError(
                f"right must have shape (n_obs, k) where n_obs={self.n_obs}, got {right.shape}"
            )

        out = np.zeros((self.n_vars, right.shape[1]), dtype=np.float64)
        for chunk in self.iter_row_chunks(chunk_size=chunk_size):
            block = chunk.block
            right_block = right[chunk.start:chunk.end, :]
            if sp.issparse(block):
                out += np.asarray(block.T.dot(right_block))
            else:
                out += np.asarray(block, dtype=np.float64).T @ right_block
        return out

    def x_dot(self, right: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Compute ``X @ right`` in streamed row chunks.

        Parameters
        ----------
        right : ndarray, shape ``(n_vars, k)``
            Dense right-hand side.
        chunk_size : int
            Rows per chunk.

        Returns
        -------
        ndarray, shape ``(n_obs, k)``
        """
        right = np.asarray(right, dtype=np.float64)
        if right.ndim != 2 or right.shape[0] != self.n_vars:
            raise ValueError(
                f"right must have shape (n_vars, k) where n_vars={self.n_vars}, got {right.shape}"
            )

        out = np.zeros((self.n_obs, right.shape[1]), dtype=np.float64)
        for chunk in self.iter_row_chunks(chunk_size=chunk_size):
            block = chunk.block
            if sp.issparse(block):
                out[chunk.start:chunk.end, :] = np.asarray(block.dot(right))
            else:
                out[chunk.start:chunk.end, :] = np.asarray(block, dtype=np.float64) @ right
        return out

    # ------------------------------------------------------------------
    # Scalar reductions
    # ------------------------------------------------------------------

    def global_min(self, chunk_size: int = 4096, row_indices: Optional[Sequence[int]] = None) -> float:
        """Compute the global minimum element.

        For sparse matrices, implicit (unstored) zeros are taken into
        account: if all stored values are positive the minimum is ``0.0``,
        not the smallest stored value.
        """
        min_val = np.inf
        if row_indices is None:
            iterator = (c.block for c in self.iter_row_chunks(chunk_size=chunk_size))
        else:
            iterator = (
                block for _, block in self.iter_selected_row_chunks(
                    row_indices,
                    chunk_size=chunk_size,
                )
            )

        for block in iterator:
            if sp.issparse(block):
                if block.nnz == 0:
                    block_min = 0.0
                else:
                    block_min = min(0.0, float(block.data.min()))
            else:
                block_min = float(np.asarray(block).min())
            min_val = min(min_val, block_min)

        if not np.isfinite(min_val):
            return 0.0
        return float(min_val)

    # ------------------------------------------------------------------
    # Row-wise in-place transform
    # ------------------------------------------------------------------

    def apply_rowwise(
        self,
        fn: Callable[[object, int, int], object],
        chunk_size: int = 4096,
    ) -> None:
        """Apply *fn* to each row chunk and write the result back in-place.

        Parameters
        ----------
        fn : callable ``(block, start, end) -> block``
            Receives the sub-matrix and its row range, returns a
            replacement of the same shape.
        chunk_size : int
            Rows per chunk.
        """
        for chunk in self.iter_row_chunks(chunk_size=chunk_size):
            updated = fn(chunk.block, chunk.start, chunk.end)
            self.set_rows(chunk.start, chunk.end, updated)
