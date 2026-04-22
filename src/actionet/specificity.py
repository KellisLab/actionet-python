"""Specificity APIs and helpers.

This module centralizes feature/archetype specificity logic that was previously
embedded in ``core.py``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from ._backed_persist import persist_updates
from ._matrix_source import MatrixSource
from .anndata_utils import anndata_to_matrix
from .backed_io import _backed_group_path, _open_backed_operator
from .lazy_transform import (
    LazyTransform,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)


def _run_specificity_backed_sparse(
    adata: AnnData,
    layer: Optional[str],
    chunk_size: int,
    *,
    H: Optional[np.ndarray] = None,
    labels_int: Optional[np.ndarray] = None,
    n_threads: int = 0,
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
) -> dict:
    """Dispatch backed sparse specificity through the C++ ABI."""
    file_path = str(adata.filename)
    group_path = _backed_group_path(layer)
    with _open_backed_operator(
        adata=adata,
        file_path=file_path,
        group_path=group_path,
        context="compute_feature_specificity_sparse",
        chunk_size=chunk_size,
        row_scale_factors=row_scale_factors,
        apply_log1p=apply_log1p,
        log_scale=log_scale,
        n_threads=n_threads,
    ) as op:
        if H is not None:
            return _core.archetype_feature_specificity_backed_operator(op, H, n_threads)
        return _core.compute_feature_specificity_backed_operator(op, labels_int, n_threads)


def _run_specificity_backed_dense(
    adata: AnnData,
    layer: Optional[str],
    chunk_size: int,
    *,
    H: Optional[np.ndarray] = None,
    labels_int: Optional[np.ndarray] = None,
    n_threads: int = 0,
    row_scale_factors: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    log_scale: float = 1.0,
) -> dict:
    """Dispatch backed dense specificity through the C++ ABI."""
    file_path = str(adata.filename)
    group_path = _backed_group_path(layer)
    with _open_backed_operator(
        adata=adata,
        file_path=file_path,
        group_path=group_path,
        context="compute_feature_specificity_dense",
        chunk_size=chunk_size,
        row_scale_factors=row_scale_factors,
        apply_log1p=apply_log1p,
        log_scale=log_scale,
        n_threads=n_threads,
    ) as op:
        if H is not None:
            return _core.archetype_feature_specificity_backed_dense_operator(op, H, n_threads)
        return _core.compute_feature_specificity_backed_dense_operator(op, labels_int, n_threads)


def _encode_labels_for_specificity(labels_arr: np.ndarray, n_obs: int) -> np.ndarray:
    """Convert user labels to contiguous 1-based int32 cluster ids.

    Semantics:
    - Non-integer labels use pandas Categorical codes (missing -> 0).
    - Integer labels treat values >= 0 as valid cluster ids and values < 0
      as missing/unassigned. Valid ids are remapped to contiguous 1..k while
      preserving ascending numeric order.
    """
    labels_arr = np.asarray(labels_arr).reshape(-1)
    if labels_arr.shape[0] != n_obs:
        raise ValueError(
            f"labels length ({labels_arr.shape[0]}) does not match number of observations ({n_obs})"
        )

    from pandas import Categorical
    from pandas.api.types import is_integer_dtype

    if is_integer_dtype(labels_arr):
        labels_raw = labels_arr.astype(np.int64, copy=False)
        labels_int = np.zeros(labels_raw.shape[0], dtype=np.int32)

        valid = labels_raw >= 0
        if np.any(valid):
            unique_ids = np.unique(labels_raw[valid])
            mapped = np.searchsorted(unique_ids, labels_raw[valid]) + 1
            if mapped.size > 0 and mapped.max() > np.iinfo(np.int32).max:
                raise ValueError("Too many unique labels to encode into int32 cluster ids")
            labels_int[valid] = mapped.astype(np.int32, copy=False)
        return labels_int

    cat = Categorical(labels_arr)
    labels_int64 = cat.codes.astype(np.int64, copy=False) + 1
    if labels_int64.size > 0 and labels_int64.max() > np.iinfo(np.int32).max:
        raise ValueError("Too many unique labels to encode into int32 cluster ids")
    return labels_int64.astype(np.int32, copy=False)


def _validate_archetype_membership(H: np.ndarray, n_obs: int) -> np.ndarray:
    """Validate and normalize an archetype membership matrix (cells x k)."""
    H = np.ascontiguousarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"Archetype matrix must be 2D (n_obs, k); got ndim={H.ndim}")
    if H.shape[0] != n_obs:
        raise ValueError(
            f"Archetype matrix row count ({H.shape[0]}) must equal n_obs ({n_obs})"
        )
    if H.shape[1] == 0:
        raise ValueError("Archetype matrix must contain at least one archetype column")
    return H


def _cluster_names_for_specificity_labels(labels_arr: np.ndarray) -> np.ndarray:
    """Return cluster names in the same column order as specificity outputs."""
    labels_arr = np.asarray(labels_arr).reshape(-1)

    from pandas import Categorical
    from pandas.api.types import is_integer_dtype

    if is_integer_dtype(labels_arr):
        labels_raw = labels_arr.astype(np.int64, copy=False)
        valid = labels_raw >= 0
        if not np.any(valid):
            return np.array([], dtype=labels_raw.dtype)
        return np.unique(labels_raw[valid])

    cat = Categorical(labels_arr)
    return np.asarray(cat.categories)


def compute_feature_specificity(
    adata: AnnData,
    labels: Union[str, np.ndarray],
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "specificity",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
    return_raw: bool = False,
    lazy_transform: Optional[LazyTransform] = None,
) -> Optional[Union[AnnData, dict]]:
    """Compute feature specificity scores for clusters/archetypes.

    Parameters
    ----------
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        When provided, the backed operator applies per-row normalization
        (target-sum scaling) and log1p on-the-fly without requiring a
        persisted ``logcounts`` layer.  Only valid when ``layer=None``
        and the input is backed.  Create with
        :func:`~actionet.lazy_transform.create_lazy_transform`.
    """
    if not inplace and not return_raw:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    if isinstance(labels, str):
        if labels not in adata.obs:
            raise ValueError(f"Labels '{labels}' not found in adata.obs.")
        labels_arr = adata.obs[labels].values
    else:
        labels_arr = np.asarray(labels)

    if hasattr(labels_arr, "categories"):
        labels_arr = np.asarray(labels_arr)

    labels_int = _encode_labels_for_specificity(labels_arr, n_obs=source.n_obs)

    if source.is_backed:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        if source.is_sparse:
            result = _run_specificity_backed_sparse(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                labels_int=labels_int,
                n_threads=n_threads,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
            )
        else:
            result = _run_specificity_backed_dense(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                labels_int=labels_int,
                n_threads=n_threads,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
            )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        if sp.issparse(S):
            result = _core.compute_feature_specificity_sparse(S, labels_int, n_threads)
        else:
            result = _core.compute_feature_specificity_dense(S, labels_int, n_threads)

    if return_raw:
        return result

    persist_updates(
        adata,
        varm={
            f"{key_added}_profile": result["average_profile"],
            f"{key_added}_upper": result["upper_significance"],
            f"{key_added}_lower": result["lower_significance"],
        },
    )
    if not inplace:
        return adata
    return None


def compute_archetype_feature_specificity(
    adata: AnnData,
    archetype_key: Union[str, np.ndarray] = "archetype_footprint",
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "archetype",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
    return_raw: bool = False,
    lazy_transform: Optional[LazyTransform] = None,
) -> Optional[Union[AnnData, dict]]:
    """Compute feature specificity scores for archetypes using archetype footprint matrix.

    For each archetype, scores each feature (gene) by how specifically it is
    expressed in cells with high membership in that archetype.  Results are
    stored in ``adata.varm`` under keys derived from ``key_added``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x genes).
    archetype_key : str or np.ndarray, default ``"archetype_footprint"``
        Key in ``adata.obsm`` holding the archetype membership matrix
        (cells x archetypes), or the matrix itself as an ndarray.
    layer : str, optional
        Layer of ``adata`` to use as the expression matrix.  If ``None``,
        ``adata.X`` is used.
    n_threads : int, default 0
        Number of threads for the C++ backend.  ``0`` lets the backend choose.
    key_added : str, default ``"archetype"``
        Prefix for the keys written to ``adata.varm``:

        - ``<key_added>_feat_profile`` — average expression profile per archetype
          (genes x archetypes).
        - ``<key_added>_feat_specificity_upper`` — upper-tail specificity scores
          (genes x archetypes).
        - ``<key_added>_feat_specificity_lower`` — lower-tail specificity scores
          (genes x archetypes).
    inplace : bool, default ``True``
        If ``True``, write results into ``adata`` and return ``None``.
        If ``False``, operate on a copy of ``adata`` and return it.
        Ignored when ``return_raw=True``.
    backed_chunk_size : int, default 4096
        Row chunk size used when streaming a backed (HDF5-on-disk) AnnData.
    return_raw : bool, default ``False``
        If ``True``, return the raw result dict from the C++ backend instead of
        writing to ``adata``.  The dict contains keys ``"archetypes"``,
        ``"upper_significance"``, and ``"lower_significance"``.
        When ``True``, ``adata`` is never modified and ``inplace`` is ignored.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        When provided, the backed operator applies per-row normalization
        (target-sum scaling) and log1p on-the-fly without requiring a
        persisted ``logcounts`` layer.  Only valid when ``layer=None``
        and the input is backed.  Create with
        :func:`~actionet.lazy_transform.create_lazy_transform`.

    Returns
    -------
    None
        When ``inplace=True`` (default).
    AnnData
        A modified copy of ``adata`` when ``inplace=False``.
    dict
        Raw C++ result dict when ``return_raw=True``.
    """
    if not inplace and not return_raw:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    if isinstance(archetype_key, str):
        if archetype_key not in adata.obsm:
            raise ValueError(f"Archetype matrix '{archetype_key}' not found in adata.obsm.")
        H = adata.obsm[archetype_key]
    else:
        H = np.asarray(archetype_key)

    H = _validate_archetype_membership(H, n_obs=source.n_obs)

    if source.is_backed:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        if source.is_sparse:
            result = _run_specificity_backed_sparse(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                H=H,
                n_threads=n_threads,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
            )
        else:
            result_stream = _run_specificity_backed_dense(
                adata,
                layer=layer,
                chunk_size=backed_chunk_size,
                H=H,
                n_threads=n_threads,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
            )
            result = {
                "archetypes": result_stream["archetypes"],
                "upper_significance": result_stream["upper_significance"],
                "lower_significance": result_stream["lower_significance"],
            }
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        if sp.issparse(S):
            result = _core.archetype_feature_specificity_sparse(S, H, n_threads)
        else:
            result = _core.archetype_feature_specificity_dense(S, H, n_threads)

    if return_raw:
        return result

    persist_updates(
        adata,
        varm={
            f"{key_added}_feat_profile": result["archetypes"],
            f"{key_added}_feat_specificity_upper": result["upper_significance"],
            f"{key_added}_feat_specificity_lower": result["lower_significance"],
        },
    )

    if not inplace:
        return adata
    return None


__all__ = [
    "_cluster_names_for_specificity_labels",
    "_encode_labels_for_specificity",
    "_run_specificity_backed_dense",
    "_run_specificity_backed_sparse",
    "_validate_archetype_membership",
    "compute_archetype_feature_specificity",
    "compute_feature_specificity",
]
