"""Marker detection and annotation functions."""

import warnings
from typing import Optional, Union, Literal, Dict, List
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata
from scipy.sparse import issparse, csr_matrix, csc_matrix

from .specificity import (
    _cluster_names_for_specificity_labels,
    compute_archetype_feature_specificity,
    compute_feature_specificity,
)
from ..io.lazy_transform import LazyTransform, _validate_lazy_transform
from ..io.operator import open_backed_operator_for
from .. import _core
from ..io.matrix_source import MatrixSource
from ..tools.anndata import as_plain_labels


def _graph_label_enrichment(G, enrichment_arr: np.ndarray, n_threads: int) -> np.ndarray:
    """Row-normalize ``G`` and return per-cell log p-values from positive marker stats.

    Encapsulates the identical three-line block used by both the enrichment
    and no-enrichment branches in ``annotate_cells`` to compute label log
    p-values from a graph and cell x celltype marker-stat matrix.
    """
    Gn = _core.normalize_graph(G, norm_method=1).T
    marker_stats_pos = np.maximum(enrichment_arr, 0)
    return _core.compute_graph_label_enrichment(Gn, marker_stats_pos, n_threads)


def _sparse_row_sum_sq(S) -> np.ndarray:
    """Compute sum of squared values per row without materializing S.power(2).

    Returns a float64 1-D array of length ``S.shape[0]``.

    For CSR: bincount over row indices derived from indptr, weighted by
    squared data values.  Handles empty rows and trailing-empty-row edge
    cases correctly.
    For CSC: bincount on row indices weighted by squared values.
    Other formats are converted to CSR first.
    """
    n_rows = S.shape[0]
    if isinstance(S, csr_matrix):
        data_sq = S.data.astype(np.float64, copy=False) ** 2
        if len(data_sq) == 0:
            return np.zeros(n_rows, dtype=np.float64)
        row_indices = np.repeat(np.arange(n_rows), np.diff(S.indptr))
        return np.bincount(row_indices, weights=data_sq, minlength=n_rows).astype(
            np.float64, copy=False
        )
    if isinstance(S, csc_matrix):
        data_sq = S.data.astype(np.float64, copy=False) ** 2
        if len(data_sq) == 0:
            return np.zeros(n_rows, dtype=np.float64)
        return np.bincount(S.indices, weights=data_sq, minlength=n_rows).astype(
            np.float64, copy=False
        )
    return _sparse_row_sum_sq(csr_matrix(S))


def find_markers(
    adata: AnnData,
    labels: Union[str, np.ndarray, pd.Series],
    labels_use: Optional[List[str]] = None,
    top_genes: Optional[int] = 50,
    features_use: Optional[str] = None,
    features_keep: Optional[Union[str, List[str], np.ndarray, pd.Series]] = None,
    layer: Optional[str] = None,
    n_threads: int = 0,
    result: Literal["table", "ranks", "scores"] = "table",
    return_type: Literal["dataframe", "dict"] = "dataframe",
    backed_chunk_size: int = 8192,
    lazy_transform: Optional[LazyTransform] = None,
) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Find marker genes for each cluster/group.

    This function identifies marker genes by computing feature specificity scores
    for each cluster and returning the top genes, their ranks, or raw scores.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels : str or np.ndarray
        Either a key in adata.obs containing cluster labels, or an array of labels.
    labels_use : list of str, optional
        Subset of labels to use. If None, uses all labels.
    top_genes : int, optional (default: 50)
        Number of top genes to return per cluster. Only used when result="table".
        If None, returns all genes.
    features_use : str, optional
        Column name in adata.var to extract feature labels from.
        If None (default), uses adata.var_names.
    features_keep : str or list of str, optional
        Additional whitelist filtering applied to the features_use set. Can be:
        - None: no additional filtering
        - str: column name in adata.var containing boolean/categorical values
        - list: explicit list of feature labels to keep
    layer : str, optional
        Layer in AnnData to use for computation. If None, uses adata.X.
    n_threads : int, optional (default: 0)
        Number of threads for computation. 0 means auto.
    result : {"table", "ranks", "scores"}, optional (default: "table")
        Type of result to return:
        - "table": Returns top marker gene names for each cluster
        - "ranks": Returns rank of each gene within each cluster (1=best)
        - "scores": Returns raw specificity scores
    return_type : {"dataframe", "dict"}, optional (default: "dataframe")
        Return format:
        - "dataframe": pandas DataFrame
        - "dict": Dictionary with cluster names as keys
    backed_chunk_size : int, optional (default: 8192)
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        When provided, the backed operator applies per-row normalization
        and log1p on-the-fly without requiring a persisted ``logcounts``
        layer.  Only valid when ``layer=None`` and the input is backed.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.

    Returns
    -------
    pd.DataFrame or dict
        Marker genes, ranks, or scores for each cluster.
        - If result="table" and return_type="dataframe": DataFrame with top genes as rows
        - If result="table" and return_type="dict": Dict with cluster names as keys, gene lists as values
        - If result="ranks" or "scores": features × clusters matrix

    Examples
    --------
    >>> # Get top 50 marker genes per cluster
    >>> markers = find_markers(adata, "clusters")

    >>> # Get ranks for all genes
    >>> ranks = find_markers(adata, "clusters", result="ranks", top_genes=None)

    >>> # Get as dictionary
    >>> markers_dict = find_markers(adata, "clusters", return_type="dict")
    """
    # Extract labels
    if isinstance(labels, str):
        if labels not in adata.obs:
            raise ValueError(f"Labels '{labels}' not found in adata.obs.")
        labels_arr = adata.obs[labels].values
    else:
        labels_arr = np.asarray(labels)

    # Normalize to a plain object array so that every downstream Categorical()
    # call produces the same (lexicographic) category order, regardless of
    # whether the original column was a pandas Categorical with a custom order.
    labels_arr = as_plain_labels(labels_arr)

    # Mask excluded observations in the label vector instead of subsetting
    # the AnnData.  The C++ specificity backend treats label=0 as
    # "unassigned" so masked-out cells contribute zero weight.
    if labels_use is not None:
        mask = np.isin(labels_arr, labels_use)
        labels_for_spec = labels_arr.copy()
        from pandas.api.types import is_integer_dtype
        if is_integer_dtype(labels_for_spec):
            labels_for_spec[~mask] = -1
        else:
            labels_for_spec = labels_for_spec.astype(object)
            labels_for_spec[~mask] = np.nan
        cluster_names = _cluster_names_for_specificity_labels(labels_arr[mask])
    else:
        labels_for_spec = labels_arr
        cluster_names = _cluster_names_for_specificity_labels(labels_arr)

    from .._feature_lookup import resolve_feature_space

    feature_labels = resolve_feature_space(adata, features_use).labels

    raw = compute_feature_specificity(
        adata,
        labels_for_spec,
        layer=layer,
        n_threads=n_threads,
        backed_chunk_size=backed_chunk_size,
        return_raw=True,
        lazy_transform=lazy_transform,
    )
    upper_sig = raw["upper_significance"]
    lower_sig = raw["lower_significance"]

    # Compute feature specificity scores
    feat_spec = upper_sig - lower_sig
    feat_spec[feat_spec < 0] = 0

    # Handle features_keep parameter (whitelist filtering)
    if features_keep is not None:
        if isinstance(features_keep, str):
            if features_keep in adata.var.columns:
                keep_values = adata.var[features_keep].to_numpy()
                if keep_values.dtype == bool:
                    keep_mask = keep_values
                else:
                    keep_mask = np.isin(feature_labels, keep_values)
            else:
                keep_mask = np.isin(feature_labels, [features_keep])
        else:
            keep_values = np.asarray(features_keep)
            if keep_values.dtype == bool:
                if len(keep_values) != len(feature_labels):
                    raise ValueError("Length of features_keep mask must match number of features")
                keep_mask = keep_values
            else:
                keep_mask = np.isin(feature_labels, keep_values)

        feat_spec = feat_spec[keep_mask, :]
        feature_labels = feature_labels[keep_mask]

    # Generate output based on result type
    if result == "table":
        # Get top genes for each cluster
        out_dict = {}
        for i, cluster in enumerate(cluster_names):
            scores = feat_spec[:, i]
            # Sort in descending order
            sorted_indices = np.argsort(scores)[::-1]
            if top_genes is not None:
                sorted_indices = sorted_indices[:top_genes]
            out_dict[cluster] = feature_labels[sorted_indices]

        if return_type == "dataframe":
            # Create DataFrame with equal-length columns
            max_len = max(len(v) for v in out_dict.values()) if out_dict else 0
            df_dict = {}
            for key, genes in out_dict.items():
                padded = list(genes) + [None] * (max_len - len(genes))
                df_dict[key] = padded
            return pd.DataFrame(df_dict)
        else:
            return out_dict

    elif result == "ranks":
        # Compute ranks for each cluster (higher score = lower rank number)
        ranks = np.zeros_like(feat_spec)
        for i in range(feat_spec.shape[1]):
            # rankdata with method='max' for ties, negate to get descending ranks
            ranks[:, i] = rankdata(-feat_spec[:, i], method='max')

        if return_type == "dataframe":
            df = pd.DataFrame(ranks, index=feature_labels, columns=cluster_names)
            return df
        else:
            out_dict = {}
            for i, cluster in enumerate(cluster_names):
                out_dict[cluster] = pd.Series(ranks[:, i], index=feature_labels)
            return out_dict

    elif result == "scores":
        # Return raw specificity scores
        if return_type == "dataframe":
            df = pd.DataFrame(feat_spec, index=feature_labels, columns=cluster_names)
            return df
        else:
            out_dict = {}
            for i, cluster in enumerate(cluster_names):
                out_dict[cluster] = pd.Series(feat_spec[:, i], index=feature_labels)
            return out_dict

    else:
        raise ValueError(f"Invalid result type: {result}. Must be 'table', 'ranks', or 'scores'.")


def annotate_cells(
    adata: AnnData,
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    method: Literal["vision", "actionet"] = "vision",
    features_use: Optional[str] = None,
    layer: Optional[str] = None,
    network_key: str = "actionet",
    norm_method: Literal["pagerank", "pagerank_sym"] = "pagerank",
    alpha: float = 0.85,
    max_it: int = 5,
    approx: bool = True,
    ignore_baseline: bool = False,
    use_enrichment: bool = True,
    use_lpa: bool = False,
    return_log_pvals: bool = False,
    n_threads: int = 0,
    backed_chunk_size: int = 8192,
    lazy_transform: Optional[LazyTransform] = None,
) -> Dict[str, np.ndarray]:
    """
    Infer cell annotations from imputed gene expression for all cells.

    This function takes marker genes, encodes them into a binary matrix, and uses
    graph-based gene expression imputation to compute enrichment scores for each
    cell type. It then assigns labels based on the highest enrichment.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers : dict, DataFrame, or ndarray
        Marker genes specification. Can be:
        - dict: Keys are cell types, values are lists of marker genes (with optional +/- suffix)
        - DataFrame: Wide format with columns as cell types, values as gene names (as returned by find_markers)
        - ndarray: Binary/weighted matrix (features × cell types)
    method : {"vision", "actionet"}, optional (default: "vision")
        Method for computing feature statistics.
    features_use : str, optional
        Column name in adata.var containing feature labels matching markers.
        If None, uses adata.var_names.
    layer : str, optional
        Layer to use for expression data. If None, uses adata.X.
    network_key : str, optional (default: "actionet")
        Key in adata.obsp containing the cell-cell network graph.
    norm_method : {"pagerank", "pagerank_sym"}, optional (default: "pagerank")
        Graph normalization method.
    alpha : float, optional (default: 0.85)
        Random-walk parameter for gene imputation (damping factor).
    max_it : int, optional (default: 5)
        Maximum iterations for imputation.
    approx : bool, optional (default: True)
        Use approximate computation.
    ignore_baseline : bool, optional (default: False)
        Ignore baseline in actionet method.
    use_enrichment : bool, optional (default: True)
        Use graph-based label enrichment for final assignment.
    use_lpa : bool, optional (default: False)
        Apply label propagation algorithm to correct labels.
    return_log_pvals : bool, optional (default: False)
        If True, include the raw graph-enriched log p-value matrix in the
        result under the key ``"log_pvals"``.
    n_threads : int, optional (default: 0)
        Number of threads (0 = auto).
    backed_chunk_size : int, optional (default: 8192)
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        Only applied when ``method="vision"`` and the input is backed.
        When provided, the backed operator applies per-row normalization
        and log1p on-the-fly without requiring a persisted ``logcounts``
        layer.  Only valid when ``layer=None``.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.

    Returns
    -------
    dict
        Dictionary with keys:
        - "labels": Inferred cell type labels (array of length n_cells)
        - "confidence": Confidence scores for labels (array of length n_cells)
        - "enrichment": Cell type score DataFrame (n_cells × n_celltypes), indexed by
          ``adata.obs_names`` with columns named by cell type
        - "labels_corrected": (optional) LPA-corrected labels if use_lpa=True
        - "log_pvals": (optional) Graph-enriched log p-value DataFrame (n_cells × n_celltypes),
          same shape and index/columns as ``"enrichment"``, present only when
          return_log_pvals=True

    Examples
    --------
    >>> # Using marker dictionary
    >>> markers = {
    ...     "T cells": ["CD3D", "CD3E", "CD3G"],
    ...     "B cells": ["CD19", "MS4A1", "CD79A"],
    ...     "Monocytes": ["CD14", "FCGR3A"]
    ... }
    >>> result = annotate_cells(adata, markers)
    >>> adata.obs["celltype"] = result["labels"]
    >>> adata.obs["celltype_confidence"] = result["confidence"]

    >>> # Using signed markers (+ for positive, - for negative)
    >>> markers = {
    ...     "CD4+ T": ["CD3D+", "CD4+", "CD8A-"],
    ...     "CD8+ T": ["CD3D+", "CD4-", "CD8A+"]
    ... }
    >>> result = annotate_cells(adata, markers)
    """

    # This function is much more verbose than the R version and generally disgusting
    # because pybind cannot pass matrices by reference and have them be modified
    # in memory like with Rcpp. Objects must be translated and passed by copy.
    # Low cost operations that can be done by libactionet are done here because
    # the cost to compute is cheaper than the cost to translate and pass by copy.

    # Get feature labels
    from .._feature_lookup import resolve_feature_space
    space = resolve_feature_space(adata, features_use, context="annotate_cells")
    feature_set = space.labels

    # Encode markers into binary/weighted matrix (sparse CSR, full gene-width)
    X_markers, celltype_names = _encode_markers(markers, feature_set)

    source = MatrixSource(adata, layer=layer)
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    # Get network graph
    from ..tools.anndata import norm_method_to_int, resolve_network

    G = resolve_network(adata, network_key)

    if not issparse(G):
        G = csr_matrix(G)

    norm_method_code = norm_method_to_int(norm_method)

    # Ensure X_markers is sparse
    if not issparse(X_markers):
        X_markers = csr_matrix(X_markers)

    n_vars = source.n_vars

    # Compute marker statistics using graph-based imputation.
    # Each method+storage combination is a self-contained block.
    # Backed paths are read-only: only MatrixSource streaming and the
    # C++ backed operator (both read-only) are used — no AnnData mutation.
    if method == "vision":
        if source.is_backed:
            # ----------------------------------------------------------
            # Vision · backed
            # ----------------------------------------------------------
            # Use the C++ backed operator for stats = S @ X (inherits
            # lazy_transform) and row_sums, then compute row_sum_sq via
            # MatrixSource streaming.  Pass pre-computed arrays to the
            # new split binding — avoids copying S across the pybind boundary.
            with open_backed_operator_for(
                adata,
                layer=layer,
                context="annotate_cells",
                chunk_size=backed_chunk_size,
                lazy_transform=lazy_transform,
                source=source,
            ) as op:
                if use_enrichment:
                    fused = _core.annotate_cells_vision_backed_fused(
                        op=op,
                        G=G,
                        X=X_markers,
                        norm_method=norm_method_code,
                        alpha=alpha,
                        max_it=max_it,
                        approx=approx,
                        enrichment_norm_method=1,
                        thread_no=n_threads,
                    )
                    marker_stats = fused["marker_stats"]
                    _fused_log_pvals = fused["log_pvals"]
                else:
                    marker_stats = _core.compute_feature_stats_vision_backed_operator(
                        op=op,
                        G=G,
                        X=X_markers,
                        norm_method=norm_method_code,
                        alpha=alpha,
                        max_it=max_it,
                        approx=approx,
                        thread_no=n_threads,
                    )
                    _fused_log_pvals = None
        else:
            # ----------------------------------------------------------
            # Vision · in-memory
            # ----------------------------------------------------------
            # Compute S @ X, mu, sigma_sq entirely in scipy/numpy so the
            # full expression matrix never crosses the pybind boundary.
            S = source.matrix  # reference — no copy

            if issparse(S):
                stats = S.dot(X_markers)
                if issparse(stats):
                    stats = stats.toarray()
                stats = np.asarray(stats, dtype=np.float64)

                row_sums = np.asarray(S.sum(axis=1), dtype=np.float64).ravel()
                row_sum_sq = _sparse_row_sum_sq(S)
            else:
                S_arr = np.asarray(S, dtype=np.float64)
                X_dense = X_markers.toarray() if issparse(X_markers) else np.asarray(X_markers)
                stats = S_arr @ X_dense
                stats = np.asarray(stats, dtype=np.float64)

                row_sums = S_arr.sum(axis=1)
                row_sum_sq = np.sum(S_arr * S_arr, axis=1)

            mu = row_sums / n_vars
            sigma_sq = (row_sum_sq - 2.0 * mu * row_sums
                        + n_vars * mu ** 2) / (n_vars - 1)

            if use_enrichment:
                fused = _core.annotate_cells_vision_fused(
                    G=G,
                    stats=stats,
                    mu=mu,
                    sigma_sq=sigma_sq,
                    X=X_markers,
                    norm_method=norm_method_code,
                    alpha=alpha,
                    max_it=max_it,
                    approx=approx,
                    enrichment_norm_method=1,
                    thread_no=n_threads,
                )
                marker_stats = fused["marker_stats"]
                _fused_log_pvals = fused["log_pvals"]
            else:
                marker_stats = _core.compute_feature_stats_vision_from_stats(
                    G=G,
                    stats=stats,
                    mu=mu,
                    sigma_sq=sigma_sq,
                    X=X_markers,
                    norm_method=norm_method_code,
                    alpha=alpha,
                    max_it=max_it,
                    approx=approx,
                    thread_no=n_threads,
                )
                _fused_log_pvals = None

    elif method == "actionet":
        # ----------------------------------------------------------
        # ACTIONet · both storage modes
        # ----------------------------------------------------------
        # Extract only the marker columns from S (typically 50-200
        # out of ~30k genes) to avoid copying the full matrix.
        required_idx = np.where(np.asarray(X_markers.getnnz(axis=1)).ravel() > 0)[0]

        if required_idx.size == 0:
            raise ValueError("Marker set does not overlap features in AnnData.")

        X_sub = X_markers[required_idx, :]

        if source.is_backed:
            S_sub = source.feature_subset(
                required_idx,
                chunk_size=backed_chunk_size,
                prefer_sparse=True,
            )
        else:
            S_sub = source.matrix[:, required_idx]

        if not issparse(S_sub):
            S_sub = csr_matrix(np.asarray(S_sub))

        if use_enrichment:
            fused = _core.annotate_cells_actionet_fused(
                G=G,
                S=S_sub,
                X=X_sub,
                norm_method=norm_method_code,
                alpha=alpha,
                max_it=max_it,
                approx=approx,
                enrichment_norm_method=1,
                thread_no=n_threads,
                ignore_baseline=ignore_baseline,
            )
            marker_stats = fused["marker_stats"]
            _fused_log_pvals = fused["log_pvals"]
        else:
            marker_stats = _core.compute_feature_stats(
                G=G,
                S=S_sub,
                X=X_sub,
                norm_method=norm_method_code,
                alpha=alpha,
                max_it=max_it,
                approx=approx,
                thread_no=n_threads,
                ignore_baseline=ignore_baseline,
            )
            _fused_log_pvals = None
    else:
        raise ValueError(f"Unknown method: {method}")

    # marker_stats is cells × celltypes
    enrichment_arr = np.nan_to_num(marker_stats, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute labels and confidence
    celltype_arr = np.asarray(celltype_names)
    if use_enrichment:
        if _fused_log_pvals is not None:
            log_pvals = _fused_log_pvals
        else:
            log_pvals = _graph_label_enrichment(G, enrichment_arr, n_threads)

        labels_idx = np.argmax(log_pvals, axis=1)
        confidence = np.max(log_pvals, axis=1)
    else:
        labels_idx = np.argmax(enrichment_arr, axis=1)
        confidence = np.max(enrichment_arr, axis=1)
        if return_log_pvals:
            log_pvals = _graph_label_enrichment(G, enrichment_arr, n_threads)

    labels = celltype_arr[labels_idx]

    enrichment = pd.DataFrame(
        enrichment_arr,
        index=adata.obs_names,
        columns=celltype_names,
    )

    result = {
        "labels": labels,
        "confidence": confidence,
        "enrichment": enrichment,
    }

    if return_log_pvals:
        result["log_pvals"] = pd.DataFrame(
            log_pvals,
            index=adata.obs_names,
            columns=celltype_names,
        )

    # Optional label propagation
    if use_lpa:
        unique_labels, numeric_labels = np.unique(labels, return_inverse=True)
        numeric_labels = numeric_labels.astype(np.float64)

        corrected_numeric = _core.run_lpa(
            G=G,
            labels=numeric_labels,
            lambda_param=1.0,
            iters=3,
            sig_threshold=3.0,
            fixed_labels=None,
            thread_no=n_threads,
        )

        result["labels_corrected"] = unique_labels[corrected_numeric.astype(np.intp)]

    return result


def _annotate_from_markers(
    *,
    adata: AnnData,
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    features_use: Optional[str],
    n_threads: int,
    spec_upper: np.ndarray,
    spec_lower: Optional[np.ndarray],
    row_names: np.ndarray,
    row_names_key: str,
    context: str,
) -> Dict[str, np.ndarray]:
    """Shared marker-mode enrichment pipeline.

    Used by both :func:`annotate_clusters` and the marker branch of
    :func:`annotate_archetypes`.  Given an ``upper``/``lower`` specificity
    pair (``lower`` may be ``None``), a marker specification, and the row
    names for the group axis, computes ``pmax(upper - lower, 0)``, encodes
    the markers to a sparse feature x celltype matrix, calls
    ``_core.assess_enrichment``, and returns the standard annotation dict.
    """
    spec = spec_upper - spec_lower if spec_lower is not None else spec_upper
    spec = np.asarray(spec)
    spec[spec < 0] = 0
    if issparse(spec):
        spec = spec.toarray()

    from .._feature_lookup import resolve_feature_space

    space = resolve_feature_space(adata, features_use, context=context)
    marker_mat, celltype_names = _encode_markers(markers, space.labels)
    if not issparse(marker_mat):
        marker_mat = csr_matrix(marker_mat)

    log_pvals = _core.assess_enrichment(spec, marker_mat, n_threads)["logPvals"].T
    log_pvals = np.nan_to_num(log_pvals, nan=0.0, posinf=0.0, neginf=0.0)

    labels_idx = np.argmax(log_pvals, axis=1)
    return {
        "labels": np.array([celltype_names[i] for i in labels_idx]),
        "confidence": np.max(log_pvals, axis=1),
        "enrichment": log_pvals,
        row_names_key: row_names,
    }


def annotate_clusters(
    adata: AnnData,
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    cluster_key: str = None,
    specificity_key: Optional[str] = None,
    features_use: Optional[str] = None,
    layer: Optional[str] = None,
    n_threads: int = 0,
    backed_chunk_size: int = 8192,
    lazy_transform: Optional[LazyTransform] = None,
) -> Dict[str, np.ndarray]:
    """
    Annotate clusters using known marker genes.

    This function assigns cell type labels to clusters by computing enrichment
    of marker gene sets within cluster-specific gene expression profiles. If
    a pre-computed feature specificity result is provided, it will be used
    instead of computing it from the cluster labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers : dict, DataFrame, or ndarray
        Marker genes specification. Can be:
        - dict: Keys are cell types, values are lists of marker genes
        - DataFrame: Wide format with columns as cell types, values as gene names
        - ndarray: Binary/weighted matrix (features × cell types)
    cluster_key : str, optional (default: None)
        Key in adata.obs containing cluster labels. Also used to extract cluster assignments
        for computing feature specificity (if not pre-computed).
    specificity_key : str, optional (default: None)
        Base key in adata.varm for pre-computed feature specificity matrices from `compute_feature_specificity()`.
        If None, computes feature specificity de novo from cluster_key.
    features_use : str, optional
        Column name in adata.var containing feature labels matching markers.
        If None, uses adata.var_names.
    layer : str, optional
        Layer to use for expression data when computing feature specificity.
        If None, uses adata.X. Only used if feature specificity needs to be computed.
    n_threads : int, optional (default: 0)
        Number of threads (0 = auto).
    backed_chunk_size : int, optional (default: 8192)
        Number of rows per chunk when streaming backed AnnData.
        Only used if feature specificity needs to be computed.
    lazy_transform : LazyTransform, optional
        Pre-built lazy logcount transform for backed AnnData inputs.
        When provided, the backed operator applies per-row normalization
        and log1p on-the-fly without requiring a persisted ``logcounts``
        layer.  Only valid when ``layer=None`` and the input is backed.
        Only used when computing feature specificity de novo
        (``specificity_key=None``); ignored with a warning otherwise.
        Create with :func:`~actionet.lazy_transform.create_lazy_transform`.

    Returns
    -------
    dict
        Dictionary with keys:
        - "labels": Inferred cell type labels for each cluster (array of length n_clusters)
        - "confidence": Confidence scores for labels (array of length n_clusters)
        - "enrichment": Enrichment score matrix (n_clusters × n_celltypes)
        - "cluster_names": Cluster names in the same order as labels/confidence/enrichment rows.
          For string clusters: lexicographically sorted unique cluster values.
          For integer clusters with sparse values (e.g., [1,2,5,8]): array indices [0,1,2,3,4,5,6,7,8]
          matching the C++ backend's sparse representation.

    Examples
    --------
    >>> # Example 1: Automatic computation (default, specificity_key=None)
    >>> markers = {
    ...     "T cells": ["CD3D", "CD3E", "CD3G"],
    ...     "B cells": ["CD19", "MS4A1", "CD79A"],
    ...     "Monocytes": ["CD14", "FCGR3A"]
    ... }
    >>> result = annotate_clusters(adata, markers, cluster_key="leiden")
    >>> # Feature specificity computed de novo from adata.obs["leiden"]

    >>> # Example 2: Using pre-computed specificity
    >>> # First compute and store specificity
    >>> from actionet import compute_feature_specificity
    >>> compute_feature_specificity(adata, "leiden", key_added="leiden_spec")
    >>> # Now use it for annotation
    >>> result = annotate_clusters(adata, markers, specificity_key="leiden_spec")
    >>> # Uses adata.varm["leiden_spec_upper"] and adata.varm["leiden_spec_lower"]

    >>> # Map annotations to cells
    >>> cluster_to_annotation = dict(zip(result["cluster_names"], result["labels"]))
    >>> adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_annotation)

    >>> # Create enrichment DataFrame
    >>> import pandas as pd
    >>> enrichment_df = pd.DataFrame(
    ...     result["enrichment"],
    ...     index=result["cluster_names"],
    ...     columns=list(markers.keys())
    ... )
    """
    # Check if we have pre-computed specificity or need to compute it
    if specificity_key is not None:
        if lazy_transform is not None:
            warnings.warn(
                "`lazy_transform` is ignored when `specificity_key` is provided "
                "(feature specificity is read from adata.varm, not computed).",
                UserWarning,
                stacklevel=2,
            )

        # Use pre-computed feature specificity
        upper_key = f"{specificity_key}_upper"
        lower_key = f"{specificity_key}_lower"

        if upper_key not in adata.varm or lower_key not in adata.varm:
            raise ValueError(
                f"Pre-computed specificity not found. Expected '{upper_key}' and '{lower_key}' "
                f"in adata.varm. Available keys: {list(adata.varm.keys())}"
            )

        upper_sig = adata.varm[upper_key]
        lower_sig = adata.varm[lower_key]

        # For pre-computed specificity, we need cluster labels to determine cluster names
        if cluster_key not in adata.obs.columns:
            raise ValueError(
                f"Cluster key '{cluster_key}' not found in adata.obs. "
                f"Needed to determine cluster names for pre-computed specificity."
            )
        cluster_labels = adata.obs[cluster_key].values
    else:
        # Compute feature specificity de novo
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        cluster_labels = adata.obs[cluster_key].values

        source = MatrixSource(adata, layer=layer)
        _validate_lazy_transform(lazy_transform, layer=layer, source=source)

        # Compute feature specificity on the fly using return_raw to avoid expensive AnnData copy
        result = compute_feature_specificity(
            adata,
            cluster_labels,
            layer=layer,
            n_threads=n_threads,
            backed_chunk_size=backed_chunk_size,
            return_raw=True,
            lazy_transform=lazy_transform,
        )
        upper_sig = result["upper_significance"]
        lower_sig = result["lower_significance"]

    # Normalize cluster labels to plain array to ensure consistent ordering.
    cluster_labels = as_plain_labels(cluster_labels)

    # Match the same label ordering logic used by compute_feature_specificity.
    n_clusters = np.asarray(upper_sig).shape[1]
    cluster_names = _cluster_names_for_specificity_labels(cluster_labels)
    if cluster_names.shape[0] != n_clusters:
        from pandas.api.types import is_integer_dtype

        if is_integer_dtype(cluster_labels):
            # Backward-compat fallback for precomputed matrices generated before
            # integer-label compaction (legacy behavior used 0..max(label)).
            cluster_names = np.arange(n_clusters)
        else:
            raise ValueError(
                "Cluster label cardinality does not match specificity matrix columns "
                f"({cluster_names.shape[0]} labels vs {n_clusters} columns)."
            )

    return _annotate_from_markers(
        adata=adata,
        markers=markers,
        features_use=features_use,
        n_threads=n_threads,
        spec_upper=upper_sig,
        spec_lower=lower_sig,
        row_names=cluster_names,
        row_names_key="cluster_names",
        context="annotate_clusters",
    )


def annotate_archetypes(
    adata: AnnData,
    markers: Optional[Union[Dict[str, List[str]], pd.DataFrame, np.ndarray]] = None,
    labels: Optional[Union[str, np.ndarray]] = None,
    scores: Optional[Union[str, np.ndarray]] = None,
    archetype_slot: str = "H_merged",
    specificity_key: Optional[str] = None,
    features_use: Optional[str] = None,
    layer: Optional[str] = None,
    n_threads: int = 0,
    backed_chunk_size: int = 8192,
    lazy_transform: Optional[LazyTransform] = None,
) -> Dict[str, np.ndarray]:
    """
    Annotate archetypes using marker genes, prior annotations, or a score matrix.

    This is the archetype-level counterpart of :func:`annotate_clusters`.  Where
    ``annotate_clusters`` operates on discrete cluster labels, this function
    operates on the continuous cell-by-archetype soft-membership matrix stored
    in ``adata.obsm[archetype_slot]`` (default ``"H_merged"``) — a simplex-
    constrained matrix of shape ``n_cells x n_archetypes`` produced by
    :func:`actionet.run_actionet` / :func:`actionet.merge_archetypes`.

    Exactly one of ``markers``, ``labels``, or ``scores`` must be provided.

    - ``markers``: known marker genes per cell type.  Uses archetype feature
      specificity (``pmax(upper - lower, 0)``) together with the Bennett
      concentration inequality via ``_core.assess_enrichment``.
    - ``labels``: a per-cell annotation (either a column name in ``adata.obs``
      or a 1-D array).  Uses ``_core.xicor_matrix`` between the continuous
      archetype soft-membership matrix and the one-hot encoded labels.
    - ``scores``: a per-cell numeric score matrix (either a key in
      ``adata.obsm`` or a 2-D array).  Uses ``_core.xicor_matrix`` between
      the archetype soft-membership matrix and the score matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers, labels, scores
        Mutually-exclusive annotation inputs.  See summary above.
    archetype_slot : str, default ``"H_merged"``
        Key in ``adata.obsm`` for the archetype soft-membership matrix
        (labels/scores modes).
    specificity_key : str, optional
        Prefix used to look up pre-computed archetype feature specificity
        in ``adata.varm``.  When provided (marker mode), the function reads
        ``{specificity_key}_upper`` and, if present, ``{specificity_key}_lower``
        and forms ``pmax(upper - lower, 0)``.  When ``None`` (default),
        archetype feature specificity is computed on the fly via
        :func:`compute_archetype_feature_specificity`.
    features_use : str, optional
        Column in ``adata.var`` supplying feature labels for marker matching.
    layer : str, optional
        Layer used when computing archetype feature specificity de novo.
    n_threads : int, default 0
        Number of parallel threads.  ``0`` lets the backend choose.
    backed_chunk_size : int, default 8192
        Backed streaming chunk size for de-novo specificity.
    lazy_transform : LazyTransform, optional
        Optional lazy transform for backed inputs (used only when computing
        specificity de novo; ignored with a warning otherwise).

    Returns
    -------
    dict
        Dictionary with keys:
        - ``"labels"``: Inferred archetype labels (array of length
          ``n_archetypes``).
        - ``"confidence"``: Confidence scores (array of length ``n_archetypes``).
        - ``"enrichment"``: ``n_archetypes x n_annotations`` enrichment matrix.
        - ``"archetype_names"``: Archetype names, in row order.

    Examples
    --------
    >>> # Marker mode with de-novo archetype specificity:
    >>> markers = {"T": ["CD3D"], "B": ["CD19"]}
    >>> res = annotate_archetypes(adata, markers=markers)
    >>> # Labels mode against an existing per-cell annotation:
    >>> res = annotate_archetypes(adata, labels="cell_type")
    """
    supplied = [x is not None for x in (markers, labels, scores)]
    if sum(supplied) != 1:
        raise ValueError(
            "Exactly one of `markers`, `labels`, or `scores` must be provided."
        )

    if markers is not None:
        # -------- Marker branch --------
        if specificity_key is not None:
            if lazy_transform is not None:
                warnings.warn(
                    "`lazy_transform` is ignored when `specificity_key` is provided.",
                    UserWarning,
                    stacklevel=2,
                )
            upper_key = f"{specificity_key}_upper"
            lower_key = f"{specificity_key}_lower"
            if upper_key not in adata.varm:
                raise ValueError(
                    f"Pre-computed archetype specificity not found. Expected "
                    f"'{upper_key}' in adata.varm. Available: {list(adata.varm.keys())}"
                )
            upper_sig = np.asarray(adata.varm[upper_key])
            if lower_key in adata.varm:
                lower_sig = np.asarray(adata.varm[lower_key])
            else:
                lower_sig = None
        else:
            spec_result = compute_archetype_feature_specificity(
                adata,
                layer=layer,
                n_threads=n_threads,
                backed_chunk_size=backed_chunk_size,
                return_raw=True,
                lazy_transform=lazy_transform,
            )
            upper_sig = spec_result["upper_significance"]
            lower_sig = spec_result.get("lower_significance")

        n_archetypes = np.asarray(upper_sig).shape[1]
        archetype_names = np.array(
            [f"A{i + 1}" for i in range(n_archetypes)], dtype=object
        )

        return _annotate_from_markers(
            adata=adata,
            markers=markers,
            features_use=features_use,
            n_threads=n_threads,
            spec_upper=upper_sig,
            spec_lower=lower_sig,
            row_names=archetype_names,
            row_names_key="archetype_names",
            context="annotate_archetypes",
        )

    # -------- Labels / scores branches: continuous-H XICOR --------
    if archetype_slot not in adata.obsm:
        raise ValueError(
            f"Archetype slot '{archetype_slot}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    X1 = np.asarray(adata.obsm[archetype_slot], dtype=np.float64)
    if X1.ndim != 2:
        raise ValueError(
            f"adata.obsm['{archetype_slot}'] must be 2-D (cells x archetypes); "
            f"got shape {X1.shape}"
        )
    n_archetypes = X1.shape[1]
    archetype_names = np.array(
        [f"A{i + 1}" for i in range(n_archetypes)], dtype=object
    )

    if labels is not None:
        if isinstance(labels, str):
            if labels not in adata.obs.columns:
                raise ValueError(f"Labels key '{labels}' not found in adata.obs.")
            label_vec = adata.obs[labels].values
        else:
            label_vec = np.asarray(labels)
        label_vec = as_plain_labels(label_vec)
        categories, inverse = np.unique(label_vec, return_inverse=True)
        n_cat = len(categories)
        X2 = np.zeros((label_vec.shape[0], n_cat), dtype=np.float64)
        X2[np.arange(label_vec.shape[0]), inverse] = 1.0
        col_names = np.asarray(categories, dtype=object)
    else:
        if isinstance(scores, str):
            if scores not in adata.obsm:
                raise ValueError(f"Scores slot '{scores}' not found in adata.obsm.")
            X2 = np.asarray(adata.obsm[scores], dtype=np.float64)
            col_names = np.array(
                [f"S{i + 1}" for i in range(X2.shape[1])], dtype=object
            )
        else:
            X2 = np.asarray(scores, dtype=np.float64)
            if X2.ndim == 1:
                X2 = X2[:, None]
            col_names = np.array(
                [f"S{i + 1}" for i in range(X2.shape[1])], dtype=object
            )

    if X1.shape[0] != X2.shape[0]:
        raise ValueError(
            f"Row count mismatch: archetype matrix has {X1.shape[0]} rows, "
            f"annotation matrix has {X2.shape[0]}."
        )

    xi_out = _core.xicor_matrix(X1, X2, True, 0, n_threads)
    z_pos = np.asarray(xi_out["Z"], dtype=np.float64)
    z_pos[z_pos < 0] = 0.0
    # Restore signed direction via classical Pearson correlation.
    corr = np.zeros_like(z_pos)
    # Vectorized column-wise Pearson correlation.
    X1c = X1 - X1.mean(axis=0, keepdims=True)
    X2c = X2 - X2.mean(axis=0, keepdims=True)
    X1_std = np.sqrt((X1c ** 2).sum(axis=0))
    X2_std = np.sqrt((X2c ** 2).sum(axis=0))
    denom = np.outer(X1_std, X2_std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (X1c.T @ X2c) / denom
    direction = np.sign(np.nan_to_num(corr, nan=0.0))
    archetype_enrichment = direction * z_pos
    archetype_enrichment = np.nan_to_num(
        archetype_enrichment, nan=0.0, posinf=0.0, neginf=0.0
    )

    labels_idx = np.argmax(archetype_enrichment, axis=1)
    confidence = np.max(archetype_enrichment, axis=1)
    annot_labels = np.array([col_names[i] for i in labels_idx])

    return {
        "labels": annot_labels,
        "confidence": confidence,
        "enrichment": archetype_enrichment,
        "archetype_names": archetype_names,
    }


def _encode_markers(
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    feature_set: np.ndarray,
) -> tuple[csr_matrix, List[str]]:
    """
    Encode marker genes into a sparse binary feature x celltype matrix.

    Uses first-match semantics: when *feature_set* contains duplicate labels,
    only the first occurrence of each label is marked (matching R behaviour).

    Parameters
    ----------
    markers : dict, DataFrame, or ndarray
        Marker specification:
        - dict: keys are labels, values are lists of feature names
        - DataFrame: columns are labels, values are feature names
        - ndarray: numeric matrix (features x labels)
    feature_set : ndarray
        Array of feature names (length n_features).

    Returns
    -------
    tuple
        (X, label_names) where X is a sparse CSR binary matrix of shape
        (n_features, n_labels) and label_names is a list of label names.
    """
    n_features = len(feature_set)

    if isinstance(markers, np.ndarray):
        X = np.asarray(markers)
        if not np.isfinite(X).all():
            raise ValueError("'markers' contains non-numeric values")
        if X.ndim != 2:
            raise ValueError("'markers' must be a 2D array")
        if X.shape[0] != n_features:
            raise ValueError("Number of rows in 'markers' does not match number of features")
        X = csr_matrix((X != 0).astype(np.float32))
        label_names = [f"Label_{i}" for i in range(X.shape[1])]
    elif isinstance(markers, (pd.DataFrame, dict)):
        from .._feature_lookup import build_first_occurrence_lookup

        lookup = build_first_occurrence_lookup(feature_set)

        if isinstance(markers, pd.DataFrame):
            if markers.columns is None or markers.columns.isnull().any():
                raise ValueError("'markers' contains unnamed entries")
            if markers.columns.duplicated().any():
                raise ValueError("'markers' contains duplicated labels")
            label_names = markers.columns.tolist()
            marker_lists = [
                markers[col].dropna().astype(str).tolist() for col in label_names
            ]
        else:
            label_names = list(markers.keys())
            if any(name is None for name in label_names):
                raise ValueError("'markers' contains unnamed entries")
            if len(set(label_names)) != len(label_names):
                raise ValueError("'markers' contains duplicated labels")
            marker_lists = []
            for name in label_names:
                vals = markers[name]
                if isinstance(vals, (list, tuple, np.ndarray, pd.Series)):
                    values = [str(v) for v in vals if v is not None]
                else:
                    values = [str(vals)] if vals is not None else []
                marker_lists.append(values)

        rows: list = []
        cols: list = []
        for j, gene_list in enumerate(marker_lists):
            seen_genes: set = set()
            for gene in gene_list:
                if gene in seen_genes:
                    continue
                seen_genes.add(gene)
                idx = lookup.get(gene)
                if idx is not None:
                    rows.append(idx)
                    cols.append(j)

        n_labels = len(label_names)
        if rows:
            data = np.ones(len(rows), dtype=np.float32)
            X = csr_matrix(
                (data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                shape=(n_features, n_labels),
            )
        else:
            X = csr_matrix((n_features, n_labels), dtype=np.float32)
    else:
        raise ValueError("'markers' must be one of: dict, DataFrame, or ndarray")

    if X.shape[1] == 0:
        raise ValueError("No markers provided")

    col_sums = np.asarray(X.sum(axis=0)).ravel()
    zero_cols = np.where(col_sums == 0)[0]
    if len(zero_cols) == X.shape[1]:
        raise ValueError("No markers in 'features_use'")
    if len(zero_cols) > 0:
        for idx in zero_cols:
            warnings.warn(
                f"Label '{label_names[idx]}' has no markers",
                UserWarning,
                stacklevel=2,
            )

    return X, label_names
