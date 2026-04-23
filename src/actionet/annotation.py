"""Marker detection and annotation functions."""

from typing import Optional, Union, Literal, Dict, List
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata
from scipy.sparse import issparse, csr_matrix, csc_matrix

from .specificity import (
    _cluster_names_for_specificity_labels,
    compute_feature_specificity,
)
from .lazy_transform import LazyTransform, _validate_lazy_transform, _resolve_lazy_backed_transform
from .backed_io import _backed_group_path, _open_backed_operator
from . import _core
from ._matrix_source import MatrixSource


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
    return_type: Literal["dataframe", "dict", ""] = "dataframe",
    backed_chunk_size: int = 4096,
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
    backed_chunk_size : int, optional (default: 4096)
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
    if hasattr(labels_arr, 'categories'):
        labels_arr = np.asarray(labels_arr)

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

    # Handle features_use parameter
    # Can be: None (use var_names) or str (column name in adata.var)
    if features_use is None:
        feature_labels = adata.var_names.values
    else:
        if features_use not in adata.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        feature_labels = adata.var[features_use].values

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
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
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
    method : {"vision", "actionet"}, optional (default: "actionet")
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
    n_threads : int, optional (default: 0)
        Number of threads (0 = auto).
    backed_chunk_size : int, optional (default: 4096)
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
        - "enrichment": Cell type score matrix (n_cells × n_celltypes)
        - "labels_corrected": (optional) LPA-corrected labels if use_lpa=True

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
    from ._feature_lookup import resolve_feature_space
    space = resolve_feature_space(adata, features_use, context="annotate_cells")
    feature_set = space.labels

    # Encode markers into binary/weighted matrix (sparse CSR, full gene-width)
    X_markers, celltype_names = _encode_markers(markers, feature_set)

    source = MatrixSource(adata, layer=layer)
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    # Get network graph
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found in adata.obsp")
    G = adata.obsp[network_key]

    if not issparse(G):
        G = csr_matrix(G)

    # Convert norm_method to integer code
    norm_method_code = 2 if norm_method == "pagerank_sym" else 0

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
            row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
                source,
                lazy_transform=lazy_transform,
                backed_chunk_size=backed_chunk_size,
            )
            file_path = str(adata.filename)
            group_path = _backed_group_path(layer)
            with _open_backed_operator(
                adata=adata,
                file_path=file_path,
                group_path=group_path,
                context="annotate_cells",
                chunk_size=backed_chunk_size,
                row_scale_factors=row_scale_factors,
                apply_log1p=apply_log1p,
                log_scale=log_scale,
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
    enrichment = np.nan_to_num(marker_stats, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute labels and confidence
    celltype_arr = np.asarray(celltype_names)
    if use_enrichment:
        if _fused_log_pvals is not None:
            log_pvals = _fused_log_pvals
        else:
            Gn = _core.normalize_graph(G, norm_method=1).T
            marker_stats_pos = np.maximum(enrichment, 0)
            log_pvals = _core.compute_graph_label_enrichment(Gn, marker_stats_pos, n_threads)

        labels_idx = np.argmax(log_pvals, axis=1)
        confidence = np.max(log_pvals, axis=1)
    else:
        labels_idx = np.argmax(enrichment, axis=1)
        confidence = np.max(enrichment, axis=1)

    labels = celltype_arr[labels_idx]

    result = {
        "labels": labels,
        "confidence": confidence,
        "enrichment": enrichment,
    }

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


def annotate_clusters(
    adata: AnnData,
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    cluster_key: str = None,
    specificity_key: Optional[str] = None,
    features_use: Optional[str] = None,
    layer: Optional[str] = None,
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
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
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk when streaming backed AnnData.
        Only used if feature specificity needs to be computed.

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
        cluster_feat_spec = upper_sig - lower_sig
        cluster_feat_spec[cluster_feat_spec < 0] = 0

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

        # Compute feature specificity on the fly using return_raw to avoid expensive AnnData copy
        result = compute_feature_specificity(
            adata,
            cluster_labels,
            layer=layer,
            n_threads=n_threads,
            backed_chunk_size=backed_chunk_size,
            return_raw=True,
        )
        # Combine upper and lower to get feature specificity
        upper_sig = result["upper_significance"]
        lower_sig = result["lower_significance"]
        cluster_feat_spec = upper_sig - lower_sig
        cluster_feat_spec[cluster_feat_spec < 0] = 0

    # Normalize cluster labels to plain array to ensure consistent ordering.
    if hasattr(cluster_labels, "categories"):
        cluster_labels = np.asarray(cluster_labels)

    # Match the same label ordering logic used by compute_feature_specificity.
    n_clusters = cluster_feat_spec.shape[1]
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

    # Get feature labels
    from ._feature_lookup import resolve_feature_space
    space = resolve_feature_space(adata, features_use, context="annotate_clusters")
    feature_set = space.labels

    # Encode markers into binary/weighted matrix
    marker_mat, celltype_names = _encode_markers(markers, feature_set)

    # Convert to dense if sparse
    if issparse(cluster_feat_spec):
        cluster_feat_spec = cluster_feat_spec.toarray()

    # Convert marker_mat to sparse for efficiency
    if not issparse(marker_mat):
        marker_mat = csr_matrix(marker_mat)

    # Compute enrichment using C++ backend
    # assess_enrichment expects (features × annotations) for both inputs
    # Returns dict with "logPvals" and "thresholds"
    enrichment_result = _core.assess_enrichment(
        cluster_feat_spec,  # features × clusters
        marker_mat,         # features × celltypes
        n_threads
    )

    # enrichment_result["logPvals"] is clusters × celltypes
    log_pvals = enrichment_result["logPvals"].T  # Transpose to clusters × celltypes

    # Handle non-finite values
    log_pvals = np.nan_to_num(log_pvals, nan=0.0, posinf=0.0, neginf=0.0)

    # Assign labels based on highest enrichment
    labels_idx = np.argmax(log_pvals, axis=1)
    confidence = np.max(log_pvals, axis=1)

    # Convert indices to label names
    labels = np.array([celltype_names[i] for i in labels_idx])

    return {
        "labels": labels,
        "confidence": confidence,
        "enrichment": log_pvals,
        "cluster_names": cluster_names,
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
        # Build first-occurrence lookup
        lookup: dict = {}
        for idx, lab in enumerate(feature_set):
            key = str(lab)
            if key not in lookup:
                lookup[key] = idx

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
            print(f"Label '{label_names[idx]}' has no markers")

    return X, label_names
