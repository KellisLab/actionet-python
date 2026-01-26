"""Marker detection and annotation functions."""

from typing import Optional, Union, Literal, Dict, List
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata
from scipy.sparse import issparse, csr_matrix

from .core import compute_feature_specificity
from . import _core


def find_markers(
    adata: AnnData,
    labels: Union[str, np.ndarray, pd.Series],
    labels_use: Optional[List[str]] = None,
    top_genes: Optional[int] = 50,
    features_use: Optional[str] = None,
    features_keep: Optional[Union[str, List[str]]] = None,
    layer: Optional[str] = None,
    n_threads: int = 0,
    result: Literal["table", "ranks", "scores"] = "table",
    return_type: Literal["dataframe", "dict", ""] = "dataframe",
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

    # Filter labels if labels_use is provided
    if labels_use is not None:
        mask = np.isin(labels_arr, labels_use)
        # Create a temporary AnnData with filtered cells
        adata_filtered = adata[mask, :].copy()
        labels_arr_filtered = labels_arr[mask]
    else:
        adata_filtered = adata
        labels_arr_filtered = labels_arr

    # Handle features_use parameter
    # Can be: None (use var_names) or str (column name in adata.var)
    if features_use is None:
        # Use all features
        feature_labels = adata_filtered.var_names.values
    else:
        # Extract from adata.var column
        if features_use not in adata_filtered.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        feature_labels = adata_filtered.var[features_use].values

    # Compute feature specificity on all data
    result_adata = compute_feature_specificity(
        adata_filtered,
        labels_arr_filtered,
        layer=layer,
        n_threads=n_threads,
        key_added="_temp_specificity",
        inplace=False,
    )

    upper_sig = result_adata.varm["_temp_specificity_upper"]
    lower_sig = result_adata.varm["_temp_specificity_lower"]

    # Compute feature specificity scores
    feat_spec = upper_sig - lower_sig
    feat_spec[feat_spec < 0] = 0

    # Get cluster names from the filtered labels
    # Use unique() to get only the categories that actually appear in filtered data
    cluster_names = np.unique(labels_arr_filtered)

    # Handle features_keep parameter (whitelist filtering)
    # Can be: None (no filtering), or list (explicit feature labels)
    if features_keep is not None:
        # Handle boolean columns
        if features_keep.dtype == bool:
            if len(features_keep) != len(feature_labels):
                raise ValueError("Length of features_keep must match number of features")
            keep_mask = features_keep
        else:
            # Explicit list provided - filter to features in the list
            keep_mask = np.isin(feature_labels, features_keep)

        # Apply the filter
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
    net_key: str = "actionet",
    norm_method: Literal["pagerank", "pagerank_sym"] = "pagerank",
    alpha: float = 0.85,
    max_it: int = 5,
    approx: bool = True,
    ignore_baseline: bool = False,
    use_enrichment: bool = True,
    use_lpa: bool = False,
    n_threads: int = 0,
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
    net_key : str, optional (default: "actionet")
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
    # Get feature labels
    if features_use is None:
        feature_set = adata.var_names.values
    else:
        if features_use not in adata.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        feature_set = adata.var[features_use].values

    # Encode markers into binary/weighted matrix
    X_markers, celltype_names = _encode_markers(markers, feature_set)

    # Get expression matrix
    if layer is None:
        S = adata.X
    else:
        S = adata.layers[layer]

    if not issparse(S):
        S = csr_matrix(S)

    # Transpose S to features × cells for C++ function
    S = S.T

    # Get network graph
    if net_key not in adata.obsp:
        raise ValueError(f"Network '{net_key}' not found in adata.obsp")
    G = adata.obsp[net_key]

    if not issparse(G):
        G = csr_matrix(G)

    # Convert norm_method to integer code
    norm_method_code = 2 if norm_method == "pagerank_sym" else 0

    # Convert X_markers to sparse matrix
    if not issparse(X_markers):
        X_markers = csr_matrix(X_markers)

    # Compute marker statistics using graph-based imputation
    if method == "vision":
        marker_stats = _core.compute_feature_stats_vision(
            G=G,
            S=S,
            X=X_markers,
            norm_method=norm_method_code,
            alpha=alpha,
            max_it=max_it,
            approx=approx,
            thread_no=n_threads,
        )
    elif method == "actionet":
        marker_stats = _core.compute_feature_stats(
            G=G,
            S=S,
            X=X_markers,
            norm_method=norm_method_code,
            alpha=alpha,
            max_it=max_it,
            approx=approx,
            thread_no=n_threads,
            ignore_baseline=ignore_baseline,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # marker_stats is cells × celltypes
    # Clean up invalid values
    marker_stats = np.nan_to_num(marker_stats, nan=0.0, posinf=0.0, neginf=0.0)
    enrichment = marker_stats.copy()

    # Compute labels and confidence
    if use_enrichment:
        # Normalize graph and compute enrichment
        Gn = _core.normalize_graph(G, norm_method=1).T  # Transpose for column-wise normalization
        marker_stats_pos = np.maximum(marker_stats, 0)

        log_pvals = _core.compute_graph_label_enrichment(Gn, marker_stats_pos, n_threads)

        labels_idx = np.argmax(log_pvals, axis=1)
        confidence = np.max(log_pvals, axis=1)
    else:
        labels_idx = np.argmax(marker_stats, axis=1)
        confidence = np.max(marker_stats, axis=1)

    # Convert indices to label names
    labels = np.array([celltype_names[i] for i in labels_idx])

    result = {
        "labels": labels,
        "confidence": confidence,
        "enrichment": enrichment,
    }

    # Optional label propagation
    if use_lpa:
        # Convert string labels to numeric
        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_idx[label] for label in labels], dtype=np.float64)

        corrected_numeric = _core.run_lpa(
            G=G,
            labels=numeric_labels,
            lambda_param=1.0,
            iters=3,
            sig_threshold=3.0,
            fixed_labels=None,
            thread_no=n_threads,
        )

        # Convert back to string labels
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        labels_corrected = np.array([idx_to_label[int(idx)] for idx in corrected_numeric])
        result["labels_corrected"] = labels_corrected

    return result


def _encode_markers(
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    feature_set: np.ndarray,
) -> tuple[np.ndarray, List[str]]:
    """
    Encode marker genes into a binary/weighted feature × celltype matrix.

    Mimics R implementation: supports dict/list, wide-format DataFrame, or matrix.
    DataFrames are treated as wide format only (columns = celltypes, values = gene names).

    Parameters
    ----------
    markers : dict, DataFrame, or ndarray
        Marker specification:
        - dict: Keys are cell types, values are lists of gene names
        - DataFrame: Wide format with columns as cell types, values as gene names
        - ndarray: Pre-computed marker matrix
    feature_set : ndarray
        Array of feature names.

    Returns
    -------
    tuple
        (X, celltype_names) where X is binary/weighted matrix of shape (n_features, n_celltypes)
        and celltype_names is a list of cell type names.
    """
    if isinstance(markers, np.ndarray):
        # Already a matrix - just return it with generic names
        n_celltypes = markers.shape[1] if markers.ndim > 1 else 1
        celltype_names = [f"Celltype_{i}" for i in range(n_celltypes)]
        return markers, celltype_names

    elif isinstance(markers, (dict, pd.DataFrame)):
        # Both dict and DataFrame are handled the same way:
        # Keys/columns are cell types, values are lists of gene names

        if isinstance(markers, pd.DataFrame):
            # Convert DataFrame to dict: columns -> cell types, values -> gene lists
            markers_dict = {}
            for col in markers.columns:
                # Get non-null gene names for this cell type
                genes = markers[col].dropna().tolist()
                markers_dict[col] = genes
        else:
            # Dict input - need to handle two cases:
            # 1. Dict of lists (expected): {'CT_1': ['gene1', 'gene2']}
            # 2. Dict of dicts (from df.to_dict()): {'CT_1': {0: 'gene1', 1: 'gene2'}}
            markers_dict = {}
            for key, val in markers.items():
                if isinstance(val, dict):
                    # Case 2: Extract values from nested dict, filtering out None
                    genes = [v for v in val.values() if v is not None and str(v) != 'nan']
                elif isinstance(val, (list, tuple)):
                    # Case 1: Already a list, just filter None/NaN
                    genes = [g for g in val if g is not None and str(g) != 'nan']
                else:
                    # Single value or other type - wrap in list
                    genes = [val] if val is not None else []
                markers_dict[key] = genes

        # Process dict of markers
        celltype_names = list(markers_dict.keys())
        n_features = len(feature_set)

        columns = []
        for celltype in celltype_names:
            col_data = np.zeros(n_features, dtype=np.float32)
            gene_list = markers_dict[celltype]

            for gene_spec in gene_list:
                # Parse signed markers (e.g., "CD3D+", "CD8A-")
                if isinstance(gene_spec, str):
                    if gene_spec.endswith('+'):
                        gene = gene_spec[:-1]
                        weight = 1.0
                    elif gene_spec.endswith('-'):
                        gene = gene_spec[:-1]
                        weight = -1.0
                    else:
                        gene = gene_spec
                        weight = 1.0
                else:
                    gene = str(gene_spec)
                    weight = 1.0

                # Find gene in feature set
                matching_idx = np.where(feature_set == gene)[0]
                if len(matching_idx) > 0:
                    idx = matching_idx[0]
                    col_data[idx] = weight

            columns.append(col_data.reshape(-1, 1))

        X = np.hstack(columns)
        return X, celltype_names

    else:
        raise ValueError(f"Unsupported markers type: {type(markers)}")

