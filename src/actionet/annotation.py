"""Marker detection and annotation functions."""

from typing import Optional, Union, Literal, Dict, List
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata
from scipy.sparse import issparse, csr_matrix

from .core import compute_feature_specificity, _compute_specificity_streamed, _labels_to_membership
from . import _core
from ._matrix_source import MatrixSource


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

    # Filter labels if labels_use is provided
    if labels_use is not None:
        mask = np.isin(labels_arr, labels_use)
        # Use a view to avoid backed-incompatible eager copies.
        adata_filtered = adata[mask, :]
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

    if getattr(adata_filtered, "isbacked", False):
        from pandas import Categorical
        from pandas.api.types import is_integer_dtype

        if not is_integer_dtype(labels_arr_filtered):
            cat = Categorical(labels_arr_filtered)
            labels_int = cat.codes.astype(np.int32)
            cluster_names = np.asarray(cat.categories)
        else:
            labels_int = labels_arr_filtered.astype(np.int32)
            cluster_names = np.unique(labels_arr_filtered)

        labels_int = labels_int + 1
        source = MatrixSource(adata_filtered, layer=layer)
        H = _labels_to_membership(labels_int, source.n_obs)
        streamed = _compute_specificity_streamed(source, H, chunk_size=backed_chunk_size)
        upper_sig = streamed["upper_significance"]
        lower_sig = streamed["lower_significance"]
    else:
        # compute_feature_specificity internally does np.asarray → Categorical
        # to map labels to integer codes.  Because we already normalised
        # labels_arr to a plain array above, the Categorical constructed here
        # will have the same lexicographic category order as the one inside
        # compute_feature_specificity, so cluster_names and the output
        # columns are guaranteed to align.
        from pandas import Categorical
        from pandas.api.types import is_integer_dtype

        if not is_integer_dtype(labels_arr_filtered):
            cat = Categorical(labels_arr_filtered)
            cluster_names = np.asarray(cat.categories)
        else:
            cluster_names = np.unique(labels_arr_filtered)

        temp_key = "_temp_specificity"
        result_adata = compute_feature_specificity(
            adata_filtered,
            labels_arr_filtered,
            layer=layer,
            n_threads=n_threads,
            key_added=temp_key,
            backed_chunk_size=backed_chunk_size,
            inplace=False,
        )
        upper_sig = result_adata.varm[f"{temp_key}_upper"]
        lower_sig = result_adata.varm[f"{temp_key}_lower"]

    # Compute feature specificity scores
    feat_spec = upper_sig - lower_sig
    feat_spec[feat_spec < 0] = 0

    # cluster_names was set above during label conversion to match
    # the column ordering of the specificity matrix.

    # Handle features_keep parameter (whitelist filtering)
    # Can be: None (no filtering), list/array of names, boolean mask, or adata.var column name
    if features_keep is not None:
        if isinstance(features_keep, str):
            if features_keep in adata_filtered.var.columns:
                keep_values = adata_filtered.var[features_keep].to_numpy()
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

    source = MatrixSource(adata, layer=layer)
    if source.is_backed:
        if issparse(X_markers):
            required_idx = np.where(np.asarray(X_markers.getnnz(axis=1)).ravel() > 0)[0]
        else:
            required_idx = np.where(np.any(np.asarray(X_markers) != 0, axis=1))[0]

        if required_idx.size == 0:
            raise ValueError("Marker set does not overlap features in AnnData.")

        X_markers = X_markers[required_idx, :]
        S_cells = source.feature_subset(
            required_idx,
            chunk_size=backed_chunk_size,
            prefer_sparse=True,
        )
        if not issparse(S_cells):
            S_cells = csr_matrix(np.asarray(S_cells))
        S = S_cells.T.tocsr()
    else:
        S = source.matrix
        if not issparse(S):
            S = csr_matrix(S)
        S = S.T

    # Get network graph
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found in adata.obsp")
    G = adata.obsp[network_key]

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


def annotate_clusters(
    adata: AnnData,
    markers: Union[Dict[str, List[str]], pd.DataFrame, np.ndarray],
    cluster_key: str = "cluster",
    features_use: Optional[str] = None,
    layer: Optional[str] = None,
    n_threads: int = 0,
    backed_chunk_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """
    Annotate clusters using known marker genes.

    This function assigns cell type labels to clusters by computing enrichment
    of marker gene sets within cluster-specific gene expression profiles. If
    cluster feature specificity has not been pre-computed, it will be calculated
    automatically.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers : dict, DataFrame, or ndarray
        Marker genes specification. Can be:
        - dict: Keys are cell types, values are lists of marker genes
        - DataFrame: Wide format with columns as cell types, values as gene names
        - ndarray: Binary/weighted matrix (features × cell types)
    cluster_key : str, optional (default: "cluster")
        Key in adata.obs containing cluster labels.
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
    >>> # Define markers and annotate (feature specificity computed automatically)
    >>> markers = {
    ...     "T cells": ["CD3D", "CD3E", "CD3G"],
    ...     "B cells": ["CD19", "MS4A1", "CD79A"],
    ...     "Monocytes": ["CD14", "FCGR3A"]
    ... }
    >>> result = annotate_clusters(adata, markers, cluster_key="cluster")
    >>> cluster_labels = result["labels"]  # Annotation for each cluster
    >>> cluster_confidence = result["confidence"]  # Confidence for each cluster
    >>> cluster_names = result["cluster_names"]  # Original cluster names

    >>> # Create a mapping from cluster name to annotation
    >>> import pandas as pd
    >>> cluster_annotation_map = pd.DataFrame({
    ...     "cluster": cluster_names,
    ...     "annotation": cluster_labels,
    ...     "confidence": cluster_confidence
    ... })

    >>> # Map annotations back to cells
    >>> cluster_to_annotation = dict(zip(cluster_names, cluster_labels))
    >>> adata.obs["cell_type"] = adata.obs["cluster"].map(cluster_to_annotation)

    >>> # Access the enrichment matrix (clusters × cell types)
    >>> enrichment = result["enrichment"]  # Row i corresponds to cluster_names[i]

    >>> # Note: For integer clusters with non-contiguous values (e.g., [1,2,5,8]),
    >>> # cluster_names will be [0,1,2,3,4,5,6,7,8] to match the C++ sparse array.
    >>> # Filter to only the clusters that actually have cells for mapping.
    """
    # Check if cluster labels exist
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

    cluster_labels = adata.obs[cluster_key].values

    # Normalize to plain array to ensure consistent ordering (matching compute_feature_specificity)
    if hasattr(cluster_labels, 'categories'):
        cluster_labels = np.asarray(cluster_labels)

    # Check if cluster feature specificity exists, if not compute it
    specificity_key = f"{cluster_key}_feat_spec"
    if specificity_key not in adata.varm:
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
    else:
        cluster_feat_spec = adata.varm[specificity_key]

    # Get cluster names in the same order as the feature specificity matrix columns
    # This matches the logic in compute_feature_specificity
    from pandas import Categorical
    from pandas.api.types import is_integer_dtype

    n_clusters = cluster_feat_spec.shape[1]

    if not is_integer_dtype(cluster_labels):
        # For string/categorical labels: use lexicographic ordering (from Categorical)
        cat = Categorical(cluster_labels)
        cluster_names = np.asarray(cat.categories)
    else:
        # For integer labels: C++ backend may create sparse array with max(label)+1 columns
        # We need to match the actual column count of the specificity matrix
        if n_clusters == len(np.unique(cluster_labels)):
            # Contiguous or small range: use actual unique values
            cluster_names = np.unique(cluster_labels)
        else:
            # Sparse integer range: create array matching matrix columns (0 to n_clusters-1)
            cluster_names = np.arange(n_clusters)

    # Get feature labels
    if features_use is None:
        feature_set = adata.var_names.values
    else:
        if features_use not in adata.var.columns:
            raise ValueError(f"Column '{features_use}' not found in adata.var")
        feature_set = adata.var[features_use].values

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
) -> tuple[np.ndarray, List[str]]:
    """
    Encode marker genes into a binary feature × celltype matrix.

    Mirrors R .encode_markers: accepts list-like (dict/DataFrame) or a numeric matrix,
    enforces named labels, and binarizes markers.

    Parameters
    ----------
    markers : dict, DataFrame, or ndarray
        Marker specification:
        - dict: keys are labels, values are lists of feature names
        - DataFrame: columns are labels, values are feature names
        - ndarray: numeric matrix (features × labels)
    feature_set : ndarray
        Array of feature names.

    Returns
    -------
    tuple
        (X, label_names) where X is a binary matrix of shape (n_features, n_labels)
        and label_names is a list of label names.
    """
    if isinstance(markers, np.ndarray):
        X = np.asarray(markers)
        if not np.isfinite(X).all():
            raise ValueError("'markers' contains non-numeric values")
        if X.ndim != 2:
            raise ValueError("'markers' must be a 2D array")
        if X.shape[0] != len(feature_set):
            raise ValueError("Number of rows in 'markers' does not match number of features")
        X = (X != 0).astype(np.float32)
        label_names = [f"Label_{i}" for i in range(X.shape[1])]
    elif isinstance(markers, pd.DataFrame):
        if markers.columns is None or markers.columns.isnull().any():
            raise ValueError("'markers' contains unnamed entries")
        if markers.columns.duplicated().any():
            raise ValueError("'markers' contains duplicated labels")
        label_names = markers.columns.tolist()
        columns = []
        for col in label_names:
            values = markers[col].dropna().astype(str).tolist()
            columns.append(np.isin(feature_set, values).astype(np.float32).reshape(-1, 1))
        X = np.hstack(columns) if columns else np.zeros((len(feature_set), 0), dtype=np.float32)
    elif isinstance(markers, dict):
        label_names = list(markers.keys())
        if any(name is None for name in label_names):
            raise ValueError("'markers' contains unnamed entries")
        if len(set(label_names)) != len(label_names):
            raise ValueError("'markers' contains duplicated labels")
        columns = []
        for name in label_names:
            vals = markers[name]
            if isinstance(vals, (list, tuple, np.ndarray, pd.Series)):
                values = [v for v in vals if v is not None]
            else:
                values = [vals] if vals is not None else []
            values = [str(v) for v in values]
            columns.append(np.isin(feature_set, values).astype(np.float32).reshape(-1, 1))
        X = np.hstack(columns) if columns else np.zeros((len(feature_set), 0), dtype=np.float32)
    else:
        raise ValueError("'markers' must be one of: dict, DataFrame, or ndarray")

    if X.shape[1] == 0:
        raise ValueError("No markers provided")

    col_sums = X.sum(axis=0)
    zero_cols = np.where(col_sums == 0)[0]
    if len(zero_cols) == X.shape[1]:
        raise ValueError("No markers in 'features_use'")
    if len(zero_cols) > 0:
        for idx in zero_cols:
            print(f"Label '{label_names[idx]}' has no markers")

    return X, label_names
