"""Utilities for translating between AnnData and C++ data structures."""

from typing import Optional, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .tools import aggregate_matrix

def anndata_to_matrix(
    adata: AnnData,
    layer: Optional[str] = None,
    transpose: bool = False,
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Extract matrix from AnnData for C++ input.
    
    Parameters
    ----------
    adata
        AnnData object.
    layer
        Layer name (None uses .X).
    transpose
        Transpose to genes x cells format.
        
    Returns
    -------
    Matrix (sparse or dense).
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    
    if transpose:
        if sp.issparse(X):
            X = X.T.tocsr()
        else:
            X = X.T
    
    return X


def matrix_to_anndata(
    X: Union[np.ndarray, sp.spmatrix],
    obs_names: Optional[np.ndarray] = None,
    var_names: Optional[np.ndarray] = None,
) -> AnnData:
    """
    Create AnnData from matrix.
    
    Parameters
    ----------
    X
        Expression matrix (cells x genes).
    obs_names
        Cell names.
    var_names
        Gene names.
        
    Returns
    -------
    AnnData object.
    """
    
    adata = AnnData(X)
    
    if obs_names is not None:
        adata.obs_names = obs_names
    if var_names is not None:
        adata.var_names = var_names
    
    return adata


def add_action_results(
    adata: AnnData,
    result: dict,
) -> None:
    """
    Add ACTION decomposition results to AnnData.
    
    Parameters
    ----------
    adata
        AnnData object to update in-place.
    result
        Result dictionary from run_action().
    """
    # Store archetype matrices
    adata.obsm["H_stacked"] = result["H_stacked"].T
    adata.obsm["H_merged"] = result["H_merged"].T

    adata.obsm["C_stacked"] = result["C_stacked"]
    adata.obsm["C_merged"] = result["C_merged"]

    # Store archetype assignments
    adata.obs["assigned_archetype"] = result["assigned_archetypes"]


def add_network_to_anndata(
    adata: AnnData,
    network: sp.spmatrix,
    key: str = "actionet",
) -> None:
    """
    Add network to AnnData.obsp.
    
    Parameters
    ----------
    adata
        AnnData object.
    network
        Sparse adjacency matrix.
    key
        Key in adata.obsp.
    """
    adata.obsp[key] = network


def _count_nonzero_grouped(
    X: Union[np.ndarray, sp.spmatrix],
    group_labels: np.ndarray,
    dim: int,
    todense: bool,
    return_inverse: bool = False,
) -> Union[np.ndarray, sp.spmatrix, tuple]:
    """
    Count nonzero entries by group.

    Parameters
    ----------
    X
        Input matrix.
    group_labels
        Group labels for aggregation.
    dim
        Dimension to aggregate (1=columns, 2=rows).
    todense
        Whether to return dense result.
    return_inverse
        Whether to return inverse indices and unique labels.

    Returns
    -------
    Aggregated count matrix, or tuple if return_inverse=True.
    """
    if sp.issparse(X):
        if todense:
            X_bool = X.toarray() != 0
            return aggregate_matrix(
                X_bool.astype(np.float64),
                group_labels,
                dim=dim,
                method="sum",
                return_sparse=False,
                return_inverse=return_inverse,
            )
        X_nz = X.copy()
        X_nz.data = np.ones_like(X_nz.data)
        return aggregate_matrix(
            X_nz,
            group_labels,
            dim=dim,
            method="sum",
            return_sparse=True,
            return_inverse=return_inverse,
        )

    X_bool = np.asarray(X) != 0
    return aggregate_matrix(
        X_bool.astype(np.float64),
        group_labels,
        dim=dim,
        method="sum",
        return_sparse=False,
        return_inverse=return_inverse,
    )


def aggregate_anndata(
    adata: AnnData,
    by: Union[str, list[str]],
    func: Union[str, list, None] = None,
    layer: Optional[str] = None,
    min_count: int = 0,
    axis: int = 0,
    todense: bool = False,
) -> AnnData:
    """
    Aggregate an AnnData object by groups defined in obs or var annotations.

    Parameters
    ----------
    adata
        Input AnnData object.
    by
        Column name(s) in obs (axis=0) or var (axis=1) to group by.
        Multiple columns will be combined with '_'.
    func
        Aggregation function(s): 'sum', 'mean', 'var', or 'count_nonzero'.
        If None, defaults to ['sum', 'mean', 'var', 'count_nonzero'].
        If a single string, applies that function only.
        If a list, applies all specified functions and stores in layers.
    layer
        Name of layer to aggregate. If None, uses .X.
    min_count
        Minimum count threshold.
    axis
        0 to aggregate observations (default), 1 to aggregate variables.
    todense
        If input matrix is sparse, convert to and return dense aggregation.

    Returns
    -------
    AnnData
        Aggregated AnnData object with .X=None and all results stored as named
        layers (one per func).
    """
    if func is None:
        func = ["sum", "mean", "var", "count_nonzero"]
    elif isinstance(func, str):
        func = [func]
    elif not isinstance(func, list):
        raise TypeError(f"func must be a string, list of strings, or None, got {type(func)}")

    valid_funcs = ["sum", "mean", "var", "count_nonzero"]
    for f in func:
        if not isinstance(f, str):
            raise TypeError(f"All elements in func must be strings, got {type(f)}")
        if f not in valid_funcs:
            raise ValueError(
                f"Invalid aggregation function '{f}'. "
                f"Must be one of {valid_funcs}"
            )

    if isinstance(by, str):
        by = [by]
    elif not isinstance(by, list):
        raise TypeError(f"by must be a string or list of strings, got {type(by)}")

    if axis not in (0, 1):
        raise ValueError("axis must be 0 (obs) or 1 (var)")

    X = adata.layers[layer] if layer is not None else adata.X
    metadata_df = adata.obs if axis == 0 else adata.var

    if len(by) == 1:
        group_labels = metadata_df[by[0]].astype(str)
    else:
        group_labels = metadata_df[by].astype(str).agg("_".join, axis=1)

    group_labels = group_labels.values
    dim = 2 if axis == 0 else 1

    # Pre-filter if min_count is specified to avoid unnecessary computation
    if min_count > 0:
        unique_labels, inverse = np.unique(group_labels, return_inverse=True)
        group_counts = np.bincount(inverse, minlength=len(unique_labels))
        keep_mask = group_counts >= min_count
        if not keep_mask.all():
            dropped = unique_labels[~keep_mask]
            dropped_list = ", ".join(map(str, dropped))
            print(f"Dropped groups: {dropped_list}")
            keep_labels = unique_labels[keep_mask]
            axis_mask = np.isin(group_labels, keep_labels)
            if axis == 0:
                adata = adata[axis_mask, :]
            else:
                adata = adata[:, axis_mask]
            X = adata.layers[layer] if layer is not None else adata.X
            metadata_df = adata.obs if axis == 0 else adata.var
            group_labels = (metadata_df[by[0]].astype(str) if len(by) == 1
                            else metadata_df[by].astype(str).agg("_".join, axis=1))
            group_labels = group_labels.values

    # Compute unique labels and inverse for aggregation
    unique_labels, inverse = np.unique(group_labels, return_inverse=True)
    group_counts = np.bincount(inverse, minlength=len(unique_labels))

    layers_dict = {}
    for f in func:
        if f == "count_nonzero":
            X_agg = _count_nonzero_grouped(
                X,
                group_labels,
                dim=dim,
                todense=todense,
                return_inverse=False,
            )
        else:
            return_sparse = sp.issparse(X) and not todense
            X_agg = aggregate_matrix(
                X,
                group_labels,
                dim=dim,
                method=f,
                return_sparse=return_sparse,
                return_inverse=False,
            )
        layers_dict[f] = X_agg

    agg_metadata_df = metadata_df[by].copy()
    agg_metadata_df["_group_label"] = group_labels
    agg_metadata_df = (
        agg_metadata_df
        .drop_duplicates(subset="_group_label")
        .set_index("_group_label")
        .loc[list(unique_labels)]
        .reset_index(drop=True)
    )
    agg_metadata_df.index = list(unique_labels)

    count_col = "n_obs" if axis == 0 else "n_var"
    agg_metadata_df[count_col] = group_counts

    if axis == 0:
        adata_agg = AnnData(X=None, obs=agg_metadata_df, var=adata.var.copy())
    else:
        adata_agg = AnnData(X=None, obs=adata.obs.copy(), var=agg_metadata_df)

    adata_agg.layers.update(layers_dict)


    return adata_agg
