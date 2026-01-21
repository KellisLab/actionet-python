"""Data preprocessing functions"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from anndata import AnnData

def import_anndata_generic(
    input_path: str,
    mtx_file: str,
    gene_annotations: str,
    sample_annotations: str,
    gene_headers: list[str] | None = None,
    sample_headers: list[str] | None = None,
    sep: str = '\t',
    prefilter: bool = False,
    prefil_params: dict | None = None,
) -> AnnData:
    """
    Python implementation of import.se.generic from R, using AnnData.
    """

    # Read gene table
    gene_path = os.path.join(input_path, gene_annotations)
    gene_table = pd.read_csv(gene_path, sep=sep, header=None, dtype=str)
    if gene_headers is not None:
        gene_table.columns = gene_headers

    # Read sample annotations
    sample_path = os.path.join(input_path, sample_annotations)
    sample_annots = pd.read_csv(sample_path, sep=sep, header=None, dtype=str)
    if sample_headers is not None:
        sample_annots.columns = sample_headers

    # Set row/col names
    obs_names = sample_annots.iloc[:, 0].tolist()
    var_names = gene_table.iloc[:, 0].tolist()

    # Read matrix and Create AnnData object
    mtx_path = os.path.join(input_path, mtx_file)
    X = mmread(mtx_path).transpose()
    adata = AnnData(
        X=csr_matrix(X),
        obs=sample_annots,
        var=gene_table
    )
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obs_names_make_unique(join="_")
    adata.var_names_make_unique(join="_")

    # Optionally prefilter
    if prefilter and prefil_params is not None:
        adata = filter_anndata(
            adata,
            layer_name=None,
            min_cells_per_feat=prefil_params.get("min_cells_per_feat", None),
            min_feats_per_cell=prefil_params.get("min_feats_per_cell", None),
            min_umis_per_cell=prefil_params.get("min_umis_per_cell", None),
            max_umis_per_cell=prefil_params.get("max_umis_per_cell", None),
            filter_adata=True
        )

    return adata

def filter_anndata(
    adata,
    layer_name: str | None = None,
    min_cells_per_feat: int | float | None = None,
    min_feats_per_cell: int | None = None,
    min_umis_per_cell: int | None = None,
    max_umis_per_cell: int | None = None,
    inplace: bool = True,
    filter_adata: bool = True
) -> Union[AnnData, dict, None]:
    """
    Filter rows (features/genes) and columns (cells) of an AnnData object in a single pass.
    Parameters:
        adata: AnnData object
        layer_name: str, layer name or None for `X` matrix
        min_cells_per_feat: int or float, minimum cells per feature (gene)
        min_feats_per_cell: int, minimum features per cell
        min_umis_per_cell: int, minimum UMIs per cell
        max_umis_per_cell: int, maximum UMIs per cell
        inplace: bool, if True filter adata in place and return None, else return a filtered copy
        filter_adata: bool, if True Filtered adata (in place or copy), else return dict containing names and indices of filtered elements
    Returns:
        AnnData, dict, or None
    """
    import numpy as np
    import pandas as pd

    X = adata.layers[layer_name] if layer_name is not None else adata.X
    obs_names = np.array(adata.obs_names)
    var_names = np.array(adata.var_names)

    # Start with all indices
    obs_idx = np.arange(X.shape[0])
    var_idx = np.arange(X.shape[1])

    prev_shape = None

    while True:
        row_mask = np.ones(X.shape[0], dtype=bool)
        col_mask = np.ones(X.shape[1], dtype=bool)

        if min_umis_per_cell is not None:
            row_mask &= np.array(X.sum(axis=1)).flatten() >= min_umis_per_cell
        if max_umis_per_cell is not None:
            row_mask &= np.array(X.sum(axis=1)).flatten() <= max_umis_per_cell
        if min_feats_per_cell is not None:
            row_mask &= np.array((X > 0).sum(axis=1)).flatten() >= min_feats_per_cell
        if min_cells_per_feat is not None:
            if isinstance(min_cells_per_feat, float) and 0 < min_cells_per_feat < 1:
                min_fc = int(np.ceil(min_cells_per_feat * X.shape[0]))
            else:
                min_fc = min_cells_per_feat
            col_mask &= np.array((X > 0).sum(axis=0)).flatten() >= min_fc

        # If shape does not change, break
        new_shape = (row_mask.sum(), col_mask.sum())
        if prev_shape == new_shape:
            break
        prev_shape = new_shape

        # Subset matrix and update index mapping
        X = X[row_mask, :][:, col_mask]
        obs_idx = obs_idx[row_mask]
        var_idx = var_idx[col_mask]

    # Final masks for original AnnData
    final_obs_mask = np.zeros(adata.n_obs, dtype=bool)
    final_var_mask = np.zeros(adata.n_vars, dtype=bool)
    final_obs_mask[obs_idx] = True
    final_var_mask[var_idx] = True

    if filter_adata:
        if inplace:
            adata._inplace_subset_obs(final_obs_mask)
            adata._inplace_subset_var(final_var_mask)
            return None
        else:
            adata_copy = adata.copy()
            adata_copy._inplace_subset_obs(final_obs_mask)
            adata_copy._inplace_subset_var(final_var_mask)
            return adata_copy
    else:
        filtered_obs = pd.DataFrame({
            "name": obs_names,
            "idx": np.arange(adata.n_obs),
            "mask": final_obs_mask
        })
        filtered_vars = pd.DataFrame({
            "name": var_names,
            "idx": np.arange(adata.n_vars),
            "mask": final_var_mask
        })
        return {
            "fil_vars": filtered_vars,
            "fil_obs": filtered_obs
        }
