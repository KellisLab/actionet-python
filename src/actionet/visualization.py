"""Visualization functions for ACTIONet."""

from typing import Optional, Union
import numpy as np
from anndata import AnnData

from . import _core


def compute_node_colors(
    adata: AnnData,
    embedding_key: str = "umap_3d_actionet",
    key_added: Optional[str] = None,
    n_threads: int = 1,
    inplace: bool = True,
) -> Optional[Union[AnnData, np.ndarray]]:
    """
    Compute RGB colors for nodes based on their 3D embedding coordinates.

    This function maps 3D coordinates to RGB colors using the Lab color space.
    It performs SVD on the coordinates, normalizes the first 3 components, and
    converts from Lab to RGB color space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    embedding_key : str, optional (default: "umap_3d_actionet")
        Key in adata.obsm containing 3D embedding coordinates.
    key_added : str, optional
        Key in adata.obsm to store the computed colors.
        If None, uses f"colors_{embedding_key}".
    n_threads : int, optional (default: 1)
        Number of threads to use.
    inplace : bool, optional (default: True)
        If True, adds colors to adata.obsm and returns adata.
        If False, returns color array without modifying adata.

    Returns
    -------
    AnnData or np.ndarray
        If inplace=True, returns adata with colors added to obsm.
        If inplace=False, returns (n_cells, 3) array of RGB colors.

    Raises
    ------
    ValueError
        If embedding_key is not found in adata.obsm.
    ValueError
        If embedding has fewer than 3 dimensions.

    Notes
    -----
    The function requires at least 3D coordinates. RGB values are in range [0, 1].

    Examples
    --------
    >>> import actionet as act
    >>> adata = act.compute_node_colors(adata, embedding_key="X_umap_3d")
    >>> print(adata.obsm['colors_X_umap_3d'])
    """
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    coordinates = adata.obsm[embedding_key]

    if coordinates.shape[1] < 3:
        raise ValueError(
            f"Embedding must have at least 3 dimensions, got {coordinates.shape[1]}"
        )

    # Call C++ implementation
    colors = _core.compute_node_colors(coordinates, n_threads)

    if not inplace:
        return colors

    # Add to adata
    if key_added is None:
        key_added = f"colors_{embedding_key}"

    adata.obsm[key_added] = colors
    return adata
