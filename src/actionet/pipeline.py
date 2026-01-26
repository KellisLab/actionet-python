"""Main ACTIONet pipeline functions."""

from typing import Optional, Literal
import numpy as np
from anndata import AnnData

from .core import (
    run_action,
    build_network,
    compute_network_diffusion,
    compute_archetype_feature_specificity,
    layout_network,
)
from .tools import scale
from .visualization import compute_node_colors


def run_actionet(
    adata: AnnData,
    k_min: int = 2,
    k_max: int = 30,
    layer: Optional[str] = None,
    reduction_key: str = "action",
    network_key: str = "actionet",
    min_observations: int = 2,
    max_iter: int = 50,
    specificity_threshold: float = -3.0,
    network_metric: Literal["jsd", "l2", "ip"] = "jsd",
    network_algorithm: Literal["k*nn", "knn"] = "k*nn",
    network_density: float = 1.0,
    mutual_edges_only: bool = True,
    layout_method: Literal["umap", "tumap"] = "umap",
    layout_epochs: int = 100,
    layout_spread: float = 1.0,
    layout_min_dist: float = 1.0,
    layout_3d: bool = True,
    layout_parallel: bool = True,
    compute_specificity_parallel: bool = False,
    n_threads: int = 0,
    seed: int = 0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run the complete ACTIONet pipeline.

    This function executes the full ACTIONet workflow including:
    1. ACTION archetypal analysis
    2. Network construction
    3. Network centrality (TODO)
    4. Network-based diffusion
    5. 2D/3D layout generation
    6. Node color computation
    7. Feature specificity analysis

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells × features).
    k_min : int, optional (default: 2)
        Minimum number of archetypes.
    k_max : int, optional (default: 30)
        Maximum number of archetypes.
    layer : str, optional
        Layer in adata to use. If None, uses adata.X.
    reduction_key : str, optional (default: "action")
        Key for storing/retrieving reduced representation in adata.obsm.
    network_key : str, optional (default: "actionet")
        Key for storing network in adata.obsp.
    min_observations : int, optional (default: 2)
        Minimum observations per archetype.
    max_iter : int, optional (default: 50)
        Maximum iterations for ACTION algorithm.
    specificity_threshold : float, optional (default: -3.0)
        Z-score threshold for filtering non-specific archetypes.
        More negative = more stringent.
    network_metric : {"jsd", "l2", "ip"}, optional (default: "jsd")
        Distance metric for network construction:
        - "jsd": Jensen-Shannon divergence
        - "l2": L2 norm
        - "ip": Inner product
    network_algorithm : {"knn", "k*nn"}, optional (default: "k*nn")
        Algorithm for network construction:
        - "knn": k-nearest neighbors
        - "k*nn": k-star nearest neighbors (adaptive)
    network_density : float, optional (default: 1.0)
        Density factor for network construction.
    mutual_edges_only : bool, optional (default: True)
        Keep only mutual nearest neighbors.
    layout_method : {"umap", "tumap"}, optional (default: "umap")
        Layout algorithm (UMAP or t-UMAP).
    layout_epochs : int, optional (default: 100)
        Number of optimization epochs for layout.
    layout_spread : float, optional (default: 1.0)
        Spread parameter for UMAP.
    layout_min_dist : float, optional (default: 1.0)
        Minimum distance parameter for UMAP.
    layout_3d : bool, optional (default: True)
        If True, compute both 2D and 3D layouts.
    layout_parallel : bool, optional (default: True)
        Use multiple threads for layout computation.
    compute_specificity_parallel : bool, optional (default: False)
        Use multiple threads for specificity computation.
        May cause memory issues on large datasets.
    n_threads : int, optional (default: 0)
        Number of threads (0=auto).
    seed : int, optional (default: 0)
        Random seed for reproducibility.
    inplace : bool, optional (default: True)
        If True, modifies adata in place. If False, returns modified copy.

    Returns
    -------
    AnnData or None
        If inplace=False, returns modified AnnData object.
        If inplace=True, returns None and modifies adata in place.

    Updates AnnData
    ---------------
    adata.obsm[reduction_key] : np.ndarray
        Reduced kernel representation.
    adata.obsm["H_stacked"] : np.ndarray
        Stacked archetype footprints.
    adata.obsm["H_merged"] : np.ndarray
        Merged archetype footprints.
    adata.obsm["archetype_footprint"] : np.ndarray
        Diffused archetype footprints.
    adata.obs["assigned_archetype"] : pd.Series
        Dominant archetype per cell.
    adata.obs["node_centrality"] : pd.Series
        Network centrality scores.
    adata.obsp[network_key] : sparse matrix
        Cell-cell similarity network.
    adata.obsm[f"{layout_method}_2d_{network_key}"] : np.ndarray
        2D layout coordinates.
    adata.obsm[f"{layout_method}_3d_{network_key}"] : np.ndarray
        3D layout coordinates (if layout_3d=True).
    adata.obsm[f"colors_{network_key}"] : np.ndarray
        RGB colors for 3D layout (if layout_3d=True).
    adata.varm["archetype_feat_profile"] : np.ndarray
        Average feature profile per archetype (features × archetypes).
    adata.varm["archetype_feat_specificity_upper"] : np.ndarray
        Upper-tail significance scores (features × archetypes).
    adata.varm["archetype_feat_specificity_lower"] : np.ndarray
        Lower-tail significance scores (features × archetypes).

    Examples
    --------
    >>> import actionet as act
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata = act.run_actionet(adata, k_max=50, inplace=False)
    >>> print(adata.obs['assigned_archetype'])

    See Also
    --------
    reduce_kernel : Compute kernel reduction
    run_action : Run ACTION decomposition
    build_network : Build cell-cell network
    """
    if not inplace:
        adata = adata.copy()

    # Step 1: ACTION archetypal analysis
    print("Running ACTION decomposition...")
    run_action(
        adata,
        k_min=k_min,
        k_max=k_max,
        reduction_key=reduction_key,
        prenormalize=True,
        max_iter=max_iter,
        specificity_threshold=specificity_threshold,
        min_observations=min_observations,
        n_threads=n_threads,
        inplace=True,
    )

    # Step 2: Build ACTIONet
    print("Building network...")
    build_network(
        adata,
        obsm_key="H_stacked",
        algorithm=network_algorithm,
        distance_metric=network_metric,
        density=network_density,
        mutual_edges_only=mutual_edges_only,
        key_added=network_key,
        n_threads=n_threads,
        inplace=True,
    )

    # Step 3: Compute network centrality
     # TODO: Implement network_centrality

    # Step 4: Smooth archetype footprints via network diffusion
    print("Computing archetype footprints via diffusion...")
    compute_network_diffusion(
        adata,
        scores="H_merged",
        network_key=network_key,
        norm_method="pagerank",
        key_added="archetype_footprint",
        n_threads=n_threads,
        inplace=True,
    )

    # Step 5: Compute 2D layout
    print(f"Computing 2D {layout_method.upper()} layout...")
    initial_coordinates = adata.obsm["archetype_footprint"].copy()

    layout_2d_key = f"{layout_method}_2d_{network_key}"
    layout_network(
        adata,
        network_key=network_key,
        initial_coords=scale(initial_coordinates),
        method=layout_method,
        n_components=2,
        min_dist=layout_min_dist,
        spread=layout_spread,
        n_epochs=layout_epochs,
        key_added=layout_2d_key,
        seed=seed,
        n_threads=n_threads if layout_parallel else 1,
        inplace=True,
    )

    # Step 6: Compute 3D layout (optional)
    if layout_3d:
        print(f"Computing 3D {layout_method.upper()} layout...")

        # Warm-start 3D layout using 2D embedding
        initial_coordinates_3d = np.column_stack([
            adata.obsm[layout_2d_key],
            initial_coordinates[:, 2] if initial_coordinates.shape[1] > 2 else np.zeros(adata.n_obs)
        ])

        layout_3d_key = f"{layout_method}_3d_{network_key}"
        layout_network(
            adata,
            network_key=network_key,
            initial_coords=scale(initial_coordinates_3d),
            method=layout_method,
            n_components=3,
            min_dist=layout_min_dist,
            spread=layout_spread,
            n_epochs=layout_epochs // 2,
            key_added=layout_3d_key,
            seed=seed,
            n_threads=n_threads if layout_parallel else 1,
            inplace=True,
        )

        # Compute node colors from 3D coordinates
        print("Computing node colors...")
        compute_node_colors(
            adata,
            embedding_key=layout_3d_key,
            key_added=f"colors_{network_key}",
            n_threads=n_threads,
            inplace=True,
        )

    # Step 7: Compute feature specificity for each archetype
    print("Computing feature specificity...")
    compute_archetype_feature_specificity(
        adata,
        archetype_key="archetype_footprint",
        layer=layer,
        key_added="archetype",
        n_threads=1 if not compute_specificity_parallel else n_threads,
        inplace=True,
    )

    if not inplace:
        return adata
    return None
