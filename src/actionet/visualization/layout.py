"""Network layout and node color visualization functions."""

from typing import Literal, Optional, Union
import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from .. import _core
from ..io.persist import persist_updates
from ..decomposition import run_svd


def layout_network(
    adata: AnnData,
    network_key: str = "actionet",
    initial_coords: Optional[Union[str, np.ndarray]] = None,
    layer: Optional[str] = None,
    method: Literal["umap", "tumap", "largevis", "leopold", "leopold2"] = "umap",
    n_components: int = 2,
    spread: float = 1.0,
    min_dist: float = 1.0,
    n_epochs: int = 0,
    learning_rate: float = 1.0,
    repulsion_strength: float = 1.0,
    negative_sample_rate: float = 3.0,
    approx_pow: bool = True,
    pcg_rand: bool = True,
    rng_type: Optional[str] = None,
    batch: bool = True,
    grain_size: int = 1,
    a: float = 0.0,
    b: float = 0.0,
    opt_method: Literal["adam", "sgd"] = "adam",
    alpha: float = -1.0,
    beta1: float = 0.5,
    beta2: float = 0.9,
    eps: float = 1e-7,
    ai: Optional[np.ndarray] = None,
    aj: Optional[np.ndarray] = None,
    seed: int = 0,
    n_threads: int = 0,
    verbose: bool = True,
    key_added: str = "X_umap",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute 2D/3D layout of ACTIONet graph using uwot methods.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    network_key
        Key in adata.obsp containing network.
    initial_coords
        Initial coordinates. Can be a key in adata.obsm, a numpy array, or None.
        If None, computes initial coordinates via SVD on the specified layer.
    layer
        Layer to use for computing initial coordinates via SVD (if initial_coords is None).
        If None, uses adata.X.
    method
        Layout method: "umap", "tumap", "largevis", "leopold", or "leopold2".
    n_components
        Number of dimensions (2 or 3).
    spread
        UMAP spread parameter.
    min_dist
        UMAP min_dist parameter.
    n_epochs
        Number of optimization epochs (0=auto).
    learning_rate
        Base learning rate.
    repulsion_strength
        Repulsion strength (uwot gamma).
    negative_sample_rate
        Negative sample rate.
    approx_pow
        Use approximate power function in UMAP gradient.
    pcg_rand
        Legacy RNG toggle (kept for backward compatibility).
    rng_type
        RNG implementation ("pcg", "tausworthe", "deterministic"). If provided,
        it takes precedence over `pcg_rand`.
    batch
        Use batch updates.
    grain_size
        Parallel grain size.
    a, b
        UMAP shape parameters. Zero values auto-compute from spread/min_dist.
    opt_method
        Optimizer: "adam" or "sgd".
    alpha, beta1, beta2, eps
        Optimizer hyperparameters.
    ai, aj
        Per-vertex coefficient vectors required for "leopold" (`ai`) and
        "leopold2" (`ai` and `aj`).
    seed
        Random seed.
    n_threads
        Number of threads.
    verbose
        Whether to print progress messages.
    key_added
        Key to store layout in adata.obsm.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Layout coordinates (cells × n_components).
    """
    valid_methods = {"umap", "tumap", "largevis", "leopold", "leopold2"}
    method = method.lower()
    if method not in valid_methods:
        raise ValueError(
            f"Invalid `method` '{method}'. Must be one of {sorted(valid_methods)}."
        )

    rng_value = ""
    if rng_type is not None:
        if not isinstance(rng_type, str):
            raise TypeError("`rng_type` must be a string when provided.")
        rng_value = rng_type.strip().lower()
        valid_rng = {"pcg", "tausworthe", "deterministic"}
        if rng_value not in valid_rng:
            raise ValueError(
                f"Invalid `rng_type` '{rng_type}'. Must be one of {sorted(valid_rng)}."
            )

    def _coerce_optional_vector(values: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"`{name}` must have length {adata.n_obs} (number of observations), "
                f"got {arr.shape[0]}."
            )
        return np.ascontiguousarray(arr, dtype=np.float64)

    ai_arr = _coerce_optional_vector(ai, "ai")
    aj_arr = _coerce_optional_vector(aj, "aj")

    if method == "leopold" and ai_arr is None:
        raise ValueError("`ai` must be provided when `method='leopold'`.")
    if method == "leopold2" and (ai_arr is None or aj_arr is None):
        raise ValueError("`ai` and `aj` must be provided when `method='leopold2'`.")

    if not inplace:
        adata = adata.copy()
    from ..tools.anndata import resolve_network
    G = resolve_network(adata, network_key)
    
    # Handle initial_coords
    if initial_coords is None:
        # Compute initial coordinates from SVD
        if verbose:
            if layer is not None:
                print(f"Computing initial coordinates from layer '{layer}' via SVD")
            else:
                print("Computing initial coordinates from adata.X via SVD")

        k = max(3, n_components)
        svd_result = run_svd(
            adata,
            n_components=k,
            layer=layer,
            algorithm=None,
            max_iter=0,
            seed=seed,
            verbose=verbose,
            return_operator_compatible=True,
        )

        # Get left singular vectors (cells × k) as initial coords
        initial_coords = svd_result["u"]  # cells x k in new orientation
        # Scale columns to have mean 0 and std 1
        initial_coords = (initial_coords - initial_coords.mean(axis=0)) / initial_coords.std(axis=0)
    elif isinstance(initial_coords, str):
        # initial_coords is a key in adata.obsm
        if initial_coords not in adata.obsm:
            raise ValueError(f"Initial coordinates '{initial_coords}' not found in adata.obsm.")
        initial_coords = adata.obsm[initial_coords]
    else:
        # initial_coords is a numpy array
        initial_coords = np.asarray(initial_coords)

    # Validate initial_coords shape
    if initial_coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"Number of rows in initial_coords ({initial_coords.shape[0]}) "
            f"does not match number of cells in adata ({adata.n_obs})"
        )

    if initial_coords.shape[1] < n_components:
        raise ValueError(
            f"Number of columns in initial_coords ({initial_coords.shape[1]}) "
            f"must be >= n_components ({n_components})"
        )

    # Ensure initial_coords is float64 and C-contiguous (binding accepts double)
    initial_coords = np.ascontiguousarray(initial_coords, dtype=np.float64)

    coords = _core.layout_network(
        G=G,
        initial_coords=initial_coords,
        method=method,
        n_components=n_components,
        spread=spread,
        min_dist=min_dist,
        n_epochs=n_epochs,
        seed=seed,
        thread_no=n_threads,
        verbose=verbose,
        learning_rate=learning_rate,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        approx_pow=approx_pow,
        pcg_rand=pcg_rand,
        rng_type=rng_value,
        batch=batch,
        grain_size=grain_size,
        a=a,
        b=b,
        opt_method=opt_method,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        ai=ai_arr,
        aj=aj_arr,
    )

    persist_updates(adata, obsm={key_added: coords})
    if not inplace:
        return adata
    return None


def compute_node_colors(
    obj: Union[AnnData, np.ndarray, sp.spmatrix],
    embedding_key: str = "umap_3d_actionet",
    key_added: Optional[str] = None,
    n_threads: int = 1,
    return_raw: bool = False,
) -> Optional[Union[AnnData, np.ndarray]]:
    """
    Compute RGB colors for nodes based on their 3D embedding coordinates.

    This function maps 3D coordinates to RGB colors using the Lab color space.
    It performs SVD on the coordinates, normalizes the first 3 components, and
    converts from Lab to RGB color space.

    Parameters
    ----------
    obj
        Annotated data matrix or matrix-like object (coerced to np.ndarray).
    embedding_key : str, optional (default: "umap_3d_actionet")
        Key in adata.obsm containing 3D embedding coordinates.
    key_added : str, optional
        Key in adata.obsm to store the computed colors.
        If None, uses f"colors_{embedding_key}".
    n_threads : int, optional (default: 1)
        Number of threads to use.
    return_raw : bool, optional (default: False)
        If True, returns color array without adding it to adata.

    Returns
    -------
    AnnData or np.ndarray
        If ``return_raw=False`` and *obj* is an AnnData, returns *obj* with
        colors added under ``obsm[key_added]``.  Otherwise returns the
        ``(n_cells, 3)`` array of RGB colors directly.

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
    >>> import actionet as an
    >>> adata = an.compute_node_colors(adata, embedding_key="X_umap_3d")
    >>> print(adata.obsm['colors_X_umap_3d'])
    """
    is_anndata = False
    if isinstance(obj, AnnData):
        is_anndata = True

    if is_anndata:
        if embedding_key not in obj.obsm:
            raise ValueError(
                f"Embedding '{embedding_key}' not found in adata.obsm. "
                f"Available keys: {list(obj.obsm.keys())}"
            )
        coordinates = obj.obsm[embedding_key]
    else:
        if isinstance(obj, sp.spmatrix):
            coordinates = obj.toarray()
        else:
            coordinates = np.asarray(obj)

    if coordinates.shape[1] < 3:
        raise ValueError(
            f"Embedding must have at least 3 dimensions, got {coordinates.shape[1]}"
        )

    # Call C++ implementation
    colors = _core.compute_node_colors(coordinates, n_threads)

    if is_anndata and not return_raw:
        # Add to adata
        if key_added is None:
            key_added = f"colors_{embedding_key}"

        persist_updates(obj, obsm={key_added: colors})
        return obj

    return colors
