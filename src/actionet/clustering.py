"""Clustering helpers and high-level clustering APIs."""

from contextlib import contextmanager
import random
from typing import Literal, Optional, Union

from anndata import AnnData
import numpy as np
import scipy.sparse as sp

from ._backed_persist import persist_updates


def cluster_network(
    adata: AnnData,
    objective_function: Literal["modularity", "CPM", "cpm"] = "modularity",
    resolution_parameter: float = 1.0,
    initial_membership: Optional[Union[str, np.ndarray]] = None,
    n_iterations: int = 3,
    min_size: int = 3,
    network_key: str = "actionet",
    key_added: Optional[str] = None,
    beta: float = 0.01,
    random_state: Optional[int] = 0,
    return_raw: bool = False,
    inplace: bool = True,
) -> Optional[Union[AnnData, np.ndarray]]:
    """
    Cluster a precomputed cell-cell graph with Leiden using only ``adata.obsp``.

    This mirrors ``actionet-r::clusterNetwork()`` behavior while avoiding Scanpy's
    full AnnData-oriented clustering path. The function operates directly on
    ``adata.obsp[network_key]`` and optional initial labels.
    """
    if not inplace and not return_raw:
        adata = adata.copy()

    if resolution_parameter <= 0:
        raise ValueError("`resolution_parameter` must be > 0.")
    if n_iterations == 0:
        raise ValueError("`n_iterations` cannot be 0. Use positive values or -1.")

    from .anndata_utils import resolve_network
    adjacency = resolve_network(adata, network_key)
    if adjacency.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(
            f"Network '{network_key}' must be shape ({adata.n_obs}, {adata.n_obs}), "
            f"got {adjacency.shape}."
        )

    if sp.issparse(adjacency):
        adjacency = adjacency.tocsr(copy=False)
        if adjacency.indptr.dtype != adjacency.indices.dtype:
            target_dtype = np.result_type(adjacency.indptr, adjacency.indices)
            adjacency.indptr = adjacency.indptr.astype(target_dtype, copy=False)
            adjacency.indices = adjacency.indices.astype(target_dtype, copy=False)
        if not np.isfinite(adjacency.data).all():
            raise ValueError(f"Network '{network_key}' contains non-finite edge weights.")
    else:
        adjacency = np.asarray(adjacency)
        if not np.isfinite(adjacency).all():
            raise ValueError(f"Network '{network_key}' contains non-finite edge weights.")

    # --- Normalize objective function string ---
    if not isinstance(objective_function, str):
        raise TypeError("`objective_function` must be a string")
    _obj_lower = objective_function.strip().lower()
    if _obj_lower == "modularity":
        objective = "modularity"
    elif _obj_lower == "cpm":
        objective = "CPM"
    else:
        raise ValueError("`objective_function` must be one of {'modularity', 'CPM'}")

    # --- Encode initial membership ---
    init_membership: Optional[np.ndarray] = None
    if initial_membership is not None:
        if isinstance(initial_membership, str):
            if initial_membership not in adata.obs:
                raise ValueError(
                    f"Initial membership key '{initial_membership}' not found in adata.obs."
                )
            _labels = np.asarray(adata.obs[initial_membership].values)
        else:
            _labels = np.asarray(initial_membership)

        _labels = _labels.reshape(-1)
        if _labels.shape[0] != adata.n_obs:
            raise ValueError(
                f"`initial_membership` length ({_labels.shape[0]}) does not match adata.n_obs ({adata.n_obs})."
            )

        if np.issubdtype(_labels.dtype, np.integer):
            _encoded = _labels.astype(np.int64, copy=False)
            if np.any(_encoded < 0):
                raise ValueError("`initial_membership` cannot contain negative labels.")
        else:
            from pandas import Categorical

            _cat = Categorical(_labels)
            _encoded = np.asarray(_cat.codes, dtype=np.int64)
            if np.any(_encoded < 0):
                raise ValueError(
                    "`initial_membership` contains missing values. Fill/drop missing values before clustering."
                )

        _, init_membership = np.unique(_encoded, return_inverse=True)
        init_membership = np.asarray(init_membership, dtype=np.int64)

    try:
        import igraph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "igraph is required for cluster_network(). "
            "Install with `pip install igraph` or `conda install -c conda-forge python-igraph`."
        ) from exc

    # Build the graph from COO edges rather than via Weighted_Adjacency, which
    # materialises a Python-level dense or slow COO roundtrip regardless of
    # sparse input.  The manual COO path is ~9x faster on realistic cell-cell
    # graphs (benchmark: 50 k cells, 30 neighbours each: 1.6 s → 0.18 s).
    n_obs = adjacency.shape[0]
    if sp.issparse(adjacency):
        coo = adjacency.tocoo(copy=False)
    else:
        coo = sp.coo_matrix(adjacency)

    # Keep only the upper triangle (including diagonal for self-loops) so that
    # each undirected edge is added exactly once.
    mask = coo.row <= coo.col
    src = coo.row[mask].tolist()
    dst = coo.col[mask].tolist()
    weights = coo.data[mask].tolist()

    # Pass zip() as a lazy iterator to avoid materialising an extra list of tuples.
    graph = igraph.Graph(n=n_obs, edges=zip(src, dst), directed=False)
    graph.es["weight"] = weights

    # --- Run Leiden with igraph random state ---
    @contextmanager
    def _igraph_rng(seed):
        if seed is None:
            yield
            return
        try:
            igraph.set_random_number_generator(random.Random(int(seed)))
            yield
        finally:
            igraph.set_random_number_generator(random)

    with _igraph_rng(random_state):
        part = graph.community_leiden(
            objective_function=objective,
            weights="weight",
            resolution=resolution_parameter,
            beta=beta,
            initial_membership=init_membership,
            n_iterations=n_iterations,
        )

    # --- Relabel clusters by size ---
    if min_size < 1:
        raise ValueError("`min_size` must be >= 1")

    _raw_labels = np.asarray(part.membership, dtype=np.int64).reshape(-1)
    if _raw_labels.size == 0:
        clusters = _raw_labels.astype(np.int32, copy=False)
    else:
        if np.any(_raw_labels < 0):
            raise ValueError("Leiden membership contains negative labels.")
        _counts = np.bincount(_raw_labels)
        _kept = np.where(_counts >= min_size)[0]
        _remap = np.zeros(_counts.shape[0], dtype=np.int32)
        if _kept.size > 0:
            _order = np.argsort(-_counts[_kept], kind="stable")
            _remap[_kept[_order]] = np.arange(1, _kept.size + 1, dtype=np.int32)
        clusters = _remap[_raw_labels]

    if return_raw:
        return clusters

    if key_added is None:
        key_added = f"leiden_{network_key}"

    params = {
        "algorithm": "leiden",
        "network_key": network_key,
        "objective_function": objective,
        "resolution_parameter": float(resolution_parameter),
        "beta": float(beta),
        "n_iterations": int(n_iterations),
        "min_size": int(min_size),
        "initial_membership": initial_membership if isinstance(initial_membership, str) else None,
        "random_state": None if random_state is None else int(random_state),
    }
    if objective == "modularity":
        params["modularity"] = float(part.modularity)

    persist_updates(
        adata,
        obs={key_added: clusters},
        uns={f"{key_added}_params": params},
    )

    if not inplace:
        return adata
    return None
