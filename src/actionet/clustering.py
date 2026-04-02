"""Clustering helpers and high-level clustering APIs."""

from contextlib import contextmanager
import random
from typing import Literal, Optional, Union

from anndata import AnnData
import numpy as np
import scipy.sparse as sp

from ._backed_persist import persist_updates


def _normalize_leiden_objective(objective_function: str) -> str:
    if not isinstance(objective_function, str):
        raise TypeError("`objective_function` must be a string")

    objective = objective_function.strip().lower()
    if objective == "modularity":
        return "modularity"
    if objective == "cpm":
        return "CPM"

    raise ValueError("`objective_function` must be one of {'modularity', 'CPM'}")


def _encode_initial_membership(
    adata: AnnData,
    initial_membership: Optional[Union[str, np.ndarray]],
) -> Optional[np.ndarray]:
    if initial_membership is None:
        return None

    if isinstance(initial_membership, str):
        if initial_membership not in adata.obs:
            raise ValueError(
                f"Initial membership key '{initial_membership}' not found in adata.obs."
            )
        labels = np.asarray(adata.obs[initial_membership].values)
    else:
        labels = np.asarray(initial_membership)

    labels = labels.reshape(-1)
    if labels.shape[0] != adata.n_obs:
        raise ValueError(
            f"`initial_membership` length ({labels.shape[0]}) does not match adata.n_obs ({adata.n_obs})."
        )

    if np.issubdtype(labels.dtype, np.integer):
        encoded = labels.astype(np.int64, copy=False)
        if np.any(encoded < 0):
            raise ValueError("`initial_membership` cannot contain negative labels.")
    else:
        from pandas import Categorical

        cat = Categorical(labels)
        encoded = np.asarray(cat.codes, dtype=np.int64)
        if np.any(encoded < 0):
            raise ValueError(
                "`initial_membership` contains missing values. Fill/drop missing values before clustering."
            )

    # Compress arbitrary integer IDs into dense 0..K-1 labels for igraph.
    _, encoded = np.unique(encoded, return_inverse=True)
    return np.asarray(encoded, dtype=np.int64)


def _relabel_clusters_by_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size < 1:
        raise ValueError("`min_size` must be >= 1")

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return labels.astype(np.int32, copy=False)
    if np.any(labels < 0):
        raise ValueError("Leiden membership contains negative labels.")

    counts = np.bincount(labels)
    kept = np.where(counts >= min_size)[0]
    remap = np.zeros(counts.shape[0], dtype=np.int32)

    if kept.size > 0:
        order = np.argsort(-counts[kept], kind="stable")
        remap[kept[order]] = np.arange(1, kept.size + 1, dtype=np.int32)

    return remap[labels]


@contextmanager
def _set_igraph_random_state(random_state: Optional[int]):
    if random_state is None:
        yield
        return

    import igraph

    try:
        igraph.set_random_number_generator(random.Random(int(random_state)))
        yield
    finally:
        igraph.set_random_number_generator(random)


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

    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found in adata.obsp.")

    adjacency = adata.obsp[network_key]
    if adjacency.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(
            f"Network '{network_key}' must be shape ({adata.n_obs}, {adata.n_obs}), "
            f"got {adjacency.shape}."
        )

    if sp.issparse(adjacency):
        adjacency = adjacency.tocsr(copy=False)
        if not np.isfinite(adjacency.data).all():
            raise ValueError(f"Network '{network_key}' contains non-finite edge weights.")
    else:
        adjacency = np.asarray(adjacency)
        if not np.isfinite(adjacency).all():
            raise ValueError(f"Network '{network_key}' contains non-finite edge weights.")

    objective = _normalize_leiden_objective(objective_function)
    init_membership = _encode_initial_membership(adata, initial_membership)

    try:
        import igraph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "igraph is required for cluster_network(). "
            "Install with `pip install igraph` or `conda install -c conda-forge python-igraph`."
        ) from exc

    try:
        graph = igraph.Graph.Weighted_Adjacency(
            adjacency,
            mode="max",
            attr="weight",
            loops="once",
        )
    except TypeError:
        graph = igraph.Graph.Weighted_Adjacency(
            adjacency,
            mode="max",
            attr="weight",
            loops=True,
        )

    with _set_igraph_random_state(random_state):
        part = graph.community_leiden(
            objective_function=objective,
            weights="weight",
            resolution=resolution_parameter,
            beta=beta,
            initial_membership=init_membership,
            n_iterations=n_iterations,
        )

    clusters = _relabel_clusters_by_size(
        np.asarray(part.membership, dtype=np.int64),
        min_size=min_size,
    )

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

