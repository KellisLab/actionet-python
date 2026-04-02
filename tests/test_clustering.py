import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import actionet as an

pytest.importorskip("igraph")


def _block_diagonal_graph(block_sizes: list[int]) -> sp.csr_matrix:
    n = int(np.sum(block_sizes))
    graph = sp.lil_matrix((n, n), dtype=np.float64)
    start = 0
    for size in block_sizes:
        stop = start + size
        graph[start:stop, start:stop] = 1.0
        start = stop
    return graph.tocsr()


def test_cluster_network_return_raw_does_not_modify_obs():
    adata = ad.AnnData(np.zeros((6, 1), dtype=np.float64))
    adata.obsp["actionet"] = _block_diagonal_graph([3, 2, 1])

    labels = an.cluster_network(
        adata,
        network_key="actionet",
        objective_function="modularity",
        min_size=1,
        return_raw=True,
    )

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (adata.n_obs,)
    assert labels.dtype == np.int32
    assert "leiden_actionet" not in adata.obs

    counts = np.bincount(labels)
    assert counts[1:].tolist() == [3, 2, 1]


def test_cluster_network_writes_obs_and_applies_min_size():
    adata = ad.AnnData(np.zeros((6, 1), dtype=np.float64))
    adata.obsp["actionet"] = _block_diagonal_graph([4, 1, 1])

    an.cluster_network(
        adata,
        network_key="actionet",
        key_added="leiden_fast",
        min_size=2,
    )

    labels = np.asarray(adata.obs["leiden_fast"])
    assert np.sum(labels == 1) == 4
    assert np.sum(labels == 0) == 2
    assert "leiden_fast_params" in adata.uns


def test_cluster_network_initial_membership_key_and_array_match():
    adata = ad.AnnData(np.zeros((6, 1), dtype=np.float64))
    adata.obsp["actionet"] = _block_diagonal_graph([3, 2, 1])
    seed_labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    adata.obs["init"] = seed_labels

    labels_from_key = an.cluster_network(
        adata,
        initial_membership="init",
        min_size=1,
        random_state=0,
        return_raw=True,
    )
    labels_from_array = an.cluster_network(
        adata,
        initial_membership=seed_labels,
        min_size=1,
        random_state=0,
        return_raw=True,
    )

    np.testing.assert_array_equal(labels_from_key, labels_from_array)


def test_cluster_network_validates_inputs():
    adata = ad.AnnData(np.zeros((4, 1), dtype=np.float64))
    adata.obsp["actionet"] = _block_diagonal_graph([2, 2])

    with pytest.raises(ValueError, match="not found"):
        an.cluster_network(adata, network_key="missing")

    with pytest.raises(ValueError, match="length"):
        an.cluster_network(adata, initial_membership=np.array([0, 1, 2]))

    adata.obs["with_na"] = np.array([0, 0, np.nan, 1], dtype=object)
    with pytest.raises(ValueError, match="missing values"):
        an.cluster_network(adata, initial_membership="with_na")
