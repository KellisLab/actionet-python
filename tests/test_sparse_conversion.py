import anndata as ad
import numpy as np
import scipy.sparse as sp

import actionet as an


def test_aggregate_matrix_sparse_output_is_csr_and_correct():
    X = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0, 4.0],
                [5.0, 0.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    group_vec = np.array(["g1", "g1", "g2", "g3"], dtype=object)

    result = an.aggregate_matrix(X, group_vec, dim=1, method="sum", return_sparse=True)
    expected = sp.csr_matrix(
        np.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    assert sp.isspmatrix_csr(result)
    assert result.has_sorted_indices
    np.testing.assert_allclose(result.toarray(), expected.toarray())


def test_build_network_returns_valid_csr_matrix():
    H = np.array(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.05, 0.95],
        ],
        dtype=np.float64,
    )
    adata = ad.AnnData(np.zeros((H.shape[0], 1), dtype=np.float64))
    adata.obsm["H_stacked"] = H

    an.build_network(
        adata,
        algorithm="knn",
        distance_metric="l2",
        k=2,
        mutual_edges_only=False,
        n_threads=1,
        obsm_key="H_stacked",
        key_added="test_graph",
    )
    graph = adata.obsp["test_graph"]

    assert sp.isspmatrix_csr(graph)
    assert graph.shape == (H.shape[0], H.shape[0])
    assert graph.nnz > 0
    assert graph.has_sorted_indices
    assert np.isfinite(graph.data).all()
