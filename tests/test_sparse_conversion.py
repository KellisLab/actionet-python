import anndata as ad
import numpy as np
import scipy.sparse as sp

import actionet as an
import actionet.pipeline as pipeline


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


def test_build_network_knn_k1_retains_nearest_neighbor():
    H = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )
    adata = ad.AnnData(np.zeros((H.shape[0], 1), dtype=np.float64))
    adata.obsm["line"] = H

    an.build_network(
        adata,
        algorithm="knn",
        distance_metric="l2",
        k=1,
        mutual_edges_only=False,
        n_threads=1,
        obsm_key="line",
        key_added="line_graph",
    )
    graph = adata.obsp["line_graph"].tocsr()

    assert graph.nnz > 0
    assert graph[0, 1] > 0


def test_build_network_knn_line_support_matches_expected_neighbors():
    H = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ],
        dtype=np.float64,
    )
    adata = ad.AnnData(np.zeros((H.shape[0], 1), dtype=np.float64))
    adata.obsm["line"] = H

    an.build_network(
        adata,
        algorithm="knn",
        distance_metric="l2",
        k=4,
        mutual_edges_only=False,
        n_threads=1,
        obsm_key="line",
        key_added="line_graph",
    )
    graph = adata.obsp["line_graph"].tocsr()

    support = set(graph[0].indices.tolist())
    assert support == {1, 2, 3, 4}


def test_run_actionet_forwards_knn_network_params(monkeypatch):
    adata = ad.AnnData(np.zeros((4, 1), dtype=np.float64))
    forwarded = {}

    def fake_run_action(adata, **kwargs):
        H = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        adata.obsm["H_stacked"] = H
        adata.obsm["H_merged"] = H.copy()

    def fake_build_network(adata, **kwargs):
        forwarded.update(kwargs)
        adata.obsp[kwargs["key_added"]] = sp.eye(adata.n_obs, format="csr")

    def fake_compute_network_diffusion(adata, **kwargs):
        adata.obsm["archetype_footprint"] = np.array(
            [
                [1.0, 0.0],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def fake_layout_network(adata, *, key_added, n_components, **kwargs):
        adata.obsm[key_added] = np.zeros((adata.n_obs, n_components), dtype=np.float64)

    def fake_compute_archetype_feature_specificity(adata, **kwargs):
        adata.varm["archetype_feat_profile"] = np.zeros((adata.n_vars, 1), dtype=np.float64)

    monkeypatch.setattr(pipeline, "run_action", fake_run_action)
    monkeypatch.setattr(pipeline, "build_network", fake_build_network)
    monkeypatch.setattr(pipeline, "compute_network_diffusion", fake_compute_network_diffusion)
    monkeypatch.setattr(pipeline, "layout_network", fake_layout_network)
    monkeypatch.setattr(
        pipeline,
        "compute_archetype_feature_specificity",
        fake_compute_archetype_feature_specificity,
    )

    pipeline.run_actionet(
        adata,
        network_algorithm="knn",
        network_M=48,
        network_k=7,
        layout_3d=False,
        n_threads=1,
        inplace=True,
    )

    assert forwarded["algorithm"] == "knn"
    assert forwarded["M"] == 48
    assert forwarded["k"] == 7
