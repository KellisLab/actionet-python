"""Within-group batch correction: design construction, golden parity, invariants.

The golden file ``tests/fixtures/batch_correction_grouped_reference.npz`` was
written by ``tests/fixtures/capture_batch_correction_reference.py`` on the build
that produced the validated prototype results (libactionet 92c0e2e). The
reduction is loaded from the file so the checks do not depend on BLAS roundoff
in ``reduce_kernel``.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp

import actionet as an

from _batch_design_reference import (
    adata_from_counts,
    notebook_make_batch_design,
    synthetic_counts,
)

REFERENCE = Path(__file__).resolve().parent / "fixtures" / "batch_correction_grouped_reference.npz"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(REFERENCE))


def _reference_adata(ref) -> ad.AnnData:
    counts = sp.csr_matrix(
        (ref["counts_data"], ref["counts_indices"], ref["counts_indptr"]),
        shape=tuple(ref["counts_shape"]),
    ).toarray()
    adata = adata_from_counts(counts, ref["cluster"], ref["stratum"], ref["batch"])
    adata.obs["donor8"] = pd.Categorical((adata.obs["donor"].cat.codes.to_numpy() % 8).astype(str))
    adata.obsm["action"] = ref["red_S_r"].copy()
    adata.obsm["action_B"] = ref["red_B"].copy()
    adata.varm["action_U"] = ref["red_U"].copy()
    adata.varm["action_A"] = ref["red_A"].copy()
    adata.uns["action_params"] = {"sigma": ref["red_sigma"].copy()}
    return adata


@pytest.fixture()
def adata(reference):
    return _reference_adata(reference)


def _sigma(adata, key="action_corrected"):
    return np.asarray(adata.uns[f"{key}_params"]["sigma"], dtype=float).ravel()


def _max_angle_deg(X, Y) -> float:
    return float(np.degrees(scipy.linalg.subspace_angles(X, Y).max()))


def _signed_frob_diff(A, B) -> float:
    signs = np.sign(np.sum(A * B, axis=0))
    signs[signs == 0] = 1.0
    return float(np.linalg.norm(A - B * signs) / np.linalg.norm(A))


def _orth(X, ref_norm=None):
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    scale = s[0] if ref_norm is None else max(ref_norm, s[0] if s.size else 0.0)
    return u[:, s > 1e-10 * scale]


def _correction_basis(S, A_prior, D):
    """Numpy mirror of the C++ correction basis: S'D projected off span([A, 1])."""
    prior = _orth(np.column_stack([A_prior, np.ones(S.shape[1])]))
    Z = S.T @ D
    Z -= prior @ (prior.T @ Z)
    Z -= prior @ (prior.T @ Z)
    return _orth(Z, ref_norm=np.linalg.norm(S.T @ D, axis=0).max())


def _brand_oracle(S_r, sigma, V, A_pert, B_pert):
    """Full Brand (2006) rank-k update of ``(S_r / sigma) diag(sigma) V'`` by ``A_pert B_pert'``.

    Uses the same, slightly non-orthonormal, ``U = S_r / sigma`` basis as the C++ path.
    """
    k = sigma.size
    U = S_r / sigma
    M = U.T @ A_pert
    P = _orth(A_pert - U @ M)
    N = V.T @ B_pert
    Q = _orth(B_pert - V @ N)
    R_P = P.T @ (A_pert - U @ M)
    R_Q = Q.T @ (B_pert - V @ N)
    K = np.block([[np.diag(sigma) + M @ N.T, M @ R_Q.T], [R_P @ N.T, R_P @ R_Q.T]])
    u, s, _ = np.linalg.svd(K)
    return s[:k], np.column_stack([U, P]) @ u[:, :k] * s[:k]


# ---------------------------------------------------------------------------
# Design construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_key", ["donor", "donor8"])
@pytest.mark.parametrize("with_depth", [False, True])
def test_design_spans_notebook_design(adata, batch_key, with_depth):
    depth = "log_counts" if with_depth else None
    notebook = notebook_make_batch_design(
        adata, cluster_key="cluster", batch_key=batch_key, strata_key="region",
        depth=adata.obs["log_counts"].to_numpy() if with_depth else None,
    )
    packaged = an.build_batch_design(
        adata, batch_key, groups="cluster", strata_key="region", depth=depth,
    )
    n_batches = adata.obs[batch_key].nunique()
    assert packaged.shape == (adata.n_obs, n_batches + with_depth)
    assert np.linalg.matrix_rank(packaged) == notebook.shape[1]
    assert _max_angle_deg(notebook, packaged) < 1e-8


def test_design_is_orthogonal_to_reduction_centring_terms(adata):
    D = an.build_batch_design(adata, "donor", groups="cluster", strata_key="region", depth="log_counts")
    C = np.column_stack([np.ones(adata.n_obs), adata.obsm["action_B"]])
    assert np.abs(C.T @ D).max() < 1e-10 * np.abs(D).max() * adata.n_obs

    raw = an.build_batch_design(
        adata, "donor", groups="cluster", strata_key="region", decorrelate_from_reduction=False,
    )
    groups = adata.obs["cluster"].astype(str) + "|" + adata.obs["region"].astype(str)
    group_sums = pd.DataFrame(raw).groupby(groups.to_numpy()).sum().to_numpy()
    assert np.abs(group_sums).max() < 1e-10


def test_array_and_key_inputs_agree(adata):
    from_key = an.build_batch_design(adata, "donor", groups="cluster", depth="log_counts")
    from_array = an.build_batch_design(
        adata, "donor",
        groups=adata.obs["cluster"].to_numpy(),
        depth=adata.obs["log_counts"].to_numpy(),
    )
    np.testing.assert_allclose(from_key, from_array)


def test_dominated_cluster_merges_into_nearest_centroid():
    rng = np.random.default_rng(3)
    n_per = 60
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [10.5, 0.5]])
    cluster = np.repeat(["A", "B", "C"], n_per)
    X = np.vstack([c + rng.normal(scale=0.2, size=(n_per, 2)) for c in centres])
    batch = np.tile(["b0", "b1", "b2", "b3"], 3 * n_per // 4)
    batch[cluster == "C"] = "b0"
    adata = ad.AnnData(
        X=sp.csr_matrix(rng.poisson(1.0, size=(3 * n_per, 20)).astype(float)),
        obs=pd.DataFrame({"cluster": cluster, "batch": batch}, index=[f"c{i}" for i in range(3 * n_per)]),
    )
    adata.obsm["action"] = X
    adata.obsm["action_B"] = rng.normal(size=(3 * n_per, 2))

    D = an.build_batch_design(adata, "batch", groups="cluster", key_added="merged")
    assert D.shape == (3 * n_per, 4)
    merged = adata.obs["merged"].astype(str).to_numpy()
    assert set(merged[cluster == "C"]) == {"B"}
    np.testing.assert_array_equal(merged[cluster != "C"], cluster[cluster != "C"])

    unmerged = an.build_batch_design(adata, "batch", groups="cluster", max_batch_frac=None, key_added="raw")
    np.testing.assert_array_equal(adata.obs["raw"].astype(str).to_numpy(), cluster)
    assert not np.allclose(D, unmerged)

    with pytest.raises(ValueError, match="must exceed 1/n_batches"):
        an.build_batch_design(adata, "batch", groups="cluster", max_batch_frac=0.25)

    adata.obs["batch"] = np.repeat(["b0", "b1", "b2"], n_per)
    with pytest.raises(ValueError, match="Every group is dominated"):
        an.build_batch_design(adata, "batch", groups="cluster", max_batch_frac=0.5)


def test_design_validation_errors(adata):
    with pytest.raises(ValueError, match="not found in adata.obs"):
        an.build_batch_design(adata, "missing")
    with pytest.raises(ValueError, match="only applies when groups='auto'"):
        an.build_batch_design(adata, "donor", groups="cluster", group_kwargs={"k": 10})
    with pytest.raises(ValueError, match="length"):
        an.build_batch_design(adata, "donor", groups=np.zeros(3))
    bad_depth = adata.obs["log_counts"].to_numpy().copy()
    bad_depth[0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        an.build_batch_design(adata, "donor", depth=bad_depth)
    adata.obs.loc[adata.obs_names[0], "cluster"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        an.build_batch_design(adata, "donor", groups="cluster")


# ---------------------------------------------------------------------------
# Golden parity and the full-Brand update
# ---------------------------------------------------------------------------


def test_notebook_design_reproduces_golden(adata, reference):
    """The validated configuration (20 donors, QR path) is unchanged by the C++ rewrite."""
    design = notebook_make_batch_design(adata, cluster_key="cluster", batch_key="donor", strata_key="region")
    assert design.shape[1] == int(reference["b20_n_design_cols"])
    an.correct_batch_effect(adata, design=design)
    np.testing.assert_allclose(_sigma(adata), reference["b20_sigma"], rtol=1e-8)
    assert _signed_frob_diff(reference["b20_S_r"], adata.obsm["action_corrected"]) < 1e-8
    assert _signed_frob_diff(reference["b20_U"], adata.varm["action_corrected_U"]) < 1e-8


def test_grouped_correction_reproduces_golden(adata, reference):
    an.correct_batch_effect(adata, batch_key="donor", groups="cluster", strata_key="region")
    np.testing.assert_allclose(_sigma(adata), reference["b20_sigma"], rtol=1e-8)
    assert _signed_frob_diff(reference["b20_S_r"], adata.obsm["action_corrected"]) < 1e-8

    params = adata.uns["action_corrected_params"]
    assert params["correction_rank"] == 19
    grouped = params["grouped_design"]
    assert grouped["groups"] == "cluster"
    assert grouped["strata_key"] == "region"
    assert grouped["n_groups"] == 8
    assert grouped["n_batches"] == 20
    assert grouped["merged_groups"] == {}
    np.testing.assert_array_equal(
        adata.obs["action_corrected_groups"].astype(str).to_numpy(),
        adata.obs["cluster"].astype(str).to_numpy(),
    )


@pytest.mark.parametrize("tag,batch_key", [("b20", "donor"), ("b8", "donor8")])
def test_correction_is_full_brand_update(adata, reference, tag, batch_key):
    """Few batches used to discard the cell-side residual; both regimes now apply the full update."""
    design = an.build_batch_design(adata, batch_key, groups="cluster", strata_key="region")
    S = adata.X.toarray()
    Z = _correction_basis(S, reference["red_A"], design)
    sigma_o, S_r_o = _brand_oracle(reference["red_S_r"], reference["red_sigma"], reference["red_U"], -S @ Z, Z)

    an.correct_batch_effect(adata, design=design)
    np.testing.assert_allclose(_sigma(adata), sigma_o, rtol=1e-8)
    assert _signed_frob_diff(S_r_o, adata.obsm["action_corrected"]) < 1e-6

    if tag == "b8":
        assert np.abs(reference["b8_sigma"] - sigma_o).max() > 1.0


def test_history_columns_avoid_prior_directions(adata, reference):
    a1 = reference["red_A"][:, 0]
    ones = np.ones(adata.n_vars)
    S = adata.X.toarray()
    for kwargs in ({"batch_key": "donor"}, {"batch_key": "donor8", "groups": "cluster", "depth": "log_counts"}):
        work = adata.copy()
        an.correct_batch_effect(work, **kwargs)
        Z = work.varm["action_corrected_A"][:, 2:]
        B = work.obsm["action_corrected_B"][:, 2:]
        np.testing.assert_allclose(work.varm["action_corrected_A"][:, :2], reference["red_A"])
        np.testing.assert_allclose(Z.T @ Z, np.eye(Z.shape[1]), atol=1e-10)
        assert np.abs(Z.T @ a1).max() < 1e-12
        assert np.abs(Z.T @ ones).max() < 1e-12
        np.testing.assert_allclose(B, -S @ Z, atol=1e-10)


def test_rank_deficient_design_keeps_depth_column(adata, reference):
    """Centred one-hots are rank deficient; the dependent column must not drop later ones."""
    without = an.build_batch_design(adata, "donor8", groups="cluster")
    with_depth = an.build_batch_design(adata, "donor8", groups="cluster", depth="log_counts")
    assert np.abs(with_depth[:, :8].sum(axis=1)).max() < 1e-10

    work = adata.copy()
    an.correct_batch_effect(work, design=with_depth)
    assert work.uns["action_corrected_params"]["correction_rank"] == 8

    S = adata.X.toarray()
    z_depth = _correction_basis(S, reference["red_A"], with_depth[:, [-1]])
    Z = work.varm["action_corrected_A"][:, 2:]
    residual = z_depth - Z @ (Z.T @ z_depth)
    assert np.linalg.norm(residual) < 1e-8

    base = adata.copy()
    an.correct_batch_effect(base, design=without)
    assert base.uns["action_corrected_params"]["correction_rank"] == 7


def test_plain_batch_key_matches_dense_design(adata):
    labels = adata.copy()
    an.correct_batch_effect(labels, batch_key="donor")
    dense = adata.copy()
    an.correct_batch_effect(dense, design=pd.get_dummies(adata.obs["donor"]).to_numpy(dtype=float))
    np.testing.assert_allclose(_sigma(labels), _sigma(dense), rtol=1e-10)
    assert "grouped_design" not in labels.uns["action_corrected_params"]
    assert "action_corrected_groups" not in labels.obs


def test_design_inside_prior_span_leaves_reduction_unchanged(adata):
    design = np.ones((adata.n_obs, 1))
    an.correct_batch_effect(adata, design=design)
    np.testing.assert_array_equal(adata.obsm["action_corrected"], adata.obsm["action"])
    assert adata.uns["action_corrected_params"]["correction_rank"] == 0


# ---------------------------------------------------------------------------
# correct_batch_effect integration
# ---------------------------------------------------------------------------


def test_grouped_arguments_require_batch_key(adata):
    design = np.ones((adata.n_obs, 1))
    with pytest.raises(ValueError, match="require 'batch_key'"):
        an.correct_batch_effect(adata, design=design, groups="cluster")
    with pytest.raises(ValueError, match="only applies when groups='auto'"):
        an.correct_batch_effect(adata, batch_key="donor", group_kwargs={"k": 10})


def test_estimate_batch_groups_recovers_clusters(adata):
    labels = an.estimate_batch_groups(adata)
    assert "action" in adata.obsm and len(adata.obsp) == 0
    table = pd.crosstab(adata.obs["cluster"].to_numpy(), labels)
    assert table.shape == (4, 4)
    assert ((table > 0).sum(axis=1) == 1).all()

    an.estimate_batch_groups(adata, key_added="crude")
    np.testing.assert_array_equal(adata.obs["crude"].to_numpy(), labels)


def test_auto_groups_in_correct_batch_effect(adata, reference):
    an.correct_batch_effect(
        adata, batch_key="donor", groups="auto", strata_key="region",
        group_kwargs={"k": 20, "random_state": 1},
    )
    assert len(adata.obsp) == 0
    assert adata.uns["action_corrected_params"]["grouped_design"]["groups"] == "auto"
    np.testing.assert_allclose(_sigma(adata), reference["b20_sigma"], rtol=1e-8)


def test_inplace_false_returns_copy(adata):
    out = an.correct_batch_effect(adata, batch_key="donor", groups="cluster", inplace=False)
    assert "action_corrected" in out.obsm and "action_corrected_groups" in out.obs
    assert "action_corrected" not in adata.obsm


def test_backed_grouped_matches_in_memory(tmp_path):
    counts, cluster, stratum, batch = synthetic_counts(n_cells=240, n_genes=80, n_batches=12, seed=4)
    adata = adata_from_counts(counts, cluster, stratum, batch)
    an.reduce_kernel(adata, n_components=8, verbose=False)
    path = tmp_path / "grouped.h5ad"
    adata.write_h5ad(path)

    an.correct_batch_effect(adata, batch_key="donor", groups="auto", strata_key="region", depth="log_counts")

    backed = ad.read_h5ad(path, backed="r+")
    try:
        an.correct_batch_effect(
            backed, batch_key="donor", groups="auto", strata_key="region", depth="log_counts",
            backed_chunk_size=64,
        )
        np.testing.assert_allclose(_sigma(backed), _sigma(adata), rtol=1e-8)
        assert _signed_frob_diff(adata.obsm["action_corrected"], np.asarray(backed.obsm["action_corrected"])) < 1e-6
    finally:
        backed.file.close()

    reopened = ad.read_h5ad(path)
    assert len(reopened.obsp) == 0
    assert "action_corrected_groups" in reopened.obs
    params = reopened.uns["action_corrected_params"]
    assert params["grouped_design"]["depth"] == "log_counts"
    assert params["grouped_design"]["strata_key"] == "region"
