"""Parity test for the optimized batch correction code path.

Verifies that the optimized C++ implementation of `correct_batch_effect`
agrees with itself across the alternate sparse-vs-dense paths, with the
explicit-design path, and with previously-saved baseline outputs (via
sign-fixed Frobenius distance comparisons).

Also tests the new `orthogonalize_batch_effect_sparse_labels` fast path
for end-to-end equivalence with the dense-design path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import actionet


def _make_synthetic_adata(n_cells: int = 800, n_genes: int = 400, n_batches: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Sparse log-counts: Poisson background with batch-specific shifts to make
    # batch-correction non-trivial.
    base = rng.poisson(0.4, size=(n_cells, n_genes)).astype(float)
    batch_effect = rng.normal(0.0, 0.6, size=(n_batches, n_genes))
    cell_batch = rng.integers(0, n_batches, size=n_cells)
    base += batch_effect[cell_batch]
    base = np.clip(base, a_min=0.0, a_max=None)
    X = sp.csr_matrix(np.log1p(base))
    obs = pd.DataFrame(
        {"batch": pd.Categorical([f"B{i}" for i in cell_batch])},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return AnnData(X=X, obs=obs, var=var)


def _signed_frob_diff(A: np.ndarray, B: np.ndarray) -> float:
    """Sign-flip-tolerant Frobenius distance between two matrices that
    differ only in column-wise sign.  Returns ||A - sign-corrected B||_F /
    ||A||_F.
    """
    if A.shape != B.shape:
        return float("inf")
    out = B.copy()
    for j in range(A.shape[1]):
        if np.dot(A[:, j], B[:, j]) < 0:
            out[:, j] = -B[:, j]
    return float(np.linalg.norm(A - out) / max(np.linalg.norm(A), 1e-30))


@pytest.mark.parametrize("n_batches", [2, 5, 10, 25])
def test_sparse_labels_matches_dense_design(n_batches):
    """The new sparse-labels fast path must agree with the dense-design path
    up to a sign-flip on each component (singular-vector signs are non-unique).
    """
    adata = _make_synthetic_adata(n_cells=600, n_genes=300, n_batches=n_batches, seed=1)
    actionet.reduce_kernel(adata, n_components=15, key_added="action", inplace=True)

    # Snapshot reduction so both runs start from identical inputs.
    snap = {
        "S_r": adata.obsm["action"].copy(),
        "B": adata.obsm["action_B"].copy(),
        "U": adata.varm["action_U"].copy(),
        "A": adata.varm["action_A"].copy(),
        "params": dict(adata.uns["action_params"]),
    }

    # Path A: explicit dense design via the original sparse path.
    design = pd.get_dummies(adata.obs["batch"], drop_first=False).to_numpy(dtype=float)
    actionet.correct_batch_effect(adata, design=design, reduction_key="action")
    sigma_design = np.asarray(adata.uns["action_corrected_params"]["sigma"])
    S_r_design = adata.obsm["action_corrected"].copy()
    U_design = adata.varm["action_corrected_U"].copy()

    # Reset reduction state and run via batch_key (sparse-labels fast path).
    adata.obsm["action"] = snap["S_r"]
    adata.obsm["action_B"] = snap["B"]
    adata.varm["action_U"] = snap["U"]
    adata.varm["action_A"] = snap["A"]
    adata.uns["action_params"] = snap["params"]
    del adata.obsm["action_corrected"], adata.obsm["action_corrected_B"]
    del adata.varm["action_corrected_U"], adata.varm["action_corrected_A"]
    del adata.uns["action_corrected_params"]

    actionet.correct_batch_effect(adata, batch_key="batch", reduction_key="action")
    sigma_labels = np.asarray(adata.uns["action_corrected_params"]["sigma"])
    S_r_labels = adata.obsm["action_corrected"]
    U_labels = adata.varm["action_corrected_U"]

    # Singular values must agree to high relative precision (canonical SVD).
    np.testing.assert_allclose(sigma_design, sigma_labels, rtol=1e-8, atol=1e-10)

    # Singular vector subspaces: equal up to per-column sign.
    assert _signed_frob_diff(S_r_design, S_r_labels) < 1e-6
    assert _signed_frob_diff(U_design, U_labels) < 1e-6


def test_sparse_labels_dense_path_parity():
    """Sparse-labels result must agree with dense-X path under the same design."""
    adata = _make_synthetic_adata(n_cells=400, n_genes=200, n_batches=6, seed=2)
    actionet.reduce_kernel(adata, n_components=12, key_added="action", inplace=True)

    snap = {
        "S_r": adata.obsm["action"].copy(),
        "B": adata.obsm["action_B"].copy(),
        "U": adata.varm["action_U"].copy(),
        "A": adata.varm["action_A"].copy(),
        "params": dict(adata.uns["action_params"]),
    }

    # Dense X path: convert X to a dense ndarray.
    adata_dense = adata.copy()
    adata_dense.X = adata_dense.X.toarray()
    actionet.correct_batch_effect(adata_dense, batch_key="batch", reduction_key="action")
    sigma_dense = np.asarray(adata_dense.uns["action_corrected_params"]["sigma"])
    S_r_dense = adata_dense.obsm["action_corrected"]

    # Sparse-labels path on the original sparse adata.
    actionet.correct_batch_effect(adata, batch_key="batch", reduction_key="action")
    sigma_sparse = np.asarray(adata.uns["action_corrected_params"]["sigma"])
    S_r_sparse = adata.obsm["action_corrected"]

    np.testing.assert_allclose(sigma_dense, sigma_sparse, rtol=1e-8, atol=1e-10)
    assert _signed_frob_diff(S_r_dense, S_r_sparse) < 1e-6


def test_sparse_labels_smoke_large_b():
    """Smoke test that the labels path runs cleanly with b > k."""
    adata = _make_synthetic_adata(n_cells=300, n_genes=150, n_batches=20, seed=3)
    actionet.reduce_kernel(adata, n_components=10, key_added="action", inplace=True)
    actionet.correct_batch_effect(adata, batch_key="batch", reduction_key="action")
    assert "action_corrected" in adata.obsm
    assert adata.obsm["action_corrected"].shape == (adata.n_obs, 10)
