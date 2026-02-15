"""Shared fixtures and helpers for backed-mode tests."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pytest


def make_test_adata(
    n_cells: int = 96,
    n_genes: int = 72,
    sparse_fmt: str = "csr",
    seed: int = 13,
) -> ad.AnnData:
    """Create a small synthetic AnnData suitable for end-to-end tests.

    Three cell types (CT_0, CT_1, CT_2) with enriched expression in
    disjoint 6-gene blocks, plus Poisson background.  Two batches (B0, B1).
    """
    rng = np.random.default_rng(seed)

    labels = np.array([f"CT_{i}" for i in (np.arange(n_cells) % 3)])
    batches = np.array(["B0" if i % 2 == 0 else "B1" for i in range(n_cells)])

    X = rng.poisson(0.2, size=(n_cells, n_genes)).astype(np.float64)
    n_enriched = min(6, n_genes // 3)
    for ct in range(3):
        rows = np.where(labels == f"CT_{ct}")[0]
        col_start = ct * n_enriched
        col_end = min(col_start + n_enriched, n_genes)
        if col_start < n_genes and col_end > col_start:
            cols = np.arange(col_start, col_end)
            X[np.ix_(rows, cols)] += rng.poisson(3.0, size=(rows.size, cols.size))

    if sparse_fmt == "csr":
        X_mat = sp.csr_matrix(X)
    elif sparse_fmt == "csc":
        X_mat = sp.csc_matrix(X)
    else:
        X_mat = X

    obs = pd.DataFrame({"CellLabel": labels, "batch": batches})
    var_names = np.array([f"G{i}" for i in range(n_genes)], dtype=object)
    var = pd.DataFrame({"Gene": var_names})

    adata = ad.AnnData(X=X_mat, obs=obs, var=var)
    adata.obs_names = np.array([f"cell_{i}" for i in range(n_cells)], dtype=object)
    adata.var_names = var_names
    adata.layers["logcounts"] = X_mat.copy()
    return adata


def open_backed(tmp_path, adata: ad.AnnData) -> ad.AnnData:
    """Write *adata* to an h5ad file and re-open in backed ``r+`` mode."""
    path = tmp_path / "test_backed.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r+")


class MatrixLike:
    """Small helper to normalise backed-matrix ``.sum()`` return types."""

    def __init__(self, X):
        self.X = X

    def sum(self, axis=0):
        out = self.X.sum(axis=axis)
        return out.A if hasattr(out, "A") else out
