"""Reference copy of the ``new_batch_corr.qmd`` prototype and a synthetic generator.

``notebook_make_batch_design`` is kept verbatim from the prototype so the
packaged :func:`actionet.build_batch_design` can be checked against the design
that produced the validated atlas results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def notebook_make_batch_design(
    adata,
    cluster_key,
    batch_key="Donor",
    strata_key=None,
    depth=None,
    reduction_key="action",
    protect_centroids=False,
    max_batch_frac=0.3,
    tol=1e-8,
):
    X = np.asarray(adata.obsm[reduction_key], dtype=float)
    B_prior = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float)
    A_prior = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float)
    if not np.allclose(A_prior[:, 1], 1.0):
        raise ValueError(f"'{reduction_key}' does not carry reduce_kernel centring terms in its first two A/B columns.")

    clusters = adata.obs[cluster_key].astype(str).to_numpy().copy()
    batches = adata.obs[batch_key].astype(str).to_numpy()

    top_frac = pd.crosstab(clusters, batches, normalize="index").max(axis=1)
    keep = top_frac.index[top_frac <= max_batch_frac]
    if len(keep) == 0:
        raise ValueError("Every cluster is dominated by one batch; lower the resolution or raise max_batch_frac.")
    centroids = pd.DataFrame(X).groupby(clusters).mean()
    for c in top_frac.index.difference(keep):
        clusters[clusters == c] = ((centroids.loc[keep] - centroids.loc[c]) ** 2).sum(axis=1).idxmin()

    groups = clusters if strata_key is None else clusters + "|" + adata.obs[strata_key].astype(str).to_numpy()
    g_codes, _ = pd.factorize(groups)

    def center(M):
        M = np.asarray(M, dtype=float).reshape(len(g_codes), -1)
        return M - pd.DataFrame(M).groupby(g_codes).transform("mean").to_numpy()

    D = center(pd.get_dummies(batches, dtype=float).to_numpy())
    if depth is not None:
        D = np.hstack([D, center(depth)])

    s = -B_prior[:, 0]
    r = A_prior[:, 0].mean() * s - B_prior[:, 1]
    constraints = [np.ones(len(g_codes)), s, r]
    if protect_centroids:
        G = pd.get_dummies(g_codes, dtype=float).to_numpy()
        constraints.append(X @ (X.T @ G / G.sum(axis=0)))
    C, _ = np.linalg.qr(np.column_stack(constraints))
    D -= C @ (C.T @ D)

    evals, evecs = np.linalg.eigh(D.T @ D)
    keep = evals > tol * evals.max()
    return D @ (evecs[:, keep] / np.sqrt(evals[keep])) * np.sqrt(len(g_codes))


def synthetic_counts(
    n_cells: int = 480,
    n_genes: int = 160,
    n_clusters: int = 4,
    n_strata: int = 2,
    n_batches: int = 20,
    seed: int = 0,
):
    """Poisson counts with cluster, stratum, batch, and per-cell depth effects.

    Cells are assigned to clusters independently of batch so no batch dominates
    a cluster. Batches carry both a gene-wise multiplicative effect and a
    shifted depth distribution.
    """
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, size=n_cells)
    stratum = rng.integers(0, n_strata, size=n_cells)
    batch = rng.integers(0, n_batches, size=n_cells)

    cluster_rate = np.exp(rng.normal(0.0, 1.0, size=(n_clusters, n_genes))) * 0.4
    stratum_mult = np.exp(rng.normal(0.0, 0.3, size=(n_strata, n_genes)))
    batch_mult = np.exp(rng.normal(0.0, 0.4, size=(n_batches, n_genes)))
    batch_depth = rng.normal(0.0, 0.3, size=n_batches)
    depth = np.exp(rng.normal(0.0, 0.25, size=n_cells) + batch_depth[batch])

    rate = cluster_rate[cluster] * stratum_mult[stratum] * batch_mult[batch] * depth[:, None]
    counts = rng.poisson(rate).astype(np.int32)
    return counts, cluster, stratum, batch


def adata_from_counts(counts, cluster, stratum, batch) -> AnnData:
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    X = np.log1p(counts / totals * np.median(totals))
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical([f"C{i}" for i in cluster]),
            "region": pd.Categorical([f"R{i}" for i in stratum]),
            "donor": pd.Categorical([f"D{i:02d}" for i in batch]),
            "log_counts": np.log1p(np.asarray(totals).ravel()),
        },
        index=[f"c{i}" for i in range(counts.shape[0])],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(counts.shape[1])])
    return AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
