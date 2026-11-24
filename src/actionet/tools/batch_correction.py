"""Batch correction and normalization utilities."""

from typing import Any, Literal, Optional, Union
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from anndata import AnnData
import pandas as pd

from .. import _core
from .anndata import anndata_to_matrix
from ..io.persist import persist_updates
from ..io.matrix_source import MatrixSource
from ..io.operator import open_backed_operator_for
from ..io.lazy_transform import (
    LazyTransform,
    _lazy_params_for_metadata,
    _resolve_lazy_backed_transform,
    _validate_lazy_transform,
)


def _load_reduction_state(adata: AnnData, reduction_key: str) -> tuple[np.ndarray, ...]:
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    params_key = f"{reduction_key}_params"
    if params_key not in adata.uns:
        raise ValueError(f"Parameters '{params_key}' not found. Run reduce_kernel first.")

    params = adata.uns[params_key]
    old_S_r = np.asarray(adata.obsm[reduction_key], dtype=float, order="C")  # cells x k
    old_U = np.asarray(adata.varm[f"{reduction_key}_U"], dtype=float, order="C")
    old_A = np.asarray(adata.varm[f"{reduction_key}_A"], dtype=float, order="C")
    old_B = np.asarray(adata.obsm[f"{reduction_key}_B"], dtype=float, order="C")
    old_sigma = np.asarray(params["sigma"], dtype=float).reshape(-1)

    if old_S_r.shape[0] != adata.n_obs:
        raise ValueError("Reduction matrix has unexpected shape; expected cells x components.")
    if old_sigma.shape[0] != old_S_r.shape[1]:
        raise ValueError("Size of 'sigma' does not match number of components in reduction.")

    return old_S_r, old_U, old_A, old_B, old_sigma


# Matches kRankRelTol in libactionet utils_decomp.hpp.
_RANK_REL_TOL = 1e-10


def _factorize_labels(values, name: str, n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode labels as dense int64 codes over the categories actually present."""
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != n_obs:
        raise ValueError(f"{name} has length {values.shape[0]}; expected {n_obs} (adata.n_obs).")
    cat = pd.Categorical(values).remove_unused_categories()
    codes = np.asarray(cat.codes, dtype=np.int64)
    if np.any(codes < 0):
        raise ValueError(f"{name} contains missing values.")
    return codes, np.asarray(cat.categories)


def _obs_labels(adata: AnnData, key: str, param: str) -> tuple[np.ndarray, np.ndarray]:
    if key not in adata.obs:
        raise ValueError(f"{param} '{key}' not found in adata.obs")
    return _factorize_labels(adata.obs[key].to_numpy(), f"adata.obs['{key}']", adata.n_obs)


def _group_means(codes: np.ndarray, n_groups: int, X: np.ndarray) -> np.ndarray:
    """Row means of ``X`` (n x p) within each group code; returns (n_groups x p)."""
    sizes = np.bincount(codes, minlength=n_groups).astype(float)
    sums = np.column_stack(
        [np.bincount(codes, weights=X[:, j], minlength=n_groups) for j in range(X.shape[1])]
    )
    return sums / sizes[:, None]


def estimate_batch_groups(
    adata: AnnData,
    reduction_key: str = "action",
    k: int = 30,
    resolution: float = 0.05,
    distance_metric: Literal["l2", "ip", "jsd"] = "l2",
    n_threads: int = 0,
    random_state: Optional[int] = 0,
    key_added: Optional[str] = None,
) -> np.ndarray:
    """Estimate coarse cell groups for within-group batch correction.

    Builds an unconstrained kNN graph on ``adata.obsm[reduction_key]`` and
    clusters it with Leiden at low resolution. The graph is discarded; nothing
    is written to ``adata.obsp``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed reduction (in-memory or backed).
    reduction_key : str
        Key in ``adata.obsm`` with the cell embedding, usually the uncorrected
        ``reduce_kernel`` output.
    k : int
        Number of nearest neighbours per cell.
    resolution : float
        Leiden resolution. Low values give a few broad groups, which is the
        intended granularity: groups should separate cell classes, not batches.
    distance_metric : {"l2", "ip", "jsd"}
        Distance used for the kNN search.
    n_threads : int
        Number of threads for the kNN search (0 = auto).
    random_state : int or None
        Seed for Leiden.
    key_added : str or None
        If given, store the labels in ``adata.obs[key_added]``.

    Returns
    -------
    ndarray of int32
        Group label per cell. Groups are numbered by decreasing size from 1;
        groups smaller than 3 cells share label 0.
    """
    from ..network.build import _build_network_matrix
    from ..network.cluster import _leiden_membership

    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    graph = _build_network_matrix(
        adata.obsm[reduction_key],
        algorithm="knn",
        distance_metric=distance_metric,
        n_threads=n_threads,
        mutual_edges_only=False,
        k=k,
        label=f"obsm['{reduction_key}']",
    )
    labels, _ = _leiden_membership(
        graph,
        resolution_parameter=resolution,
        random_state=random_state,
        label=f"kNN graph of obsm['{reduction_key}']",
    )
    del graph

    if key_added is not None:
        persist_updates(adata, obs={key_added: labels})
    return labels


def _batch_design(
    adata: AnnData,
    batch_key: str,
    groups: Optional[Union[str, np.ndarray]] = None,
    strata_key: Optional[str] = None,
    depth: Optional[Union[str, np.ndarray]] = None,
    max_batch_frac: Optional[float] = 0.3,
    reduction_key: str = "action",
    decorrelate_from_reduction: bool = False,
    group_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the within-group batch design and return it with its provenance."""
    n = adata.n_obs
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")
    if group_kwargs is not None and not (isinstance(groups, str) and groups == "auto"):
        raise ValueError("'group_kwargs' only applies when groups='auto'.")

    b_codes, b_names = _obs_labels(adata, batch_key, "Batch key")
    n_batches = b_names.shape[0]
    if n_batches < 2:
        raise ValueError("'batch_key' must have at least 2 categories for correction.")

    if max_batch_frac is not None:
        max_batch_frac = float(max_batch_frac)
        if not 0.0 < max_batch_frac <= 1.0:
            raise ValueError("'max_batch_frac' must be in (0, 1], or None to disable merging.")
        if max_batch_frac <= 1.0 / n_batches:
            raise ValueError(
                f"'max_batch_frac' ({max_batch_frac:g}) must exceed 1/n_batches "
                f"({1.0 / n_batches:.3g} for {n_batches} batches): every group would count as "
                "dominated by one batch. Raise it, or pass None to disable merging."
            )

    # --- Resolve cluster labels ---
    if groups is None:
        groups_source = "none"
        c_codes = np.zeros(n, dtype=np.int64)
        c_names = np.asarray(["all"], dtype=object)
    elif isinstance(groups, str) and groups == "auto":
        groups_source = "auto"
        kwargs = {"reduction_key": reduction_key, **(group_kwargs or {})}
        c_codes, c_names = _factorize_labels(
            estimate_batch_groups(adata, **kwargs), "Estimated groups", n
        )
    elif isinstance(groups, str):
        groups_source = groups
        c_codes, c_names = _obs_labels(adata, groups, "Group key")
    else:
        groups_source = "array"
        c_codes, c_names = _factorize_labels(groups, "'groups'", n)
    c_names = c_names.astype(str).astype(object)
    n_clusters = c_names.shape[0]

    # --- Merge batch-dominated clusters into the nearest remaining centroid ---
    merged: dict[str, str] = {}
    if max_batch_frac is not None and n_clusters > 1:
        counts = np.bincount(c_codes * n_batches + b_codes, minlength=n_clusters * n_batches)
        counts = counts.reshape(n_clusters, n_batches)
        top_frac = counts.max(axis=1) / counts.sum(axis=1)
        kept = np.flatnonzero(top_frac <= max_batch_frac)
        dominated = np.flatnonzero(top_frac > max_batch_frac)
        if kept.size == 0:
            raise ValueError(
                "Every group is dominated by one batch; lower the clustering resolution "
                "or raise 'max_batch_frac'."
            )
        if dominated.size > 0:
            X = np.asarray(adata.obsm[reduction_key], dtype=float)
            centroids = _group_means(c_codes, n_clusters, X)
            remap = np.arange(n_clusters)
            for c in dominated:
                dist = ((centroids[kept] - centroids[c]) ** 2).sum(axis=1)
                remap[c] = kept[np.argmin(dist)]
                merged[str(c_names[c])] = str(c_names[remap[c]])
            c_codes = remap[c_codes]

    # --- Groups = cluster x stratum ---
    if strata_key is not None:
        s_codes, s_names = _obs_labels(adata, strata_key, "Strata key")
        g_codes = c_codes * s_names.shape[0] + s_codes
    else:
        g_codes = c_codes
    _, g_codes = np.unique(g_codes, return_inverse=True)
    g_codes = g_codes.reshape(-1)
    n_groups = int(g_codes.max()) + 1
    g_sizes = np.bincount(g_codes, minlength=n_groups).astype(float)

    # --- Batch one-hots (and optional depth) centred within groups ---
    depth_values: Optional[np.ndarray] = None
    if depth is not None:
        if isinstance(depth, str):
            if depth not in adata.obs:
                raise ValueError(f"Depth key '{depth}' not found in adata.obs")
            depth_values = np.asarray(adata.obs[depth].to_numpy(), dtype=float).reshape(-1)
        else:
            depth_values = np.asarray(depth, dtype=float).reshape(-1)
        if depth_values.shape[0] != n:
            raise ValueError(f"'depth' has length {depth_values.shape[0]}; expected {n} (adata.n_obs).")
        if not np.isfinite(depth_values).all():
            raise ValueError("'depth' contains NaN or Inf values.")

    gb_counts = np.bincount(g_codes * n_batches + b_codes, minlength=n_groups * n_batches)
    batch_frac = gb_counts.reshape(n_groups, n_batches) / g_sizes[:, None]
    D = np.empty((n, n_batches + (depth_values is not None)), dtype=float)
    np.negative(batch_frac[g_codes], out=D[:, :n_batches])
    D[np.arange(n), b_codes] += 1.0
    if depth_values is not None:
        depth_mean = np.bincount(g_codes, weights=depth_values, minlength=n_groups) / g_sizes
        D[:, n_batches] = depth_values - depth_mean[g_codes]

    # --- Keep the design orthogonal to the reduction's cell-side centring terms ---
    if decorrelate_from_reduction:
        b_key = f"{reduction_key}_B"
        if b_key not in adata.obsm:
            raise ValueError(f"'{b_key}' not found. Run reduce_kernel first.")
        C = np.column_stack([np.ones(n), np.asarray(adata.obsm[b_key], dtype=float)])
        Q, R, _ = scipy.linalg.qr(C, mode="economic", pivoting=True)
        r_diag = np.abs(np.diag(R))
        rank = int(np.count_nonzero(r_diag > _RANK_REL_TOL * r_diag[0]))
        Q = Q[:, :rank]
        D -= Q @ (Q.T @ D)

    info = {
        "groups": c_names[c_codes],
        "groups_source": groups_source,
        "merged_groups": merged,
        "n_groups": n_groups,
        "n_batches": int(n_batches),
    }
    return D, info


def build_batch_design(
    adata: AnnData,
    batch_key: str,
    groups: Optional[Union[str, np.ndarray]] = None,
    strata_key: Optional[str] = None,
    depth: Optional[Union[str, np.ndarray]] = None,
    max_batch_frac: Optional[float] = 0.3,
    reduction_key: str = "action",
    decorrelate_from_reduction: bool = False,
    group_kwargs: Optional[dict] = None,
    key_added: Optional[str] = None,
) -> np.ndarray:
    """Build a design matrix that removes batch effects within cell groups.

    The returned matrix is meant for ``correct_batch_effect(design=...)``, or is
    built implicitly when ``correct_batch_effect`` receives ``groups``,
    ``strata_key`` or ``depth``. Columns are batch one-hot indicators centred
    within each group (cluster x stratum), so differences in group composition
    between batches are not treated as batch effects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed ``reduce_kernel`` reduction.
    batch_key : str
        Column in ``adata.obs`` with batch labels (e.g. donor).
    groups : str, ndarray, "auto", or None
        Cell groups within which batches are compared. An ``adata.obs`` column,
        a per-cell label array, ``"auto"`` to estimate them with
        :func:`estimate_batch_groups` on ``obsm[reduction_key]``, or ``None``
        for a single group.
    strata_key : str or None
        Column in ``adata.obs`` with a covariate to protect (e.g. brain
        region). Groups become cluster x stratum, so batch effects are only
        estimated between cells of the same stratum.
    depth : str, ndarray, or None
        Optional per-cell depth covariate (e.g. log total counts), added as
        one extra column centred within groups. Experimental. It has little
        effect with ``decorrelate_from_reduction=True``, which projects most
        of it away.
    max_batch_frac : float or None
        Groups in which a single batch exceeds this fraction of cells are
        merged into the group with the nearest centroid in
        ``obsm[reduction_key]``. Otherwise their batch effect would be kept as
        if it were biology. Must exceed ``1 / n_batches``. ``None`` disables
        merging.
    reduction_key : str
        Key prefix of the ``reduce_kernel`` output that will be corrected.
    decorrelate_from_reduction : bool
        Make the design orthogonal over cells to the all-ones vector and to
        ``obsm[f"{reduction_key}_B"]`` (for a fresh reduction, the per-cell
        projection on the gene-mean direction and the mean expression).
        ``True`` reproduces the original prototype design. Because those terms
        track sequencing depth, the constraint also keeps most within-batch
        depth variation out of the correction.
    group_kwargs : dict or None
        Keyword arguments for :func:`estimate_batch_groups` when
        ``groups="auto"``.
    key_added : str or None
        If given, store the cluster labels after merging in
        ``adata.obs[key_added]``.

    Returns
    -------
    ndarray
        Design matrix ``(n_obs, n_batches [+ 1])``. It is rank deficient by
        construction; ``correct_batch_effect`` drops redundant directions.
    """
    D, info = _batch_design(
        adata,
        batch_key,
        groups=groups,
        strata_key=strata_key,
        depth=depth,
        max_batch_frac=max_batch_frac,
        reduction_key=reduction_key,
        decorrelate_from_reduction=decorrelate_from_reduction,
        group_kwargs=group_kwargs,
    )
    if key_added is not None:
        persist_updates(adata, obs={key_added: pd.Categorical(info["groups"])})
    return D


def correct_batch_effect(
    adata: AnnData,
    batch_key: Optional[str] = None,
    design: Optional[np.ndarray] = None,
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_suffix: str = "corrected",
    inplace: bool = True,
    backed_chunk_size: int = 8192,
    lazy_transform: Optional[LazyTransform] = None,
    groups: Optional[Union[str, np.ndarray]] = None,
    strata_key: Optional[str] = None,
    depth: Optional[Union[str, np.ndarray]] = None,
    max_batch_frac: Optional[float] = 0.3,
    group_kwargs: Optional[dict] = None,
    decorrelate_from_reduction: bool = False,
) -> Optional[AnnData]:
    """Correct batch effects using orthogonalization.

    The gene-space directions spanned by ``S' D`` (``S`` the cells x genes
    matrix, ``D`` the design) are removed from the reduction. Setting any of
    *groups*, *strata_key*, *depth* or *decorrelate_from_reduction* switches
    to within-group correction: the design is built by
    :func:`build_batch_design` from *batch_key*, so batches are only compared
    between cells of the same group (cluster x stratum).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed reduction.
    batch_key : str or None
        Column in ``adata.obs`` containing batch labels (categorical).
        Mutually exclusive with *design*.
    design : ndarray or None
        Explicit design matrix ``(n_obs, n_covariates)``. Columns may be
        linearly dependent; redundant directions are dropped.
        Mutually exclusive with *batch_key*.
    reduction_key : str
        Key prefix for the existing reduction in ``adata.obsm``.
    layer : str or None
        Layer to use.  ``None`` uses ``adata.X``.
    corrected_suffix : str
        Suffix appended to *reduction_key* for storing corrected results.
    inplace : bool
        If True, modifies *adata* in place and returns None.
        If False, returns a modified copy.
    backed_chunk_size : int
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform or None
        Pre-computed lazy transform for backed inputs on ``.X`` only.
        ``None`` means no transformation is applied.
    groups : str, ndarray, "auto", or None
        Cell groups for within-group correction: an ``adata.obs`` column, a
        per-cell label array, or ``"auto"`` to estimate them with
        :func:`estimate_batch_groups`. Requires *batch_key*.
    strata_key : str or None
        Column in ``adata.obs`` with a covariate to protect (e.g. region).
        Requires *batch_key*.
    depth : str, ndarray, or None
        Experimental per-cell depth covariate added to the design.
        Requires *batch_key*.
    max_batch_frac : float or None
        Groups dominated by one batch beyond this fraction are merged into the
        nearest group. Only used for within-group correction.
    group_kwargs : dict or None
        Keyword arguments for :func:`estimate_batch_groups` when
        ``groups="auto"``.
    decorrelate_from_reduction : bool
        Make the design orthogonal to the reduction's cell-side centring
        terms, as in the original prototype. See :func:`build_batch_design`.
        Requires *batch_key*.

    Returns
    -------
    None or AnnData
        If ``inplace=True``, returns ``None`` and modifies *adata* in place.
        If ``inplace=False``, returns a new AnnData with corrected results.
        Within-group correction also stores the cluster labels after merging
        in ``adata.obs[f"{reduction_key}_{corrected_suffix}_groups"]``.
    """
    corrected_key = f"{reduction_key}_{corrected_suffix}"

    if (batch_key is None) == (design is None):
        raise ValueError("Provide exactly one of 'batch_key' or 'design'.")

    grouped = (
        groups is not None or strata_key is not None or depth is not None
        or decorrelate_from_reduction
    )
    if grouped and batch_key is None:
        raise ValueError(
            "'groups', 'strata_key', 'depth' and 'decorrelate_from_reduction' require 'batch_key'."
        )
    if group_kwargs is not None and not grouped:
        raise ValueError("'group_kwargs' only applies when groups='auto'.")

    if not inplace:
        adata = adata.copy()

    if batch_key is not None and batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    source = MatrixSource(adata, layer=layer)
    # Validate before expensive reduction-state loading or operator setup.
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    old_S_r, old_U, old_A, old_B, old_sigma = _load_reduction_state(adata, reduction_key)

    design_info: Optional[dict[str, Any]] = None
    if grouped:
        design, design_info = _batch_design(
            adata,
            batch_key,
            groups=groups,
            strata_key=strata_key,
            depth=depth,
            max_batch_frac=max_batch_frac,
            reduction_key=reduction_key,
            decorrelate_from_reduction=decorrelate_from_reduction,
            group_kwargs=group_kwargs,
        )

    # When the user provides batch_key, build a categorical batch vector and capture
    # integer labels.  These are used directly by the sparse one-hot fast path so we
    # can avoid materialising a dense (n_obs × n_batches) design matrix.
    batch_labels: Optional[pd.Categorical] = None
    if batch_key is not None and not grouped:
        batch_series = pd.Series(adata.obs[batch_key].to_numpy(), index=adata.obs_names).astype("category")
        if len(batch_series.cat.categories) < 2:
            raise ValueError("'batch_key' must have at least 2 categories for correction.")
        batch_labels = batch_series
        # Lazily build the dense design only when the fast path cannot be used.
        design = None
    else:
        design = np.asarray(design, dtype=float)
        if design.ndim != 2:
            raise ValueError("'design' must be a 2D matrix (cells x covariates).")
        if design.shape[0] != adata.n_obs:
            raise ValueError("Design matrix does not match number of observations.")
        design = np.ascontiguousarray(design)

    apply_log1p = False

    if source.is_backed:
        # Backed/operator path: still requires a dense design matrix.
        if design is None:
            design = pd.get_dummies(batch_labels, drop_first=False).to_numpy(dtype=float)
            design = np.ascontiguousarray(design)
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        with open_backed_operator_for(
            adata,
            layer=layer,
            context="correct_batch_effect",
            chunk_size=backed_chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
        ) as op:
            result = _core.orthogonalize_batch_effect_operator(
                op, old_S_r, old_U, old_A, old_B, old_sigma, design,
            )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        if sp.issparse(S):
            if batch_labels is not None:
                # Sparse one-hot fast path: pass integer labels directly.
                # C++ accumulates Z = S' * D in O(nnz + g*b) without ever
                # materialising the dense design matrix.
                codes = batch_labels.cat.codes.to_numpy(dtype=np.int64, copy=True)
                n_batches = int(len(batch_labels.cat.categories))
                result = _core.orthogonalize_batch_effect_sparse_labels(
                    S, old_S_r, old_U, old_A, old_B, old_sigma, codes, n_batches,
                )
            else:
                result = _core.orthogonalize_batch_effect_sparse(
                    S, old_S_r, old_U, old_A, old_B, old_sigma, design
                )
        else:
            S = np.asarray(S, dtype=float, order="C")
            if design is None:
                design = pd.get_dummies(batch_labels, drop_first=False).to_numpy(dtype=float)
                design = np.ascontiguousarray(design)
            result = _core.orthogonalize_batch_effect_dense(
                S, old_S_r, old_U, old_A, old_B, old_sigma, design
            )

    corrected_params = {
        "sigma": result["sigma"],
        "batch_key": batch_key,
        "original_reduction": reduction_key,
        "correction_rank": int(result["A"].shape[1] - old_A.shape[1]),
    }
    corrected_params.update(_lazy_params_for_metadata(lazy_transform if apply_log1p else None))

    obs_updates = {}
    if design_info is not None:
        corrected_params["grouped_design"] = {
            "groups": design_info["groups_source"],
            "strata_key": strata_key,
            "depth": depth if isinstance(depth, str) else ("array" if depth is not None else None),
            "max_batch_frac": None if max_batch_frac is None else float(max_batch_frac),
            "decorrelate_from_reduction": bool(decorrelate_from_reduction),
            "merged_groups": design_info["merged_groups"],
            "n_groups": design_info["n_groups"],
            "n_batches": design_info["n_batches"],
            "design_columns": int(design.shape[1]),
        }
        obs_updates[f"{corrected_key}_groups"] = pd.Categorical(design_info["groups"])

    persist_updates(
        adata,
        obs=obs_updates,
        obsm={
            corrected_key: result["S_r"],          # cells x k, direct
            f"{corrected_key}_B": result["B"],
        },
        varm={
            f"{corrected_key}_U": result["U"],
            f"{corrected_key}_A": result["A"],
        },
        uns={f"{corrected_key}_params": corrected_params},
    )
    if not inplace:
        return adata
    return None


def correct_basal_expression(
    adata: AnnData,
    basal_genes: Union[list, np.ndarray],
    reduction_key: str = "action",
    layer: Optional[str] = None,
    corrected_key: str = "action_basal_corrected",
    inplace: bool = True,
    backed_chunk_size: int = 8192,
    lazy_transform: Optional[LazyTransform] = None,
) -> Optional[AnnData]:
    """Correct for basal expression levels by orthogonalizing their effects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed reduction.
    basal_genes : list or ndarray
        Gene names whose basal expression should be removed.
    reduction_key : str
        Key prefix for the existing reduction.
    layer : str or None
        Layer to use.  ``None`` uses ``adata.X``.
    corrected_key : str
        Key for storing corrected results.
    inplace : bool
        If True, modifies *adata* in place and returns None.
        If False, returns a modified copy.
    backed_chunk_size : int
        Number of rows per chunk when streaming backed AnnData.
        Ignored for in-memory objects.
    lazy_transform : LazyTransform or None
        Pre-computed lazy transform for backed inputs on ``.X`` only.
        ``None`` means no transformation is applied.

    Returns
    -------
    None or AnnData
        If ``inplace=True``, returns ``None`` and modifies *adata* in place.
        If ``inplace=False``, returns a new AnnData with corrected results.
    """
    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)
    # Validate before expensive reduction-state loading or operator setup.
    _validate_lazy_transform(lazy_transform, layer=layer, source=source)

    old_S_r, old_U, old_A, old_B, old_sigma = _load_reduction_state(adata, reduction_key)

    from .._feature_lookup import resolve_feature_space, resolve_requested_features

    space = resolve_feature_space(adata, features_use=None, context="correct_basal_expression")
    resolved = resolve_requested_features(
        list(basal_genes), space, context="correct_basal_expression"
    )
    if resolved.matched_indices.size == 0:
        raise ValueError("None of the specified basal genes found in adata.var_names")

    basal = np.zeros((adata.n_vars, 1), dtype=float)
    basal[resolved.matched_indices, 0] = 1.0
    basal = np.ascontiguousarray(basal)

    row_scale_factors: Optional[np.ndarray] = None
    apply_log1p = False
    log_scale = 1.0

    if source.is_backed:
        row_scale_factors, apply_log1p, log_scale = _resolve_lazy_backed_transform(
            source,
            lazy_transform=lazy_transform,
            backed_chunk_size=backed_chunk_size,
        )
        with open_backed_operator_for(
            adata,
            layer=layer,
            context="correct_basal_expression",
            chunk_size=backed_chunk_size,
            row_scale_factors=row_scale_factors,
            apply_log1p=apply_log1p,
            log_scale=log_scale,
        ) as op:
            result = _core.orthogonalize_basal_operator(
                op, old_S_r, old_U, old_A, old_B, old_sigma, basal,
            )
    else:
        S = anndata_to_matrix(adata, layer=layer)  # cells x genes, native
        if sp.issparse(S):
            result = _core.orthogonalize_basal_sparse(
                S, old_S_r, old_U, old_A, old_B, old_sigma, basal
            )
        else:
            S = np.asarray(S, dtype=float, order="C")
            result = _core.orthogonalize_basal_dense(
                S, old_S_r, old_U, old_A, old_B, old_sigma, basal
            )

    corrected_params = {
        "sigma": result["sigma"],
        "basal_genes": np.asarray(basal_genes)[np.isin(basal_genes, adata.var_names)].tolist(),
        "original_reduction": reduction_key,
    }
    corrected_params.update(_lazy_params_for_metadata(lazy_transform if apply_log1p else None))

    persist_updates(
        adata,
        obsm={
            corrected_key: result["S_r"],
            f"{corrected_key}_B": result["B"],
        },
        varm={
            f"{corrected_key}_U": result["U"],
            f"{corrected_key}_A": result["A"],
        },
        uns={f"{corrected_key}_params": corrected_params},
    )

    if not inplace:
        return adata
    return None
