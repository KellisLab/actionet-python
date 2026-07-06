"""ACTION archetypal analysis wrapper."""

from typing import Optional
import numpy as np
from anndata import AnnData

from .. import _core
from ..io.persist import persist_updates
from .. import tools


def run_action(
    adata: AnnData,
    k_min: int = 2,
    k_max: int = 30,
    reduction_key: str = "action",
    prenormalize: bool = True,
    max_iter: int = 50,
    tolerance: float = 1e-100,
    specificity_threshold: float = -3.0,
    min_observations: int = 2,
    n_threads: int = 0,
    return_c_matrices: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run ACTION archetypal analysis decomposition.

    Parameters
    ----------
    adata
        Annotated data matrix with reduced representation.
    k_min
        Minimum number of archetypes.
    k_max
        Maximum number of archetypes.
    reduction_key
        Key in adata.obsm containing reduced representation.
    prenormalize
        If True (default), L1-normalize the rows of ``adata.obsm[reduction_key]``
        before decomposition. Disable only if the caller has already pre-scaled the
        reduction.
    max_iter
        Maximum iterations for AA.
    tolerance
        Convergence tolerance.
    specificity_threshold
        Threshold for filtering archetypes (z-score).
    min_observations
        Minimum observations per archetype.
    n_threads
        Number of threads (0=auto).
    return_c_matrices
        If True, persist ``C_stacked`` and ``C_merged`` in ``adata.obsm``.
        If False, only ``H`` matrices and assignments are returned/stored.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm["H_stacked"] : np.ndarray
        Stacked archetype matrix (cells × archetypes).
    adata.obsm["H_merged"] : np.ndarray
        Merged archetype matrix (cells × archetypes, after merging similar archetypes).
    adata.obsm["C_stacked"] : np.ndarray, optional
        Stacked cell-archetype coefficient matrix (cells × archetypes).
        Written only when ``return_c_matrices=True``.
    adata.obsm["C_merged"] : np.ndarray, optional
        Merged cell-archetype coefficient matrix (cells × archetypes).
        Written only when ``return_c_matrices=True``.
    adata.obs["assigned_archetype"] : pd.Series or np.ndarray
        Cell-to-archetype assignments.

    Notes
    -----
    The Python defaults for ``max_iter`` (50), ``tolerance`` (1e-100), and
    ``min_observations`` (2) intentionally diverge from the C++ core /
    ``decompose_action`` defaults (100, 1e-6, 3 respectively).  The Python
    values were chosen to prioritize convergence speed and sensitivity for
    interactive exploratory workflows.  Pass explicit values if you need
    C++-identical behaviour.
    """
    if not inplace:
        adata = adata.copy()
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    S_r = adata.obsm[reduction_key]  # cells x k, native orientation

    if prenormalize:
        # Match the shared AnnData contract and R wrapper: each cell is a row in
        # ``obsm[reduction_key]``, so prenormalization must scale rows, not columns.
        S_r = tools.l1_norm_scale(S_r, axis=1)

    # Ensure C-contiguous memory layout for C++ compatibility
    S_r = np.ascontiguousarray(S_r)

    result = _core.run_action(
        S_r, k_min, k_max, max_iter, tolerance,
        specificity_threshold, min_observations, n_threads, return_c_matrices
    )

    obsm_updates = {
        "H_stacked": result["H_stacked"],    # cells x archetypes, direct
        "H_merged": result["H_merged"],      # direct
    }
    if return_c_matrices:
        obsm_updates["C_stacked"] = result["C_stacked"]
        obsm_updates["C_merged"] = result["C_merged"]

    persist_updates(
        adata,
        obsm=obsm_updates,
        obs={"assigned_archetype": result["assigned_archetypes"]},
    )
    if not inplace:
        return adata
    return None
