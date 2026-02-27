"""High-level Python API wrapping C++ bindings with AnnData integration."""

import warnings
from typing import Any, Optional, Union, Literal
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from . import _core
from .anndata_utils import anndata_to_matrix
from ._backed_persist import persist_updates
from ._backed_compression import (
    format_compression_summary,
    get_storage_metadata_from_adata,
    get_storage_metadata_from_matrix,
    is_compressed_storage,
)
from ._matrix_source import MatrixSource
from . import tools


_WARNED_COMPRESSED_BACKED_SVD: set[tuple[str, str, str]] = set()


def _is_backed_matrix(X: Any) -> bool:
    """Detect whether ``X`` is backed/on-disk rather than fully in memory."""
    if sp.issparse(X) or isinstance(X, np.ndarray):
        return False

    if hasattr(X, "isbacked"):
        return bool(X.isbacked)

    if hasattr(X, "group"):
        return True

    mod = type(X).__module__
    if mod and mod.startswith("h5py"):
        return True

    return False


def _warn_if_compressed_backed_svd(
    metadata: Optional[dict],
    *,
    context: str,
    recommendation: str,
) -> None:
    """Warn once per (file, matrix key, context) for compressed backed SVD."""
    if not is_compressed_storage(metadata):
        return

    filename = str((metadata or {}).get("filename") or "<unknown>")
    matrix_key = str((metadata or {}).get("matrix_key") or "<unknown>")
    dedupe_key = (filename, matrix_key, context)
    if dedupe_key in _WARNED_COMPRESSED_BACKED_SVD:
        return
    _WARNED_COMPRESSED_BACKED_SVD.add(dedupe_key)

    codecs = format_compression_summary(metadata)
    warnings.warn(
        (
            f"Backed operator SVD in `{context}` is reading compressed storage "
            f"for `{matrix_key}` ({codecs}). This can cause major runtime "
            f"slowdowns due to repeated decompression during matvec passes. "
            f"Recommended: `{recommendation}`."
        ),
        UserWarning,
        stacklevel=3,
    )


class _TransposeMatrixOperator:
    """Matrix operator for S = X.T where X is cells x genes (backed sparse).

    Implements chunked matrix-vector products that stream through X in
    row-blocks of ``chunk_size`` cells at a time, keeping peak memory
    proportional to ``chunk_size * n_genes`` rather than ``n_cells * n_genes``.

    Parameters
    ----------
    X
        Backed sparse dataset (cells × genes).  Accessed via ``X[start:end, :]``.
    chunk_size : int
        Number of cell-rows to load per chunk.  Larger values reduce Python
        loop overhead and improve BLAS utilisation, but increase peak memory.
        The default (4096) works well for datasets where each chunk fits
        comfortably in L3 cache (~256 MB at 64k genes, float64).

    Performance notes
    -----------------
    - Each PRIMME iteration calls ``matvec`` and ``rmatvec`` once per block
      vector, so I/O is the dominant cost.  Consider increasing ``chunk_size``
      if the dataset is on fast local SSD, or decreasing it on network storage.
    - Each chunk is converted to dense (if sparse) via ``np.asarray``, which
      creates a temporary ``(chunk_size, n_genes)`` float64 array.  For very
      large gene dimensions (>100k) this temporary may dominate memory.
    - Accumulated floating-point rounding across chunks is negligible for
      typical single-cell data but is documented here for completeness.

    Possible future improvements
    ----------------------------
    - Accept a ``dtype`` parameter and accumulate in float32 for memory savings
      when full double precision is unnecessary.
    - Pre-fetch the next chunk in a background thread while computing on the
      current chunk (double-buffering) to hide I/O latency.
    - Expose a ``rechunk`` helper that rewrites backed data into an optimal
      on-disk chunk layout for row-slicing.
    """

    def __init__(self, X: Any, chunk_size: int = 4096):
        self._X = X
        self._chunk_size = int(max(1, chunk_size))
        n_cells, n_genes = X.shape
        # Logical shape exposed to C++: features x cells
        self.shape = (int(n_genes), int(n_cells))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        # y = S @ x = X.T @ x
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.shape[1]:
            raise ValueError(f"matvec expected vector of length {self.shape[1]}, got {x.shape}")

        y = np.zeros(self.shape[0], dtype=np.float64)
        for start in range(0, self.shape[1], self._chunk_size):
            end = min(start + self._chunk_size, self.shape[1])
            block = self._X[start:end, :]
            if sp.issparse(block):
                y += np.asarray(block.T.dot(x[start:end])).ravel()
            else:
                y += np.asarray(block, dtype=np.float64).T @ x[start:end]
        return y

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        # y = S.T @ x = X @ x
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.shape[0]:
            raise ValueError(f"rmatvec expected vector of length {self.shape[0]}, got {x.shape}")

        y = np.zeros(self.shape[1], dtype=np.float64)
        for start in range(0, self.shape[1], self._chunk_size):
            end = min(start + self._chunk_size, self.shape[1])
            block = self._X[start:end, :]
            if sp.issparse(block):
                y[start:end] = np.asarray(block.dot(x)).ravel()
            else:
                y[start:end] = np.asarray(block, dtype=np.float64) @ x
        return y


def _select_svd_algorithm(
    S: Any,
    svd_algorithm: Optional[int],
    verbose: bool = True
) -> int:
    """
    Select the optimal SVD algorithm based on matrix properties.

    Parameters
    ----------
    S : scipy.sparse matrix, numpy.ndarray, or backed dataset
        Input matrix (features × cells).
    svd_algorithm : int or None
        User-specified algorithm (if None, automatic selection is performed).
    verbose : bool
        Whether to print selection rationale.

    Returns
    -------
    int
        Selected algorithm code (0=IRLB, 1=Halko, 2=Feng, 3=PRIMME).

    Selection Logic
    ---------------
    1. Backed inputs always use PRIMME (operator path).
    2. If user specifies an algorithm explicitly, use it.
    3. If matrix exceeds 32-bit indexing limits (>2^31-1 elements), force PRIMME.
    4. For sparse matrices:
       - Large & very sparse (>70% sparse, >1B elements): PRIMME
       - Otherwise: IRLB
    5. For dense matrices: Halko (fastest).
    """
    # If algorithm is explicitly specified, use it.
    # Guard backed inputs first — they must go through the operator path.
    if _is_backed_matrix(S):
        if svd_algorithm is not None and svd_algorithm != 3:
            raise ValueError("Backed matrices currently support only PRIMME (svd_algorithm=3)")
        if verbose:
            print("Detected backed matrix: selecting PRIMME operator path")
        return 3

    if svd_algorithm is not None:
        if svd_algorithm not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid svd_algorithm {svd_algorithm}. Must be 0 (IRLB), 1 (Halko), 2 (Feng), or 3 (PRIMME).")
        return svd_algorithm

    # Calculate matrix properties
    if sp.issparse(S):
        total_elements = S.nnz
    else:
        total_elements = np.prod(S.shape)

    # Check for 32-bit overflow (2^31 - 1 = 2,147,483,647)
    # Many sparse matrix libraries use 32-bit integers for indexing
    MAX_32BIT = 2_147_483_647

    if total_elements > MAX_32BIT:
        if verbose:
            print(f"⚠ Matrix exceeds 32-bit indexing limit ({total_elements:,} > {MAX_32BIT:,} elements)")
            print(f"→ Auto-selected PRIMME for safe handling of large matrices")
        return 3  # PRIMME

    # Determine sparsity and select algorithm
    if sp.issparse(S):
        sparsity = 1.0 - (total_elements / np.prod(S.shape))

        # For large, very sparse matrices, PRIMME is most memory-efficient
        if sparsity > 0.7 and total_elements > 2_000_000_000:
            if verbose:
                print(f"Auto-selected PRIMME for large sparse matrix "
                      f"({sparsity:.1%} sparse, {total_elements:,} elements)")
            return 3  # PRIMME
        else:
            if verbose:
                print(f"Auto-selected IRLB for sparse matrix "
                      f"({sparsity:.1%} sparse, {total_elements:,} elements)")
            return 0  # IRLB
    else:
        # Dense matrices: Halko is typically fastest
        if verbose:
            print(f"Auto-selected Halko for dense matrix ({total_elements:,} elements)")
        return 1  # Halko


def reduce_kernel(
    adata: AnnData,
    n_components: int = 30,
    layer: Optional[str] = None,
    key_added: str = "action",
    svd_algorithm: Optional[int] = None,
    max_iter: int = 0,
    seed: int = 0,
    verbose: bool = True,
    precomputed_svd: Optional[dict] = None,
    backed_chunk_size: int = 4096,
    inplace: bool = True
) -> Optional[AnnData]:
    """
    Compute a low-rank approximation of the kernel matrix for ACTION decomposition and store the results in AnnData.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells × features).
    n_components : int, optional (default: 30)
        Number of singular vectors (components) to compute.
    layer : str or None, optional (default: None)
        Layer in AnnData to use for computation. If None, uses adata.X.
    key_added : str, optional (default: "action")
        Key under which to store the results in adata.obsm and related fields.
    svd_algorithm : int or None, optional (default: None)
        SVD algorithm to use:
        - 0: IRLB (Implicitly Restarted Lanczos Bidiagonalization)
        - 1: Halko (Randomized SVD)
        - 2: Feng (Feng's randomized algorithm)
        - 3: PRIMME (PReconditioned Iterative MultiMethod Eigensolver)
        - None: Automatic selection based on matrix properties (recommended)
    max_iter : int, optional (default: 0)
        Maximum number of iterations for SVD solver (0=auto).
    seed : int, optional (default: 0)
        Random seed for reproducibility.
    verbose : bool, optional (default: True)
        Whether to print progress messages.
    precomputed_svd : dict or None, optional (default: None)
        If provided, skip SVD computation and use this precomputed result.
        Expected keys: ``"u"`` (features × k), ``"d"`` (k,), ``"v"`` (cells × k).
        Obtain via :func:`run_svd`.
    backed_chunk_size : int, optional (default: 4096)
        Number of cell-rows per chunk when streaming backed sparse data.
        Ignored for in-memory matrices.  See :class:`_TransposeMatrixOperator`
        for tuning guidance.
    inplace : bool, optional (default: True)
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Reduced representation (cells × n_components).
    adata.obsm[f"{key_added}_B"] : np.ndarray
        B matrix from decomposition (cells × n_components).
    adata.varm[f"{key_added}_U"] : np.ndarray
        Left singular vectors (features × n_components).
    adata.varm[f"{key_added}_A"] : np.ndarray
        A matrix from decomposition (features × n_components).
    adata.uns[f"{key_added}_params"] : dict
        Parameters used for reduction (e.g., sigma, n_components, svd_algorithm).
    """
    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)
    X = source.matrix  # cells x genes
    use_operator = source.is_backed

    if use_operator:
        if precomputed_svd is None:
            matrix_metadata = get_storage_metadata_from_adata(adata, layer=layer)
            _warn_if_compressed_backed_svd(
                matrix_metadata,
                context="reduce_kernel",
                recommendation=(
                    f'actionet.decompress_backed_storage(adata, layer={repr(layer)}, scope="matrix")'
                ),
            )

        svd_algorithm = 3  # OOM v1 supports PRIMME only.
        op = _TransposeMatrixOperator(X, chunk_size=backed_chunk_size)
        if precomputed_svd is None:
            result = _core.reduce_kernel_operator(op, n_components, max_iter, seed, verbose)
        else:
            result = _core.reduce_kernel_from_svd_operator(
                op,
                precomputed_svd["u"],
                precomputed_svd["d"],
                precomputed_svd["v"],
                verbose,
            )
    else:
        S = anndata_to_matrix(adata, layer=layer, transpose=True)
        svd_algorithm = _select_svd_algorithm(S, svd_algorithm, verbose)

        if precomputed_svd is None:
            if sp.issparse(S):
                result = _core.reduce_kernel_sparse(S, n_components, svd_algorithm, max_iter, seed, verbose)
            else:
                result = _core.reduce_kernel_dense(S, n_components, svd_algorithm, max_iter, seed, verbose)
        else:
            # Use the efficient in-memory path that computes perturbation terms
            # directly via Armadillo instead of the chunked operator path.
            if sp.issparse(S):
                result = _core.reduce_kernel_from_svd_sparse(
                    S, precomputed_svd["u"], precomputed_svd["d"],
                    precomputed_svd["v"], verbose,
                )
            else:
                result = _core.reduce_kernel_from_svd_dense(
                    S, precomputed_svd["u"], precomputed_svd["d"],
                    precomputed_svd["v"], verbose,
                )

    # Map algorithm code to name for better user understanding
    algorithm_names = {0: 'IRLB', 1: 'Halko', 2: 'Feng', 3: 'PRIMME'}

    params = {
        "sigma": np.asarray(result["sigma"]).ravel(),
        "n_components": n_components,
        "svd_algorithm": svd_algorithm,
        "svd_algorithm_name": algorithm_names.get(svd_algorithm, f'Unknown({svd_algorithm})'),
        "used_precomputed_svd": precomputed_svd is not None,
        "operator_mode": use_operator,
    }
    persist_updates(
        adata,
        obsm={
            key_added: result["S_r"].T,  # cells x components
            f"{key_added}_B": result["B"],
        },
        varm={
            f"{key_added}_U": result["U"],
            f"{key_added}_A": result["A"],
        },
        uns={f"{key_added}_params": params},
    )

    if not inplace:
        return adata
    return None


def reduce_kernel_from_svd(
    adata: AnnData,
    svd_result: dict,
    layer: Optional[str] = None,
    key_added: str = "action",
    verbose: bool = True,
    backed_chunk_size: int = 4096,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Compute reduced kernel using a precomputed SVD result.

    Convenience wrapper around :func:`reduce_kernel` that infers ``n_components``
    from the SVD result and passes it as ``precomputed_svd``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells × features).
    svd_result : dict
        Precomputed SVD with keys ``"u"``, ``"d"``, ``"v"`` (as returned by
        :func:`run_svd`).
    layer : str or None
        Layer to use (None uses .X).
    key_added : str
        Key under which to store results.
    verbose : bool
        Print progress messages.
    backed_chunk_size : int
        Chunk size for backed sparse streaming.
    inplace : bool
        Modify adata in place or return a copy.
    """
    return reduce_kernel(
        adata=adata,
        n_components=int(np.asarray(svd_result["d"]).size),
        layer=layer,
        key_added=key_added,
        svd_algorithm=None,
        max_iter=0,
        seed=0,
        verbose=verbose,
        precomputed_svd=svd_result,
        backed_chunk_size=backed_chunk_size,
        inplace=inplace,
    )


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
    adata.obs["assigned_archetype"] : pd.Series or np.ndarray
        Cell-to-archetype assignments.
    """
    if not inplace:
        adata = adata.copy()
    if reduction_key not in adata.obsm:
        raise ValueError(f"Reduction '{reduction_key}' not found. Run reduce_kernel first.")

    S_r = adata.obsm[reduction_key].T  # Transpose to components x cells

    if prenormalize:
        S_r = tools.l1_norm_scale(S_r, axis=0)

    # Ensure C-contiguous memory layout for C++ compatibility
    S_r = np.ascontiguousarray(S_r)

    result = _core.run_action(
        S_r, k_min, k_max, max_iter, tolerance,
        specificity_threshold, min_observations, n_threads
    )

    persist_updates(
        adata,
        obsm={
            "H_stacked": result["H_stacked"].T,
            "H_merged": result["H_merged"].T,
            "C_stacked": result["C_stacked"],
            "C_merged": result["C_merged"],
        },
        obs={"assigned_archetype": result["assigned_archetypes"]},
    )
    if not inplace:
        return adata
    return None


def build_network(
    adata: AnnData,
    algorithm: Literal["knn", "k*nn"] = "k*nn",
    distance_metric: Literal["jsd", "l2", "ip"] = "jsd",
    density: float = 1.0,
    n_threads: int = 0,
    mutual_edges_only: bool = True,
    M: float = 16,
    ef_construction: float = 200,
    ef: float = 200,
    k: int = 10,
    obsm_key: str = "H_stacked",
    key_added: str = "actionet",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Build cell-cell interaction network from archetype footprints.

    Parameters
    ----------
    adata
        Annotated data matrix with ACTION results.
    obsm_key
        Key in adata.obsm containing archetype matrix.
    algorithm
        Network construction algorithm.
    distance_metric
        Distance metric for similarity.
    density
        Graph density factor.
    k
        Number of nearest neighbors.
    mutual_edges_only
        Only keep mutual nearest neighbors.
    n_threads
        Number of threads (0=auto).
    key_added
        Key to store network in adata.obsp.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsp[key_added] : scipy.sparse matrix or np.ndarray
        Network adjacency matrix (cells × cells).
    """
    if not inplace:
        adata = adata.copy()
    if obsm_key not in adata.obsm:
        raise ValueError(f"Archetype matrix '{obsm_key}' not found. Run run_action first.")

    H = adata.obsm[obsm_key]

    # Ensure C-contiguous memory layout for C++ compatibility
    H = np.ascontiguousarray(H.T)

    G = _core.build_network(
        H, algorithm, distance_metric, density, n_threads,
        M, ef_construction, ef, mutual_edges_only, k
    )

    persist_updates(adata, obsp={key_added: G})
    if not inplace:
        return adata
    return None


def compute_network_diffusion(
    adata: AnnData,
    scores: Union[str, np.ndarray],
    norm_method: Literal["pagerank", "pagerank_sym"] = "pagerank",
    alpha: float = 0.85,
    n_threads: int = 0,
    approx: bool = True,
    max_iter: int = 5,
    tol = 1e-8,
    network_key: str = "actionet",
    key_added: str = "diffused",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute network diffusion/smoothing over ACTIONet graph.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    scores
        Either key in adata.obsm or array of scores to diffuse.
    network_key
        Key in adata.obsp containing network.
    alpha
        Diffusion parameter (0-1).
    max_iter
        Maximum iterations.
    n_threads
        Number of threads.
    key_added
        Key to store diffused scores.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Diffused scores (cells × features or cells × 1).
    """
    if not inplace:
        adata = adata.copy()
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found. Run build_network first.")
    
    G = adata.obsp[network_key]
    
    if isinstance(scores, str):
        if scores not in adata.obsm:
            raise ValueError(f"Scores '{scores}' not found in adata.obsm.")
        X0 = adata.obsm[scores]
    else:
        X0 = np.asarray(scores)
    
    if X0.ndim == 1:
        X0 = X0.reshape(-1, 1)

    # Ensure C-contiguous memory layout for C++ compatibility
    X0 = np.ascontiguousarray(X0)

    X_diffused = _core.compute_network_diffusion(
        G = G,
        X0 = X0,
        alpha = alpha,
        max_it = max_iter,
        thread_no = n_threads,
        approx = approx,
        norm_method = 2 if norm_method == "pagerank_sym" else 0,
        tol = tol
    )

    persist_updates(adata, obsm={key_added: X_diffused})
    if not inplace:
        return adata
    return None


def _labels_to_membership(labels_int: np.ndarray, n_obs: int) -> np.ndarray:
    labels_int = np.asarray(labels_int, dtype=np.int64).reshape(-1)
    if labels_int.shape[0] != n_obs:
        raise ValueError(
            f"labels length ({labels_int.shape[0]}) does not match number of observations ({n_obs})"
        )

    max_label = int(labels_int.max()) if labels_int.size > 0 else 0
    if max_label <= 0:
        raise ValueError("No valid labels after conversion; ensure labels contain at least one non-missing category.")

    H = np.zeros((n_obs, max_label), dtype=np.float64)
    valid = labels_int > 0
    H[np.arange(n_obs)[valid], labels_int[valid] - 1] = 1.0
    return H


def _compute_specificity_streamed(
    source: MatrixSource,
    H_cells: np.ndarray,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Streamed equivalent of ``libactionet::computeFeatureSpecificity()``.

    Implements Bernstein-style tail-probability scoring to identify
    features (genes) whose expression is specifically enriched or depleted
    in each group defined by *H_cells*.

    **Algorithm overview** (mirrors ``specificity.cpp``):

    1. Shift the matrix so all values are non-negative (subtract global
       minimum).
    2. Normalise the membership matrix *H* column-wise by dividing each
       column by its mean.
    3. Accumulate, in a single streaming pass over row-chunks:
       - ``row_count``  -- nnz count per feature (column)
       - ``col_count``  -- nnz count per observation (row)
       - ``row_factor_sum`` -- column sums of the shifted matrix
       - ``obs``        -- ``X_shifted.T @ H_norm`` (observed feature--group
         co-occurrence)
    4. Derive per-feature and per-observation density estimates
       ``row_p``, ``col_p`` and a relative-density weight ``beta``.
    5. Compute *expected* co-occurrence ``exp`` and its variance proxy
       ``nu`` under a null model, then evaluate one-sided Bernstein-type
       tail bounds, yielding ``log10``-scaled upper (enrichment) and lower
       (depletion) significance matrices.

    Parameters
    ----------
    source : MatrixSource
        Expression matrix accessor (cells x features).
    H_cells : ndarray, shape ``(n_obs, k)``
        Group-membership or archetype-footprint matrix.
    chunk_size : int
        Rows per streaming chunk.

    Returns
    -------
    dict with keys ``"average_profile"``, ``"upper_significance"``,
    ``"lower_significance"`` -- all ``(n_vars, k)`` arrays.
    """
    H_cells = np.asarray(H_cells, dtype=np.float64, order="C")
    if H_cells.ndim != 2 or H_cells.shape[0] != source.n_obs:
        raise ValueError(
            f"H_cells must have shape (n_obs, k) where n_obs={source.n_obs}, got {H_cells.shape}"
        )
    if H_cells.shape[1] == 0:
        raise ValueError("H_cells must contain at least one archetype/label column.")

    col_mean = H_cells.mean(axis=0)
    col_mean[col_mean == 0] = 1.0
    Ht = H_cells / col_mean[np.newaxis, :]

    min_val = source.global_min(chunk_size=chunk_size)

    row_count = np.zeros(source.n_vars, dtype=np.float64)
    row_factor_sum = np.zeros(source.n_vars, dtype=np.float64)
    col_count = np.zeros(source.n_obs, dtype=np.float64)
    obs = np.zeros((source.n_vars, Ht.shape[1]), dtype=np.float64)

    for chunk in source.iter_row_chunks(chunk_size=chunk_size):
        block = chunk.block
        h_block = Ht[chunk.start:chunk.end, :]

        if sp.issparse(block):
            block_csr = block.tocsr(copy=False)

            # Count nnz on the *original* block -- this matches the C++ sparse
            # iterator which visits all stored elements before and after the
            # min-shift.  Using nnz rather than positivity-after-shift ensures
            # numerical parity with the in-memory C++ path.
            row_count += np.asarray(block_csr.getnnz(axis=0)).ravel()
            col_count[chunk.start:chunk.end] = np.asarray(block_csr.getnnz(axis=1)).ravel()

            if min_val != 0.0:
                block_csr = block_csr.copy()
                block_csr.data = block_csr.data - min_val

            row_factor_sum += np.asarray(block_csr.sum(axis=0)).ravel()
            obs += np.asarray(block_csr.T.dot(h_block))
        else:
            arr = np.asarray(block, dtype=np.float64)
            if min_val != 0.0:
                arr = arr - min_val

            pos = arr > 0
            row_count += pos.sum(axis=0)
            col_count[chunk.start:chunk.end] = pos.sum(axis=1)
            row_factor_sum += arr.sum(axis=0)
            obs += arr.T @ h_block

    row_factor = np.divide(
        row_factor_sum,
        row_count,
        out=np.zeros_like(row_factor_sum),
        where=row_count > 0,
    )

    row_p = row_count / float(source.n_obs if source.n_obs > 0 else 1)
    col_p = col_count / float(source.n_vars if source.n_vars > 0 else 1)

    rho = float(col_p.mean()) if col_p.size > 0 else 0.0
    beta = np.zeros_like(col_p) if rho == 0.0 else (col_p / rho)

    gamma = Ht * beta[:, np.newaxis]
    a = gamma.max(axis=0) if gamma.size > 0 else np.zeros(Ht.shape[1], dtype=np.float64)

    exp = np.outer(row_p * row_factor, gamma.sum(axis=0))
    nu = np.outer(row_p * np.square(row_factor), np.square(gamma).sum(axis=0))
    A = np.outer(row_factor, a)
    lamb = obs - exp

    with np.errstate(divide="ignore", invalid="ignore"):
        log_lower = np.square(lamb) / (2.0 * nu)
        log_upper = np.square(lamb) / (2.0 * (nu + (lamb * A / 3.0)))

    log_lower[lamb >= 0] = 0.0
    log_upper[lamb <= 0] = 0.0

    scale = np.log(10.0)
    log_lower = np.nan_to_num(log_lower, nan=0.0, posinf=0.0, neginf=0.0) / scale
    log_upper = np.nan_to_num(log_upper, nan=0.0, posinf=0.0, neginf=0.0) / scale

    return {
        "average_profile": obs / float(source.n_obs if source.n_obs > 0 else 1),
        "upper_significance": log_upper,
        "lower_significance": log_lower,
    }


def compute_feature_specificity(
    adata: AnnData,
    labels: Union[str, np.ndarray],
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "specificity",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
) -> Optional[AnnData]:
    """
    Compute feature specificity scores for clusters/archetypes.

    Parameters
    ----------
    adata
        Annotated data matrix.
    labels
        Either key in adata.obs or array of cluster labels.
    layer
        Layer to use (None uses .X).
    n_threads
        Number of threads.
    key_added
        Key prefix for storing results in adata.varm.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk for streamed specificity computation on
        backed AnnData.  Ignored for in-memory objects.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.varm[f"{key_added}_profile"] : np.ndarray
        Average feature profile per cluster/archetype (features × clusters).
    adata.varm[f"{key_added}_upper"] : np.ndarray
        Upper-tail significance scores (features × clusters).
    adata.varm[f"{key_added}_lower"] : np.ndarray
        Lower-tail significance scores (features × clusters).
    """
    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    if isinstance(labels, str):
        if labels not in adata.obs:
            raise ValueError(f"Labels '{labels}' not found in adata.obs.")
        labels_arr = adata.obs[labels].values
    else:
        labels_arr = np.asarray(labels)

    # Normalise pandas Categorical to a plain object array so that the
    # subsequent Categorical() call always produces a deterministic
    # (lexicographic) category ordering, independent of any pre-existing
    # category order stored in the h5ad file.
    if hasattr(labels_arr, 'categories'):
        labels_arr = np.asarray(labels_arr)

    # Convert to integer labels
    from pandas import Categorical
    if not np.issubdtype(labels_arr.dtype, np.integer):
        cat = Categorical(labels_arr)
        labels_int = cat.codes.astype(np.int32)
    else:
        labels_int = labels_arr.astype(np.int32)

    # Function expects 1-based labels, so add 1
    labels_int = labels_int + 1

    if source.is_backed:
        H = _labels_to_membership(labels_int, source.n_obs)
        result = _compute_specificity_streamed(source, H, chunk_size=backed_chunk_size)
    else:
        S = anndata_to_matrix(adata, layer=layer, transpose=True)
        if sp.issparse(S):
            result = _core.compute_feature_specificity_sparse(S, labels_int, n_threads)
        else:
            result = _core.compute_feature_specificity_dense(S, labels_int, n_threads)

    persist_updates(
        adata,
        varm={
            f"{key_added}_profile": result["average_profile"],
            f"{key_added}_upper": result["upper_significance"],
            f"{key_added}_lower": result["lower_significance"],
        },
    )
    if not inplace:
        return adata
    return None


def compute_archetype_feature_specificity(
    adata: AnnData,
    archetype_key: Union[str, np.ndarray] = "archetype_footprint",
    layer: Optional[str] = None,
    n_threads: int = 0,
    key_added: str = "archetype",
    inplace: bool = True,
    backed_chunk_size: int = 4096,
) -> Optional[AnnData]:
    """
    Compute feature specificity scores for archetypes using archetype matrix.

    This function is analogous to archetypeFeatureSpecificity() in R.
    It computes feature enrichment for each archetype using the archetype
    footprint matrix (typically the diffused H_merged matrix).

    Parameters
    ----------
    adata
        Annotated data matrix.
    archetype_key
        Either key in adata.obsm containing archetype matrix (cells × archetypes)
        or the archetype matrix itself as numpy array.
    layer
        Layer to use (None uses .X).
    n_threads
        Number of threads.
    key_added
        Prefix for storing results in adata.varm.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.
    backed_chunk_size : int, optional (default: 4096)
        Number of rows per chunk for streamed specificity computation on
        backed AnnData.  Ignored for in-memory objects.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place.
        If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.varm[f"{key_added}_feat_profile"] : np.ndarray
        Average feature profile per archetype (features × archetypes).
    adata.varm[f"{key_added}_feat_specificity_upper"] : np.ndarray
        Upper-tail significance scores (features × archetypes).
    adata.varm[f"{key_added}_feat_specificity_lower"] : np.ndarray
        Lower-tail significance scores (features × archetypes).
    """
    if not inplace:
        adata = adata.copy()

    source = MatrixSource(adata, layer=layer)

    if isinstance(archetype_key, str):
        if archetype_key not in adata.obsm:
            raise ValueError(f"Archetype matrix '{archetype_key}' not found in adata.obsm.")
        H = adata.obsm[archetype_key]
    else:
        H = np.asarray(archetype_key)

    H = np.ascontiguousarray(H, dtype=np.float64)

    if source.is_backed:
        result_stream = _compute_specificity_streamed(source, H, chunk_size=backed_chunk_size)
        result = {
            "archetypes": result_stream["average_profile"],
            "upper_significance": result_stream["upper_significance"],
            "lower_significance": result_stream["lower_significance"],
        }
    else:
        S = anndata_to_matrix(adata, layer=layer, transpose=True)
        H_t = np.ascontiguousarray(H.T, dtype=np.float64)
        if sp.issparse(S):
            result = _core.archetype_feature_specificity_sparse(S, H_t, n_threads)
        else:
            result = _core.archetype_feature_specificity_dense(S, H_t, n_threads)

    persist_updates(
        adata,
        varm={
            f"{key_added}_feat_profile": result["archetypes"],
            f"{key_added}_feat_specificity_upper": result["upper_significance"],
            f"{key_added}_feat_specificity_lower": result["lower_significance"],
        },
    )

    if not inplace:
        return adata
    return None


def layout_network(
    adata: AnnData,
    network_key: str = "actionet",
    initial_coords: Optional[Union[str, np.ndarray]] = None,
    layer: Optional[str] = None,
    method: Literal["umap", "tumap"] = "umap",
    n_components: int = 2,
    spread: float = 1.0,
    min_dist: float = 1.0,
    n_epochs: int = 0,
    seed: int = 0,
    n_threads: int = 0,
    verbose: bool = True,
    key_added: str = "X_umap",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute 2D/3D layout of ACTIONet graph using UMAP.

    Parameters
    ----------
    adata
        Annotated data matrix with network.
    network_key
        Key in adata.obsp containing network.
    initial_coords
        Initial coordinates. Can be a key in adata.obsm, a numpy array, or None.
        If None, computes initial coordinates via SVD on the specified layer.
    layer
        Layer to use for computing initial coordinates via SVD (if initial_coords is None).
        If None, uses adata.X.
    method
        Layout method.
    n_components
        Number of dimensions (2 or 3).
    spread
        UMAP spread parameter.
    min_dist
        UMAP min_dist parameter.
    n_epochs
        Number of optimization epochs (0=auto).
    seed
        Random seed.
    n_threads
        Number of threads.
    verbose
        Whether to print progress messages.
    key_added
        Key to store layout in adata.obsm.
    inplace
        If True, modifies the AnnData object in place. If False, returns a new AnnData object with the results.

    Returns
    -------
    None or AnnData
        If inplace=True, returns None and modifies adata in place. If inplace=False, returns a new AnnData object with the results.

    Updates AnnData
    --------------
    adata.obsm[key_added] : np.ndarray
        Layout coordinates (cells × n_components).
    """
    if not inplace:
        adata = adata.copy()
    if network_key not in adata.obsp:
        raise ValueError(f"Network '{network_key}' not found.")
    
    G = adata.obsp[network_key]
    
    # Handle initial_coords
    if initial_coords is None:
        # Compute initial coordinates from SVD
        if verbose:
            if layer is not None:
                print(f"Computing initial coordinates from layer '{layer}' via SVD")
            else:
                print("Computing initial coordinates from adata.X via SVD")

        X = MatrixSource(adata, layer=layer).matrix
        k = max(3, n_components)
        svd_result = run_svd(
            X,
            n_components=k,
            algorithm=None,
            max_iter=0,
            seed=seed,
            verbose=verbose,
            return_operator_compatible=True,
        )

        # Get right singular vectors and scale them
        initial_coords = svd_result["v"]  # Already transposed to cells x components
        # Scale columns to have mean 0 and std 1
        initial_coords = (initial_coords - initial_coords.mean(axis=0)) / initial_coords.std(axis=0)
    elif isinstance(initial_coords, str):
        # initial_coords is a key in adata.obsm
        if initial_coords not in adata.obsm:
            raise ValueError(f"Initial coordinates '{initial_coords}' not found in adata.obsm.")
        initial_coords = adata.obsm[initial_coords]
    else:
        # initial_coords is a numpy array
        initial_coords = np.asarray(initial_coords)

    # Validate initial_coords shape
    if initial_coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"Number of rows in initial_coords ({initial_coords.shape[0]}) "
            f"does not match number of cells in adata ({adata.n_obs})"
        )

    if initial_coords.shape[1] < n_components:
        raise ValueError(
            f"Number of columns in initial_coords ({initial_coords.shape[1]}) "
            f"must be >= n_components ({n_components})"
        )

    # Ensure initial_coords is float32 and C-contiguous
    initial_coords = np.ascontiguousarray(initial_coords, dtype=np.float32)

    coords = _core.layout_network(
        G, initial_coords, method, n_components,
        spread, min_dist, n_epochs, seed, n_threads, verbose
    )

    persist_updates(adata, obsm={key_added: coords})
    if not inplace:
        return adata
    return None


def run_svd(
    X: Union[np.ndarray, sp.spmatrix, Any],
    n_components: int = 30,
    algorithm: Union[int] = None,
    max_iter: int = 0,
    seed: int = 0,
    verbose: bool = True,
    return_operator_compatible: bool = True,
    backed_chunk_size: int = 4096,
) -> dict:
    """
    Compute truncated SVD decomposition.

    Transposes X internally (to features × cells) before decomposing, so the
    caller should pass the matrix in its natural obs × vars orientation.

    Parameters
    ----------
    X : numpy.ndarray, scipy.sparse matrix, or backed sparse dataset
        Matrix to decompose (obs × vars, e.g. cells × genes).
    n_components : int
        Number of singular components to compute.
    algorithm : int or None
        SVD algorithm code (0=IRLB, 1=Halko, 2=Feng, 3=PRIMME).
        None enables automatic selection.
    max_iter : int
        Maximum iterations (0 = auto).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress messages.
    return_operator_compatible : bool
        If True (default), normalise output keys to ``{"u", "d", "v"}`` so the
        result can be passed directly as ``precomputed_svd`` to :func:`reduce_kernel`.
    backed_chunk_size : int
        Chunk size for the operator path when X is a backed sparse dataset.

    Returns
    -------
    dict
        ``{"u": ndarray, "d": ndarray, "v": ndarray}`` — left singular vectors
        (features × k), singular values (k,), and right singular vectors (cells × k).
    """

    if _is_backed_matrix(X):
        _warn_if_compressed_backed_svd(
            get_storage_metadata_from_matrix(X),
            context="run_svd",
            recommendation='actionet.decompress_backed_storage(adata, layer=None, scope="matrix")',
        )
        algorithm = 3
        op = _TransposeMatrixOperator(X, chunk_size=backed_chunk_size)
        result = _core.run_svd_operator(op, n_components, max_iter, seed, verbose)
    elif sp.issparse(X):
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()

        algorithm = _select_svd_algorithm(X, algorithm, verbose)
        result = _core.run_svd_sparse(X.T, n_components, max_iter, seed, algorithm, verbose)
    else:
        algorithm = _select_svd_algorithm(X, algorithm, verbose)
        result = _core.run_svd_dense(X.T, n_components, max_iter, seed, algorithm, verbose)

    if return_operator_compatible:
        # Ensure fields expected by reduce_kernel(..., precomputed_svd=...) are always present.
        result = {
            "u": result["u"],
            "d": result["d"],
            "v": result["v"],
        }

    return result
