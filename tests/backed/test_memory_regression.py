"""Parity checks and RSS micro-benchmark for the three backed-mode stages
affected by the orientation-unification memory regressions:
  - reduce_kernel
  - build_network
  - compute_network_diffusion

Part A tests assert structural equivalence and contiguity.
Part B prints peak RSS deltas per stage (informational, always passes).
"""

import gc
import os
import sys
import threading
import time

import numpy as np
import pytest
import scipy.sparse as sp

from .conftest import make_test_adata, open_backed

try:
    import actionet as an

    _has_ext = hasattr(an, "_core")
except Exception:
    _has_ext = False

requires_ext = pytest.mark.skipif(not _has_ext, reason="C extension not built")

_N_CELLS = 96
_N_GENES = 72
_SEED = 42
_N_COMPONENTS = 12
_K_MIN = 2
_K_MAX = 6
_CHUNK = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_pair(tmp_path):
    """Return (adata_mem, adata_backed) both normalised."""
    adata_mem = make_test_adata(n_cells=_N_CELLS, n_genes=_N_GENES, sparse_fmt="csr", seed=_SEED)
    an.normalize_anndata(adata_mem, target_sum=1e4, inplace=True)
    adata_backed = open_backed(tmp_path, adata_mem)
    return adata_mem, adata_backed


def _run_through_build_network(adata, *, backed_chunk_size=None):
    """Run reduce_kernel -> run_action -> build_network on *adata*."""
    kw = {"backed_chunk_size": backed_chunk_size} if backed_chunk_size else {}
    an.reduce_kernel(adata, n_components=_N_COMPONENTS, seed=1, inplace=True, **kw)
    an.run_action(
        adata,
        reduction_key="action",
        k_min=_K_MIN,
        k_max=_K_MAX,
        n_threads=1,
        inplace=True,
    )
    an.build_network(adata, obsm_key="H_stacked", key_added="actionet", n_threads=1, inplace=True)


def _subspace_overlap(A, B):
    """Compute mean principal angle cosine between column spaces of A and B.

    Returns a value in [0, 1]; 1.0 means identical subspaces.
    """
    Q_a, _ = np.linalg.qr(A)
    Q_b, _ = np.linalg.qr(B)
    svs = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    return float(np.mean(np.minimum(svs, 1.0)))


# ===================================================================
# Part A: Parity checks
# ===================================================================


@requires_ext
class TestReduceKernelParity:
    """Backed vs in-memory reduce_kernel must produce structurally equivalent results."""

    def test_shapes_match(self, tmp_path):
        adata_mem, adata_backed = _prepare_pair(tmp_path)

        an.reduce_kernel(adata_mem, n_components=_N_COMPONENTS, seed=1, inplace=True)
        an.reduce_kernel(
            adata_backed, n_components=_N_COMPONENTS, seed=1,
            backed_chunk_size=_CHUNK, inplace=True,
        )

        assert adata_mem.obsm["action"].shape == adata_backed.obsm["action"].shape
        assert adata_mem.varm["action_U"].shape == adata_backed.varm["action_U"].shape

    def test_subspace_overlap(self, tmp_path):
        """IRLB vs Halko produce different bases; check subspace overlap >= 0.8."""
        adata_mem, adata_backed = _prepare_pair(tmp_path)

        an.reduce_kernel(adata_mem, n_components=_N_COMPONENTS, seed=1, inplace=True)
        an.reduce_kernel(
            adata_backed, n_components=_N_COMPONENTS, seed=1,
            backed_chunk_size=_CHUNK, inplace=True,
        )

        overlap = _subspace_overlap(adata_mem.obsm["action"], adata_backed.obsm["action"])
        assert overlap >= 0.80, f"Subspace overlap too low: {overlap:.3f}"

    def test_output_contiguity(self, tmp_path):
        adata_mem, adata_backed = _prepare_pair(tmp_path)

        an.reduce_kernel(adata_mem, n_components=_N_COMPONENTS, seed=1, inplace=True)
        an.reduce_kernel(
            adata_backed, n_components=_N_COMPONENTS, seed=1,
            backed_chunk_size=_CHUNK, inplace=True,
        )

        for key in ("action", "action_B"):
            if key in adata_mem.obsm:
                arr = np.asarray(adata_mem.obsm[key])
                assert arr.flags["C_CONTIGUOUS"], f"in-memory obsm['{key}'] not C-contiguous"
            if key in adata_backed.obsm:
                arr = np.asarray(adata_backed.obsm[key])
                assert arr.flags["C_CONTIGUOUS"], f"backed obsm['{key}'] not C-contiguous"


@requires_ext
class TestBuildNetworkParity:
    """Backed vs in-memory build_network must produce structurally similar graphs."""

    def test_parity(self, tmp_path):
        adata_mem, adata_backed = _prepare_pair(tmp_path)

        _run_through_build_network(adata_mem)
        _run_through_build_network(adata_backed, backed_chunk_size=_CHUNK)

        g_mem = adata_mem.obsp["actionet"]
        g_bak = adata_backed.obsp["actionet"]

        assert g_mem.shape == g_bak.shape, "Graph shape mismatch"
        # NNZ may differ slightly due to different SVD bases; check within 30%.
        ratio = g_bak.nnz / max(g_mem.nnz, 1)
        assert 0.7 <= ratio <= 1.3, f"Graph NNZ ratio out of range: {ratio:.2f}"


@requires_ext
class TestNetworkDiffusionParity:
    """Diffusion from the same graph and scores must give equivalent results."""

    def test_self_consistency(self, tmp_path):
        """Run diffusion twice on the same in-memory data; results must match exactly."""
        adata_mem, _ = _prepare_pair(tmp_path)
        _run_through_build_network(adata_mem)

        an.compute_network_diffusion(
            adata_mem, scores="H_merged", network_key="actionet",
            key_added="archetype_footprint", n_threads=1, inplace=True,
        )
        fp1 = adata_mem.obsm["archetype_footprint"].copy()

        an.compute_network_diffusion(
            adata_mem, scores="H_merged", network_key="actionet",
            key_added="archetype_footprint", n_threads=1, inplace=True,
        )
        fp2 = adata_mem.obsm["archetype_footprint"]

        np.testing.assert_allclose(fp1, fp2, atol=1e-10,
                                   err_msg="Diffusion not deterministic")

    def test_backed_runs_without_error(self, tmp_path):
        """Backed mode pipeline through diffusion completes successfully."""
        _, adata_backed = _prepare_pair(tmp_path)
        _run_through_build_network(adata_backed, backed_chunk_size=_CHUNK)
        an.compute_network_diffusion(
            adata_backed, scores="H_merged", network_key="actionet",
            key_added="archetype_footprint", n_threads=1, inplace=True,
        )
        fp = adata_backed.obsm["archetype_footprint"]
        assert fp.shape[0] == _N_CELLS
        assert fp.shape[1] > 0
        assert np.all(np.isfinite(fp)), "Diffusion output contains non-finite values"


# ===================================================================
# Part B: RSS micro-benchmark
# ===================================================================

def _get_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def _measure_peak_rss(func):
    """Run *func* while sampling RSS; return peak delta in MB."""
    gc.collect()
    baseline = _get_rss_mb()
    peak = baseline
    stop_event = threading.Event()

    def sampler():
        nonlocal peak
        while not stop_event.is_set():
            current = _get_rss_mb()
            if current > peak:
                peak = current
            time.sleep(0.005)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        func()
    finally:
        stop_event.set()
        t.join(timeout=1.0)
    return peak - baseline


@requires_ext
@pytest.mark.benchmark
class TestRSSBenchmark:
    """Measure peak RSS delta for each stage in both modes.

    Always passes -- informational output only.  Run with ``-s`` to see output.
    """

    def test_rss_stages(self, tmp_path):
        results = {}

        for mode in ("in_memory", "backed"):
            adata_base = make_test_adata(
                n_cells=_N_CELLS, n_genes=_N_GENES, sparse_fmt="csr", seed=_SEED,
            )
            an.normalize_anndata(adata_base, target_sum=1e4, inplace=True)

            if mode == "backed":
                adata = open_backed(tmp_path / mode, adata_base)
                kw = {"backed_chunk_size": _CHUNK}
            else:
                adata = adata_base
                kw = {}

            results[("reduce_kernel", mode)] = _measure_peak_rss(
                lambda: an.reduce_kernel(
                    adata, n_components=_N_COMPONENTS, seed=1, inplace=True, **kw,
                )
            )

            an.run_action(
                adata, reduction_key="action",
                k_min=_K_MIN, k_max=_K_MAX, n_threads=1, inplace=True,
            )

            results[("build_network", mode)] = _measure_peak_rss(
                lambda: an.build_network(
                    adata, obsm_key="H_stacked", key_added="actionet",
                    n_threads=1, inplace=True,
                )
            )

            results[("network_diffusion", mode)] = _measure_peak_rss(
                lambda: an.compute_network_diffusion(
                    adata, scores="H_merged", network_key="actionet",
                    key_added="archetype_footprint", n_threads=1, inplace=True,
                )
            )

        print("\n--- RSS Micro-Benchmark (peak RSS delta in MB) ---")
        for stage in ("reduce_kernel", "build_network", "network_diffusion"):
            bak = results.get((stage, "backed"), 0.0)
            mem = results.get((stage, "in_memory"), 0.0)
            print(f"[RSS] {stage:<24s} backed: {bak:7.1f} MB  in_memory: {mem:7.1f} MB")
        print("---------------------------------------------------")
