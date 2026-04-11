"""Tests for checkpoint_backed: flush + compact round-trip."""

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import pytest

from actionet._backed_persist import (
    checkpoint_backed,
    is_backed_adata,
    persist_updates,
)

from .conftest import make_test_adata, open_backed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_slots(adata: ad.AnnData, rng: np.random.Generator) -> dict:
    """Write synthetic data into every annotation slot via persist_updates."""
    n = adata.n_obs
    p = adata.n_vars

    obsm_emb = rng.standard_normal((n, 5)).astype(np.float32)
    varm_scores = rng.standard_normal((p, 3)).astype(np.float64)
    obsp_graph = sp.random(n, n, density=0.05, format="csr", random_state=42)
    obs_labels = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    uns_params = {"alpha": 0.5, "k": 10}

    persist_updates(
        adata,
        obs={"cluster": obs_labels},
        obsm={"X_embed": obsm_emb},
        varm={"scores": varm_scores},
        obsp={"graph": obsp_graph},
        uns={"params": uns_params},
    )

    return {
        "obs_cluster": np.asarray(obs_labels),
        "obsm_embed": obsm_emb,
        "varm_scores": varm_scores,
        "obsp_graph": obsp_graph,
        "uns_params": uns_params,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointBacked:
    """checkpoint_backed flush and round-trip correctness."""

    @pytest.fixture()
    def backed_adata(self, tmp_path):
        mem = make_test_adata(n_cells=48, n_genes=36, seed=99)
        return open_backed(tmp_path, mem)

    def test_raises_on_inmemory(self):
        mem = make_test_adata(n_cells=10, n_genes=8)
        with pytest.raises(ValueError, match="backed"):
            checkpoint_backed(mem)

    def test_raises_on_readonly(self, tmp_path):
        mem = make_test_adata(n_cells=10, n_genes=8)
        path = tmp_path / "ro.h5ad"
        mem.write_h5ad(path)
        ro = ad.read_h5ad(path, backed="r")
        with pytest.raises(ValueError, match="read-only"):
            checkpoint_backed(ro)

    def test_roundtrip_slots(self, backed_adata, tmp_path):
        """Slots written via persist_updates survive checkpoint + reopen."""
        rng = np.random.default_rng(7)
        expected = _populate_slots(backed_adata, rng)

        checkpoint_backed(backed_adata, verbose=True)

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")

        np.testing.assert_array_equal(
            np.asarray(reloaded.obs["cluster"]),
            expected["obs_cluster"],
        )
        np.testing.assert_allclose(
            reloaded.obsm["X_embed"],
            expected["obsm_embed"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            reloaded.varm["scores"],
            expected["varm_scores"],
            rtol=1e-6,
        )
        reloaded_graph = reloaded.obsp["graph"]
        if sp.issparse(reloaded_graph):
            reloaded_graph = reloaded_graph.toarray()
        expected_graph = expected["obsp_graph"]
        if sp.issparse(expected_graph):
            expected_graph = expected_graph.toarray()
        np.testing.assert_allclose(reloaded_graph, expected_graph, rtol=1e-6)

        assert reloaded.uns["params"]["alpha"] == expected["uns_params"]["alpha"]
        assert reloaded.uns["params"]["k"] == expected["uns_params"]["k"]

        reloaded.file.close()

    def test_noop_checkpoint(self, tmp_path):
        """Checkpoint with no in-memory annotations does not error."""
        mem = make_test_adata(n_cells=10, n_genes=8)
        backed = open_backed(tmp_path, mem)

        size_before = backed.filename.stat().st_size
        checkpoint_backed(backed)
        size_after = backed.filename.stat().st_size

        assert size_after >= size_before
        backed.file.close()


class TestCheckpointCompact:
    """checkpoint_backed with compact=True reclaims dead space."""

    @pytest.fixture()
    def backed_adata(self, tmp_path):
        mem = make_test_adata(n_cells=48, n_genes=36, seed=99)
        return open_backed(tmp_path, mem)

    def test_compact_reduces_size(self, backed_adata):
        """Repeated overwrites inflate the file; compact shrinks it back."""
        rng = np.random.default_rng(42)

        for _ in range(5):
            _populate_slots(backed_adata, rng)

        checkpoint_backed(backed_adata)
        size_bloated = backed_adata.filename.stat().st_size

        checkpoint_backed(backed_adata, compact=True)

        size_compacted = backed_adata.filename.stat().st_size
        assert size_compacted <= size_bloated

    def test_handle_works_after_compact(self, backed_adata):
        """AnnData handle is usable after compact (refresh succeeded)."""
        rng = np.random.default_rng(11)
        expected = _populate_slots(backed_adata, rng)

        checkpoint_backed(backed_adata, compact=True)

        assert is_backed_adata(backed_adata)
        np.testing.assert_allclose(
            backed_adata.obsm["X_embed"],
            expected["obsm_embed"],
            rtol=1e-6,
        )
        assert backed_adata.uns["params"]["alpha"] == 0.5

        backed_adata.file.close()

    def test_compact_preserves_X(self, backed_adata):
        """The primary .X matrix is intact after compact."""
        import h5py

        path = str(backed_adata.filename)
        with h5py.File(path, "r") as f:
            has_X_before = "X" in f

        rng = np.random.default_rng(3)
        _populate_slots(backed_adata, rng)
        checkpoint_backed(backed_adata, compact=True)

        with h5py.File(path, "r") as f:
            has_X_after = "X" in f

        assert has_X_before == has_X_after

        backed_adata.file.close()

    def test_compact_preserves_compression(self, tmp_path):
        """Datasets that were gzip-compressed remain compressed after compact."""
        import h5py

        mem = make_test_adata(n_cells=48, n_genes=36, seed=7)
        path = tmp_path / "compressed.h5ad"
        mem.write_h5ad(path)

        with h5py.File(path, "r") as f:
            x_node = f["X"]
            if hasattr(x_node, "keys") and "data" in x_node:
                original_compression = x_node["data"].compression
            elif hasattr(x_node, "compression"):
                original_compression = x_node.compression
            else:
                original_compression = None

        backed = ad.read_h5ad(path, backed="r+")
        rng = np.random.default_rng(3)
        _populate_slots(backed, rng)
        checkpoint_backed(backed, compact=True)

        with h5py.File(str(backed.filename), "r") as f:
            x_node = f["X"]
            if hasattr(x_node, "keys") and "data" in x_node:
                after_compression = x_node["data"].compression
            elif hasattr(x_node, "compression"):
                after_compression = x_node.compression
            else:
                after_compression = None

        assert original_compression == after_compression

        backed.file.close()
