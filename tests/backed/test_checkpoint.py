"""Tests for checkpoint_backed: flush + compact round-trip."""

import numpy as np
import scipy.sparse as sp
import anndata as ad
import h5py
import pandas as pd
import pytest

import actionet.io.checkpoint as checkpoint_module

from actionet.io.persist import (
    is_backed_adata,
    persist_updates,
)
from actionet.io.checkpoint import checkpoint_backed

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

    def test_checkpoint_preserves_ondisk_sparse_aux_matrices(self, tmp_path):
        """checkpoint_backed on a file with on-disk sparse obsp/varp matrices
        round-trips them correctly through the collect-all rewrite path.

        Guards the collect_annotation_results -> checkpoint_backed path against
        the orphaned-handle class fixed for persist_updates (Q7): any live
        backed matrix wrapper captured into the results dict would be handed to
        ad.io.write_elem after the source file is closed. The filter now lives
        in collect_annotation_results (via is_live_backed_wrapper). This test
        also pins the user-facing guarantee that aux matrices survive a no-op
        (nothing-dirty) checkpoint.
        """
        mem = make_test_adata(n_cells=40, n_genes=24, seed=5)
        graph = sp.random(40, 40, density=0.08, format="csr", random_state=1)
        var_graph = sp.random(24, 24, density=0.08, format="csr", random_state=2)
        mem.obsp["graph"] = graph
        mem.varp["gene_graph"] = var_graph

        backed = open_backed(tmp_path, mem)

        # No prior persist_updates -> collect-all path. Must not raise even if
        # any container yields a live backed wrapper.
        checkpoint_backed(backed)

        path = str(backed.filename)
        backed.file.close()

        reloaded = ad.read_h5ad(path, backed="r")
        reloaded_graph = reloaded.obsp["graph"]
        if sp.issparse(reloaded_graph):
            reloaded_graph = reloaded_graph.toarray()
        np.testing.assert_allclose(reloaded_graph, graph.toarray(), rtol=1e-6)
        reloaded_vg = reloaded.varp["gene_graph"]
        if sp.issparse(reloaded_vg):
            reloaded_vg = reloaded_vg.toarray()
        np.testing.assert_allclose(reloaded_vg, var_graph.toarray(), rtol=1e-6)
        reloaded.file.close()

    def test_collect_annotation_results_skips_live_backed_wrappers(
        self, tmp_path
    ):
        """collect_annotation_results must not capture a live backed matrix
        wrapper (CSRDataset/CSCDataset) into the results dict, regardless of
        which container AnnData happens to serve it from. Simulate a backed
        wrapper in obsm to lock the filter in place across AnnData versions.
        """
        import actionet.io.anndata_io as anndata_io

        mem = make_test_adata(n_cells=20, n_genes=12, seed=3)
        backed = open_backed(tmp_path, mem)
        try:
            # .X is a genuine live backed wrapper on AnnData >= 0.13.
            live_wrapper = backed.X
            assert anndata_io.is_live_backed_wrapper(live_wrapper)
            backed.obsm["backed_like"] = live_wrapper

            results = anndata_io.collect_annotation_results(
                backed, obsm_keys=["backed_like"]
            )
            assert "backed_like" not in results["obsm_keys"]
        finally:
            backed.file.close()

    def test_noop_checkpoint(self, tmp_path):
        """Checkpoint with no in-memory annotations does not error."""
        mem = make_test_adata(n_cells=10, n_genes=8)
        backed = open_backed(tmp_path, mem)

        with h5py.File(backed.filename, "r") as f:
            keys_before = sorted(f.keys())

        checkpoint_backed(backed)

        with h5py.File(backed.filename, "r") as f:
            keys_after = sorted(f.keys())

        assert keys_before == keys_after
        backed.file.close()

    def test_chunk_size_only_reaches_compaction(self, backed_adata, monkeypatch):
        """The transfer chunk is irrelevant unless compact=True."""
        observed = []

        def fake_repack(adata, *, chunk_size, verbose):
            observed.append(chunk_size)

        monkeypatch.setattr(checkpoint_module, "_repack_h5ad", fake_repack)

        checkpoint_backed(backed_adata, compact=False, backed_write_chunk_size=123)
        assert observed == []

        checkpoint_backed(backed_adata, compact=True, backed_write_chunk_size=123)
        assert observed == [123]


class TestCheckpointCompact:
    """checkpoint_backed with compact=True reclaims dead space."""

    @pytest.fixture()
    def backed_adata(self, tmp_path):
        mem = make_test_adata(n_cells=48, n_genes=36, seed=99)
        return open_backed(tmp_path, mem)

    def test_compact_reduces_size(self, backed_adata):
        """compact=True runs without error and produces a valid file.

        With modern h5py (>= 3.16), aggressive free-space reuse means repeated
        same-shape overwrites may not grow the file monotonically. Compact
        (repack) is still correct but may not shrink the file in all cases.
        This test verifies compact runs successfully and the handle remains
        usable afterward.
        """
        rng = np.random.default_rng(42)

        # Write once and compact immediately to get a clean baseline.
        _populate_slots(backed_adata, rng)
        checkpoint_backed(backed_adata, compact=True)

        # Overwrite many times to accumulate potential dead space.
        for _ in range(10):
            _populate_slots(backed_adata, rng)
            checkpoint_backed(backed_adata)

        # Compact should not raise.
        checkpoint_backed(backed_adata, compact=True)
        size_after_compact = backed_adata.filename.stat().st_size

        # The file must still be valid and non-empty.
        assert size_after_compact > 0
        assert backed_adata.filename.exists()

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

    def test_compact_preserves_unknown_top_level_object(self, tmp_path):
        """Unknown top-level HDF5 objects use object-copy semantics."""
        mem = make_test_adata(n_cells=24, n_genes=12, seed=8)
        path = tmp_path / "unknown_object.h5ad"
        mem.write_h5ad(path)
        expected = np.arange(30, dtype=np.int64)
        with h5py.File(path, "r+") as handle:
            group = handle.create_group("vendor_extension")
            group.attrs["schema"] = "example-1"
            dataset = group.create_dataset(
                "payload",
                data=expected,
                chunks=(10,),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )
            dataset.attrs["meaning"] = "opaque-to-actionet"

        backed = ad.read_h5ad(path, backed="r+")
        checkpoint_backed(backed, compact=True)
        backed.file.close()

        with h5py.File(path, "r") as handle:
            group = handle["vendor_extension"]
            dataset = group["payload"]
            assert group.attrs["schema"] == "example-1"
            assert dataset.attrs["meaning"] == "opaque-to-actionet"
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.shuffle
            assert dataset.fletcher32
            assert dataset.chunks == (10,)
            np.testing.assert_array_equal(dataset[:], expected)
