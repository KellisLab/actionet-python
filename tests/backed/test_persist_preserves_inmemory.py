"""Regression tests: persist_updates must not discard in-memory user modifications.

Verifies that obs/var columns, obsm keys, and uns entries that the user
added or modified directly on a backed AnnData survive a subsequent
persist_updates call (which triggers an atomic file rewrite + handle refresh).
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from actionet._backed_persist import persist_updates, is_backed_adata
from .conftest import make_test_adata, open_backed


class TestPersistPreservesInmemoryState:
    """persist_updates must preserve user-modified annotations on backed objects."""

    @pytest.fixture()
    def backed_adata(self, tmp_path):
        mem = make_test_adata(n_cells=48, n_genes=36, seed=42)
        return open_backed(tmp_path, mem)

    def test_new_obs_column_survives_persist(self, backed_adata):
        """A user-added obs column must survive a subsequent persist_updates."""
        n = backed_adata.n_obs
        user_labels = np.array([f"label_{i % 5}" for i in range(n)])
        backed_adata.obs["user_col"] = user_labels

        persist_updates(
            backed_adata,
            obsm={"embedding": np.random.default_rng(7).standard_normal((n, 3))},
        )

        assert "user_col" in backed_adata.obs.columns
        np.testing.assert_array_equal(backed_adata.obs["user_col"].values, user_labels)

        # Verify on-disk persistence by reopening
        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        assert "user_col" in reloaded.obs.columns
        np.testing.assert_array_equal(reloaded.obs["user_col"].values, user_labels)
        reloaded.file.close()

    def test_modified_existing_obs_column_survives_persist(self, backed_adata):
        """Modifications to an existing obs column must survive persist_updates."""
        n = backed_adata.n_obs
        new_labels = np.array([f"modified_{i}" for i in range(n)])
        backed_adata.obs["CellLabel"] = new_labels

        persist_updates(
            backed_adata,
            obs={"score": np.arange(n, dtype=np.float64)},
        )

        np.testing.assert_array_equal(backed_adata.obs["CellLabel"].values, new_labels)

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        np.testing.assert_array_equal(reloaded.obs["CellLabel"].values, new_labels)
        reloaded.file.close()

    def test_new_var_column_survives_persist(self, backed_adata):
        """A user-added var column must survive persist_updates."""
        p = backed_adata.n_vars
        var_scores = np.random.default_rng(3).standard_normal(p)
        backed_adata.var["importance"] = var_scores

        persist_updates(
            backed_adata,
            obsm={"X_test": np.zeros((backed_adata.n_obs, 2))},
        )

        assert "importance" in backed_adata.var.columns
        np.testing.assert_allclose(backed_adata.var["importance"].values, var_scores)

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        assert "importance" in reloaded.var.columns
        np.testing.assert_allclose(reloaded.var["importance"].values, var_scores)
        reloaded.file.close()

    def test_new_obsm_key_survives_persist(self, backed_adata):
        """A user-added obsm entry must survive persist_updates."""
        n = backed_adata.n_obs
        user_embedding = np.random.default_rng(5).standard_normal((n, 10))
        backed_adata.obsm["X_user_pca"] = user_embedding

        persist_updates(
            backed_adata,
            obs={"cluster": np.zeros(n, dtype=np.int32)},
        )

        assert "X_user_pca" in backed_adata.obsm
        np.testing.assert_allclose(backed_adata.obsm["X_user_pca"], user_embedding)

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        assert "X_user_pca" in reloaded.obsm
        np.testing.assert_allclose(reloaded.obsm["X_user_pca"], user_embedding)
        reloaded.file.close()

    def test_new_uns_entry_survives_persist(self, backed_adata):
        """A user-added uns entry must survive persist_updates."""
        n = backed_adata.n_obs
        backed_adata.uns["user_param"] = {"alpha": 0.9, "k": 15}

        persist_updates(
            backed_adata,
            obsm={"X_layout": np.zeros((n, 2))},
        )

        assert "user_param" in backed_adata.uns
        assert backed_adata.uns["user_param"]["alpha"] == 0.9
        assert backed_adata.uns["user_param"]["k"] == 15

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        assert "user_param" in reloaded.uns
        assert reloaded.uns["user_param"]["alpha"] == 0.9
        assert reloaded.uns["user_param"]["k"] == 15
        reloaded.file.close()

    def test_multiple_persist_calls_preserve_all(self, backed_adata):
        """Multiple persist_updates calls must not drop prior user modifications."""
        n = backed_adata.n_obs
        rng = np.random.default_rng(11)

        # User adds a column
        backed_adata.obs["manual_annotation"] = np.array(
            [f"type_{i % 3}" for i in range(n)]
        )

        # First ACTIONet function persists
        persist_updates(
            backed_adata,
            obsm={"H_stacked": rng.standard_normal((n, 20)).astype(np.float32)},
        )

        # User adds another column
        backed_adata.obs["quality_score"] = rng.uniform(0, 1, size=n)

        # Second ACTIONet function persists
        persist_updates(
            backed_adata,
            obs={"assigned_archetype": np.arange(n, dtype=np.int32)},
        )

        # All must survive
        assert "manual_annotation" in backed_adata.obs.columns
        assert "quality_score" in backed_adata.obs.columns
        assert "assigned_archetype" in backed_adata.obs.columns
        assert "H_stacked" in backed_adata.obsm

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        assert "manual_annotation" in reloaded.obs.columns
        assert "quality_score" in reloaded.obs.columns
        assert "assigned_archetype" in reloaded.obs.columns
        assert "H_stacked" in reloaded.obsm
        reloaded.file.close()

    def test_persist_priority_for_current_call(self, backed_adata):
        """Keys explicitly passed to persist_updates take priority over in-memory state."""
        n = backed_adata.n_obs

        # User sets a value in obs
        backed_adata.obs["score"] = np.ones(n)

        # persist_updates is called with a DIFFERENT value for the same key
        new_scores = np.arange(n, dtype=np.float64)
        persist_updates(backed_adata, obs={"score": new_scores})

        # The persist_updates value wins
        np.testing.assert_array_equal(backed_adata.obs["score"].values, new_scores)

        path = str(backed_adata.filename)
        backed_adata.file.close()
        reloaded = ad.read_h5ad(path, backed="r+")
        np.testing.assert_array_equal(reloaded.obs["score"].values, new_scores)
        reloaded.file.close()
