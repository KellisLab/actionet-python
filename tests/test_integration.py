"""Integration tests for complete ACTIONet pipeline."""

import pytest
import numpy as np
import actionet as an


def test_full_pipeline(synthetic_adata):
    """Test a complete ACTION pipeline."""
    adata = synthetic_adata.copy()

    # Step 1: Reduce kernel
    an.reduce_kernel(adata, k=15, reduction_key="action")
    assert "action" in adata.obsm
    assert adata.obsm["action"].shape == (adata.n_obs, 15)
    assert "action_params" in adata.uns

    # Step 2: Build network
    an.build_network(adata, reduction_key="action", network_key="actionet")
    assert "actionet" in adata.obsp
    assert adata.obsp["actionet"].shape == (adata.n_obs, adata.n_obs)

    # Step 3: Run ACTION
    an.run_action(adata, k_min=3, k_max=8, reduction_key="action")
    assert "H_merged" in adata.obsm
    assert "C_merged" in adata.varm
    assert "assigned_archetypes" in adata.obs

    # Step 4: Compute feature specificity
    an.compute_feature_specificity(adata, H_key="H_merged")
    assert "feature_specificity" in adata.varm

    # Verify shapes and basic properties
    n_archetypes = adata.obsm["H_merged"].shape[1]
    assert adata.varm["C_merged"].shape == (adata.n_vars, n_archetypes)
    assert adata.obs["assigned_archetypes"].nunique() == n_archetypes


def test_batch_correction_pipeline(adata_with_reduction):
    """Test batch correction in a pipeline."""
    adata = adata_with_reduction.copy()

    # Add batch labels
    n_cells = adata.n_obs
    adata.obs["batch"] = ["batch1"] * (n_cells // 2) + ["batch2"] * (n_cells - n_cells // 2)

    # Correct batch effect
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        reduction_key="action",
        corrected_key="action_corrected"
    )

    assert "action_corrected" in adata.obsm
    assert "action_corrected_params" in adata.uns

    # Build network on corrected reduction
    an.build_network(adata, reduction_key="action_corrected", network_key="actionet_corrected")
    assert "actionet_corrected" in adata.obsp


def test_imputation_pipeline(adata_with_network):
    """Test imputation in a pipeline."""
    adata = adata_with_network.copy()

    # Select genes to impute
    genes_to_impute = adata.var_names[:10].tolist()

    # Impute features
    X_imputed = an.impute_features(
        adata,
        features=genes_to_impute,
        network_key="actionet",
        alpha=0.85
    )

    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))

    # Smooth kernel
    an.smooth_kernel(
        adata,
        reduction_key="action",
        smoothed_key="action_smoothed",
        alpha=0.85
    )

    assert "action_smoothed" in adata.obsm


def test_basal_correction_pipeline(adata_with_reduction):
    """Test basal expression correction in a pipeline."""
    adata = adata_with_reduction.copy()

    # Select some genes as basal
    basal_genes = adata.var_names[:10].tolist()

    # Correct basal expression
    an.correct_basal_expression(
        adata,
        basal_genes=basal_genes,
        reduction_key="action",
        corrected_key="action_basal_corrected"
    )

    assert "action_basal_corrected" in adata.obsm
    assert "action_basal_corrected_params" in adata.uns
    assert "basal_genes" in adata.uns["action_basal_corrected_params"]


def test_multiple_reductions(synthetic_adata):
    """Test multiple reductions can coexist."""
    adata = synthetic_adata.copy()

    # Create multiple reductions
    an.reduce_kernel(adata, k=10, reduction_key="action_10")
    an.reduce_kernel(adata, k=20, reduction_key="action_20")

    assert "action_10" in adata.obsm
    assert "action_20" in adata.obsm
    assert adata.obsm["action_10"].shape[1] == 10
    assert adata.obsm["action_20"].shape[1] == 20


def test_multiple_networks(adata_with_reduction):
    """Test multiple networks can be built."""
    adata = adata_with_reduction.copy()

    # Build networks with different parameters
    an.build_network(
        adata,
        reduction_key="action",
        network_key="actionet_k5",
        k=5
    )
    an.build_network(
        adata,
        reduction_key="action",
        network_key="actionet_k10",
        k=10
    )

    assert "actionet_k5" in adata.obsp
    assert "actionet_k10" in adata.obsp
    # Different k should produce different networks
    assert not np.array_equal(
        adata.obsp["actionet_k5"].toarray(),
        adata.obsp["actionet_k10"].toarray()
    )


def test_pipeline_with_layers(synthetic_adata):
    """Test pipeline can work with layers."""
    adata = synthetic_adata.copy()

    # Add a normalized layer
    from scipy.sparse import csr_matrix
    adata.layers["normalized"] = csr_matrix(adata.X / (adata.X.sum(axis=1) + 1))

    # Run reduction on layer
    an.reduce_kernel(adata, k=15, layer="normalized", reduction_key="action")

    assert "action" in adata.obsm
    assert adata.obsm["action"].shape[1] == 15


def test_archetype_imputation_pipeline(adata_with_archetypes):
    """Test imputation from archetypes."""
    adata = adata_with_archetypes.copy()

    # Select genes to impute
    genes_to_impute = adata.var_names[:10].tolist()

    # Impute from archetypes
    X_imputed = an.impute_from_archetypes(
        adata,
        features=genes_to_impute,
        H_key="H_merged"
    )

    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))
    assert not np.allclose(X_imputed, 0)
