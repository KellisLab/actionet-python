"""Tests for imputation functions."""

import pytest
import numpy as np
from scipy.sparse import random as sparse_random
import actionet as an


def test_impute_features(adata_with_network):
    """Test feature imputation using network diffusion."""
    adata = adata_with_network.copy()

    # Select a few genes to impute
    genes_to_impute = adata.var_names[:5].tolist()

    # Run imputation
    X_imputed = an.impute_features(
        adata,
        features=genes_to_impute,
        network_key="actionet",
        alpha=0.85,
        rescale=True
    )

    # Check results
    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))
    assert not np.allclose(X_imputed, 0)  # Should have non-zero values

    # Check that imputation doesn't dramatically change values
    original = adata[:, genes_to_impute].X.toarray() if hasattr(adata.X, 'toarray') else adata[:, genes_to_impute].X
    correlation = np.corrcoef(X_imputed.flatten(), original.flatten())[0, 1]
    assert correlation > 0.5  # Should maintain some correlation


def test_impute_features_no_rescale(adata_with_network):
    """Test imputation without rescaling."""
    adata = adata_with_network.copy()

    genes_to_impute = adata.var_names[:3].tolist()

    X_imputed = an.impute_features(
        adata,
        features=genes_to_impute,
        rescale=False
    )

    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))


def test_impute_features_invalid_network(adata_with_reduction):
    """Test imputation with missing network."""
    adata = adata_with_reduction.copy()

    with pytest.raises(ValueError, match="Network"):
        an.impute_features(adata, features=["gene_0"], network_key="nonexistent")


def test_impute_features_invalid_genes(adata_with_network):
    """Test imputation with non-existent genes."""
    adata = adata_with_network.copy()

    with pytest.raises(ValueError, match="None of the specified features"):
        an.impute_features(adata, features=["NonExistentGene1", "NonExistentGene2"])


def test_impute_from_archetypes(adata_with_archetypes):
    """Test imputation from archetype profiles."""
    adata = adata_with_archetypes.copy()

    # Select genes to impute
    genes_to_impute = adata.var_names[:10].tolist()

    # Run archetype-based imputation
    X_imputed = an.impute_from_archetypes(
        adata,
        features=genes_to_impute,
        H_key="H_merged",
        rescale=True
    )

    # Check results
    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))
    assert not np.allclose(X_imputed, 0)


def test_impute_from_archetypes_missing_H(adata_with_reduction):
    """Test archetype imputation with missing H matrix."""
    adata = adata_with_reduction.copy()

    with pytest.raises(ValueError, match="H matrix"):
        an.impute_from_archetypes(adata, features=["gene_0"], H_key="nonexistent")


def test_smooth_kernel(adata_with_reduction):
    """Test kernel smoothing of reduced representation."""
    adata = adata_with_reduction.copy()

    # Run kernel smoothing
    an.smooth_kernel(
        adata,
        reduction_key="action",
        smoothed_key="action_smoothed",
        alpha=0.85,
        max_iter=10
    )

    # Check results
    assert "action_smoothed" in adata.obsm
    assert adata.obsm["action_smoothed"].shape == adata.obsm["action"].shape

    # Smoothed version should be different but correlated
    assert not np.array_equal(adata.obsm["action_smoothed"], adata.obsm["action"])

    # Check correlation for first component
    correlation = np.corrcoef(
        adata.obsm["action"][:, 0],
        adata.obsm["action_smoothed"][:, 0]
    )[0, 1]
    assert correlation > 0.7


def test_smooth_kernel_invalid_key(adata_with_reduction):
    """Test kernel smoothing with invalid reduction key."""
    adata = adata_with_reduction.copy()

    with pytest.raises(ValueError, match="Reduction"):
        an.smooth_kernel(adata, reduction_key="nonexistent")


def test_impute_features_layer(adata_with_network):
    """Test imputation from a specific layer."""
    adata = adata_with_network.copy()

    # Add a layer
    adata.layers["test_layer"] = adata.X.copy()

    genes_to_impute = adata.var_names[:5].tolist()

    X_imputed = an.impute_features(
        adata,
        features=genes_to_impute,
        layer="test_layer"
    )

    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))


def test_impute_from_archetypes_layer(adata_with_archetypes):
    """Test archetype imputation from a specific layer."""
    adata = adata_with_archetypes.copy()

    # Add a layer
    adata.layers["test_layer"] = adata.X.copy()

    genes_to_impute = adata.var_names[:5].tolist()

    X_imputed = an.impute_from_archetypes(
        adata,
        features=genes_to_impute,
        H_key="H_merged",
        layer="test_layer"
    )

    assert X_imputed.shape == (adata.n_obs, len(genes_to_impute))
