"""Tests for advanced ACTIONet functions."""

import pytest
import numpy as np
import actionet as an


def test_run_archetypal_analysis():
    """Test basic archetypal analysis."""
    np.random.seed(42)

    # Create simple data
    data = np.random.randn(50, 100)
    k = 5
    W0 = data[:, :k].copy()

    result = an.run_archetypal_analysis(data, W0, max_iter=10)

    assert "C" in result
    assert "H" in result
    assert "W" in result
    assert result["C"].shape == (100, k)
    assert result["H"].shape == (k, 100)


def test_decompose_action(adata_with_reduction):
    """Test ACTION decomposition trace."""
    adata = adata_with_reduction.copy()

    S_r = adata.obsm["action"].T
    result = an.decompose_action(S_r, k_min=2, k_max=5, max_iter=10)

    assert "C" in result
    assert "H" in result
    assert len(result["C"]) == 4  # k=2,3,4,5
    assert len(result["H"]) == 4


def test_collect_archetypes():
    """Test archetype collection and filtering."""
    np.random.seed(42)

    # Create mock trace
    C_trace = [np.random.randn(10, i) for i in range(2, 6)]
    H_trace = [np.random.randn(i, 50) for i in range(2, 6)]

    result = an.collect_archetypes(C_trace, H_trace, specificity_threshold=-5.0, min_observations=1)

    assert "selected_archs" in result
    assert "C_stacked" in result
    assert "H_stacked" in result


def test_merge_archetypes():
    """Test archetype merging."""
    np.random.seed(42)

    S_r = np.random.randn(10, 50)
    C_stacked = np.random.randn(10, 15)
    H_stacked = np.random.randn(15, 50)

    result = an.merge_archetypes(S_r, C_stacked, H_stacked)

    assert "selected_archetypes" in result
    assert "C_merged" in result
    assert "H_merged" in result
    assert "assigned_archetypes" in result


def test_run_simplex_regression():
    """Test simplex regression."""
    np.random.seed(42)

    A = np.random.randn(50, 10)
    B = np.random.randn(50, 20)

    X = an.run_simplex_regression(A, B)

    assert X.shape == (10, 20)
    # Check simplex constraint (columns sum to 1)
    assert np.allclose(X.sum(axis=0), 1.0, atol=0.01)


def test_run_spa():
    """Test successive projections algorithm."""
    np.random.seed(42)

    data = np.random.randn(50, 100)
    k = 10

    result = an.run_spa(data, k)

    assert "selected_cols" in result
    assert "norms" in result
    assert len(result["selected_cols"]) == k


def test_run_label_propagation(adata_with_network):
    """Test label propagation."""
    adata = adata_with_network.copy()

    # Create initial labels
    n_cells = adata.n_obs
    initial_labels = np.random.choice([1.0, 2.0, 3.0], size=n_cells)

    an.run_label_propagation(
        adata,
        initial_labels,
        network_key="actionet",
        lambda_param=0.5,
        iterations=2
    )

    assert "propagated_labels" in adata.obs
    assert len(adata.obs["propagated_labels"]) == n_cells


def test_run_label_propagation_with_fixed(adata_with_network):
    """Test label propagation with fixed labels."""
    adata = adata_with_network.copy()

    n_cells = adata.n_obs
    initial_labels = np.random.choice([1.0, 2.0, 3.0], size=n_cells)
    fixed = np.array([1, 2, 3], dtype=np.int32)  # Fix first 3 cells

    an.run_label_propagation(
        adata,
        initial_labels,
        fixed_labels=fixed,
        key_added="fixed_propagated"
    )

    assert "fixed_propagated" in adata.obs


def test_compute_coreness(adata_with_network):
    """Test graph coreness computation."""
    adata = adata_with_network.copy()

    an.compute_coreness(adata, network_key="actionet")

    assert "coreness" in adata.obs
    assert adata.obs["coreness"].dtype == np.int32 or adata.obs["coreness"].dtype == np.int64


def test_compute_archetype_centrality(adata_with_network):
    """Test archetype centrality computation."""
    adata = adata_with_network.copy()

    # Create mock assignments
    assignments = np.random.randint(0, 3, size=adata.n_obs)

    an.compute_archetype_centrality(
        adata,
        assignments,
        network_key="actionet"
    )

    assert "centrality" in adata.obs
    assert len(adata.obs["centrality"]) == adata.n_obs


def test_advanced_pipeline(adata_with_reduction):
    """Test advanced pipeline with decomposition and merging."""
    adata = adata_with_reduction.copy()

    # Get reduced representation
    S_r = adata.obsm["action"].T

    # Run decomposition
    trace = an.decompose_action(S_r, k_min=3, k_max=5, max_iter=10)

    # Collect archetypes
    collected = an.collect_archetypes(
        trace["C"], trace["H"],
        specificity_threshold=-5.0,
        min_observations=1
    )

    # Merge archetypes
    merged = an.merge_archetypes(
        S_r,
        collected["C_stacked"],
        collected["H_stacked"]
    )

    assert "assigned_archetypes" in merged
    assert len(merged["assigned_archetypes"]) == adata.n_obs


def test_spa_with_archetypal_analysis():
    """Test SPA followed by AA."""
    np.random.seed(42)

    data = np.random.randn(50, 100)
    k = 5

    # Run SPA to find initial archetypes
    spa_result = an.run_spa(data, k)
    selected = spa_result["selected_cols"] - 1  # Convert to 0-indexed

    # Use selected columns as initial archetypes
    W0 = data[:, selected.astype(int)]

    # Run AA
    aa_result = an.run_archetypal_analysis(data, W0, max_iter=10)

    assert aa_result["C"].shape == (100, k)
    assert aa_result["H"].shape == (k, 100)
