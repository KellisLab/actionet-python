"""Tests for batch correction functions."""

import pytest
import numpy as np
from scipy.sparse import random as sparse_random
import actionet as an


def test_correct_batch_effect(adata_with_reduction):
    """Test batch effect correction."""
    adata = adata_with_reduction.copy()
    
    # Add batch labels
    n_cells = adata.n_obs
    adata.obs["batch"] = ["batch1"] * (n_cells // 2) + ["batch2"] * (n_cells - n_cells // 2)
    
    # Run batch correction
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        reduction_key="action",
        corrected_key="action_corrected"
    )
    
    # Check results
    assert "action_corrected" in adata.obsm
    assert adata.obsm["action_corrected"].shape == adata.obsm["action"].shape
    assert "action_corrected_params" in adata.uns
    
    # Check that correction was applied
    assert not np.array_equal(adata.obsm["action_corrected"], adata.obsm["action"])


def test_correct_batch_effect_invalid_key(adata_with_reduction):
    """Test batch correction with invalid batch key."""
    adata = adata_with_reduction.copy()
    
    with pytest.raises(ValueError, match="Batch key"):
        an.correct_batch_effect(adata, batch_key="nonexistent")


def test_correct_basal_expression(adata_with_reduction):
    """Test basal expression correction."""
    adata = adata_with_reduction.copy()
    
    # Select some genes as basal
    basal_genes = adata.var_names[:10].tolist()
    
    # Run basal correction
    an.correct_basal_expression(
        adata,
        basal_genes=basal_genes,
        reduction_key="action",
        corrected_key="action_basal_corrected"
    )
    
    # Check results
    assert "action_basal_corrected" in adata.obsm
    assert adata.obsm["action_basal_corrected"].shape == adata.obsm["action"].shape
    assert "action_basal_corrected_params" in adata.uns
    
    # Check that basal genes were recorded
    assert "basal_genes" in adata.uns["action_basal_corrected_params"]
    assert len(adata.uns["action_basal_corrected_params"]["basal_genes"]) == 10


def test_correct_basal_expression_no_genes(adata_with_reduction):
    """Test basal correction with no matching genes."""
    adata = adata_with_reduction.copy()
    
    with pytest.raises(ValueError, match="None of the specified basal genes"):
        an.correct_basal_expression(adata, basal_genes=["NonExistentGene1", "NonExistentGene2"])


def test_correct_batch_effect_multiple_batches(adata_with_reduction):
    """Test batch correction with multiple batches."""
    adata = adata_with_reduction.copy()
    
    # Add 3 batches
    n_cells = adata.n_obs
    batch_size = n_cells // 3
    adata.obs["batch"] = (
        ["batch1"] * batch_size +
        ["batch2"] * batch_size +
        ["batch3"] * (n_cells - 2 * batch_size)
    )
    
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        corrected_key="action_corrected_3batch"
    )
    
    assert "action_corrected_3batch" in adata.obsm
    assert adata.obsm["action_corrected_3batch"].shape[0] == n_cells
