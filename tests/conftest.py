"""Pytest fixtures for ACTIONet tests."""

import pytest
import numpy as np
from scipy.sparse import random as sparse_random
from anndata import AnnData
import actionet as an


@pytest.fixture
def synthetic_adata():
    """Create a synthetic AnnData object."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 200

    # Create sparse count matrix
    X = sparse_random(n_obs, n_vars, density=0.3, format='csr', random_state=42)
    X.data = np.random.poisson(5, size=X.data.shape)

    # Create AnnData
    adata = AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    return adata


@pytest.fixture
def adata_with_reduction(synthetic_adata):
    """AnnData with computed ACTION reduction."""
    adata = synthetic_adata.copy()

    # Run reduction
    an.reduce_kernel(adata, k=15, reduction_key="action")

    return adata


@pytest.fixture
def adata_with_network(adata_with_reduction):
    """AnnData with network built from reduction."""
    adata = adata_with_reduction.copy()

    # Build network
    an.build_network(adata, reduction_key="action", network_key="actionet")

    return adata


@pytest.fixture
def adata_with_archetypes(adata_with_network):
    """AnnData with full ACTION decomposition."""
    adata = adata_with_network.copy()

    # Run ACTION
    an.run_action(adata, k_min=3, k_max=8, reduction_key="action")

    return adata
