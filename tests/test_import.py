"""Smoke tests for package import."""

import pytest


def test_import_actionet():
    """Test that the main package can be imported."""
    import actionet
    assert actionet is not None


def test_import_core():
    """Test that core module can be imported."""
    import actionet.core
    assert actionet.core is not None


def test_import_batch_correction():
    """Test that batch_correction module can be imported."""
    import actionet.batch_correction
    assert actionet.batch_correction is not None


def test_import_imputation():
    """Test that imputation module can be imported."""
    import actionet.imputation
    assert actionet.imputation is not None


def test_core_functions_available():
    """Test that core functions are available in the namespace."""
    import actionet as an

    assert hasattr(an, 'reduce_kernel')
    assert hasattr(an, 'run_action')
    assert hasattr(an, 'build_network')
    assert hasattr(an, 'compute_network_diffusion')
    assert hasattr(an, 'compute_feature_specificity')
    assert hasattr(an, 'layout_network')
    assert hasattr(an, 'run_svd')


def test_batch_correction_functions_available():
    """Test that batch correction functions are available."""
    import actionet as an

    assert hasattr(an, 'correct_batch_effect')
    assert hasattr(an, 'correct_basal_expression')


def test_imputation_functions_available():
    """Test that imputation functions are available."""
    import actionet as an

    assert hasattr(an, 'impute_features')
    assert hasattr(an, 'impute_from_archetypes')
    assert hasattr(an, 'smooth_kernel')


def test_cpp_core_available():
    """Test that C++ core module is available."""
    import actionet._core as core
    assert core is not None

    # Check for key C++ functions
    assert hasattr(core, 'reduce_kernel_sparse')
    assert hasattr(core, 'reduce_kernel_dense')
    assert hasattr(core, 'build_network')
    assert hasattr(core, 'orthogonalize_batch_effect_sparse')
    assert hasattr(core, 'orthogonalize_batch_effect_dense')
    assert hasattr(core, 'orthogonalize_basal_sparse')
