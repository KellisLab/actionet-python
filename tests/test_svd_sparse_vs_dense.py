"""
Test script to verify consistency between sparse and dense matrix inputs for all SVD methods,
including PRIMME operator (backed) mode.

This script:
1. Tests each SVD method with both sparse and dense input
2. Tests PRIMME in operator/backed mode (simulating out-of-memory datasets)
3. Validates that results are numerically consistent
4. Validates all output keys use the correct naming convention (_U, _A, _B)
5. Visualizes differences between sparse and dense results
"""

import sys
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import anndata
import scanpy as sc
import scipy.sparse as sp
import h5py

# Add parent directory to path if needed
sys.path.insert(0, '../src')

import actionet

# SVD algorithm constants (from svd_main.hpp)
ALG_IRLB = 0
ALG_HALKO = 1
ALG_FENG = 2
ALG_PRIMME = 3

# Dictionary mapping algorithm codes to names
SVD_METHODS = {
    ALG_IRLB: 'IRLB',
    ALG_HALKO: 'Halko',
    ALG_FENG: 'Feng',
    ALG_PRIMME: 'PRIMME',
}


def load_and_prepare_data():
    """Load test data and perform basic preprocessing."""
    print("Loading test data...")
    adata = anndata.read_h5ad("../data/test_adata.h5ad")

    print(f"Data shape: {adata.shape}")

    # Basic filtering
    print("Filtering data...")
    sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))

    # Normalize and log-transform
    print("Normalizing and log-transforming...")
    adata.layers['logcounts'] = sc.pp.normalize_total(
        adata, target_sum=1e4, inplace=False
    )['X']
    sc.pp.log1p(adata, layer='logcounts', base=2, copy=False)

    print(f"Preprocessed shape: {adata.shape}")
    print(f"Data type: {type(adata.layers['logcounts'])}")
    print(f"Is sparse: {sp.issparse(adata.layers['logcounts'])}")

    return adata


def validate_reduction_keys(adata, key, n_components, label=""):
    """Validate that all expected output keys exist with correct shapes and naming."""
    prefix = f"[{label}] " if label else ""
    errors = []

    # obsm keys
    if key not in adata.obsm:
        errors.append(f"{prefix}Missing adata.obsm['{key}']")
    else:
        S_r = adata.obsm[key]
        if S_r.shape != (adata.n_obs, n_components):
            errors.append(f"{prefix}obsm['{key}'] shape {S_r.shape} != expected ({adata.n_obs}, {n_components})")

    if f"{key}_B" not in adata.obsm:
        errors.append(f"{prefix}Missing adata.obsm['{key}_B']")

    # varm keys — must use _U (not _V)
    if f"{key}_U" not in adata.varm:
        errors.append(f"{prefix}Missing adata.varm['{key}_U'] (left singular vectors)")
    else:
        U = adata.varm[f"{key}_U"]
        if U.shape[0] != adata.n_vars:
            errors.append(f"{prefix}varm['{key}_U'] has {U.shape[0]} rows, expected {adata.n_vars}")
        if U.shape[1] != n_components:
            errors.append(f"{prefix}varm['{key}_U'] has {U.shape[1]} cols, expected {n_components}")

    if f"{key}_A" not in adata.varm:
        errors.append(f"{prefix}Missing adata.varm['{key}_A']")

    # Verify OLD naming convention (_V) is NOT present
    if f"{key}_V" in adata.varm:
        errors.append(f"{prefix}STALE KEY: adata.varm['{key}_V'] should be '{key}_U'")

    # uns params
    params_key = f"{key}_params"
    if params_key not in adata.uns:
        errors.append(f"{prefix}Missing adata.uns['{params_key}']")
    else:
        params = adata.uns[params_key]
        if "sigma" not in params:
            errors.append(f"{prefix}Missing 'sigma' in params")

    # NaN/Inf checks
    for slot_name in [key, f"{key}_B"]:
        if slot_name in adata.obsm:
            arr = adata.obsm[slot_name]
            if np.any(np.isnan(arr)):
                errors.append(f"{prefix}NaN in obsm['{slot_name}']")
            if np.any(np.isinf(arr)):
                errors.append(f"{prefix}Inf in obsm['{slot_name}']")

    for slot_name in [f"{key}_U", f"{key}_A"]:
        if slot_name in adata.varm:
            arr = adata.varm[slot_name]
            if np.any(np.isnan(arr)):
                errors.append(f"{prefix}NaN in varm['{slot_name}']")
            if np.any(np.isinf(arr)):
                errors.append(f"{prefix}Inf in varm['{slot_name}']")

    return errors


def test_svd_sparse_vs_dense(adata, method_code, method_name, n_components=30):
    """Test a single SVD method with both sparse and dense inputs."""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name} (Sparse vs Dense)")
    print(f"{'='*60}")

    results = {
        'sparse': {'success': False, 'error': None},
        'dense': {'success': False, 'error': None}
    }

    # Test with sparse input
    print(f"\n--- Testing {method_name} with SPARSE input ---")
    try:
        adata_sparse = adata.copy()

        # Ensure data is sparse
        if not sp.issparse(adata_sparse.layers['logcounts']):
            adata_sparse.layers['logcounts'] = sp.csr_matrix(adata_sparse.layers['logcounts'])

        actionet.reduce_kernel(
            adata_sparse,
            n_components=n_components,
            layer='logcounts',
            key_added='action_sparse',
            svd_algorithm=method_code,
            max_iter=0,
            seed=42,
            verbose=True,
            inplace=True
        )

        results['sparse']['reduction'] = adata_sparse.obsm['action_sparse']
        results['sparse']['sigma'] = adata_sparse.uns['action_sparse_params']['sigma']
        results['sparse']['U'] = adata_sparse.varm['action_sparse_U']
        results['sparse']['success'] = True

        # Validate all output keys
        key_errors = validate_reduction_keys(adata_sparse, 'action_sparse', n_components, f"{method_name} SPARSE")
        if key_errors:
            for err in key_errors:
                print(f"  ⚠ {err}")
        else:
            print(f"  ✓ All output keys valid (using _U convention)")

        print(f"✓ {method_name} SPARSE completed successfully!")
        print(f"  Reduction shape: {results['sparse']['reduction'].shape}")
        print(f"  Sparsity: {sp.issparse(adata_sparse.layers['logcounts'])}")

    except Exception as e:
        print(f"✗ Error with {method_name} SPARSE: {str(e)}")
        results['sparse']['error'] = str(e)
        import traceback
        traceback.print_exc()

    # Test with dense input
    print(f"\n--- Testing {method_name} with DENSE input ---")
    try:
        adata_dense = adata.copy()

        # Convert to dense
        if sp.issparse(adata_dense.layers['logcounts']):
            adata_dense.layers['logcounts'] = adata_dense.layers['logcounts'].toarray()

        actionet.reduce_kernel(
            adata_dense,
            n_components=n_components,
            layer='logcounts',
            key_added='action_dense',
            svd_algorithm=method_code,
            max_iter=0,
            seed=42,
            verbose=True,
            inplace=True
        )

        results['dense']['reduction'] = adata_dense.obsm['action_dense']
        results['dense']['sigma'] = adata_dense.uns['action_dense_params']['sigma']
        results['dense']['U'] = adata_dense.varm['action_dense_U']
        results['dense']['success'] = True

        # Validate all output keys
        key_errors = validate_reduction_keys(adata_dense, 'action_dense', n_components, f"{method_name} DENSE")
        if key_errors:
            for err in key_errors:
                print(f"  ⚠ {err}")
        else:
            print(f"  ✓ All output keys valid (using _U convention)")

        print(f"✓ {method_name} DENSE completed successfully!")
        print(f"  Reduction shape: {results['dense']['reduction'].shape}")
        print(f"  Sparsity: {sp.issparse(adata_dense.layers['logcounts'])}")

    except Exception as e:
        print(f"✗ Error with {method_name} DENSE: {str(e)}")
        results['dense']['error'] = str(e)
        import traceback
        traceback.print_exc()

    # Compare results if both succeeded
    if results['sparse']['success'] and results['dense']['success']:
        print(f"\n--- Comparing {method_name} Sparse vs Dense ---")

        sparse_red = results['sparse']['reduction']
        dense_red = results['dense']['reduction']
        sparse_sigma = results['sparse']['sigma']
        dense_sigma = results['dense']['sigma']

        # Flatten sigma if needed
        if hasattr(sparse_sigma, 'shape') and len(sparse_sigma.shape) == 2:
            sparse_sigma = sparse_sigma.flatten()
        if hasattr(dense_sigma, 'shape') and len(dense_sigma.shape) == 2:
            dense_sigma = dense_sigma.flatten()

        # Compute differences
        red_diff = np.abs(sparse_red - dense_red)
        sigma_diff = np.abs(sparse_sigma - dense_sigma)

        # Compute relative differences
        red_rel_diff = red_diff / (np.abs(sparse_red) + 1e-10)
        sigma_rel_diff = sigma_diff / (np.abs(sparse_sigma) + 1e-10)

        # Compute correlation (accounting for sign flips)
        corr = np.corrcoef(sparse_red.flatten(), dense_red.flatten())[0, 1]
        abs_corr = np.abs(corr)

        print(f"  Reduction absolute diff - Mean: {np.mean(red_diff):.6e}, Max: {np.max(red_diff):.6e}")
        print(f"  Reduction relative diff - Mean: {np.mean(red_rel_diff):.6e}, Max: {np.max(red_rel_diff):.6e}")
        print(f"  Sigma absolute diff     - Mean: {np.mean(sigma_diff):.6e}, Max: {np.max(sigma_diff):.6e}")
        print(f"  Sigma relative diff     - Mean: {np.mean(sigma_rel_diff):.6e}, Max: {np.max(sigma_rel_diff):.6e}")
        print(f"  Correlation             - {corr:.10f} (abs: {abs_corr:.10f})")

        # Check if results are consistent (high correlation, low relative error)
        is_consistent = abs_corr > 0.9999 and np.mean(red_rel_diff) < 0.01

        if is_consistent:
            print(f"  ✓ Results are CONSISTENT between sparse and dense")
        else:
            print(f"  ⚠ Results show DIFFERENCES between sparse and dense")

        results['comparison'] = {
            'correlation': corr,
            'abs_correlation': abs_corr,
            'reduction_mean_abs_diff': np.mean(red_diff),
            'reduction_max_abs_diff': np.max(red_diff),
            'reduction_mean_rel_diff': np.mean(red_rel_diff),
            'reduction_max_rel_diff': np.max(red_rel_diff),
            'sigma_mean_abs_diff': np.mean(sigma_diff),
            'sigma_max_abs_diff': np.max(sigma_diff),
            'is_consistent': is_consistent
        }
    else:
        results['comparison'] = None

    return results


def visualize_comparison(all_results, method_names):
    """Visualize sparse vs dense comparison for all methods."""
    successful_comparisons = [
        name for name in method_names
        if all_results[name]['comparison'] is not None
    ]

    if not successful_comparisons:
        print("\nNo successful comparisons - cannot create visualization.")
        return

    n_methods = len(successful_comparisons)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, n_methods, figure=fig, hspace=0.35, wspace=0.3)

    for idx, method in enumerate(successful_comparisons):
        results = all_results[method]
        sparse_red = results['sparse']['reduction']
        dense_red = results['dense']['reduction']
        comp = results['comparison']

        # Plot 1: Sparse results (first 2 components)
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.scatter(sparse_red[:, 0], sparse_red[:, 1], alpha=0.5, s=5, c='blue')
        ax1.set_title(f'{method} - Sparse', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Dense results (first 2 components)
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.scatter(dense_red[:, 0], dense_red[:, 1], alpha=0.5, s=5, c='red')
        ax2.set_title(f'{method} - Dense', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scatter plot comparing sparse vs dense
        ax3 = fig.add_subplot(gs[2, idx])
        # Sample points for clarity
        sample_idx = np.random.choice(sparse_red.shape[0] * 2,
                                     min(1000, sparse_red.shape[0] * 2),
                                     replace=False)
        sparse_flat_sample = sparse_red[:, :2].flatten()[sample_idx]
        dense_flat_sample = dense_red[:, :2].flatten()[sample_idx]

        ax3.scatter(sparse_flat_sample, dense_flat_sample, alpha=0.3, s=3)

        # Add diagonal line
        lims = [
            np.min([ax3.get_xlim(), ax3.get_ylim()]),
            np.max([ax3.get_xlim(), ax3.get_ylim()]),
        ]
        ax3.plot(lims, lims, 'k-', alpha=0.5, zorder=1)
        ax3.set_xlim(lims)
        ax3.set_ylim(lims)

        ax3.set_title(f'Corr: {comp["abs_correlation"]:.6f}', fontsize=10)
        ax3.set_xlabel('Sparse values')
        ax3.set_ylabel('Dense values')
        ax3.grid(True, alpha=0.3)

    fig.suptitle('Sparse vs Dense Input Comparison',
                 fontsize=16, fontweight='bold')

    # Save figure
    output_path = '../tests/svd_sparse_vs_dense_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def plot_correlation_summary(all_results, method_names):
    """Plot bar chart of correlations for all methods."""
    successful_comparisons = [
        name for name in method_names
        if all_results[name]['comparison'] is not None
    ]

    if not successful_comparisons:
        return

    correlations = [all_results[name]['comparison']['abs_correlation']
                   for name in successful_comparisons]
    mean_rel_diffs = [all_results[name]['comparison']['reduction_mean_rel_diff']
                     for name in successful_comparisons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot correlations
    ax1.bar(successful_comparisons, correlations, color='steelblue', alpha=0.7)
    ax1.axhline(y=0.9999, color='r', linestyle='--', label='0.9999 threshold')
    ax1.set_ylabel('Absolute Correlation', fontsize=12)
    ax1.set_title('Sparse vs Dense Correlation', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(correlations) - 0.001, 1.0001])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot relative differences
    ax2.bar(successful_comparisons, mean_rel_diffs, color='coral', alpha=0.7)
    ax2.axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
    ax2.set_ylabel('Mean Relative Difference', fontsize=12)
    ax2.set_title('Sparse vs Dense Mean Relative Error', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = '../tests/svd_sparse_vs_dense_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {output_path}")

    plt.show()


def test_primme_backed_mode(adata, n_components=30):
    """
    Test PRIMME in backed (operator) mode by creating a backed AnnData on disk.

    This simulates an out-of-memory dataset where the matrix is stored on disk
    and accessed via chunked matrix-vector products through the operator path.
    """
    print(f"\n{'='*60}")
    print(f"Testing: PRIMME BACKED MODE (operator path)")
    print(f"{'='*60}")

    result = {
        'success': False,
        'error': None,
        'reduction': None,
        'sigma': None,
    }

    try:
        # Create a backed h5ad file with sparse data in CSR format
        with tempfile.TemporaryDirectory() as tmpdir:
            backed_path = os.path.join(tmpdir, "backed_test.h5ad")

            # Write the dataset to disk
            adata_write = adata.copy()
            logcounts = adata_write.layers['logcounts']
            if sp.issparse(logcounts):
                logcounts = sp.csr_matrix(logcounts)
            else:
                logcounts = sp.csr_matrix(logcounts)
            adata_write.layers['logcounts'] = logcounts
            adata_write.write_h5ad(backed_path)
            del adata_write

            # Open in backed mode
            adata_backed = anndata.read_h5ad(backed_path, backed='r')

            print(f"  Backed data type: {type(adata_backed.X)}")
            print(f"  Shape: {adata_backed.shape}")

            # Run reduce_kernel — should automatically select PRIMME operator path
            actionet.reduce_kernel(
                adata_backed,
                n_components=n_components,
                layer=None,  # Use .X which is backed
                key_added='action_backed',
                svd_algorithm=3,  # PRIMME
                max_iter=0,
                seed=42,
                verbose=True,
                inplace=True,
                backed_chunk_size=2048,
            )

            result['reduction'] = adata_backed.obsm['action_backed']
            result['sigma'] = adata_backed.uns['action_backed_params']['sigma']
            result['success'] = True

            # Validate output keys
            key_errors = validate_reduction_keys(adata_backed, 'action_backed', n_components, "PRIMME BACKED")
            if key_errors:
                for err in key_errors:
                    print(f"  ⚠ {err}")
            else:
                print(f"  ✓ All output keys valid (using _U convention)")

            # Validate operator_mode is recorded
            params = adata_backed.uns['action_backed_params']
            if params.get('operator_mode', False):
                print(f"  ✓ Operator mode correctly recorded in params")
            else:
                print(f"  ⚠ operator_mode not set in params (may not be backed)")

            print(f"\n  Reduction shape: {result['reduction'].shape}")
            print(f"  First 5 singular values: {result['sigma'][:5]}")
            print(f"✓ PRIMME BACKED mode completed successfully!")

    except Exception as e:
        print(f"✗ Error with PRIMME BACKED: {str(e)}")
        result['error'] = str(e)
        import traceback
        traceback.print_exc()

    return result


def test_primme_backed_vs_inmemory(adata, n_components=30):
    """
    Compare PRIMME backed mode against PRIMME in-memory mode.

    Both should produce numerically similar results (up to iterative solver tolerance).
    """
    print(f"\n{'='*60}")
    print(f"Testing: PRIMME BACKED vs IN-MEMORY consistency")
    print(f"{'='*60}")

    results = {
        'backed': {'success': False},
        'inmemory': {'success': False},
        'comparison': None,
    }

    # In-memory PRIMME
    print(f"\n--- PRIMME in-memory ---")
    try:
        adata_mem = adata.copy()
        if not sp.issparse(adata_mem.layers['logcounts']):
            adata_mem.layers['logcounts'] = sp.csr_matrix(adata_mem.layers['logcounts'])

        actionet.reduce_kernel(
            adata_mem,
            n_components=n_components,
            layer='logcounts',
            key_added='action_primme',
            svd_algorithm=3,
            max_iter=0,
            seed=42,
            verbose=True,
            inplace=True
        )

        results['inmemory']['reduction'] = adata_mem.obsm['action_primme']
        results['inmemory']['sigma'] = adata_mem.uns['action_primme_params']['sigma']
        results['inmemory']['success'] = True
        print(f"  ✓ In-memory PRIMME succeeded")
    except Exception as e:
        print(f"  ✗ In-memory PRIMME failed: {e}")
        results['inmemory']['error'] = str(e)

    # Backed PRIMME
    print(f"\n--- PRIMME backed ---")
    backed_result = test_primme_backed_mode(adata, n_components)
    results['backed']['success'] = backed_result['success']
    if backed_result['success']:
        results['backed']['reduction'] = backed_result['reduction']
        results['backed']['sigma'] = backed_result['sigma']

    # Compare
    if results['inmemory']['success'] and results['backed']['success']:
        print(f"\n--- Comparing PRIMME backed vs in-memory ---")

        mem_red = results['inmemory']['reduction']
        bck_red = results['backed']['reduction']
        mem_sigma = np.asarray(results['inmemory']['sigma']).flatten()
        bck_sigma = np.asarray(results['backed']['sigma']).flatten()

        # Compute correlation (sign may differ per component)
        component_corrs = []
        for i in range(min(mem_red.shape[1], bck_red.shape[1])):
            c = np.abs(np.corrcoef(mem_red[:, i], bck_red[:, i])[0, 1])
            component_corrs.append(c)

        mean_corr = np.mean(component_corrs)
        min_corr = np.min(component_corrs)

        sigma_rel_diff = np.mean(np.abs(mem_sigma - bck_sigma) / (np.abs(mem_sigma) + 1e-10))

        print(f"  Component correlation (mean): {mean_corr:.6f}")
        print(f"  Component correlation (min):  {min_corr:.6f}")
        print(f"  Sigma relative diff (mean):   {sigma_rel_diff:.6e}")

        # Operator path uses chunked matvec so results may differ slightly
        is_consistent = mean_corr > 0.99 and sigma_rel_diff < 0.05
        if is_consistent:
            print(f"  ✓ Backed and in-memory results are CONSISTENT")
        else:
            print(f"  ⚠ Results show DIFFERENCES (expected with iterative solver)")

        results['comparison'] = {
            'mean_component_corr': mean_corr,
            'min_component_corr': min_corr,
            'sigma_rel_diff': sigma_rel_diff,
            'is_consistent': is_consistent,
        }

    return results


def main():
    """Main test function."""
    print("="*60)
    print("ACTIONet SVD Sparse vs Dense Validation Test")
    print("="*60)

    # Load and prepare data
    adata = load_and_prepare_data()

    # Test parameters
    n_components = 30

    # Test all SVD methods
    all_results = {}
    for method_code, method_name in SVD_METHODS.items():
        all_results[method_name] = test_svd_sparse_vs_dense(
            adata, method_code, method_name, n_components
        )

    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)

    for method_name in SVD_METHODS.values():
        results = all_results[method_name]
        sparse_status = "✓" if results['sparse']['success'] else "✗"
        dense_status = "✓" if results['dense']['success'] else "✗"

        print(f"\n{method_name}:")
        print(f"  Sparse: {sparse_status}")
        if not results['sparse']['success'] and results['sparse']['error']:
            print(f"    Error: {results['sparse']['error']}")

        print(f"  Dense:  {dense_status}")
        if not results['dense']['success'] and results['dense']['error']:
            print(f"    Error: {results['dense']['error']}")

        if results['comparison'] is not None:
            comp = results['comparison']
            consistency = "✓ CONSISTENT" if comp['is_consistent'] else "⚠ DIFFERENT"
            print(f"  Comparison: {consistency}")
            print(f"    Correlation: {comp['abs_correlation']:.10f}")
            print(f"    Mean relative diff: {comp['reduction_mean_rel_diff']:.6e}")

    # Visualize results
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    method_names = list(SVD_METHODS.values())
    visualize_comparison(all_results, method_names)
    plot_correlation_summary(all_results, method_names)

    # Test PRIMME backed mode
    print("\n" + "="*60)
    print("Testing PRIMME backed (operator) mode...")
    print("="*60)

    backed_vs_mem = test_primme_backed_vs_inmemory(adata, n_components)

    # Final verdict
    print("\n" + "="*60)
    print("Final Verdict:")
    print("="*60)

    all_consistent = all(
        all_results[name]['comparison'] is not None and
        all_results[name]['comparison']['is_consistent']
        for name in method_names
        if all_results[name]['sparse']['success'] and all_results[name]['dense']['success']
    )

    if all_consistent:
        print("✓ All methods show CONSISTENT results between sparse and dense inputs!")
    else:
        print("⚠ Some methods show differences between sparse and dense inputs.")

    if backed_vs_mem['comparison'] is not None:
        if backed_vs_mem['comparison']['is_consistent']:
            print("✓ PRIMME backed vs in-memory results are CONSISTENT!")
        else:
            print("⚠ PRIMME backed vs in-memory results differ (may be expected with iterative solver).")
    elif backed_vs_mem['backed']['success']:
        print("✓ PRIMME backed mode completed (no in-memory comparison available).")
    else:
        print("⚠ PRIMME backed mode did not complete successfully.")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
