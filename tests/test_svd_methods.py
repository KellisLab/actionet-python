"""
Test script to validate all supported SVD methods in ACTIONet.

This script tests:
1. reduce_kernel with different SVD algorithms (IRLB, Halko, Feng, PRIMME)
2. Validates basic statistics and dimensions
3. Compares results across methods using correlation analysis
4. Visualizes the first 2 components in side-by-side facets
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import anndata
import scanpy as sc

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
    print(f"Features: {adata.n_vars}, Cells: {adata.n_obs}")

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
    return adata


def test_svd_method(adata, method_code, method_name, n_components=30):
    """Test a single SVD method and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name} (algorithm code: {method_code})")
    print(f"{'='*60}")

    try:
        # Create a copy for testing
        adata_test = adata.copy()

        # Run reduce_kernel with the specified algorithm
        actionet.reduce_kernel(
            adata_test,
            n_components=n_components,
            layer='logcounts',
            key_added=f'action_{method_name.lower()}',
            svd_algorithm=method_code,
            max_iter=0,  # Use default for each method
            seed=42,
            verbose=True,
            inplace=True
        )

        # Extract results
        reduction = adata_test.obsm[f'action_{method_name.lower()}']
        sigma = adata_test.uns[f'action_{method_name.lower()}_params']['sigma']

        # Validate dimensions
        assert reduction.shape == (adata_test.n_obs, n_components), \
            f"Unexpected shape: {reduction.shape}"

        # Print statistics
        print(f"\nResults for {method_name}:")
        print(f"  Reduction shape: {reduction.shape}")
        print(f"  Sigma shape: {sigma.shape if hasattr(sigma, 'shape') else len(sigma)}")
        print(f"  Reduction - Mean: {np.mean(reduction):.6f}")
        print(f"  Reduction - Std:  {np.std(reduction):.6f}")
        print(f"  Reduction - Min:  {np.min(reduction):.6f}")
        print(f"  Reduction - Max:  {np.max(reduction):.6f}")
        print(f"  First 5 singular values: {sigma[:5] if hasattr(sigma, '__len__') else sigma}")

        # Check for NaN or Inf
        assert not np.any(np.isnan(reduction)), f"{method_name}: Found NaN in reduction"
        assert not np.any(np.isinf(reduction)), f"{method_name}: Found Inf in reduction"

        print(f"✓ {method_name} completed successfully!")

        return {
            'reduction': reduction,
            'sigma': sigma,
            'success': True,
            'error': None
        }

    except Exception as e:
        print(f"✗ Error with {method_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'reduction': None,
            'sigma': None,
            'success': False,
            'error': str(e)
        }


def compare_methods(results, method_names):
    """Compute pairwise correlations between methods."""
    print("\n" + "="*60)
    print("Pairwise Correlations (first 2 components):")
    print("="*60)

    successful_methods = [name for name in method_names if results[name]['success']]

    for i, method1 in enumerate(successful_methods):
        for method2 in successful_methods[i+1:]:
            red1 = results[method1]['reduction'][:, :2].flatten()
            red2 = results[method2]['reduction'][:, :2].flatten()

            # Compute absolute correlation (sign can differ)
            corr = np.abs(np.corrcoef(red1, red2)[0, 1])

            print(f"{method1:10s} vs {method2:10s}: {corr:.6f}")


def visualize_results(results, adata, method_names):
    """Create faceted visualization of first 2 components."""
    successful_methods = [name for name in method_names if results[name]['success']]

    if not successful_methods:
        print("\nNo methods succeeded - cannot create visualization.")
        return

    n_methods = len(successful_methods)
    fig = plt.figure(figsize=(5 * n_methods, 5))
    gs = GridSpec(1, n_methods, figure=fig, hspace=0.3, wspace=0.3)

    # Try to get colors from cell labels
    if 'CellLabel' in adata.obs.columns:
        labels = adata.obs['CellLabel'].astype('category')
        colors = labels.cat.codes
        cmap = 'tab20'
        has_labels = True
    else:
        colors = 'steelblue'
        cmap = None
        has_labels = False

    for idx, method in enumerate(successful_methods):
        ax = fig.add_subplot(gs[0, idx])

        reduction = results[method]['reduction']

        scatter = ax.scatter(
            reduction[:, 0],
            reduction[:, 1],
            c=colors,
            cmap=cmap,
            alpha=0.6,
            s=10,
            rasterized=True
        )

        ax.set_title(f'{method}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add colorbar for first plot if we have labels
        if idx == 0 and has_labels:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cell Type', fontsize=10)

    fig.suptitle('SVD Method Comparison - First 2 Components',
                 fontsize=16, fontweight='bold', y=1.02)

    # Save figure
    output_path = '../tests/svd_methods_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def plot_singular_values(results, method_names):
    """Plot singular value decay for each method."""
    successful_methods = [name for name in method_names if results[name]['success']]

    if not successful_methods:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in successful_methods:
        sigma = results[method]['sigma']
        if hasattr(sigma, '__len__'):
            ax.plot(range(1, len(sigma) + 1), sigma, 'o-', label=method, alpha=0.7)

    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Decay', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    output_path = '../tests/svd_singular_values.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Singular values plot saved to: {output_path}")

    plt.show()


def main():
    """Main test function."""
    print("="*60)
    print("ACTIONet SVD Methods Validation Test")
    print("="*60)

    # Load and prepare data
    adata = load_and_prepare_data()

    # Test parameters
    n_components = 30

    # Test all SVD methods
    results = {}
    for method_code, method_name in SVD_METHODS.items():
        results[method_name] = test_svd_method(
            adata, method_code, method_name, n_components
        )

    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    for method_name, result in results.items():
        status = "✓ PASSED" if result['success'] else "✗ FAILED"
        print(f"{method_name:10s}: {status}")
        if not result['success']:
            print(f"  Error: {result['error']}")

    # Compare methods
    method_names = list(SVD_METHODS.values())
    compare_methods(results, method_names)

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, adata, method_names)
    plot_singular_values(results, method_names)

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
