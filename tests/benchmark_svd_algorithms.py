"""
Comprehensive Benchmark Test for SVD Algorithms in ACTIONet.

This script benchmarks all 4 SVD methods (IRLB, Halko, Feng, PRIMME) with:
1. Both sparse and dense matrix inputs
2. PRIMME in operator/backed mode (simulating out-of-memory datasets)
3. Speed measurements (execution time)
4. Memory usage measurements (peak memory consumption)
5. Multiple matrix sizes (if specified)
6. Statistical comparison and visualization

Dependencies:
- memory_profiler (optional, for detailed memory profiling)
- psutil (for memory tracking)

Usage:
    python benchmark_svd_algorithms.py [--detailed-memory] [--include-backed]
"""

import sys
import os
import time
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import anndata
import scanpy as sc
import scipy.sparse as sp
import pandas as pd
import traceback
import argparse
import gc

# Try to import memory profiling libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # Define in except block to avoid linting warning
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory tracking will be limited.")

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available. Using psutil for memory tracking.")

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

# Colors for visualization
METHOD_COLORS = {
    'IRLB': '#1f77b4',
    'Halko': '#ff7f0e',
    'Feng': '#2ca02c',
    'PRIMME': '#d62728',
}


class MemoryTracker:
    """Track memory usage during execution."""

    def __init__(self, use_psutil=True):
        self.use_psutil = use_psutil and PSUTIL_AVAILABLE
        self.process = psutil.Process() if self.use_psutil else None
        self.start_memory = None
        self.peak_memory = None

    def start(self):
        """Start tracking memory."""
        gc.collect()  # Force garbage collection before measurement
        if self.use_psutil:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        else:
            self.start_memory = 0
            self.peak_memory = 0

    def update(self):
        """Update peak memory if current is higher."""
        if self.use_psutil:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory

    def get_peak_memory_increase(self):
        """Get peak memory increase since start."""
        if self.use_psutil:
            return self.peak_memory - self.start_memory
        return 0.0


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
    print(f"Data type: {type(adata.layers['logcounts'])}")
    print(f"Is sparse: {sp.issparse(adata.layers['logcounts'])}")

    if sp.issparse(adata.layers['logcounts']):
        sparsity = 1.0 - (adata.layers['logcounts'].nnz / np.prod(adata.layers['logcounts'].shape))
        print(f"Sparsity: {sparsity:.2%}")

    return adata


def benchmark_svd_method(adata, method_code, method_name, input_type='sparse',
                         n_components=30, n_runs=3, detailed_memory=False):
    """
    Benchmark a single SVD method.

    Args:
        adata: AnnData object with preprocessed data
        method_code: Integer code for SVD algorithm
        method_name: Name of the method
        input_type: 'sparse' or 'dense'
        n_components: Number of components to compute
        n_runs: Number of benchmark runs for timing
        detailed_memory: Whether to use detailed memory profiling

    Returns:
        dict: Benchmark results including timing and memory usage
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {method_name} ({input_type.upper()} input)")
    print(f"{'='*70}")

    results = {
        'method': method_name,
        'input_type': input_type,
        'success': False,
        'error': None,
        'times': [],
        'mean_time': None,
        'std_time': None,
        'memory_mb': None,
    }

    # Prepare data
    adata_test = adata.copy()

    if input_type == 'sparse':
        if not sp.issparse(adata_test.layers['logcounts']):
            adata_test.layers['logcounts'] = sp.csr_matrix(adata_test.layers['logcounts'])
    else:  # dense
        if sp.issparse(adata_test.layers['logcounts']):
            adata_test.layers['logcounts'] = adata_test.layers['logcounts'].toarray()

    # Run multiple times for timing
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=' ', flush=True)

        # Fresh copy for each run
        adata_run = adata_test.copy()

        # Setup memory tracking
        memory_tracker = MemoryTracker()

        try:
            # Start timing and memory tracking
            memory_tracker.start()
            start_time = time.perf_counter()

            # Run SVD
            actionet.reduce_kernel(
                adata_run,
                n_components=n_components,
                layer='logcounts',
                key_added=f'action_benchmark',
                svd_algorithm=method_code,
                max_iter=0,  # Use default
                seed=42,
                verbose=False,
                inplace=True
            )

            # End timing
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Get memory usage
            memory_tracker.update()
            memory_increase = memory_tracker.get_peak_memory_increase()

            results['times'].append(elapsed)
            if run == 0:  # Use first run for memory estimate
                results['memory_mb'] = memory_increase

            print(f"✓ {elapsed:.3f}s, Memory: {memory_increase:.1f} MB")

            # Save results from first successful run
            if run == 0:
                results['reduction'] = adata_run.obsm['action_benchmark']
                results['sigma'] = adata_run.uns['action_benchmark_params']['sigma']
                results['shape'] = adata_run.obsm['action_benchmark'].shape

            # Clean up
            del adata_run
            gc.collect()

        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            results['error'] = str(e)
            traceback.print_exc()
            return results

    # Compute timing statistics
    results['mean_time'] = np.mean(results['times'])
    results['std_time'] = np.std(results['times'])
    results['min_time'] = np.min(results['times'])
    results['max_time'] = np.max(results['times'])
    results['success'] = True

    print(f"\n  Summary:")
    print(f"    Mean time: {results['mean_time']:.3f} ± {results['std_time']:.3f} s")
    print(f"    Min time:  {results['min_time']:.3f} s")
    print(f"    Max time:  {results['max_time']:.3f} s")
    print(f"    Memory:    {results['memory_mb']:.1f} MB")

    return results


def benchmark_primme_backed(adata, n_components=30, n_runs=3, backed_chunk_size=4096):
    """
    Benchmark PRIMME in backed (operator) mode.

    Creates a temporary h5ad file and opens it in backed mode to simulate
    out-of-memory datasets processed via chunked matrix-vector products.

    Args:
        adata: AnnData object with preprocessed data
        n_components: Number of components to compute
        n_runs: Number of benchmark runs for timing
        backed_chunk_size: Chunk size for the operator path

    Returns:
        dict: Benchmark results including timing and memory usage
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: PRIMME BACKED (operator path, chunk_size={backed_chunk_size})")
    print(f"{'='*70}")

    results = {
        'method': 'PRIMME',
        'input_type': 'backed',
        'success': False,
        'error': None,
        'times': [],
        'mean_time': None,
        'std_time': None,
        'memory_mb': None,
    }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            backed_path = os.path.join(tmpdir, "backed_benchmark.h5ad")

            # Write sparse data to disk
            adata_write = adata.copy()
            logcounts = adata_write.layers['logcounts']
            if not sp.issparse(logcounts):
                logcounts = sp.csr_matrix(logcounts)
            adata_write.X = sp.csr_matrix(logcounts)
            adata_write.write_h5ad(backed_path)
            del adata_write
            gc.collect()

            for run in range(n_runs):
                print(f"  Run {run + 1}/{n_runs}...", end=' ', flush=True)

                adata_backed = anndata.read_h5ad(backed_path, backed='r')

                memory_tracker = MemoryTracker()
                memory_tracker.start()
                start_time = time.perf_counter()

                actionet.reduce_kernel(
                    adata_backed,
                    n_components=n_components,
                    layer=None,
                    key_added='action_benchmark',
                    svd_algorithm=3,
                    max_iter=0,
                    seed=42,
                    verbose=False,
                    inplace=True,
                    backed_chunk_size=backed_chunk_size,
                )

                end_time = time.perf_counter()
                elapsed = end_time - start_time

                memory_tracker.update()
                memory_increase = memory_tracker.get_peak_memory_increase()

                results['times'].append(elapsed)
                if run == 0:
                    results['memory_mb'] = memory_increase
                    results['reduction'] = adata_backed.obsm['action_benchmark']
                    results['sigma'] = adata_backed.uns['action_benchmark_params']['sigma']
                    results['shape'] = adata_backed.obsm['action_benchmark'].shape

                print(f"✓ {elapsed:.3f}s, Memory: {memory_increase:.1f} MB")

                del adata_backed
                gc.collect()

            results['mean_time'] = np.mean(results['times'])
            results['std_time'] = np.std(results['times'])
            results['min_time'] = np.min(results['times'])
            results['max_time'] = np.max(results['times'])
            results['success'] = True

            print(f"\n  Summary:")
            print(f"    Mean time: {results['mean_time']:.3f} ± {results['std_time']:.3f} s")
            print(f"    Min time:  {results['min_time']:.3f} s")
            print(f"    Max time:  {results['max_time']:.3f} s")
            print(f"    Memory:    {results['memory_mb']:.1f} MB")

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        results['error'] = str(e)
        traceback.print_exc()

    return results


def run_all_benchmarks(adata, n_components=30, n_runs=3, detailed_memory=False,
                       include_backed=False, backed_chunk_size=4096):
    """Run benchmarks for all methods and both input types."""
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE SVD BENCHMARK")
    print("="*70)
    print(f"Data shape: {adata.shape}")
    print(f"Components: {n_components}")
    print(f"Runs per test: {n_runs}")
    print(f"Memory tracking: {'psutil' if PSUTIL_AVAILABLE else 'disabled'}")

    all_results = []

    for method_code, method_name in SVD_METHODS.items():
        for input_type in ['sparse', 'dense']:
            result = benchmark_svd_method(
                adata,
                method_code,
                method_name,
                input_type=input_type,
                n_components=n_components,
                n_runs=n_runs,
                detailed_memory=detailed_memory
            )
            all_results.append(result)

    # Optionally benchmark PRIMME in backed mode
    if include_backed:
        backed_result = benchmark_primme_backed(
            adata, n_components=n_components, n_runs=n_runs,
            backed_chunk_size=backed_chunk_size
        )
        all_results.append(backed_result)

    return all_results


def create_summary_table(results):
    """Create a pandas DataFrame summarizing benchmark results."""
    summary_data = []

    for result in results:
        if result['success']:
            summary_data.append({
                'Method': result['method'],
                'Input Type': result['input_type'],
                'Mean Time (s)': result['mean_time'],
                'Std Time (s)': result['std_time'],
                'Min Time (s)': result['min_time'],
                'Max Time (s)': result['max_time'],
                'Memory (MB)': result['memory_mb'],
            })
        else:
            summary_data.append({
                'Method': result['method'],
                'Input Type': result['input_type'],
                'Mean Time (s)': None,
                'Std Time (s)': None,
                'Min Time (s)': None,
                'Max Time (s)': None,
                'Memory (MB)': None,
            })

    df = pd.DataFrame(summary_data)
    return df


def visualize_benchmark_results(results, output_path='../tests/benchmark_svd_results.png'):
    """Create comprehensive visualization of benchmark results."""

    # Filter successful results
    successful = [r for r in results if r['success']]

    if not successful:
        print("\nNo successful results to visualize.")
        return

    # Collect all methods and input types actually present
    methods = sorted(list(set([r['method'] for r in successful])))
    all_input_types = sorted(list(set([r['input_type'] for r in successful])))
    has_backed = 'backed' in all_input_types

    INPUT_COLORS = {
        'sparse': 'steelblue',
        'dense': 'coral',
        'backed': '#2ca02c',
    }

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    n_types = len(all_input_types)
    width = 0.8 / n_types  # bar width scales with number of types

    # Helper to find a result
    def _find(method, input_type):
        return next((r for r in successful
                     if r['method'] == method and r['input_type'] == input_type), None)

    # 1. Execution Time Comparison (Bar plot)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))

    for ti, itype in enumerate(all_input_types):
        times = []
        stds = []
        for method in methods:
            r = _find(method, itype)
            times.append(r['mean_time'] if r else 0)
            stds.append(r['std_time'] if r else 0)
        offset = (ti - (n_types - 1) / 2) * width
        ax1.bar(x + offset, times, width, label=itype.capitalize(), alpha=0.8,
                color=INPUT_COLORS.get(itype, 'gray'), yerr=stds, capsize=4)

    ax1.set_xlabel('SVD Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Memory Usage Comparison (Bar plot)
    ax2 = fig.add_subplot(gs[0, 1])

    for ti, itype in enumerate(all_input_types):
        mems = []
        for method in methods:
            r = _find(method, itype)
            mems.append(r['memory_mb'] if r and r['memory_mb'] is not None else 0)
        offset = (ti - (n_types - 1) / 2) * width
        ax2.bar(x + offset, mems, width, label=itype.capitalize(), alpha=0.8,
                color=INPUT_COLORS.get(itype, 'gray'))

    ax2.set_xlabel('SVD Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Peak Memory Increase (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    if not PSUTIL_AVAILABLE:
        ax2.text(0.5, 0.5, 'Memory tracking unavailable\n(psutil not installed)',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='red', weight='bold')

    # 3. Speedup Factor (Sparse vs Dense, and optionally Sparse vs Backed)
    ax3 = fig.add_subplot(gs[1, 0])

    # Build comparison pairs: dense/sparse, and backed/sparse if available
    speed_pairs = [('dense', 'sparse')]
    if has_backed:
        speed_pairs.append(('backed', 'sparse'))

    pair_width = 0.8 / len(speed_pairs)
    for pi, (num_type, den_type) in enumerate(speed_pairs):
        factors = []
        for method in methods:
            r_num = _find(method, num_type)
            r_den = _find(method, den_type)
            if r_num and r_den and r_den['mean_time'] > 0:
                factors.append(r_num['mean_time'] / r_den['mean_time'])
            else:
                factors.append(0)
        offset = (pi - (len(speed_pairs) - 1) / 2) * pair_width
        colors_bar = [INPUT_COLORS.get(num_type, 'gray')] * len(methods)
        ax3.bar(x + offset, factors, pair_width, alpha=0.7,
                color=colors_bar,
                label=f'{num_type.capitalize()} / {den_type.capitalize()}')

        for i, v in enumerate(factors):
            if v > 0:
                ax3.text(i + offset, v + 0.05, f'{v:.2f}x',
                         ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax3.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No difference')
    ax3.set_xlabel('SVD Method', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time Ratio (vs Sparse)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Speed', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Detailed Timing Box Plot
    ax4 = fig.add_subplot(gs[1, 1])

    timing_data = []
    labels = []
    colors_box = []

    for method in methods:
        for input_type in all_input_types:
            result = _find(method, input_type)
            if result:
                timing_data.append(result['times'])
                labels.append(f"{method}\n{input_type}")
                colors_box.append(INPUT_COLORS.get(input_type, 'gray'))

    if timing_data:
        bp = ax4.boxplot(timing_data, tick_labels=labels, patch_artist=True, showmeans=True)

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Overall title
    fig.suptitle('SVD Algorithm Benchmark Results', fontsize=18, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark visualization saved to: {output_path}")

    plt.close()


def print_summary(results):
    """Print a formatted summary of benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    # Create and display summary table
    df = create_summary_table(results)
    print("\n" + df.to_string(index=False))

    # Find fastest methods
    print("\n" + "-"*70)
    print("FASTEST METHODS:")
    print("-"*70)

    successful = [r for r in results if r['success']]

    if successful:
        # Sparse fastest
        sparse_results = [r for r in successful if r['input_type'] == 'sparse']
        if sparse_results:
            fastest_sparse = min(sparse_results, key=lambda x: x['mean_time'])
            print(f"Sparse input:  {fastest_sparse['method']:10s} - {fastest_sparse['mean_time']:.3f} s")

        # Dense fastest
        dense_results = [r for r in successful if r['input_type'] == 'dense']
        if dense_results:
            fastest_dense = min(dense_results, key=lambda x: x['mean_time'])
            print(f"Dense input:   {fastest_dense['method']:10s} - {fastest_dense['mean_time']:.3f} s")

        # Backed fastest
        backed_results = [r for r in successful if r['input_type'] == 'backed']
        if backed_results:
            fastest_backed = min(backed_results, key=lambda x: x['mean_time'])
            print(f"Backed input:  {fastest_backed['method']:10s} - {fastest_backed['mean_time']:.3f} s")

    # Memory efficiency
    if PSUTIL_AVAILABLE:
        print("\n" + "-"*70)
        print("MEMORY EFFICIENCY:")
        print("-"*70)

        memory_results = [r for r in successful if r.get('memory_mb') is not None]

        if memory_results:
            # Sparse lowest memory
            sparse_memory = [r for r in memory_results if r['input_type'] == 'sparse']
            if sparse_memory:
                lowest_sparse = min(sparse_memory, key=lambda x: x['memory_mb'])
                print(f"Sparse input:  {lowest_sparse['method']:10s} - {lowest_sparse['memory_mb']:.1f} MB")

            # Dense lowest memory
            dense_memory = [r for r in memory_results if r['input_type'] == 'dense']
            if dense_memory:
                lowest_dense = min(dense_memory, key=lambda x: x['memory_mb'])
                print(f"Dense input:   {lowest_dense['method']:10s} - {lowest_dense['memory_mb']:.1f} MB")

            # Backed memory
            backed_memory = [r for r in memory_results if r['input_type'] == 'backed']
            if backed_memory:
                lowest_backed = min(backed_memory, key=lambda x: x['memory_mb'])
                print(f"Backed input:  {lowest_backed['method']:10s} - {lowest_backed['memory_mb']:.1f} MB")

    print("\n" + "="*70)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Benchmark SVD algorithms in ACTIONet')
    parser.add_argument('--components', type=int, default=30,
                       help='Number of components to compute (default: 30)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of benchmark runs per test (default: 3)')
    parser.add_argument('--detailed-memory', action='store_true',
                       help='Use detailed memory profiling (requires memory_profiler)')
    parser.add_argument('--include-backed', action='store_true',
                       help='Include PRIMME backed (operator) mode benchmark')
    parser.add_argument('--backed-chunk-size', type=int, default=4096,
                       help='Chunk size for backed mode operator (default: 4096)')

    args = parser.parse_args()

    print("="*70)
    print("SVD ALGORITHM BENCHMARK")
    print("="*70)
    print(f"Components: {args.components}")
    print(f"Runs: {args.runs}")
    print(f"Detailed memory: {args.detailed_memory}")
    print(f"Include backed mode: {args.include_backed}")

    # Load and prepare data
    adata = load_and_prepare_data()

    # Run benchmarks
    results = run_all_benchmarks(
        adata,
        n_components=args.components,
        n_runs=args.runs,
        detailed_memory=args.detailed_memory,
        include_backed=args.include_backed,
        backed_chunk_size=args.backed_chunk_size,
    )

    # Print summary
    print_summary(results)

    # Create visualizations
    visualize_benchmark_results(results)

    # Save detailed results to CSV
    df = create_summary_table(results)
    output_csv = '../tests/benchmark_svd_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
