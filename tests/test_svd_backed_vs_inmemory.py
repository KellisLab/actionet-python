"""
Test script to verify consistency between PRIMME backed (operator) and in-memory modes.

Mirrors the structure of test_svd_sparse_vs_dense.py but compares:
  - PRIMME in-memory sparse  vs  PRIMME backed (operator path)
  - PRIMME in-memory dense   vs  PRIMME backed (operator path)

Backed mode creates a temporary h5ad file on disk and opens it with
``anndata.read_h5ad(..., backed='r')``.  The operator path then performs
chunked matrix-vector products through a Python callback, exercising the
full OOM SVD pipeline (PythonMatrixOperator -> PRIMME -> perturbedSVD ->
kernel reduction).

Output figures (saved alongside this script):
  - svd_backed_vs_inmemory_comparison.png   (scatter facets)
  - svd_backed_vs_inmemory_summary.png      (correlation / error bars)
  - svd_backed_vs_inmemory_sigma.png        (singular-value overlay)
"""

import sys
import os
import tempfile
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import anndata
import scanpy as sc
import scipy.sparse as sp

# Add parent directory to path if needed
sys.path.insert(0, '../src')

import actionet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALG_PRIMME = 3
INPUT_MODES = {
    'sparse':  'PRIMME in-memory (sparse)',
    'dense':   'PRIMME in-memory (dense)',
    'backed':  'PRIMME backed (operator)',
}
# Backed-mode comparison is inherently less tight than sparse-vs-dense because
# the operator path routes through Python callbacks and PRIMME's iterative
# solver may converge to a slightly different subspace.
CORR_THRESHOLD = 0.99        # per-component |correlation|
SIGMA_REL_THRESHOLD = 0.05   # mean |Δσ/σ|


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """Load test data and perform basic preprocessing."""
    print("Loading test data...")
    adata = anndata.read_h5ad("../data/test_adata.h5ad")
    print(f"Data shape: {adata.shape}")

    print("Filtering data...")
    sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))

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
    """Validate output keys exist with correct shapes and _U (not _V) naming."""
    prefix = f"[{label}] " if label else ""
    errors = []

    if key not in adata.obsm:
        errors.append(f"{prefix}Missing adata.obsm['{key}']")
    else:
        S_r = adata.obsm[key]
        if S_r.shape != (adata.n_obs, n_components):
            errors.append(f"{prefix}obsm['{key}'] shape {S_r.shape} != "
                          f"({adata.n_obs}, {n_components})")

    if f"{key}_B" not in adata.obsm:
        errors.append(f"{prefix}Missing adata.obsm['{key}_B']")

    if f"{key}_U" not in adata.varm:
        errors.append(f"{prefix}Missing adata.varm['{key}_U']")
    else:
        U = adata.varm[f"{key}_U"]
        if U.shape[0] != adata.n_vars:
            errors.append(f"{prefix}varm['{key}_U'] rows {U.shape[0]} != {adata.n_vars}")
        if U.shape[1] != n_components:
            errors.append(f"{prefix}varm['{key}_U'] cols {U.shape[1]} != {n_components}")

    if f"{key}_A" not in adata.varm:
        errors.append(f"{prefix}Missing adata.varm['{key}_A']")

    if f"{key}_V" in adata.varm:
        errors.append(f"{prefix}STALE KEY: varm['{key}_V'] should be '{key}_U'")

    if f"{key}_params" not in adata.uns:
        errors.append(f"{prefix}Missing adata.uns['{key}_params']")
    elif "sigma" not in adata.uns[f"{key}_params"]:
        errors.append(f"{prefix}Missing 'sigma' in params")

    for slot in [key, f"{key}_B"]:
        if slot in adata.obsm:
            arr = adata.obsm[slot]
            if np.any(np.isnan(arr)):
                errors.append(f"{prefix}NaN in obsm['{slot}']")
            if np.any(np.isinf(arr)):
                errors.append(f"{prefix}Inf in obsm['{slot}']")

    for slot in [f"{key}_U", f"{key}_A"]:
        if slot in adata.varm:
            arr = adata.varm[slot]
            if np.any(np.isnan(arr)):
                errors.append(f"{prefix}NaN in varm['{slot}']")
            if np.any(np.isinf(arr)):
                errors.append(f"{prefix}Inf in varm['{slot}']")

    return errors


# ---------------------------------------------------------------------------
# Per-mode runners
# ---------------------------------------------------------------------------
def run_primme_sparse(adata, n_components, seed=42):
    """Run PRIMME with in-memory sparse input."""
    ad = adata.copy()
    lc = ad.layers['logcounts']
    if not sp.issparse(lc):
        ad.layers['logcounts'] = sp.csr_matrix(lc)

    actionet.reduce_kernel(
        ad, n_components=n_components, layer='logcounts',
        key_added='action', svd_algorithm=ALG_PRIMME,
        max_iter=0, seed=seed, verbose=True, inplace=True,
    )
    return ad


def run_primme_dense(adata, n_components, seed=42):
    """Run PRIMME with in-memory dense input."""
    ad = adata.copy()
    lc = ad.layers['logcounts']
    if sp.issparse(lc):
        ad.layers['logcounts'] = lc.toarray()

    actionet.reduce_kernel(
        ad, n_components=n_components, layer='logcounts',
        key_added='action', svd_algorithm=ALG_PRIMME,
        max_iter=0, seed=seed, verbose=True, inplace=True,
    )
    return ad


def run_primme_backed(adata, n_components, seed=42, chunk_size=2048):
    """Run PRIMME with backed (operator) input via a temp h5ad on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "backed.h5ad")

        # Write sparse .X to disk
        ad_w = adata.copy()
        lc = ad_w.layers['logcounts']
        ad_w.X = sp.csr_matrix(lc) if not sp.issparse(lc) else sp.csr_matrix(lc)
        ad_w.write_h5ad(path)
        del ad_w

        ad = anndata.read_h5ad(path, backed='r')
        print(f"  Backed type: {type(ad.X)}, shape: {ad.shape}")

        actionet.reduce_kernel(
            ad, n_components=n_components, layer=None,
            key_added='action', svd_algorithm=ALG_PRIMME,
            max_iter=0, seed=seed, verbose=True, inplace=True,
            backed_chunk_size=chunk_size,
        )
        # Copy results out before the tmpdir is removed
        result = {
            'obsm_action': ad.obsm['action'].copy(),
            'sigma': np.array(ad.uns['action_params']['sigma']).copy(),
            'U': ad.varm['action_U'].copy(),
            'A': ad.varm['action_A'].copy(),
            'B': ad.obsm['action_B'].copy(),
            'params': dict(ad.uns['action_params']),
            'adata': ad,   # kept alive until we validate keys
        }
        key_errors = validate_reduction_keys(ad, 'action', n_components, "BACKED")
        result['key_errors'] = key_errors
        return result


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def test_single_pair(adata, mode_a, mode_b, n_components=30):
    """
    Run two modes and compare their results.

    Returns a dict with 'a', 'b' sub-dicts and a 'comparison' sub-dict.
    """
    label_a = INPUT_MODES[mode_a]
    label_b = INPUT_MODES[mode_b]
    print(f"\n{'='*60}")
    print(f"Comparing: {label_a}  vs  {label_b}")
    print(f"{'='*60}")

    runners = {
        'sparse': run_primme_sparse,
        'dense':  run_primme_dense,
        'backed': run_primme_backed,
    }

    out = {
        'a': {'mode': mode_a, 'success': False, 'error': None},
        'b': {'mode': mode_b, 'success': False, 'error': None},
        'comparison': None,
    }

    for side, mode in [('a', mode_a), ('b', mode_b)]:
        label = INPUT_MODES[mode]
        print(f"\n--- Running {label} ---")
        try:
            if mode == 'backed':
                res = runners[mode](adata, n_components)
                out[side]['reduction'] = res['obsm_action']
                out[side]['sigma'] = res['sigma']
                key_errors = res['key_errors']
                if res['params'].get('operator_mode', False):
                    print(f"  ✓ Operator mode recorded in params")
            else:
                ad = runners[mode](adata, n_components)
                out[side]['reduction'] = ad.obsm['action']
                out[side]['sigma'] = np.asarray(
                    ad.uns['action_params']['sigma']
                ).flatten()
                key_errors = validate_reduction_keys(
                    ad, 'action', n_components, label
                )

            if key_errors:
                for e in key_errors:
                    print(f"  ⚠ {e}")
            else:
                print(f"  ✓ All output keys valid (_U convention)")

            out[side]['success'] = True
            print(f"  Shape: {out[side]['reduction'].shape}")
            print(f"  ✓ {label} succeeded")

        except Exception as exc:
            out[side]['error'] = str(exc)
            print(f"  ✗ {label} failed: {exc}")
            traceback.print_exc()

    # Compare
    if out['a']['success'] and out['b']['success']:
        out['comparison'] = _compare_reductions(
            out['a']['reduction'], out['b']['reduction'],
            out['a']['sigma'], out['b']['sigma'],
            label_a, label_b,
        )

    return out


def _compare_reductions(red_a, red_b, sigma_a, sigma_b, label_a, label_b):
    """Per-component correlation + sigma comparison."""
    sigma_a = np.asarray(sigma_a).flatten()
    sigma_b = np.asarray(sigma_b).flatten()

    print(f"\n--- Comparison ---")

    # Per-component absolute correlation (handles sign flips)
    n_comp = min(red_a.shape[1], red_b.shape[1])
    comp_corrs = np.array([
        np.abs(np.corrcoef(red_a[:, i], red_b[:, i])[0, 1])
        for i in range(n_comp)
    ])

    # Sigma relative difference
    sigma_abs_diff = np.abs(sigma_a[:n_comp] - sigma_b[:n_comp])
    sigma_rel_diff = sigma_abs_diff / (np.abs(sigma_a[:n_comp]) + 1e-10)

    # Flattened reduction correlation (for overview)
    flat_corr = np.abs(np.corrcoef(
        red_a[:, :n_comp].flatten(), red_b[:, :n_comp].flatten()
    )[0, 1])

    # Reduction element-wise absolute & relative diffs
    red_abs_diff = np.abs(red_a[:, :n_comp] - red_b[:, :n_comp])
    red_rel_diff = red_abs_diff / (np.abs(red_a[:, :n_comp]) + 1e-10)

    print(f"  Per-component |corr|  mean: {np.mean(comp_corrs):.6f}  "
          f"min: {np.min(comp_corrs):.6f}")
    print(f"  Flattened |corr|:           {flat_corr:.6f}")
    print(f"  Reduction abs diff   mean: {np.mean(red_abs_diff):.6e}  "
          f"max: {np.max(red_abs_diff):.6e}")
    print(f"  Reduction rel diff   mean: {np.mean(red_rel_diff):.6e}  "
          f"max: {np.max(red_rel_diff):.6e}")
    print(f"  Sigma abs diff       mean: {np.mean(sigma_abs_diff):.6e}  "
          f"max: {np.max(sigma_abs_diff):.6e}")
    print(f"  Sigma rel diff       mean: {np.mean(sigma_rel_diff):.6e}  "
          f"max: {np.max(sigma_rel_diff):.6e}")

    is_consistent = (
        np.mean(comp_corrs) > CORR_THRESHOLD and
        np.mean(sigma_rel_diff) < SIGMA_REL_THRESHOLD
    )
    status = "✓ CONSISTENT" if is_consistent else "⚠ DIFFERENT"
    print(f"  Verdict: {status}")

    return {
        'component_corrs': comp_corrs,
        'mean_component_corr': float(np.mean(comp_corrs)),
        'min_component_corr': float(np.min(comp_corrs)),
        'flat_correlation': float(flat_corr),
        'reduction_mean_abs_diff': float(np.mean(red_abs_diff)),
        'reduction_max_abs_diff': float(np.max(red_abs_diff)),
        'reduction_mean_rel_diff': float(np.mean(red_rel_diff)),
        'reduction_max_rel_diff': float(np.max(red_rel_diff)),
        'sigma_mean_abs_diff': float(np.mean(sigma_abs_diff)),
        'sigma_max_abs_diff': float(np.max(sigma_abs_diff)),
        'sigma_mean_rel_diff': float(np.mean(sigma_rel_diff)),
        'sigma_max_rel_diff': float(np.max(sigma_rel_diff)),
        'is_consistent': is_consistent,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_comparison(all_results):
    """
    Three-row faceted figure per pair:
      row 0 – mode A scatter (component 1 vs 2)
      row 1 – mode B scatter
      row 2 – A vs B value scatter (sampled) with diagonal
    """
    pairs = [k for k, v in all_results.items() if v['comparison'] is not None]
    if not pairs:
        print("\nNo successful comparisons — cannot create visualization.")
        return

    n = len(pairs)
    fig = plt.figure(figsize=(6 * n, 14))
    gs = GridSpec(3, n, figure=fig, hspace=0.35, wspace=0.30)

    for idx, pair_key in enumerate(pairs):
        res = all_results[pair_key]
        red_a = res['a']['reduction']
        red_b = res['b']['reduction']
        comp = res['comparison']

        label_a = INPUT_MODES[res['a']['mode']]
        label_b = INPUT_MODES[res['b']['mode']]

        # Row 0: mode A
        ax0 = fig.add_subplot(gs[0, idx])
        ax0.scatter(red_a[:, 0], red_a[:, 1], alpha=0.5, s=5, c='#1f77b4')
        ax0.set_title(label_a, fontsize=11, fontweight='bold')
        ax0.set_xlabel('Component 1')
        ax0.set_ylabel('Component 2')
        ax0.grid(True, alpha=0.3)

        # Row 1: mode B
        ax1 = fig.add_subplot(gs[1, idx])
        ax1.scatter(red_b[:, 0], red_b[:, 1], alpha=0.5, s=5, c='#d62728')
        ax1.set_title(label_b, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.grid(True, alpha=0.3)

        # Row 2: value scatter
        ax2 = fig.add_subplot(gs[2, idx])
        flat_a = red_a[:, :2].flatten()
        flat_b = red_b[:, :2].flatten()
        n_pts = len(flat_a)
        sample = np.random.choice(n_pts, min(2000, n_pts), replace=False)
        ax2.scatter(flat_a[sample], flat_b[sample], alpha=0.3, s=3)
        lims = [
            min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1]),
        ]
        ax2.plot(lims, lims, 'k-', alpha=0.5, zorder=1)
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_title(f"|corr|={comp['flat_correlation']:.6f}", fontsize=10)
        ax2.set_xlabel(label_a)
        ax2.set_ylabel(label_b)
        ax2.grid(True, alpha=0.3)

    fig.suptitle('PRIMME Backed vs In-Memory Comparison',
                 fontsize=16, fontweight='bold')

    out = '../tests/svd_backed_vs_inmemory_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nComparison figure saved to: {out}")
    plt.show()


def plot_summary_bars(all_results):
    """Bar charts: per-component correlation and mean relative error."""
    pairs = [k for k, v in all_results.items() if v['comparison'] is not None]
    if not pairs:
        return

    corrs = [all_results[k]['comparison']['mean_component_corr'] for k in pairs]
    min_corrs = [all_results[k]['comparison']['min_component_corr'] for k in pairs]
    rel_diffs = [all_results[k]['comparison']['reduction_mean_rel_diff'] for k in pairs]
    sigma_diffs = [all_results[k]['comparison']['sigma_mean_rel_diff'] for k in pairs]

    labels = [f"{INPUT_MODES[all_results[k]['a']['mode']]}\nvs\n"
              f"{INPUT_MODES[all_results[k]['b']['mode']]}" for k in pairs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1 – Mean per-component correlation
    x = np.arange(len(pairs))
    ax = axes[0]
    bars = ax.bar(x, corrs, color='steelblue', alpha=0.7, label='mean')
    ax.bar(x, min_corrs, color='navy', alpha=0.4, width=0.4, label='min')
    ax.axhline(CORR_THRESHOLD, color='r', ls='--', label=f'{CORR_THRESHOLD} threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('|Correlation|')
    ax.set_title('Per-Component Correlation', fontweight='bold')
    ax.set_ylim([min(min_corrs + [CORR_THRESHOLD]) - 0.02, 1.001])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 2 – Reduction mean relative diff
    ax = axes[1]
    ax.bar(x, rel_diffs, color='coral', alpha=0.7)
    ax.axhline(0.01, color='r', ls='--', label='1% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean Relative Difference')
    ax.set_title('Reduction Mean Relative Error', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 3 – Sigma mean relative diff
    ax = axes[2]
    ax.bar(x, sigma_diffs, color='mediumpurple', alpha=0.7)
    ax.axhline(SIGMA_REL_THRESHOLD, color='r', ls='--',
               label=f'{SIGMA_REL_THRESHOLD*100:.0f}% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean |Δσ/σ|')
    ax.set_title('Singular Value Relative Error', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = '../tests/svd_backed_vs_inmemory_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Summary figure saved to: {out}")
    plt.show()


def plot_singular_values(all_results):
    """Overlay singular-value curves for each mode tested."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'sparse': '#1f77b4', 'dense': '#ff7f0e', 'backed': '#d62728'}
    plotted = set()

    for pair_key, res in all_results.items():
        for side in ('a', 'b'):
            if not res[side]['success']:
                continue
            mode = res[side]['mode']
            if mode in plotted:
                continue
            plotted.add(mode)

            sigma = np.asarray(res[side]['sigma']).flatten()
            ax.plot(range(1, len(sigma) + 1), sigma, 'o-',
                    label=INPUT_MODES[mode], color=colors.get(mode, 'gray'),
                    alpha=0.7, markersize=4)

    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Decay — PRIMME Modes', fontsize=14,
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    out = '../tests/svd_backed_vs_inmemory_sigma.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Singular-value figure saved to: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ACTIONet PRIMME Backed vs In-Memory Validation Test")
    print("=" * 60)

    adata = load_and_prepare_data()
    n_components = 30

    # Two comparison pairs:
    #   1. sparse in-memory  vs  backed
    #   2. dense  in-memory  vs  backed
    all_results = {}

    for inmem_mode in ('sparse', 'dense'):
        key = f"{inmem_mode}_vs_backed"
        all_results[key] = test_single_pair(
            adata, inmem_mode, 'backed', n_components
        )

    # Summary table
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    all_ok = True
    for pair_key, res in all_results.items():
        label_a = INPUT_MODES[res['a']['mode']]
        label_b = INPUT_MODES[res['b']['mode']]

        status_a = "✓" if res['a']['success'] else "✗"
        status_b = "✓" if res['b']['success'] else "✗"

        print(f"\n{pair_key}:")
        print(f"  {label_a}: {status_a}")
        if not res['a']['success'] and res['a'].get('error'):
            print(f"    Error: {res['a']['error']}")
        print(f"  {label_b}: {status_b}")
        if not res['b']['success'] and res['b'].get('error'):
            print(f"    Error: {res['b']['error']}")

        if res['comparison'] is not None:
            c = res['comparison']
            tag = "✓ CONSISTENT" if c['is_consistent'] else "⚠ DIFFERENT"
            print(f"  Comparison: {tag}")
            print(f"    Mean component |corr|:  {c['mean_component_corr']:.6f}")
            print(f"    Min  component |corr|:  {c['min_component_corr']:.6f}")
            print(f"    Reduction mean rel err: {c['reduction_mean_rel_diff']:.6e}")
            print(f"    Sigma mean rel err:     {c['sigma_mean_rel_diff']:.6e}")
            if not c['is_consistent']:
                all_ok = False
        else:
            all_ok = False

    # Visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    visualize_comparison(all_results)
    plot_summary_bars(all_results)
    plot_singular_values(all_results)

    # Final verdict
    print("\n" + "=" * 60)
    print("Final Verdict:")
    print("=" * 60)

    if all_ok:
        print("✓ All PRIMME backed vs in-memory comparisons are CONSISTENT!")
    else:
        print("⚠ Some comparisons show differences "
              "(may be expected with iterative solver + chunked I/O).")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
