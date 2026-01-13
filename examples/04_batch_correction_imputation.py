"""
Example 4: Batch Correction and Imputation with ACTIONet

This example demonstrates:
1. Correcting batch effects using orthogonalization
2. Imputing gene expression using network diffusion
3. Correcting for basal/housekeeping gene expression
4. Smoothing reduced representations
"""

import numpy as np
from scipy.sparse import random as sparse_random
from anndata import AnnData
import actionet as an

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Example 4: Batch Correction and Imputation")
print("=" * 80)

# ============================================================================
# 1. Create synthetic data with batch effects
# ============================================================================
print("\n1. Creating synthetic data with batch effects...")

n_cells_per_batch = 50
n_genes = 200
n_batches = 3

# Create data for each batch
X_batches = []
batch_labels = []

for batch_idx in range(n_batches):
    # Each batch has a different baseline expression level
    baseline = 5 + batch_idx * 2

    X_batch = sparse_random(n_cells_per_batch, n_genes, density=0.3, format='csr', random_state=batch_idx)
    X_batch.data = np.random.poisson(baseline, size=X_batch.data.shape)

    X_batches.append(X_batch)
    batch_labels.extend([f"batch_{batch_idx+1}"] * n_cells_per_batch)

# Concatenate batches
from scipy.sparse import vstack
X = vstack(X_batches)

# Create AnnData
adata = AnnData(X)
adata.obs["batch"] = batch_labels
adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
adata.var_names = [f"gene_{i}" for i in range(n_genes)]

print(f"   Created AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"   Batches: {adata.obs['batch'].value_counts().to_dict()}")

# ============================================================================
# 2. Run basic preprocessing and reduction
# ============================================================================
print("\n2. Running ACTION reduction...")

an.reduce_kernel(adata, k=15, reduction_key="action", verbose=False)
print(f"   Reduced to {adata.obsm['action'].shape[1]} dimensions")

# ============================================================================
# 3. Batch effect correction
# ============================================================================
print("\n3. Correcting batch effects...")

an.correct_batch_effect(
    adata,
    batch_key="batch",
    reduction_key="action",
    corrected_key="action_corrected"
)

print("   Batch-corrected reduction stored in adata.obsm['action_corrected']")
print(f"   Original reduction shape: {adata.obsm['action'].shape}")
print(f"   Corrected reduction shape: {adata.obsm['action_corrected'].shape}")

# Compare variance before and after correction
var_before = np.var(adata.obsm["action"], axis=0).mean()
var_after = np.var(adata.obsm["action_corrected"], axis=0).mean()
print(f"   Average variance before correction: {var_before:.4f}")
print(f"   Average variance after correction: {var_after:.4f}")

# ============================================================================
# 4. Build networks on both original and corrected reductions
# ============================================================================
print("\n4. Building cell-cell networks...")

an.build_network(adata, reduction_key="action", network_key="actionet_original")
an.build_network(adata, reduction_key="action_corrected", network_key="actionet_corrected")

print("   Networks built on both original and corrected reductions")
print(f"   Original network density: {adata.obsp['actionet_original'].nnz / (adata.n_obs ** 2):.4f}")
print(f"   Corrected network density: {adata.obsp['actionet_corrected'].nnz / (adata.n_obs ** 2):.4f}")

# ============================================================================
# 5. Feature imputation using network diffusion
# ============================================================================
print("\n5. Imputing gene expression...")

# Select a subset of genes to impute
genes_to_impute = adata.var_names[:10].tolist()

# Impute using the corrected network
X_imputed = an.impute_features(
    adata,
    features=genes_to_impute,
    network_key="actionet_corrected",
    alpha=0.85,
    rescale=True
)

print(f"   Imputed expression for {len(genes_to_impute)} genes")
print(f"   Imputed matrix shape: {X_imputed.shape}")

# Compare original and imputed expression
original_expr = adata[:, genes_to_impute].X
if hasattr(original_expr, 'toarray'):
    original_expr = original_expr.toarray()

print(f"   Original expression range: [{original_expr.min():.2f}, {original_expr.max():.2f}]")
print(f"   Imputed expression range: [{X_imputed.min():.2f}, {X_imputed.max():.2f}]")

# Calculate correlation
correlation = np.corrcoef(X_imputed.flatten(), original_expr.flatten())[0, 1]
print(f"   Correlation between original and imputed: {correlation:.4f}")

# ============================================================================
# 6. Basal expression correction
# ============================================================================
print("\n6. Correcting for basal/housekeeping gene expression...")

# Select some genes as "housekeeping" genes
basal_genes = adata.var_names[:20].tolist()

an.correct_basal_expression(
    adata,
    basal_genes=basal_genes,
    reduction_key="action",
    corrected_key="action_basal_corrected"
)

print(f"   Corrected for {len(basal_genes)} basal genes")
print(f"   Basal-corrected reduction stored in adata.obsm['action_basal_corrected']")
print(f"   Basal genes recorded in adata.uns['action_basal_corrected_params']")

# ============================================================================
# 7. Kernel smoothing
# ============================================================================
print("\n7. Smoothing reduced representation...")

an.smooth_kernel(
    adata,
    reduction_key="action_corrected",
    smoothed_key="action_smoothed",
    alpha=0.85,
    max_iter=10
)

print("   Smoothed reduction stored in adata.obsm['action_smoothed']")

# Compare variance before and after smoothing
var_before_smooth = np.var(adata.obsm["action_corrected"], axis=0).mean()
var_after_smooth = np.var(adata.obsm["action_smoothed"], axis=0).mean()
print(f"   Average variance before smoothing: {var_before_smooth:.4f}")
print(f"   Average variance after smoothing: {var_after_smooth:.4f}")

# ============================================================================
# 8. Run ACTION on corrected data
# ============================================================================
print("\n8. Running ACTION decomposition on corrected reduction...")

an.run_action(
    adata,
    k_min=3,
    k_max=8,
    reduction_key="action_corrected"
)

n_archetypes = adata.obsm["H_merged"].shape[1]
print(f"   Identified {n_archetypes} archetypes")
print(f"   Cell assignments: {adata.obs['assigned_archetypes'].value_counts().to_dict()}")

# ============================================================================
# 9. Impute from archetypes
# ============================================================================
print("\n9. Imputing gene expression from archetype profiles...")

genes_for_archetype_impute = adata.var_names[10:20].tolist()

X_arch_imputed = an.impute_from_archetypes(
    adata,
    features=genes_for_archetype_impute,
    H_key="H_merged",
    rescale=True
)

print(f"   Imputed expression for {len(genes_for_archetype_impute)} genes using archetypes")
print(f"   Imputed matrix shape: {X_arch_imputed.shape}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary of stored results:")
print("=" * 80)
print("\nObservation matrices (adata.obsm):")
for key in adata.obsm.keys():
    print(f"   - {key}: {adata.obsm[key].shape}")

print("\nObservation pairwise matrices (adata.obsp):")
for key in adata.obsp.keys():
    shape = adata.obsp[key].shape
    nnz = adata.obsp[key].nnz
    print(f"   - {key}: {shape}, {nnz} non-zero entries")

print("\nVariable matrices (adata.varm):")
for key in adata.varm.keys():
    print(f"   - {key}: {adata.varm[key].shape}")

print("\nUnstructured annotations (adata.uns):")
for key in adata.uns.keys():
    print(f"   - {key}")

print("\nObservation annotations (adata.obs):")
for key in adata.obs.columns:
    print(f"   - {key}")

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
