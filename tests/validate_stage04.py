#!/usr/bin/env python3
"""Stage 04 Validation: Python Frontend Adaptation + Boundary Optimization.

Verifies:
 - All obsm slots have shape (n_obs, k): cells x components
 - All varm slots have shape (n_vars, k): genes x components
 - SVD u/v roles are correct with new cells x genes orientation
 - No transpose operations at the C++ boundary
 - Full pipeline runs without errors

Usage:
    cd actionet-python
    source .venv/bin/activate
    python tests/validate_stage04.py
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import anndata

# Locate fixtures relative to this script
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(TESTS_DIR, "fixtures", "parity_fixture.h5ad")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, msg: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f": {msg}" if msg else ""))
        FAIL += 1


def main():
    global PASS, FAIL
    print("=" * 65)
    print("  Stage 04 Validation: Python Frontend Adaptation")
    print("=" * 65)

    import actionet

    adata = anndata.read_h5ad(FIXTURE)
    print(f"\nDataset: {adata.n_obs} cells × {adata.n_vars} genes\n")

    # ----------------------------------------------------------------
    # 1. reduce_kernel — S_r shape
    # ----------------------------------------------------------------
    print("1. reduce_kernel")
    K = 15
    actionet.reduce_kernel(adata, n_components=K, seed=42, verbose=False)
    check("S_r shape is (cells, k)", adata.obsm["action"].shape == (adata.n_obs, K),
          f"got {adata.obsm['action'].shape}")
    check("U shape is (genes, k)", adata.varm["action_U"].shape == (adata.n_vars, K),
          f"got {adata.varm['action_U'].shape}")
    check("B shape is (cells, p)", adata.obsm["action_B"].shape[0] == adata.n_obs,
          f"got {adata.obsm['action_B'].shape}")
    check("A shape is (genes, p)", adata.varm["action_A"].shape[0] == adata.n_vars,
          f"got {adata.varm['action_A'].shape}")
    sigma = np.asarray(adata.uns["action_params"]["sigma"]).ravel()
    check("sigma is 1D (k,)", sigma.ndim == 1 and sigma.shape[0] == K,
          f"got {sigma.shape}")

    # Verify SVD consistency: X * U / sigma ≈ S_r
    X = adata.X
    U = adata.varm["action_U"]
    XU = np.asarray(X.dot(U) if sp.issparse(X) else X @ U)
    XU_scaled = XU / sigma[np.newaxis, :]
    corrs = [abs(np.corrcoef(adata.obsm["action"][:, i], XU_scaled[:, i])[0, 1])
             for i in range(K)]
    check("SVD consistency (X*U/sigma ≈ S_r)", np.mean(corrs) > 0.90,
          f"mean abs-corr={np.mean(corrs):.4f}")

    # ----------------------------------------------------------------
    # 2. run_action — H/C shapes
    # ----------------------------------------------------------------
    print("\n2. run_action")
    actionet.run_action(adata, k_max=10)
    for key in ["H_stacked", "H_merged", "C_stacked", "C_merged"]:
        check(f"{key} shape (cells, archetypes)",
              adata.obsm[key].shape[0] == adata.n_obs,
              f"got {adata.obsm[key].shape}")

    # ----------------------------------------------------------------
    # 3. build_network
    # ----------------------------------------------------------------
    print("\n3. build_network")
    actionet.build_network(adata)
    check("network shape (cells, cells)",
          adata.obsp["actionet"].shape == (adata.n_obs, adata.n_obs),
          f"got {adata.obsp['actionet'].shape}")

    # ----------------------------------------------------------------
    # 4. compute_network_diffusion
    # ----------------------------------------------------------------
    print("\n4. compute_network_diffusion")
    actionet.compute_network_diffusion(adata, scores="H_merged", key_added="footprint")
    check("footprint shape (cells, archetypes)",
          adata.obsm["footprint"].shape[0] == adata.n_obs,
          f"got {adata.obsm['footprint'].shape}")

    # ----------------------------------------------------------------
    # 5. layout_network
    # ----------------------------------------------------------------
    print("\n5. layout_network")
    actionet.layout_network(adata, seed=42, verbose=False)
    check("umap shape (cells, 2)",
          adata.obsm["X_umap"].shape == (adata.n_obs, 2),
          f"got {adata.obsm['X_umap'].shape}")

    # ----------------------------------------------------------------
    # 6. compute_archetype_feature_specificity
    # ----------------------------------------------------------------
    print("\n6. compute_archetype_feature_specificity")
    actionet.compute_archetype_feature_specificity(adata, archetype_key="footprint")
    for key in ["archetype_feat_profile", "archetype_feat_specificity_upper",
                "archetype_feat_specificity_lower"]:
        check(f"{key} shape (genes, archetypes)",
              adata.varm[key].shape[0] == adata.n_vars,
              f"got {adata.varm[key].shape}")

    # ----------------------------------------------------------------
    # 7. compute_feature_specificity (label-based)
    # ----------------------------------------------------------------
    print("\n7. compute_feature_specificity")
    labels = np.random.default_rng(42).integers(1, 6, size=adata.n_obs)
    actionet.compute_feature_specificity(adata, labels=labels)
    check("specificity_profile shape (genes, k)",
          adata.varm["specificity_profile"].shape[0] == adata.n_vars,
          f"got {adata.varm['specificity_profile'].shape}")

    # ----------------------------------------------------------------
    # 8. run_svd — u/v orientation with new cells x genes contract
    # ----------------------------------------------------------------
    print("\n8. run_svd (cells × genes input)")
    X = adata.X
    svd = actionet.run_svd(X, n_components=5, verbose=False)
    # With cells x genes input: u = cells x k, v = genes x k
    check("svd['u'] shape is (cells, k)",
          svd["u"].shape == (adata.n_obs, 5),
          f"got {svd['u'].shape}")
    check("svd['v'] shape is (genes, k)",
          svd["v"].shape == (adata.n_vars, 5),
          f"got {svd['v'].shape}")

    # ----------------------------------------------------------------
    # 9. Verify no transpose=True warnings when running normal pipeline
    # ----------------------------------------------------------------
    print("\n9. No DeprecationWarning from anndata_to_matrix")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adata2 = anndata.read_h5ad(FIXTURE)
        actionet.reduce_kernel(adata2, n_components=5, seed=42, verbose=False)
        actionet.run_action(adata2, k_max=5)
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                    and "transpose=True" in str(x.message)]
    check("no transpose=True deprecation warnings", len(dep_warnings) == 0,
          f"got {len(dep_warnings)} warnings")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"  Results: {PASS} PASS, {FAIL} FAIL")
    print("=" * 65)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
