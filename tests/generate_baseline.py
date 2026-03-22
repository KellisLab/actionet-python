"""
Baseline capture script for Plan 00 — Parity Baseline Infrastructure.

Runs the full canonical ACTIONet pipeline on the shared parity fixture and
saves all parity-critical slots to:
  - tests/fixtures/baseline_python.h5ad   (full AnnData)
  - tests/fixtures/baseline_python.npz    (individual arrays for comparison)

Usage (from the actionet-python repo root):
    python tests/generate_baseline.py

All outputs are deterministic given a fixed seed (42) and the frozen fixture.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

# Allow running from the repo root without pip install -e .
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import actionet as an

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "parity_fixture.h5ad"
)
OUT_H5AD = os.path.join(
    os.path.dirname(__file__), "fixtures", "baseline_python.h5ad"
)
OUT_NPZ = os.path.join(
    os.path.dirname(__file__), "fixtures", "baseline_python.npz"
)
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_fixture(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fixture not found: {path}\n"
            "Run libactionet/test/fixtures/generate_fixture.py first."
        )


def _extract_arrays(adata: ad.AnnData) -> dict[str, np.ndarray]:
    """Pull every parity-critical slot out of the AnnData into a flat dict."""
    arrays: dict[str, np.ndarray] = {}

    def _get(container, key: str, label: str) -> None:
        if key in container:
            v = container[key]
            if sp.issparse(v):
                # Store sparse as dense for .npz; keep COO for reconstruction
                arrays[f"{label}_data"] = v.data
                arrays[f"{label}_indices"] = v.indices
                arrays[f"{label}_indptr"] = v.indptr
                arrays[f"{label}_shape"] = np.array(v.shape)
            else:
                arrays[label] = np.asarray(v)
        else:
            print(f"  [MISSING] {label}")

    # Reduction
    _get(adata.obsm, "action",   "obsm_action")
    _get(adata.obsm, "action_B", "obsm_action_B")
    _get(adata.varm, "action_U", "varm_action_U")
    _get(adata.varm, "action_A", "varm_action_A")
    if "action_params" in adata.uns:
        arrays["uns_action_sigma"] = np.asarray(adata.uns["action_params"]["sigma"])
    else:
        print("  [MISSING] uns_action_params/sigma")

    # ACTION
    _get(adata.obsm, "H_stacked", "obsm_H_stacked")
    _get(adata.obsm, "H_merged",  "obsm_H_merged")
    _get(adata.obsm, "C_stacked", "obsm_C_stacked")
    _get(adata.obsm, "C_merged",  "obsm_C_merged")
    if "assigned_archetype" in adata.obs.columns:
        arrays["obs_assigned_archetype"] = np.asarray(adata.obs["assigned_archetype"])
    else:
        print("  [MISSING] obs/assigned_archetype")

    # Network
    if "actionet" in adata.obsp:
        G = adata.obsp["actionet"]
        if sp.issparse(G):
            G_csr = G.tocsr()
            arrays["obsp_actionet_data"]    = G_csr.data
            arrays["obsp_actionet_indices"] = G_csr.indices
            arrays["obsp_actionet_indptr"]  = G_csr.indptr
            arrays["obsp_actionet_shape"]   = np.array(G_csr.shape)
        else:
            arrays["obsp_actionet"] = np.asarray(G)
    else:
        print("  [MISSING] obsp/actionet")

    # Specificity (cluster)
    _get(adata.varm, "specificity_profile", "varm_specificity_profile")
    _get(adata.varm, "specificity_upper",   "varm_specificity_upper")
    _get(adata.varm, "specificity_lower",   "varm_specificity_lower")

    # Specificity (archetype)
    _get(adata.varm, "archetype_feat_profile",             "varm_archetype_feat_profile")
    _get(adata.varm, "archetype_feat_specificity_upper",   "varm_archetype_feat_specificity_upper")
    _get(adata.varm, "archetype_feat_specificity_lower",   "varm_archetype_feat_specificity_lower")

    # Batch correction
    _get(adata.obsm, "action_corrected",   "obsm_action_corrected")
    _get(adata.varm, "action_corrected_U", "varm_action_corrected_U")
    _get(adata.varm, "action_corrected_A", "varm_action_corrected_A")

    # Layout (UMAP coordinates)
    _get(adata.obsm, "actionet_2d", "obsm_actionet_2d")

    return arrays


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(adata: ad.AnnData) -> ad.AnnData:
    print("Step 1: reduce_kernel ...")
    an.reduce_kernel(adata, n_components=20, layer="logcounts", seed=SEED, verbose=False)

    print("Step 2: run_action ...")
    an.run_action(adata, k_min=2, k_max=20, reduction_key="action")

    print("Step 3: build_network ...")
    an.build_network(adata)

    # Compute cluster labels from archetype assignment for specificity
    labels = adata.obs["assigned_archetype"].astype(int).values

    print("Step 4: compute_feature_specificity ...")
    an.compute_feature_specificity(
        adata,
        labels=labels,
        layer="logcounts",
        key_added="specificity",
    )

    print("Step 5: compute_archetype_feature_specificity ...")
    # The archetype footprint key expected is H_merged (cells x archetypes)
    an.compute_archetype_feature_specificity(
        adata,
        archetype_key="H_merged",
        layer="logcounts",
        key_added="archetype",
    )

    print("Step 6: correct_batch_effect ...")
    an.correct_batch_effect(
        adata,
        batch_key="batch",
        reduction_key="action",
        corrected_suffix="corrected",
        layer="logcounts",
    )

    print("Step 7: layout_network ...")
    an.layout_network(adata, seed=SEED, verbose=False, key_added="actionet_2d")

    return adata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _check_fixture(FIXTURE_PATH)

    print(f"Loading fixture: {FIXTURE_PATH}")
    adata = ad.read_h5ad(FIXTURE_PATH)
    print(f"  Shape: {adata.shape}")

    print("\nRunning pipeline (seed={})...".format(SEED))
    adata = run_pipeline(adata)

    print(f"\nSaving full AnnData → {OUT_H5AD}")
    adata.write_h5ad(OUT_H5AD, compression="gzip")

    print(f"Extracting parity arrays → {OUT_NPZ}")
    arrays = _extract_arrays(adata)
    np.savez_compressed(OUT_NPZ, **arrays)

    print(f"\nSaved {len(arrays)} arrays:")
    for k, v in sorted(arrays.items()):
        print(f"  {k:50s}  shape={np.asarray(v).shape}  dtype={np.asarray(v).dtype}")

    print("\nPython baseline complete.")


if __name__ == "__main__":
    main()
