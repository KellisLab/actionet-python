#!/usr/bin/env python3
"""Parity tests: In-memory vs Backed mode on the small dataset.

Runs the full workflow in both modes with identical seeds and parameters,
then compares every output artifact at each pipeline stage.
"""
import os, sys, time, gc, shutil, warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path
from collections import OrderedDict

import actionet as an

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)

SRC = DATA_DIR / "test_adata.h5ad"
BACKED_PATH = OUT_DIR / "parity_backed.h5ad"

SVD_ALG = "primme"
N_COMP = 30
K_MAX = 30
BCS = 4096


def parity_check(name, val_mem, val_bck, rtol=1e-4, atol=1e-6):
    """Compare two arrays/values and print result."""
    if val_mem is None or val_bck is None:
        print(f"  {name}: SKIP (one or both missing)")
        return None

    if isinstance(val_mem, (pd.DataFrame, pd.Series)):
        val_mem = val_mem.values
    if isinstance(val_bck, (pd.DataFrame, pd.Series)):
        val_bck = val_bck.values

    val_mem = np.asarray(val_mem, dtype=float) if not isinstance(val_mem, str) else val_mem
    val_bck = np.asarray(val_bck, dtype=float) if not isinstance(val_bck, str) else val_bck

    if isinstance(val_mem, str):
        match = val_mem == val_bck
        print(f"  {name}: {'PASS' if match else 'FAIL'} (str match)")
        return match

    if val_mem.shape != val_bck.shape:
        print(f"  {name}: FAIL (shape mismatch: {val_mem.shape} vs {val_bck.shape})")
        return False

    if val_mem.size == 0:
        print(f"  {name}: PASS (empty)")
        return True

    close = np.allclose(val_mem, val_bck, rtol=rtol, atol=atol, equal_nan=True)
    if close:
        max_diff = np.nanmax(np.abs(val_mem - val_bck))
        print(f"  {name}: PASS (max_diff={max_diff:.2e})")
        return True

    # Column-wise correlation for matrices
    if val_mem.ndim == 2 and val_mem.shape[1] > 1:
        corrs = []
        for i in range(val_mem.shape[1]):
            x, y = val_mem[:, i], val_bck[:, i]
            if np.std(x) > 0 and np.std(y) > 0:
                corrs.append(abs(np.corrcoef(x, y)[0, 1]))
        mean_corr = np.mean(corrs) if corrs else 0
        max_diff = np.nanmax(np.abs(val_mem - val_bck))
        rmse = np.sqrt(np.nanmean((val_mem - val_bck) ** 2))
        sign_match = "PASS" if mean_corr > 0.99 else "WARN" if mean_corr > 0.90 else "FAIL"
        print(f"  {name}: {sign_match} (mean_abs_corr={mean_corr:.6f}, max_diff={max_diff:.2e}, rmse={rmse:.2e})")
        return mean_corr > 0.90
    else:
        max_diff = np.nanmax(np.abs(val_mem - val_bck))
        rmse = np.sqrt(np.nanmean((val_mem - val_bck) ** 2))
        # For vectors, also report correlation
        flat_m, flat_b = val_mem.ravel(), val_bck.ravel()
        if np.std(flat_m) > 0 and np.std(flat_b) > 0:
            corr = np.corrcoef(flat_m, flat_b)[0, 1]
            sign = "PASS" if abs(corr) > 0.99 else "WARN" if abs(corr) > 0.90 else "FAIL"
            print(f"  {name}: {sign} (corr={corr:.6f}, max_diff={max_diff:.2e}, rmse={rmse:.2e})")
            return abs(corr) > 0.90
        else:
            print(f"  {name}: WARN (zero-variance, max_diff={max_diff:.2e})")
            return max_diff < atol


def parity_overlap(name, set_mem, set_bck, topn=None):
    """Compare overlap of gene sets or label arrays."""
    if topn:
        set_mem = set(list(set_mem)[:topn])
        set_bck = set(list(set_bck)[:topn])
    else:
        set_mem = set(set_mem)
        set_bck = set(set_bck)
    overlap = len(set_mem & set_bck) / max(len(set_mem | set_bck), 1)
    sign = "PASS" if overlap > 0.8 else "WARN" if overlap > 0.5 else "INFO"
    print(f"  {name}: {sign} (Jaccard={overlap:.3f}, |mem|={len(set_mem)}, |bck|={len(set_bck)}, |intersect|={len(set_mem & set_bck)})")
    return overlap


def main():
    results = OrderedDict()
    print("=" * 70)
    print("  PARITY TEST: In-Memory vs Backed — Small Dataset")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------
    print("\n[1] Loading datasets...")
    adata_mem = ad.read_h5ad(str(SRC))
    shutil.copy2(str(SRC), str(BACKED_PATH))
    adata_bck = ad.read_h5ad(str(BACKED_PATH), backed="r+")
    print(f"  In-memory: {adata_mem.shape}, dtype={adata_mem.X.dtype}")
    print(f"  Backed:    {adata_bck.shape}, isbacked={adata_bck.isbacked}")

    # ---------------------------------------------------------------
    # Filter
    # ---------------------------------------------------------------
    print("\n[2] Filter (min_cells_per_feat=0.01)...")
    an.filter_anndata(adata_mem, min_cells_per_feat=0.01, backed_chunk_size=BCS)
    an.filter_anndata(adata_bck, min_cells_per_feat=0.01, backed_chunk_size=BCS)
    print(f"  mem shape: {adata_mem.shape}")
    print(f"  bck shape: {adata_bck.shape}")
    results["filter_shape"] = adata_mem.shape == adata_bck.shape
    print(f"  filter_shape: {'PASS' if results['filter_shape'] else 'FAIL'}")

    # Compare gene sets retained
    results["filter_genes"] = parity_overlap(
        "filter_genes",
        adata_mem.var_names.tolist(),
        adata_bck.var_names.tolist(),
    )

    # ---------------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------------
    print("\n[3] Normalize (target_sum=1e4, log_base=2)...")
    an.normalize_anndata(adata_mem, target_sum=1e4, log_transform=True, log_base=2,
                         backed_chunk_size=BCS, inplace=True)
    an.normalize_anndata(adata_bck, target_sum=1e4, log_transform=True, log_base=2,
                         backed_chunk_size=BCS, inplace=True)

    # Sample rows and compare
    n_sample = min(500, adata_mem.n_obs)
    rng = np.random.default_rng(42)
    sample_idx = sorted(rng.choice(adata_mem.n_obs, size=n_sample, replace=False))

    X_mem_sample = adata_mem.X[sample_idx, :]
    X_bck_sample = adata_bck.X[sample_idx, :]
    if sp.issparse(X_mem_sample):
        X_mem_sample = X_mem_sample.toarray()
    if sp.issparse(X_bck_sample):
        X_bck_sample = np.asarray(X_bck_sample.toarray() if hasattr(X_bck_sample, 'toarray') else X_bck_sample)
    X_bck_sample = np.asarray(X_bck_sample)

    results["normalize_values"] = parity_check(
        "normalize_values (500 rows)", X_mem_sample, X_bck_sample, rtol=1e-5, atol=1e-8
    )

    # Row sums
    if sp.issparse(X_mem_sample):
        mem_rowsums = np.asarray(X_mem_sample.sum(axis=1)).ravel()
    else:
        mem_rowsums = np.asarray(X_mem_sample, dtype=np.float64).sum(axis=1).ravel()
    bck_rowsums = np.asarray(X_bck_sample, dtype=np.float64).sum(axis=1).ravel()
    results["normalize_rowsums"] = parity_check(
        "normalize_rowsums (500 rows)", mem_rowsums, bck_rowsums, rtol=1e-4
    )

    # ---------------------------------------------------------------
    # Reduce kernel
    # ---------------------------------------------------------------
    print("\n[4] Reduce kernel (PRIMME, k=30)...")
    an.reduce_kernel(adata_mem, n_components=N_COMP, key_added="action",
                     svd_algorithm=SVD_ALG, verbose=False, backed_chunk_size=BCS, inplace=True)
    an.reduce_kernel(adata_bck, n_components=N_COMP, key_added="action",
                     svd_algorithm=SVD_ALG, verbose=False, backed_chunk_size=BCS, inplace=True)

    S_mem = np.asarray(adata_mem.obsm["action"])
    S_bck = np.asarray(adata_bck.obsm["action"])
    results["reduce_S"] = parity_check("reduction S (obsm['action'])", S_mem, S_bck, rtol=1e-3)

    B_mem = np.asarray(adata_mem.obsm["action_B"])
    B_bck = np.asarray(adata_bck.obsm["action_B"])
    results["reduce_B"] = parity_check("reduction B (obsm['action_B'])", B_mem, B_bck, rtol=1e-3)

    U_mem = np.asarray(adata_mem.varm["action_U"])
    U_bck = np.asarray(adata_bck.varm["action_U"])
    results["reduce_U"] = parity_check("reduction U (varm['action_U'])", U_mem, U_bck, rtol=1e-3)

    sigma_mem = adata_mem.uns["action_params"]["sigma"]
    sigma_bck = adata_bck.uns["action_params"]["sigma"]
    results["reduce_sigma"] = parity_check("sigma", np.array(sigma_mem), np.array(sigma_bck), rtol=1e-4)

    # ---------------------------------------------------------------
    # run_action
    # ---------------------------------------------------------------
    print("\n[5] run_action (k_min=2, k_max=30)...")
    an.run_action(adata_mem, k_min=2, k_max=K_MAX, reduction_key="action", inplace=True)
    an.run_action(adata_bck, k_min=2, k_max=K_MAX, reduction_key="action", inplace=True)

    for key in ["H_stacked", "H_merged", "C_stacked", "C_merged"]:
        if key in adata_mem.obsm and key in adata_bck.obsm:
            results[f"action_{key}"] = parity_check(
                f"action {key}", np.asarray(adata_mem.obsm[key]), np.asarray(adata_bck.obsm[key]), rtol=1e-3
            )

    if "assigned_archetype" in adata_mem.obs and "assigned_archetype" in adata_bck.obs:
        aa_mem = np.asarray(adata_mem.obs["assigned_archetype"])
        aa_bck = np.asarray(adata_bck.obs["assigned_archetype"])
        agree = np.mean(aa_mem == aa_bck) if len(aa_mem) == len(aa_bck) else 0
        sign = "PASS" if agree > 0.8 else "WARN" if agree > 0.5 else "INFO"
        print(f"  assigned_archetype agreement: {sign} ({agree:.3f})")
        results["action_assigned_archetype"] = agree

    # ---------------------------------------------------------------
    # build_network
    # ---------------------------------------------------------------
    print("\n[6] build_network...")
    an.build_network(adata_mem, obsm_key="H_stacked", inplace=True)
    an.build_network(adata_bck, obsm_key="H_stacked", inplace=True)

    G_mem = adata_mem.obsp["actionet"]
    G_bck = adata_bck.obsp["actionet"]
    # Compare network structure: nnz, density
    nnz_mem = G_mem.nnz if sp.issparse(G_mem) else np.count_nonzero(G_mem)
    nnz_bck = G_bck.nnz if sp.issparse(G_bck) else np.count_nonzero(G_bck)
    nnz_ratio = min(nnz_mem, nnz_bck) / max(nnz_mem, nnz_bck) if max(nnz_mem, nnz_bck) > 0 else 1
    sign = "PASS" if nnz_ratio > 0.95 else "WARN" if nnz_ratio > 0.80 else "FAIL"
    print(f"  network nnz: {sign} (mem={nnz_mem:,}, bck={nnz_bck:,}, ratio={nnz_ratio:.4f})")
    results["network_nnz_ratio"] = nnz_ratio

    # ---------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------
    print("\n[7] compute_network_diffusion...")
    an.compute_network_diffusion(adata_mem, scores="H_merged", key_added="archetype_footprint", inplace=True)
    an.compute_network_diffusion(adata_bck, scores="H_merged", key_added="archetype_footprint", inplace=True)

    results["diffusion_footprint"] = parity_check(
        "archetype_footprint",
        np.asarray(adata_mem.obsm["archetype_footprint"]),
        np.asarray(adata_bck.obsm["archetype_footprint"]),
        rtol=1e-3,
    )

    # ---------------------------------------------------------------
    # layout_2d
    # ---------------------------------------------------------------
    print("\n[8] layout_network (2D, initial_coords='action')...")
    an.layout_network(adata_mem, n_components=2, initial_coords="action",
                      key_added="umap_2d_actionet", seed=0, inplace=True)
    an.layout_network(adata_bck, n_components=2, initial_coords="action",
                      key_added="umap_2d_actionet", seed=0, inplace=True)

    results["layout_2d"] = parity_check(
        "umap_2d_actionet",
        np.asarray(adata_mem.obsm["umap_2d_actionet"]),
        np.asarray(adata_bck.obsm["umap_2d_actionet"]),
        rtol=0.1,
    )

    # ---------------------------------------------------------------
    # layout_3d + colors
    # ---------------------------------------------------------------
    print("\n[9] layout_network (3D, initial_coords='action') + node_colors...")
    an.layout_network(adata_mem, n_components=3, initial_coords="action",
                      key_added="umap_3d_actionet", seed=0, inplace=True)
    an.layout_network(adata_bck, n_components=3, initial_coords="action",
                      key_added="umap_3d_actionet", seed=0, inplace=True)

    results["layout_3d"] = parity_check(
        "umap_3d_actionet",
        np.asarray(adata_mem.obsm["umap_3d_actionet"]),
        np.asarray(adata_bck.obsm["umap_3d_actionet"]),
        rtol=0.1,
    )

    an.compute_node_colors(adata_mem, embedding_key="umap_3d_actionet", key_added="colors_actionet")
    an.compute_node_colors(adata_bck, embedding_key="umap_3d_actionet", key_added="colors_actionet")

    results["node_colors"] = parity_check(
        "colors_actionet",
        np.asarray(adata_mem.obsm["colors_actionet"]),
        np.asarray(adata_bck.obsm["colors_actionet"]),
        rtol=0.1,
    )

    # ---------------------------------------------------------------
    # archetype feature specificity
    # ---------------------------------------------------------------
    print("\n[10] compute_archetype_feature_specificity...")
    an.compute_archetype_feature_specificity(
        adata_mem, archetype_key="archetype_footprint", key_added="archetype",
        backed_chunk_size=BCS, inplace=True)
    an.compute_archetype_feature_specificity(
        adata_bck, archetype_key="archetype_footprint", key_added="archetype",
        backed_chunk_size=BCS, inplace=True)

    for key in ["archetype_feat_profile", "archetype_feat_specificity_upper", "archetype_feat_specificity_lower"]:
        if key in adata_mem.varm and key in adata_bck.varm:
            results[f"specificity_{key}"] = parity_check(
                key,
                np.asarray(adata_mem.varm[key]),
                np.asarray(adata_bck.varm[key]),
                rtol=1e-2,
            )

    # ---------------------------------------------------------------
    # find_markers
    # ---------------------------------------------------------------
    print("\n[11] find_markers (top 30)...")
    markers_mem = an.find_markers(adata_mem, labels="CellLabel", features_use="Gene",
                                  top_genes=30, return_type="dataframe", backed_chunk_size=BCS)
    markers_bck = an.find_markers(adata_bck, labels="CellLabel", features_use="Gene",
                                  top_genes=30, return_type="dataframe", backed_chunk_size=BCS)

    print(f"  markers_mem shape: {markers_mem.shape}")
    print(f"  markers_bck shape: {markers_bck.shape}")
    results["markers_shape"] = markers_mem.shape == markers_bck.shape

    # Per-class marker overlap
    shared_cols = sorted(set(markers_mem.columns) & set(markers_bck.columns))
    overlaps = []
    for col in shared_cols:
        ol = parity_overlap(
            f"markers[{col}] top30",
            markers_mem[col].dropna().tolist(),
            markers_bck[col].dropna().tolist(),
            topn=30,
        )
        overlaps.append(ol)
    results["markers_mean_overlap"] = np.mean(overlaps) if overlaps else 0

    # Also get ranks for correlation
    ranks_mem = an.find_markers(adata_mem, labels="CellLabel", features_use="Gene",
                                result="ranks", return_type="dataframe", backed_chunk_size=BCS)
    ranks_bck = an.find_markers(adata_bck, labels="CellLabel", features_use="Gene",
                                result="ranks", return_type="dataframe", backed_chunk_size=BCS)
    rank_corrs = []
    for col in sorted(set(ranks_mem.columns) & set(ranks_bck.columns)):
        x = ranks_mem[col].values.astype(float)
        y = ranks_bck[col].values.astype(float)
        if np.std(x) > 0 and np.std(y) > 0:
            rank_corrs.append(np.corrcoef(x, y)[0, 1])
    results["marker_rank_corr"] = np.mean(rank_corrs) if rank_corrs else 0
    sign = "PASS" if results["marker_rank_corr"] > 0.9 else "WARN" if results["marker_rank_corr"] > 0.7 else "FAIL"
    print(f"  marker_rank_corr: {sign} ({results['marker_rank_corr']:.4f})")

    # ---------------------------------------------------------------
    # annotate_cells
    # ---------------------------------------------------------------
    print("\n[12] annotate_cells (vision)...")
    annot_mem = an.annotate_cells(adata_mem, markers_mem, method="vision",
                                  features_use="Gene", backed_chunk_size=BCS)
    annot_bck = an.annotate_cells(adata_bck, markers_bck, method="vision",
                                  features_use="Gene", backed_chunk_size=BCS)

    labels_mem = np.array(annot_mem["labels"])
    labels_bck = np.array(annot_bck["labels"])
    if len(labels_mem) == len(labels_bck):
        agree = np.mean(labels_mem == labels_bck)
        sign = "PASS" if agree > 0.8 else "WARN" if agree > 0.5 else "INFO"
        print(f"  label_agreement: {sign} ({agree:.4f})")
        results["annotation_agreement"] = agree

    conf_mem = np.array(annot_mem["confidence"], dtype=float)
    conf_bck = np.array(annot_bck["confidence"], dtype=float)
    results["annotation_confidence"] = parity_check("confidence", conf_mem, conf_bck, rtol=0.1)

    # Also test: annotate both with the SAME markers (mem markers) to isolate annotation vs marker effect
    print("\n  [12b] annotate_cells with SHARED markers (isolate annotation parity)...")
    annot_mem2 = an.annotate_cells(adata_mem, markers_mem, method="vision",
                                   features_use="Gene", backed_chunk_size=BCS)
    annot_bck2 = an.annotate_cells(adata_bck, markers_mem, method="vision",
                                   features_use="Gene", backed_chunk_size=BCS)
    labels_mem2 = np.array(annot_mem2["labels"])
    labels_bck2 = np.array(annot_bck2["labels"])
    if len(labels_mem2) == len(labels_bck2):
        agree2 = np.mean(labels_mem2 == labels_bck2)
        sign2 = "PASS" if agree2 > 0.8 else "WARN" if agree2 > 0.5 else "INFO"
        print(f"  shared_marker_label_agreement: {sign2} ({agree2:.4f})")
        results["annotation_shared_markers"] = agree2

    # Ground truth validation (CellLabel)
    print("\n  [12c] Annotation vs ground truth (CellLabel)...")
    gt = np.array(adata_mem.obs["CellLabel"])
    gt_agree_mem = np.mean(labels_mem == gt)
    gt_agree_bck = np.mean(labels_bck == gt)
    sign_gt_mem = "PASS" if gt_agree_mem > 0.8 else "WARN" if gt_agree_mem > 0.5 else "FAIL"
    sign_gt_bck = "PASS" if gt_agree_bck > 0.8 else "WARN" if gt_agree_bck > 0.5 else "FAIL"
    print(f"  mem vs ground truth: {sign_gt_mem} ({gt_agree_mem:.4f})")
    print(f"  bck vs ground truth: {sign_gt_bck} ({gt_agree_bck:.4f})")
    results["annotation_gt_mem"] = gt_agree_mem
    results["annotation_gt_bck"] = gt_agree_bck

    # ---------------------------------------------------------------
    # impute_features
    # ---------------------------------------------------------------
    print("\n[13] impute_features (10 genes)...")
    rng = np.random.default_rng(42)
    all_genes = adata_mem.var["Gene"].dropna().unique()
    impute_genes = list(rng.choice(all_genes, size=10, replace=False))

    imp_mem = an.impute_features(adata_mem, features=impute_genes, features_use="Gene",
                                 reduction_key="action", backed_chunk_size=BCS)
    imp_bck = an.impute_features(adata_bck, features=impute_genes, features_use="Gene",
                                 reduction_key="action", backed_chunk_size=BCS)

    shared_feats = sorted(set(imp_mem.columns) & set(imp_bck.columns))
    imp_corrs = []
    for f in shared_feats:
        x = np.asarray(imp_mem[f], dtype=float)
        y = np.asarray(imp_bck[f], dtype=float)
        if np.std(x) > 0 and np.std(y) > 0:
            c = np.corrcoef(x, y)[0, 1]
            imp_corrs.append(c)
            sign = "PASS" if abs(c) > 0.99 else "WARN" if abs(c) > 0.90 else "FAIL"
            print(f"  impute[{f}]: {sign} (corr={c:.6f})")
    results["imputation_mean_corr"] = np.mean(imp_corrs) if imp_corrs else 0
    print(f"  imputation_mean_corr: {results['imputation_mean_corr']:.6f}")

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------
    if hasattr(adata_bck, "file") and adata_bck.file is not None:
        try:
            adata_bck.file.close()
        except Exception:
            pass
    try:
        os.remove(str(BACKED_PATH))
    except Exception:
        pass

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PARITY TEST SUMMARY")
    print("=" * 70)

    pass_count = 0
    warn_count = 0
    fail_count = 0
    info_count = 0

    for name, val in results.items():
        if val is None:
            info_count += 1
        elif isinstance(val, bool):
            if val:
                pass_count += 1
            else:
                fail_count += 1
        elif isinstance(val, float):
            if val > 0.99:
                pass_count += 1
            elif val > 0.80:
                warn_count += 1
            elif val > 0.50:
                info_count += 1
            else:
                fail_count += 1

    print(f"\n  PASS: {pass_count}")
    print(f"  WARN: {warn_count}")
    print(f"  INFO: {info_count}")
    print(f"  FAIL: {fail_count}")
    print()

    print("  Detailed results:")
    for name, val in results.items():
        if isinstance(val, bool):
            print(f"    {name}: {'PASS' if val else 'FAIL'}")
        elif isinstance(val, float):
            sign = "PASS" if val > 0.99 else "WARN" if val > 0.80 else "INFO" if val > 0.50 else "FAIL"
            print(f"    {name}: {sign} ({val:.4f})")
        else:
            print(f"    {name}: {val}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
