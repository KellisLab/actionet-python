#!/usr/bin/env python3
"""Benchmark: In-memory vs Backed mode for ACTIONet workflows.

Compares runtime, peak memory, and result parity across two datasets.
Run with: python -u tests/benchmark_backed_extension.py
"""
import os, sys, time, gc, shutil, json, traceback
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import actionet as an
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)

N_TRIALS = 3
SVD_ALGORITHM = 3  # PRIMME


class Profiler:
    """Wall-time + RSS delta tracker."""
    def __init__(self, label):
        self.label = label
        self.elapsed = 0.0
        self.peak_rss_mb = 0.0
    def __enter__(self):
        gc.collect()
        self._rss0 = psutil.Process().memory_info().rss / 1e6
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        self.peak_rss_mb = max(0, psutil.Process().memory_info().rss / 1e6 - self._rss0)
        return False


def run_workflow(adata, ds_cfg, mode_label, backed_chunk_size=4096):
    """Run the full benchmark workflow. Returns (metrics_dict, result_artifacts)."""
    metrics = {}
    results = {}
    is_backed = getattr(adata, "isbacked", False)
    bcs = backed_chunk_size if is_backed else 4096

    label_col = ds_cfg["label_col"]
    batch_key = ds_cfg["batch_key"]
    n_comp = ds_cfg["n_components"]
    k_max = ds_cfg["k_max"]
    use_batch = batch_key is not None
    reduction_key = "action"
    effective_reduction = "action_corrected" if use_batch else "action"

    # 1) Filter
    with Profiler("filter") as p:
        an.filter_anndata(adata, min_cells_per_feat=0.01, backed_chunk_size=bcs)
    metrics["filter"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] filter: {p.elapsed:.2f}s, shape={adata.shape}", flush=True)

    # 2) Normalize + log2
    with Profiler("normalize") as p:
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True,
                             log_base=2, backed_chunk_size=bcs, inplace=True)
    metrics["normalize"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] normalize: {p.elapsed:.2f}s", flush=True)

    # 3) Reduce kernel (PRIMME)
    with Profiler("reduce_kernel") as p:
        an.reduce_kernel(adata, n_components=n_comp, key_added=reduction_key,
                         svd_algorithm=SVD_ALGORITHM, backed_chunk_size=bcs,
                         verbose=False, inplace=True)
    metrics["reduce_kernel"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] reduce_kernel: {p.elapsed:.2f}s", flush=True)

    # 4) Batch correction (if applicable)
    if use_batch:
        with Profiler("batch_correction") as p:
            an.correct_batch_effect(adata, batch_key=batch_key,
                                    reduction_key=reduction_key,
                                    backed_chunk_size=bcs, inplace=True)
        metrics["batch_correction"] = (p.elapsed, p.peak_rss_mb)
        print(f"  [{mode_label}] batch_correction: {p.elapsed:.2f}s", flush=True)

    # 5a) ACTION decomposition
    with Profiler("run_action") as p:
        an.run_action(adata, k_min=2, k_max=k_max,
                      reduction_key=effective_reduction, inplace=True)
    metrics["run_action"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] run_action: {p.elapsed:.2f}s", flush=True)

    # 5b) Build network
    with Profiler("build_network") as p:
        an.build_network(adata, obsm_key="H_stacked", inplace=True)
    metrics["build_network"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] build_network: {p.elapsed:.2f}s", flush=True)

    # 5c) Archetype diffusion
    with Profiler("diffusion") as p:
        an.compute_network_diffusion(adata, scores="H_merged",
                                     key_added="archetype_footprint", inplace=True)
    metrics["diffusion"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] diffusion: {p.elapsed:.2f}s", flush=True)

    # 5d) UMAP 2D
    with Profiler("layout_2d") as p:
        an.layout_network(adata, n_components=2, key_added="umap_2d_actionet",
                          inplace=True)
    metrics["layout_2d"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] layout_2d: {p.elapsed:.2f}s", flush=True)

    # 5e) UMAP 3D + colors
    with Profiler("layout_3d") as p:
        an.layout_network(adata, n_components=3, key_added="umap_3d_actionet",
                          inplace=True)
    metrics["layout_3d"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] layout_3d: {p.elapsed:.2f}s", flush=True)

    with Profiler("node_colors") as p:
        an.compute_node_colors(adata, embedding_key="umap_3d_actionet",
                               key_added="colors_actionet")
    metrics["node_colors"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] node_colors: {p.elapsed:.2f}s", flush=True)

    # 5f) Archetype feature specificity
    with Profiler("arch_specificity") as p:
        an.compute_archetype_feature_specificity(
            adata, archetype_key="archetype_footprint",
            key_added="archetype", backed_chunk_size=bcs, inplace=True)
    metrics["arch_specificity"] = (p.elapsed, p.peak_rss_mb)
    print(f"  [{mode_label}] arch_specificity: {p.elapsed:.2f}s", flush=True)

    # 6) Find markers (top 30)
    with Profiler("find_markers") as p:
        markers_df = an.find_markers(adata, labels=label_col, features_use="Gene",
                                     top_genes=30, return_type="dataframe",
                                     backed_chunk_size=bcs)
    metrics["find_markers"] = (p.elapsed, p.peak_rss_mb)
    results["markers"] = markers_df
    print(f"  [{mode_label}] find_markers: {p.elapsed:.2f}s, shape={markers_df.shape}", flush=True)

    # 7) Annotate cells (vision)
    with Profiler("annotate_cells") as p:
        annot = an.annotate_cells(adata, markers_df, method="vision",
                                  features_use="Gene", backed_chunk_size=bcs)
    metrics["annotate_cells"] = (p.elapsed, p.peak_rss_mb)
    results["annotation"] = annot
    print(f"  [{mode_label}] annotate_cells: {p.elapsed:.2f}s", flush=True)

    # 8) Impute 10 features
    all_genes = adata.var["Gene"].dropna().unique()
    rng = np.random.default_rng(42)
    impute_genes = list(rng.choice(all_genes, size=min(10, len(all_genes)), replace=False))
    with Profiler("impute_features") as p:
        imp_df = an.impute_features(adata, features=impute_genes,
                                    features_use="Gene",
                                    reduction_key=effective_reduction,
                                    backed_chunk_size=bcs)
    metrics["impute_features"] = (p.elapsed, p.peak_rss_mb)
    results["imputation"] = imp_df
    results["impute_genes"] = impute_genes
    print(f"  [{mode_label}] impute_features: {p.elapsed:.2f}s, shape={imp_df.shape}", flush=True)

    # Total
    total_time = sum(v[0] for v in metrics.values())
    total_peak = max(v[1] for v in metrics.values())
    metrics["TOTAL"] = (total_time, total_peak)
    print(f"  [{mode_label}] TOTAL: {total_time:.2f}s, peak RSS delta: {total_peak:.1f} MB", flush=True)

    if effective_reduction in adata.obsm:
        results["reduction"] = np.array(adata.obsm[effective_reduction])

    return metrics, results


def check_parity(res_mem, res_backed):
    """Compare in-memory vs backed results."""
    parity = {}
    if "reduction" in res_mem and "reduction" in res_backed:
        r_mem, r_bck = res_mem["reduction"], res_backed["reduction"]
        n = min(r_mem.shape[1], r_bck.shape[1])
        corrs = []
        for i in range(n):
            x, y = r_mem[:, i].ravel(), r_bck[:, i].ravel()
            if np.std(x) > 0 and np.std(y) > 0:
                corrs.append(abs(np.corrcoef(x, y)[0, 1]))
        parity["reduction_mean_abs_corr"] = float(np.mean(corrs)) if corrs else float("nan")

    if "markers" in res_mem and "markers" in res_backed:
        m1, m2 = res_mem["markers"], res_backed["markers"]
        cols = sorted(set(m1.columns) & set(m2.columns))
        overlaps = []
        for c in cols:
            s1 = set(m1[c].dropna().tolist()[:30])
            s2 = set(m2[c].dropna().tolist()[:30])
            if s1 and s2:
                overlaps.append(len(s1 & s2) / max(len(s1), len(s2)))
        parity["marker_overlap"] = float(np.mean(overlaps)) if overlaps else float("nan")

    if "annotation" in res_mem and "annotation" in res_backed:
        l1 = np.array(res_mem["annotation"]["labels"])
        l2 = np.array(res_backed["annotation"]["labels"])
        if len(l1) == len(l2):
            parity["annotation_agreement"] = float(np.mean(l1 == l2))

    if "imputation" in res_mem and "imputation" in res_backed:
        i1, i2 = res_mem["imputation"], res_backed["imputation"]
        shared = sorted(set(i1.columns) & set(i2.columns))
        corrs = []
        for f in shared:
            x = np.asarray(i1[f], dtype=float)
            y = np.asarray(i2[f], dtype=float)
            if np.std(x) > 0 and np.std(y) > 0:
                corrs.append(np.corrcoef(x, y)[0, 1])
        parity["imputation_mean_corr"] = float(np.mean(corrs)) if corrs else float("nan")
    return parity


def aggregate_metrics(all_metrics):
    valid = [m for m in all_metrics if m is not None]
    if not valid:
        return {}
    agg = {}
    for k in valid[0].keys():
        times = [m[k][0] for m in valid]
        mems = [m[k][1] for m in valid]
        agg[k] = {
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "mem_mean": float(np.mean(mems)),
            "mem_std": float(np.std(mems)),
        }
    return agg


def benchmark_dataset(ds_name, ds_cfg):
    """Run N_TRIALS of in-memory and backed workflows for one dataset."""
    print(f"\n{'='*70}", flush=True)
    print(f"  BENCHMARKING: {ds_name} ({ds_cfg['file']})", flush=True)
    print(f"{'='*70}", flush=True)

    src_path = DATA_DIR / ds_cfg["file"]
    all_mem_metrics = []
    all_bck_metrics = []
    last_mem_res = None
    last_bck_res = None

    for trial in range(1, N_TRIALS + 1):
        print(f"\n--- {ds_name} Trial {trial}/{N_TRIALS} ---", flush=True)

        # In-memory run
        print("  Loading in-memory...", flush=True)
        adata_mem = ad.read_h5ad(str(src_path))
        gc.collect()
        try:
            m, r = run_workflow(adata_mem, ds_cfg, f"mem-t{trial}")
            all_mem_metrics.append(m)
            last_mem_res = r
        except Exception as e:
            print(f"  ERROR (in-memory trial {trial}): {e}", flush=True)
            traceback.print_exc()
            all_mem_metrics.append(None)
        del adata_mem
        gc.collect()

        # Backed run
        backed_path = str(OUT_DIR / f"backed_{ds_name}_t{trial}.h5ad")
        print("  Preparing backed copy...", flush=True)
        shutil.copy2(str(src_path), backed_path)
        print("  Opening backed...", flush=True)
        adata_bck = ad.read_h5ad(backed_path, backed="r+")
        gc.collect()
        try:
            m, r = run_workflow(adata_bck, ds_cfg, f"bck-t{trial}")
            all_bck_metrics.append(m)
            last_bck_res = r
        except Exception as e:
            print(f"  ERROR (backed trial {trial}): {e}", flush=True)
            traceback.print_exc()
            all_bck_metrics.append(None)
        if hasattr(adata_bck, "file") and adata_bck.file is not None:
            try:
                adata_bck.file.close()
            except Exception:
                pass
        del adata_bck
        gc.collect()
        try:
            os.remove(backed_path)
        except Exception:
            pass

    parity = {}
    if last_mem_res and last_bck_res:
        parity = check_parity(last_mem_res, last_bck_res)

    return all_mem_metrics, all_bck_metrics, parity


def make_comparison_table(agg_mem, agg_bck):
    all_steps = list(agg_mem.keys())
    steps = [k for k in all_steps if k != "TOTAL"] + (["TOTAL"] if "TOTAL" in all_steps else [])
    rows = []
    for step in steps:
        m = agg_mem.get(step, {})
        b = agg_bck.get(step, {})
        mt = m.get("time_mean", 0)
        bt = b.get("time_mean", 0)
        speedup = f"{mt / max(bt, 0.001):.2f}x" if mt and bt else "N/A"
        rows.append({
            "Step": step,
            "Mem Time (s)": f"{mt:.2f} +/- {m.get('time_std', 0):.2f}",
            "Bck Time (s)": f"{bt:.2f} +/- {b.get('time_std', 0):.2f}",
            "Speedup": speedup,
            "Mem RSS (MB)": f"{m.get('mem_mean', 0):.0f} +/- {m.get('mem_std', 0):.0f}",
            "Bck RSS (MB)": f"{b.get('mem_mean', 0):.0f} +/- {b.get('mem_std', 0):.0f}",
        })
    return pd.DataFrame(rows)


def plot_comparison(agg_mem, agg_bck, ds_name):
    steps = [k for k in agg_mem.keys() if k != "TOTAL"]
    mem_times = [agg_mem[s]["time_mean"] for s in steps]
    bck_times = [agg_bck.get(s, {}).get("time_mean", 0) for s in steps]
    x = np.arange(len(steps))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(x - w / 2, mem_times, w, label="In-Memory", color="#4C72B0")
    axes[0].bar(x + w / 2, bck_times, w, label="Backed", color="#DD8452")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title(f"Runtime by Step - {ds_name}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
    axes[0].legend()

    mem_mems = [agg_mem[s]["mem_mean"] for s in steps]
    bck_mems = [agg_bck.get(s, {}).get("mem_mean", 0) for s in steps]
    axes[1].bar(x - w / 2, mem_mems, w, label="In-Memory", color="#4C72B0")
    axes[1].bar(x + w / 2, bck_mems, w, label="Backed", color="#DD8452")
    axes[1].set_ylabel("RSS Delta (MB)")
    axes[1].set_title(f"Peak Memory by Step - {ds_name}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
    axes[1].legend()

    plt.tight_layout()
    fig_path = OUT_DIR / f"benchmark_{ds_name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig_path}", flush=True)
    return fig_path


def make_scaling_projection(agg_mem, agg_bck, ds_shape):
    n_cells, n_genes = ds_shape
    target_cells = [100_000, 1_000_000, 10_000_000]
    rows = []
    for mode, agg in [("in-memory", agg_mem), ("backed", agg_bck)]:
        if not agg:
            continue
        base_total = agg.get("TOTAL", {}).get("time_mean", 0)
        base_mem = agg.get("TOTAL", {}).get("mem_mean", 0)
        base_kernel = agg.get("reduce_kernel", {}).get("time_mean", 0)
        base_action = agg.get("run_action", {}).get("time_mean", 0)
        base_net = agg.get("build_network", {}).get("time_mean", 0)
        base_norm = agg.get("normalize", {}).get("time_mean", 0)
        base_filter = agg.get("filter", {}).get("time_mean", 0)

        for nc in target_cells:
            cell_ratio = nc / n_cells
            est_time = (
                base_filter * cell_ratio
                + base_norm * cell_ratio
                + base_kernel * cell_ratio
                + base_action * cell_ratio
                + base_net * cell_ratio * np.log(nc) / max(np.log(n_cells), 1)
                + (base_total - base_filter - base_norm - base_kernel - base_action - base_net)
                * cell_ratio
            )
            est_mem = base_mem * cell_ratio
            rows.append({
                "Mode": mode,
                "Cells": f"{nc:,}",
                "Genes": "10,000",
                "Est. Time (min)": f"{est_time / 60:.1f}",
                "Est. Peak RSS (GB)": f"{est_mem / 1000:.1f}",
            })
    return pd.DataFrame(rows)


def make_batch_scaling(agg_mem, agg_bck, ds_shape, ref_batches):
    rows = []
    target_batches = [1, 5, 25, 50, 100]
    for mode, agg in [("in-memory", agg_mem), ("backed", agg_bck)]:
        if not agg:
            continue
        bc_time = agg.get("batch_correction", {}).get("time_mean", 0)
        bc_mem = agg.get("batch_correction", {}).get("mem_mean", 0)
        if bc_time == 0:
            continue
        for nb in target_batches:
            ratio = nb / ref_batches
            rows.append({
                "Mode": mode,
                "Batches": nb,
                "Est. Batch Corr Time (s)": f"{bc_time * ratio:.1f}",
                "Est. Batch Corr RSS (MB)": f"{bc_mem * ratio:.0f}",
            })
    return pd.DataFrame(rows)


def main():
    datasets = {
        "small": {
            "file": "test_adata.h5ad",
            "label_col": "CellLabel",
            "batch_key": None,
            "n_components": 30,
            "k_max": 30,
            "ref_batches": 0,
        },
        "large": {
            "file": "adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad",
            "label_col": "CellType",
            "batch_key": "UID",
            "n_components": 30,
            "k_max": 30,
            "ref_batches": 25,
        },
    }

    report = []
    report.append("# ACTIONet Backed Extension Benchmark Report\n")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Trials per mode: {N_TRIALS}")
    report.append(f"SVD Algorithm: PRIMME (id={SVD_ALGORITHM})\n")

    all_results = {}

    for ds_name, ds_cfg in datasets.items():
        src = DATA_DIR / ds_cfg["file"]
        if not src.exists():
            print(f"SKIP {ds_name}: {src} not found", flush=True)
            continue

        tmp = ad.read_h5ad(str(src), backed="r")
        shape = tmp.shape
        tmp.file.close()
        del tmp
        gc.collect()

        mem_metrics, bck_metrics, parity = benchmark_dataset(ds_name, ds_cfg)

        agg_mem = aggregate_metrics(mem_metrics)
        agg_bck = aggregate_metrics(bck_metrics)

        tbl = make_comparison_table(agg_mem, agg_bck)

        report.append(f"\n## Dataset: {ds_name} ({ds_cfg['file']})")
        report.append(f"Shape: {shape[0]:,} cells x {shape[1]:,} genes")
        if ds_cfg["batch_key"]:
            report.append(f"Batch key: {ds_cfg['batch_key']} ({ds_cfg['ref_batches']} batches)")
        report.append(f"Label column: {ds_cfg['label_col']}\n")

        report.append("### Runtime & Memory Comparison\n")
        report.append(tabulate(tbl, headers="keys", tablefmt="pipe", showindex=False))
        report.append("")

        report.append("\n### Result Parity (In-Memory vs Backed)\n")
        if parity:
            for k, v in parity.items():
                report.append(f"- **{k}**: {v:.4f}")
        else:
            report.append("- No parity data (one or both modes failed)")
        report.append("")

        if agg_mem and agg_bck:
            fig_path = plot_comparison(agg_mem, agg_bck, ds_name)
            report.append(f"\n![Benchmark {ds_name}]({fig_path.name})\n")

        proj = make_scaling_projection(agg_mem, agg_bck, shape)
        if not proj.empty:
            report.append("\n### Scaling Projections (Cell Count)\n")
            report.append(tabulate(proj, headers="keys", tablefmt="pipe", showindex=False))
            report.append("")

        if ds_cfg["ref_batches"] > 0:
            bproj = make_batch_scaling(agg_mem, agg_bck, shape, ds_cfg["ref_batches"])
            if not bproj.empty:
                report.append("\n### Scaling Projections (Batch Count)\n")
                report.append(tabulate(bproj, headers="keys", tablefmt="pipe", showindex=False))
                report.append("")

        all_results[ds_name] = {
            "shape": list(shape),
            "agg_mem": agg_mem,
            "agg_bck": agg_bck,
            "parity": parity,
        }

    # Write report
    report_path = OUT_DIR / "BENCHMARK_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport written to: {report_path}", flush=True)

    json_path = OUT_DIR / "benchmark_raw.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data: {json_path}", flush=True)


if __name__ == "__main__":
    main()
