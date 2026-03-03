#!/usr/bin/env python3
"""Benchmark in-memory vs backed ACTIONet workflows.

This script implements the benchmark workflow specified in
tests/AGENT_BENCHMARK_BACKED_EXTENSION.md.
"""

import gc
import json
import os
import shutil
import threading
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
from tabulate import tabulate

import actionet as an

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)

N_TRIALS = int(os.environ.get("ACTIONET_BENCH_TRIALS", "3"))
SVD_ALGORITHM = "primme"
BACKED_CHUNK_SIZE = int(os.environ.get("ACTIONET_BENCH_CHUNK_SIZE", "4096"))
SELECTED_DATASETS = {
    x.strip() for x in os.environ.get("ACTIONET_BENCH_DATASETS", "small,large").split(",") if x.strip()
}


class Profiler:
    """Wall-time and peak RSS delta tracker."""

    def __init__(self, label: str, sample_interval: float = 0.02):
        self.label = label
        self.sample_interval = sample_interval
        self.elapsed = 0.0
        self.peak_rss_mb = 0.0
        self.peak_abs_mb = 0.0

    def _sample_peak(self):
        while not self._stop_event.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / 1e6
                self.peak_abs_mb = max(self.peak_abs_mb, rss_mb)
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def __enter__(self):
        gc.collect()
        self._proc = psutil.Process()
        self._rss0_mb = self._proc.memory_info().rss / 1e6
        self.peak_abs_mb = self._rss0_mb
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_peak, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            self.peak_abs_mb = max(self.peak_abs_mb, self._proc.memory_info().rss / 1e6)
        except Exception:
            pass
        self.peak_rss_mb = max(0.0, self.peak_abs_mb - self._rss0_mb)
        return False


def run_workflow(adata, ds_cfg, mode_label, backed_chunk_size=4096):
    """Run benchmark workflow and return (metrics, result_artifacts)."""
    metrics = {}
    results = {}
    step_peak_abs = []

    is_backed = bool(getattr(adata, "isbacked", False))
    bcs = backed_chunk_size if is_backed else 4096

    label_col = ds_cfg["label_col"]
    batch_key = ds_cfg["batch_key"]
    n_comp = ds_cfg["n_components"]
    k_max = ds_cfg["k_max"]
    use_batch = batch_key is not None
    reduction_key = "action"
    effective_reduction = "action_corrected" if use_batch else "action"

    workflow_start_rss_mb = psutil.Process().memory_info().rss / 1e6

    def run_step(step_name, fn):
        with Profiler(step_name) as p:
            out = fn()
        metrics[step_name] = (p.elapsed, p.peak_rss_mb)
        step_peak_abs.append(p.peak_abs_mb)
        return out

    run_step(
        "filter",
        lambda: an.filter_anndata(
            adata,
            min_cells_per_feat=0.01,
            backed_chunk_size=bcs,
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] filter: {metrics['filter'][0]:.2f}s, shape={adata.shape}", flush=True)

    run_step(
        "normalize",
        lambda: an.normalize_anndata(
            adata,
            target_sum=1e4,
            log_transform=True,
            log_base=2,
            backed_chunk_size=bcs,
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] normalize: {metrics['normalize'][0]:.2f}s", flush=True)

    run_step(
        "reduce_kernel",
        lambda: an.reduce_kernel(
            adata,
            n_components=n_comp,
            key_added=reduction_key,
            svd_algorithm=SVD_ALGORITHM,
            backed_chunk_size=bcs,
            verbose=False,
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] reduce_kernel: {metrics['reduce_kernel'][0]:.2f}s", flush=True)

    if use_batch:
        run_step(
            "batch_correction",
            lambda: an.correct_batch_effect(
                adata,
                batch_key=batch_key,
                reduction_key=reduction_key,
                backed_chunk_size=bcs,
                inplace=True,
            ),
        )
        print(f"  [{mode_label}] batch_correction: {metrics['batch_correction'][0]:.2f}s", flush=True)

    run_step(
        "action_decomposition",
        lambda: an.run_action(
            adata,
            k_min=2,
            k_max=k_max,
            reduction_key=effective_reduction,
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] action_decomposition: {metrics['action_decomposition'][0]:.2f}s", flush=True)

    run_step(
        "network_construction",
        lambda: an.build_network(
            adata,
            obsm_key="H_stacked",
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] network_construction: {metrics['network_construction'][0]:.2f}s", flush=True)

    run_step(
        "archetype_diffusion",
        lambda: an.compute_network_diffusion(
            adata,
            scores="H_merged",
            key_added="archetype_footprint",
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] archetype_diffusion: {metrics['archetype_diffusion'][0]:.2f}s", flush=True)

    run_step(
        "umap_2d",
        lambda: an.layout_network(
            adata,
            n_components=2,
            key_added="umap_2d_actionet",
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] umap_2d: {metrics['umap_2d'][0]:.2f}s", flush=True)

    run_step(
        "umap_3d",
        lambda: an.layout_network(
            adata,
            n_components=3,
            key_added="umap_3d_actionet",
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] umap_3d: {metrics['umap_3d'][0]:.2f}s", flush=True)

    run_step(
        "color_computation",
        lambda: an.compute_node_colors(
            adata,
            embedding_key="umap_3d_actionet",
            key_added="colors_actionet",
        ),
    )
    print(f"  [{mode_label}] color_computation: {metrics['color_computation'][0]:.2f}s", flush=True)

    run_step(
        "feature_specificity",
        lambda: an.compute_archetype_feature_specificity(
            adata,
            archetype_key="archetype_footprint",
            key_added="archetype",
            backed_chunk_size=bcs,
            inplace=True,
        ),
    )
    print(f"  [{mode_label}] feature_specificity: {metrics['feature_specificity'][0]:.2f}s", flush=True)

    markers_df = run_step(
        "find_markers",
        lambda: an.find_markers(
            adata,
            labels=label_col,
            features_use="Gene",
            top_genes=30,
            return_type="dataframe",
            backed_chunk_size=bcs,
        ),
    )
    results["markers"] = markers_df
    print(f"  [{mode_label}] find_markers: {metrics['find_markers'][0]:.2f}s, shape={markers_df.shape}", flush=True)

    annot = run_step(
        "annotate_cells",
        lambda: an.annotate_cells(
            adata,
            markers_df,
            method="vision",
            features_use="Gene",
            backed_chunk_size=bcs,
        ),
    )
    results["annotation"] = annot
    print(f"  [{mode_label}] annotate_cells: {metrics['annotate_cells'][0]:.2f}s", flush=True)

    all_genes = adata.var["Gene"].dropna().astype(str).unique()
    rng = np.random.default_rng(42)
    impute_genes = list(rng.choice(all_genes, size=min(10, len(all_genes)), replace=False))
    imp_df = run_step(
        "impute_features",
        lambda: an.impute_features(
            adata,
            features=impute_genes,
            features_use="Gene",
            reduction_key=effective_reduction,
            backed_chunk_size=bcs,
        ),
    )
    results["imputation"] = imp_df
    results["impute_genes"] = impute_genes
    print(f"  [{mode_label}] impute_features: {metrics['impute_features'][0]:.2f}s, shape={imp_df.shape}", flush=True)

    total_time = sum(v[0] for v in metrics.values())
    total_peak = max(0.0, (max(step_peak_abs) - workflow_start_rss_mb) if step_peak_abs else 0.0)
    metrics["TOTAL"] = (total_time, total_peak)
    print(f"  [{mode_label}] TOTAL: {total_time:.2f}s, peak RSS delta: {total_peak:.1f} MB", flush=True)

    if effective_reduction in adata.obsm:
        results["reduction"] = np.array(adata.obsm[effective_reduction])

    return metrics, results


def check_parity(res_mem, res_backed):
    """Compare in-memory and backed outputs."""
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
        l1 = np.asarray(res_mem["annotation"]["labels"])
        l2 = np.asarray(res_backed["annotation"]["labels"])
        if len(l1) == len(l2):
            parity["annotation_agreement"] = float(np.mean(l1 == l2))

    if "imputation" in res_mem and "imputation" in res_backed:
        i1, i2 = res_mem["imputation"], res_backed["imputation"]
        shared = sorted(set(i1.columns) & set(i2.columns))
        corrs = []
        for feat in shared:
            x = np.asarray(i1[feat], dtype=float)
            y = np.asarray(i2[feat], dtype=float)
            if np.std(x) > 0 and np.std(y) > 0:
                corrs.append(np.corrcoef(x, y)[0, 1])
        parity["imputation_mean_corr"] = float(np.mean(corrs)) if corrs else float("nan")

    return parity


def aggregate_metrics(all_metrics):
    valid = [m for m in all_metrics if m is not None]
    if not valid:
        return {}
    keys = sorted({k for m in valid for k in m.keys()}, key=lambda x: (x == "TOTAL", x))
    agg = {}
    for key in keys:
        times = [m[key][0] for m in valid if key in m]
        mems = [m[key][1] for m in valid if key in m]
        if not times:
            continue
        agg[key] = {
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "mem_mean": float(np.mean(mems)),
            "mem_std": float(np.std(mems)),
        }
    return agg


def benchmark_dataset(ds_name, ds_cfg, trials, backed_chunk_size):
    """Run multiple in-memory and backed trials for one dataset."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"  BENCHMARKING: {ds_name} ({ds_cfg['file']})", flush=True)
    print(f"{'=' * 70}", flush=True)

    src_path = DATA_DIR / ds_cfg["file"]
    all_mem_metrics = []
    all_bck_metrics = []
    last_mem_res = None
    last_bck_res = None

    for trial in range(1, trials + 1):
        print(f"\n--- {ds_name} Trial {trial}/{trials} ---", flush=True)

        print("  Loading in-memory...", flush=True)
        adata_mem = ad.read_h5ad(str(src_path))
        gc.collect()
        try:
            m, r = run_workflow(adata_mem, ds_cfg, f"mem-t{trial}", backed_chunk_size=backed_chunk_size)
            all_mem_metrics.append(m)
            last_mem_res = r
        except Exception as exc:
            print(f"  ERROR (in-memory trial {trial}): {exc}", flush=True)
            traceback.print_exc()
            all_mem_metrics.append(None)
        del adata_mem
        gc.collect()

        backed_path = str(OUT_DIR / f"backed_{ds_name}_t{trial}.h5ad")
        print("  Preparing backed copy...", flush=True)
        shutil.copy2(str(src_path), backed_path)
        print("  Opening backed...", flush=True)
        adata_bck = ad.read_h5ad(backed_path, backed="r+")
        gc.collect()
        try:
            print("  Decompressing backed file (scope='file')...", flush=True)
            an.decompress_backed_storage(
                adata_bck,
                scope="file",
                chunk_size=backed_chunk_size,
                verbose=False,
            )
            m, r = run_workflow(adata_bck, ds_cfg, f"bck-t{trial}", backed_chunk_size=backed_chunk_size)
            all_bck_metrics.append(m)
            last_bck_res = r
        except Exception as exc:
            print(f"  ERROR (backed trial {trial}): {exc}", flush=True)
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
    all_steps = sorted(set(agg_mem.keys()) | set(agg_bck.keys()), key=lambda x: (x == "TOTAL", x))
    rows = []
    for step in all_steps:
        m = agg_mem.get(step, {})
        b = agg_bck.get(step, {})
        mt = m.get("time_mean", 0.0)
        bt = b.get("time_mean", 0.0)
        speedup = f"{mt / max(bt, 1e-9):.2f}x" if mt > 0 and bt > 0 else "N/A"
        rows.append(
            {
                "Step": step,
                "Mem Time (s)": f"{mt:.2f} +/- {m.get('time_std', 0.0):.2f}",
                "Bck Time (s)": f"{bt:.2f} +/- {b.get('time_std', 0.0):.2f}",
                "Speedup (Mem/Bck)": speedup,
                "Mem Peak RSS (MB)": f"{m.get('mem_mean', 0.0):.0f} +/- {m.get('mem_std', 0.0):.0f}",
                "Bck Peak RSS (MB)": f"{b.get('mem_mean', 0.0):.0f} +/- {b.get('mem_std', 0.0):.0f}",
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(agg_mem, agg_bck, ds_name):
    steps = [k for k in agg_mem.keys() if k != "TOTAL"]
    if not steps:
        return None

    mem_times = [agg_mem[s]["time_mean"] for s in steps]
    bck_times = [agg_bck.get(s, {}).get("time_mean", 0.0) for s in steps]
    mem_mems = [agg_mem[s]["mem_mean"] for s in steps]
    bck_mems = [agg_bck.get(s, {}).get("mem_mean", 0.0) for s in steps]

    x = np.arange(len(steps))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(x - width / 2, mem_times, width, label="In-Memory", color="#4C72B0")
    axes[0].bar(x + width / 2, bck_times, width, label="Backed", color="#DD8452")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title(f"Runtime by Step - {ds_name}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
    axes[0].legend()

    axes[1].bar(x - width / 2, mem_mems, width, label="In-Memory", color="#4C72B0")
    axes[1].bar(x + width / 2, bck_mems, width, label="Backed", color="#DD8452")
    axes[1].set_ylabel("Peak RSS Delta (MB)")
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
    """Estimate runtime/memory for target cell counts at 10k genes."""
    n_cells, n_genes = ds_shape
    n_cells = max(int(n_cells), 1)
    n_genes = max(int(n_genes), 1)
    target_cells = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    target_genes = 10_000
    gene_ratio = target_genes / n_genes

    rows = []
    for mode, agg in [("in-memory", agg_mem), ("backed", agg_bck)]:
        if not agg:
            continue

        base_total = agg.get("TOTAL", {}).get("time_mean", 0.0)
        base_mem = agg.get("TOTAL", {}).get("mem_mean", 0.0)
        base_network = agg.get("network_construction", {}).get("time_mean", 0.0)

        base_linear = max(0.0, base_total - base_network)
        for nc in target_cells:
            cell_ratio = nc / n_cells
            network_term = base_network * cell_ratio * np.log1p(nc) / max(np.log1p(n_cells), 1.0)
            est_time = base_linear * cell_ratio * gene_ratio + network_term
            est_mem = base_mem * cell_ratio * gene_ratio
            rows.append(
                {
                    "Mode": mode,
                    "Cells": f"{nc:,}",
                    "Genes": f"{target_genes:,}",
                    "Est. Time (min)": f"{est_time / 60:.1f}",
                    "Est. Peak RSS (GB)": f"{est_mem / 1000:.1f}",
                }
            )
    return pd.DataFrame(rows)


def make_batch_scaling(agg_mem, agg_bck, ref_batches):
    """Estimate batch-correction scaling with batch count."""
    rows = []
    target_batches = [1, 5, 25, 50, 100]
    ref_batches = max(int(ref_batches), 1)

    for mode, agg in [("in-memory", agg_mem), ("backed", agg_bck)]:
        if not agg:
            continue
        bc_time = agg.get("batch_correction", {}).get("time_mean", 0.0)
        bc_mem = agg.get("batch_correction", {}).get("mem_mean", 0.0)
        if bc_time <= 0:
            continue
        for nb in target_batches:
            ratio = nb / ref_batches
            rows.append(
                {
                    "Mode": mode,
                    "Batches": nb,
                    "Est. Batch Corr Time (s)": f"{bc_time * ratio:.1f}",
                    "Est. Batch Corr Peak RSS (MB)": f"{bc_mem * ratio:.0f}",
                }
            )
    return pd.DataFrame(rows)


def _resolve_dataset_metadata(path: Path, batch_key: str | None):
    tmp = ad.read_h5ad(str(path), backed="r")
    shape = tmp.shape
    n_batches = 0
    if batch_key is not None and batch_key in tmp.obs:
        n_batches = int(pd.Series(tmp.obs[batch_key]).nunique())
    if hasattr(tmp, "file") and tmp.file is not None:
        tmp.file.close()
    del tmp
    gc.collect()
    return shape, n_batches


def main():
    datasets = {
        "small": {
            "file": "test_adata.h5ad",
            "label_col": "CellLabel",
            "batch_key": None,
            "n_components": 30,
            "k_max": 30,
        },
        "large": {
            "file": "adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad",
            "label_col": "CellType",
            "batch_key": "UID",
            "n_components": 30,
            "k_max": 30,
        },
    }

    report = []
    report.append("# ACTIONet Backed Extension Benchmark Report")
    report.append("")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Trials per mode: {N_TRIALS}")
    report.append(f"SVD algorithm: {SVD_ALGORITHM}")
    report.append(f"Backed chunk size: {BACKED_CHUNK_SIZE}")
    report.append("Backed decompression: scope='file' before every backed trial")
    report.append("")

    all_results = {}

    for ds_name, ds_cfg in datasets.items():
        if SELECTED_DATASETS and ds_name not in SELECTED_DATASETS:
            continue

        src = DATA_DIR / ds_cfg["file"]
        if not src.exists():
            print(f"SKIP {ds_name}: {src} not found", flush=True)
            continue

        shape, n_batches = _resolve_dataset_metadata(src, ds_cfg["batch_key"])
        ds_cfg_run = dict(ds_cfg)
        ds_cfg_run["ref_batches"] = n_batches

        mem_metrics, bck_metrics, parity = benchmark_dataset(
            ds_name,
            ds_cfg_run,
            trials=N_TRIALS,
            backed_chunk_size=BACKED_CHUNK_SIZE,
        )

        agg_mem = aggregate_metrics(mem_metrics)
        agg_bck = aggregate_metrics(bck_metrics)
        table = make_comparison_table(agg_mem, agg_bck)

        report.append(f"## Dataset: {ds_name} ({ds_cfg['file']})")
        report.append(f"Shape: {shape[0]:,} cells x {shape[1]:,} genes")
        if ds_cfg["batch_key"]:
            report.append(f"Batch key: {ds_cfg['batch_key']} ({n_batches} batches)")
        report.append(f"Label column: {ds_cfg['label_col']}")
        report.append("")
        report.append("### Runtime & Peak Memory Comparison")
        report.append("")
        report.append(tabulate(table, headers="keys", tablefmt="pipe", showindex=False))
        report.append("")
        report.append("### Result Parity (In-Memory vs Backed)")
        report.append("")
        if parity:
            for key, value in parity.items():
                report.append(f"- **{key}**: {value:.4f}")
        else:
            report.append("- No parity data (one or both modes failed)")
        report.append("")

        fig_path = plot_comparison(agg_mem, agg_bck, ds_name)
        if fig_path is not None:
            report.append(f"![Benchmark {ds_name}]({fig_path.name})")
            report.append("")

        proj = make_scaling_projection(agg_mem, agg_bck, shape)
        if not proj.empty:
            report.append("### Scaling Projections (Cell Count)")
            report.append("")
            report.append(tabulate(proj, headers="keys", tablefmt="pipe", showindex=False))
            report.append("")

        if n_batches > 0:
            batch_proj = make_batch_scaling(agg_mem, agg_bck, n_batches)
            if not batch_proj.empty:
                report.append("### Scaling Projections (Batch Count)")
                report.append("")
                report.append(tabulate(batch_proj, headers="keys", tablefmt="pipe", showindex=False))
                report.append("")

        all_results[ds_name] = {
            "shape": list(shape),
            "n_batches": int(n_batches),
            "agg_mem": agg_mem,
            "agg_bck": agg_bck,
            "parity": parity,
        }

    report_path = OUT_DIR / "BENCHMARK_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print(f"\nReport written to: {report_path}", flush=True)

    json_path = OUT_DIR / "benchmark_raw.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, default=str)
    print(f"Raw data: {json_path}", flush=True)


if __name__ == "__main__":
    main()
