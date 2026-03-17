#!/usr/bin/env python3
"""benchmark_backed_extension.py — ACTIONet Scaling Benchmark Suite orchestrator.

See: tests/ACTIONet Scaling Benchmark Suite.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_support as bs
from benchmark_support import (
    DATASET_MANIFEST, DATA_DIR, BENCHMARK_DATA_DIR,
    N_THREADS, SVD_ALGORITHM, N_COMPONENTS,
    TIMEOUT_SECONDS, RSS_KILL_GB,
    STAGE_RSS_LIMIT_GB, STAGE_TIME_LIMIT_SECONDS,
    TIER_LABELS, TIERS, MIN_LABEL_CELLS, MIN_LABELS_FOR_ANNOTATION,
    StageProfiler, ResultRow, append_result, is_case_complete, make_case_id,
    dataset_path, trials_for_tier, generate_report, build_summary_csv,
    PYTHON_EXE, BENCHMARK_RESULTS_DIR,
)

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scipy.sparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
# HDF5 file locking: when the Python h5py handle holds the file open (backed
# mode) and C++ _core.create_backed_operator tries to open the same file, the
# second open fails with EAGAIN (errno=11). The benchmark closes adata.file
# before calling reduce_kernel, but some OS/kernel configurations still have the
# lock transiently held. Setting HDF5_USE_FILE_LOCKING=FALSE allows both handles
# to coexist safely for single-process benchmark use.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------------------------------------------------------------------------
# Constants for this run
# ---------------------------------------------------------------------------

ALL_TIER_LABELS = [TIER_LABELS[t] for t in TIERS]

PROFILE_NETWORK_PARAMS = {
    "default": {
        "algorithm": "k*nn",
        "mutual_edges_only": True,
        "obsm_key": "H_stacked",
    },
    "knn_ceiling": {
        "algorithm": "knn",
        "k": 100,
        "ef": 500,
        "ef_construction": 400,
        "mutual_edges_only": True,
        "obsm_key": "action_corrected",  # overridden for small_full
    },
}

# ---------------------------------------------------------------------------
# Core workflow runner (runs inside child process)
# ---------------------------------------------------------------------------

def _get_n_batches(adata: ad.AnnData, batch_key: Optional[str]) -> int:
    if batch_key and batch_key in adata.obs:
        return int(pd.Series(adata.obs[batch_key]).nunique())
    return 0


def _get_nnz(adata: ad.AnnData) -> int:
    try:
        if scipy.sparse.issparse(adata.X):
            return int(adata.X.nnz)
        if hasattr(adata.X, "shape"):
            return int(np.prod(adata.X.shape))
    except Exception:
        pass
    return 0


def _make_result(
    case_id: str,
    dataset: str,
    tier: Optional[int],
    profile: str,
    mode: str,
    stage: str,
    params: Dict[str, Any],
    adata: ad.AnnData,
    n_batches: int,
    execution_kind: str,
    wall_s: float,
    peak_rss_mb: float,
    io_read_mb: float,
    io_write_mb: float,
    status: str,
    failure_reason: Optional[str] = None,
) -> ResultRow:
    return ResultRow(
        case_id=case_id,
        dataset=dataset,
        tier=tier,
        profile=profile,
        mode=mode,
        stage=stage,
        params=params,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        nnz=_get_nnz(adata),
        n_batches=n_batches,
        representation_dim=N_COMPONENTS,
        execution_kind=execution_kind,
        wall_s=wall_s,
        peak_rss_mb=peak_rss_mb,
        io_read_mb=io_read_mb,
        io_write_mb=io_write_mb,
        status=status,
        failure_reason=failure_reason,
    )


def _compute_colors_with_3d(adata: ad.AnnData, n_threads: int) -> None:
    """Generate a 3D UMAP solely for color computation, then compute colors."""
    import actionet as an
    an.layout_network(adata, n_components=3, key_added="_umap_3d_colors",
                      n_threads=n_threads, inplace=True)
    an.compute_node_colors(adata, embedding_key="_umap_3d_colors")


def run_workflow_in_process(
    dataset_handle: str,
    profile: str,
    mode: str,
    trial: int,
    output_dir: Path,
    n_threads: int = N_THREADS,
    n_components: int = N_COMPONENTS,
    network_params_override: Optional[Dict[str, Any]] = None,
    sweep_label: str = "",
) -> None:
    """Full workflow runner — intended to run inside a child subprocess."""
    import actionet as an

    case_id = make_case_id(dataset_handle, profile, mode, trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    batch_key = cfg["batch_key"]
    label_key = cfg["label_key"]
    features_key = cfg["features_key"]
    use_batch = cfg["batch_correction"]
    tier = cfg["tier"]

    src_path = dataset_path(dataset_handle)
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset not found: {src_path}")

    # Determine backed copy path — use output_dir/work/ so it stays in the workspace
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine backed copy path (on /data so we have space)
    if mode == "backed_decompressed":
        backed_copy_path = work_dir / f"_work_{case_id}.h5ad"
        print(f"  [{case_id}] Copying to backed store: {backed_copy_path}", flush=True)
        import shutil
        shutil.copy2(str(src_path), str(backed_copy_path))
        adata = ad.read_h5ad(str(backed_copy_path), backed="r+")
        from actionet._backed_compression import (
            get_storage_metadata_from_adata, is_compressed_storage,
        )
        if is_compressed_storage(get_storage_metadata_from_adata(adata)):
            print(f"  [{case_id}] Decompressing backed storage ...", flush=True)
            an.decompress_backed_storage(adata, scope="file", chunk_size=4096, verbose=False)
        else:
            print(f"  [{case_id}] Backed storage already uncompressed — skipping decompress.", flush=True)
        execution_kind = "backed_streamed"
    elif mode == "backed_compressed":
        backed_copy_path = work_dir / f"_work_{case_id}.h5ad"
        import shutil
        shutil.copy2(str(src_path), str(backed_copy_path))
        adata = ad.read_h5ad(str(backed_copy_path), backed="r+")
        execution_kind = "backed_streamed"
    else:
        backed_copy_path = None
        adata = ad.read_h5ad(str(src_path))
        execution_kind = "in_memory"

    n_batches = _get_n_batches(adata, batch_key)
    bcs = 4096

    def save_row(stage, wall_s, peak_rss_mb, io_r, io_w, status, failure_reason=None, extra_params=None):
        p = dict(extra_params or {})
        row = _make_result(
            case_id=case_id,
            dataset=dataset_handle,
            tier=tier,
            profile=profile,
            mode=mode,
            stage=stage,
            params=p,
            adata=adata,
            n_batches=n_batches,
            execution_kind=execution_kind,
            wall_s=wall_s,
            peak_rss_mb=peak_rss_mb,
            io_read_mb=io_r,
            io_write_mb=io_w,
            status=status,
            failure_reason=failure_reason,
        )
        append_result(jsonl_path, row)

    def run_stage(stage_name, fn, stage_params=None):
        io_r0, io_w0 = bs._io_counters_mb()
        with StageProfiler(stage_name) as p:
            result = fn()
        io_r1, io_w1 = bs._io_counters_mb()
        save_row(
            stage_name, p.elapsed, p.peak_rss_mb,
            max(0.0, io_r1 - io_r0), max(0.0, io_w1 - io_w0),
            "ok", extra_params=stage_params,
        )
        print(
            f"  [{case_id}] {stage_name}: {p.elapsed:.2f}s  peak_rss={p.peak_rss_mb:.0f}MB",
            flush=True,
        )
        return result

    total_t0 = time.perf_counter()
    total_rss0 = psutil.Process().memory_info().rss / 1e6
    total_io_r0, total_io_w0 = bs._io_counters_mb()

    try:
        # Stage 1: filter
        run_stage("filter", lambda: an.filter_anndata(
            adata, min_cells_per_feat=0.01, backed_chunk_size=bcs, inplace=True,
        ), {"min_cells_per_feat": 0.01})

        # Stage 2: normalize
        run_stage("normalize", lambda: an.normalize_anndata(
            adata, target_sum=1e4, log_transform=True, log_base=2,
            backed_chunk_size=bcs, inplace=True,
        ), {"target_sum": 1e4, "log_base": 2})

        # For backed mode: close and reopen the file after normalize writes to disk
        # to release HDF5 locks before the C++ backed operator in reduce_kernel opens it
        if mode != "in_memory" and backed_copy_path is not None:
            try:
                if hasattr(adata, "file") and adata.file is not None:
                    adata.file.close()
            except Exception:
                pass
            del adata
            gc.collect()
            import time as _t; _t.sleep(0.5)  # Allow HDF5 lock release
            adata = ad.read_h5ad(str(backed_copy_path), backed="r+")

        # Stage 3: reduce_kernel
        # For backed mode: the C++ backed operator opens the file independently,
        # so we must close adata.file before reduce_kernel and reopen after.
        def _run_reduce_kernel():
            nonlocal adata
            if mode != "in_memory" and backed_copy_path is not None:
                try:
                    if hasattr(adata, "file") and adata.file is not None:
                        adata.file.close()
                except Exception:
                    pass
            an.reduce_kernel(
                adata, n_components=n_components, key_added="action",
                svd_algorithm=SVD_ALGORITHM, backed_chunk_size=bcs,
                verbose=False, inplace=True,
            )
            if mode != "in_memory" and backed_copy_path is not None:
                adata = ad.read_h5ad(str(backed_copy_path), backed="r+")

        run_stage("reduce_kernel", _run_reduce_kernel,
                  {"svd_algorithm": SVD_ALGORITHM, "n_components": n_components})

        # Stage 4: batch_correction (skipped for small_full)
        effective_reduction = "action"
        if use_batch:
            run_stage("batch_correction", lambda: an.correct_batch_effect(
                adata, batch_key=batch_key, reduction_key="action",
                backed_chunk_size=bcs, inplace=True,
            ), {"batch_key": batch_key})
            effective_reduction = "action_corrected"

        # Stage 5: action_decomposition (always in-memory — operates on obsm)
        run_stage("action_decomposition", lambda: an.run_action(
            adata, k_min=2, k_max=30, reduction_key=effective_reduction,
            n_threads=n_threads, inplace=True,
        ), {"k_min": 2, "k_max": 30})

        # Stage 6: network_construction
        net_params = dict(
            network_params_override if network_params_override is not None
            else PROFILE_NETWORK_PARAMS.get(profile, PROFILE_NETWORK_PARAMS["default"])
        )
        # For small_full (no batch correction) with knn_ceiling, use 'action' obsm
        if not use_batch and net_params.get("algorithm") == "knn":
            net_params["obsm_key"] = "H_stacked"

        run_stage("network_construction", lambda: an.build_network(
            adata, n_threads=n_threads, inplace=True, **net_params,
        ), net_params)

        # Stage 7: archetype_diffusion
        run_stage("archetype_diffusion", lambda: an.compute_network_diffusion(
            adata, scores="H_merged", key_added="archetype_footprint",
            n_threads=n_threads, inplace=True,
        ), {"scores": "H_merged"})

        # Stage 8: layout_2d
        run_stage("layout_2d", lambda: an.layout_network(
            adata, n_components=2, key_added="umap_2d_actionet",
            n_threads=n_threads, inplace=True,
        ), {"n_components": 2})

        # Stage 9: color_computation — requires 3D embedding; generate one internally
        run_stage("color_computation", lambda: _compute_colors_with_3d(adata, n_threads),
                  {"embedding_key": "umap_2d_actionet"})

        # Stage 10: feature_specificity
        run_stage("feature_specificity", lambda: an.compute_archetype_feature_specificity(
            adata, archetype_key="archetype_footprint",
            n_threads=n_threads, backed_chunk_size=bcs, inplace=True,
        ), {"archetype_key": "archetype_footprint"})

        # Check label guard for legacy stages
        run_legacy = True
        skip_reason = None
        if label_key in adata.obs:
            label_counts = adata.obs[label_key].value_counts()
            valid_labels = label_counts[label_counts >= MIN_LABEL_CELLS].index.tolist()
            if len(valid_labels) < MIN_LABELS_FOR_ANNOTATION:
                run_legacy = False
                skip_reason = f"fewer than {MIN_LABELS_FOR_ANNOTATION} labels with >={MIN_LABEL_CELLS} cells"

        # Stage 11: marker_detection
        markers_df = None
        if run_legacy:
            markers_df = run_stage("marker_detection", lambda: an.find_markers(
                adata, labels=label_key, features_use=features_key,
                top_genes=30, return_type="dataframe",
                n_threads=n_threads, backed_chunk_size=bcs,
            ), {"top_genes": 30, "features_use": features_key})
        else:
            save_row("marker_detection", 0.0, 0.0, 0.0, 0.0, "skipped", skip_reason)
            print(f"  [{case_id}] marker_detection: skipped ({skip_reason})", flush=True)

        # Stage 12: annotation
        if run_legacy and markers_df is not None:
            run_stage("annotation", lambda: an.annotate_cells(
                adata, markers_df, method="vision",
                features_use=features_key,
                n_threads=n_threads, backed_chunk_size=bcs,
            ), {"method": "vision", "features_use": features_key})
        else:
            save_row("annotation", 0.0, 0.0, 0.0, 0.0, "skipped", skip_reason)

        # Stage 13: imputation
        if run_legacy:
            all_genes = adata.var[features_key].dropna().astype(str).unique()
            rng = np.random.default_rng(42)
            impute_genes = list(rng.choice(all_genes, size=min(10, len(all_genes)), replace=False))
            run_stage("imputation", lambda: an.impute_features(
                adata, features=impute_genes, features_use=features_key,
                reduction_key=effective_reduction,
                n_threads=n_threads, backed_chunk_size=bcs,
            ), {"n_features": len(impute_genes), "features_use": features_key})
        else:
            save_row("imputation", 0.0, 0.0, 0.0, 0.0, "skipped", skip_reason)

    except Exception as exc:
        total_wall = time.perf_counter() - total_t0
        total_peak = max(0.0, psutil.Process().memory_info().rss / 1e6 - total_rss0)
        io_r1, io_w1 = bs._io_counters_mb()
        save_row("total", total_wall, total_peak,
                 max(0.0, io_r1 - total_io_r0), max(0.0, io_w1 - total_io_w0),
                 "stage_failed", str(exc))
        raise
    finally:
        # Capture adata attributes before deletion so total row can use them
        try:
            _n_obs_final = int(adata.n_obs)
            _n_vars_final = int(adata.n_vars)
        except Exception:
            _n_obs_final = 0
            _n_vars_final = 0
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        if backed_copy_path and backed_copy_path.exists():
            try:
                backed_copy_path.unlink()
            except Exception:
                pass

    total_wall = time.perf_counter() - total_t0
    total_peak = max(0.0, psutil.Process().memory_info().rss / 1e6 - total_rss0)
    io_r1, io_w1 = bs._io_counters_mb()
    # Write total row directly (adata is already deleted)
    row = ResultRow(
        case_id=case_id, dataset=dataset_handle, tier=tier,
        profile=profile, mode=mode, stage="total",
        params={}, n_obs=_n_obs_final, n_vars=_n_vars_final,
        nnz=0, n_batches=n_batches, representation_dim=n_components,
        execution_kind=execution_kind,
        wall_s=total_wall, peak_rss_mb=total_peak,
        io_read_mb=max(0.0, io_r1 - total_io_r0),
        io_write_mb=max(0.0, io_w1 - total_io_w0),
        status="ok", failure_reason=None,
    )
    append_result(jsonl_path, row)
    print(f"  [{case_id}] TOTAL: {total_wall:.2f}s  peak_rss_delta={total_peak:.0f}MB", flush=True)


# ---------------------------------------------------------------------------
# Child-process dispatch helpers
# ---------------------------------------------------------------------------

def _child_main(fn_name: str, kwargs_json: str) -> None:
    """Entry point for child subprocesses dispatched via subprocess.run."""
    import importlib
    import json as _json
    kwargs = _json.loads(kwargs_json)
    # Convert Path values
    for k, v in kwargs.items():
        if isinstance(v, str) and v.startswith("/") and ("benchmark" in v or "data" in v):
            pass  # leave as str; callee will convert
    if fn_name == "run_workflow_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_workflow_in_process(**kwargs)
    elif fn_name == "run_network_sweep_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_network_sweep_in_process(**kwargs)
    elif fn_name == "run_thread_sweep_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_thread_sweep_in_process(**kwargs)
    elif fn_name == "run_ef_sweep_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_ef_sweep_in_process(**kwargs)
    elif fn_name == "run_reduction_sweep_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_reduction_sweep_in_process(**kwargs)
    elif fn_name == "run_batch_sweep_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_batch_sweep_in_process(**kwargs)
    elif fn_name == "run_kmax_sweep_in_process":
        kwargs["output_dir"] = Path(kwargs["output_dir"])
        run_kmax_sweep_in_process(**kwargs)
    else:
        raise ValueError(f"Unknown fn_name: {fn_name}")


def dispatch_child(
    fn_name: str,
    kwargs: Dict[str, Any],
    timeout_s: float = TIMEOUT_SECONDS,
    rss_limit_gb: float = RSS_KILL_GB,
) -> Tuple[str, Optional[str]]:
    """Launch fn_name(**kwargs) in a fresh subprocess. Returns (status, failure_reason)."""
    kwargs_serial = {}
    for k, v in kwargs.items():
        kwargs_serial[k] = str(v) if isinstance(v, Path) else v

    kwargs_json = json.dumps(kwargs_serial)
    cmd = [
        PYTHON_EXE, "-c",
        f"import sys; sys.path.insert(0,'{Path(__file__).resolve().parent}'); "
        f"from benchmark_backed_extension import _child_main; "
        f"_child_main({repr(fn_name)}, {repr(kwargs_json)})"
    ]

    import subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rss_limit_bytes = rss_limit_gb * 1e9
    deadline = time.monotonic() + timeout_s

    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        ps_proc = None

    stdout_lines = []
    while True:
        # Non-blocking line read
        try:
            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
                stdout_lines.append(line)
        except Exception:
            pass

        if proc.poll() is not None:
            # Drain remaining output
            try:
                rest = proc.stdout.read()
                if rest:
                    print(rest, end="", flush=True)
            except Exception:
                pass
            break

        # Check timeout
        if time.monotonic() > deadline:
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait(timeout=5)
            return "timeout", f"wall time exceeded {timeout_s/3600:.1f}h limit"

        # Check RSS
        if ps_proc is not None:
            try:
                rss = ps_proc.memory_info().rss
                if rss > rss_limit_bytes:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc.wait(timeout=5)
                    return "oom", f"RSS {rss/1e9:.1f} GB exceeded {rss_limit_gb} GB limit"
            except psutil.NoSuchProcess:
                ps_proc = None

        time.sleep(0.5)

    rc = proc.returncode
    if rc == 0:
        return "ok", None
    return "stage_failed", f"child exited with code {rc}"


# ---------------------------------------------------------------------------
# Focused sweep runners (run inside child processes)
# ---------------------------------------------------------------------------

def run_network_sweep_in_process(
    dataset_handle: str,
    algorithm: str,
    obsm_key: str,
    k: int,
    ef: int,
    output_dir: Path,
    trial: int = 1,
    n_threads: int = N_THREADS,
) -> None:
    """Run only reduce_kernel + build_network on a dataset for the size frontier sweep."""
    import actionet as an

    sweep_label = f"net_{algorithm}_{obsm_key}"
    case_id = make_case_id(dataset_handle, "network_sweep", "in_memory", trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    src_path = dataset_path(dataset_handle)
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))

    adata = ad.read_h5ad(str(src_path))
    n_batches = bs._get_n_batches(adata, cfg["batch_key"])
    use_batch = cfg["batch_correction"]

    def save_row(stage, wall_s, peak_rss_mb, status, failure_reason=None, params=None):
        row = ResultRow(
            case_id=case_id, dataset=dataset_handle, tier=cfg["tier"],
            profile="network_sweep", mode="in_memory", stage=stage,
            params=params or {}, n_obs=adata.n_obs, n_vars=adata.n_vars,
            nnz=_get_nnz(adata), n_batches=n_batches, representation_dim=N_COMPONENTS,
            execution_kind="in_memory", wall_s=wall_s, peak_rss_mb=peak_rss_mb,
            io_read_mb=0.0, io_write_mb=0.0, status=status, failure_reason=failure_reason,
        )
        append_result(jsonl_path, row)

    try:
        an.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, inplace=True)

        with StageProfiler("reduce_kernel") as p:
            an.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action",
                             svd_algorithm=SVD_ALGORITHM, verbose=False, inplace=True)
        save_row("reduce_kernel", p.elapsed, p.peak_rss_mb, "ok", params={"n_components": N_COMPONENTS})

        if use_batch and cfg["batch_key"]:
            an.correct_batch_effect(adata, batch_key=cfg["batch_key"],
                                    reduction_key="action", inplace=True)

        an.run_action(adata, k_min=2, k_max=30,
                      reduction_key="action_corrected" if use_batch else "action",
                      n_threads=n_threads, inplace=True)

        net_params: Dict[str, Any] = {
            "algorithm": algorithm,
            "obsm_key": obsm_key,
            "mutual_edges_only": True,
            "n_threads": n_threads,
        }
        if algorithm == "knn":
            net_params["k"] = k
            net_params["ef"] = ef
            net_params["ef_construction"] = 400

        with StageProfiler("network_construction") as p:
            an.build_network(adata, inplace=True, **net_params)
        save_row("network_construction", p.elapsed, p.peak_rss_mb, "ok", params=net_params)

    finally:
        del adata
        gc.collect()


def run_thread_sweep_in_process(
    dataset_handle: str,
    n_threads: int,
    output_dir: Path,
    trial: int = 1,
) -> None:
    """Thread-count sweep: run_action, build_network (k*nn + knn), layout_network."""
    import actionet as an

    sweep_label = f"threads_{n_threads}"
    case_id = make_case_id(dataset_handle, "thread_sweep", "in_memory", trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    src_path = dataset_path(dataset_handle)
    adata = ad.read_h5ad(str(src_path))
    use_batch = cfg["batch_correction"]

    def save(stage, wall_s, rss, status="ok", params=None):
        row = ResultRow(
            case_id=case_id, dataset=dataset_handle, tier=cfg["tier"],
            profile="thread_sweep", mode="in_memory", stage=stage,
            params=params or {}, n_obs=adata.n_obs, n_vars=adata.n_vars,
            nnz=_get_nnz(adata), n_batches=bs._get_n_batches(adata, cfg["batch_key"]),
            representation_dim=N_COMPONENTS, execution_kind="in_memory",
            wall_s=wall_s, peak_rss_mb=rss, io_read_mb=0.0, io_write_mb=0.0,
            status=status, failure_reason=None,
        )
        append_result(jsonl_path, row)

    try:
        an.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, inplace=True)
        an.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action",
                         svd_algorithm=SVD_ALGORITHM, verbose=False, inplace=True)
        if use_batch and cfg["batch_key"]:
            an.correct_batch_effect(adata, batch_key=cfg["batch_key"],
                                    reduction_key="action", inplace=True)
        eff_key = "action_corrected" if use_batch else "action"

        with StageProfiler() as p:
            an.run_action(adata, k_min=2, k_max=30, reduction_key=eff_key,
                          n_threads=n_threads, inplace=True)
        save("run_action", p.elapsed, p.peak_rss_mb, params={"n_threads": n_threads})

        with StageProfiler() as p:
            an.build_network(adata, algorithm="k*nn", obsm_key="H_stacked",
                             mutual_edges_only=True, n_threads=n_threads, inplace=True)
        save("build_network_kstarnn", p.elapsed, p.peak_rss_mb,
             params={"algorithm": "k*nn", "n_threads": n_threads})

        with StageProfiler() as p:
            an.build_network(adata, algorithm="knn", k=100, ef=500,
                             ef_construction=400, obsm_key="H_stacked",
                             mutual_edges_only=True, n_threads=n_threads, inplace=True)
        save("build_network_knn", p.elapsed, p.peak_rss_mb,
             params={"algorithm": "knn", "k": 100, "n_threads": n_threads})

        an.compute_network_diffusion(adata, scores="H_merged",
                                     key_added="archetype_footprint", inplace=True)

        with StageProfiler() as p:
            an.layout_network(adata, n_components=2, key_added="umap_2d_actionet",
                              n_threads=n_threads, inplace=True)
        save("layout_2d", p.elapsed, p.peak_rss_mb, params={"n_threads": n_threads})

    finally:
        del adata
        gc.collect()


def run_ef_sweep_in_process(
    dataset_handle: str,
    ef: int,
    output_dir: Path,
    trial: int = 1,
    n_threads: int = N_THREADS,
) -> None:
    """ef sweep for knn at k=100 on 100k dataset."""
    import actionet as an

    sweep_label = f"ef_{ef}"
    case_id = make_case_id(dataset_handle, "ef_sweep", "in_memory", trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    src_path = dataset_path(dataset_handle)
    adata = ad.read_h5ad(str(src_path))
    use_batch = cfg["batch_correction"]

    def save(stage, wall_s, rss, params=None):
        row = ResultRow(
            case_id=case_id, dataset=dataset_handle, tier=cfg["tier"],
            profile="ef_sweep", mode="in_memory", stage=stage,
            params=params or {}, n_obs=adata.n_obs, n_vars=adata.n_vars,
            nnz=_get_nnz(adata), n_batches=bs._get_n_batches(adata, cfg["batch_key"]),
            representation_dim=N_COMPONENTS, execution_kind="in_memory",
            wall_s=wall_s, peak_rss_mb=rss, io_read_mb=0.0, io_write_mb=0.0,
            status="ok", failure_reason=None,
        )
        append_result(jsonl_path, row)

    try:
        an.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, inplace=True)
        an.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action",
                         svd_algorithm=SVD_ALGORITHM, verbose=False, inplace=True)
        if use_batch and cfg["batch_key"]:
            an.correct_batch_effect(adata, batch_key=cfg["batch_key"],
                                    reduction_key="action", inplace=True)
        eff_key = "action_corrected" if use_batch else "action"
        an.run_action(adata, k_min=2, k_max=30, reduction_key=eff_key,
                      n_threads=n_threads, inplace=True)

        net_params = {"algorithm": "knn", "k": 100, "ef": ef,
                      "ef_construction": 400, "obsm_key": "H_stacked",
                      "mutual_edges_only": True, "n_threads": n_threads}
        with StageProfiler() as p:
            an.build_network(adata, inplace=True, **net_params)
        save("network_construction", p.elapsed, p.peak_rss_mb, params=net_params)

    finally:
        del adata
        gc.collect()


def run_reduction_sweep_in_process(
    dataset_handle: str,
    mode: str,
    chunk_size: int,
    output_dir: Path,
    trial: int = 1,
) -> None:
    """reduce_kernel storage/chunk-size sweep."""
    import actionet as an

    sweep_label = f"red_{mode}_cs{chunk_size}"
    case_id = make_case_id(dataset_handle, "reduction_sweep", mode, trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    src_path = dataset_path(dataset_handle)

    backed_copy_path = None
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    if mode == "backed_decompressed":
        import shutil
        backed_copy_path = work_dir / f"_red_{case_id}.h5ad"
        shutil.copy2(str(src_path), str(backed_copy_path))
        adata = ad.read_h5ad(str(backed_copy_path), backed="r+")
        from actionet._backed_compression import (
            get_storage_metadata_from_adata, is_compressed_storage,
        )
        if is_compressed_storage(get_storage_metadata_from_adata(adata)):
            an.decompress_backed_storage(adata, scope="file", chunk_size=chunk_size, verbose=False)
        execution_kind = "backed_streamed"
    elif mode == "backed_compressed":
        import shutil
        backed_copy_path = work_dir / f"_red_{case_id}.h5ad"
        shutil.copy2(str(src_path), str(backed_copy_path))
        adata = ad.read_h5ad(str(backed_copy_path), backed="r+")
        execution_kind = "backed_streamed"
    else:
        adata = ad.read_h5ad(str(src_path))
        execution_kind = "in_memory"

    def save(stage, wall_s, rss, status="ok", params=None):
        row = ResultRow(
            case_id=case_id, dataset=dataset_handle, tier=cfg["tier"],
            profile="reduction_sweep", mode=mode, stage=stage,
            params=params or {}, n_obs=adata.n_obs, n_vars=adata.n_vars,
            nnz=_get_nnz(adata), n_batches=bs._get_n_batches(adata, cfg["batch_key"]),
            representation_dim=N_COMPONENTS, execution_kind=execution_kind,
            wall_s=wall_s, peak_rss_mb=rss, io_read_mb=0.0, io_write_mb=0.0,
            status=status, failure_reason=None,
        )
        append_result(jsonl_path, row)

    try:
        an.filter_anndata(adata, min_cells_per_feat=0.01,
                          backed_chunk_size=chunk_size, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2,
                             backed_chunk_size=chunk_size, inplace=True)
        with StageProfiler() as p:
            an.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action",
                             svd_algorithm=SVD_ALGORITHM, backed_chunk_size=chunk_size,
                             verbose=False, inplace=True)
        save("reduce_kernel", p.elapsed, p.peak_rss_mb,
             params={"mode": mode, "chunk_size": chunk_size, "n_components": N_COMPONENTS})
    finally:
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        if backed_copy_path and backed_copy_path.exists():
            try:
                backed_copy_path.unlink()
            except Exception:
                pass


def run_batch_sweep_in_process(
    dataset_handle: str,
    n_batches_use: int,
    output_dir: Path,
    trial: int = 1,
    n_threads: int = N_THREADS,
) -> None:
    """Batch-count sweep for correct_batch_effect."""
    import actionet as an

    sweep_label = f"batches_{n_batches_use}"
    case_id = make_case_id(dataset_handle, "batch_sweep", "in_memory", trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    src_path = dataset_path(dataset_handle)
    adata_src = ad.read_h5ad(str(src_path))

    # Subset to n_batches_use batches
    batch_key = cfg["batch_key"]
    all_batches = list(adata_src.obs[batch_key].unique())
    if n_batches_use < len(all_batches):
        rng = np.random.default_rng(42)
        chosen = list(rng.choice(all_batches, size=n_batches_use, replace=False))
        adata = adata_src[adata_src.obs[batch_key].isin(chosen)].copy()
        del adata_src
    else:
        adata = adata_src

    def save(stage, wall_s, rss, status="ok", params=None):
        row = ResultRow(
            case_id=case_id, dataset=dataset_handle, tier=cfg["tier"],
            profile="batch_sweep", mode="in_memory", stage=stage,
            params=params or {}, n_obs=adata.n_obs, n_vars=adata.n_vars,
            nnz=_get_nnz(adata), n_batches=n_batches_use,
            representation_dim=N_COMPONENTS, execution_kind="in_memory",
            wall_s=wall_s, peak_rss_mb=rss, io_read_mb=0.0, io_write_mb=0.0,
            status=status, failure_reason=None,
        )
        append_result(jsonl_path, row)

    try:
        an.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, inplace=True)
        an.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action",
                         svd_algorithm=SVD_ALGORITHM, verbose=False, inplace=True)
        with StageProfiler() as p:
            an.correct_batch_effect(adata, batch_key=batch_key,
                                    reduction_key="action", inplace=True)
        save("batch_correction", p.elapsed, p.peak_rss_mb,
             params={"n_batches": n_batches_use, "batch_key": batch_key})
    finally:
        del adata
        gc.collect()


def run_kmax_sweep_in_process(
    dataset_handle: str,
    k_max: int,
    output_dir: Path,
    trial: int = 1,
    n_threads: int = N_THREADS,
) -> None:
    """k_max sweep for run_action."""
    import actionet as an

    sweep_label = f"kmax_{k_max}"
    case_id = make_case_id(dataset_handle, "kmax_sweep", "in_memory", trial, sweep_label)
    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_MANIFEST[dataset_handle]
    src_path = dataset_path(dataset_handle)
    adata = ad.read_h5ad(str(src_path))
    use_batch = cfg["batch_correction"]

    def save(stage, wall_s, rss, params=None):
        row = ResultRow(
            case_id=case_id, dataset=dataset_handle, tier=cfg["tier"],
            profile="kmax_sweep", mode="in_memory", stage=stage,
            params=params or {}, n_obs=adata.n_obs, n_vars=adata.n_vars,
            nnz=_get_nnz(adata), n_batches=bs._get_n_batches(adata, cfg["batch_key"]),
            representation_dim=N_COMPONENTS, execution_kind="in_memory",
            wall_s=wall_s, peak_rss_mb=rss, io_read_mb=0.0, io_write_mb=0.0,
            status="ok", failure_reason=None,
        )
        append_result(jsonl_path, row)

    try:
        an.filter_anndata(adata, min_cells_per_feat=0.01, inplace=True)
        an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, inplace=True)
        an.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action",
                         svd_algorithm=SVD_ALGORITHM, verbose=False, inplace=True)
        if use_batch and cfg["batch_key"]:
            an.correct_batch_effect(adata, batch_key=cfg["batch_key"],
                                    reduction_key="action", inplace=True)
        eff_key = "action_corrected" if use_batch else "action"
        with StageProfiler() as p:
            an.run_action(adata, k_min=2, k_max=k_max, reduction_key=eff_key,
                          n_threads=n_threads, inplace=True)
        save("action_decomposition", p.elapsed, p.peak_rss_mb,
             params={"k_min": 2, "k_max": k_max})
    finally:
        del adata
        gc.collect()


# ---------------------------------------------------------------------------
# Orchestrator: full-workflow suite
# ---------------------------------------------------------------------------

def run_workflow_suite(
    datasets: List[str],
    profiles: List[str],
    modes: List[str],
    output_dir: Path,
    resume: bool,
    max_tier: Optional[str],
    trials_override: Optional[int],
    n_threads: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    default_frontier_failed = False

    max_tier_n = None
    if max_tier:
        try:
            max_tier_n = int(max_tier.lower().replace("k", "000").replace("m", "000000"))
        except ValueError:
            pass

    for dataset_handle in datasets:
        cfg = DATASET_MANIFEST.get(dataset_handle)
        if cfg is None:
            print(f"  [suite] Unknown dataset: {dataset_handle}", flush=True)
            continue
        tier = cfg.get("tier")
        if max_tier_n is not None and tier is not None and tier > max_tier_n:
            print(f"  [suite] Skipping {dataset_handle} (tier {tier} > max {max_tier_n})", flush=True)
            continue

        src_path = dataset_path(dataset_handle)
        if not src_path.exists():
            print(f"  [suite] Dataset file not found, skipping: {src_path}", flush=True)
            continue

        n_obs = tier if tier else 0
        if n_obs == 0:
            try:
                tmp = ad.read_h5ad(str(src_path), backed="r")
                n_obs = tmp.n_obs
                if hasattr(tmp, "file") and tmp.file:
                    tmp.file.close()
                del tmp
            except Exception:
                n_obs = 1

        n_trials = trials_override if trials_override else trials_for_tier(n_obs)

        for profile in profiles:
            if profile == "default" and default_frontier_failed and tier is not None:
                print(f"  [suite] default frontier failed — marking {dataset_handle} infeasible", flush=True)
                continue

            for mode in modes:
                for trial in range(1, n_trials + 1):
                    case_id = make_case_id(dataset_handle, profile, mode, trial)
                    jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"

                    if resume and is_case_complete(jsonl_path, case_id):
                        print(f"  [suite] SKIP (already complete): {case_id}", flush=True)
                        continue

                    print(f"\n{'='*70}", flush=True)
                    print(f"  [suite] Running: {case_id}", flush=True)
                    print(f"{'='*70}", flush=True)

                    status, reason = dispatch_child(
                        "run_workflow_in_process",
                        {
                            "dataset_handle": dataset_handle,
                            "profile": profile,
                            "mode": mode,
                            "trial": trial,
                            "output_dir": str(output_dir),
                            "n_threads": n_threads,
                        },
                    )

                    if status != "ok":
                        print(f"  [suite] FAILED {case_id}: {status} — {reason}", flush=True)
                        try:
                            tmp = ad.read_h5ad(str(src_path), backed="r")
                            fail_row = ResultRow(
                                case_id=case_id, dataset=dataset_handle, tier=tier,
                                profile=profile, mode=mode, stage="total",
                                params={}, n_obs=tmp.n_obs, n_vars=tmp.n_vars,
                                nnz=0, n_batches=0, representation_dim=N_COMPONENTS,
                                execution_kind="in_memory" if mode == "in_memory" else "backed_streamed",
                                wall_s=0.0, peak_rss_mb=0.0,
                                io_read_mb=0.0, io_write_mb=0.0,
                                status=status, failure_reason=reason,
                            )
                            if hasattr(tmp, "file") and tmp.file:
                                tmp.file.close()
                            del tmp
                        except Exception:
                            fail_row = ResultRow(
                                case_id=case_id, dataset=dataset_handle, tier=tier,
                                profile=profile, mode=mode, stage="total",
                                params={}, n_obs=0, n_vars=0, nnz=0,
                                n_batches=0, representation_dim=N_COMPONENTS,
                                execution_kind="in_memory",
                                wall_s=0.0, peak_rss_mb=0.0,
                                io_read_mb=0.0, io_write_mb=0.0,
                                status=status, failure_reason=reason,
                            )
                        append_result(jsonl_path, fail_row)
                        if profile == "default" and tier is not None:
                            default_frontier_failed = True


# ---------------------------------------------------------------------------
# Orchestrator: focused sweeps
# ---------------------------------------------------------------------------

def run_network_size_sweep(output_dir: Path, resume: bool, n_threads: int) -> None:
    print("\n[sweep] network size frontier", flush=True)
    scale_handles = [f"scale_subset_{TIER_LABELS[t]}" for t in TIERS] + ["scale_full"]
    obsm_keys_per_batch = {
        True: ["H_stacked", "H_merged", "action_corrected"],
        False: ["H_stacked", "H_merged"],
    }
    for handle in scale_handles:
        if not dataset_path(handle).exists():
            print(f"  [sweep] skipping {handle} — file not found", flush=True)
            continue
        cfg = DATASET_MANIFEST[handle]
        use_batch = cfg["batch_correction"]
        for algorithm, k, ef in [("k*nn", 10, 200), ("knn", 100, 500)]:
            for obsm_key in obsm_keys_per_batch[use_batch]:
                case_id = make_case_id(handle, "network_sweep", "in_memory", 1,
                                       f"net_{algorithm}_{obsm_key}")
                jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
                if resume and is_case_complete(jsonl_path, case_id):
                    print(f"  [sweep] SKIP {case_id}", flush=True)
                    continue
                dispatch_child("run_network_sweep_in_process", {
                    "dataset_handle": handle, "algorithm": algorithm,
                    "obsm_key": obsm_key, "k": k, "ef": ef,
                    "output_dir": str(output_dir), "trial": 1, "n_threads": n_threads,
                })


def run_thread_sweep(output_dir: Path, resume: bool) -> None:
    print("\n[sweep] thread count", flush=True)
    handle = "scale_subset_100k"
    if not dataset_path(handle).exists():
        print(f"  [sweep] skipping — {handle} not found", flush=True)
        return
    for n_th in [1, 8, 16, 44]:
        case_id = make_case_id(handle, "thread_sweep", "in_memory", 1, f"threads_{n_th}")
        jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
        if resume and is_case_complete(jsonl_path, case_id):
            continue
        dispatch_child("run_thread_sweep_in_process", {
            "dataset_handle": handle, "n_threads": n_th,
            "output_dir": str(output_dir), "trial": 1,
        })


def run_ef_sweep(output_dir: Path, resume: bool, n_threads: int) -> None:
    print("\n[sweep] knn ef values", flush=True)
    handle = "scale_subset_100k"
    if not dataset_path(handle).exists():
        print(f"  [sweep] skipping — {handle} not found", flush=True)
        return
    for ef in [200, 350, 500, 750]:
        case_id = make_case_id(handle, "ef_sweep", "in_memory", 1, f"ef_{ef}")
        jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
        if resume and is_case_complete(jsonl_path, case_id):
            continue
        dispatch_child("run_ef_sweep_in_process", {
            "dataset_handle": handle, "ef": ef,
            "output_dir": str(output_dir), "trial": 1, "n_threads": n_threads,
        })


def run_reduction_storage_sweep(output_dir: Path, resume: bool) -> None:
    print("\n[sweep] reduce_kernel storage", flush=True)
    handles_for_chunk = ["sparse_medium", "scale_subset_100k"]
    all_handles = [f"scale_subset_{TIER_LABELS[t]}" for t in TIERS] + ["sparse_medium"]
    for handle in all_handles:
        if not dataset_path(handle).exists():
            continue
        for mode in ["in_memory", "backed_decompressed"]:
            case_id = make_case_id(handle, "reduction_sweep", mode, 1, f"red_{mode}_cs4096")
            jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
            if resume and is_case_complete(jsonl_path, case_id):
                continue
            dispatch_child("run_reduction_sweep_in_process", {
                "dataset_handle": handle, "mode": mode, "chunk_size": 4096,
                "output_dir": str(output_dir), "trial": 1,
            })
    for handle in handles_for_chunk:
        if not dataset_path(handle).exists():
            continue
        for mode in ["backed_compressed", "backed_decompressed"]:
            for chunk_size in [1024, 4096, 16384]:
                case_id = make_case_id(handle, "reduction_sweep", mode, 1,
                                       f"red_{mode}_cs{chunk_size}")
                jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
                if resume and is_case_complete(jsonl_path, case_id):
                    continue
                dispatch_child("run_reduction_sweep_in_process", {
                    "dataset_handle": handle, "mode": mode, "chunk_size": chunk_size,
                    "output_dir": str(output_dir), "trial": 1,
                })


def run_batch_count_sweep(output_dir: Path, resume: bool, n_threads: int) -> None:
    print("\n[sweep] batch count", flush=True)
    handles = ["sparse_medium", "scale_subset_100k", "scale_full"]
    for handle in handles:
        if not dataset_path(handle).exists():
            continue
        cfg = DATASET_MANIFEST[handle]
        try:
            tmp = ad.read_h5ad(str(dataset_path(handle)), backed="r")
            max_b = int(tmp.obs[cfg["batch_key"]].nunique())
            if hasattr(tmp, "file") and tmp.file:
                tmp.file.close()
            del tmp
        except Exception:
            max_b = 25
        for n_b in [1, 5, 10, 20, max_b]:
            if n_b > max_b:
                continue
            case_id = make_case_id(handle, "batch_sweep", "in_memory", 1, f"batches_{n_b}")
            jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
            if resume and is_case_complete(jsonl_path, case_id):
                continue
            dispatch_child("run_batch_sweep_in_process", {
                "dataset_handle": handle, "n_batches_use": n_b,
                "output_dir": str(output_dir), "trial": 1, "n_threads": n_threads,
            })


def run_kmax_sweep(output_dir: Path, resume: bool, n_threads: int) -> None:
    print("\n[sweep] k_max sweep", flush=True)
    handle = "scale_subset_50k"
    if not dataset_path(handle).exists():
        print(f"  [sweep] skipping — {handle} not found", flush=True)
        return
    for k_max in [15, 30, 50]:
        case_id = make_case_id(handle, "kmax_sweep", "in_memory", 1, f"kmax_{k_max}")
        jsonl_path = output_dir / "raw" / f"{case_id}.jsonl"
        if resume and is_case_complete(jsonl_path, case_id):
            continue
        dispatch_child("run_kmax_sweep_in_process", {
            "dataset_handle": handle, "k_max": k_max,
            "output_dir": str(output_dir), "trial": 1, "n_threads": n_threads,
        })


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke(output_dir: Path, n_threads: int) -> None:
    print("\n[smoke] Running smoke test on small_full ...", flush=True)
    for mode in ["in_memory", "backed_decompressed"]:
        case_id = make_case_id("small_full", "default", mode, 1)
        status, reason = dispatch_child(
            "run_workflow_in_process",
            {
                "dataset_handle": "small_full", "profile": "default",
                "mode": mode, "trial": 1, "output_dir": str(output_dir),
                "n_threads": n_threads,
            },
            timeout_s=1800,
        )
        print(f"  [smoke] {mode}: {status}  ({reason})", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACTIONet Scaling Benchmark Suite")
    parser.add_argument(
        "--suite", default="all",
        choices=["all", "workflow", "network", "reduction", "batch", "thread", "ef", "kmax", "smoke"],
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=(
            ["small_full", "sparse_medium"]
            + [f"scale_subset_{TIER_LABELS[t]}" for t in TIERS]
            + ["scale_full"]
        ),
    )
    parser.add_argument("--profiles", nargs="+", default=["default", "knn_ceiling"],
                        choices=["default", "knn_ceiling"])
    parser.add_argument("--modes", nargs="+", default=["backed_decompressed", "in_memory"],
                        choices=["in_memory", "backed_decompressed", "backed_compressed"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tier", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--threads", type=int, default=N_THREADS)
    parser.add_argument("--skip-subsets", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        output_dir = BENCHMARK_RESULTS_DIR / run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)

    print(f"ACTIONet Scaling Benchmark Suite", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Suite: {args.suite}  Threads: {args.threads}", flush=True)

    if args.report_only:
        generate_report(output_dir)
        return

    bs.ensure_sparse_medium()

    if not args.skip_subsets:
        max_tier_n = None
        if args.max_tier:
            try:
                max_tier_n = int(args.max_tier.lower().replace("k", "000").replace("m", "000000"))
            except ValueError:
                pass
        tiers_to_gen = [t for t in TIERS if max_tier_n is None or t <= max_tier_n]
        if any("scale" in d for d in args.datasets):
            print("\nGenerating scale subsets ...", flush=True)
            bs.generate_all_subsets(tiers=tiers_to_gen)

    if args.suite == "smoke":
        run_smoke(output_dir, args.threads)
    elif args.suite in ("all", "workflow"):
        run_workflow_suite(
            datasets=args.datasets, profiles=args.profiles, modes=args.modes,
            output_dir=output_dir, resume=args.resume, max_tier=args.max_tier,
            trials_override=args.trials, n_threads=args.threads,
        )

    if args.suite in ("all", "network"):
        run_network_size_sweep(output_dir, resume=args.resume, n_threads=args.threads)
    if args.suite in ("all", "thread"):
        run_thread_sweep(output_dir, resume=args.resume)
    if args.suite in ("all", "ef"):
        run_ef_sweep(output_dir, resume=args.resume, n_threads=args.threads)
    if args.suite in ("all", "reduction"):
        run_reduction_storage_sweep(output_dir, resume=args.resume)
    if args.suite in ("all", "batch"):
        run_batch_count_sweep(output_dir, resume=args.resume, n_threads=args.threads)
    if args.suite in ("all", "kmax"):
        run_kmax_sweep(output_dir, resume=args.resume, n_threads=args.threads)

    print("\nGenerating report ...", flush=True)
    generate_report(output_dir)
    print(f"\nDone. Results in: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
