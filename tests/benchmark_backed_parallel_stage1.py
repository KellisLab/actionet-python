#!/usr/bin/env python3
"""Stage-1 backed parallelization benchmark for run_svd/reduce_kernel.

Collects:
  - wall time
  - disk I/O (read/write MB)
  - peak RSS (absolute process RSS observed during each stage)
  - backed-vs-in-memory parity on a bounded subset

Outputs:
  tests/benchmark_results/<run_id>/raw/results.jsonl
  tests/benchmark_results/<run_id>/summary.csv
  tests/benchmark_results/<run_id>/summary.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

import actionet as an

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_support as bs  # noqa: E402


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _parse_threads(spec: str) -> list[int]:
    threads: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("Thread values must be >= 1")
        threads.append(value)
    if not threads:
        raise ValueError("At least one thread value is required")
    return sorted(set(threads))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_sanitize(row)) + "\n")


def _svd_sigma_metrics(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    obs = np.asarray(observed, dtype=float).reshape(-1)
    k = min(ref.size, obs.size)
    if k == 0:
        return float("nan"), float("nan")
    ref = np.sort(ref[:k])[::-1]
    obs = np.sort(obs[:k])[::-1]
    denom = np.linalg.norm(ref)
    rel = float(np.linalg.norm(obs - ref) / denom) if denom > 0 else float("nan")
    if k < 2 or np.std(ref) == 0 or np.std(obs) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ref, obs)[0, 1])
    return corr, rel


def _procrustes_rel_error(reference: np.ndarray, observed: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if ref.shape != obs.shape or ref.ndim != 2:
        return float("nan")
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    obs_centered = obs - obs.mean(axis=0, keepdims=True)
    cross = obs_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rot = u @ vt
    aligned = obs_centered @ rot
    denom = np.linalg.norm(ref_centered)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(aligned - ref_centered) / denom)


def _stage_row(
    *,
    run_id: str,
    dataset_handle: str,
    mode: str,
    stage: str,
    thread_count: int,
    trial: int,
    wall_s: float,
    peak_rss_mb: float,
    io_read_mb: float,
    io_write_mb: float,
    status: str,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "dataset_handle": dataset_handle,
        "mode": mode,
        "stage": stage,
        "thread_count": int(thread_count),
        "trial": int(trial),
        "wall_s": float(wall_s),
        "peak_rss_mb": float(peak_rss_mb),
        "io_read_mb": float(io_read_mb),
        "io_write_mb": float(io_write_mb),
        "status": status,
        "error": error,
    }
    if extra:
        row.update(extra)
    return row


def _metric_delta(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    return max(0.0, float(after) - float(before))


def _perf_run(
    *,
    run_id: str,
    dataset_handle: str,
    dataset_src: Path,
    work_dir: Path,
    thread_count: int,
    trial: int,
    n_components: int,
    chunk_size: int,
    jsonl_path: Path,
) -> None:
    case_copy = work_dir / f"{run_id}_{dataset_handle}_t{thread_count}_r{trial}.h5ad"
    shutil.copy2(str(dataset_src), str(case_copy))
    adata = ad.read_h5ad(str(case_copy), backed="r+")
    try:
        io_r0, io_w0 = bs._io_counters_mb()
        with bs.StageProfiler(label=f"run_svd[t={thread_count},trial={trial}]") as prof:
            an.run_svd(
                adata,
                n_components=n_components,
                algorithm="halko",
                seed=42 + trial,
                verbose=False,
                backed_chunk_size=chunk_size,
                backed_n_threads=thread_count,
            )
        io_r1, io_w1 = bs._io_counters_mb()
        _append_jsonl(
            jsonl_path,
            _stage_row(
                run_id=run_id,
                dataset_handle=dataset_handle,
                mode="performance",
                stage="run_svd",
                thread_count=thread_count,
                trial=trial,
                wall_s=prof.elapsed,
                peak_rss_mb=prof.peak_rss_abs_mb,
                io_read_mb=_metric_delta(io_r1, io_r0),
                io_write_mb=_metric_delta(io_w1, io_w0),
                status="ok",
            ),
        )

        io_r0, io_w0 = bs._io_counters_mb()
        with bs.StageProfiler(label=f"reduce_kernel[t={thread_count},trial={trial}]") as prof:
            an.reduce_kernel(
                adata,
                n_components=n_components,
                svd_algorithm="halko",
                seed=42 + trial,
                verbose=False,
                backed_chunk_size=chunk_size,
                backed_n_threads=thread_count,
                inplace=True,
            )
        io_r1, io_w1 = bs._io_counters_mb()
        _append_jsonl(
            jsonl_path,
            _stage_row(
                run_id=run_id,
                dataset_handle=dataset_handle,
                mode="performance",
                stage="reduce_kernel",
                thread_count=thread_count,
                trial=trial,
                wall_s=prof.elapsed,
                peak_rss_mb=prof.peak_rss_abs_mb,
                io_read_mb=_metric_delta(io_r1, io_r0),
                io_write_mb=_metric_delta(io_w1, io_w0),
                status="ok",
            ),
        )
    except Exception as exc:
        _append_jsonl(
            jsonl_path,
            _stage_row(
                run_id=run_id,
                dataset_handle=dataset_handle,
                mode="performance",
                stage="run_svd+reduce_kernel",
                thread_count=thread_count,
                trial=trial,
                wall_s=0.0,
                peak_rss_mb=0.0,
                io_read_mb=0.0,
                io_write_mb=0.0,
                status="error",
                error=str(exc),
            ),
        )
    finally:
        try:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        if case_copy.exists():
            case_copy.unlink()


def _prepare_parity_reference(
    dataset_src: Path,
    parity_max_obs: int,
    n_components: int,
) -> tuple[ad.AnnData, dict[str, Any], ad.AnnData]:
    adata_backed = ad.read_h5ad(str(dataset_src), backed="r")
    try:
        n_obs = int(adata_backed.n_obs)
        keep = min(int(parity_max_obs), n_obs)
        subset = adata_backed[:keep].to_memory()
    finally:
        try:
            if hasattr(adata_backed, "file") and adata_backed.file is not None:
                adata_backed.file.close()
        except Exception:
            pass
        del adata_backed

    ref_svd = an.run_svd(
        subset,
        n_components=n_components,
        algorithm="halko",
        verbose=False,
    )
    ref_reduce = subset.copy()
    an.reduce_kernel(
        ref_reduce,
        n_components=n_components,
        svd_algorithm="halko",
        verbose=False,
        inplace=True,
    )
    return subset, ref_svd, ref_reduce


def _parity_run(
    *,
    run_id: str,
    dataset_handle: str,
    thread_count: int,
    parity_template: ad.AnnData,
    ref_svd: dict[str, Any],
    ref_reduce: ad.AnnData,
    work_dir: Path,
    n_components: int,
    chunk_size: int,
    jsonl_path: Path,
) -> None:
    parity_file = work_dir / f"{run_id}_{dataset_handle}_parity_t{thread_count}.h5ad"
    parity_template.write_h5ad(parity_file)
    adata_backed = ad.read_h5ad(str(parity_file), backed="r+")
    try:
        io_r0, io_w0 = bs._io_counters_mb()
        with bs.StageProfiler(label=f"parity_run_svd[t={thread_count}]") as prof:
            backed_svd = an.run_svd(
                adata_backed,
                n_components=n_components,
                algorithm="halko",
                verbose=False,
                backed_chunk_size=chunk_size,
                backed_n_threads=thread_count,
            )
        io_r1, io_w1 = bs._io_counters_mb()
        svd_corr, svd_rel = _svd_sigma_metrics(ref_svd["d"], backed_svd["d"])
        _append_jsonl(
            jsonl_path,
            _stage_row(
                run_id=run_id,
                dataset_handle=dataset_handle,
                mode="parity",
                stage="run_svd",
                thread_count=thread_count,
                trial=0,
                wall_s=prof.elapsed,
                peak_rss_mb=prof.peak_rss_abs_mb,
                io_read_mb=_metric_delta(io_r1, io_r0),
                io_write_mb=_metric_delta(io_w1, io_w0),
                status="ok",
                extra={
                    "sigma_corr_vs_inmemory": svd_corr,
                    "sigma_rel_diff_vs_inmemory": svd_rel,
                },
            ),
        )

        io_r0, io_w0 = bs._io_counters_mb()
        with bs.StageProfiler(label=f"parity_reduce_kernel[t={thread_count}]") as prof:
            an.reduce_kernel(
                adata_backed,
                n_components=n_components,
                svd_algorithm="halko",
                verbose=False,
                backed_chunk_size=chunk_size,
                backed_n_threads=thread_count,
                inplace=True,
            )
        io_r1, io_w1 = bs._io_counters_mb()
        sigma_backed = np.asarray(adata_backed.uns["action_params"]["sigma"]).reshape(-1)
        sigma_ref = np.asarray(ref_reduce.uns["action_params"]["sigma"]).reshape(-1)
        _, sigma_rel = _svd_sigma_metrics(sigma_ref, sigma_backed)
        action_err = _procrustes_rel_error(ref_reduce.obsm["action"], adata_backed.obsm["action"])
        _append_jsonl(
            jsonl_path,
            _stage_row(
                run_id=run_id,
                dataset_handle=dataset_handle,
                mode="parity",
                stage="reduce_kernel",
                thread_count=thread_count,
                trial=0,
                wall_s=prof.elapsed,
                peak_rss_mb=prof.peak_rss_abs_mb,
                io_read_mb=_metric_delta(io_r1, io_r0),
                io_write_mb=_metric_delta(io_w1, io_w0),
                status="ok",
                extra={
                    "sigma_rel_diff_vs_inmemory": sigma_rel,
                    "procrustes_rel_error_vs_inmemory": action_err,
                },
            ),
        )
    except Exception as exc:
        _append_jsonl(
            jsonl_path,
            _stage_row(
                run_id=run_id,
                dataset_handle=dataset_handle,
                mode="parity",
                stage="run_svd+reduce_kernel",
                thread_count=thread_count,
                trial=0,
                wall_s=0.0,
                peak_rss_mb=0.0,
                io_read_mb=0.0,
                io_write_mb=0.0,
                status="error",
                error=str(exc),
            ),
        )
    finally:
        try:
            if hasattr(adata_backed, "file") and adata_backed.file is not None:
                adata_backed.file.close()
        except Exception:
            pass
        del adata_backed
        gc.collect()
        if parity_file.exists():
            parity_file.unlink()


def _summarize(df: pd.DataFrame, output_dir: Path, strict_gates: bool) -> int:
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"

    perf = df[(df["mode"] == "performance") & (df["status"] == "ok")].copy()
    parity = df[(df["mode"] == "parity") & (df["status"] == "ok")].copy()

    perf_summary = pd.DataFrame()
    if not perf.empty:
        perf_summary = (
            perf.groupby(["stage", "thread_count"])[["wall_s", "io_read_mb", "io_write_mb", "peak_rss_mb"]]
            .mean()
            .reset_index()
            .sort_values(["stage", "thread_count"])
        )

    parity_summary = pd.DataFrame()
    if not parity.empty:
        parity_metrics = [c for c in parity.columns if c.endswith("_vs_inmemory")]
        parity_summary = (
            parity.groupby(["stage", "thread_count"])[parity_metrics]
            .mean()
            .reset_index()
            .sort_values(["stage", "thread_count"])
        )

    combined = perf_summary.copy()
    if not parity_summary.empty:
        combined = combined.merge(parity_summary, on=["stage", "thread_count"], how="outer")
    combined.to_csv(summary_csv, index=False)

    failures: list[str] = []
    if not perf_summary.empty and (perf_summary["thread_count"] == 1).any():
        baseline = perf_summary[perf_summary["thread_count"] == 1].set_index("stage")
        for _, row in perf_summary[perf_summary["thread_count"] != 1].iterrows():
            stage = row["stage"]
            if stage not in baseline.index:
                continue
            base = baseline.loc[stage]
            for metric in ("wall_s", "io_read_mb", "io_write_mb", "peak_rss_mb"):
                b = float(base[metric])
                v = float(row[metric])
                if not np.isfinite(b) or not np.isfinite(v):
                    continue
                # Percentage-based regression checks are unstable when the
                # baseline is near zero. Skip tiny I/O/RSS baselines.
                if metric in ("io_read_mb", "io_write_mb", "peak_rss_mb") and b < 1.0:
                    continue
                if b > 0 and (v / b) > 1.10:
                    failures.append(
                        f"{stage} threads={int(row['thread_count'])}: {metric} regression {(v / b - 1.0) * 100:.1f}% (>10%)"
                    )

    if not parity_summary.empty:
        bad_sigma = parity_summary[
            (parity_summary.get("sigma_rel_diff_vs_inmemory", 0.0) > 1e-4)
        ]
        for _, row in bad_sigma.iterrows():
            failures.append(
                f"{row['stage']} threads={int(row['thread_count'])}: sigma_rel_diff_vs_inmemory={row['sigma_rel_diff_vs_inmemory']:.3e} (>1e-4)"
            )
        if "procrustes_rel_error_vs_inmemory" in parity_summary.columns:
            bad_action = parity_summary[parity_summary["procrustes_rel_error_vs_inmemory"] > 1e-3]
            for _, row in bad_action.iterrows():
                failures.append(
                    f"{row['stage']} threads={int(row['thread_count'])}: procrustes_rel_error_vs_inmemory={row['procrustes_rel_error_vs_inmemory']:.3e} (>1e-3)"
                )

    lines: list[str] = []
    lines.append("# Stage 1 Backed Parallelization Benchmark")
    lines.append("")
    lines.append("## Performance Means")
    lines.append("")
    if perf_summary.empty:
        lines.append("_No successful performance rows._")
    else:
        lines.append(perf_summary.to_markdown(index=False))
        io_missing = (
            perf_summary["io_read_mb"].isna().all()
            and perf_summary["io_write_mb"].isna().all()
        )
        if io_missing:
            lines.append("")
            lines.append("_I/O counters unavailable on this platform/environment (reported as NaN)._")
    lines.append("")
    lines.append("## Parity Means (vs in-memory subset)")
    lines.append("")
    if parity_summary.empty:
        lines.append("_No successful parity rows._")
    else:
        lines.append(parity_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    if failures:
        for failure in failures:
            lines.append(f"- FAIL: {failure}")
    else:
        lines.append("- PASS: all non-regression and parity gates met.")
    lines.append("")
    lines.append(f"Strict gates: {'enabled' if strict_gates else 'disabled'}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if strict_gates and failures:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1 backed parallelization benchmark")
    parser.add_argument("--dataset-handle", default="sparse_medium", help="Dataset handle from tests/benchmark_support.py")
    parser.add_argument("--threads", default="1,4,8,16", help="Comma-separated thread counts")
    parser.add_argument("--trials", type=int, default=2, help="Trials per thread count")
    parser.add_argument("--n-components", type=int, default=30, help="Number of SVD/reduction components")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Backed chunk size")
    parser.add_argument("--parity-max-obs", type=int, default=50_000, help="Max observations for parity subset")
    parser.add_argument("--output-dir", default=None, help="Output directory (default tests/benchmark_results/<run_id>)")
    parser.add_argument("--no-strict-gates", action="store_true", help="Do not fail non-regression/parity gate violations")
    args = parser.parse_args()

    dataset_src = bs.dataset_path(args.dataset_handle)
    if not dataset_src.exists():
        raise FileNotFoundError(f"Dataset not found for handle '{args.dataset_handle}': {dataset_src}")
    thread_counts = _parse_threads(args.threads)

    run_id = f"stage1_backed_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else (bs.BENCHMARK_RESULTS_DIR / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "raw" / "results.jsonl"

    print(f"[stage1] dataset={args.dataset_handle} path={dataset_src}")
    print(f"[stage1] threads={thread_counts} trials={args.trials} n_components={args.n_components}")
    print(f"[stage1] output_dir={output_dir}")

    for thread_count in thread_counts:
        for trial in range(1, args.trials + 1):
            print(f"[stage1] performance thread={thread_count} trial={trial}")
            _perf_run(
                run_id=run_id,
                dataset_handle=args.dataset_handle,
                dataset_src=dataset_src,
                work_dir=work_dir,
                thread_count=thread_count,
                trial=trial,
                n_components=args.n_components,
                chunk_size=args.chunk_size,
                jsonl_path=jsonl_path,
            )

    print("[stage1] preparing parity subset + in-memory references")
    parity_template, ref_svd, ref_reduce = _prepare_parity_reference(
        dataset_src=dataset_src,
        parity_max_obs=args.parity_max_obs,
        n_components=args.n_components,
    )

    for thread_count in thread_counts:
        print(f"[stage1] parity thread={thread_count}")
        _parity_run(
            run_id=run_id,
            dataset_handle=args.dataset_handle,
            thread_count=thread_count,
            parity_template=parity_template,
            ref_svd=ref_svd,
            ref_reduce=ref_reduce,
            work_dir=work_dir,
            n_components=args.n_components,
            chunk_size=args.chunk_size,
            jsonl_path=jsonl_path,
        )

    rows = bs.load_results(jsonl_path)
    df = pd.DataFrame(rows)
    exit_code = _summarize(df, output_dir=output_dir, strict_gates=not args.no_strict_gates)
    print(f"[stage1] summary={output_dir / 'summary.md'}")
    print(f"[stage1] csv={output_dir / 'summary.csv'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
