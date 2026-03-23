#!/usr/bin/env python3
"""Focused post-refactor benchmark for comparing two Actionet branches.

This driver is intentionally self-contained:
- the top-level orchestrator uses only the Python standard library
- dependency-heavy work runs inside branch-specific virtualenv workers
- every benchmark case executes in a fresh subprocess

Usage:
    python3 tests/benchmark_branch_compare.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


BASELINE_BRANCH = "dev-backed"
CANDIDATE_BRANCH = "feature/orientation-unification"

BACKED_MODE = "backed_decompressed"
INMEMORY_MODE = "in_memory"

DEFAULT_BACKED_CHUNK_SIZE = 4096
DEFAULT_CASE_TIMEOUT_S = 4 * 3600
DEFAULT_MAX_INMEMORY_RSS_GB = 30.0

PRIMARY_INMEMORY_TIER = "100k"
INMEMORY_FALLBACK_TIER = "50k"
BACKED_FALLBACK_TIER = "250k"

TIER_SIZES = {
    "50k": 50_000,
    "100k": 100_000,
    "200k": 200_000,
    "250k": 250_000,
}

STAGE_ORDER = [
    "reduce_kernel",
    "correct_batch_effect",
    "run_action",
    "build_network",
    "compute_network_diffusion",
    "layout_network_2d",
    "compute_archetype_feature_specificity",
    "total",
]

DATASET_ORDER = [
    "sparse_medium",
    "scale_subset_50k",
    "scale_subset_100k",
    "scale_subset_200k",
    "scale_subset_250k",
    "scale_full",
]

BENCHMARK_ENV_OVERRIDES = {
    "HDF5_USE_FILE_LOCKING": "FALSE",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass
class BranchRuntime:
    branch: str
    worktree: str
    python: str


@dataclass
class CaseSpec:
    branch: str
    dataset: str
    dataset_path: str
    tier: Optional[int]
    mode: str
    trial: int
    case_id: str
    output_jsonl: str
    work_dir: str
    timeout_s: int
    backed_chunk_size: int


@dataclass
class ResultRow:
    branch: str
    dataset: str
    tier: Optional[int]
    mode: str
    trial: int
    case_id: str
    stage: str
    wall_s: float
    peak_rss_mb: float
    io_read_mb: float
    io_write_mb: float
    status: str
    failure_reason: Optional[str]
    n_obs: int
    n_vars: int
    nnz: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def script_path() -> Path:
    return Path(__file__).resolve()


def log(message: str) -> None:
    print(message, flush=True)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def default_output_dir() -> Path:
    return repo_root() / "tests" / "benchmark_results" / f"branch_compare_{timestamp_id()}"


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    display_cwd = str(cwd) if cwd is not None else os.getcwd()
    log(f"$ (cd {display_cwd} && {' '.join(cmd)})")
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def stream_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> int:
    display_cwd = str(cwd) if cwd is not None else os.getcwd()
    log(f"$ (cd {display_cwd} && {' '.join(cmd)})")
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    assert proc.stdout is not None

    while True:
        line = proc.stdout.readline()
        if line:
            print(line, end="", flush=True)
        if proc.poll() is not None:
            remainder = proc.stdout.read()
            if remainder:
                print(remainder, end="", flush=True)
            return int(proc.returncode or 0)
        if deadline is not None and time.monotonic() > deadline:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return 124
        time.sleep(0.2)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_case_total_row(case_jsonl: Path) -> Optional[Dict[str, Any]]:
    rows = load_jsonl(case_jsonl)
    totals = [row for row in rows if row.get("stage") == "total"]
    return totals[-1] if totals else None


def base_python_executable() -> str:
    return sys.executable or "python3"


def worker_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(BENCHMARK_ENV_OVERRIDES)
    return env


def create_branch_runtime(branch: str, work_root: Path) -> BranchRuntime:
    repo = repo_root()
    branch_dir = work_root / "worktrees" / sanitize_name(branch)
    branch_dir.parent.mkdir(parents=True, exist_ok=True)

    run_command(["git", "worktree", "add", "--detach", str(branch_dir), branch], cwd=repo)
    run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=branch_dir)

    venv_dir = branch_dir / ".venv-bench"
    run_command([base_python_executable(), "-m", "venv", str(venv_dir)], cwd=branch_dir)
    python_path = venv_dir / "bin" / "python"

    env = worker_env()
    run_command([str(python_path), "-m", "pip", "install", ".", "psutil"], cwd=branch_dir, env=env)

    return BranchRuntime(branch=branch, worktree=str(branch_dir), python=str(python_path))


def run_internal(
    python_exe: str,
    subcommand: str,
    args: Sequence[str],
    *,
    timeout_s: Optional[int] = None,
) -> int:
    cmd = [python_exe, str(script_path()), subcommand, *args]
    return stream_subprocess(cmd, env=worker_env(), timeout_s=timeout_s)


def prepare_datasets(prepare_python: str, data_root: Path, manifest_path: Path) -> Dict[str, Any]:
    args = [
        "--repo-root", str(repo_root()),
        "--data-root", str(data_root),
        "--manifest-path", str(manifest_path),
    ]
    exit_code = run_internal(prepare_python, "prepare-data", args, timeout_s=DEFAULT_CASE_TIMEOUT_S)
    if exit_code != 0:
        raise RuntimeError(f"Dataset preparation failed with exit code {exit_code}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def smoke_check(branch_runtime: BranchRuntime, smoke_root: Path) -> None:
    args = [
        "--branch", branch_runtime.branch,
        "--work-dir", str(smoke_root / sanitize_name(branch_runtime.branch)),
    ]
    exit_code = run_internal(branch_runtime.python, "smoke-check", args, timeout_s=900)
    if exit_code != 0:
        raise RuntimeError(f"Smoke check failed for branch {branch_runtime.branch}")


def make_case_spec(
    branch_runtime: BranchRuntime,
    dataset: str,
    dataset_path: str,
    tier: Optional[int],
    mode: str,
    trial: int,
    raw_dir: Path,
    case_work_root: Path,
    backed_chunk_size: int,
    timeout_s: int,
) -> CaseSpec:
    case_id = f"{sanitize_name(branch_runtime.branch)}__{dataset}__{mode}__t{trial}"
    return CaseSpec(
        branch=branch_runtime.branch,
        dataset=dataset,
        dataset_path=dataset_path,
        tier=tier,
        mode=mode,
        trial=trial,
        case_id=case_id,
        output_jsonl=str(raw_dir / f"{case_id}.jsonl"),
        work_dir=str(case_work_root / case_id),
        timeout_s=timeout_s,
        backed_chunk_size=backed_chunk_size,
    )


def write_case_config(spec: CaseSpec, cases_dir: Path) -> Path:
    cases_dir.mkdir(parents=True, exist_ok=True)
    config_path = cases_dir / f"{spec.case_id}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2, sort_keys=True)
    return config_path


def append_total_failure_row(spec: CaseSpec, status: str, failure_reason: str) -> None:
    total_row = ResultRow(
        branch=spec.branch,
        dataset=spec.dataset,
        tier=spec.tier,
        mode=spec.mode,
        trial=spec.trial,
        case_id=spec.case_id,
        stage="total",
        wall_s=0.0,
        peak_rss_mb=0.0,
        io_read_mb=0.0,
        io_write_mb=0.0,
        status=status,
        failure_reason=failure_reason,
        n_obs=0,
        n_vars=0,
        nnz=0,
    )
    append_jsonl(Path(spec.output_jsonl), total_row.to_dict())


def execute_case(branch_runtime: BranchRuntime, spec: CaseSpec) -> Dict[str, Any]:
    config_path = write_case_config(spec, Path(spec.output_jsonl).parent.parent / "cases")
    exit_code = run_internal(
        branch_runtime.python,
        "run-case",
        ["--config", str(config_path)],
        timeout_s=spec.timeout_s,
    )

    case_jsonl = Path(spec.output_jsonl)
    total_row = load_case_total_row(case_jsonl)
    if exit_code == 124:
        append_total_failure_row(spec, "timeout", f"Case timed out after {spec.timeout_s}s")
        total_row = load_case_total_row(case_jsonl)
    elif exit_code != 0 and total_row is None:
        append_total_failure_row(spec, "failed", f"Case subprocess exited with code {exit_code}")
        total_row = load_case_total_row(case_jsonl)

    if total_row is None:
        raise RuntimeError(f"No total row produced for case {spec.case_id}")
    return total_row


def is_ok_under_rss_limit(row: Dict[str, Any], limit_gb: float) -> bool:
    if row.get("status") != "ok":
        return False
    peak_rss_mb = float(row.get("peak_rss_mb") or 0.0)
    return peak_rss_mb <= limit_gb * 1024.0


def is_host_limit_failure(row: Dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    reason = str(row.get("failure_reason") or "").lower()
    if status in {"timeout", "memory_limit", "rss_limit"}:
        return True
    if "memory" in reason or "killed" in reason or "timeout" in reason:
        return True
    return False


def load_all_rows(raw_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def trial_median(values: Iterable[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(statistics.median(clean))


def build_comparison_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("mode")),
            str(row.get("stage")),
            str(row.get("branch")),
        )
        grouped.setdefault(key, {}).setdefault("rows", []).append(row)

    comparisons: List[Dict[str, Any]] = []
    comparable_keys = set(
        (dataset, mode, stage)
        for dataset, mode, stage, _branch in grouped.keys()
    )

    for dataset, mode, stage in sorted(
        comparable_keys,
        key=lambda item: (
            DATASET_ORDER.index(item[0]) if item[0] in DATASET_ORDER else len(DATASET_ORDER),
            item[1],
            STAGE_ORDER.index(item[2]) if item[2] in STAGE_ORDER else len(STAGE_ORDER),
        ),
    ):
        baseline_rows = grouped.get((dataset, mode, stage, BASELINE_BRANCH), {}).get("rows", [])
        candidate_rows = grouped.get((dataset, mode, stage, CANDIDATE_BRANCH), {}).get("rows", [])
        if not baseline_rows or not candidate_rows:
            continue

        baseline_wall = trial_median(numeric(row.get("wall_s")) for row in baseline_rows)
        candidate_wall = trial_median(numeric(row.get("wall_s")) for row in candidate_rows)
        baseline_rss = trial_median(numeric(row.get("peak_rss_mb")) for row in baseline_rows)
        candidate_rss = trial_median(numeric(row.get("peak_rss_mb")) for row in candidate_rows)
        baseline_tier = baseline_rows[0].get("tier")
        candidate_tier = candidate_rows[0].get("tier")
        tier = baseline_tier if baseline_tier is not None else candidate_tier

        if baseline_wall is None or candidate_wall is None or baseline_rss is None or candidate_rss is None:
            continue

        wall_delta_pct = (
            ((candidate_wall - baseline_wall) / baseline_wall) * 100.0
            if baseline_wall > 0 else None
        )
        rss_delta_pct = (
            ((candidate_rss - baseline_rss) / baseline_rss) * 100.0
            if baseline_rss > 0 else None
        )

        repeated = len(baseline_rows) > 1 and len(candidate_rows) > 1
        wall_threshold = 20.0 if repeated else 25.0
        rss_threshold = 15.0 if repeated else 20.0
        obvious_regression = (
            (wall_delta_pct is not None and wall_delta_pct > wall_threshold) or
            (rss_delta_pct is not None and rss_delta_pct > rss_threshold)
        )

        comparisons.append(
            {
                "dataset": dataset,
                "tier": tier,
                "mode": mode,
                "stage": stage,
                "baseline_trials": len(baseline_rows),
                "candidate_trials": len(candidate_rows),
                "baseline_wall_s": baseline_wall,
                "candidate_wall_s": candidate_wall,
                "baseline_peak_rss_mb": baseline_rss,
                "candidate_peak_rss_mb": candidate_rss,
                "wall_delta_pct": wall_delta_pct,
                "rss_delta_pct": rss_delta_pct,
                "obvious_regression": obvious_regression,
            }
        )

    return comparisons


def failure_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = [row for row in rows if row.get("status") != "ok"]
    failures.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]) if row.get("dataset") in DATASET_ORDER else len(DATASET_ORDER),
            str(row.get("branch")),
            str(row.get("mode")),
            STAGE_ORDER.index(row["stage"]) if row.get("stage") in STAGE_ORDER else len(STAGE_ORDER),
            int(row.get("trial") or 0),
        )
    )
    return failures


def format_float(value: Any, digits: int = 2) -> str:
    number = numeric(value)
    if number is None or not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def format_pct(value: Any, digits: int = 1) -> str:
    number = numeric(value)
    if number is None or not math.isfinite(number):
        return "NA"
    return f"{number:+.{digits}f}%"


def escape_md(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_None._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    threshold_wall_repeated: float = 20.0,
    threshold_rss_repeated: float = 15.0,
    threshold_wall_single: float = 25.0,
    threshold_rss_single: float = 20.0,
) -> str:
    failures = failure_rows(rows)
    flagged = [row for row in comparisons if row.get("obvious_regression")]

    lines: List[str] = []
    lines.append("# Branch Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Baseline branch: `{BASELINE_BRANCH}`")
    lines.append(f"Candidate branch: `{CANDIDATE_BRANCH}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- Repeated cases: wall `>{threshold_wall_repeated:.0f}%`, RSS `>{threshold_rss_repeated:.0f}%`")
    lines.append(f"- Single-trial large backed cases: wall `>{threshold_wall_single:.0f}%`, RSS `>{threshold_rss_single:.0f}%`")
    lines.append("")

    lines.append("## Datasets")
    lines.append("")
    dataset_rows = []
    for dataset_name in DATASET_ORDER:
        entry = manifest.get(dataset_name)
        if not entry:
            continue
        dataset_rows.append([
            dataset_name,
            entry.get("tier", "NA"),
            entry.get("path", "NA"),
        ])
    lines.append(markdown_table(["Dataset", "Tier", "Path"], dataset_rows))
    lines.append("")

    lines.append("## Failures")
    lines.append("")
    if failures:
        failure_table = [
            [
                row.get("dataset"),
                row.get("mode"),
                row.get("trial"),
                row.get("branch"),
                row.get("stage"),
                row.get("status"),
                row.get("failure_reason"),
            ]
            for row in failures
        ]
        lines.append(markdown_table(
            ["Dataset", "Mode", "Trial", "Branch", "Stage", "Status", "Reason"],
            failure_table,
        ))
    else:
        lines.append("_No failures recorded._")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    if flagged:
        flagged_sorted = sorted(
            flagged,
            key=lambda row: max(
                numeric(row.get("wall_delta_pct")) or 0.0,
                numeric(row.get("rss_delta_pct")) or 0.0,
            ),
            reverse=True,
        )
        lines.append("Obvious regressions:")
        lines.append("")
        for row in flagged_sorted[:12]:
            lines.append(
                f"- `{row['dataset']}` `{row['mode']}` `{row['stage']}`: "
                f"wall {format_pct(row['wall_delta_pct'])}, "
                f"RSS {format_pct(row['rss_delta_pct'])}"
            )
    else:
        lines.append("_No obvious regressions crossed the configured thresholds among comparable successful cases._")

    asymmetric_failures = [row for row in failures if row.get("stage") == "total"]
    if asymmetric_failures:
        lines.append("")
        lines.append("Asymmetric or outright case failures remain important findings even when no thresholded regression appears above.")
    lines.append("")

    lines.append("## Comparable Stage Deltas")
    lines.append("")
    comparison_table = [
        [
            row["dataset"],
            row["mode"],
            row["stage"],
            format_float(row["baseline_wall_s"]),
            format_float(row["candidate_wall_s"]),
            format_pct(row["wall_delta_pct"]),
            format_float(row["baseline_peak_rss_mb"]),
            format_float(row["candidate_peak_rss_mb"]),
            format_pct(row["rss_delta_pct"]),
            "yes" if row["obvious_regression"] else "",
        ]
        for row in comparisons
    ]
    lines.append(markdown_table(
        [
            "Dataset",
            "Mode",
            "Stage",
            f"{BASELINE_BRANCH} wall (s)",
            f"{CANDIDATE_BRANCH} wall (s)",
            "Wall delta",
            f"{BASELINE_BRANCH} RSS (MB)",
            f"{CANDIDATE_BRANCH} RSS (MB)",
            "RSS delta",
            "Flag",
        ],
        comparison_table,
    ))
    lines.append("")

    backed_growth = [
        row for row in comparisons
        if row.get("mode") == BACKED_MODE and row.get("stage") == "total"
    ]
    backed_growth.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]) if row.get("dataset") in DATASET_ORDER else len(DATASET_ORDER),
        )
    )
    lines.append("## Backed Growth Summary")
    lines.append("")
    if backed_growth:
        growth_table = [
            [
                row["dataset"],
                format_float(row["baseline_wall_s"]),
                format_float(row["candidate_wall_s"]),
                format_pct(row["wall_delta_pct"]),
                format_float(row["baseline_peak_rss_mb"]),
                format_float(row["candidate_peak_rss_mb"]),
                format_pct(row["rss_delta_pct"]),
            ]
            for row in backed_growth
        ]
        lines.append(markdown_table(
            [
                "Dataset",
                f"{BASELINE_BRANCH} total (s)",
                f"{CANDIDATE_BRANCH} total (s)",
                "Wall delta",
                f"{BASELINE_BRANCH} RSS (MB)",
                f"{CANDIDATE_BRANCH} RSS (MB)",
                "RSS delta",
            ],
            growth_table,
        ))
    else:
        lines.append("_No comparable backed total rows were available._")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Raw rows: `{output_dir / 'raw'}`")
    lines.append(f"- Summary CSV: `{output_dir / 'summary.csv'}`")
    lines.append(f"- Comparison CSV: `{output_dir / 'comparison.csv'}`")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def create_report(output_dir: Path, manifest: Dict[str, Any]) -> None:
    raw_dir = output_dir / "raw"
    rows = load_all_rows(raw_dir)
    write_csv(rows, output_dir / "summary.csv")
    comparisons = build_comparison_rows(rows)
    write_csv(comparisons, output_dir / "comparison.csv")
    report = build_report(output_dir, rows, comparisons, manifest)
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(report)


def orchestrate(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    smoke_root = work_root / "smoke"
    case_work_root = work_root / "cases"
    data_root = out_dir / "data"
    manifest_path = out_dir / "dataset_manifest.json"

    branch_runtimes = {
        branch: create_branch_runtime(branch, work_root)
        for branch in args.branches
    }

    prepare_python = branch_runtimes[args.branches[0]].python
    manifest = prepare_datasets(prepare_python, data_root, manifest_path)

    for branch in args.branches:
        smoke_check(branch_runtimes[branch], smoke_root)

    def dataset_entry(name: str) -> Dict[str, Any]:
        entry = manifest.get(name)
        if entry is None:
            raise KeyError(f"Missing dataset entry for {name}")
        return entry

    def run_pair(dataset_name: str, mode: str, trial: int) -> Dict[str, Dict[str, Any]]:
        entry = dataset_entry(dataset_name)
        results: Dict[str, Dict[str, Any]] = {}
        for branch in args.branches:
            runtime = branch_runtimes[branch]
            spec = make_case_spec(
                runtime,
                dataset_name,
                entry["path"],
                entry.get("tier"),
                mode,
                trial,
                raw_dir,
                case_work_root,
                args.backed_chunk_size,
                args.case_timeout_s,
            )
            results[branch] = execute_case(runtime, spec)
        return results

    # Mandatory repeated cases.
    for trial in (1, 2):
        run_pair("sparse_medium", INMEMORY_MODE, trial)
        run_pair("sparse_medium", BACKED_MODE, trial)
        run_pair("scale_subset_100k", BACKED_MODE, trial)

    # In-memory frontier gate.
    first_100k_results = run_pair("scale_subset_100k", INMEMORY_MODE, 1)
    both_100k_ok = all(
        is_ok_under_rss_limit(first_100k_results[branch], args.max_inmemory_rss_gb)
        for branch in args.branches
    )
    if both_100k_ok:
        run_pair("scale_subset_100k", INMEMORY_MODE, 2)
    else:
        for trial in (1, 2):
            run_pair("scale_subset_50k", INMEMORY_MODE, trial)

    # Larger backed growth tiers.
    run_pair("scale_subset_200k", BACKED_MODE, 1)
    full_results = run_pair("scale_full", BACKED_MODE, 1)
    if all(is_host_limit_failure(full_results[branch]) for branch in args.branches):
        run_pair("scale_subset_250k", BACKED_MODE, 1)

    create_report(out_dir, manifest)
    log(f"[benchmark] Report written to {out_dir / 'report.md'}")
    return 0


def internal_prepare_data(args: argparse.Namespace) -> int:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    repo = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    sparse_src = repo / "data" / "adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad"
    scale_src = repo / "data" / "adata_agg_Scn4b_OX_fil.h5ad"

    if not sparse_src.exists():
        raise FileNotFoundError(f"Missing sparse_medium source file: {sparse_src}")
    if not scale_src.exists():
        raise FileNotFoundError(f"Missing scale_full source file: {scale_src}")

    manifest: Dict[str, Any] = {}

    sparse_dst = data_root / "sparse_medium.h5ad"
    if not sparse_dst.exists():
        log(f"[prepare-data] Copying sparse_medium to {sparse_dst}")
        shutil.copy2(sparse_src, sparse_dst)

    adata_sparse = ad.read_h5ad(str(sparse_dst), backed="r")
    try:
        is_sparse = sp.issparse(adata_sparse.X)
        sparse_n_obs = int(adata_sparse.n_obs)
    finally:
        if getattr(adata_sparse, "file", None) is not None:
            try:
                adata_sparse.file.close()
            except Exception:
                pass
        del adata_sparse
        gc.collect()

    if not is_sparse:
        log("[prepare-data] Converting sparse_medium run-local copy to CSR")
        adata_sparse_mem = ad.read_h5ad(str(sparse_dst))
        if not sp.issparse(adata_sparse_mem.X):
            adata_sparse_mem.X = sp.csr_matrix(adata_sparse_mem.X)
        adata_sparse_mem.write_h5ad(str(sparse_dst))
        del adata_sparse_mem
        gc.collect()

    manifest["sparse_medium"] = {
        "path": str(sparse_dst),
        "tier": sparse_n_obs,
    }

    obs_adata = ad.read_h5ad(str(scale_src), backed="r")
    try:
        obs = obs_adata.obs.copy()
    finally:
        if getattr(obs_adata, "file", None) is not None:
            try:
                obs_adata.file.close()
            except Exception:
                pass
        del obs_adata
        gc.collect()

    manifest["scale_full"] = {
        "path": str(scale_src),
        "tier": int(len(obs)),
    }

    counts = obs["UID"].value_counts()
    eligible = counts[counts >= 100].index.tolist()
    eligible_obs = obs[obs["UID"].isin(eligible)].copy()
    total_eligible = len(eligible_obs)

    for label, tier in TIER_SIZES.items():
        subset_name = f"scale_subset_{label}"
        subset_path = data_root / f"{subset_name}.h5ad"
        if subset_path.exists():
            manifest[subset_name] = {
                "path": str(subset_path),
                "tier": tier,
            }
            continue

        log(f"[prepare-data] Generating {subset_name} -> {subset_path}")
        if total_eligible < tier:
            selected_idx = eligible_obs.index.tolist()
        else:
            rng = np.random.default_rng(42)
            sample_sizes: Dict[str, int] = {}
            for uid in eligible:
                n_uid = int(counts[uid])
                sample_sizes[uid] = max(0, int(round(n_uid / total_eligible * tier)))

            diff = tier - sum(sample_sizes.values())
            if diff != 0:
                sorted_uids = sorted(eligible, key=lambda uid: -int(counts[uid]))
                for idx in range(abs(diff)):
                    uid = sorted_uids[idx % len(sorted_uids)]
                    sample_sizes[uid] += 1 if diff > 0 else -1
                    sample_sizes[uid] = max(0, min(int(counts[uid]), sample_sizes[uid]))

            selected_idx = []
            for uid in eligible:
                n_take = sample_sizes.get(uid, 0)
                if n_take <= 0:
                    continue
                uid_idx = eligible_obs.index[eligible_obs["UID"] == uid].tolist()
                chosen = rng.choice(uid_idx, size=min(n_take, len(uid_idx)), replace=False).tolist()
                selected_idx.extend(chosen)

        full_adata = ad.read_h5ad(str(scale_src))
        subset = full_adata[selected_idx].copy()
        del full_adata
        gc.collect()
        subset.write_h5ad(str(subset_path))
        del subset
        gc.collect()

        manifest[subset_name] = {
            "path": str(subset_path),
            "tier": tier,
        }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    log(f"[prepare-data] Wrote manifest to {manifest_path}")
    return 0


def internal_smoke_check(args: argparse.Namespace) -> int:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import actionet as an
    from actionet.tools import scale as scale_coords

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "smoke.h5ad"

    rng = np.random.default_rng(7)
    n_cells = 48
    n_genes = 36
    labels = np.array([f"CT_{i % 3}" for i in range(n_cells)], dtype=object)
    batches = np.array([f"B{i % 2}" for i in range(n_cells)], dtype=object)

    X = rng.poisson(0.2, size=(n_cells, n_genes)).astype(np.float64)
    for ct in range(3):
        rows = np.where(labels == f"CT_{ct}")[0]
        cols = np.arange(ct * 4, min(n_genes, ct * 4 + 4))
        X[np.ix_(rows, cols)] += rng.poisson(2.5, size=(rows.size, cols.size))

    X = sp.csr_matrix(X)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"CellLabel": labels, "UID": batches}),
        var=pd.DataFrame({"Gene": np.array([f"G{i}" for i in range(n_genes)], dtype=object)}),
    )
    adata.obs_names = np.array([f"cell_{i}" for i in range(n_cells)], dtype=object)
    adata.var_names = np.array([f"G{i}" for i in range(n_genes)], dtype=object)
    adata.write_h5ad(path)
    del adata

    backed = ad.read_h5ad(str(path), backed="r+")
    an.filter_anndata(backed, min_cells_per_feat=0.01, backed_chunk_size=32, inplace=True)
    an.normalize_anndata(backed, target_sum=1e4, log_transform=True, log_base=2, backed_chunk_size=32, inplace=True)
    try:
        if getattr(backed, "file", None) is not None:
            backed.file.close()
    except Exception:
        pass

    an.reduce_kernel(backed, n_components=8, key_added="action", svd_algorithm="halko", seed=42, backed_chunk_size=32, inplace=True)
    backed = ad.read_h5ad(str(path), backed="r+")
    an.correct_batch_effect(backed, batch_key="UID", reduction_key="action", backed_chunk_size=32, inplace=True)
    an.run_action(backed, reduction_key="action_corrected", k_min=2, k_max=8, inplace=True)
    an.build_network(backed, algorithm="k*nn", mutual_edges_only=True, inplace=True)
    an.compute_network_diffusion(backed, scores="H_merged", key_added="archetype_footprint", inplace=True)
    initial = scale_coords(np.asarray(backed.obsm["archetype_footprint"], dtype=np.float64))
    an.layout_network(backed, initial_coords=initial, method="umap", n_components=2, n_epochs=100, seed=42, key_added="umap_2d_actionet", inplace=True)
    an.compute_archetype_feature_specificity(
        backed,
        archetype_key="archetype_footprint",
        key_added="archetype",
        n_threads=1,
        backed_chunk_size=32,
        inplace=True,
    )

    required_keys = [
        "action",
        "action_corrected",
        "H_stacked",
        "H_merged",
        "archetype_footprint",
        "umap_2d_actionet",
    ]
    for key in required_keys:
        if key not in backed.obsm:
            raise RuntimeError(f"Smoke check missing adata.obsm[{key!r}]")

    if getattr(backed, "file", None) is not None:
        try:
            backed.file.close()
        except Exception:
            pass
    log(f"[smoke-check] Branch {args.branch} passed")
    return 0


class WorkerStageProfiler:
    def __init__(self, sample_interval: float = 0.05):
        self.sample_interval = sample_interval
        self.elapsed = 0.0
        self.peak_rss_mb = 0.0
        self._peak_abs = 0.0

    def _sampler(self) -> None:
        import psutil

        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / 1e6
                if rss > self._peak_abs:
                    self._peak_abs = rss
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def __enter__(self) -> "WorkerStageProfiler":
        import psutil

        gc.collect()
        self._proc = psutil.Process()
        self._rss0 = self._proc.memory_info().rss / 1e6
        self._peak_abs = self._rss0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.elapsed = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            rss = self._proc.memory_info().rss / 1e6
            self._peak_abs = max(self._peak_abs, rss)
        except Exception:
            pass
        self.peak_rss_mb = max(0.0, self._peak_abs - self._rss0)
        return False


def worker_io_counters_mb() -> tuple[float, float]:
    import psutil

    try:
        counters = psutil.Process().io_counters()
        return counters.read_bytes / 1e6, counters.write_bytes / 1e6
    except Exception:
        return 0.0, 0.0


def worker_append_row(path: Path, row: ResultRow) -> None:
    append_jsonl(path, row.to_dict())


def internal_run_case(args: argparse.Namespace) -> int:
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import actionet as an
    from actionet.tools import scale as scale_coords
    from actionet._backed_compression import get_storage_metadata_from_adata, is_compressed_storage

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    spec = CaseSpec(**config)
    output_jsonl = Path(spec.output_jsonl)
    work_dir = Path(spec.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_src = Path(spec.dataset_path)
    backed_case_path: Optional[Path] = None

    adata: Any = None

    def current_nnz(obj: Any) -> int:
        try:
            if sp.issparse(obj.X):
                return int(obj.X.nnz)
            if hasattr(obj.X, "shape"):
                return int(np.prod(obj.X.shape))
        except Exception:
            pass
        return 0

    def append_stage_row(
        *,
        stage: str,
        wall_s: float,
        peak_rss_mb: float,
        io_read_mb: float,
        io_write_mb: float,
        status: str,
        failure_reason: Optional[str] = None,
    ) -> None:
        row = ResultRow(
            branch=spec.branch,
            dataset=spec.dataset,
            tier=spec.tier,
            mode=spec.mode,
            trial=spec.trial,
            case_id=spec.case_id,
            stage=stage,
            wall_s=wall_s,
            peak_rss_mb=peak_rss_mb,
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb,
            status=status,
            failure_reason=failure_reason,
            n_obs=int(getattr(adata, "n_obs", 0) or 0),
            n_vars=int(getattr(adata, "n_vars", 0) or 0),
            nnz=current_nnz(adata) if adata is not None else 0,
        )
        worker_append_row(output_jsonl, row)

    def run_stage(stage: str, fn: Any) -> None:
        io_r0, io_w0 = worker_io_counters_mb()
        try:
            with WorkerStageProfiler() as profiler:
                fn()
        except Exception as exc:
            io_r1, io_w1 = worker_io_counters_mb()
            append_stage_row(
                stage=stage,
                wall_s=profiler.elapsed,
                peak_rss_mb=profiler.peak_rss_mb,
                io_read_mb=max(0.0, io_r1 - io_r0),
                io_write_mb=max(0.0, io_w1 - io_w0),
                status="failed",
                failure_reason=str(exc),
            )
            raise
        else:
            io_r1, io_w1 = worker_io_counters_mb()
            append_stage_row(
                stage=stage,
                wall_s=profiler.elapsed,
                peak_rss_mb=profiler.peak_rss_mb,
                io_read_mb=max(0.0, io_r1 - io_r0),
                io_write_mb=max(0.0, io_w1 - io_w0),
                status="ok",
                failure_reason=None,
            )

    total_io_r0 = 0.0
    total_io_w0 = 0.0
    total_profiler: Optional[WorkerStageProfiler] = None

    try:
        if spec.mode == BACKED_MODE:
            backed_case_path = work_dir / f"{spec.case_id}.h5ad"
            shutil.copy2(dataset_src, backed_case_path)
            adata = ad.read_h5ad(str(backed_case_path), backed="r+")
            if is_compressed_storage(get_storage_metadata_from_adata(adata)):
                log(f"[run-case] Decompressing backed storage for {spec.case_id}")
                an.decompress_backed_storage(adata, scope="file", chunk_size=spec.backed_chunk_size, verbose=False)
        else:
            adata = ad.read_h5ad(str(dataset_src))

        # Untimed setup.
        an.filter_anndata(adata, min_cells_per_feat=0.01, backed_chunk_size=spec.backed_chunk_size, inplace=True)
        an.normalize_anndata(
            adata,
            target_sum=1e4,
            log_transform=True,
            log_base=2,
            backed_chunk_size=spec.backed_chunk_size,
            inplace=True,
        )

        if spec.mode == BACKED_MODE and backed_case_path is not None:
            try:
                if getattr(adata, "file", None) is not None:
                    adata.file.close()
            except Exception:
                pass
            del adata
            gc.collect()
            adata = ad.read_h5ad(str(backed_case_path), backed="r+")

        total_io_r0, total_io_w0 = worker_io_counters_mb()
        total_profiler = WorkerStageProfiler()
        total_profiler.__enter__()

        def reduce_kernel_stage() -> None:
            nonlocal adata
            if spec.mode == BACKED_MODE and backed_case_path is not None:
                try:
                    if getattr(adata, "file", None) is not None:
                        adata.file.close()
                except Exception:
                    pass
            an.reduce_kernel(
                adata,
                n_components=30,
                key_added="action",
                svd_algorithm="halko" if spec.mode == BACKED_MODE else "irlb",
                seed=42,
                backed_chunk_size=spec.backed_chunk_size,
                verbose=False,
                inplace=True,
            )
            if spec.mode == BACKED_MODE and backed_case_path is not None:
                adata = ad.read_h5ad(str(backed_case_path), backed="r+")

        run_stage("reduce_kernel", reduce_kernel_stage)
        run_stage(
            "correct_batch_effect",
            lambda: an.correct_batch_effect(
                adata,
                batch_key="UID",
                reduction_key="action",
                backed_chunk_size=spec.backed_chunk_size,
                inplace=True,
            ),
        )
        run_stage(
            "run_action",
            lambda: an.run_action(
                adata,
                reduction_key="action_corrected",
                k_min=2,
                k_max=30,
                inplace=True,
            ),
        )
        run_stage(
            "build_network",
            lambda: an.build_network(
                adata,
                algorithm="k*nn",
                mutual_edges_only=True,
                inplace=True,
            ),
        )
        run_stage(
            "compute_network_diffusion",
            lambda: an.compute_network_diffusion(
                adata,
                scores="H_merged",
                key_added="archetype_footprint",
                inplace=True,
            ),
        )

        def layout_stage() -> None:
            initial = scale_coords(np.asarray(adata.obsm["archetype_footprint"], dtype=np.float64))
            an.layout_network(
                adata,
                network_key="actionet",
                initial_coords=initial,
                method="umap",
                n_components=2,
                spread=1.0,
                min_dist=1.0,
                n_epochs=100,
                seed=42,
                key_added="umap_2d_actionet",
                inplace=True,
                verbose=False,
            )

        run_stage("layout_network_2d", layout_stage)
        run_stage(
            "compute_archetype_feature_specificity",
            lambda: an.compute_archetype_feature_specificity(
                adata,
                archetype_key="archetype_footprint",
                key_added="archetype",
                n_threads=1,
                backed_chunk_size=spec.backed_chunk_size,
                inplace=True,
            ),
        )

        total_profiler.__exit__(None, None, None)
        total_io_r1, total_io_w1 = worker_io_counters_mb()
        append_stage_row(
            stage="total",
            wall_s=total_profiler.elapsed,
            peak_rss_mb=total_profiler.peak_rss_mb,
            io_read_mb=max(0.0, total_io_r1 - total_io_r0),
            io_write_mb=max(0.0, total_io_w1 - total_io_w0),
            status="ok",
            failure_reason=None,
        )
        log(f"[run-case] Completed {spec.case_id}")
        return 0
    except MemoryError as exc:
        if total_profiler is not None:
            total_profiler.__exit__(None, None, None)
        total_io_r1, total_io_w1 = worker_io_counters_mb()
        append_stage_row(
            stage="total",
            wall_s=0.0 if total_profiler is None else total_profiler.elapsed,
            peak_rss_mb=0.0 if total_profiler is None else total_profiler.peak_rss_mb,
            io_read_mb=max(0.0, total_io_r1 - total_io_r0),
            io_write_mb=max(0.0, total_io_w1 - total_io_w0),
            status="memory_limit",
            failure_reason=str(exc),
        )
        return 2
    except Exception as exc:
        if total_profiler is not None:
            total_profiler.__exit__(None, None, None)
        total_io_r1, total_io_w1 = worker_io_counters_mb()
        append_stage_row(
            stage="total",
            wall_s=0.0 if total_profiler is None else total_profiler.elapsed,
            peak_rss_mb=0.0 if total_profiler is None else total_profiler.peak_rss_mb,
            io_read_mb=max(0.0, total_io_r1 - total_io_r0),
            io_write_mb=max(0.0, total_io_w1 - total_io_w0),
            status="failed",
            failure_reason=str(exc),
        )
        return 1
    finally:
        try:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand")

    prepare = subparsers.add_parser("prepare-data", help="Internal: prepare local benchmark datasets")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--manifest-path", required=True)

    smoke = subparsers.add_parser("smoke-check", help="Internal: smoke-test one branch env")
    smoke.add_argument("--branch", required=True)
    smoke.add_argument("--work-dir", required=True)

    run_case = subparsers.add_parser("run-case", help="Internal: run one benchmark case")
    run_case.add_argument("--config", required=True)

    parser.add_argument(
        "--branches",
        nargs=2,
        default=[BASELINE_BRANCH, CANDIDATE_BRANCH],
        help="Branches to compare. Baseline must come first.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for raw results, CSVs, and markdown report.",
    )
    parser.add_argument(
        "--work-root",
        default=str(Path(tempfile.gettempdir()) / f"actionet_branch_compare_{timestamp_id()}"),
        help="Temporary work root for worktrees, virtualenvs, smoke data, and per-case files.",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=int,
        default=DEFAULT_CASE_TIMEOUT_S,
        help="Per-case timeout in seconds.",
    )
    parser.add_argument(
        "--backed-chunk-size",
        type=int,
        default=DEFAULT_BACKED_CHUNK_SIZE,
        help="Chunk size for backed operators.",
    )
    parser.add_argument(
        "--max-inmemory-rss-gb",
        type=float,
        default=DEFAULT_MAX_INMEMORY_RSS_GB,
        help="If the first 100k in-memory trial on either branch exceeds this RSS limit, switch to the 50k in-memory fallback tier.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.subcommand == "prepare-data":
        return internal_prepare_data(args)
    if args.subcommand == "smoke-check":
        return internal_smoke_check(args)
    if args.subcommand == "run-case":
        return internal_run_case(args)

    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
