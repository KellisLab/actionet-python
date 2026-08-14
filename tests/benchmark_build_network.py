#!/usr/bin/env python3
"""Focused benchmark for the build_network hot path (k*nn and knn).

Isolates network construction from the rest of the pipeline by operating on a
cached, precomputed ``H_stacked`` embedding.  Two phases:

  prep   Compute ``H_stacked`` for one or more datasets (backed mode for large
         data to keep RSS low) and cache each embedding to a .npy file.

  bench  Load a cached embedding and time ``_core.build_network`` directly for
         both algorithms across a thread sweep, recording wall time and graph
         statistics.  Optionally loads a separately built baseline ``_core`` so
         the current and baseline implementations run in one process against
         identical inputs.

The benchmark calls the compiled ``_core.build_network`` entry point directly
(bypassing AnnData persistence) so the measured time is the C++ builder plus the
scipy CSR construction only.

Usage examples:
    # 1) Prepare embeddings (backed) for a few sizes.
    python tests/benchmark_build_network.py prep \
        --dataset data/actionet_benchmark/scale_subset_100k.h5ad \
        --out tests/_bench_cache/h_100k.npy

    # 2) Benchmark current build.
    python tests/benchmark_build_network.py bench \
        --embedding tests/_bench_cache/h_100k.npy \
        --threads 1,8 --algorithms k*nn,knn

    # 3) Compare against a baseline _core built elsewhere.
    python tests/benchmark_build_network.py bench \
        --embedding tests/_bench_cache/h_100k.npy \
        --baseline-core /path/to/baseline/_core.cpython-*.so
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# prep: compute and cache H_stacked
# --------------------------------------------------------------------------
def cmd_prep(args: argparse.Namespace) -> int:
    import anndata as ad
    import actionet as an

    src = Path(args.dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        emb = np.load(out)
        log(f"[prep] {out} exists (shape {emb.shape}); use --force to recompute")
        return 0

    log(f"[prep] Opening {src} (backed={args.backed})")
    if args.backed:
        # Work on a private copy so normalization/reduction rewrites do not
        # mutate the shared source file.
        import shutil

        work = out.parent / (src.stem + ".work.h5ad")
        if not work.exists() or args.force:
            log(f"[prep] Copying to {work}")
            shutil.copy2(src, work)
        adata = ad.read_h5ad(str(work), backed="r+")
    else:
        adata = ad.read_h5ad(str(src))

    t0 = time.perf_counter()
    an.filter_anndata(adata, min_cells_per_feat=0.01,
                      backed_chunk_size=args.chunk, inplace=True)
    an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2,
                         backed_chunk_size=args.chunk, inplace=True)

    # Re-open backed handle after the normalization rewrite transaction.
    if args.backed:
        try:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        adata = ad.read_h5ad(str(work), backed="r+")

    an.reduce_kernel(
        adata, n_components=args.n_components, key_added="action",
        svd_algorithm="halko" if args.backed else "irlb",
        seed=42, backed_chunk_size=args.chunk, verbose=False, inplace=True,
    )
    if args.backed:
        try:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()
        adata = ad.read_h5ad(str(work), backed="r+")

    an.run_action(adata, reduction_key="action", k_min=args.k_min, k_max=args.k_max,
                  n_threads=args.threads, inplace=True)

    emb = np.ascontiguousarray(adata.obsm["H_stacked"], dtype=np.float32)
    elapsed = time.perf_counter() - t0
    np.save(out, emb)
    log(f"[prep] Wrote {out} shape={emb.shape} dtype={emb.dtype} in {elapsed:.1f}s")

    try:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    except Exception:
        pass
    return 0


# --------------------------------------------------------------------------
# _core loading (current or baseline)
#
# The compiled module always exports PyInit__core, so a baseline .so can only be
# imported under the name "_core".  To run current and baseline in the same
# session we would collide in sys.modules, so each build is benchmarked in its
# own subprocess (see run_bench_worker / _worker_main).
# --------------------------------------------------------------------------
def load_core_as_core(baseline_path: Optional[str]) -> Any:
    """Import the _core extension. If baseline_path is given, load that .so
    under the canonical module name "_core"; otherwise import the installed one.
    Must run in a fresh process so the module name is not already bound."""
    import importlib

    if baseline_path is None:
        from actionet import _core
        return _core

    path = Path(baseline_path).resolve()
    sys.modules.pop("_core", None)
    spec = importlib.util.spec_from_file_location("_core", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline _core from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_core"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# bench: time build_network
# --------------------------------------------------------------------------
def bench_one(core: Any, H: np.ndarray, *, algorithm: str, metric: str,
              k: int, threads: int, mutual: bool, trials: int) -> Dict[str, Any]:
    times: List[float] = []
    nnz = 0
    n = H.shape[0]
    for _ in range(trials):
        gc.collect()
        t0 = time.perf_counter()
        G = core.build_network(
            H, algorithm, metric, 1.0, threads,
            16.0, 200.0, 200.0, mutual, k,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        nnz = int(G.nnz)
        del G
    times.sort()
    return {
        "algorithm": algorithm,
        "metric": metric,
        "k": k,
        "threads": threads,
        "n": n,
        "dim": int(H.shape[1]),
        "trials": trials,
        "best_s": times[0],
        "median_s": times[len(times) // 2],
        "nnz": nnz,
        "mean_degree": (nnz / n) if n else 0.0,
    }


def cmd_bench(args: argparse.Namespace) -> int:
    import subprocess

    H = np.ascontiguousarray(np.load(args.embedding), dtype=np.float32)
    log(f"[bench] Embedding {args.embedding} shape={H.shape} dtype={H.dtype}")

    labels: List[tuple[str, Optional[str]]] = [("current", None)]
    if args.baseline_core:
        labels.append(("baseline", args.baseline_core))

    results: List[Dict[str, Any]] = []
    for label, core_path in labels:
        # Run each build in its own process so the _core module name is free.
        worker_cmd = [
            sys.executable, str(Path(__file__).resolve()), "_worker",
            "--embedding", args.embedding,
            "--algorithms", args.algorithms,
            "--metric", args.metric,
            "--k", str(args.k),
            "--threads", args.threads,
            "--trials", str(args.trials),
        ]
        if args.no_mutual:
            worker_cmd.append("--no-mutual")
        if core_path:
            worker_cmd += ["--worker-core", core_path]

        log(f"[bench] === {label} core ===")
        proc = subprocess.run(worker_cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            log(proc.stdout)
            log(proc.stderr)
            raise RuntimeError(f"worker failed for {label}")
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            r["build"] = label
            results.append(r)
            log(f"[bench] {label:8s} {r['algorithm']:5s} k={r['k']} thr={r['threads']:2d} "
                f"best={r['best_s']:.3f}s median={r['median_s']:.3f}s "
                f"nnz={r['nnz']} deg={r['mean_degree']:.1f}")

    if args.baseline_core:
        log("\n[bench] === speedup (baseline_median / current_median) ===")
        by_key: Dict[tuple, Dict[str, float]] = {}
        for r in results:
            key = (r["algorithm"], r["threads"])
            by_key.setdefault(key, {})[r["build"]] = r["median_s"]
        for key in sorted(by_key.keys()):
            d = by_key[key]
            if "current" in d and "baseline" in d and d["current"] > 0:
                sp = d["baseline"] / d["current"]
                log(f"[bench] {key[0]:5s} thr={key[1]:2d}  "
                    f"baseline={d['baseline']:.3f}s current={d['current']:.3f}s  "
                    f"speedup={sp:.2f}x")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        log(f"[bench] Wrote {args.out}")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    """Benchmark a single build in this process; emit one JSON line per config."""
    H = np.ascontiguousarray(np.load(args.embedding), dtype=np.float32)
    core = load_core_as_core(args.worker_core)
    thread_list = [int(t) for t in args.threads.split(",") if t.strip()]
    algo_list = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for algo in algo_list:
        for threads in thread_list:
            r = bench_one(
                core, H,
                algorithm=algo, metric=args.metric, k=args.k,
                threads=threads, mutual=not args.no_mutual, trials=args.trials,
            )
            print(json.dumps(r, sort_keys=True), flush=True)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="Compute and cache H_stacked")
    pp.add_argument("--dataset", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--backed", action="store_true", default=True)
    pp.add_argument("--in-memory", dest="backed", action="store_false")
    pp.add_argument("--chunk", type=int, default=4096)
    pp.add_argument("--n-components", type=int, default=30)
    pp.add_argument("--k-min", type=int, default=2)
    pp.add_argument("--k-max", type=int, default=30)
    pp.add_argument("--threads", type=int, default=0)
    pp.add_argument("--force", action="store_true")

    pb = sub.add_parser("bench", help="Benchmark build_network on a cached embedding")
    pb.add_argument("--embedding", required=True)
    pb.add_argument("--algorithms", default="k*nn,knn")
    pb.add_argument("--metric", default="jsd")
    pb.add_argument("--k", type=int, default=100)
    pb.add_argument("--threads", default="1,8")
    pb.add_argument("--trials", type=int, default=3)
    pb.add_argument("--no-mutual", action="store_true")
    pb.add_argument("--baseline-core", default=None,
                    help="Path to a baseline _core .so to compare against")
    pb.add_argument("--out", default=None)

    pw = sub.add_parser("_worker", help="(internal) benchmark one build in-process")
    pw.add_argument("--embedding", required=True)
    pw.add_argument("--algorithms", default="k*nn,knn")
    pw.add_argument("--metric", default="jsd")
    pw.add_argument("--k", type=int, default=100)
    pw.add_argument("--threads", default="1,8")
    pw.add_argument("--trials", type=int, default=3)
    pw.add_argument("--no-mutual", action="store_true")
    pw.add_argument("--worker-core", default=None)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "prep":
        return cmd_prep(args)
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "_worker":
        return cmd_worker(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
