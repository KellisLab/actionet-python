#!/usr/bin/env python3
"""Atlas-scale validation of within-group batch correction.

Two stages, run with different builds against the same input file:

``capture``  (pre-change build: actionet-python e9e33a7, libactionet 92c0e2e)
    Replays the ``new_batch_corr.qmd`` prototype: filter → lazy transform →
    ``reduce_kernel`` → crude kNN + Leiden groups → prototype design →
    ``correct_batch_effect(design=...)``, plus a plain ``batch_key`` run.
    Saves the reduction, crude groups, and both corrected reductions.
    Uses only APIs that exist in the pre-change build.

``compare``  (new build)
    Loads the captured reduction and groups, then runs these arms:

    ========  ==========================================================
    nb        prototype design, captured groups           (≈ capture)
    pkg       ``build_batch_design``, captured groups      (≈ nb)
    auto      ``correct_batch_effect(groups="auto")``      (end to end)
    proj      projected rank-k update K_k (I - Z Z') with ``pkg``'s Z
              (candidate one-pass alternative; computed in numpy)
    nodecor   ``pkg`` with ``decorrelate_from_reduction=False``
    depth     ``pkg`` plus the experimental depth column
    nodecor_depth  depth column without decorrelation (the decorrelation
              projects most of the depth column away)
    plain     ``correct_batch_effect(batch_key=...)``     (behaviour change)
    ========  ==========================================================

    and reports sigma / principal-angle agreement with the capture, timings,
    the depth signal left in each embedding, and kNN-based batch mixing and
    label preservation. Writes ``compare.json`` and ``compare.md``.

Usage::

    # pre-change build
    python tests/validate_grouped_batch_correction_atlas.py capture \\
        --input tmp/adata_agg_ALL_fil.h5ad --workdir validation/

    # new build
    python tests/validate_grouped_batch_correction_atlas.py compare \\
        --input tmp/adata_agg_ALL_fil.h5ad --workdir validation/ \\
        --label-key CellType

The input file is copied into ``--workdir`` before filtering (filtering a
backed file rewrites it) unless ``--no-copy`` is given. After filtering,
auto-persist is disabled so the arms do not rewrite the file; results live in
memory and in the ``.npz`` / ``.json`` outputs only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _batch_design_reference import notebook_make_batch_design  # noqa: E402

REDUCTION_KEYS = ("S_r", "U", "A", "B", "sigma")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Stage:
    """Wall time and peak RSS (absolute, MB) of a block."""

    def __init__(self, name: str, log: dict):
        self.name, self.log = name, log

    def _sample(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss / 1e6)
            time.sleep(0.1)

    def __enter__(self):
        self._proc = psutil.Process()
        self._peak = self._proc.memory_info().rss / 1e6
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        print(f"[{self.name}] ...", flush=True)
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.log[self.name] = {"wall_s": elapsed, "peak_rss_mb": self._peak}
        print(f"[{self.name}] {elapsed:.1f}s, peak RSS {self._peak:.0f} MB", flush=True)
        return False


def git_rev(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    rev = out.stdout.strip() or "unknown"
    return rev + ("-dirty" if dirty.stdout.strip() else "")


def build_revs() -> dict[str, str]:
    """Revisions of the installed (editable) actionet checkout and its libactionet submodule."""
    import actionet

    root = Path(actionet.__file__).resolve().parents[2]
    return {"actionet_rev": git_rev(root), "libactionet_rev": git_rev(root / "src" / "libactionet"),
            "actionet_path": str(root)}


def open_and_prepare(args) -> tuple[Any, Any]:
    """Copy, open backed, filter, and build the lazy transform as in the prototype."""
    import anndata
    import actionet as an

    path = Path(args.input)
    if not args.no_copy:
        work = Path(args.workdir) / f"work_{args.stage}.h5ad"
        if not work.exists() or args.recopy:
            print(f"[copy] {path} -> {work}", flush=True)
            shutil.copyfile(path, work)
        path = work
    adata = anndata.read_h5ad(path, backed="r+")
    an.filter_anndata(adata, min_cells_per_feat=args.min_cells_per_feat, inplace=True)
    an.set_auto_persist(adata, False)
    lzt = an.create_lazy_transform(adata, target_sum=np.median(an.matrix_sums(adata, axis=1)))
    return adata, lzt


def reduction_state(adata, key: str) -> dict[str, np.ndarray]:
    return {
        "S_r": np.asarray(adata.obsm[key], dtype=float),
        "U": np.asarray(adata.varm[f"{key}_U"], dtype=float),
        "A": np.asarray(adata.varm[f"{key}_A"], dtype=float),
        "B": np.asarray(adata.obsm[f"{key}_B"], dtype=float),
        "sigma": np.asarray(adata.uns[f"{key}_params"]["sigma"], dtype=float).ravel(),
    }


def pop_corrected(adata, key: str) -> tuple[dict[str, np.ndarray], dict]:
    """Extract S_r, U, A, sigma of a corrected reduction and drop its slots from ``adata``."""
    state = {
        "S_r": np.asarray(adata.obsm.pop(key), dtype=float),
        "U": np.asarray(adata.varm.pop(f"{key}_U"), dtype=float),
        "A": np.asarray(adata.varm.pop(f"{key}_A"), dtype=float),
    }
    params = dict(adata.uns.pop(f"{key}_params"))
    state["sigma"] = np.asarray(params["sigma"], dtype=float).ravel()
    adata.obsm.pop(f"{key}_B", None)
    return state, params


def projected_update(S_r: np.ndarray, V: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-k SVD of K_k (I - Z Z') with K_k = S_r V': the full update minus -(K - K_k) Z Z'."""
    W = V - Z @ (Z.T @ V)
    _, R = np.linalg.qr(W)
    u, s, _ = np.linalg.svd(S_r @ R.T, full_matrices=False)
    return u * s, s


def install_reduction(adata, state: dict[str, np.ndarray], key: str = "action") -> None:
    """Write a captured reduction into the in-memory slots without a disk rewrite."""
    adata.obsm[key] = state["S_r"]
    adata.obsm[f"{key}_B"] = state["B"]
    adata.varm[f"{key}_U"] = state["U"]
    adata.varm[f"{key}_A"] = state["A"]
    params = dict(adata.uns.get(f"{key}_params", {}))
    params["sigma"] = state["sigma"]
    adata.uns[f"{key}_params"] = params


def crude_groups(adata, k: int, resolution: float) -> np.ndarray:
    """Prototype crude clustering using public APIs available in both builds."""
    import anndata
    import actionet as an

    crude = anndata.AnnData(obs=pd.DataFrame(index=adata.obs_names),
                            obsm={"action": np.asarray(adata.obsm["action"])})
    an.build_network(crude, algorithm="knn", distance_metric="l2", k=k,
                     mutual_edges_only=False, obsm_key="action", key_added="unc_knn")
    an.cluster_network(crude, network_key="unc_knn", resolution_parameter=resolution,
                       key_added="leiden_unc")
    return crude.obs["leiden_unc"].to_numpy()


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index from the contingency table."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    n = comb(len(a))
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(axis=1)).sum()
    sum_b = comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / n
    return float((sum_ij - expected) / (0.5 * (sum_a + sum_b) - expected))


def principal_angles_deg(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    s = np.clip(np.linalg.svd(qx.T @ qy, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def within_group_r2(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> float:
    """R^2 of regressing y on X after centring both within groups."""
    codes = pd.factorize(groups)[0]
    sizes = np.bincount(codes).astype(float)
    yc = y - (np.bincount(codes, weights=y) / sizes)[codes]
    Xc = X - np.column_stack([np.bincount(codes, weights=X[:, j]) / sizes for j in range(X.shape[1])])[codes]
    coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    resid = yc - Xc @ coef
    return float(1.0 - resid @ resid / (yc @ yc))


def knn_label_agreement(X: np.ndarray, labels: dict[str, np.ndarray], k: int, n_threads: int) -> dict[str, float]:
    """Mean fraction of each cell's kNN sharing its label, per label set."""
    import anndata
    import actionet as an

    tmp = anndata.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]), obsm={"X": X})
    an.build_network(tmp, algorithm="knn", distance_metric="l2", k=k, mutual_edges_only=False,
                     obsm_key="X", key_added="knn", n_threads=n_threads)
    G = tmp.obsp["knn"].tocsr()
    G.data[:] = 1.0
    deg = np.asarray(G.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    out = {}
    for name, lab in labels.items():
        codes = pd.factorize(lab)[0]
        same = (codes[G.indices] == np.repeat(codes, np.diff(G.indptr))).astype(float)
        G_same = G.copy()
        G_same.data = same
        out[name] = float(np.mean(np.asarray(G_same.sum(axis=1)).ravel() / deg))
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def capture(args) -> None:
    import actionet as an

    timings: dict = {}
    workdir = Path(args.workdir)
    with Stage("prepare", timings):
        adata, lzt = open_and_prepare(args)
    n_donors = int(adata.obs[args.batch_key].nunique())
    print(f"[info] n_obs={adata.n_obs:,} n_vars={adata.n_vars:,} donors={n_donors}", flush=True)
    if n_donors <= 16:
        print("[warn] <= 16 donors: the pre-change build discards the cell-side residual "
              "for the prototype design (projected rank-k path).", flush=True)

    with Stage("reduce_kernel", timings):
        an.reduce_kernel(adata, n_components=args.k, lazy_transform=lzt)
    red = reduction_state(adata, "action")

    with Stage("crude_groups", timings):
        groups = crude_groups(adata, args.group_k, args.resolution)

    adata.obs["leiden_unc"] = groups
    with Stage("nb_design", timings):
        design = notebook_make_batch_design(adata, cluster_key="leiden_unc", batch_key=args.batch_key,
                                            strata_key=args.strata_key, max_batch_frac=args.max_batch_frac)
    with Stage("nb_correct", timings):
        an.correct_batch_effect(adata, design=design, lazy_transform=lzt, corrected_suffix="nb")
    nb, _ = pop_corrected(adata, "action_nb")
    n_design_cols = design.shape[1]
    del design

    with Stage("plain_correct", timings):
        an.correct_batch_effect(adata, batch_key=args.batch_key, lazy_transform=lzt, corrected_suffix="plain")
    plain, _ = pop_corrected(adata, "action_plain")

    payload = {f"red_{k}": v for k, v in red.items()}
    payload.update({f"nb_{k}": v for k, v in nb.items()})
    payload.update(plain_S_r=plain["S_r"], plain_sigma=plain["sigma"])
    payload.update(
        groups=np.asarray(groups).astype(str),
        obs_names=np.asarray(adata.obs_names).astype(str),
        var_names=np.asarray(adata.var_names).astype(str),
    )
    np.savez(workdir / "capture.npz", **payload)
    meta = {
        **build_revs(),
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "n_donors": n_donors,
        "nb_design_columns": int(n_design_cols), "n_groups": int(pd.Series(groups).nunique()),
        "args": vars(args), "timings": timings,
    }
    (workdir / "capture.json").write_text(json.dumps(meta, indent=2, default=str))
    adata.file.close()
    print(f"[done] wrote {workdir / 'capture.npz'}", flush=True)


def compare(args) -> None:
    import actionet as an

    workdir = Path(args.workdir)
    cap = dict(np.load(workdir / "capture.npz", allow_pickle=False))
    cap_meta = json.loads((workdir / "capture.json").read_text())
    timings: dict = {}

    with Stage("prepare", timings):
        adata, lzt = open_and_prepare(args)
    if not np.array_equal(np.asarray(adata.var_names).astype(str), cap["var_names"]):
        raise SystemExit("Filtered var_names differ from the capture; use the same input and --min-cells-per-feat.")
    if not np.array_equal(np.asarray(adata.obs_names).astype(str), cap["obs_names"]):
        raise SystemExit("obs_names differ from the capture.")

    red = {k: cap[f"red_{k}"] for k in REDUCTION_KEYS}
    install_reduction(adata, red)
    groups = cap["groups"]
    n_donors = int(adata.obs[args.batch_key].nunique())

    if args.check_reduce_kernel:
        with Stage("reduce_kernel", timings):
            an.reduce_kernel(adata, n_components=args.k, lazy_transform=lzt, key_added="action_check")
        fresh = reduction_state(adata, "action_check")
        reduce_check = {
            "sigma_max_abs_diff": float(np.abs(fresh["sigma"] - red["sigma"]).max()),
            "max_angle_deg": float(principal_angles_deg(fresh["S_r"], red["S_r"]).max()),
        }
        for slot, suffix in (("obsm", ""), ("obsm", "_B"), ("varm", "_U"), ("varm", "_A"), ("uns", "_params")):
            getattr(adata, slot).pop(f"action_check{suffix}", None)
        del fresh
    else:
        reduce_check = None

    common = dict(batch_key=args.batch_key, strata_key=args.strata_key, max_batch_frac=args.max_batch_frac)
    arms: dict[str, dict[str, np.ndarray]] = {}
    arm_meta: dict[str, dict] = {}

    def run_design(name: str, design: np.ndarray) -> None:
        with Stage(f"{name}_correct", timings):
            an.correct_batch_effect(adata, design=design, lazy_transform=lzt, corrected_suffix=name)
        arms[name], params = pop_corrected(adata, f"action_{name}")
        arm_meta[name] = {"correction_rank": int(params["correction_rank"]),
                          "design_columns": int(design.shape[1])}

    adata.obs["leiden_unc"] = groups
    with Stage("nb_design", timings):
        design = notebook_make_batch_design(adata, cluster_key="leiden_unc", **common)
    run_design("nb", design)

    with Stage("pkg_design", timings):
        design = an.build_batch_design(adata, groups=groups, **common)
    run_design("pkg", design)

    Z = arms["pkg"]["A"][:, red["A"].shape[1]:]
    with Stage("proj_update", timings):
        proj_S_r, proj_sigma = projected_update(red["S_r"], red["U"], Z)
    arms["proj"] = {"S_r": proj_S_r, "sigma": proj_sigma}
    arm_meta["proj"] = {"correction_rank": int(Z.shape[1]), "note": "K_k (I - ZZ') with Z from pkg"}
    del Z

    with Stage("nodecor_design", timings):
        design = an.build_batch_design(adata, groups=groups, decorrelate_from_reduction=False, **common)
    run_design("nodecor", design)

    depth = np.log1p(np.asarray(an.matrix_sums(adata, axis=1), dtype=float).ravel())
    with Stage("depth_design", timings):
        design = an.build_batch_design(adata, groups=groups, depth=depth, **common)
    run_design("depth", design)

    with Stage("nodecor_depth_design", timings):
        design = an.build_batch_design(adata, groups=groups, depth=depth, decorrelate_from_reduction=False, **common)
    run_design("nodecor_depth", design)
    del design

    with Stage("auto_groups", timings):
        auto_groups = an.estimate_batch_groups(adata, k=args.group_k, resolution=args.resolution)
    with Stage("auto_correct", timings):
        an.correct_batch_effect(adata, groups=auto_groups, lazy_transform=lzt, corrected_suffix="auto", **common)
    arms["auto"], params = pop_corrected(adata, "action_auto")
    arm_meta["auto"] = {
        "correction_rank": int(params["correction_rank"]),
        "ari_vs_captured_groups": ari(auto_groups, groups),
        "n_groups": int(len(np.unique(auto_groups))),
        "merged_groups": dict(params["grouped_design"]["merged_groups"]),
    }

    with Stage("plain_correct", timings):
        an.correct_batch_effect(adata, batch_key=args.batch_key, lazy_transform=lzt, corrected_suffix="plain")
    arms["plain"], params = pop_corrected(adata, "action_plain")
    arm_meta["plain"] = {"correction_rank": int(params["correction_rank"])}

    # --- Agreement ---
    def agreement(X, sx, Y, sy) -> dict:
        ang = principal_angles_deg(X, Y)
        top = min(10, X.shape[1])
        return {
            "sigma_max_abs_diff": float(np.abs(sx - sy).max()),
            "sigma_max_rel_diff": float(np.max(np.abs(sx - sy) / np.maximum(np.abs(sy), 1e-300))),
            "max_angle_deg": float(ang.max()),
            f"max_angle_top{top}_deg": float(principal_angles_deg(X[:, :top], Y[:, :top]).max()),
        }

    agree = {
        "nb_vs_capture": agreement(arms["nb"]["S_r"], arms["nb"]["sigma"], cap["nb_S_r"], cap["nb_sigma"]),
        "pkg_vs_nb": agreement(arms["pkg"]["S_r"], arms["pkg"]["sigma"], arms["nb"]["S_r"], arms["nb"]["sigma"]),
        "auto_vs_pkg": agreement(arms["auto"]["S_r"], arms["auto"]["sigma"], arms["pkg"]["S_r"], arms["pkg"]["sigma"]),
        "nodecor_vs_pkg": agreement(arms["nodecor"]["S_r"], arms["nodecor"]["sigma"],
                                    arms["pkg"]["S_r"], arms["pkg"]["sigma"]),
        "depth_vs_pkg": agreement(arms["depth"]["S_r"], arms["depth"]["sigma"], arms["pkg"]["S_r"], arms["pkg"]["sigma"]),
        "nodecor_depth_vs_pkg": agreement(arms["nodecor_depth"]["S_r"], arms["nodecor_depth"]["sigma"],
                                          arms["pkg"]["S_r"], arms["pkg"]["sigma"]),
        "proj_vs_pkg": agreement(arms["proj"]["S_r"], arms["proj"]["sigma"], arms["pkg"]["S_r"], arms["pkg"]["sigma"]),
        "plain_vs_capture_plain": agreement(arms["plain"]["S_r"], arms["plain"]["sigma"],
                                            cap["plain_S_r"], cap["plain_sigma"]),
    }

    # --- Embedding quality ---
    rng = np.random.default_rng(0)
    n_sub = min(args.metric_cells, adata.n_obs)
    sub = np.sort(rng.choice(adata.n_obs, size=n_sub, replace=False))
    label_sets = {"donor": adata.obs[args.batch_key].astype(str).to_numpy()[sub]}
    if args.strata_key:
        label_sets["strata"] = adata.obs[args.strata_key].astype(str).to_numpy()[sub]
    if args.label_key:
        label_sets["label"] = adata.obs[args.label_key].astype(str).to_numpy()[sub]
    sub_groups = groups[sub]
    pair = pd.Series(1, index=pd.MultiIndex.from_arrays([sub_groups, label_sets["donor"]]))
    pair_counts = pair.groupby(level=[0, 1]).transform("size").to_numpy()
    group_counts = pd.Series(sub_groups).map(pd.Series(sub_groups).value_counts()).to_numpy()
    expected_same_donor = float(np.mean(pair_counts / group_counts))

    embeddings = {"uncorrected": red["S_r"], "capture_nb": cap["nb_S_r"], "capture_plain": cap["plain_S_r"]}
    embeddings.update({name: state["S_r"] for name, state in arms.items()})

    quality = {}
    for name, X in embeddings.items():
        with Stage(f"metrics_{name}", timings):
            quality[name] = {
                "depth_r2_within_groups": within_group_r2(depth, X, groups),
                "depth_max_abs_corr": float(np.max(np.abs(
                    [np.corrcoef(X[:, j], depth)[0, 1] for j in range(X.shape[1])]))),
                "knn_same_label_frac": knn_label_agreement(X[sub], label_sets, args.metric_k, args.n_threads),
            }

    result = {
        **build_revs(),
        "capture": {k: cap_meta[k] for k in ("actionet_rev", "libactionet_rev", "n_donors", "nb_design_columns",
                                             "n_groups", "timings")},
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "n_donors": n_donors,
        "reduce_kernel_check": reduce_check,
        "arms": arm_meta, "agreement": agree, "quality": quality, "timings": timings,
        "metric_cells": int(n_sub), "expected_same_donor_frac": expected_same_donor, "args": vars(args),
    }
    (workdir / "compare.json").write_text(json.dumps(result, indent=2, default=str))
    (workdir / "compare.md").write_text(render_markdown(result))
    adata.file.close()
    print((workdir / "compare.md").read_text())


def render_markdown(r: dict) -> str:
    lines = [
        "# Grouped batch correction: atlas validation",
        "",
        f"- new build: actionet {r['actionet_rev']}, libactionet {r['libactionet_rev']}",
        f"- capture build: actionet {r['capture']['actionet_rev']}, libactionet {r['capture']['libactionet_rev']}",
        f"- cells {r['n_obs']:,}, genes {r['n_vars']:,}, donors {r['n_donors']}"
        f" (prototype design columns {r['capture']['nb_design_columns']}, crude groups {r['capture']['n_groups']})",
        "",
        "## Agreement",
        "",
        "| comparison | max abs Δσ | max rel Δσ | max angle (°) | top-10 angle (°) |",
        "|---|---|---|---|---|",
    ]
    for name, a in r["agreement"].items():
        top_key = next(k for k in a if k.startswith("max_angle_top"))
        lines.append(f"| {name} | {a['sigma_max_abs_diff']:.3g} | {a['sigma_max_rel_diff']:.3g} | "
                     f"{a['max_angle_deg']:.3g} | {a[top_key]:.3g} |")
    lines += ["", "## Embedding quality", "",
              f"kNN agreement computed on {r['metric_cells']:,} cells. Same-donor fraction expected under "
              f"perfect mixing within crude groups: {r['expected_same_donor_frac']:.3f}.", "",
              "| embedding | depth R² within groups | max abs corr(depth) | " +
              " | ".join(f"kNN same {k}" for k in next(iter(r["quality"].values()))["knn_same_label_frac"]) + " |",
              "|---" * (3 + len(next(iter(r["quality"].values()))["knn_same_label_frac"])) + "|"]
    for name, q in r["quality"].items():
        knn = " | ".join(f"{v:.3f}" for v in q["knn_same_label_frac"].values())
        lines.append(f"| {name} | {q['depth_r2_within_groups']:.3f} | {q['depth_max_abs_corr']:.3f} | {knn} |")
    lines += ["", "## Arms", "", "```json", json.dumps(r["arms"], indent=2, default=str), "```",
              "", "## Timings (s)", "", "| stage | wall | peak RSS (MB) |", "|---|---|---|"]
    for name, t in r["timings"].items():
        lines.append(f"| {name} | {t['wall_s']:.1f} | {t['peak_rss_mb']:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["capture", "compare"])
    p.add_argument("--input", required=True, help="Atlas h5ad (opened backed)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--no-copy", action="store_true", help="Filter the input file in place")
    p.add_argument("--recopy", action="store_true", help="Overwrite an existing working copy")
    p.add_argument("--batch-key", default="Donor")
    p.add_argument("--strata-key", default="Region")
    p.add_argument("--label-key", default=None, help="Cell-type annotation used for preservation metrics")
    p.add_argument("--k", type=int, default=30, help="reduce_kernel components")
    p.add_argument("--min-cells-per-feat", type=float, default=0.001)
    p.add_argument("--group-k", type=int, default=30)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-batch-frac", type=float, default=0.3)
    p.add_argument("--metric-cells", type=int, default=200_000)
    p.add_argument("--metric-k", type=int, default=30)
    p.add_argument("--n-threads", type=int, default=0)
    p.add_argument("--check-reduce-kernel", action="store_true",
                   help="compare: recompute reduce_kernel on the new build and compare to the capture")
    args = p.parse_args(argv)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    capture(args) if args.stage == "capture" else compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
