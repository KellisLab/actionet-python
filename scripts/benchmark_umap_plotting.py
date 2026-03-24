#!/usr/bin/env python3
"""Manual benchmark for ACTIONet static UMAP rendering modes."""

from __future__ import annotations

import argparse
import io
import os
import time

import numpy as np
import pandas as pd
from anndata import AnnData

import actionet as act


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


def _build_adata(n_obs: int, seed: int) -> AnnData:
    rng = np.random.default_rng(seed)
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.var_names = pd.Index(["dummy"])
    adata.obs["cluster"] = pd.Categorical(rng.choice(["a", "b", "c", "d"], size=n_obs))
    adata.obs["score"] = rng.normal(size=n_obs).astype(np.float32)
    adata.obsm["umap_2d_actionet"] = rng.normal(size=(n_obs, 2)).astype(np.float32)
    adata.obsm["colors_actionet"] = rng.random(size=(n_obs, 3)).astype(np.float32)
    return adata


def _timed(label: str, fn) -> tuple[str, float]:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    return label, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=250_000, help="Number of synthetic cells.")
    parser.add_argument("--seed", type=int, default=37, help="Random seed.")
    args = parser.parse_args()

    from lets_plot import LetsPlot

    LetsPlot.setup_html(no_js=True)

    adata = _build_adata(args.n_obs, args.seed)
    results: list[tuple[str, float]] = []

    results.append(
        _timed(
            "lets_plot_spec",
            lambda: act.plot_umap(adata, color="cluster"),
        )
    )

    def _render_lets_plot_png() -> None:
        buf = io.BytesIO()
        act.plot_umap(adata, color="cluster").to_png(buf)

    results.append(_timed("lets_plot_png", _render_lets_plot_png))

    def _render_raster_png() -> None:
        buf = io.BytesIO()
        act.plot_umap_raster(adata, color="cluster").savefig(buf, format="png")

    results.append(_timed("raster_png", _render_raster_png))

    for label, elapsed in results:
        print(f"{label}: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
