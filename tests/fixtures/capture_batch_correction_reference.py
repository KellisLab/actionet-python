#!/usr/bin/env python3
"""Capture pre-change ``correct_batch_effect`` outputs for the grouped-design regression test.

Run once against the build that produced the validated prototype results::

    python tests/fixtures/capture_batch_correction_reference.py

Writes ``tests/fixtures/batch_correction_grouped_reference.npz`` containing the
synthetic counts and labels, the ``reduce_kernel`` state, and the outputs of
``correct_batch_effect`` on the prototype designs for 20 batches (QR path in
``perturbedSVD``) and 8 batches (Gram-Schmidt path, where the near-zero
intercept column discarded the cell-side residual). The reduction is stored so
the regression test is insensitive to BLAS-dependent ``reduce_kernel`` roundoff.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import actionet  # noqa: E402
from _batch_design_reference import (  # noqa: E402
    adata_from_counts,
    notebook_make_batch_design,
    synthetic_counts,
)

OUTPUT = HERE / "batch_correction_grouped_reference.npz"
N_COMPONENTS = 10


def _correct(adata, design):
    work = adata.copy()
    actionet.correct_batch_effect(work, design=design, reduction_key="action")
    return {
        "S_r": np.asarray(work.obsm["action_corrected"]),
        "U": np.asarray(work.varm["action_corrected_U"]),
        "sigma": np.asarray(work.uns["action_corrected_params"]["sigma"]).ravel(),
    }


def main() -> None:
    counts, cluster, stratum, batch = synthetic_counts()
    adata = adata_from_counts(counts, cluster, stratum, batch)
    adata.obs["donor8"] = adata.obs["donor"].cat.codes.to_numpy() % 8
    adata.obs["donor8"] = adata.obs["donor8"].astype(str).astype("category")
    actionet.reduce_kernel(adata, n_components=N_COMPONENTS, key_added="action", verbose=False)

    payload = {}
    rev = subprocess.run(
        ["git", "-C", str(HERE / ".." / ".." / "src" / "libactionet"), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    payload["libactionet_rev"] = np.asarray(rev or "unknown")
    csr = sp.csr_matrix(counts)
    payload.update(
        counts_data=csr.data.astype(np.int32),
        counts_indices=csr.indices.astype(np.int32),
        counts_indptr=csr.indptr.astype(np.int64),
        counts_shape=np.asarray(csr.shape, dtype=np.int64),
        cluster=cluster.astype(np.int32),
        stratum=stratum.astype(np.int32),
        batch=batch.astype(np.int32),
        red_S_r=np.asarray(adata.obsm["action"]),
        red_U=np.asarray(adata.varm["action_U"]),
        red_A=np.asarray(adata.varm["action_A"]),
        red_B=np.asarray(adata.obsm["action_B"]),
        red_sigma=np.asarray(adata.uns["action_params"]["sigma"]).ravel(),
    )

    for tag, batch_key in (("b20", "donor"), ("b8", "donor8")):
        design = notebook_make_batch_design(
            adata, cluster_key="cluster", batch_key=batch_key, strata_key="region"
        )
        out = _correct(adata, design)
        payload[f"{tag}_n_design_cols"] = np.int64(design.shape[1])
        for name, value in out.items():
            payload[f"{tag}_{name}"] = value

    plain = adata.copy()
    actionet.correct_batch_effect(plain, batch_key="donor", reduction_key="action")
    payload["plain_sigma"] = np.asarray(plain.uns["action_corrected_params"]["sigma"]).ravel()

    np.savez_compressed(OUTPUT, **payload)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")
    for tag in ("b20", "b8"):
        print(tag, "design cols:", int(payload[f"{tag}_n_design_cols"]), "sigma:", np.round(payload[f"{tag}_sigma"], 4))


if __name__ == "__main__":
    main()
