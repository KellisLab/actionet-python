"""Backed reduce_kernel smoke test (originally tests/test_backed.py)."""

import numpy as np
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import actionet as an

from .conftest import make_test_adata, open_backed


def test_backed_reduce_kernel_smoke(tmp_path):
    """reduce_kernel runs on a backed AnnData without error."""
    adata = open_backed(tmp_path, make_test_adata(sparse_fmt="csr", seed=99))

    an.normalize_anndata(adata, target_sum=1e4, backed_chunk_size=32, inplace=True)

    an.reduce_kernel(
        adata,
        n_components=10,
        layer=None,
        key_added="action",
        backed_chunk_size=32,
        inplace=True,
    )

    assert "action" in adata.obsm
    assert adata.obsm["action"].shape == (adata.n_obs, 10)
    assert not np.any(np.isnan(adata.obsm["action"]))
