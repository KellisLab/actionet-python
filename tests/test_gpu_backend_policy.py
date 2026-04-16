#!/usr/bin/env python3
"""Execution-policy tests for Phase 1 GPU backend dispatch."""

import os

import anndata as ad
import numpy as np
import pytest

import actionet as an


def _random_dense_matrix(n_obs: int = 80, n_vars: int = 40, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n_obs, n_vars))


def test_run_svd_gpu_invalid_device_falls_back_to_cpu_dense():
    X = _random_dense_matrix()

    out = an.run_svd(
        X,
        n_components=8,
        algorithm="primme",
        compute_backend="gpu",
        device_id=99999,
        allow_cpu_fallback=True,
        verbose=False,
    )

    assert out["u"].shape == (X.shape[0], 8)
    assert np.asarray(out["d"]).reshape(-1).shape == (8,)
    assert out["v"].shape == (X.shape[1], 8)


def test_run_svd_gpu_invalid_device_without_fallback_raises_dense():
    X = _random_dense_matrix(seed=11)

    with pytest.raises(RuntimeError, match="GPU backend requested"):
        an.run_svd(
            X,
            n_components=8,
            algorithm="primme",
            compute_backend="gpu",
            device_id=99999,
            allow_cpu_fallback=False,
            verbose=False,
        )


def test_run_svd_backed_operator_gpu_policy(tmp_path):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    X = _random_dense_matrix(n_obs=64, n_vars=32, seed=19)
    adata = ad.AnnData(X=X)
    h5ad_path = tmp_path / "gpu_policy_backed.h5ad"
    adata.write_h5ad(h5ad_path)

    backed = ad.read_h5ad(h5ad_path, backed="r+")
    try:
        with pytest.raises(RuntimeError, match="GPU backend requested"):
            an.run_svd(
                backed,
                n_components=6,
                algorithm="primme",
                compute_backend="gpu",
                allow_cpu_fallback=False,
                verbose=False,
            )

        out = an.run_svd(
            backed,
            n_components=6,
            algorithm="primme",
            compute_backend="gpu",
            allow_cpu_fallback=True,
            verbose=False,
        )
        assert out["u"].shape == (X.shape[0], 6)
        assert np.asarray(out["d"]).reshape(-1).shape == (6,)
        assert out["v"].shape == (X.shape[1], 6)
    finally:
        backed.file.close()
