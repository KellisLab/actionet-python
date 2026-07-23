"""Tests for independent backed compute and write chunk controls."""

from __future__ import annotations

import contextlib
import importlib

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from actionet.io.chunking import (
    DEFAULT_BACKED_READ_CHUNK_SIZE,
    DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    resolve_backed_write_chunk_size,
    validate_chunk_size,
)

from .conftest import make_test_adata, open_backed

svd_module = importlib.import_module("actionet.decomposition.svd")
kernel_module = importlib.import_module("actionet.decomposition.kernel")
filter_module = importlib.import_module("actionet.preprocessing.filter")
normalize_module = importlib.import_module("actionet.preprocessing.normalize")


@pytest.mark.parametrize("value", [0, -1])
def test_validate_chunk_size_rejects_nonpositive(value):
    with pytest.raises(ValueError, match="must be > 0"):
        validate_chunk_size(value, name="backed_chunk_size")


@pytest.mark.parametrize("value", [True, 1.5, "4096"])
def test_validate_chunk_size_rejects_nonintegers(value):
    with pytest.raises(TypeError, match="positive integer"):
        validate_chunk_size(value, name="backed_chunk_size")


def test_write_chunk_none_uses_independent_write_default():
    """None for the write chunk falls back to the shared write default,
    not to the read/compute chunk value."""
    assert resolve_backed_write_chunk_size(4096, None) == (
        4096,
        DEFAULT_BACKED_WRITE_CHUNK_SIZE,
    )
    assert resolve_backed_write_chunk_size(4096, 32768) == (4096, 32768)


def test_default_chunk_sizes_are_read_8192_write_16384():
    """Regression guard: the shared Python defaults are read=8192 and
    write=16384."""
    assert DEFAULT_BACKED_READ_CHUNK_SIZE == 8192
    assert DEFAULT_BACKED_WRITE_CHUNK_SIZE == 16384


@pytest.mark.parametrize(
    "call",
    [
        lambda obj, value: svd_module.run_svd(
            np.eye(4),
            backed_chunk_size=value,
            verbose=False,
        ),
        lambda obj, value: kernel_module.reduce_kernel(
            obj,
            n_components=2,
            backed_write_chunk_size=value,
            verbose=False,
        ),
        lambda obj, value: filter_module.filter_anndata(
            obj,
            backed_write_chunk_size=value,
            filter_adata=False,
        ),
        lambda obj, value: normalize_module.normalize_anndata(
            obj,
            backed_write_chunk_size=value,
        ),
    ],
)
@pytest.mark.parametrize("value", [0, -4])
def test_public_apis_reject_nonpositive_chunks(call, value):
    obj = ad.AnnData(np.eye(4))
    with pytest.raises(ValueError, match="must be > 0"):
        call(obj, value)


@pytest.mark.parametrize("write_chunk_size", [None, 53])
def test_run_svd_routes_compute_and_write_chunks(
    tmp_path,
    monkeypatch,
    write_chunk_size,
):
    backed = open_backed(
        tmp_path,
        make_test_adata(n_cells=16, n_genes=12, seed=101),
    )
    observed: dict[str, int] = {}

    def fake_decompress(*args, **kwargs):
        observed["write"] = kwargs["write_chunk_size"]
        return None

    @contextlib.contextmanager
    def fake_open(*args, **kwargs):
        observed["compute"] = kwargs["chunk_size"]
        yield object()

    def fake_run(*args, **kwargs):
        return {
            "u": np.zeros((backed.n_obs, 2)),
            "d": np.ones(2),
            "v": np.zeros((backed.n_vars, 2)),
        }

    monkeypatch.setattr(svd_module, "_maybe_decompress_backed_path", fake_decompress)
    monkeypatch.setattr(svd_module, "open_backed_operator_for", fake_open)
    monkeypatch.setattr(svd_module._core, "run_svd_backed_operator", fake_run)

    svd_module.run_svd(
        backed,
        n_components=2,
        backed_chunk_size=17,
        backed_write_chunk_size=write_chunk_size,
        verbose=False,
    )

    assert observed == {
        "compute": 17,
        "write": DEFAULT_BACKED_WRITE_CHUNK_SIZE if write_chunk_size is None else write_chunk_size,
    }
    backed.file.close()


@pytest.mark.parametrize("write_chunk_size", [None, 59])
def test_reduce_kernel_routes_compute_and_write_chunks(
    tmp_path,
    monkeypatch,
    write_chunk_size,
):
    backed = open_backed(
        tmp_path,
        make_test_adata(n_cells=16, n_genes=12, seed=102),
    )
    observed: dict[str, int] = {}

    def fake_decompress(*args, **kwargs):
        observed["write"] = kwargs["write_chunk_size"]
        return None

    @contextlib.contextmanager
    def fake_open(*args, **kwargs):
        observed["compute"] = kwargs["chunk_size"]
        yield object()

    def fake_reduce(*args, **kwargs):
        return {
            "S_r": np.zeros((backed.n_obs, 2)),
            "B": np.zeros((2, 2)),
            "U": np.zeros((backed.n_vars, 2)),
            "A": np.zeros((2, 2)),
            "sigma": np.ones(2),
        }

    monkeypatch.setattr(kernel_module, "_maybe_decompress_backed_path", fake_decompress)
    monkeypatch.setattr(kernel_module, "open_backed_operator_for", fake_open)
    monkeypatch.setattr(kernel_module._core, "reduce_kernel_backed_operator", fake_reduce)
    monkeypatch.setattr(kernel_module, "persist_updates", lambda *args, **kwargs: None)

    kernel_module.reduce_kernel(
        backed,
        n_components=2,
        backed_chunk_size=19,
        backed_write_chunk_size=write_chunk_size,
        verbose=False,
    )

    assert observed == {
        "compute": 19,
        "write": DEFAULT_BACKED_WRITE_CHUNK_SIZE if write_chunk_size is None else write_chunk_size,
    }
    backed.file.close()


def test_reduce_kernel_from_svd_forwards_write_chunk(monkeypatch):
    observed = {}

    def fake_reduce_kernel(**kwargs):
        observed.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(kernel_module, "reduce_kernel", fake_reduce_kernel)
    result = kernel_module.reduce_kernel_from_svd(
        ad.AnnData(np.eye(4)),
        {"u": np.eye(4, 2), "d": np.ones(2), "v": np.eye(4, 2)},
        backed_chunk_size=23,
        backed_write_chunk_size=61,
    )

    assert result == "sentinel"
    assert observed["backed_chunk_size"] == 23
    assert observed["backed_write_chunk_size"] == 61


@pytest.mark.parametrize("write_chunk_size", [None, 67])
def test_filter_routes_compute_and_write_chunks(monkeypatch, write_chunk_size):
    adata = ad.AnnData(np.eye(4))
    observed = {}

    def fake_masks(*args, **kwargs):
        observed["compute"] = kwargs["backed_chunk_size"]
        return np.ones(adata.n_obs, dtype=bool), np.ones(adata.n_vars, dtype=bool)

    def fake_apply(*args, **kwargs):
        observed["write"] = kwargs["backed_write_chunk_size"]
        return None

    monkeypatch.setattr(filter_module, "compute_filter_masks", fake_masks)
    monkeypatch.setattr(filter_module, "apply_filter", fake_apply)

    filter_module.filter_anndata(
        adata,
        backed_chunk_size=29,
        backed_write_chunk_size=write_chunk_size,
    )

    assert observed == {
        "compute": 29,
        "write": DEFAULT_BACKED_WRITE_CHUNK_SIZE if write_chunk_size is None else write_chunk_size,
    }


@pytest.mark.parametrize("write_chunk_size", [None, 71])
def test_normalize_routes_compute_and_write_chunks(monkeypatch, write_chunk_size):
    adata = ad.AnnData(np.eye(4))
    fake_source = type(
        "FakeSource",
        (),
        {"is_backed": True, "is_sparse": False},
    )()
    observed = {}

    monkeypatch.setattr(normalize_module, "MatrixSource", lambda *args, **kwargs: fake_source)

    def fake_normalize(*args, **kwargs):
        observed["compute"] = kwargs["read_chunk_size"]
        observed["write"] = kwargs["write_chunk_size"]

    monkeypatch.setattr(normalize_module, "_normalize_backed", fake_normalize)

    normalize_module.normalize_anndata(
        adata,
        backed_chunk_size=31,
        backed_write_chunk_size=write_chunk_size,
    )

    assert observed == {
        "compute": 31,
        "write": DEFAULT_BACKED_WRITE_CHUNK_SIZE if write_chunk_size is None else write_chunk_size,
    }


@pytest.mark.parametrize("sparse_fmt", ["csr", "csc"])
def test_normalize_write_override_preserves_output(tmp_path, sparse_fmt):
    source = make_test_adata(
        n_cells=30,
        n_genes=21,
        sparse_fmt=sparse_fmt,
        seed=103,
    )
    legacy = open_backed(tmp_path / "legacy", source.copy())
    split = open_backed(tmp_path / "split", source.copy())

    normalize_module.normalize_anndata(
        legacy,
        backed_chunk_size=7,
        log_transform=True,
        inplace=True,
    )
    normalize_module.normalize_anndata(
        split,
        backed_chunk_size=7,
        backed_write_chunk_size=19,
        log_transform=True,
        inplace=True,
    )

    legacy_x = legacy.X.to_memory()
    split_x = split.X.to_memory()
    np.testing.assert_allclose(
        legacy_x.toarray() if sp.issparse(legacy_x) else legacy_x,
        split_x.toarray() if sp.issparse(split_x) else split_x,
        rtol=1e-6,
        atol=1e-7,
    )
    legacy.file.close()
    split.file.close()


def test_filter_write_override_preserves_output(tmp_path):
    source = make_test_adata(n_cells=36, n_genes=24, seed=104)
    legacy = open_backed(tmp_path / "legacy", source.copy())
    split = open_backed(tmp_path / "split", source.copy())

    filter_module.filter_anndata(
        legacy,
        min_cells_per_feat=3,
        backed_chunk_size=7,
        inplace=True,
    )
    filter_module.filter_anndata(
        split,
        min_cells_per_feat=3,
        backed_chunk_size=7,
        backed_write_chunk_size=19,
        inplace=True,
    )

    legacy_x = legacy.X.to_memory()
    split_x = split.X.to_memory()
    np.testing.assert_array_equal(legacy.obs_names, split.obs_names)
    np.testing.assert_array_equal(legacy.var_names, split.var_names)
    np.testing.assert_allclose(legacy_x.toarray(), split_x.toarray())
    legacy.file.close()
    split.file.close()


def test_svd_write_override_preserves_compressed_result(tmp_path):
    source = make_test_adata(n_cells=36, n_genes=24, seed=105)
    paths = [tmp_path / "legacy.h5ad", tmp_path / "split.h5ad"]
    for path in paths:
        source.write_h5ad(path, compression="gzip")

    legacy = ad.read_h5ad(paths[0], backed="r")
    split = ad.read_h5ad(paths[1], backed="r")
    legacy_result = svd_module.run_svd(
        legacy,
        n_components=4,
        seed=7,
        backed_chunk_size=7,
        verbose=False,
    )
    split_result = svd_module.run_svd(
        split,
        n_components=4,
        seed=7,
        backed_chunk_size=7,
        backed_write_chunk_size=19,
        verbose=False,
    )

    for key in ("u", "d", "v"):
        np.testing.assert_allclose(legacy_result[key], split_result[key], rtol=1e-10, atol=1e-12)
    legacy.file.close()
    split.file.close()
