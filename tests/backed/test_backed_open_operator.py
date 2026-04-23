"""Unit tests for lock-safe backed-operator opener."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import actionet.backed_io as backed_io

    _has_actionet = True
except Exception:
    _has_actionet = False


requires_actionet = pytest.mark.skipif(not _has_actionet, reason="actionet extension not built")


@requires_actionet
def test_open_backed_operator_retry_then_success(tmp_path, monkeypatch):
    src = tmp_path / "retry_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    calls: list[str] = []
    sentinel = object()

    def fake_create(**kwargs):
        calls.append(kwargs["file_path"])
        if len(calls) == 1:
            raise RuntimeError("createBackedOperator: failed to open h5ad file: simulated lock")
        return sentinel

    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    def _no_fallback(*args, **kwargs):
        raise AssertionError("fallback copy should not be created when retry succeeds")

    monkeypatch.setattr(backed_io.tempfile, "mkstemp", _no_fallback)

    op, cleanup = backed_io._open_backed_operator(
        adata=None,
        file_path=str(src),
        group_path="/X",
        context="unit_retry",
        chunk_size=32,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )
    try:
        assert op is sentinel
        assert calls == [str(src), str(src)]
    finally:
        cleanup()


@requires_actionet
def test_open_backed_operator_fallback_copy_then_cleanup(tmp_path, monkeypatch):
    src = tmp_path / "fallback_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    create_calls: list[str] = []

    def fake_create(**kwargs):
        file_path = str(kwargs["file_path"])
        create_calls.append(file_path)
        if file_path == str(src):
            raise RuntimeError("createBackedOperator: failed to open h5ad file: simulated lock")
        return {"file_path": file_path}

    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    op, cleanup = backed_io._open_backed_operator(
        adata=None,
        file_path=str(src),
        group_path="/X",
        context="unit_fallback",
        chunk_size=32,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )

    fallback_paths = [Path(p) for p in create_calls if p != str(src)]
    assert len(fallback_paths) == 1

    fallback_path = fallback_paths[0]
    assert fallback_path.parent == src.parent
    assert fallback_path.exists()
    assert op["file_path"] == str(fallback_path)

    cleanup()
    assert not fallback_path.exists()


@requires_actionet
def test_open_backed_operator_non_lock_error_fails_fast(tmp_path, monkeypatch):
    src = tmp_path / "hard_fail_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    def fake_create(**kwargs):
        raise ValueError("invalid group path")

    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    def _no_fallback(*args, **kwargs):
        raise AssertionError("fallback should not be attempted for non-lock errors")

    monkeypatch.setattr(backed_io.tempfile, "mkstemp", _no_fallback)

    with pytest.raises(ValueError, match="invalid group path"):
        backed_io._open_backed_operator(
            adata=None,
            file_path=str(src),
            group_path="/X",
            context="unit_hard_fail",
            chunk_size=32,
            retry_attempts=4,
            retry_backoff_seconds=0.0,
        )
