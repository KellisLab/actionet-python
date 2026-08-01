"""Failure injection for the shared H5AD rewrite transaction."""

from pathlib import Path

import pytest

from actionet.io.rewrite import RewriteTransaction


def test_replace_failure_restores_source_handle_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"original")
    lifecycle: list[str] = []
    temp_path: Path | None = None

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("actionet.io.rewrite.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        with RewriteTransaction(str(source), str(source)) as transaction:
            temp_path = Path(transaction.temp_path)
            temp_path.write_bytes(b"replacement")
            transaction.commit(
                close_source=lambda: lifecycle.append("close"),
                restore_source=lambda: lifecycle.append("restore"),
            )

    assert source.read_bytes() == b"original"
    assert lifecycle == ["close", "restore"]
    assert temp_path is not None and not temp_path.exists()


def test_close_failure_attempts_restore_and_preserves_source(tmp_path):
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"original")
    lifecycle: list[str] = []
    temp_path: Path | None = None

    def fail_close():
        lifecycle.append("close")
        raise OSError("synthetic close failure")

    with pytest.raises(OSError, match="synthetic close failure"):
        with RewriteTransaction(str(source), str(source)) as transaction:
            temp_path = Path(transaction.temp_path)
            temp_path.write_bytes(b"replacement")
            transaction.commit(
                close_source=fail_close,
                restore_source=lambda: lifecycle.append("restore"),
            )

    assert source.read_bytes() == b"original"
    assert lifecycle == ["close", "restore"]
    assert temp_path is not None and not temp_path.exists()


def test_fingerprint_change_refuses_commit_before_close(tmp_path):
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"original")
    closed = False
    temp_path: Path | None = None

    with pytest.raises(RuntimeError, match="changed during rewrite"):
        with RewriteTransaction(str(source), str(source)) as transaction:
            temp_path = Path(transaction.temp_path)
            temp_path.write_bytes(b"replacement")
            source.write_bytes(b"external-change")

            def close_source():
                nonlocal closed
                closed = True

            transaction.commit(close_source=close_source)

    assert source.read_bytes() == b"external-change"
    assert not closed
    assert temp_path is not None and not temp_path.exists()
