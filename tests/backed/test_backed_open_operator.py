"""Unit tests for lock-safe backed-operator opener."""

from __future__ import annotations

import warnings

import pytest

try:
    import actionet.io.operator as backed_io

    _has_actionet = True
except Exception:
    _has_actionet = False


requires_actionet = pytest.mark.skipif(not _has_actionet, reason="actionet extension not built")


class _FakeFileManager:
    """Minimal stand-in for AnnData's h5py file-manager surface.

    Records ``close``/``open`` calls and mimics ``is_open`` / ``_filemode``
    so ``_release_backed_handle`` and ``_restore_backed_handle`` operate on
    it exactly as they would on the real anndata manager.
    """

    def __init__(
        self,
        *,
        filemode: str = "r+",
        raise_on_open: Exception | None = None,
        events: list[str] | None = None,
    ):
        self._filemode = filemode
        self.is_open = True
        self.calls: list[tuple[str, dict]] = []
        self._raise_on_open = raise_on_open
        self._events = events

    def close(self) -> None:
        self.calls.append(("close", {}))
        if self._events is not None:
            self._events.append("manager.close")
        self.is_open = False

    def open(self, *, filemode: str) -> None:
        self.calls.append(("open", {"filemode": filemode}))
        if self._events is not None:
            self._events.append("manager.open")
        if self._raise_on_open is not None:
            raise self._raise_on_open
        self._filemode = filemode
        self.is_open = True

    def flush(self) -> None:
        self.calls.append(("flush", {}))


class _FakeOperator:
    """Operator whose explicit close can be observed after a retained yield."""

    def __init__(self, *, events: list[str] | None = None):
        self.is_open = True
        self._events = events

    def close(self) -> None:
        if self._events is not None:
            self._events.append("operator.close")
        self.is_open = False


class _FakeBackedAdata:
    """Minimal stand-in for a backed AnnData with a fake file manager."""

    def __init__(self, manager: _FakeFileManager):
        self.file = manager
        self.isbacked = True
        # Never accessed inside `_open_backed_operator`; touching this
        # attribute inside a `with` block is a safety-envelope violation.
        self._backed_x_guard_tripped = False

    @property
    def X(self):  # noqa: N802 - AnnData API
        # Any read inside the `with` block trips this; used by
        # test_open_backed_operator_does_not_touch_adata_X.
        self._backed_x_guard_tripped = True
        raise RuntimeError("adata.X must not be accessed while backed handle is released")


@requires_actionet
def test_open_backed_operator_retry_then_success(tmp_path, monkeypatch):
    src = tmp_path / "retry_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    calls: list[str] = []
    sentinel = _FakeOperator()

    def fake_create(**kwargs):
        calls.append(kwargs["file_path"])
        if len(calls) == 1:
            raise RuntimeError("createBackedOperator: failed to open h5ad file: simulated lock")
        return sentinel

    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    with backed_io._open_backed_operator(
        adata=None,
        file_path=str(src),
        group_path="/X",
        context="unit_retry",
        chunk_size=32,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    ) as op:
        assert op is sentinel
        assert calls == [str(src), str(src)]


@requires_actionet
def test_open_backed_operator_retries_exhausted_raises(tmp_path, monkeypatch):
    """After every retry fails with a lock error, no shadow copy is created."""
    src = tmp_path / "exhausted_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    attempts: list[str] = []

    def fake_create(**kwargs):
        attempts.append(str(kwargs["file_path"]))
        raise RuntimeError("createBackedOperator: failed to open h5ad file: simulated lock")

    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    with pytest.raises(RuntimeError, match=r"failed to open backed operator .* lock-open retries"):
        with backed_io._open_backed_operator(
            adata=None,
            file_path=str(src),
            group_path="/X",
            context="unit_exhausted",
            chunk_size=32,
            retry_attempts=2,
            retry_backoff_seconds=0.0,
        ):
            pytest.fail("context body must not execute when open never succeeds")

    # Every attempt targeted the original path — no shadow-file was created.
    assert attempts == [str(src), str(src)]
    residual = list(src.parent.glob("actionet_lock_fallback_*.h5ad"))
    assert residual == [], f"no shadow copies expected, found {residual}"


@requires_actionet
def test_open_backed_operator_non_lock_error_fails_fast(tmp_path, monkeypatch):
    src = tmp_path / "hard_fail_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    calls: list[str] = []

    def fake_create(**kwargs):
        calls.append(kwargs["file_path"])
        raise ValueError("invalid group path")

    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    with pytest.raises(ValueError, match="invalid group path"):
        with backed_io._open_backed_operator(
            adata=None,
            file_path=str(src),
            group_path="/X",
            context="unit_hard_fail",
            chunk_size=32,
            retry_attempts=4,
            retry_backoff_seconds=0.0,
        ):
            pass

    # Fails fast on the first attempt; no retries.
    assert len(calls) == 1


@requires_actionet
def test_open_backed_operator_closes_and_restores_h5py_handle(tmp_path, monkeypatch):
    """h5py handle is closed before the C++ open and reopened with the saved mode."""
    src = tmp_path / "close_reopen_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    events: list[str] = []
    manager = _FakeFileManager(filemode="r+", events=events)
    adata = _FakeBackedAdata(manager)

    sentinel = _FakeOperator(events=events)
    manager_state_at_create: list[bool] = []

    def fake_create(**kwargs):
        events.append("operator.create")
        # Snapshot manager.is_open at the moment of C++ open — it must be closed.
        manager_state_at_create.append(manager.is_open)
        return sentinel

    # _release_backed_handle -> _flush_backed_handle -> BackedAnnDataAdapter,
    # which touches the real anndata surface. Bypass that layer for this
    # unit test by patching the private flush helper directly.
    monkeypatch.setattr(backed_io, "_flush_backed_handle", lambda adata, *, context: None)
    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    with backed_io._open_backed_operator(
        adata=adata,
        file_path=str(src),
        group_path="/X",
        context="unit_close_reopen",
        chunk_size=32,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    ) as op:
        assert op is sentinel
        assert manager.is_open is False, "h5py handle must be closed while op is active"

    # After exit, close then open must have been called exactly once each,
    # and the open must use the original filemode.
    op_calls = [name for name, _ in manager.calls]
    assert op_calls == ["close", "open"], f"unexpected manager calls: {manager.calls}"
    assert manager.calls[1] == ("open", {"filemode": "r+"})
    assert manager.is_open is True
    assert manager_state_at_create == [False]
    assert op is sentinel, "the caller intentionally retains the yielded wrapper"
    assert sentinel.is_open is False, "native handles must close despite that retained wrapper"
    assert events == [
        "manager.close",
        "operator.create",
        "operator.close",
        "manager.open",
    ]


@requires_actionet
def test_open_backed_operator_closes_native_handles_before_restore(tmp_path):
    """The real pybind operator is unusable after exit while AnnData is restored."""
    import anndata as ad
    import numpy as np

    from actionet import _core

    src = tmp_path / "native_close_source.h5ad"
    matrix = np.arange(12, dtype=np.float64).reshape(3, 4)
    ad.AnnData(matrix).write_h5ad(src)
    adata = ad.read_h5ad(src, backed="r+")

    try:
        with backed_io._open_backed_operator(
            adata=adata,
            file_path=str(src),
            group_path="/X",
            context="unit_native_close",
            chunk_size=2,
            retry_attempts=1,
            retry_backoff_seconds=0.0,
        ) as op:
            selected = _core.backed_take_columns(
                op,
                np.array([1], dtype=np.int64),
                prefer_sparse=False,
            )
            np.testing.assert_array_equal(selected[:, 0], matrix[:, 1])

        assert adata.file.is_open is True
        with pytest.raises(RuntimeError):
            _core.backed_take_columns(
                op,
                np.array([1], dtype=np.int64),
                prefer_sparse=False,
            )
    finally:
        adata.file.close()


@requires_actionet
def test_open_backed_operator_restores_handle_on_exception(tmp_path, monkeypatch):
    """Even when the C++ open raises, the h5py handle is restored."""
    src = tmp_path / "raise_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    manager = _FakeFileManager(filemode="r")
    adata = _FakeBackedAdata(manager)

    def fake_create(**kwargs):
        raise ValueError("simulated non-lock failure")

    monkeypatch.setattr(backed_io, "_flush_backed_handle", lambda adata, *, context: None)
    monkeypatch.setattr(backed_io, "_create_backed_operator", fake_create)

    with pytest.raises(ValueError, match="simulated non-lock failure"):
        with backed_io._open_backed_operator(
            adata=adata,
            file_path=str(src),
            group_path="/X",
            context="unit_raise",
            chunk_size=32,
            retry_attempts=1,
            retry_backoff_seconds=0.0,
        ):
            pytest.fail("body must not execute when C++ open raises")

    op_calls = [name for name, _ in manager.calls]
    assert op_calls == ["close", "open"]
    assert manager.calls[1] == ("open", {"filemode": "r"})
    assert manager.is_open is True


@requires_actionet
def test_open_backed_operator_reopen_failure_warns_not_raises(tmp_path, monkeypatch):
    """A restore-time failure surfaces as a UserWarning; caller result is not masked."""
    src = tmp_path / "reopen_fail_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    manager = _FakeFileManager(
        filemode="r+",
        raise_on_open=OSError("simulated reopen failure"),
    )
    adata = _FakeBackedAdata(manager)

    sentinel = _FakeOperator()
    monkeypatch.setattr(backed_io, "_flush_backed_handle", lambda adata, *, context: None)
    monkeypatch.setattr(backed_io, "_create_backed_operator", lambda **kwargs: sentinel)

    yielded = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with backed_io._open_backed_operator(
            adata=adata,
            file_path=str(src),
            group_path="/X",
            context="unit_reopen_fail",
            chunk_size=32,
            retry_attempts=1,
            retry_backoff_seconds=0.0,
        ) as op:
            yielded = op

    assert yielded is sentinel, "caller must still receive the operator"
    matching = [w for w in caught if "failed to reopen backed AnnData handle" in str(w.message)]
    assert len(matching) == 1, f"expected one reopen-failure warning, got {[str(w.message) for w in caught]}"
    assert matching[0].category is UserWarning
    assert "unit_reopen_fail" in str(matching[0].message)
    assert "'r+'" in str(matching[0].message)


@requires_actionet
def test_open_backed_operator_skips_release_when_adata_none(tmp_path, monkeypatch):
    """No release/restore machinery runs when adata is None."""
    src = tmp_path / "no_adata.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    sentinel = _FakeOperator()
    flush_calls: list = []

    def fake_flush(adata, *, context):
        flush_calls.append((adata, context))

    monkeypatch.setattr(backed_io, "_flush_backed_handle", fake_flush)
    monkeypatch.setattr(backed_io, "_create_backed_operator", lambda **kwargs: sentinel)

    with backed_io._open_backed_operator(
        adata=None,
        file_path=str(src),
        group_path="/X",
        context="unit_no_adata",
        chunk_size=32,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    ) as op:
        assert op is sentinel

    assert flush_calls == []


@requires_actionet
def test_open_backed_operator_does_not_touch_adata_X(tmp_path, monkeypatch):
    """Documenting invariant: no path inside the context should read adata.X.

    _FakeBackedAdata.X raises immediately. If any future edit makes
    _open_backed_operator poke adata.X (directly or via a lazy accessor) while
    the h5py handle is released, this test catches it.
    """
    src = tmp_path / "guard_source.h5ad"
    src.write_bytes(b"dummy h5ad payload")

    manager = _FakeFileManager(filemode="r")
    adata = _FakeBackedAdata(manager)

    monkeypatch.setattr(backed_io, "_flush_backed_handle", lambda adata, *, context: None)
    monkeypatch.setattr(backed_io, "_create_backed_operator", lambda **kwargs: _FakeOperator())

    with backed_io._open_backed_operator(
        adata=adata,
        file_path=str(src),
        group_path="/X",
        context="unit_no_backed_X",
        chunk_size=32,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    ):
        pass

    assert adata._backed_x_guard_tripped is False
