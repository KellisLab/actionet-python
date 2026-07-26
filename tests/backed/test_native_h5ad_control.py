"""Rollback-switch and native preflight behavior."""

import numpy as np
import pytest

from actionet import _core
from actionet.io.backed_adapter import BackedMatrixLocation
from actionet.io.native_h5ad import (
    NativeCapabilityError,
    backed_io_engine,
    plan_native_subset,
)


def test_backed_io_engine_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("ACTIONET_BACKED_IO_ENGINE", "surprise")
    with pytest.raises(ValueError, match="auto, native, or python"):
        backed_io_engine()


def test_python_engine_skips_native_inspection(monkeypatch):
    monkeypatch.setenv("ACTIONET_BACKED_IO_ENGINE", "python")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("native inspection should not run")

    monkeypatch.setattr(_core, "h5ad_inspect_matrix", fail_if_called)
    location = BackedMatrixLocation("/source.h5ad", "/X")
    assert (
        plan_native_subset(
            location,
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
        is None
    )


@pytest.mark.parametrize(
    ("engine", "raises"),
    [("auto", False), ("native", True)],
)
def test_filter_capability_is_rejected_during_preflight(
    monkeypatch,
    engine,
    raises,
):
    monkeypatch.setenv("ACTIONET_BACKED_IO_ENGINE", engine)
    monkeypatch.setattr(
        _core,
        "h5ad_inspect_matrix",
        lambda *_args: {
            "encoding": "csr",
            "datasets": [
                {
                    "filters": [
                        {
                            "id": 32001,
                            "name": "external",
                            "decode_available": True,
                            "encode_available": False,
                        }
                    ]
                }
            ],
        },
    )
    location = BackedMatrixLocation("/source.h5ad", "/X")

    def call():
        return plan_native_subset(
            location,
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )

    if raises:
        with pytest.raises(NativeCapabilityError, match="no encoder"):
            call()
    else:
        assert call() is None
