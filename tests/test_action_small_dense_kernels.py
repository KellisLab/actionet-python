"""Correctness and catastrophic-regression coverage for ACTION dense kernels."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from actionet import _core
from actionet.action import run_simplex_regression


RTOL = 1e-8
ATOL = 1e-10


def _assert_simplex_feasible(solution: np.ndarray) -> None:
    assert np.all(np.isfinite(solution))
    assert np.all(solution >= -ATOL)
    assert np.all(solution <= 1.0 + ATOL)
    np.testing.assert_allclose(solution.sum(axis=0), 1.0, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (17, 37),  # skinny inline path
        (128, 131),  # exact inline cutoff
        (129, 130),  # both dimensions exceed the cutoff: CBLAS fallback
    ],
)
def test_cached_and_noncached_simplex_solvers_agree(rows: int, columns: int) -> None:
    rng = np.random.default_rng(20260713 + rows)
    design = np.ascontiguousarray(rng.normal(size=(rows, columns)))
    targets = np.ascontiguousarray(rng.normal(size=(rows, 3)))

    noncached = run_simplex_regression(design, targets, compute_XtX=False)
    cached = run_simplex_regression(design, targets, compute_XtX=True)

    _assert_simplex_feasible(noncached)
    _assert_simplex_feasible(cached)
    np.testing.assert_allclose(noncached, cached, rtol=RTOL, atol=ATOL)


def test_simplex_solvers_handle_duplicate_and_zero_columns() -> None:
    design = np.ascontiguousarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = np.ascontiguousarray(
        [
            [1.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.0],
        ]
    )

    noncached = run_simplex_regression(design, targets, compute_XtX=False)
    cached = run_simplex_regression(design, targets, compute_XtX=True)

    _assert_simplex_feasible(noncached)
    _assert_simplex_feasible(cached)
    np.testing.assert_allclose(noncached, cached, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("landmark_coefficient", "expected_retained"),
    [
        pytest.param(0.0, False, id="exact-zero"),
        pytest.param(1e-18, False, id="roundoff-residue"),
        pytest.param(1e-6, False, id="exact-support-threshold"),
        pytest.param(
            np.nextafter(1e-6, np.inf),
            True,
            id="above-support-threshold",
        ),
    ],
)
def test_landmark_support_uses_meaningful_simplex_coefficients(
    landmark_coefficient: float,
    expected_retained: bool,
) -> None:
    # Each H row has one unambiguous landmark. Archetype zero has meaningful
    # membership on a non-landmark cell, so only its candidate coefficient at
    # cell zero controls the reproducibility decision. This isolates landmark
    # support from the independent trivial-membership filter.
    H_stacked = np.ascontiguousarray(np.eye(4, dtype=np.float64))
    C_stacked = np.zeros((4, 4), dtype=np.float64)
    C_stacked[1, 0] = 0.5
    C_stacked[0, 0] = landmark_coefficient
    C_stacked[1, 1] = 1.0
    C_stacked[2, 2] = 1.0
    C_stacked[3, 3] = 1.0
    C_stacked = np.ascontiguousarray(C_stacked)

    result = _core.collect_archetypes(
        C_stacked,
        H_stacked,
        -np.inf,  # Disable specificity pruning for this focused policy test.
        1,
    )
    retained = np.asarray(result["selected_archs"], dtype=np.int64)

    assert (0 in retained) is expected_retained


_THREAD_PARITY_PROGRAM = textwrap.dedent(
    """
    import sys
    import numpy as np

    from actionet import _core
    from actionet.action import decompose_action, run_archetypal_analysis

    thread_count = int(sys.argv[1])
    output_path = sys.argv[2]
    rng = np.random.default_rng(90210)
    data = np.ascontiguousarray(rng.normal(size=(12, 48)))
    reduction = np.ascontiguousarray(data.T)
    initial = np.ascontiguousarray(data[:, [0, 7, 15, 23]])

    aa = run_archetypal_analysis(
        data, initial, max_iter=8, tolerance=0.0
    )
    decomposition = decompose_action(
        reduction, k_min=2, k_max=5, max_iter=8,
        tolerance=0.0, n_threads=thread_count,
    )
    action = _core.run_action(
        reduction, 2, 5, 8, 0.0, -3.0, 1, thread_count, True
    )

    np.savez_compressed(
        output_path,
        aa_C=aa["C"],
        aa_H=aa["H"],
        decompose_C=decomposition["C_stacked"],
        decompose_H=decomposition["H_stacked"],
        action_C_stacked=action["C_stacked"],
        action_H_stacked=action["H_stacked"],
        action_C_merged=action["C_merged"],
        action_H_merged=action["H_merged"],
        action_assignments=action["assigned_archetypes"],
    )
    """
)


def _run_thread_parity_case(tmp_path: Path, thread_count: int) -> np.lib.npyio.NpzFile:
    output_path = tmp_path / f"action-threads-{thread_count}.npz"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(thread_count)
    # Large BLAS calls are not under test here; keep them serial so the test
    # measures ACTION's own OpenMP decomposition on every backend.
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(
        [sys.executable, "-c", _THREAD_PARITY_PROGRAM, str(thread_count), str(output_path)],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
    )
    return np.load(output_path)


def test_aa_decompose_and_run_action_are_thread_deterministic(tmp_path: Path) -> None:
    single = _run_thread_parity_case(tmp_path, 1)
    multi = _run_thread_parity_case(tmp_path, 4)

    try:
        assert np.array_equal(single["action_assignments"], multi["action_assignments"])
        for key in single.files:
            if key == "action_assignments":
                continue
            np.testing.assert_allclose(single[key], multi[key], rtol=RTOL, atol=ATOL)
    finally:
        single.close()
        multi.close()


@pytest.mark.openblas_smoke
def test_action_synthetic_catastrophic_regression_smoke() -> None:
    if os.environ.get("ACTIONET_RUN_OPENBLAS_SMOKE") != "1":
        pytest.skip("set ACTIONET_RUN_OPENBLAS_SMOKE=1 to run the timing smoke test")

    rng = np.random.default_rng(4815162342)
    reduction = rng.normal(size=(3000, 30))
    reduction /= np.maximum(np.abs(reduction).sum(axis=1, keepdims=True), 1e-15)
    reduction = np.ascontiguousarray(reduction)

    started = time.perf_counter()
    result = _core.run_action(
        reduction, 2, 20, 20, 0.0, -3.0, 2, 8, False
    )
    elapsed = time.perf_counter() - started

    assert result["assigned_archetypes"].shape == (reduction.shape[0],)
    assert elapsed < 30.0, f"synthetic ACTION smoke test took {elapsed:.2f}s"
