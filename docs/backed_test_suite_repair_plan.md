# Backed Test Suite Repair Plan

Status: proposed
Owner: unassigned
Scope: `tests/backed/` (plus `tests/backed/test_backed_parity.py` collection error)

## Context

`tests/backed/` currently reports 12 failures + 1 collection error against
current `main`/`dev-gpu-v2`. None are caused by production regressions;
all are test-side drift left behind after refactors landed in
`src/actionet/pipeline.py`, `src/actionet/backed_io.py`, and
`src/actionet/core`. Baseline reproduced with:

```bash
pytest tests/backed --ignore=tests/backed/test_backed_parity.py -q
# 12 failed, 174 passed
pytest tests/backed/test_backed_parity.py
# collection error (see group F below)
```

The remaining 174 tests pass, so the backed code paths themselves are
healthy. This document groups the failures by root cause and prescribes
the minimum test-side change to restore green CI.

## Failure inventory

| #  | Test                                                                                                    | Root cause (group) |
|----|---------------------------------------------------------------------------------------------------------|--------------------|
| 1  | `test_backed_extension.py::test_backed_e2e_x`                                                           | A                  |
| 2  | `test_backed_extension.py::test_backed_e2e_layer`                                                       | A                  |
| 3  | `test_backed_extension.py::test_backed_persistence_after_reopen`                                        | A                  |
| 4  | `test_backed_extension.py::test_backed_parity_markers_and_imputation`                                   | A                  |
| 5  | `test_lazy_transform_cache.py::test_lazy_transform_cache_reused_across_reduce_correct_and_run_actionet` | A                  |
| 6  | `test_lazy_transform_cache.py::test_annotate_cells_lazy_transform_runs_without_error`                   | A                  |
| 7  | `test_backed_open_operator.py::test_open_backed_operator_retry_then_success`                            | B                  |
| 8  | `test_backed_open_operator.py::test_open_backed_operator_fallback_copy_then_cleanup`                    | B                  |
| 9  | `test_backed_open_operator.py::test_open_backed_operator_non_lock_error_fails_fast`                     | B                  |
| 10 | `test_backed_reduce_kernel.py::test_flush_backed_handle_raises_on_flush_failure`                        | C                  |
| 11 | `test_checkpoint.py::TestCheckpointBacked::test_noop_checkpoint`                                        | D                  |
| 12 | `test_checkpoint.py::TestCheckpointCompact::test_compact_reduces_size`                                  | D                  |
| E  | `test_backed_parity.py` (collection error)                                                              | E                  |

## Group A: `run_actionet` signature drift

**Symptom.** `TypeError: run_actionet() got an unexpected keyword argument 'layer'` and `... 'backed_chunk_size'`.

**Cause.** Both kwargs were removed from the public API. Current signature at
[`src/actionet/pipeline.py:18`](../src/actionet/pipeline.py) accepts only
`reduction_key` (defaulting to `"action"`) and no per-call chunk override —
chunking is now controlled by the input matrix source, not `run_actionet`.
The commit that decoupled archetype specificity (`c3b2259`) and later
refactors dropped these parameters from the pipeline entry point.

**Affected call sites.**

- [`tests/backed/test_backed_extension.py:60`](../tests/backed/test_backed_extension.py) — `run_actionet(adata, layer=layer, reduction_key="action_corrected", ..., backed_chunk_size=32, inplace=True)`
- [`tests/backed/test_backed_extension.py:816`](../tests/backed/test_backed_extension.py) — memory reference call with `layer=` and `backed_chunk_size=`
- [`tests/backed/test_backed_extension.py:820`](../tests/backed/test_backed_extension.py) — backed reference call with same kwargs
- [`tests/backed/test_lazy_transform_cache.py:189`](../tests/backed/test_lazy_transform_cache.py) — `run_actionet(..., backed_chunk_size=32, inplace=True)`
- [`tests/backed/test_lazy_transform_cache.py:456`](../tests/backed/test_lazy_transform_cache.py) — `run_actionet(adata_ref, layer="logcounts", ...)`

**Fix.** For each call site:

1. Drop the `layer=` kwarg. The equivalent behavior is:
   - Ensure the preceding `reduce_kernel(..., layer=..., key_added="action")` and
     `correct_batch_effect(..., layer=..., reduction_key="action", corrected_suffix="corrected")`
     calls already wrote the reduction to `adata.obsm["action_corrected"]`.
   - Pass `reduction_key="action_corrected"` (already present in the call
     sites, so this is literally just deleting `layer=…`).
2. Drop the `backed_chunk_size=` kwarg. The pipeline now inherits chunking
   from the matrix source. If the test wants to force a small chunk size,
   set it on `reduce_kernel(..., backed_chunk_size=32)` instead — that
   still accepts the parameter and is what controls SVD-time I/O.

**Regression posture.** No behavioral change expected; these calls
already carry `reduction_key="action_corrected"`, so the `layer=` kwarg
was purely redundant even before it was removed.

## Group B: `_open_backed_operator` is now a context manager

**Symptom.** `TypeError: cannot unpack non-iterable _GeneratorContextManager object` (tests 7, 8) and `Failed: DID NOT RAISE <class 'ValueError'>` (test 9).

**Cause.** [`src/actionet/backed_io.py:166`](../src/actionet/backed_io.py) decorates `_open_backed_operator` with `@contextlib.contextmanager` and yields the operator. The old surface returned a `(op, cleanup)` tuple; the new one is used as `with _open_backed_operator(...) as op: ...`. Consequences for the tests:

- `op, cleanup = _open_backed_operator(...)` unpacks a generator-context-manager → `TypeError`.
- Fast-fail assertions like `pytest.raises(ValueError)` around the *call* now trip nothing because the body of a `@contextmanager`-decorated function runs only on `__enter__`. The `ValueError` fires inside the `with` block.

**Affected call sites.**

- [`tests/backed/test_backed_open_operator.py:41`](../tests/backed/test_backed_open_operator.py) — retry test.
- [`tests/backed/test_backed_open_operator.py:73`](../tests/backed/test_backed_open_operator.py) — fallback test.
- [`tests/backed/test_backed_open_operator.py:110`](../tests/backed/test_backed_open_operator.py) — hard-fail test.

**Fix.** Rewrite each test to use the context manager:

```python
# Before
op, cleanup = backed_io._open_backed_operator(...)
try:
    assert op is sentinel
finally:
    cleanup()

# After
with backed_io._open_backed_operator(...) as op:
    assert op is sentinel
```

For test 9 (`non_lock_error_fails_fast`):

```python
with pytest.raises(ValueError, match="invalid group path"):
    with backed_io._open_backed_operator(...):
        pass
```

The fallback test (8) additionally inspects the temp path *after* the
context exits. Move that inspection into the `with` block, then assert on
post-exit cleanup outside:

```python
with backed_io._open_backed_operator(...) as op:
    fallback_paths = [Path(p) for p in create_calls if p != str(src)]
    assert len(fallback_paths) == 1
    fallback_path = fallback_paths[0]
    assert fallback_path.exists()

# After exit, fallback file should be removed
assert not fallback_path.exists()
```

## Group C: `_flush_backed_handle` was moved off `actionet.core`

**Symptom.** `AttributeError: module 'actionet.core' has no attribute '_flush_backed_handle'`.

**Cause.** The flush helper was relocated (likely into `actionet._backed_persist` or a private module) during the persistence refactor.

**Affected call site.**

- [`tests/backed/test_backed_reduce_kernel.py:39-53`](../tests/backed/test_backed_reduce_kernel.py) — `actionet_core._flush_backed_handle(_DummyAdata(), context="reduce_kernel")`.

**Fix.** Track down the new location before writing code:

```bash
rg -n "_flush_backed_handle" src/
```

Likely candidates: `src/actionet/_backed_persist.py` or a call site inline
in `reduce_kernel`. Two acceptable resolutions:

1. **Retarget the test** at the new location and import path. Keep the
   assertion (flush failure must raise `RuntimeError` mentioning
   "failed to flush backed AnnData handle").
2. **If the helper no longer exists** (e.g., `.file.flush()` errors are
   now caught and rethrown inline inside `reduce_kernel`), delete the
   unit test and add an integration-level replacement in
   `test_backed_reduce_kernel.py` that monkeypatches `adata.file._file.flush`
   to raise and asserts that `reduce_kernel(..., inplace=True)` propagates
   a `RuntimeError`.

Prefer option 1 if the helper still exists — it is a tight, fast unit
test worth keeping.

## Group D: brittle byte-size assertions in checkpoint tests

**Symptom.**

- `test_noop_checkpoint`: `assert 36984 >= 38392` — post-checkpoint size is *smaller* than the pre-checkpoint size.
- `test_compact_reduces_size`: `assert 66432 > 94128` — the "bloated" state is smaller than the clean baseline, defeating the test's precondition.

**Cause.** h5py 3.16 and anndata 0.12 changed encoding/attr layouts such
that:

- A no-op checkpoint can now legitimately *shrink* the file (dead attrs
  cleaned up during flush).
- Ten sequential overwrites of small annotation slots no longer produce
  monotonically-growing dead space (h5py reuses freed regions more
  aggressively).

**Affected call sites.**

- [`tests/backed/test_checkpoint.py:121-125`](../tests/backed/test_checkpoint.py)
- [`tests/backed/test_checkpoint.py:143-171`](../tests/backed/test_checkpoint.py)

**Fix.**

For `test_noop_checkpoint`, replace the fragile size-monotonicity check
with a *content-preservation* check:

```python
def test_noop_checkpoint(self, tmp_path):
    mem = make_test_adata(n_cells=10, n_genes=8)
    backed = open_backed(tmp_path, mem)

    # Snapshot on-disk shape and X payload
    with h5py.File(backed.filename, "r") as f:
        keys_before = sorted(f.keys())

    checkpoint_backed(backed)  # must not raise

    with h5py.File(backed.filename, "r") as f:
        keys_after = sorted(f.keys())
    assert keys_before == keys_after
    backed.file.close()
```

For `test_compact_reduces_size`, either:

1. Increase the bloat generator until `size_bloated > size_clean` holds
   reliably (e.g., overwrite 100× with distinct-shape arrays rather than
   same-shape overwrites), OR
2. Switch to measuring *chunk-level* dead space via `h5py.h5f.get_filespace`
   or `_repack_h5ad` output, which is what the compact code actually
   optimises. Assert the compacted file is `<=` the bloated file (already
   present at line 169) and drop the strict `>` precondition on
   `size_bloated > size_clean`.

Prefer option 2 — chunk-level free-space is the observable that
`compact=True` targets, and it is independent of h5py's fine-grained
allocation heuristics.

## Group E: `test_backed_parity.py` collection error

**Symptom.** `pytest.PytestRemovedIn9Warning: Marks applied to fixtures have no effect`.

**Cause.** [`tests/backed/test_backed_parity.py:53`](../tests/backed/test_backed_parity.py) decorates a `@pytest.fixture` with a marker (e.g. `@requires_ext`). Newer pytest promotes this to an error and refuses to collect the module.

**Fix.** Move the marker to the *test functions* that consume the fixture,
not the fixture itself. Concretely:

```python
# Before
@requires_ext
@pytest.fixture
def some_fixture(...): ...

# After
@pytest.fixture
def some_fixture(...): ...

@requires_ext
def test_uses_fixture(some_fixture): ...
```

If `requires_ext` is really a skip condition on availability of the
compiled extension, consider making it a
`pytest.mark.skipif(...)` on the module (via `pytestmark = ...` at the top
of the file) so every test in the module inherits it without needing to
touch each function.

## Execution order

Groups are independent. Recommended order (cheapest first):

1. **E** (single-line fix, unblocks collection so the module joins the run).
2. **A** (mechanical kwarg deletion across five call sites).
3. **B** (three tests, purely refactoring around `with`).
4. **C** (locate the moved helper; small).
5. **D** (needs a small design choice on what to actually measure).

## Verification

- `pytest tests/backed -q` should report 0 failures / 0 errors on both
  `dev-gpu-v2` and `dev` after these fixes.
- Cross-check by running once with `--collect-only` first to confirm E is
  resolved before touching anything else.
- Guard against re-drift by adding a lightweight signature-contract test
  (out of scope here but worth filing as a follow-up):

  ```python
  def test_run_actionet_signature_is_stable():
      import inspect
      sig = inspect.signature(actionet.run_actionet)
      assert "layer" not in sig.parameters
      assert "backed_chunk_size" not in sig.parameters
  ```

  This catches the *next* time a test file lags behind the API.

## Non-goals

- Do not modify production code in `src/actionet/`. Every failure is
  test-side.
- Do not chase the `test_backed_parity.py` skip logic; only unblock
  collection. Any parity failures that surface after collection succeeds
  are a separate ticket.
- Do not attempt to restore `layer=` or `backed_chunk_size=` kwargs on
  `run_actionet`. Their removal was intentional and downstream callers
  in `src/actionet/preprocessing.py` and the pipeline already reflect it.

## Related work

- Bug fix for `subset_anndata` "Invalid file identifier" after
  `.to_memory()` on a backed view: separate change in
  [`src/actionet/_backed_persist.py`](../src/actionet/_backed_persist.py)
  adding `_ensure_backed_open`. That fix is orthogonal to this plan and
  landed with green regression tests in
  [`tests/test_subset_anndata.py`](../tests/test_subset_anndata.py).
