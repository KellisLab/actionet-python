# Codebase Cleanup Audit — Context Document (Python front-end)

Use this document as standing context when auditing individual modules of
`actionet-python` (the Python package and its pybind11 wrappers). Pass it to
an agent alongside the specific module(s) being audited.

The companion document `context/AUDIT_CONTEXT_CPP.md` covers the C++ core
(`libactionet`, git submodule). Audit the two sides separately using their
respective context files.

---

## Project Overview

ACTIONet is a multi-language computational biology toolkit for single-cell
multi-resolution analysis. This repository (`actionet-python`) provides the
Python front-end via pybind11 bindings to the C++ core (`libactionet`,
included as a git submodule at `src/libactionet/`).

- **Repository root:** `actionet-python`
- **Branch under audit:** `dev-gpu` (diverged from `dev`, which is the branch
  the last cleanup pass targeted). The Python front-end cleanup pass for
  change streams (1)–(4) completed on 2026-07-14; see "Cleanup Pass Status"
  below.
- **Python package:** `src/actionet/`
- **pybind11 wrappers:** `src/actionet/bindings/`
- **C++ core (submodule):** `src/libactionet/`
- **Build:** CMake + scikit-build-core + pybind11 (C++17, OpenMP required)
- **Runtime deps:** BLAS/LAPACK, HDF5, OpenMP
- **Python version:** ≥3.12
- **Package version:** `0.4.0` (bumped from `0.3.0` on this branch; the bump
  is intended to ship together with the 2026-07-14 cleanup commit).

---

## Motivation for This Audit

`dev-gpu` accumulated four overlapping change streams since diverging from
`dev`. GPU work is the next major feature; before that lands, the Python
surface needs a targeted cleanup pass to remove residue from those change
streams and flag inconsistencies they introduced.

### Change streams landed on `dev-gpu`

1. **SVD path removal — Feng and PRIMME retired from the public Python API.**
   `_SVD_ALGORITHM_TO_ID` in `src/actionet/decomposition/svd.py` now exposes
   only `{"irlb": 0, "halko": 1}`. `_normalize_algorithm` raises `ValueError`
   for `"feng"` / `"primme"` in any casing. The pybind entry points
   (`run_svd_sparse`, `run_svd_dense`, `run_svd_backed_operator`,
   `reduce_kernel_sparse`, `reduce_kernel_dense`,
   `reduce_kernel_backed_operator`) all call `validate_python_svd_algorithm`
   defined in `src/actionet/bindings/wp_utils.h` so private `_core` callers
   cannot bypass the string validation. See `context/DECISIONS.md`
   ("SVD algorithm strategy").

2. **64-bit extensions to surviving SVD paths.** IRLB and Halko now guard
   per-axis row/column dimensions against `INT_MAX` inside `libactionet`
   (see the C++ audit context). Python-side documentation in
   `src/actionet/decomposition/svd.py` and `README.md` was updated to
   describe the new contract: sparse `nnz > 2^31 - 1` is supported directly,
   but per-axis dimensions must fit in `INT_MAX`.

3. **Threading changes for cross-BLAS harmony.** ACTION's AA/simplex-
   regression path now uses inline small-dense kernels at the C++ layer and
   a new `OuterParallelRegionScope` / `get_num_threads_nested_safe` pair to
   avoid oversubscription across MKL, OpenBLAS-pthread, and OpenBLAS-OpenMP.
   No public Python API change; the affected Python surface is documented
   in `plans/openblas_threading_fix_handoff.md` and
   `plans/openblas_threading_and_odr_findings.md`. New test suites live in
   `tests/benchmark_action_blas_backends.py` (~429 lines, benchmark) and
   `tests/test_action_small_dense_kernels.py` (~208 lines, unit tests).

4. **AnnData >= 0.13 compatibility patches.** `src/actionet/io/persist.py`
   grew a `_real_layer_keys` helper (filters the `None` alias `layers[None]`
   that AnnData 0.13 aliases to `.X`) and a substantially expanded
   `_init_from_reopened` implementation that unpacks the reopened AnnData
   into explicit kwargs and drives the file-init branch. Callers in
   `src/actionet/io/subset.py` and `src/actionet/io/checkpoint.py` were
   updated to use `_real_layer_keys` when listing layers. The 2026-07-14
   cleanup pass additionally added inline `None` filters at two remaining
   layer-iteration sites (`persist_updates` in `io/persist.py` and
   `collect_annotation_results` in `io/anndata_io.py`) so the patch is now
   local to the io layer. TODO.md documents this as a
   "Deferred: simplify anndata 0.13 backed compatibility patch once
   `anndata>=0.13` is the floor".

### Audit goals

1. Remove residue from stream (1): stale docstrings, comments, TODO entries,
   or Python-only helpers that still reference Feng/PRIMME. Confirm the
   negative tests in `tests/test_irlb_svd_parity.py` remain intentional.
2. Verify stream (2): the docstring/`README.md` contract for 64-bit SVD is
   consistent with the pybind entry points and the C++ guards, and that the
   `svd_backend_requested` / `svd_backend_resolved` output keys added to
   `run_svd`/`reduce_kernel` are documented and tested.
3. Verify stream (3): the Python surface has no coupling to BLAS backend
   detection, no environment-variable requirements, and no import warnings.
   Confirm `tests/benchmark_action_blas_backends.py` and
   `tests/test_action_small_dense_kernels.py` cover the new seams, and that
   the AA/simplex tolerance decisions documented in `context/DECISIONS.md`
   ("ACTION numerical decision stability") are only asserted at the C++
   layer (not silently duplicated in Python).
4. Verify stream (4): the anndata compat patch is scoped exactly to the
   three files listed in TODO.md, does not leak `None` into any other key
   iteration, and has direct test coverage (e.g. `tests/test_backed_layers_none.py`).
5. Flag any coupling hotspots or documentation drift that will block GPU
   dispatch insertion (`ExecutionPolicy`, backend selection) at the Python
   surface.

---

## Cleanup Pass Status (2026-07-14)

The audit driven by this document was executed against `dev-gpu` on
2026-07-14 and its recommendations were implemented in a single
uncommitted change set (staged alongside the `0.3.0 -> 0.4.0` version
bump). Subsequent audit runs should treat the items below as **already
resolved** and focus instead on new drift.

Landed cleanup items (all in the working tree, unstaged):

1. **Failure 3 fix**:
   `tests/test_irlb_svd_parity.py::test_core_backed_operator_rejects_retired_algorithm_ids`
   now sets `HDF5_USE_FILE_LOCKING=FALSE` and closes the anndata `r+`
   handle before opening the file through
   `_core.create_backed_operator`.
2. **`reduce_kernel` precomputed-SVD metadata honesty**: when
   `precomputed_svd` is supplied,
   `src/actionet/decomposition/kernel.py` now persists
   `svd_algorithm=None` / `svd_algorithm_name="none"` and
   `used_precomputed_svd=True` instead of a resolved-but-unused id.
3. **AnnData `layers[None]` alias**: `src/actionet/io/anndata_io.py`
   (`collect_annotation_results`) and `src/actionet/io/persist.py`
   (`persist_updates`, the `layers_keys` `_dirty_tracker.mark` call)
   now filter the `None` key inline. The dedicated `_real_layer_keys`
   helper is unchanged (still the primary shared abstraction).
4. **`run_svd` backed-branch simplification**: dropped the redundant
   `selected_algorithm`/`algorithm_id` pair in
   `src/actionet/decomposition/svd.py`; a single `algorithm_id`
   variable is used end-to-end.
5. **Duplicate `// svd_main` banner** in
   `src/actionet/bindings/wp_decomposition.cpp` renamed to
   `// perturbed_svd` for the earlier section; the true `svd_main`
   banner remains at the correct location.
6. **Docstring coverage** for the SVD provenance metadata added by
   stream (2): `run_svd` (svd.py) and `reduce_kernel` (kernel.py)
   docstrings now document `svd_algorithm`, `svd_algorithm_name`,
   `svd_backend_requested`, `svd_backend_resolved`,
   `used_precomputed_svd`, and `operator_mode`.
7. **Negative-test tightening**:
   `test_reduce_kernel_rejects_primme` and
   `test_reduce_kernel_rejects_feng` now assert the error message
   contains the allowed set `{auto, halko, irlb}`, matching the
   pattern already used for `run_svd`.
8. **New provenance regression tests** in
   `tests/test_irlb_svd_parity.py`
   (`test_run_svd_emits_backend_metadata_*`,
   `test_run_svd_operator_compatible_omits_backend_metadata`,
   `test_reduce_kernel_persists_backend_metadata_inmemory`,
   `test_reduce_kernel_precomputed_svd_marks_algorithm_none`) covering
   inmem sparse, inmem dense, backed, operator-compatible short form,
   and the precomputed-SVD `svd_algorithm=None` contract.
9. **Halko synthetic-parity convergence (Failure 1) — test-side fix**:
   the four cross-algorithm parity tests
   (`test_{inmemory,backed}_{sparse,dense}_parity_irlb_vs_halko`) now
   pass `max_iter=HALKO_SYNTHETIC_MAX_IT=20` so Halko can converge on
   the small full-rank synthetic matrices. `SIGMA_RTOL` remains at
   5%. Same-algorithm smoke tests keep `max_iter=0` so the public
   default (`default_max_it=5`, tuned for real single-cell matrices)
   is still exercised. The C++ Halko default is intentionally
   unchanged.
10. **Stale benchmark artefacts** at
    `tests/benchmark_results/svd_backed_full/` and
    `tests/benchmark_results/svd_inmem_full/` (both full of Feng
    references) were deleted. `tests/benchmark_results/batch_correction/`
    is preserved.
11. **Version bump** `pyproject.toml` `0.3.0 -> 0.4.0` is retained in
    the working tree and intended to ship with the cleanup commit.

Items intentionally **not** changed during the pass:

- Halko `default_max_it=5` in
  `src/libactionet/src/decomposition/svd_main.cpp` — correct default
  for real single-cell data; Failure 1 was fixed test-side.
- Python-vs-C++ default-value discrepancies where the docstring
  wording was already accurate (e.g. `max_iter=0` documented as
  "0 = solver default").
- The `_real_layer_keys` helper, `_init_from_reopened`, and the
  Feng/PRIMME negative-tests block — retained per guardrails.

---

## Cleanup Pass Status (2026-07-14, part 2 — libactionet + pybind11 audit)

A follow-up cleanup pass driven by
`.cursor/plans/libactionet_cleanup_plan_fd9dcfd0.plan.md` covered
the C++ core (`libactionet`) and the pybind11 wrappers
(`src/actionet/bindings/`) on `dev-gpu`. All items landed as an
uncommitted change set on 2026-07-14 alongside the part-1 pass
above. The full Python test suite passes (547 passed, 1 skipped)
after an editable rebuild. See
`context/AUDIT_CONTEXT_CPP.md` for the full item-by-item list; the
Python-front-end-visible items are:

1. **pybind11 GIL discipline aligned.** All eight `wp_decomposition`
   entry points now release the GIL through
   `py::gil_scoped_release` blocks — including the previously
   omitted `orthogonalize_batch_effect_operator` and
   `orthogonalize_basal_operator`. Four bare
   `release; ...; acquire;` sites in `wp_annotation.cpp` were also
   converted to scoped-block form.
2. **All `run_svd_*` and `perturbed_svd` pybind entry points now
   return C-order numpy arrays.** `wp_decomposition.cpp` moved from
   `arma_mat_to_numpy` (Fortran-order) to `arma_mat_to_numpy_c` for
   `run_svd_sparse`, `run_svd_dense`, and `perturbed_svd`; this
   matches the already-C-order `svd_to_dict` in `wp_io.cpp`.
   Downstream numpy consumers touching `.strides` or `.tobytes()`
   now see consistent behavior across all four entry points.
3. **Shared `int_array_to_uvec<T>` template.** Promoted from the
   private helper in `wp_io.cpp` to `wp_utils.h`, parameterized on
   element type, adopted across `wp_action.cpp`, `wp_annotation.cpp`
   (four label-unpacking sites), and `wp_network.cpp`. `wp_io.cpp`'s
   `int64_array_to_uvec` is now a thin delegation.
4. **`run_action` `tol` divergence documented.** A cross-referencing
   comment block was added above `decomp_action` in `wp_action.cpp`
   explaining the intentional 3-way default divergence (`C++ 1e-6`,
   pybind `1e-16`, Python wrapper `1e-100`). Defaults were not
   aligned — Python's `run_action.py` always passes an explicit
   `tolerance`, so the pybind default is effectively unused from
   Python.
5. **[Behavior-visible from C++/R only] Unknown SVD algorithm codes
   now throw.** `runSVD` and `runSVD_Operator` no longer silently
   fall through to IRLB when given a retired code (`2`, `3`). Python
   is unaffected — `validate_python_svd_algorithm` already rejects
   these at the binding boundary — but it closes the last back door
   into the retired paths from direct C++/R callers.

There is no known Python-surface regression from any of the above.

---

## Architecture (Layer Diagram)

```
┌──────────────────────────────────────────────────────────────┐
│  Python API  (src/actionet/*.py)                             │
│  User-facing functions, AnnData integration, orchestration   │
├──────────────────────────────────────────────────────────────┤
│  pybind11 Wrappers  (src/actionet/bindings/*.cpp)            │
│  Thin C++ → Python bridge; data conversion, lifetime mgmt    │
├──────────────────────────────────────────────────────────────┤
│  libactionet C++ Core  (src/libactionet/)                    │
│  Armadillo-based numerics, OpenMP parallel, HDF5 backed I/O  │
└──────────────────────────────────────────────────────────────┘
```

Audit scope for **this** document stops at the pybind boundary. The C++
core is covered in `context/AUDIT_CONTEXT_CPP.md`.

---

## Module Map — Python Package (`src/actionet/`)

The package mirrors the `libactionet` C++ subpackage layout. Each subpackage
below corresponds to a `libactionet/src/<name>/` directory.

| Subpackage / File | Role | Change stream(s) since `dev` |
|-------------------|------|------------------------------|
| `__init__.py` | Public API surface; re-exports from all subpackages | none |
| `pipeline.py` | End-to-end `run_actionet` convenience pipeline | none |
| `_feature_lookup.py` | Feature-name resolution (shared internal helper) | none |
| **`action/`** | `run_action`, archetypal analysis, ACTION decomposition | (3) via C++ only |
| **`annotation/`** | Markers, cell/cluster annotation, feature specificity | none |
| **`bindings/`** | pybind11 wrappers — built as `actionet._core` | (1) SVD alg guards |
| **`decomposition/`** | `svd.py`, `kernel.py` — Python surface for SVD + kernel reduction | (1), (2) |
| **`io/`** | Backed HDF5 stack: matrix source, operator, lazy transform, persistence, checkpoint, subset | (4) anndata 0.13 |
| **`network/`** | Network build, diffusion, centrality, clustering, imputation | none |
| **`preprocessing/`** | Import, filter/subset, normalize, backed decompression (`io.py`, `filter.py`, `normalize.py`) | none |
| **`tools/`** | Matrix utilities, AnnData helpers, batch correction, guide calling | none |
| **`visualization/`** | Layout, node colors, all plotting variants | none |
| `_data/` | Bundled data / resources | none |

---

## Module Map — pybind11 Wrappers (`src/actionet/bindings/`)

| Wrapper File | C++ Module Bound | Change stream(s) |
|--------------|------------------|------------------|
| `_core.cpp` | Module entry; dispatches to `init_*` functions | none |
| `wp_action.cpp` | action (AA, ACTION decomp, reduce_kernel, SPA, simplex) | (1) added `validate_python_svd_algorithm` to both `reduce_kernel_*` |
| `wp_decomposition.cpp` | decomposition (SVD algorithms, orthogonalization) | (1) added `validate_python_svd_algorithm` to both `run_svd_*` |
| `wp_network.cpp` | network (build, diffusion, measures, label propagation) | none |
| `wp_annotation.cpp` | annotation (specificity, marker_stats) | none |
| `wp_io.cpp` | io (backed HDF5 operators) | (1) added guard to both backed-operator SVD entry points |
| `wp_tools.cpp` | tools (autocorrelation, enrichment, aggregation, xicor, MWM, guide calling) | none |
| `wp_visualization.cpp` | visualization (layout, color map) | none |
| `wp_utils.{cpp,h}` | Shared conversion utilities (arma ↔ numpy/scipy); now hosts `validate_python_svd_algorithm` inline | (1) |

The `validate_python_svd_algorithm` guard rejects any algorithm id outside
`{ALG_IRLB=0, ALG_HALKO=1}` with a `RuntimeError` message pattern of
`"unsupported SVD algorithm id ...; valid IDs: 0 (IRLB), 1 (Halko)"`.
Negative tests exercise it from every entry point (see the
"Regression tests: PRIMME removal and 64-bit sparse support" and
"Regression tests: SVD backend / algorithm provenance metadata"
sections of `tests/test_irlb_svd_parity.py`).

---

## Key Architectural Decisions (Non-Negotiable)

Recorded in `context/DECISIONS.md`; must not be re-litigated during audit.

1. **OpenMP is a hard requirement** — no OFF option; used pervasively.
2. **Public Python SVD surface: IRLB, Halko.** Feng and PRIMME are removed
   from both the Python API and `libactionet`. `"auto"` selects IRLB for
   sparse in-memory, Halko for dense in-memory, Halko for backed operator.
3. **Halko is the default backed SVD** — predictable I/O cost model.
4. **BLAS policy for ACTION.** Inline small-dense kernels below
   `INLINE_DIMENSION_LIMIT = 128` (min dimension); larger matrices route
   through the configured BLAS. No public API change, no BLAS-vendor
   detection, no runtime thread guard, no import warnings.
5. **ACTION numerical decision stability.** Simplex support is defined as
   coefficient strictly greater than `1e-6`. Enforced at the C++ layer
   only; do not duplicate the constant in Python.
6. **GPU: NVIDIA CUDA, Python-first, SVD-first.** CUDA 12.2+, Ampere+ only,
   Linux x86_64 (WSL2 for dev). GPU is a backend, not a new SVD algorithm.
7. **`libactionet` APIs are contracts** — Python wrappers map to them
   cleanly; do not silently introduce Python-only defaults that shift C++
   behavior.
8. **Breaking changes are allowed** if they substantially improve
   performance, resource usage, ease-of-use, or reproducibility.

---

## GPU Support Context (Upcoming)

Python-facing GPU work will primarily affect:

- `decomposition/` — `run_svd` and `reduce_kernel` need a `backend=` kwarg
  and the `svd_backend_requested`/`svd_backend_resolved` output keys
  already added on this branch (in `svd.py` and `kernel.py`) are the
  hook for that. Confirm both keys are emitted from every non-operator-
  compatible return path.
- `bindings/` — new C++ entry points and `ExecutionPolicy` plumbing.
- `io/` — device memory management for backed streaming (deferred; not
  in scope for this audit).

Do not recommend restructuring Python entry points around a GPU backend
in this audit — flag readiness only where the current shape blocks
insertion. The `plans/GPU_INTEGRATION.md`, `plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`, and `src/libactionet/plans/GPU_BACKEND_PLAN.md` docs
describe the intended shape.

---

## Audit Execution Instructions

When auditing a module, apply the `/codebase-cleanup-audit` skill with
these repo-specific instructions.

### Scope

- Audit the specified Python module(s) and their pybind11 wrappers only.
  Do **not** audit the C++ core in this pass — that is covered by
  `context/AUDIT_CONTEXT_CPP.md`.
- When a Python module has a corresponding `wp_<module>.cpp`, audit them
  together (they are one architectural layer).
- Cross-reference the change streams in "Motivation for This Audit" to
  decide whether a finding is a residue of the recent changes or a
  pre-existing issue. Flag both, but distinguish them.

### Priorities (ordered)

1. **Confirmed residue from the four change streams**: stale docstrings,
   comments, dead imports, unused helpers, negative-test drift, TODO
   entries whose remediation already landed. This is the highest-value
   category for this pass because the streams overlapped and touched
   many files.
2. **Documentation and contract drift.** The `svd_backend_requested` and
   `svd_backend_resolved` output keys are new; verify every non-
   `return_operator_compatible=True` return path emits them and that
   docstrings cover the new keys. Verify the 64-bit contract wording is
   identical between `run_svd`, `reduce_kernel`, and `README.md`.
3. **API and contract drift across the pybind boundary.** Signature
   mismatches between Python wrappers and the C++ entry points they
   call, algorithm ID drift, and missing/incorrect default values.
4. **Test coverage on new seams.** Confirm each new C++/Python seam
   introduced by streams (2)–(4) has direct Python-level tests. Missing
   coverage should be flagged, not fabricated.
5. **Coupling hotspots that will block GPU dispatch insertion.**
   Especially in `decomposition/svd.py` and `decomposition/kernel.py`.
6. **General consolidation opportunities and easy efficiency wins**
   (repeated normalization, redundant AnnData copies on cold paths,
   duplicate config parsing). Keep this bounded — do not turn the audit
   into a rewrite.
7. **Known-untouched surfaces.** Streams (1)–(4) barely touched
   `network/`, `annotation/`, `preprocessing/`, `tools/`, and
   `visualization/`. Audit those modules only for pre-existing issues
   already listed in `TODO.md`, and default to "no material findings" if
   nothing has drifted.

### Guardrails

- Audit only. Do not edit code.
- Do not re-litigate decisions in `context/DECISIONS.md`.
- Do not recommend deleting the anndata 0.13 compatibility patch: it is
  intentionally retained until `anndata>=0.13` is the floor (see
  `TODO.md` "Deferred" entry).
- Do not recommend deleting the negative Feng/PRIMME tests in
  `tests/test_irlb_svd_parity.py`: they enforce the public API contract
  for removed algorithms and are explicitly required by
  `context/DECISIONS.md`.
- Treat `__init__.py` exports as the public API contract.
- Do not recommend restructuring the `layers[None]` filter or
  `_init_from_reopened` logic beyond what `TODO.md` describes.
- Do not audit `context/_old/` — those documents describe a prior
  cleanup cycle and are kept for reference.

### Report Format

Use the findings-first report template from the skill. Include:

- File paths and line numbers.
- Severity classification (`Confirmed issue` / `Likely risk` / `Open question`).
- Which change stream (1–4) the finding relates to, or "pre-existing" if it
  predates this branch.
- Concrete evidence for each finding.
- Quick wins vs. larger follow-ups at the bottom.

---

## Known Open Issues (do not re-discover; verify status)

These were documented in `plans/investigate_svd_parity_test_failures.md`.
Status reflects the state after the 2026-07-14 cleanup pass; verify
before re-auditing.

- **Failure 1 (RESOLVED — test-side)**:
  `test_inmemory_sparse_parity_irlb_vs_halko` and the three sibling
  cross-algorithm parity tests now pass `max_iter=20` to Halko so it
  converges on the small full-rank synthetic matrices. The C++
  in-memory Halko default (`default_max_it=5`) is intentionally
  unchanged — it is correct for real single-cell matrices where
  spectra decay quickly, and same-algorithm smoke tests still
  exercise it.
- **Failure 2 (RESOLVED)**:
  `test_core_run_svd_rejects_retired_algorithm_ids` and
  `test_core_reduce_kernel_rejects_retired_algorithm_ids` pass; the
  `validate_python_svd_algorithm` guard covers all six entry points.
- **Failure 3 (RESOLVED)**:
  `test_core_backed_operator_rejects_retired_algorithm_ids` now sets
  `HDF5_USE_FILE_LOCKING=FALSE` and closes the anndata `r+` handle
  before invoking `_core.create_backed_operator`. Fixed in the
  2026-07-14 cleanup pass.

Additionally:

- **Pyproject `0.3.0 -> 0.4.0` bump** is present in the working tree
  and will ship together with the 2026-07-14 cleanup commit.

---

## Suggested Audit Order

Audit in dependency order (leaves first, roots last). Each row lists the
Python subpackage alongside its pybind11 wrapper counterpart. Change-stream
column indicates which streams touched the module.

| Phase | Module (Python + wrapper) | Change streams | Rationale |
|-------|----------------------------|----------------|-----------|
| 1 | `bindings/wp_utils.{cpp,h}` + `bindings/_core.cpp` | (1) | Foundational; new SVD guard lives here |
| 2 | `decomposition/svd.py` + `decomposition/kernel.py` + `bindings/wp_decomposition.cpp` + `bindings/wp_action.cpp` (reduce_kernel arms) | (1), (2) | Highest churn on this branch; new backend metadata keys, algorithm guards, 64-bit doc contract |
| 3 | `io/persist.py` + `io/subset.py` + `io/checkpoint.py` + `bindings/wp_io.cpp` | (1), (4) | AnnData 0.13 patch; backed-operator SVD guard |
| 4 | `action/` (Python) + `bindings/wp_action.cpp` | (3) via C++ | AA/simplex wrappers should be pure passthrough — verify no Python-side numerical constants |
| 5 | `network/` + `bindings/wp_network.cpp` | none | Regression sweep only |
| 6 | `annotation/` + `bindings/wp_annotation.cpp` | none | Regression sweep only |
| 7 | `preprocessing/` + `_feature_lookup.py` | none | Python-only orchestration; regression sweep |
| 8 | `tools/` + `bindings/wp_tools.cpp` | none | Regression sweep only |
| 9 | `visualization/` + `bindings/wp_visualization.cpp` | none | Regression sweep only |
| 10 | `pipeline.py` + `__init__.py` | none | Top-level API surface; last because it depends on everything |
| 11 | `tests/` | (1)–(4) | Verify coverage of new seams; check for stale test data or algorithm parametrizations |

---

## Phase-Specific Focus Areas

### Phase 1: `bindings/wp_utils.{cpp,h}` + `bindings/_core.cpp`

- Confirm `validate_python_svd_algorithm` in `wp_utils.h` is the only SVD
  algorithm-id guard site and that every `run_svd_*` / `reduce_kernel_*`
  wrapper calls it before any GIL release.
- Check for orphan helpers or includes left behind by the PRIMME/Feng
  removal (e.g. any lingering `svd_primme.hpp` / `svd_feng.hpp` includes,
  dead `#if defined(...)` blocks).

### Phase 2: `decomposition/` + `wp_decomposition.cpp` + reduce_kernel arms in `wp_action.cpp`

- Verify `_SVD_ALGORITHM_TO_ID`, `_SVD_ID_TO_ALGORITHM`, and
  `_normalize_algorithm` are the single source of truth on the Python
  side. No Python module should hardcode algorithm IDs.
- Verify the `svd_backend_requested` / `svd_backend_resolved` keys are
  emitted from `run_svd` (`return_operator_compatible=False` branch) and
  from `reduce_kernel` (via `.uns[...]`). Check docstrings for both keys.
- Confirm the `algorithm_id` variable in `run_svd` is now assigned on the
  backed branch (the diff shows this was added — verify it did not break
  the operator-compatible return path).
- Cross-check `_select_svd_algorithm_inmemory` and
  `_select_svd_algorithm_backed` docstrings against the actual defaults
  and the "SVD algorithm strategy" and "Backed SVD algorithm default"
  entries in `context/DECISIONS.md`.
- Confirm the 64-bit contract wording is consistent across `run_svd`,
  `reduce_kernel`, and `README.md`.
- Look for any pre-PRIMME dead branches or comments still referencing
  32-bit overflow handling that no longer applies.
- Verify `_maybe_decompress_backed_path` and `_chunk_target_bytes` are
  the only backed-path helpers shared between `svd.py` and `kernel.py`;
  flag any duplication.

### Phase 3: `io/persist.py` + `io/subset.py` + `io/checkpoint.py` + `wp_io.cpp`

- Confirm `_real_layer_keys` is used everywhere layers are enumerated
  during backed persistence, subset, and checkpoint. Flag any remaining
  `list(adata.layers.keys())` call that could reintroduce the `None`
  alias into HDF5 output.
- Verify `_init_from_reopened` matches the expanded implementation
  documented in `TODO.md` "Deferred" section (explicit kwargs,
  file-init branch, raw-arg handling, file-handle close-and-adopt).
- Check `wp_io.cpp` for consistency with the SVD algorithm guard added
  to both backed-operator SVD entry points; verify no other backed
  entry point silently accepts an algorithm id.
- Test coverage: `tests/test_backed_layers_none.py` (72 lines, added
  this branch) is the direct regression test for stream (4). Confirm it
  exercises persist, subset, and checkpoint.

### Phase 4: `action/` + `wp_action.cpp`

- Verify no Python code duplicates the ACTION numerical constants
  (simplex support tolerance, SPA tie tolerance, active-set thresholds).
  Those live only in
  `src/libactionet/include/utils_internal/utils_action_numeric_policy.hpp`.
- Confirm no Python module reads `OMP_NUM_THREADS` or otherwise tries
  to negotiate thread counts with the C++ layer — the C++ policy is
  authoritative (see `context/DECISIONS.md` "BLAS policy for ACTION").

### Phases 5–10: regression sweep only

Change streams (1)–(4) barely touched `network/`, `annotation/`,
`preprocessing/`, `tools/`, `visualization/`, `pipeline.py`, or
`__init__.py`. For each of those modules, audit only for:

- **Confirmed issues** from any of the four streams (should be rare).
- Pre-existing `TODO.md` items whose fix is a quick win.
- Documentation drift against the current `README.md` and `docs/`.

Default to `No material cleanup or quality findings were confirmed in
the audited scope.` when nothing has drifted. Do not invent work.

### Phase 11: `tests/`

- Confirm `tests/test_irlb_svd_parity.py` negative tests still cover
  every removed-algorithm entry point (in-memory sparse, in-memory
  dense, backed sparse, backed dense; via both `_core.*` and the public
  Python surface). The `reduce_kernel_rejects_{primme,feng}` tests
  also assert the error message lists `{auto, halko, irlb}`.
- Confirm the SVD backend / algorithm provenance regression tests
  (`test_run_svd_emits_backend_metadata_*`,
  `test_run_svd_operator_compatible_omits_backend_metadata`,
  `test_reduce_kernel_persists_backend_metadata_inmemory`,
  `test_reduce_kernel_precomputed_svd_marks_algorithm_none`) still
  cover the new metadata contract from stream (2).
- Failure 3 (`test_core_backed_operator_rejects_retired_algorithm_ids`)
  is resolved; the test now disables HDF5 file locking and closes the
  anndata handle before opening the file from C++.
- Halko synthetic-parity convergence (Failure 1) is fixed test-side
  via `max_iter=HALKO_SYNTHETIC_MAX_IT=20` on the four cross-algorithm
  parity tests; C++ `default_max_it` unchanged.
- New test files added this branch:
  `tests/test_action_small_dense_kernels.py`,
  `tests/test_backed_layers_none.py`,
  `tests/benchmark_action_blas_backends.py`,
  `tests/benchmark_svd_inmemory_defaults.py`. Verify each is discovered
  by `pytest` (check markers, `python_files` patterns) and that
  benchmark markers `benchmark` and `openblas_smoke` in
  `pyproject.toml` match the tests that use them.
- `tests/benchmark_results/svd_{backed,inmem}_full/` were deleted in
  the 2026-07-14 cleanup pass (stale Feng-referenced artefacts).
  `tests/benchmark_results/batch_correction/` is preserved.
- Check notebooks (`tests/*.ipynb`) for stale algorithm arguments
  (`svd_algorithm="feng"` etc.) — the diffs showed several were
  touched but the audit should confirm none still request removed
  algorithms.

---

## Example Agent Prompt

```
Read context/AUDIT_CONTEXT.md for project context and audit instructions.

Audit Phase [N]: [module list]

Apply the /codebase-cleanup-audit skill to the following files:
- [list specific file paths]

Focus on: [phase-specific focus areas]

Produce a findings-first report. Do not edit code.
```

---

## Cross-References

- `context/AUDIT_CONTEXT_CPP.md` — companion audit context for the C++
  core; use for anything under `src/libactionet/`.
- `context/DECISIONS.md` — Python-side architectural decisions.
- `context/_old/AUDIT_CONTEXT.md` and `context/_old/AUDIT_CONTEXT_CPP.md`
  — prior cleanup cycle; kept for reference only.
- `TODO.md` — outstanding Python work items and the anndata 0.13
  "Deferred" entry.
- `plans/openblas_threading_fix_handoff.md` — final BLAS policy handoff.
- `plans/openblas_threading_and_odr_findings.md` — full diagnosis of the
  ACTION performance regression and the small-dense kernel remedy.
- `plans/investigate_svd_parity_test_failures.md` — the three
  post-removal SVD parity failures and recommended fixes.
- `plans/GPU_INTEGRATION.md` — Python-facing GPU roadmap.
- `plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md` — GPU-backed SVD launchpad.
- `docs/svd_algorithm_benchmark.md` — benchmark evidence for the retired
  Feng and current auto-selection defaults.
- `src/libactionet/TODO.md` — remaining C++-side cleanup notes, in
  particular the `actionet-r` Feng/PRIMME patch that must still land
  out-of-tree.
