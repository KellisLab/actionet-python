# Agent Playbook — actionet-python (pybind11 front-end)

## Purpose of this repository

This repository provides the **Python user-facing interface** to the Actionet C++ core via **pybind11**. It is also the primary interface used by the Illumina perturb-seq preprocessing pipeline and is the forward-looking, performance-first front-end.

Downstream dependencies:

- `perturbseq_preprocessing` pipeline (private; HPC-deployed)

Upstream dependency:

- `libactionet` C++ core library

---

## Repository layout (high level)

- `src/` — Python package source, pybind11 bindings, and `libactionet` submodule
- `tests/` — tests (import, API sanity, regressions)
- `docs/` — user and developer documentation
- `data/` — example or reference datasets (not synced; lives locally)
- `scripts/` — helper scripts
- `R/` — Deprecated: Copy of the R API for parity reference. (not synced)
- `CMakeLists.txt` — CMake build for extension module
- `pyproject.toml`, `setup.py`, `MANIFEST.in` — packaging and build configuration
- `install_optimized.sh` — optimized build/install helper

---

## What success looks like

- Python API is complete enough to support core analysis functionality and current pipeline needs
- Stable and well-documented public interfaces
- Reliable builds using CMake + pybind11 within conda environments
- Tests that protect against regressions and wrapper/core mismatch
- Behavior parity with `actionet-r` where applicable and intentional divergence where documented

---

## Hard guardrails (must follow)

- Assume HPC builds occur in **de-containerized conda environments** (no Docker/Singularity).
- Avoid introducing dependencies that are difficult to build on shared clusters.

---

## How to work safely in this repo

### Adding or modifying Python-facing functionality

1. Define the Python API first (function signature, docstring, return types).
2. Map the API cleanly to the C++ core (avoid leaking C++-specific complexity).
3. Add or update pybind11 bindings with clear ownership and lifetime semantics.
4. Add tests in `tests/` using minimal, representative inputs.
5. Update documentation (`README.md` or `docs/`) if user-visible behavior changes.

### Coordinating with the C++ core

- Treat `libactionet` APIs as a contract.
- If a core change is required:
  - propose the C++ change explicitly
  - note required wrapper updates
  - avoid situations where Python temporarily breaks due to core drift

---

## Parity with R front-end

The R package (`actionet-r`) serves as a living **reference implementation** for many behaviors.

When implementing or modifying features:

- Match R semantics and outputs when feasible.
- If Python diverges intentionally (e.g., performance-driven changes):
  - document the difference clearly
  - ensure downstream pipeline expectations are updated accordingly

Recommended parity dimensions to track:

- function names
- parameter names and defaults
- output object structure and field names
- ordering and determinism guarantees

---

## Build and environment assumptions

- Build system: CMake + pybind11
- Typical environment: virtualenv or conda-based Python environment
- Must compile with common HPC toolchains (GCC/Clang)

Avoid:

- assuming system-wide libraries or Python packages

---

## Common pitfalls

- Exposing unstable or experimental C++ APIs directly to users
- Mismatch of row/column ordering of matrices passed to C++ API
- Tight coupling between pipeline-specific logic and the general-purpose API
- Python-only features that implicitly change shared core semantics

---

## When blocked, ask for

- Target `libactionet` version or commit
- Build environment details (Python version, compiler)
- Example pipeline calls into the library
- Expected outputs or golden test cases
