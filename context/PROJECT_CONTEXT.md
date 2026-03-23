# Project Context

## Overview

This work aims to develop a multi-language computational biology data analysis toolkit:

### ACTIONet software toolchain

- C++ core library: `KellisLab/libactionet`
- R front-end: `KellisLab/actionet-r` (Rcpp)
- Python front-end: `KellisLab/actionet-python` (pybind11)

## Dependency Graph

- `libactionet` is a core dependency for both `actionet-r` and `actionet-python`
- Both front-end packages are typically used for interactive and iterative data processing and analysis
- `actionet-python` is often used in non-permissive and headless HPC environments (e.g. conda, SGE, slurm) for data processing pipelines

## Current Status

- All branches are usable but still in active development.
- Remaining work:
  - libactionet: Optimize and add critical features
  - actionet-python: Port and test core R functions
  - actionet-r: Modernize and consolidate codebase. Improve build system. Fix bugs
  - all: Document

## Build/Binding Stack

- C++: CMake
- R bindings: Rcpp
- Python bindings: pybind11
- No containerization

## Key Principles (for humans + agents)

- Changes must preserve correctness and (where applicable) parity between R and Python front-ends.
- R front-end is more feature complete. Python version is more barebones and higher performance. This may change in the future.
- Prefer spec/contract-driven interfaces across repos (I/O schemas, parameter names, output formats).
- Assume all repos are present and readable on local machine.
