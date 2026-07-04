# ACTIONet (Python)

Python front-end for **ACTIONet** — a single-cell multi-resolution data analysis
toolkit built on the [`libactionet`](https://github.com/KellisLab/libactionet)
C++ core, with [AnnData](https://anndata.readthedocs.io/) as the primary data
container.

## Quick start

```bash
git clone https://github.com/KellisLab/actionet-python.git
cd actionet-python
git submodule update --init --recursive
pip install -e .
```

```python
import actionet as an
import anndata as ad

adata = ad.read_h5ad("my_data.h5ad")
an.run_actionet(adata)          # full pipeline
an.plot_umap(adata, color="assigned_archetype")
```

See the [Guides](guide_calling.md) for end-to-end examples and the
[API Reference](api/index.md) for every public function.

## Scope

This site documents the **Python front-end only**:

- The public Python API defined by `__all__` in `actionet/__init__.py`.
- Python-specific concepts: AnnData integration, `LazyTransform`, backed
  (HDF5-streamed) persistence, GPU error taxonomy.
- Install and build instructions for the Python package.
- Python-user-facing guides.

Not documented here (see the respective repositories):

- **C++ core library** — classes, headers, and algorithms live in
  [`libactionet`](https://github.com/KellisLab/libactionet). Both the Python
  and R front-ends bind against the same C++ contract.
- **R front-end** — see [`actionet-r`](https://github.com/KellisLab/actionet-r).
- **pybind11 wrappers** (`wp_*.cpp`, `_core.cpp`) are implementation detail and
  intentionally hidden from this reference.

For a picture of how the layers fit together, see [Architecture](architecture.md).

## Where to go next

- New to ACTIONet? Start with **[Architecture](architecture.md)** for a 30-second
  mental model.
- Running the full pipeline? See **[`run_actionet`](api/pipeline.md)**.
- Building step-by-step? See **[Core](api/core.md)** and **[Reduction](api/reduction.md)**.
- Plotting? See **[Plotting](api/plotting.md)**.

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open <http://127.0.0.1:8000> for a live-reloading preview.
