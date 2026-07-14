## Primary
* Consolidate plotting paths with render backend (lets_plot or matplotlib)
* NaN and Inf support for _anndata_io
* Recreate LazyTransform from uns
* Documentation audit
* Numerical precision inconsistency
  * Mix of float32 and float64 (mainly in operator, sometimes in-mem) causes drift within and across modes
  * C++ uses 64-bit, R/Rcpp does also — always consistent
* Fix R/Python result parity
* Reorganize and consolidate code
  * Audit for dead functions
* Interactive cell selector
* UMAP points have no stroke
* `compute_archetype_feature_specificity()`: `key_added` > `key_prefix`
* Make archetype specificity and network centrality optional in `run_actionet()`
* Split _backed_persist.py 
* [Deferred] Simplify anndata 0.13 backed compatibility patch once `anndata>=0.13` is the floor
  * Currently `src/actionet/io/persist.py`, `src/actionet/io/subset.py`, and `src/actionet/io/checkpoint.py`
    filter out the `None` key from `adata.layers.keys()` via a `_real_layer_keys` helper to avoid writing
    spurious `layers/None` HDF5 groups (anndata 0.13 aliases `.X` as `layers[None]`).
  * `_init_from_reopened` in `src/actionet/io/persist.py` also unpacks the reopened AnnData into explicit
    kwargs (and drives the "init from file" branch via `filename=`) to sidestep the `X is layers[None]`
    identity check that fails when backed `_CSRDataset` wrappers are recreated per attribute access.
  * When we drop `anndata<0.13` support, revisit both workarounds: the `_real_layer_keys` helper can
    likely be inlined or removed entirely, and `_init_from_reopened` can be simplified now that
    `layers[None]` is a stable, documented alias for `.X`.
## Secondary
* Consolidate normalization code-paths
* Add network centrality to run_actionet?
* Lazy transform in-memory?
* Allow changes to lazy transform params
  * Force recompute of attributes
* compute_transparency() use scale()
* Explore more accurate and faster log approximations for JSD and lazy transform
* Combine plot_umap* paths with `raster=bool`
* Python `compute_feature_specificity()` in-mem is still garbage (maybe not?)


## Done
* ARMA_DONT_USE_WRAPPER multiple redefinition
* Parallel specificity bug (fixed???)
* Standardize key args
* Test impute features
* Implement plotFeatures
* Test backed SVD
* MatrixSource supports layers, but layers can't be backed.
  * Added validation logic
* Add pseudocount to `normalize_anndata()`
* `compute_feature_specificity` backed/in-mem parity
* Fix memory usage in `compute_network_diffusion()` (in-memory)
* Speed up plot/impute features
  * I/O bound. Probably as good as it's gonna get
* Add run_lpa/propagate_labels
* Decouple archetype specificity from run_actionet
  * Compute specificity on the fly for `impute_from_archetypes()`
* Decouple compute_network_diffusion from anndata
* Add 3D to plotly
* Optionally omit C_* and specificity matrices to reduce object size
* Document everything (OpenMP hard requirement, README overhaul, context files updated)
* annotate_cells enrichment → DataFrame
* Fixed violin/boxplot alignment
* Delete PRIMME sources, headers, vendored tree, CMake wiring, `ALG_PRIMME`, and remaining test/wrapper references (see `context/DECISIONS.md` "SVD algorithm strategy").
* Delete Feng SVD C++ sources (`svd_feng.{cpp,hpp}`), the `ALG_FENG` enum, and Feng switch cases in `runSVD`/`runSVD_Operator`. The `libactionet/wrappers_r/` copies here are reference-only and were intentionally left untouched; a reminder to patch the standalone `actionet-r` package is tracked in `src/libactionet/TODO.md`.
