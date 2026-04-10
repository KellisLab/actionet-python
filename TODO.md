## Primary
* recreate LazyTransform from uns
* Numerical percision inconsistency
  * Mix of float64 and float64 (mainly in operator, sometimes in-mem) causes drift within and across modes
  * C++ uses 64-bit, R/Rcpp does also - always consistent
* Fix R/Python result parity
* Document everything
* Change output from numpy to pandas (annotate_*)
* Reorganize and consolidate code
* * Audit for dead functions.
* Interactive cell selector

## Secondary
* Consolidate normalization code-paths
* Add network centrality to run_actionet?
* Lazy transform in-memory?
* Allow changes to lazy transform params
  * Force recompute of attributes
* `lazy_logcounts` in `_validate_lazy_logcounts_params()` does nothing?
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
* test backed SVD
* MatrixSource supports layers, but layers can't be backed.
  * Added validation logic
* Add pseudocount to `normalize_anndata()`
* `compute_feature_specificity` backed/in-mem parity
* Fix memory usage in `compute_network_diffusion()` (in-memory)
  * Probably fixed???
* Speed up plot/impute features
  * I/O bound. Probably as good as it's gonna get
* Add run_lpa/propagate_labels
* Decouple archetype specificity from run_actionet
  * Compute specificity on the fly for `impuimpute_from_archetypes()`
* Decouple compute_network_diffusion from anndata
* Add 3D to plotly
* Optionally omit C_* and specificity matrices to reduce object size
