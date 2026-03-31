## Primary
* Numerical percision inconsistency
  * Mix of float64 and float64 (mainly in operator, sometimes in-mem) causes drift within and across modes
  * C++ uses 64-bit, R/Rcpp does also - always consistent
* Fix R/Python result parity
* obs label as legend title
* Test plot umap interactive
* Document everything
* Add run_lpa/propagate_labels
* Decouple compute_network_diffusion from anndata
* Change output from numpy to pandas (annotate_*)
* Reorganize and consolidate code

## Secondary
* Normalize with C++ API (nah?)
* compute_transparency() use scale()


## Done
* ARMA_DONT_USE_WRAPPER multiple redefinition
* Parallel specificity bug (fixed???)
* Standardize key args
* Test impute features
* Implement plotFeatures
* test backed SVD
