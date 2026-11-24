# Batch correction

Batch-effect correction and basal-expression correction routines.

`correct_batch_effect` removes the gene-space directions spanned by `S' D`
(`S` the cells x genes matrix, `D` the design) from a `reduce_kernel`
reduction with a rank-k SVD update. With only `batch_key`, `D` is the batch
one-hot matrix. Passing `groups`, `strata_key`, `depth` or
`decorrelate_from_reduction=True` switches to within-group correction:

- Batch one-hots are centred within each group (cluster x stratum), so
  differences in cell-type or region composition between batches are not
  removed as batch effects.
- Groups dominated by one batch (more than `max_batch_frac` of their cells)
  are merged into the group with the nearest centroid.
- `decorrelate_from_reduction=True` makes the design orthogonal to the
  reduction's cell-side centring terms (`obsm[f"{reduction_key}_B"]`), as in
  the original prototype. It is off by default. Those terms track
  sequencing depth, so with it on, most within-batch depth variation is left
  in the corrected embedding.

```python
import actionet as an

an.reduce_kernel(adata, n_components=30, lazy_transform=lzt)

# Groups estimated from a coarse kNN + Leiden clustering of the reduction,
# region protected as a stratum.
an.correct_batch_effect(
    adata, batch_key="Donor", groups="auto", strata_key="Region",
    lazy_transform=lzt,
)

# Step by step, with user-provided cluster labels.
design = an.build_batch_design(
    adata, "Donor", groups="cell_class", strata_key="Region",
)
an.correct_batch_effect(adata, design=design, lazy_transform=lzt)
```

Within-group runs store the cluster labels after merging in
`obs[f"{corrected_key}_groups"]` and the design provenance in
`uns[f"{corrected_key}_params"]["grouped_design"]`. All corrections record the
number of removed directions as `uns[f"{corrected_key}_params"]["correction_rank"]`.

::: actionet.tools.batch_correction
    options:
      members:
        - correct_batch_effect
        - build_batch_design
        - estimate_batch_groups
        - correct_basal_expression
