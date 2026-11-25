# Batch correction

Batch-effect correction and basal-expression correction routines.

`correct_batch_effect` removes the gene-space directions spanned by `S' D`
(`S` the cells x genes matrix, `D` the design) from a `reduce_kernel`
reduction with a rank-k SVD update. With only `batch_key`, `D` is the batch
one-hot matrix and each batch's mean deviation from the overall mean is
removed. That deviation contains:

- technical effects (protocol, site, processing date);
- differences in cell-type composition between batches;
- biology shared by every cell of a batch, such as donor disease status,
  genotype or sex;
- differences in sequencing depth between batches.

Passing `groups`, `strata_key`, `depth` or `decorrelate_from_reduction=True`
switches to within-group correction. Batch one-hots are then centred within
each group (cluster x stratum), and each option keeps one part of the batch
means out of the correction:

| Option | Effect |
|---|---|
| `groups` | Batches are compared only within cell groups, so composition differences are not removed. |
| `strata_key` | Batches are compared only within the levels of one or more covariates, so differences between those levels are kept. |
| `decorrelate_from_reduction=True` | The design is made orthogonal to the reduction's depth-tracking terms (`obsm[f"{reduction_key}_B"]`). Between-batch depth differences, and donor structure tied to them, are kept. Off by default. |
| `depth` | One extra column removes the sequencing-depth axis within groups, both within and between batches. Experimental. |
| `max_batch_frac` | Groups in which one batch has more than this fraction of the cells are merged into the nearest group, so a batch-specific cluster is not kept as biology. |

## Choosing a configuration

**Strong technical batches, similar composition.** Typical signs: several
sites, protocols or years, and batches that form separate islands without
correction.

- Use `batch_key` alone, set to the finest technical unit (library or
  sample, not donor, when a donor was sequenced more than once).
- Optionally run two passes:
  - Annotate coarse cell classes on the corrected result.
  - Re-run on the original reduction with `groups=<class labels>`. This
    matches or slightly improves vanilla mixing and reduces cross-lineage
    leakage.
- Avoid `groups="auto"` here. Clusters of an uncorrected reduction follow
  the batches, and the design then removes little of the batch effect. A
  `UserWarning` reports this.
- Avoid `decorrelate_from_reduction=True`, and avoid covariates that are
  partly technical as strata.

**Large heterogeneous collections, where batches differ in composition.**
Typical signs: dissections, regions or cell types missing from some batches,
moderate batch effects, and uneven sequencing depth.

- Use `groups="auto"` or coarse cell-class labels. Without groups, rare cell
  types present in few batches can be pulled into other lineages.
- Use `strata_key` for biologically distinct tissues or regions.
- Keep the default `decorrelate_from_reduction=False` for maximal depth
  removal and mixing. Set it to `True` to keep more donor-level structure;
  depth differences between donors are kept with it.
- Add `depth=` if a depth gradient remains within cell types.

**Little technical variation, one batch per individual.** Typical signs:
littermates processed together, one sample per animal.

- Correction mostly removes biology; consider not correcting.
- If you correct, set `strata_key` to the experimental condition (for
  example `["Genotype", "Treatment"]`).
- Use `groups` for composition outliers and `depth=` for depth gradients.

**Any variable that is constant within each batch** (disease status per
donor, genotype per animal):

- Its effect is removed unless it is a stratum.
- With `groups`, cell-type-specific effects of such variables are removed
  more completely than with `batch_key` alone. Comparing batches within a
  cell type isolates them.
- Stratify only on variables with several batches per level. With one or two
  batches per level, technical and biological differences cannot be
  separated, and the strata keep both.

## Examples

```python
import actionet as an

an.reduce_kernel(adata, n_components=30, lazy_transform=lzt)

# Technical batches: batch key alone.
an.correct_batch_effect(adata, batch_key="Library", lazy_transform=lzt)

# Heterogeneous collection: estimated cell groups, region protected.
an.correct_batch_effect(
    adata, batch_key="Donor", groups="auto", strata_key="Region",
    lazy_transform=lzt,
)

# One batch per animal: cell-class labels, condition protected, depth removed.
an.correct_batch_effect(
    adata, batch_key="Animal", groups="cell_class",
    strata_key=["Genotype", "Treatment"], depth="log_counts",
    lazy_transform=lzt,
)

# Step by step.
design = an.build_batch_design(
    adata, "Donor", groups="cell_class", strata_key="Region",
)
an.correct_batch_effect(adata, design=design, lazy_transform=lzt)
```

Within-group runs store the cluster labels after merging in
`obs[f"{corrected_key}_groups"]` and the design provenance in
`uns[f"{corrected_key}_params"]["grouped_design"]`. The provenance includes
`merged_groups` and `batch_confounding`, the diagnostic behind the warning:

- `dominated_cell_frac`: the share of cells in batch-dominated groups.
- `batch_entropy_ratio`: within-group batch entropy over overall batch
  entropy, where 1 means the groups mix batches like the whole dataset.

All corrections record the number of removed directions as
`uns[f"{corrected_key}_params"]["correction_rank"]`.

::: actionet.tools.batch_correction
    options:
      members:
        - correct_batch_effect
        - build_batch_design
        - estimate_batch_groups
        - correct_basal_expression
