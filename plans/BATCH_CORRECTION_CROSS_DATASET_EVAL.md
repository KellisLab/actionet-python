# Batch correction: cross-dataset evaluation of vanilla and grouped designs

Status: evaluation done on three datasets that span the regimes users will hit.
No default values changed. All proposals in section 7 were approved: 1–4
are implemented, and the extra atlas graphs (proposal 5) are done.

Inputs: `plans/GROUPED_BATCH_CORRECTION_HANDOFF.md` and the package at the
current branch. Scripts and outputs: `build/batchcorr_eval/` (gitignored, see
section 9). Full tables: `build/batchcorr_eval/out/report_tables.md`.

## 1. Summary

- **No configuration wins on all three datasets, and none should be expected
  to.** Vanilla correction removes each batch's mean deviation. On these data
  that deviation mixes four things:
  - technical batch effects;
  - cell-type composition;
  - biology nested in the batch (donor phenotype, genotype, sex);
  - between-batch depth differences.

  Each grouped-design parameter keeps one of these out of the correction.
  The right setting depends on which components dominate a dataset, so the
  guidance has to be by regime (section 6).
- **The current API defaults are sound and should stay:**
  - `groups=None`, i.e. vanilla;
  - `decorrelate_from_reduction=False`;
  - `max_batch_frac=0.3`;
  - `depth=None`.

  What should change is the documentation. README and `docs/api` lead with the
  atlas configuration (`groups="auto", strata_key="Region"`) as if it were
  the standard call. On MSN-like data it is the worst sensible choice.
- **MSN** (strong technical batches):
  - Grouping itself is not what hurts. Cell-class label groups match the
    pre-change vanilla on ACTIONet-graph mixing (same-UID fraction 0.212 vs
    0.213; 0.094 is perfect mixing).
  - The losses come from four places:
    - **auto groups** (0.30): the clusters follow batches;
    - **decorrelation** (0.33–0.42);
    - **disabling the merge of dominated groups** (0.65);
    - **Region strata**: they protect what is mostly donor and library
      identity.
- **Scn4b** (no technical variation):
  - Correction gives little mixing: same-mouse 0.151 → 0.127 on the graph,
    against 0.058 for perfect mixing.
  - Vanilla removes about 38% of the genotype structure and raises
    endothelial leak 12-fold.
  - Lineage or auto groups without strata remove 58–61%. Their apparent
    mixing gain (0.103) is that genotype structure being removed.
  - Lineage groups plus strata on Genotype × Condition keep all of the
    genotype structure with no leak.
  - Adding the depth column also removes most of the within-type depth axis
    (R² 0.74 → 0.21).
- **Atlas** (composition-heavy, asymmetric depth):
  - Groups are what remove the vanilla endothelial leak: 0.0147 → 0.0017 on
    the graph.
    - Lineage-label groups do the same (0.0015).
    - Region strata alone leave the leak at 0.0145.
  - The current default and the prototype trade off rather than one
    dominating:
    - The default removes more depth (embedding R² 0.30 vs 0.66) and mixes
      donors more.
    - The prototype keeps more donor-level phenotype-associated structure
      (76% vs 67%, measured within sequencing date).
  - Phenotype strata do not recover that structure. It is depth-linked donor
    structure, not a phenotype mean shift.
- **Grouping removes more batch-nested biology than vanilla, on all three
  datasets.**
  - Comparing batches within a cell type isolates cell-type-specific
    condition effects, which vanilla dilutes across the whole batch.
  - Graph-level evidence:
    - Scn4b genotype kept: 0.42 with lineage groups vs 0.62 for vanilla.
    - MSN Condition enrichment: 0.053 with label groups vs 0.097.
    - Atlas phenotype enrichment: 0.065 with the grouped default vs 0.109.
  - Strata are therefore more important when groups are used.
- **Pre-change vs new vanilla:**
  - The old build's reflected gene-mean axis raises MSN graph purity
    (0.946 vs 0.934). Adding the axis back to the new output reproduces the
    gain; removing it from the old output removes the gain.
  - On the atlas the same axis caused the endothelial-to-oligodendrocyte
    collapse: 12% graph leak vs 1.5% after the fix.
  - The fix should stay. But "prior vanilla" comparisons mix two changes:
    the C++ fix and the design.

## 2. Review of the handoff

### What holds

- The engineering:
  - C++ projection and rank detection;
  - the full Brand update;
  - parity with the prototype.

  Rescoring the stored atlas arms reproduces the handoff numbers.
- `decorrelate_from_reduction=False` as the default. MSN and Scn4b agree with
  the atlas on mixing and depth (section 5.3). The atlas trade-off belongs in
  the docs.
- The full update over the projected update. At graph level, the projected
  update buys nothing (MSN 0.202 vs 0.212 same-UID; atlas see section 5.7).
- Region strata on the atlas, and the handoff's own note that MSN auto groups
  follow donors.

### Incorrect or incomplete

1. **MSN batch structure.** The handoff describes 25 UIDs as "10 donors ×
   Caudate/Putamen".
   - The UIDs are libraries from 16 donor × region tissues. Six tissues were
     sequenced 2–3 times, with different protocols (whole tissue, BVE v1,
     BVE v2) and years (2019–2024).
   - These technical replicates give a purely technical mixing target.
   - UID is the right batch key (section 5.6).
2. **MSN region is not the biology the handoff took it for.** Within subtype,
   in the uncorrected embedding:

   | library pair | median centroid distance |
   |---|---|
   | same donor and protocol, other region | 0.13 |
   | same tissue re-sequenced | 0.17–0.26 |
   | other donors, same region | 0.256 |
   | other donors, other region | 0.276 |

   Region moves libraries only 0.02 apart; donor and processing dominate.
   Region retention under strata (0.89 in the handoff) therefore mostly
   retains library identity. Every grouped MSN arm in the handoff had Region
   strata, which is part of why grouped looked uniformly worse.
3. **Mixing was scored as monotone good.** No metric measured donor-level
   biology, so the claim that `nodecor` "beats `pkg` on every metric" holds
   only for the metrics measured. Once cross-donor phenotype structure is
   scored, it is a trade-off (section 5.3).
4. **Atlas phenotype is confounded with sequencing date.**
   - The TDP-43 cohort (ALS, FTLD) was sequenced in 2021.
   - The tau cohort (CBD, FTDP, PSP) and all PN controls were sequenced in
     2022.

   Donor-level phenotype comparisons on the atlas must condition on date.
   This report scores phenotype structure within date.
5. **Atlas-specific settings documented as general.**
   - README and `docs/api/batch_correction.md` lead with `groups="auto",
     strata_key="Region"`.
   - The README says within-group correction "removes the within-cluster
     sequencing-depth axis". The data contradict this:
     - On MSN, groups alone leave depth R² unchanged (0.607 vs 0.607).
     - Depth removal comes from `decorrelate_from_reduction=False` and the
       `depth` column.
     - Groups help depth only where composition differs (atlas 0.52 → 0.30).
6. **Batch nested in biology is not discussed.**
   - Any effective donor- or mouse-level correction removes biology nested
     in the batch.
     - Sex structure loses over 90% under vanilla and the grouped defaults on
       MSN and the atlas.
     - Scn4b genotype, graph level: 38% removed under vanilla and 58–61%
       under grouped designs without strata.
   - Strata protect nested biology when it is a mean shift:
     - Scn4b genotype: 100% kept.
     - MSN Condition: enrichment 0.62 vs 0.05–0.10.

## 3. Datasets and regimes

| | Atlas PFC/VMB | MSN (human striatum) | Scn4b (mouse) |
|---|---|---|---|
| cells | 1,154,246 | 41,007 | 300,157 |
| batch key (n) | Donor (129) | UID library (25) | UID mouse (26) |
| uncorrected same-batch kNN fraction (perfect mixing) | 0.585 (0.017) | 0.921 (0.094) | 0.254 (0.058) |
| composition: median / max TVD of a batch vs pooled | 0.26 / 0.94 | 0.06 / 0.18 | 0.13 / 0.97 |
| depth: IQR of batch-median fold / batch share of within-label log-depth variance | 1.58 / 0.35 | 1.30 / 0.38 | 2.10 / 0.24 |
| label share of log-depth variance | 0.41 | 0.01 | 0.75 |
| batch nested in | Phenotype (6), sex, sequencing date | donor, region, protocol, year, condition (HD: 2 donors) | Genotype × Condition (6–7 mice each) |
| auto groups (resolution 0.05): cells in batch-dominated clusters / batch-entropy ratio | 1.2% / 0.94 | 64% / 0.38 | 0.4% / 0.98 |
| ground truth available | region-specific types; within-date phenotype | technical replicates | genotype signature |

Notes:
- **MSN technical structure.** Protocol and year structure is fully removed by
  vanilla correction: kNN same-protocol 0.518 against 0.526 expected.
- **Scn4b composition.** The large maximum TVD comes from 6 mice whose cells
  are mostly endothelial.
- **Batch-entropy ratio.** Mean within-cluster batch entropy divided by the
  global batch entropy; 1 means clusters mix batches like the whole data.

## 4. Method

**Preprocessing**
- MSN follows `tests/test_batchcorr.ipynb`:
  - `filter_anndata(0.01)`;
  - `normalize_anndata(1e4, log2)`;
  - `reduce_kernel(30)`.
- Scn4b is the same with `min_cells_per_feat=0.001`, to keep rare
  non-neuronal types.
- Atlas:
  - stored k=50 reduction with the lazy transform;
  - arms in `build/atlas_pfc_vmb` reused;
  - 7 new arms.

**Arms**
- Every arm calls the public API: `correct_batch_effect`,
  `build_batch_design`, or `design=`.
- The pre-change vanilla is the `../actionet-pre` build on the same
  reduction.
- 30–45 arms per dataset. The sweep covers:
  - groups: none, auto at four resolutions, lineage labels, full labels;
  - strata;
  - decorrelation;
  - the depth column;
  - `max_batch_frac`;
  - batch key;
  - projected update;
  - a batch × group interaction design;
  - depth column alone.

**Two scoring levels**
- Embedding: k=30 kNN on L1-normalized rows.
- ACTIONet graph, for shortlisted arms:
  - `run_action(k_max=30)`;
  - `build_network` with the `run_actionet` defaults;
  - weighted by edge.

  The graph level is what users analyse. Several embedding-level rankings
  change there (sections 5.1, 5.7, 5.8).

**Metrics.** Lower is better for leak, mixing and depth; higher for purity
and retention.
- Label purity, lineage leak, and endothelial leak (Vasc.Endo).
- Batch mixing: same-batch fraction among same-label, same-stratum
  neighbours.
- MSN replicate mixing: same-UID fraction among neighbours from the same
  donor × region. This is purely technical.
- Strata retention.
- Donor-level biology:
  - Among same-label, same-stratum neighbours from other donors, compute the
    enrichment of same phenotype as (obs − exp)/(1 − exp).
  - "Kept" is that enrichment relative to the uncorrected embedding.
  - Atlas phenotype is scored within sequencing date.
- Depth:
  - within-label linear R² (embedding);
  - depth kNN: the correlation of a cell's centred log depth with its
    same-label neighbours' mean (embedding and graph).

**Noise and caveats**
- Re-running MSN with another SVD seed reproduces every metric to three
  decimals.
- Labels were annotated on corrected data: probably vanilla for MSN and
  Scn4b, and the prototype for the atlas. Purity therefore favours those
  configurations slightly.

## 5. Results by parameter

Values are graph level unless marked (emb). MSN mixing is same-UID,
perfect 0.094. Scn4b mixing is same-mouse, perfect 0.058. Atlas mixing is
same-donor, perfect 0.017.

### 5.1 `groups`

| | MSN mixing | MSN purity | MSN Condition enr. | Scn4b mixing | Scn4b genotype kept | Scn4b Endo leak | Atlas Endo leak (emb) | Atlas phenotype kept (emb) |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0.232 | 0.934 | 0.097 | 0.127 | 0.62 | 0.0049 | 0.0116 | 0.72 |
| lineage / cell-class labels | **0.212** | 0.937 | 0.053 | 0.103 | 0.42 | 0.0003 | 0.0017 | 0.67 |
| auto (0.05) | 0.302 | 0.937 | 0.003 | 0.103 | 0.39 | 0.0004 | 0.0024 | 0.67 |

- **Table note.** The atlas grouped rows include Region strata. Strata alone,
  without groups, leave the vanilla leak in place (graph 0.0145, emb 0.0132).
- **Atlas graph level.** Lineage-label groups match the auto-group default:
  - purity 0.882 vs 0.887;
  - endothelial leak 0.0015 vs 0.0017;
  - donor mixing 0.066 vs 0.062;
  - region retention 0.853 vs 0.848.
- **Labels.** Coarse labels (cell class or lineage) were never worse than
  vanilla. Full subtype labels behaved like lineage labels on MSN and Scn4b.
- **Auto groups need clusters that follow cell type.** On the atlas and
  Scn4b they do (ARI vs lineage 0.75 and 0.69). On MSN they follow batches
  (ARI vs lineage 0.003; 64% of cells in batch-dominated clusters).
  Clusters are then merged across batches and the correction is undone.
  Lower resolution makes it worse (0.411 at 0.02).
- **Resolution can act as strata.** On Scn4b, resolution 0.2 splits by
  genotype and keeps 100% of the genotype structure, like explicit strata.
  This is data-dependent, not something to rely on.
- **Groups remove more batch-nested biology than vanilla.**
  - A condition effect that differs by cell type (such as the HD-model
    signature in Scn4b MSNs) is diluted in a batch's overall mean. Vanilla
    therefore removes only part of it.
  - Within-group batch comparison isolates that effect, and the correction
    removes it.
  - On Scn4b, the lower mouse mixing under groups (0.103 vs 0.127) is that
    genotype structure going: with genotype strata, mixing is back at
    0.125.
  - Use groups together with strata on any condition that is constant
    within batches.

### 5.2 `strata_key`

- **Use strata for biology that is a mean shift and shared across batches.**
  - Scn4b, Genotype × Condition: genotype kept 1.00 with lineage groups,
    0.62 → 1.07 at graph level. Mixing is unchanged (0.125 vs 0.127).
  - MSN Condition: HD enrichment 0.62 vs 0.05 with groups.
    - Mixing worsens (0.288 vs 0.212) because HD has only 3 libraries from
      2 donors: their technical differences cannot be separated from HD.
  - Atlas Region: needed. Region retention is 0.918 (emb) vs 0.877 without
    strata, and region-only types stay apart.
- **Do not use technical covariates as strata.** MSN Region strata keep
  region (0.81 vs 0.59) at no mixing cost with label groups, but region there
  is mostly library identity (section 2).
- **Strata don't help depth-linked donor structure.** Atlas Region ×
  Phenotype strata keep less phenotype structure (0.63 vs 0.67, emb). They
  raise endothelial leak (0.0066 vs 0.0024) because the finer strata leave
  small groups.
- **`strata_key` takes one column.** Scn4b needed a combined
  Genotype_Condition column.

### 5.3 `decorrelate_from_reduction`

| | MSN mixing | MSN depth kNN | Atlas mixing | Atlas phenotype enr. (graph) / kept (emb) | Atlas depth R² (emb) | Atlas purity |
|---|---|---|---|---|---|---|
| False (default) | 0.212 (labels) | 0.70 | 0.062 | 0.065 / 0.67 | 0.30 | 0.887 |
| True | 0.330 (labels) | 0.86 | 0.078 | 0.100 / 0.76 | 0.66 | 0.868 |

- **MSN:** False is better on mixing and depth in every pairing. With
  decorrelation, mixing is 0.330 (labels), 0.337 (no groups) and 0.420
  (prototype).
- **Scn4b** (lineage groups, graph):
  - Decorrelation keeps more genotype structure (0.52 vs 0.42), but it also
    keeps depth (depth kNN 0.785 vs 0.688).
  - Genotype strata are the cleaner protection: 1.07 kept, depth kNN 0.682.
- **Atlas: a real trade-off.**
  - True keeps between-donor depth differences, and with them donor
    structure. That includes phenotype-associated structure within
    sequencing date.
  - Vanilla keeps similar phenotype structure to the prototype (graph
    0.109, emb 0.72), but it leaks endothelium. The phenotype structure lost
    under the default is partly the grouping effect of section 5.1;
    decorrelation offsets it.
  - Whether that structure is biology is unresolved. Neurodegeneration can
    change RNA content, and phenotype strata did not recover it.
  - The handoff's conclusion that False is better on every metric does not
    hold once this is measured.

### 5.4 `depth`

| | MSN depth kNN | MSN mixing | Scn4b depth R² (emb) | Scn4b genotype kept | Atlas depth kNN | Atlas purity |
|---|---|---|---|---|---|---|
| without | 0.695 (labels) | 0.212 | 0.606 (labels + strata) | 1.00 | 0.684 | 0.887 |
| with | 0.457 | 0.228 | 0.214 | 1.00 | 0.659 | 0.879 |

- **Small cost, large depth reduction.** The cost is about 0.01–0.02 in
  mixing or purity; depth falls sharply. Genotype is untouched on Scn4b.
- **Depth alone gives no batch mixing** (MSN 0.854, atlas 0.582 emb).
- **Within-batch depth goes too.** The column removes within-batch depth
  variation as well as between-batch differences, so it should stay opt-in.
  Depth often tracks real cell-state differences, such as activity or cell
  size.

### 5.5 `max_batch_frac` and auto-group resolution

- **`max_batch_frac=0.3` is best or tied everywhere.**
  - On MSN, 0.5 gives mixing 0.46 and None gives 0.65: dominated clusters
    left unmerged keep their batch identity.
  - On the atlas, None is about equal (emb).
- **Resolution:**
  - 0.05 is right for the atlas.
  - On Scn4b, finer resolutions split by genotype (section 5.1).
  - On MSN, every resolution follows batches.

### 5.6 `batch_key`

- **Use the finest technical unit** that is not nested in the biology of
  interest.
  - MSN UID vs Donor: mixing 0.212 vs 0.46, replicate mixing 0.62 vs 0.82.
    Libraries of one donor differ by protocol and year.
  - Atlas, where UID is donor × region: Donor is better (emb).
    - Vanilla with UID lowers purity (0.907 vs 0.916) and triples
      endothelial leak (0.036 vs 0.012).
    - With Region strata the gap is smaller: purity 0.911 vs 0.920, mixing
      about equal.

### 5.7 Projected update and interaction design

- **Projected update** (`K_k (I − ZZ')`):
  - Better embedding mixing, but it keeps depth (MSN emb depth R² 0.80 vs
    0.61).
  - At graph level it is about equal. MSN mixing is 0.202 vs 0.212. On the
    atlas, in the prototype configuration, purity is 0.874 vs 0.868 and
    mixing 0.077 vs 0.078.
  - There is no reason to expose it.
- **Batch × group interaction design** (per-group batch effects):
  - It over-corrects once its rank exceeds k: Scn4b genotype kept 0.20.
  - It gives no MSN gain (0.244).

### 5.8 Pre-change vanilla (reflected gene-mean axis)

| MSN graph | purity | mixing |
|---|---|---|
| old vanilla | 0.946 | 0.213 |
| old, axis removed | 0.936 | 0.202 |
| new vanilla | 0.934 | 0.232 |
| new, axis added | 0.945 | 0.225 |

- **Why the axis matters.** The pre-change build re-adds the all-positive
  gene-mean axis as a dominant component: σ₁ is 16× the next on MSN. ACTION
  L1-normalizes embedding rows, so the axis acts as an implicit per-cell
  normalization.
- **MSN.** The axis explains the old build's purity advantage. The mixing
  difference comes from the rest of the fix and favours the fixed
  projection.
- **Atlas.** The axis concentrates endothelial leak, as in the handoff's
  Endo→Oligo finding:
  - graph: 0.120 old vs 0.015 new;
  - embedding: 0.024 old vs 0.012 new.

  It also costs region retention on the graph (0.729 vs 0.761).
- **Scn4b.** No purity gain from the axis: 0.976 old and with the axis
  added, vs 0.978 new.
- **Conclusion.** Keep the fix. Treat any remaining "old vanilla looks
  better on homogeneous data" comparisons as this effect.

## 6. Proposed user guidance

Draft text for `docs/api/batch_correction.md` (replacing the atlas-first
example) and the README section.

### What correction removes

`correct_batch_effect(adata, batch_key)` removes each batch's mean deviation
from the overall mean. That deviation contains:
- technical effects;
- differences in cell-type composition;
- biology shared by all cells of a batch (donor condition, genotype, sex);
- between-batch sequencing-depth differences.

Each option keeps one of these out of the correction:

| option | effect |
|---|---|
| `groups` | Compare batches only within cell groups, so composition differences are not removed. |
| `strata_key` | Compare batches only within a covariate's levels, so differences between those levels are kept. |
| `decorrelate_from_reduction=True` | Keep between-batch depth differences, and donor structure tied to them. |
| `depth` | Also remove the within-group sequencing-depth axis, within and between batches. |
| `max_batch_frac` | Merge groups dominated by one batch, so a batch-specific cluster is not mistaken for biology. |

### Choosing a configuration

1. **Strong technical batches with similar composition.**
   - Typical signs: multiple sites, protocols or years, and batches that form
     separate islands without correction.
   - Use vanilla: `batch_key` set to the finest technical unit (library or
     sample).
   - Optionally use a two-pass workflow:
     - Cluster or annotate the vanilla result coarsely.
     - Re-run with `groups=<coarse labels>` on the original reduction. This
       equals or slightly improves vanilla and reduces lineage leak.
   - Avoid `groups="auto"`: clusters of an uncorrected embedding follow the
     batches.
   - Avoid `decorrelate_from_reduction=True` and covariates that are partly
     technical as strata.
2. **Large heterogeneous collection where batches differ in composition.**
   - Typical signs: dissection, regions, or cell types missing from some
     batches; moderate batch effects; asymmetric depth.
   - Use `groups="auto"` (or coarse labels).
   - Use `strata_key` for biologically distinct tissues or regions that are
     present in only some batches.
   - Keep `decorrelate_from_reduction=False` for maximal depth removal and
     mixing. Use `True` to keep more donor-level structure: depth
     differences between donors are kept with it.
   - Add `depth=` if a depth gradient remains within cell types.
3. **Little technical variation, with batch equal to individual nested in
   condition.**
   - Typical signs: littermates processed together, one sample per animal.
   - Correction mostly removes biology; consider not correcting.
   - If correcting, always set `strata_key` to the condition (combine
     genotype × treatment into one column). Use `groups` (labels or
     `"auto"`) for composition outliers, and `depth=` for depth gradients.
   - Never use `groups` without condition strata here. Within-group
     comparison removes more of the condition than vanilla does.
4. **Any design where a variable of interest is constant within each
   batch** (disease status per donor, genotype per animal):
   - Its mean effect is removed unless it is a stratum. With `groups`, its
     cell-type-specific effects are removed more completely than under
     vanilla.
   - Stratify only on variables with several batches per level. With one or
     two batches per level, technical and biological differences cannot be
     separated, and strata keep both.

**Checks before choosing**
- **Batch strength.** Look at uncorrected batch islands, or the kNN
  same-batch fraction vs the batch-size expectation.
- **Composition.** Look at per-batch cell-class proportions.
- **Nesting.** Crosstab batch against condition.
- **Auto groups.** Check whether they follow batches: look at
  `uns[f"{key}_params"]["grouped_design"]` merged groups. If many clusters
  were merged, use labels or vanilla.

## 7. Proposed changes

Items 1–4 are implemented:
- **Docs:** `docs/api/batch_correction.md` and README.
- **Warning and list strata:** `_batch_design`, with tests in
  `tests/test_batch_correction_grouped.py`.
- **Decision record:** `context/DECISIONS.md`.

The warning also applies to user-supplied groups. Label groupings on all
three datasets have a batch-entropy ratio of 0.95 or higher and at most 0.4%
of cells in dominated groups, so they do not trigger it.

1. **Docs.**
   - Rewrite the batch-correction section of `docs/api/batch_correction.md`
     and README around section 6, with vanilla first.
   - Remove the README claim that within-group correction removes the
     depth axis.
2. **Auto-group warning.**
   - In `_batch_design`, compute the share of cells in batch-dominated
     clusters and the batch-entropy ratio. Warn when the share exceeds 20%
     or the ratio is below 0.7.
   - Record both in `grouped_design` provenance.
   - The thresholds separate MSN (64%, 0.38) from the atlas (1.2%, 0.94)
     and Scn4b (0.4%, 0.98).
   - The separation holds at resolutions 0.02 to 1.0:
     - MSN: 45–98% of cells in dominated clusters, ratio 0.20–0.43.
     - Scn4b: 0.4%, ratio 0.84–0.98. The lower values come from the
       genotype split at finer resolution.
3. **`strata_key` accepts a list of columns**, combined internally.
4. **`context/DECISIONS.md`.** Add this evidence under batch correction:
   - decorrelation off confirmed on MSN and Scn4b;
   - the atlas phenotype and depth trade-off recorded;
   - auto groups not a general default;
   - donor-nested biology needs strata, especially with groups.
5. **Atlas graph runs for lineage-label groups and strata-only.** Done; they
   confirm the embedding results (section 5.1, section 8).
   - Measured cost per run at 16 threads, one at a time on the 48-core
     machine:
     - ACTION: 76–83 min;
     - network: 45–50 min;
     - peak RSS: 48–67 GB.
   - Disk: about 10 GB per graph.

No default values change under these proposals.

## 8. Main tables (graph level)

Generated by `build/batchcorr_eval/report_tables.py`. "enr" is the
cross-donor same-phenotype enrichment; "kept" is relative to uncorrected.
Embedding tables, and atlas arms without graphs, are in `out/report_tables.md`.

### MSN (perfect mixing: same UID 0.094; replicates 0.436)

| configuration | purity | lineage leak | same UID | same UID (replicates) | same Region | Condition enr | Sex kept | depth kNN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| uncorrected | 0.896 | 0.0313 | 0.869 | 0.851 | 0.873 | -0.44 | 1 | 0.803 |
| vanilla, pre-change build | 0.946 | 0.0073 | 0.213 | 0.612 | 0.583 | 0.084 | 0.011 | 0.68 |
| vanilla | 0.934 | 0.0074 | 0.232 | 0.633 | 0.592 | 0.097 | -0.008 | 0.7 |
| groups=CellType | 0.937 | 0.0054 | 0.212 | 0.621 | 0.585 | 0.053 | -0.028 | 0.695 |
| groups=full_label | 0.94 | 0.005 | 0.213 | 0.619 | 0.585 | 0.049 | -0.02 | 0.696 |
| groups=CellType, strata=Region | 0.938 | 0.0053 | 0.209 | 0.608 | 0.812 | 0.05 | -0.022 | 0.673 |
| groups=auto | 0.937 | 0.0066 | 0.302 | 0.66 | 0.644 | 0.003 | 0.011 | 0.703 |
| groups=auto, strata=Region | 0.937 | 0.0063 | 0.303 | 0.654 | 0.809 | 0.032 | 0.004 | 0.701 |
| groups=auto (resolution 0.02) | 0.916 | 0.0141 | 0.411 | 0.658 | 0.635 | -0.08 | -0.045 | 0.719 |
| groups=auto, max_batch_frac=0.5 | 0.934 | 0.0066 | 0.46 | 0.675 | 0.632 | -0.132 | 0.532 | 0.762 |
| groups=auto, max_batch_frac=None | 0.936 | 0.0064 | 0.646 | 0.669 | 0.733 | -0.705 | 1.137 | 0.722 |
| decorrelate (no groups) | 0.932 | 0.0081 | 0.337 | 0.666 | 0.643 | 0.084 | 0.055 | 0.857 |
| groups=CellType, decorrelate | 0.933 | 0.0067 | 0.33 | 0.655 | 0.637 | 0.064 | 0.073 | 0.857 |
| groups=auto, strata=Region, decorrelate (prototype) | 0.933 | 0.0073 | 0.42 | 0.695 | 0.81 | 0.072 | 0.074 | 0.846 |
| depth (no groups) | 0.937 | 0.0071 | 0.232 | 0.646 | 0.588 | 0.071 | -0.001 | 0.481 |
| groups=CellType, depth | 0.937 | 0.0052 | 0.228 | 0.634 | 0.589 | 0.064 | -0.023 | 0.457 |
| groups=CellType, strata=Condition | 0.938 | 0.0053 | 0.288 | 0.64 | 0.605 | 0.616 | 0.042 | 0.721 |
| vanilla, batch_key=Donor | 0.931 | 0.0072 | 0.46 | 0.819 | 0.737 | -0.062 | 0.155 | 0.715 |
| groups=CellType, projected update | 0.937 | 0.005 | 0.202 | 0.605 | 0.586 | 0.035 | -0.02 | 0.708 |
| UID x CellType interaction design | 0.936 | 0.0052 | 0.244 | 0.659 | 0.602 | 0.037 | -0.031 | 0.714 |
| depth column only (no batch) | 0.918 | 0.0139 | 0.854 | 0.848 | 0.874 | -0.399 | 0.356 | 0.708 |

### Scn4b (perfect mixing: same mouse 0.058)

| configuration | purity | lineage leak | Endo leak | same mouse | Genotype kept | depth kNN |
|:--|--:|--:|--:|--:|--:|--:|
| uncorrected | 0.975 | 0.0006 | 0.0004 | 0.151 | 1 | 0.756 |
| vanilla, pre-change build | 0.976 | 0.001 | 0.008 | 0.122 | 0.612 | 0.681 |
| vanilla | 0.978 | 0.0008 | 0.0049 | 0.127 | 0.62 | 0.694 |
| groups=lineage | 0.979 | 0.0006 | 0.0003 | 0.103 | 0.416 | 0.688 |
| groups=auto | 0.979 | 0.0006 | 0.0004 | 0.103 | 0.389 | 0.689 |
| groups=auto (resolution 0.2) | 0.978 | 0.0005 | 0.0004 | 0.124 | 1.018 | 0.681 |
| strata=Genotype_Condition (no groups) | 0.977 | 0.0006 | 0.0027 | 0.141 | 1.023 | 0.725 |
| groups=lineage, strata=Genotype_Condition | 0.978 | 0.0006 | 0.0002 | 0.125 | 1.073 | 0.682 |
| groups=auto, strata=Genotype_Condition | 0.977 | 0.0006 | 0.0003 | 0.125 | 1.106 | 0.685 |
| groups=lineage, decorrelate | 0.978 | 0.0006 | 0.0006 | 0.127 | 0.516 | 0.785 |
| groups=lineage, depth | 0.977 | 0.0006 | 0.0002 | 0.099 | 0.361 | 0.647 |
| groups=lineage, strata=Genotype_Condition, depth | 0.978 | 0.0006 | 0.0002 | 0.124 | 1.061 | 0.659 |

### Atlas (perfect mixing: same donor 0.017; phenotype scored within sequencing date)

No uncorrected atlas graph exists, so the phenotype and sex columns are raw
enrichments, not "kept".

| configuration | purity | lineage leak | Endo leak | same Donor | same Region | Phenotype enr | Sex enr | depth kNN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| vanilla, pre-change build | 0.879 | 0.0041 | 0.1198 | 0.068 | 0.729 | 0.084 | 0.011 | 0.743 |
| vanilla | 0.886 | 0.0017 | 0.0147 | 0.107 | 0.761 | 0.109 | 0.016 | 0.749 |
| strata=Region (no groups) | 0.887 | 0.0017 | 0.0145 | 0.103 | 0.865 | 0.111 | 0.019 | 0.756 |
| groups=leiden_unc, strata=Region (current default config) | 0.887 | 0.001 | 0.0017 | 0.062 | 0.848 | 0.065 | 0.012 | 0.684 |
| groups=lineage, strata=Region | 0.882 | 0.001 | 0.0015 | 0.066 | 0.853 | 0.067 | 0.012 | 0.695 |
| groups=leiden_unc, strata=Region, decorrelate (prototype) | 0.868 | 0.0013 | 0.0021 | 0.078 | 0.84 | 0.1 | 0.011 | 0.772 |
| stored prototype result | 0.872 | 0.0011 | 0.0018 | 0.083 | 0.835 | 0.104 | 0.012 | 0.79 |
| groups=leiden_unc, strata=Region, depth | 0.879 | 0.0009 | 0.0016 | 0.057 | 0.835 | 0.057 | 0.01 | 0.659 |

## 9. Reproduction

All scripts are in `build/batchcorr_eval/`, run with the repo `.venv`
(`old_plain.py` with `../actionet-pre/.venv`):

```bash
python characterize.py {atlas,msn,scn4b}     # regime descriptors
python prep.py {msn,scn4b}                    # filter, normalize, reduce_kernel(30)
python arms.py {msn,scn4b}                    # correction arms (public API)
/data/git_projects/actionet-pre/.venv/bin/python old_plain.py {msn,scn4b}
python atlas_arms.py                          # new atlas arms; reuses build/atlas_pfc_vmb
python diag_axis.py; python seed_replicate.py msn
python metrics.py {msn,scn4b,atlas}           # embedding metrics (cached kNN)
python graph_arm.py DATASET ARM...            # ACTIONet graph + graph metrics
bash run_atlas_metrics.sh                     # atlas rescoring incl. stored graphs
python summarize.py DATASET [--graph]; python report_tables.py
```
