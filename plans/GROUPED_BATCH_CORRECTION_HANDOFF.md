# Within-group batch correction: implementation handoff

Status: implemented and tested on synthetic data and MSN. Validated on the
PFC/VMB subset of the atlas (section 4c):
- The grouped path reproduces the prototype result.
- The plain `batch_key` path still connects dissimilar lineages.

Decisions after validation (recorded in DECISIONS.md):
- `decorrelate_from_reduction` now defaults to `False`, and
  `correct_batch_effect` exposes it.
- The full Brand update stays.
- The plain path is unchanged.

All changes are **uncommitted** on branch `dev` in both repositories:
actionet-python on top of `e9e33a7`, libactionet on top of `92c0e2e`.

---

## 1. What changed

### libactionet (C++), `src/libactionet`

- `orthogonalization.cpp` was rewritten around one core,
  `orthogonalize_core_(MatrixOperator, reduction, Z_raw)`, shared by every
  entry point (batch/basal; dense, sparse, sparse labels, operator; field and
  struct layouts). Steps:
  1. `Z_raw = S' D`.
  2. Project `Z_raw` off `span([A_prior, 1_g])`.
  3. Orthonormalize with rank detection. If the rank is 0, return the
     reduction unchanged.
  4. Apply the Brand update with `B = -S Z`.
  - The intercept augmentation is deleted, because it is identically zero
    after step 2.
  - `deflate_reduction_struct_/field_`, `orthogonalize_common_*` and
    `ORTHOG_SMALL_C_THRESHOLD` are removed. Public signatures are unchanged.
  - The correction rank is printed (`Correction rank: r of q requested
    directions`).
- New helpers in `utils_decomp.{hpp,cpp}`: `orthonormal_basis` (SVD,
  relative tolerance `kRankRelTol = 1e-10`), `orthonormal_factor` and
  `max_column_norm`.
  - `orthonormal_factor` keeps the old fast path: Gram-Schmidt for at most 16
    columns, `qr_econ` above that. It falls back to an SVD factorization only
    when a column is dependent.
- `perturbedSVD` (`svd_main.cpp`) uses `orthonormal_factor` for `P` and `Q`.
- `docs/batch_orthogonalization_algorithm.md` was rewritten (orientation,
  steps, rank detection, what the update represents). `TODO.md` gained the R
  follow-up.

**Bugs fixed:**

- **Few-donor bug.** With the prototype design, `Z ⊥ 1_g`, so the
  intercept column was about 1e-13. With at most 16 correction columns, the
  `gram_schmidt` 1e-4 absolute threshold then zeroed that column *and every
  later column*. That discarded the cell-side residual, and the result was
  effectively the projected rank-k update.
  - Above 16 columns, `qr_econ` (with no rank check) applied the full Brand
    update.
  - Now every column count applies the full update.
- **Reflection bug.** One-hot designs are dominated by the gene-mean
  direction `a1`. Deflating them removed `a1` a second time and reflected
  `s = S a1` back into the embedding as a depth axis.

### actionet-python

- `tools/batch_correction.py`:
  - `estimate_batch_groups(adata, reduction_key="action", k=30,
    resolution=0.05, distance_metric="l2", ...)` builds a kNN graph
    (`algorithm="knn"`, not mutual) and runs Leiden. Nothing is written to
    `obsp`.
  - `build_batch_design(adata, batch_key, groups=None, strata_key=None,
    depth=None, max_batch_frac=0.3, reduction_key="action",
    decorrelate_from_reduction=False, group_kwargs=None, key_added=None)`.
    The default was `True` until the atlas validation (section 4c).
    `groups` accepts an obs key, an array, `"auto"` or `None`.
  - Design steps:
    1. Merge dominated clusters into the nearest centroid.
    2. Form groups as cluster × stratum.
    3. Centre one-hots within groups (bincount, no pandas groupby).
    4. Add the optional centred depth column.
    5. With `decorrelate_from_reduction=True`, project off
       `[1_n, obsm[key_B]]` with a pivoted QR.
  - The prototype's eigh-whitening is gone, because the C++ handles rank
    deficiency.
  - `max_batch_frac` must be greater than `1/n_batches`; `None` disables
    merging.
  - `correct_batch_effect` gained `groups`, `strata_key`, `depth`,
    `max_batch_frac`, `group_kwargs` and `decorrelate_from_reduction`.
    Setting any of `groups`, `strata_key`, `depth`, or
    `decorrelate_from_reduction=True` builds the design from `batch_key`.
    - Outputs: `obs[f"{corrected_key}_groups"]` (cluster labels after
      merging) and `uns[f"{corrected_key}_params"]["grouped_design"]` (group
      source, `strata_key`, `depth`, `max_batch_frac`,
      `decorrelate_from_reduction`, merge map, `n_groups`, `n_batches`,
      design columns).
    - Every correction now records `correction_rank`.
- `network/build.py`: added the private `_build_network_matrix` (returns CSR,
  no AnnData). `network/cluster.py`: added the private `_leiden_membership`.
  `build_network` / `cluster_network` call these and behave as before.
- Exports were added in `tools/__init__.py` and `actionet/__init__.py`.
- Docs: `docs/api/batch_correction.md`, README batch-correction section
  (which also fixes the wrong `corrected_key=` argument), and a new
  `context/DECISIONS.md` entry "Batch correction".
- Tests and tools:
  - `tests/test_batch_correction_grouped.py` (21 tests).
  - `tests/_batch_design_reference.py`: verbatim copy of the prototype
    function, plus a synthetic generator.
  - `tests/fixtures/batch_correction_grouped_reference.npz`: golden outputs
    from the pre-change build, written by
    `tests/fixtures/capture_batch_correction_reference.py`. They are in
    `fixtures/` because `.gitignore` excludes every `data/` directory.
  - `tests/benchmark_batch_correction.py --mode grouped`: times each stage.
  - `tests/validate_grouped_batch_correction_atlas.py`: the atlas validation
    driver (section 4).

### Behavior changes (breaking, recorded in DECISIONS.md)

- Plain `batch_key`, `design` and basal corrections no longer reflect the
  gene-mean axis. On the synthetic regression data, `sigma` changed from
  `[46.37, 36.20, 30.62, 29.07, 17.37, …]` to `[36.64, 30.65, 29.22, 17.42,
  …]`: the reflected leading axis is gone.
- Prototype-style designs with at most 16 correction columns (at most 16
  donors) now apply the full update instead of the projected one.
- R is affected once it bumps the submodule (tracked in
  `src/libactionet/TODO.md`).

---

## 2. Verified locally (macOS arm64, synthetic data)

| Check | Result |
|---|---|
| `reduce_kernel` vs pre-change build | bitwise identical (`S_r`, `U`, `A`, `B`, `sigma`) |
| Prototype design, 20 donors (QR path) vs golden | sigma diff 5.7e-14, max angle 1.9e-6° |
| `build_batch_design` vs prototype design | same column span, max principal angle 2.9e-13° |
| Grouped `correct_batch_effect` vs golden | sigma diff 1.1e-13 |
| 8 donors (formerly the Gram-Schmidt path) | old golden was more than 1 sigma unit away from full Brand; new output matches a numpy Brand oracle to rtol 1e-8 |
| New history columns | orthonormal, ⊥ `a1` and ⊥ `1_g` (< 1e-12); `B = -S Z` |
| Rank-deficient design (centred one-hots + depth) | rank 8 vs 7 without depth; depth direction is removed |
| Backed vs in-memory, `groups="auto"` + strata + depth | sigma rtol 1e-8; no `obsp` written; provenance round-trips through h5ad |
| `estimate_batch_groups` on 4 synthetic clusters | exact recovery |
| Full suite | 676 passed, 1 skipped, 1 failed (pre-existing, see below) |

The one failure, `tests/backed/test_backed_parity.py::test_annotate_cells_backed_handles_retryable_open_conflict`,
is unrelated. A test double (`object()`) reaches
`_core.close_backed_operator` in `src/actionet/io/operator.py`, a file this
work does not touch.

---

## 3. Moving to the atlas machine

1. Commit libactionet first, then the parent:
   ```bash
   git -C src/libactionet add -A && git -C src/libactionet commit -m "Project batch basis off prior history; rank-revealing orthonormalization"
   git -C src/libactionet push origin dev
   git add -A src/libactionet src/actionet tests README.md docs context plans
   git commit -m "Within-group batch correction"
   git push origin dev
   ```
   Make sure `tests/fixtures/` and `tests/_batch_design_reference.py` are
   included. `new_batch_corr.qmd` is untracked scratch; add it only if wanted.
2. New build on the atlas machine (OpenMP, BLAS/LAPACK and HDF5 as in the
   README):
   ```bash
   git clone git@github.com:KellisLab/actionet-python.git && cd actionet-python
   git checkout dev && git submodule update --init --recursive
   python -m venv .venv && .venv/bin/pip install -e .
   .venv/bin/pip install pytest psutil
   ```
   After the first install, rebuild with
   `.venv/bin/python -m pip install -e . --no-deps`. A plain
   `cmake --build build/...` fails because the editable build uses an
   isolated environment that pybind11 is not visible from.
3. Pre-change build for the capture stage, in a separate worktree and venv:
   ```bash
   git worktree add ../actionet-pre e9e33a7
   git -C ../actionet-pre submodule update --init --recursive   # pins libactionet 92c0e2e
   python -m venv ../actionet-pre/.venv
   ../actionet-pre/.venv/bin/pip install -e ../actionet-pre psutil
   ```
4. Sanity check on the new build:
   `.venv/bin/python -m pytest tests/test_batch_correction_grouped.py tests/test_batch_correction_parity.py -q`.
   The golden tolerances (rtol 1e-8) leave room for BLAS differences between
   machines. If they fail on x86/MKL, look at the size of the gap before
   suspecting the algorithm.

---

## 4. Outstanding validation (atlas)

Run the driver twice, with the same input and working directory:

```bash
# Pre-change build: replays new_batch_corr.qmd (Donor / Region, k=30, crude kNN k=30, resolution 0.05)
../actionet-pre/.venv/bin/python tests/validate_grouped_batch_correction_atlas.py capture \
    --input tmp/adata_agg_ALL_fil.h5ad --workdir validation/

# New build
.venv/bin/python tests/validate_grouped_batch_correction_atlas.py compare \
    --input tmp/adata_agg_ALL_fil.h5ad --workdir validation/ \
    --label-key <cell-type column> --check-reduce-kernel
```

How the driver works:

- Each stage copies the input into `--workdir` before filtering (filtering a
  backed file rewrites it), then disables auto-persist, so the arms never
  rewrite the file.
- `capture` saves the reduction, the crude groups and the prototype and plain
  outputs to `capture.npz`. `compare` reuses them, so the arms differ only in
  design and update.
- Outputs: `validation/compare.json` and `validation/compare.md`.
- Disk: two working copies of the atlas. RAM: on the order of
  `n_obs × (30 + n_donors) × 8` bytes per arm, plus the design.

Compare arms: `nb` (prototype design), `pkg` (`build_batch_design`), `auto`
(`estimate_batch_groups` + design), `proj` (projected update, numpy),
`nodecor` (no decorrelation), `depth` (opt-in depth column),
`nodecor_depth` (depth column without decorrelation) and `plain`.

The checks, in priority order:

1. **Regime and parity.** `capture.json → n_donors`.
   - **More than 16 donors** (expected; the user reported more than 15):
     `nb_vs_capture` should agree to about 1e-10 relative in sigma, with
     tiny angles. `pkg_vs_nb` should agree to about 1e-12.
   - **Exactly 16 or fewer:** the validated atlas result came from the
     projected path. `nb_vs_capture` will differ, and `capture_nb` should
     instead be close to the `proj` arm. That changes which update is "the
     validated one"; resolve it before anything else.
2. **Full vs projected update.** This is the most important open question.
   On synthetic data (table below), the full update raised the same-donor
   kNN fraction above the uncorrected embedding, while `proj` approached the
   perfect-mixing value and preserved region better. `proj` did not reduce
   within-group depth signal, though.
   - Compare `pkg` and `proj` on: kNN same-donor fraction vs
     `expected_same_donor_frac`, same-region and same-label fractions, and
     `depth_r2_within_groups`.
   - Then run `an.run_actionet` on both (e.g. write `proj` into
     `obsm["action_proj"]`) and compare archetypes and layout with the
     notebook result.
   - Outcome: keep full Brand, or add an `update="projected"` option (C++:
     skip `P`, use `K_k(I - ZZ')`; one data pass) and possibly make it the
     default. This decision needs human sign-off (DECISIONS.md).
3. **Group estimation.** `arms.auto.ari_vs_captured_groups` and
   `auto_vs_pkg`. HNSW with multiple threads is not bit-reproducible, so
   expect ARI below 1. Check that the downstream metrics stay within noise of
   `pkg`. Also review `merged_groups` (which clusters were dominated and
   where they went).
4. **Decorrelation default (`decorrelate_from_reduction=True`).** Compare
   `nodecor` with `pkg` on `depth_r2_within_groups`, `depth_max_abs_corr`
   and the kNN metrics. The evidence conflicts:
   - Synthetic data favoured decorrelation (within-group depth R² 0.095
     without vs 0.034 with).
   - The real MSN dataset (section 4b) favoured *no* decorrelation on every
     metric.
   - The `[1, s, r]` constraint was validated on the old build, where it
     also blocked the gene-mean reflection. The C++ now handles the
     reflection itself, so the constraint may only be suppressing the
     donor-confounded depth signal.
   - If the atlas agrees with MSN, flip the default. That needs sign-off,
     because it reverses the validated prototype.
5. **Depth column.** Compare `depth` and `nodecor_depth` with `pkg`. With
   decorrelation on, the depth column is mostly projected away, since
   `s` and `r` are depth proxies. On MSN, the depth column helped only
   without decorrelation: within-label depth R² 0.367 vs 0.781 with it.
6. **Plain-path change.** `plain_vs_capture_plain` should differ (reflection
   removed). Confirm `plain` has lower depth correlation than
   `capture_plain` and that nothing else regressed.
7. **Performance and memory.** `timings`: `auto_groups` (kNN + Leiden; watch
   peak RSS, see opportunity 5), `pkg_design` (dense `n × b` float64), and
   `*_correct` vs `plain_correct`. Also run
   `tests/benchmark_batch_correction.py --mode grouped --strata-key Region`
   on a mid-size dataset if a before/after table is wanted for
   `docs/batch_correction_benchmark.md`.
8. **Reduction determinism.** `reduce_kernel_check` (with
   `--check-reduce-kernel`): the new build's `reduce_kernel` vs the capture.
   It was bitwise identical locally. On another machine the comparison is
   between two builds on the same BLAS, so any gap points at the build.

### 4b. Real-data sanity check: Hm STR MSN (local)

Data: `data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad`, 41,007 cells,
25 UIDs (10 donors × Caudate/Putamen), 2 cell types and 5 subtypes.
Pipeline as in `tests/test_batchcorr.ipynb`:
- `filter_anndata(0.01)`
- `normalize_anndata(1e4, log2)`
- `reduce_kernel(k=30)`

Script and outputs: `build/sanity_msn/` (gitignored). Metrics are over all
cells with kNN k = 30. Within-label depth R² is the fraction of log-depth
variance, within each subtype, that the embedding explains linearly.
Expected same-UID fraction under perfect mixing: 0.047 within subtype, 0.094
within subtype × region.

| embedding | depth R² within subtype | kNN same UID | kNN same region | kNN same subtype |
|---|---|---|---|---|
| uncorrected | 0.825 | 0.850 | 0.923 | 0.920 |
| plain `batch_key` | 0.630 | 0.486 | 0.762 | 0.946 |
| auto groups + Region (default) | 0.796 | 0.637 | 0.891 | 0.939 |
| … projected update | 0.825 | 0.550 | 0.870 | 0.941 |
| … no decorrelation | 0.636 | 0.579 | 0.888 | 0.944 |
| … depth column | 0.781 | 0.603 | 0.885 | 0.935 |
| … depth column, no decorrelation | 0.367 | 0.587 | 0.891 | 0.945 |
| CellType + Region | 0.806 | 0.583 | 0.875 | 0.942 |
| CellType + Region, no decorrelation | 0.621 | 0.523 | 0.872 | 0.945 |
| CellType + Region, depth column, no decorrelation | 0.462 | 0.537 | 0.875 | 0.946 |

Takeaways:

- **Parity.** The packaged design reproduces the prototype exactly (sigma
  3e-14 relative, 5e-6°).
- **Region strata work.** Plain correction erases region (0.762, against
  0.511 under perfect mixing); with strata it stays near uncorrected
  (0.89).
- **Auto groups follow donors here.** Their ARI against CellType is 0.003,
  and 9 of 12 groups were dominated by one UID and merged into 3.
  - A broad-class prior (user `groups`) or a check that flags donor-driven
    crude clusters would help. Lower resolution alone will not, because
    donor effects dominate the uncorrected reduction.
  - Run to run, the crude clustering is not reproducible (11 vs 12 groups),
    because the multi-threaded HNSW search varies.
- **Decorrelation.** Decorrelation on and the depth column fight each
  other. With decorrelation off, the depth column cuts within-subtype depth
  R² from 0.83 to 0.37 with no loss of subtype or region structure.

### Synthetic smoke run of the driver (3,000 cells, 400 genes, 24 donors, 6 clusters, 2 regions, k = 12)

Both stages used the new build; this checks the plumbing, not the regime.
`nb`, `pkg`, `auto` and `plain` matched their capture counterparts to
3e-6°. Same-donor fraction expected under perfect mixing within groups:
0.043.

| embedding | depth R² within groups | max abs corr(depth) | kNN same donor | kNN same region |
|---|---|---|---|---|
| uncorrected | 0.063 | 0.138 | 0.197 | 0.949 |
| `pkg` (full update, = prototype) | 0.034 | 0.084 | 0.238 | 0.941 |
| `proj` (projected update) | 0.063 | 0.135 | 0.094 | 0.978 |
| `nodecor` | 0.095 | 0.178 | 0.227 | 0.945 |
| `depth` | 0.074 | 0.149 | 0.254 | 0.931 |
| `plain` | 0.082 | 0.205 | 0.234 | 0.941 |

Interpretation, to confirm on the atlas: the full update solves
`K_k - K Z Z' = K_k (I - Z Z') - (K - K_k) Z Z'`. The second term re-injects
the batch content of the residual outside the top k. That would explain the
higher same-donor fraction, and possibly also part of the depth reduction.

### 4c. Atlas validation: PFC/VMB subset (Linux x86_64, MKL)

Data: `/data/adatas/adata_agg_PFC_VMB_fil.h5ad`, opened read-only and backed.
- Size: 1,154,246 cells × 27,384 genes, 4.19 × 10⁹ nonzeros.
- Design factors: 129 donors, 198 samples (`UID`), and 2 regions (PFC
  761,793 cells, VMB 392,453).
- Annotation: 59 `full_label` types.

The file holds the prototype's validated result, called "gold" below:
- `obsm["action_corrected"]`, with its ACTIONet graph;
- the uncorrected reduction `obsm["action"]`;
- the crude groups `obs["leiden_unc"]` (21 clusters).

Method:
- **Arms.** The reduction was computed on 1,167,728 cells, and the file was
  subset afterwards. The section-4 driver recomputes `reduce_kernel`, so it
  cannot be compared against the gold, and was not used. Instead, every arm
  corrects the stored `obsm["action"]` through `correct_batch_effect`, using
  the stored lazy transform (target sum 7928, natural log1p).
- **Location.** Scripts and full tables are in `build/atlas_pfc_vmb/`
  (gitignored). `out/summary.md` has every arm in both spaces.
- **Embedding metrics.** These use a k = 30 kNN graph on L1-normalized rows,
  which is what `run_action(prenormalize=True)` sees. Raw-space values agree.
  - *Cross-lineage*: fraction of neighbours in another of 12 lineages
    (`full_label` truncated to Glut, GABA, DA, Glia.*, Imm.*, Vasc.*).
    Imm.Micro ↔ Imm.PVM is not counted. The Vasc.Endo column restricts this
    to endothelial cells.
  - *Same sample*: among a cell's neighbours with the same label and
    region, the fraction from its own `UID`. Perfect mixing gives 0.017.
  - *Same region*: within label, over labels with at least 200 cells per
    region. Perfect mixing gives 0.535.
  - *Depth R²*: the fraction of within-label log-total-count variance that
    one linear map of the embedding explains, with both centred per label.
- **Graph metrics.** These apply the same definitions, edge-weighted, to
  the ACTIONet graph: `run_action(k_max=30)`, then `build_network` with the
  `run_actionet` defaults (k\*nn, JSD).

Embedding kNN:

| arm | change | cross-lineage | Vasc.Endo cross-lineage | same label | same sample | same region | depth R² |
|---|---|---|---|---|---|---|---|
| uncorrected | | 0.0011 | 0.0036 | 0.911 | 0.585 | 0.932 | 0.741 |
| gold | stored prototype result | 0.0006 | 0.0032 | 0.920 | 0.461 | 0.914 | 0.670 |
| `pkg` | `groups="leiden_unc"`, Donor, Region (prototype design) | 0.0006 | 0.0026 | 0.920 | 0.457 | 0.916 | 0.662 |
| `auto` | `groups="auto"` | 0.0007 | 0.0024 | 0.920 | 0.455 | 0.915 | 0.663 |
| `proj` | projected update | 0.0007 | 0.0024 | 0.925 | 0.310 | 0.910 | 0.709 |
| `nodecor` | `decorrelate_from_reduction=False` | 0.0004 | 0.0014 | 0.924 | 0.427 | 0.918 | 0.298 |
| `depth` | depth column | 0.0008 | 0.0026 | 0.920 | 0.460 | 0.917 | 0.686 |
| `nodecor_depth` | depth column, no decorrelation | 0.0004 | 0.0018 | 0.923 | 0.429 | 0.918 | 0.146 |
| `proj_nodecor` | projected, no decorrelation | 0.0004 | 0.0019 | 0.929 | 0.290 | 0.916 | 0.717 |
| `nostrata` | no Region strata | 0.0006 | 0.0028 | 0.920 | 0.458 | 0.883 | 0.675 |
| `pkg_uid` | `UID` as batch | 0.0007 | 0.0030 | 0.911 | 0.462 | 0.917 | 0.665 |
| plain Donor | `batch_key="Donor"` | 0.0021 | 0.0168 | 0.916 | 0.470 | 0.874 | 0.522 |
| plain Donor, pre-change build | | 0.0024 | 0.0311 | 0.918 | 0.438 | 0.863 | 0.596 |
| plain UID | `batch_key="UID"` | 0.0034 | 0.0405 | 0.907 | 0.480 | 0.879 | 0.455 |
| plain UID, pre-change build | | 0.0027 | 0.0303 | 0.906 | 0.446 | 0.868 | 0.510 |

ACTIONet graph:

| graph | archetypes (stacked/merged) | cross-lineage | Vasc.Endo cross-lineage | Endo → Oligo | same label | same sample | same region |
|---|---|---|---|---|---|---|---|
| gold, stored graph (original, larger `k_max`) | 855/32 | 0.0011 | 0.0020 | 0.0002 | 0.892 | 0.137 | 0.860 |
| gold | 398/22 | 0.0011 | 0.0018 | 0.0003 | 0.872 | 0.083 | 0.835 |
| `pkg` | 368/21 | 0.0013 | 0.0021 | 0.0003 | 0.869 | 0.078 | 0.840 |
| `proj` | 367/21 | 0.0012 | 0.0017 | 0.0003 | 0.874 | 0.077 | 0.838 |
| `nodecor` | 377/20 | 0.0010 | 0.0017 | 0.0002 | 0.887 | 0.062 | 0.848 |
| `nodecor_depth` | 355/20 | 0.0009 | 0.0016 | 0.0002 | 0.879 | 0.057 | 0.835 |
| plain Donor | 401/22 | 0.0017 | 0.0147 | 0.0114 | 0.886 | 0.107 | 0.761 |
| plain Donor, pre-change build | 292/17 | 0.0041 | 0.1198 | 0.1037 | 0.879 | 0.068 | 0.729 |

"Endo → Oligo" is the fraction of Vasc.Endo cells' neighbours that are
Glia.Oligo. Compare arms against the gold row built with the same settings,
not the stored graph.

Results, by the checks in section 4:

1. **Regime and parity: pass.** There are 129 donors, so the full Brand
   path applies.
   - On the prototype design, the new build reproduces the pre-change build
     to 2.8e-14 relative in sigma and 2.7e-6°.
   - `pkg` equals the prototype design.
   - Against the stored gold: per-column correlation is 0.987–0.9999, the
     largest top-10 principal angle is 4.6°, kNN Jaccard is 0.65, and every
     metric is within 0.01. The residual gap comes from the 13,482 cells
     removed after the gold was computed.
   - With the same ACTION settings, the `pkg` and gold graphs agree on every
     metric to within 0.006.
   - The stored graph's higher label and region purity comes from its larger
     `k_max`, not from the correction.
2. **Full vs projected update: keep full.**
   - In the embedding, `proj` mixes samples better (0.31 vs 0.46), as on
     synthetic data.
   - It cannot remove any cell-level confound that the uncorrected
     embedding carries. Its cell side stays in `span(U_k)`, so its
     embedding is a linear re-weighting of the uncorrected one. Raw-space
     depth R² is exactly the uncorrected 0.756, with or without
     decorrelation or a depth column.
   - It also collapses about 8 of the 50 components (σ₅₀ is 76, against
     773 for the full update).
   - In the ACTIONet graph the two are indistinguishable: same sample is
     0.077 vs 0.078, and same label 0.874 vs 0.869. The embedding-level
     mixing gap does not survive ACTION, so an `update="projected"` option
     would buy nothing downstream here.
3. **Group estimation: pass.**
   - `auto` reproduces `leiden_unc` (ARI 0.955) and merges the same
     dominated cluster (12 into 1).
   - Every metric is within 0.003 of `pkg`.
   - Cost at 1.15M cells: 206 s and 8.9 GB peak RSS.
   - Unlike MSN, the crude clusters here follow cell type, not donor.
4. **Decorrelation default: evidence agrees with MSN; the default is now
   off.** `nodecor` beats `pkg` on every metric:
   - within-label depth R²: 0.30 vs 0.66;
   - same sample: 0.43 vs 0.46;
   - same label: 0.924 vs 0.920;
   - same region: 0.918 vs 0.916;
   - Vasc.Endo cross-lineage: 0.0014 vs 0.0026.

   The graph agrees, against the gold built with the same settings:
   - same label: 0.887 vs 0.872;
   - same sample: 0.062 vs 0.083;
   - same region: 0.848 vs 0.835;
   - cross-lineage: 0.0010 vs 0.0011.

   The constraint makes every batch column orthogonal over cells to the
   per-cell depth proxies `s` and `r`, so the correction cannot touch
   within-batch depth variation. The gold, built with the constraint, keeps
   the same depth signal as `pkg` (0.67). Flipping the default therefore
   moves away from the validated result: `nodecor` vs gold has kNN Jaccard
   0.34 and top-10 angles up to 41°.
5. **Depth column: only without decorrelation, and not free.**
   - Without decorrelation, it halves depth R² again in the embedding
     (`nodecor_depth`, 0.30 to 0.15), with no other embedding-level change.
   - With decorrelation, it slightly raises it (0.66 to 0.69). This matches
     MSN.
   - In the graph, it gives the best mixing (same sample 0.057 vs 0.062).
     It costs some structure relative to `nodecor`: same label 0.879 vs
     0.887, same region 0.835 vs 0.848.
   - It still matches or beats the gold on every graph metric.
   - Keep it opt-in.
6. **Plain path: the reflection fix works, but the plain path is still
   unsafe on this atlas.** The reported Vasc.Endo/Glia.Oligo collapse has
   two causes, and the new build fixes only one.
   - **Reflection (fixed).** The pre-change build re-adds `s` as the top
     component:
     - σ is 41,387, against 18,468 for the uncorrected top and 7,449
       grouped;
     - it correlates 1.000 with `s` and 0.84 with log depth;
     - it is positive in every cell and carries a median 50% of each
       cell's L1 norm, which ACTION's row normalization then compresses
       biology against.

     The new build removes this axis: σ₁ for the plain Donor correction is
     3,055.

     This is what produced the reported collapse. In the pre-change
     plain-Donor graph, 10.4% of Vasc.Endo neighbours are Glia.Oligo. The
     new build brings that down to 1.1%.
   - **Composition (not fixed).** Raw donor one-hots span each donor's mean
     profile, and so its cell-type composition. Plain correction:
     - removes 23–30% of the grouped embedding's top axis (Micro/PVM/Oligo
       vs Astro/OPC) and 35–56% of its fifth;
     - lets Vasc.Endo cross-lineage neighbours reach 1.7% (Donor) or 4.1%
       (UID), against 0.3% for gold;
     - drops region retention to 0.87–0.88.

     For UID, the new plain path is no better than the old one.
   - In the graph, the new plain-Donor path still has:
     - Endo → Oligo at 1.1%, against 0.03% for any grouped arm;
     - same region at 0.761, against 0.835–0.848;
     - same sample at 0.107, worse than grouped.
   - Every grouped arm keeps Vasc.Endo cross-lineage at or below the
     uncorrected 0.36% in the embedding.
7. **Performance.** Arms ran three to four at a time on 48 cores, so these
   wall times include contention.
   - Correction: 342–670 s per arm (two backed passes over 4.19 × 10⁹
     nonzeros), 12–17 GB peak RSS.
   - Designs: 2–5 s.
   - ACTION with `k_max=30`: 100–127 min per arm.
   - k\*nn network: 56–97 min per arm.
8. **Reduction determinism: not run.** This atlas is a subset, so the
   stored reduction was reused.

Also on this atlas:
- **Strata.** Region strata keep region structure: 0.916, against 0.883
  without strata.
- **Batch key.** Donor beats UID. Within a Region stratum a UID is a donor,
  so `UID` only adds donor × region columns: rank 196 vs 128. Those columns
  cost label purity (0.911 vs 0.920) and gain no mixing.

Code findings, all fixed after the evaluation:
- `correct_batch_effect` did not pass through `decorrelate_from_reduction`.
  It now does.
- The `build_batch_design` docstring for `depth` said decorrelation "already
  removes the main depth confound". On both real datasets, decorrelation
  retains it.
- `persist_updates` called `_ensure_backed_writable` before checking
  `get_auto_persist`. Any tool therefore raised on a `backed="r"` object
  even with auto-persist off. The check now runs only when auto-persist is
  on.

Decisions taken on this evidence:
- `decorrelate_from_reduction` defaults to `False`.
- The full Brand update stays.
- The plain `batch_key` path is unchanged.
- The tests and the validation driver pass `decorrelate_from_reduction=True`
  wherever they check parity with the prototype.

Visual comparison, new default vs prototype configuration:
- **Figures.** In `build/atlas_pfc_vmb/out/viz/`:
  `side_by_side_{Donor,Region,full_label}.png` and `side_by_side_all.png`.
- **Columns.**
  - the stored original layout;
  - `pkg`, the prototype configuration;
  - `nodecor`, the new default.

  Both new columns use the graphs from the table above. Each header shows
  that graph's metrics.
- **Layouts.** `layout_network` with the `run_actionet` defaults (UMAP, 100
  epochs, spread 1, `min_dist` 1, seed 0). Both started from the
  standardized stored layout.
  - UMAP still rearranges clusters, so placement is only roughly comparable
    across columns.
  - Cost: 16 and 20 min, run in parallel. Each process peaked at 69–91 GB
    RSS, about 160 GB together, because the C++ layout keeps its own edge
    lists.
- **Re-running.** `bash build/atlas_pfc_vmb/run_viz.sh`.
  - It can be launched detached and skips finished layouts.
  - Progress goes to `out/viz/STATUS`.
  - It needs matplotlib (the `plotting` extra), which was installed in the
    atlas machine's venv for this.
- **Observations.**
  - Both configurations keep lineages apart, with no Vasc.Endo–Glia.Oligo
    bridging.
  - Both keep the PFC/VMB split within oligodendrocytes, and the VMB-only
    populations, as in the original.
  - Donors look well mixed in both, with no donor-specific islands. At 129
    donors only the graph metric separates them: same sample is 0.078 for
    the prototype and 0.062 for the new default.
  - The new-default layout is more compact, because its graph is denser
    (mean degree 961 vs 759). That is a layout property, not merged cell
    types: its same-label purity is higher, 0.887 vs 0.868.

---

## 5. Out-of-scope opportunities (noted while tracing, not implemented)

1. **Projected rank-k update.** `K_k (I - Z Z')` needs one data pass (`S' D`)
   instead of two, and has no residual artifact term (section 4, check 2).
2. **Halko `gram_schmidt` threshold.** `gram_schmidt` keeps its absolute
   1e-4 threshold and its early exit, which zeroes all later columns.
   - Batch correction no longer calls it, but the Halko SVD path still does.
   - Switch it to `orthonormal_factor` or a relative threshold, after
     checking that the Halko outputs are unchanged.
3. **Binding copies.**
   - `numpy_to_arma_mat` makes a C-order copy and then an Armadillo copy.
   - `scipy_to_arma_sparse` copies the whole matrix.
   - For atlas-sized designs (`n × b`) and in-memory sparse `S`, zero-copy
     views (F-order, or a transposed view) would remove one to two full
     copies.
4. **History growth.** Each correction appends `r` columns to
   `obsm[f"{key}_B"]` (`n × r` float64). At 2M cells and 100 donors that is
   1.6 GB per chained correction. Options: store float32, or make the
   history opt-out when `smooth_kernel` replay is not needed.
5. **Leiden graph construction at scale.** `_leiden_membership` builds the
   igraph graph from COO `.tolist()`. At millions of cells × 30 neighbours,
   that is hundreds of millions of Python objects. Options: build the graph
   without Python lists (check what the pinned python-igraph accepts), or
   estimate groups on a subsample and assign the rest by nearest centroid.
6. **Label-0 bin.** `estimate_batch_groups` gives every cluster smaller than
   `min_size = 3` the label 0, which then acts as a real group. Fold those
   cells into the nearest centroid instead.
7. **Dominance per cluster × stratum.** Merging checks batch dominance per
   cluster, as the prototype did. A cluster × stratum cell dominated by one
   donor is still centred on its own. Checking at that level would protect
   region-specific populations sampled from few donors.
8. **Non-orthonormal stored `U`.** `reduce_kernel` rounds `S_r` to a
   `0.01/sqrt(n)` grid, so `U = S_r / sigma` is orthonormal only to about
   5.8e-4 on the test data. Brand's update assumes orthonormality.
   - Compared with an exact SVD of the same matrix, this caused up to 0.05
     sigma shift and a 5° rotation of near-degenerate trailing components.
     It affects old and new builds alike.
   - Fixes: re-orthonormalize in `svd_from_reduction_`, or store unrounded
     factors.
9. **Matrix-free design.** `build_batch_design` materializes `n × b`
   float64.
   - `S' D` could be formed from labels like the sparse-labels path:
     `S' D = S'G_b − (S'G_g) P'`, where `G_b` and `G_g` are the batch and
     group indicators. The decorrelation would become
     `− (S'C)(C'D)`, which only needs `S'C` (`g × (p+1)`).
   - The design would then never exist in memory. This needs a C++ entry
     point and an operator variant for backed data.
10. **Backed persistence cost.** Every `correct_batch_effect` on a writable
    backed file rewrites the whole file (`persist_updates`). For multi-arm
    experiments, document `set_auto_persist(adata, False)` +
    `checkpoint_backed`, as the validation driver does.
11. **R front-end.** Bump the submodule in actionet-r, regenerate its golden
    outputs, and port `build_batch_design` (tracked in
    `src/libactionet/TODO.md`).

---

## 6. File index

- C++: `src/libactionet/src/decomposition/orthogonalization.cpp`,
  `src/libactionet/include/decomposition/orthogonalization.hpp`,
  `src/libactionet/src/decomposition/svd_main.cpp`,
  `src/libactionet/{include,src}/utils_internal/utils_decomp.{hpp,cpp}`
- Python: `src/actionet/tools/batch_correction.py`,
  `src/actionet/network/{build,cluster}.py`, `src/actionet/{,tools/}__init__.py`,
  `src/actionet/io/persist.py` (read-only fix from section 4c)
- Tests and tools: `tests/test_batch_correction_grouped.py`,
  `tests/_batch_design_reference.py`, `tests/fixtures/`,
  `tests/benchmark_batch_correction.py`,
  `tests/validate_grouped_batch_correction_atlas.py`,
  `tests/backed/test_infrastructure.py`
- Atlas evaluation (gitignored, atlas machine only): `build/atlas_pfc_vmb/`.
  - Scripts: arms, metrics and graphs (`run_arms.py`, `metrics.py`,
    `eta2.py`, `graph_arm.py`), the summary (`summarize.py`), and the
    layouts and plots (`viz_layouts.py`, `run_viz.sh`).
  - Outputs: `out/summary.md` and `out/viz/`.
- Docs: `docs/api/batch_correction.md`, `README.md`, `context/DECISIONS.md`,
  `src/libactionet/docs/batch_orthogonalization_algorithm.md`,
  `src/libactionet/TODO.md`
