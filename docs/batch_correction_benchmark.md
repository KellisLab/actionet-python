# Batch correction benchmark

Empirical comparison of `correct_batch_effect()` before and after the
optimization pass tracked in `tests/benchmark_batch_correction.py`.

## Setup

- Dataset: `data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad`
- After preprocessing (`filter_anndata` `min_cells_per_feat=0.01`,
  `normalize_total target_sum=1e4`, `log1p base=2`):
  - `n_obs = 41,007`, `n_vars = 20,109`, `nnz = 226,473,393`.
- Reduction: `reduce_kernel(k=30)`, IRLB sparse path.
- Sweep: `b ∈ {2, 5, 10, 25, 50, 100}` synthetic batch labels stratified
  over `obs['UID']`.
- 3 trials per `b`. Median wall and median peak-RSS-delta reported.
- Hardware: macOS arm64 (M-series), portable build (no `--native-macos`).

## Results

| n_batches | baseline wall (s) | optimized wall (s) | speedup | baseline ΔRSS (MB) | optimized ΔRSS (MB) |
|-----------|-------------------|--------------------|---------|--------------------|---------------------|
|         2 |             2.206 |              2.294 |   0.96× |                1.3 |                 7.1 |
|         5 |             2.543 |              2.477 |   1.03× |                2.5 |                 7.9 |
|        10 |             3.543 |              2.961 |   1.20× |                0.4 |                 3.0 |
|        25 |             6.566 |              4.701 |   1.40× |                0.4 |                 0.1 |
|        50 |            13.049 |              7.562 |   1.73× |                0.2 |                 0.0 |
|       100 |            36.349 |             19.184 |   1.89× |             5435.4 |                 0.1 |

Maximum absolute difference between baseline and optimized singular values
across all `b` and trials: **6.4 × 10⁻¹¹** (well below the 10⁻⁶ tolerance).

## Interpretation

- Wall time scaling improved from ~16.5× (`b=2 → b=100`) to ~8.4× — closer
  to the algorithmic floor set by the `(k+b+1)³` core SVD plus the
  `(g+n)·k·(k+b+1)` factor expansion.
- The 5.4 GB transient memory peak at `b=100` was caused by
  `arma::join_horiz([U|P]) * U_p` materialising a `(g+n) × (k+b+1)`
  temporary. Splitting that into two narrow GEMMs eliminates it.
- For small `b` (≤5) the new code is comparable to baseline; the algorithmic
  wins are concentrated where `b` was previously hurting the most.

## Reproducing

```bash
# Baseline (run on git rev preceding the optimization pass):
python tests/benchmark_batch_correction.py \
    --output tests/benchmark_results/batch_correction/baseline.json \
    --tag baseline

# Optimized (run on the post-optimization revision):
python tests/benchmark_batch_correction.py \
    --output tests/benchmark_results/batch_correction/optimized.json \
    --tag optimized \
    --baseline-json tests/benchmark_results/batch_correction/baseline.json
```

The benchmark script accepts `--batch-counts`, `--trials`, `--seed`, and
`--natural-batch-key` for configuring the sweep.

## Underlying changes

The benchmark exercises four independent optimizations:

1. **Truncated final GEMMs** in `actionet::perturbedSVD`
   (`src/libactionet/src/decomposition/svd_main.cpp`). The expansion
   `U_new = [U|P] · U_p[:, :k]` is split into
   `U · U_p_top + P · U_p_bot`, avoiding the
   `(g × (k+b+1))` `join_horiz` allocation. Same for `V_new`.
2. **Householder QR** (`arma::qr_econ`) replaces classical Gram-Schmidt
   for orthonormalising the residual subspaces (`P, Q`) and the batch
   subspace (`Z`).
3. **Sparse one-hot fast path**
   (`orthogonalizeBatchEffect_sparse_labels`): when the design is one-hot
   (typical for `batch_key` use), `Z = S' · D` is built by a single
   sparse-iterator pass over `S`, costing `O(nnz + g·b)` instead of
   `O(nnz·b)` for the generic sparse-dense product. The Python wrapper
   dispatches to this path automatically when `adata.X` is sparse.
4. **Block-write of the core matrix `K`** in `perturbedSVD` removes the
   `join_vert` / `arma::trans(arma::join_vert(...))` allocations.

Validation: `tests/test_batch_correction_parity.py` asserts that singular
values match within `1e-8`, and that the singular subspaces match within a
sign-fixed Frobenius distance of `1e-6` against both the dense-design and
dense-X paths, across `b ∈ {2, 5, 10, 25}`.
