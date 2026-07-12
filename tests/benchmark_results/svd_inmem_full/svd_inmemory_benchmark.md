# In-Memory SVD Algorithm Benchmark: IRLB vs Halko vs Feng

Generated: 2026-07-12 03:35:46

## Configuration

- n_components: 30
- trials per config: 2
- reconstruction probe rows: 500
- reference algorithm for sigma_corr: IRLB

## Summary (mean across trials)

| storage_form   | dataset   |   n_obs | algorithm   |    wall_s |   peak_rss_mb |   sigma_corr |   reconstruction_err |
|:---------------|:----------|--------:|:------------|----------:|--------------:|-------------:|---------------------:|
| dense          | 25k       |   25000 | feng        |   1.29643 |   2988.12     |     0.999999 |             0.653377 |
| dense          | 25k       |   25000 | halko       |   1.1192  |   2973.89     |     0.999998 |             0.65339  |
| dense          | 25k       |   25000 | irlb        |   2.66972 |   2978.68     |   nan        |             0.653286 |
| dense          | 50k       |   50000 | feng        |   2.58176 |   5933.97     |     1        |             0.659343 |
| dense          | 50k       |   50000 | halko       |   2.34657 |   5921.07     |     1        |             0.659345 |
| dense          | 50k       |   50000 | irlb        |   5.49205 |   5917.1      |   nan        |             0.659315 |
| dense          | 100k      |  100000 | feng        |   9.60082 |      0.04096  |     1        |             0.651954 |
| dense          | 100k      |  100000 | halko       |   9.11276 |     82.4361   |     0.999999 |             0.651984 |
| dense          | 100k      |  100000 | irlb        |  15.1666  |    113.5      |   nan        |             0.651929 |
| sparse         | 25k       |   25000 | feng        |  16.3381  |   2095.88     |     0.999999 |             0.653377 |
| sparse         | 25k       |   25000 | halko       |  14.2492  |   2179.01     |     0.999998 |             0.65339  |
| sparse         | 25k       |   25000 | irlb        |   4.32961 |   2087.39     |   nan        |             0.653286 |
| sparse         | 50k       |   50000 | feng        |  34.5365  |   4128.92     |     1        |             0.659343 |
| sparse         | 50k       |   50000 | halko       |  28.2597  |   4131.87     |     1        |             0.659345 |
| sparse         | 50k       |   50000 | irlb        |   8.71163 |   4158.96     |   nan        |             0.659315 |
| sparse         | 100k      |  100000 | feng        |  98.608   |   8244.28     |     1        |             0.651954 |
| sparse         | 100k      |  100000 | halko       |  75.3714  |   8238.6      |     0.999999 |             0.651984 |
| sparse         | 100k      |  100000 | irlb        |  17.5686  |   8268.58     |   nan        |             0.651929 |
| sparse         | 150k      |  150000 | feng        | 188.169   |   3945.86     |     1        |             0.651259 |
| sparse         | 150k      |  150000 | halko       | 148.853   |    849.789    |     0.999999 |             0.65133  |
| sparse         | 150k      |  150000 | irlb        |  28.1537  |   3235.26     |   nan        |             0.651237 |
| sparse         | 200k      |  200000 | feng        | 264.582   |    839.942    |     0.999999 |             0.660437 |
| sparse         | 200k      |  200000 | halko       | 216.19    |    198.541    |     0.999999 |             0.660444 |
| sparse         | 200k      |  200000 | irlb        |  43.5161  |      0.270336 |   nan        |             0.660425 |

## Sparse in-memory

### Wall time (s), ratios vs IRLB

| dataset   |     feng |    halko |     irlb |   feng_vs_irlb |   halko_vs_irlb |
|:----------|---------:|---------:|---------:|---------------:|----------------:|
| 100k      |  98.608  |  75.3714 | 17.5686  |        5.61275 |         4.29013 |
| 150k      | 188.169  | 148.853  | 28.1537  |        6.68362 |         5.28715 |
| 200k      | 264.582  | 216.19   | 43.5161  |        6.0801  |         4.96805 |
| 25k       |  16.3381 |  14.2492 |  4.32961 |        3.77358 |         3.29111 |
| 50k       |  34.5365 |  28.2597 |  8.71163 |        3.96442 |         3.24391 |

### Peak RSS (MB), ratios vs IRLB

| dataset   |     feng |    halko |        irlb |   feng_vs_irlb |   halko_vs_irlb |
|:----------|---------:|---------:|------------:|---------------:|----------------:|
| 100k      | 8244.28  | 8238.6   | 8268.58     |       0.997061 |        0.996374 |
| 150k      | 3945.86  |  849.789 | 3235.26     |       1.21964  |        0.262665 |
| 200k      |  839.942 |  198.541 |    0.270336 |    3107.03     |      734.424    |
| 25k       | 2095.88  | 2179.01  | 2087.39     |       1.00407  |        1.0439   |
| 50k       | 4128.92  | 4131.87  | 4158.96     |       0.992777 |        0.993486 |

### Accuracy (vs IRLB reference)

> `sigma_corr`: Pearson correlation of singular values vs IRLB.
> `reconstruction_err`: relative Frobenius error on a random 500-row probe.

| dataset   |   n_obs | algorithm   |   sigma_corr |   reconstruction_err |
|:----------|--------:|:------------|-------------:|---------------------:|
| 25k       |   25000 | feng        |     0.999999 |             0.653377 |
| 25k       |   25000 | halko       |     0.999998 |             0.65339  |
| 50k       |   50000 | feng        |     1        |             0.659343 |
| 50k       |   50000 | halko       |     1        |             0.659345 |
| 100k      |  100000 | feng        |     1        |             0.651954 |
| 100k      |  100000 | halko       |     0.999999 |             0.651984 |
| 150k      |  150000 | feng        |     1        |             0.651259 |
| 150k      |  150000 | halko       |     0.999999 |             0.65133  |
| 200k      |  200000 | feng        |     0.999999 |             0.660437 |
| 200k      |  200000 | halko       |     0.999999 |             0.660444 |

## Dense in-memory

### Wall time (s), ratios vs IRLB

| dataset   |    feng |   halko |     irlb |   feng_vs_irlb |   halko_vs_irlb |
|:----------|--------:|--------:|---------:|---------------:|----------------:|
| 100k      | 9.60082 | 9.11276 | 15.1666  |       0.633022 |        0.600842 |
| 25k       | 1.29643 | 1.1192  |  2.66972 |       0.485603 |        0.419219 |
| 50k       | 2.58176 | 2.34657 |  5.49205 |       0.47009  |        0.427268 |

### Peak RSS (MB), ratios vs IRLB

| dataset   |       feng |     halko |    irlb |   feng_vs_irlb |   halko_vs_irlb |
|:----------|-----------:|----------:|--------:|---------------:|----------------:|
| 100k      |    0.04096 |   82.4361 |  113.5  |    0.000360881 |        0.726308 |
| 25k       | 2988.12    | 2973.89   | 2978.68 |    1.00317     |        0.998394 |
| 50k       | 5933.97    | 5921.07   | 5917.1  |    1.00285     |        1.00067  |

### Accuracy (vs IRLB reference)

> `sigma_corr`: Pearson correlation of singular values vs IRLB.
> `reconstruction_err`: relative Frobenius error on a random 500-row probe.

| dataset   |   n_obs | algorithm   |   sigma_corr |   reconstruction_err |
|:----------|--------:|:------------|-------------:|---------------------:|
| 25k       |   25000 | feng        |     0.999999 |             0.653377 |
| 25k       |   25000 | halko       |     0.999998 |             0.65339  |
| 50k       |   50000 | feng        |     1        |             0.659343 |
| 50k       |   50000 | halko       |     1        |             0.659345 |
| 100k      |  100000 | feng        |     1        |             0.651954 |
| 100k      |  100000 | halko       |     0.999999 |             0.651984 |

## Recommendations

**Sparse: keep current default (IRLB).** Median wall=17.569s beats all accuracy-qualified alternatives.

**Dense: keep current default (HALKO).** Median wall=2.347s beats all accuracy-qualified alternatives.

