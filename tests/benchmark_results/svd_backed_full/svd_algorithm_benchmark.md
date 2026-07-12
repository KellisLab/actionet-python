# Backed SVD Algorithm Benchmark: Halko vs IRLB vs Feng

Generated: 2026-07-12 02:44:16

## Configuration

- n_components: 30
- chunk_size: 4096
- trials per config: 2
- reconstruction probe rows: 500

## Summary (mean across trials)

| dataset   |   n_obs | algorithm   |    wall_s |   peak_rss_mb |   io_read_mb |   sigma_corr |   reconstruction_err |
|:----------|--------:|:------------|----------:|--------------:|-------------:|-------------:|---------------------:|
| 25k       |   25000 | feng        |   4.47305 |       6.22592 |            0 |     1        |             0.653377 |
| 25k       |   25000 | halko       |   4.41073 |      10.199   |            0 |   nan        |             0.65339  |
| 25k       |   25000 | irlb        |  31.3704  |       5.65248 |            0 |     0.999998 |             0.653286 |
| 50k       |   50000 | feng        |   8.99256 |      33.6773  |            0 |     1        |             0.659343 |
| 50k       |   50000 | halko       |   8.58623 |      24.6252  |            0 |   nan        |             0.659345 |
| 50k       |   50000 | irlb        |  63.0184  |      23.1588  |            0 |     1        |             0.659315 |
| 100k      |  100000 | feng        |  18.0143  |      29.0324  |            0 |     1        |             0.651954 |
| 100k      |  100000 | halko       |  17.2331  |      32.7434  |            0 |   nan        |             0.651984 |
| 100k      |  100000 | irlb        | 125.739   |      14.7784  |            0 |     0.999999 |             0.651929 |
| 150k      |  150000 | feng        |  27.1974  |      35.3239  |            0 |     1        |             0.651259 |
| 150k      |  150000 | halko       |  25.7886  |      25.6819  |            0 |   nan        |             0.65133  |
| 150k      |  150000 | irlb        | 187.756   |      26.1161  |            0 |     0.999999 |             0.651237 |
| 200k      |  200000 | feng        |  36.774   |      19.6772  |            0 |     1        |             0.660437 |
| 200k      |  200000 | halko       |  42.1798  |      14.2049  |            0 |   nan        |             0.660444 |
| 200k      |  200000 | irlb        | 272.756   |      14.3032  |            0 |     0.999999 |             0.660425 |

## Speed Comparison (wall seconds)

| dataset   |     feng |    halko |     irlb |   feng_vs_halko |   irlb_vs_halko |
|:----------|---------:|---------:|---------:|----------------:|----------------:|
| 100k      | 18.0143  | 17.2331  | 125.739  |        1.04533  |         7.29637 |
| 150k      | 27.1974  | 25.7886  | 187.756  |        1.05463  |         7.28058 |
| 200k      | 36.774   | 42.1798  | 272.756  |        0.871839 |         6.46651 |
| 25k       |  4.47305 |  4.41073 |  31.3704 |        1.01413  |         7.11229 |
| 50k       |  8.99256 |  8.58623 |  63.0184 |        1.04732  |         7.33947 |

## Memory Comparison (peak RSS MB)

| dataset   |     feng |   halko |     irlb |   feng_vs_halko |   irlb_vs_halko |
|:----------|---------:|--------:|---------:|----------------:|----------------:|
| 100k      | 29.0324  | 32.7434 | 14.7784  |        0.886665 |        0.451339 |
| 150k      | 35.3239  | 25.6819 | 26.1161  |        1.37544  |        1.01691  |
| 200k      | 19.6772  | 14.2049 | 14.3032  |        1.38524  |        1.00692  |
| 25k       |  6.22592 | 10.199  |  5.65248 |        0.610442 |        0.554217 |
| 50k       | 33.6773  | 24.6252 | 23.1588  |        1.3676   |        0.940452 |

## Accuracy (vs Halko reference)

> `sigma_corr`: Pearson correlation of singular values between the algorithm and Halko.
> `reconstruction_err`: relative Frobenius error ||A_probe - U D V'||_F / ||A_probe||_F
> estimated on a random 500-row probe of the normalised matrix.

| dataset   |   n_obs | algorithm   |   sigma_corr |   reconstruction_err |
|:----------|--------:|:------------|-------------:|---------------------:|
| 25k       |   25000 | feng        |     1        |             0.653377 |
| 25k       |   25000 | irlb        |     0.999998 |             0.653286 |
| 50k       |   50000 | feng        |     1        |             0.659343 |
| 50k       |   50000 | irlb        |     1        |             0.659315 |
| 100k      |  100000 | feng        |     1        |             0.651954 |
| 100k      |  100000 | irlb        |     0.999999 |             0.651929 |
| 150k      |  150000 | feng        |     1        |             0.651259 |
| 150k      |  150000 | irlb        |     0.999999 |             0.651237 |
| 200k      |  200000 | feng        |     1        |             0.660437 |
| 200k      |  200000 | irlb        |     0.999999 |             0.660425 |

## Recommendation

**Keep Halko as default.** Halko has the lowest median wall time (17.23s) among accuracy-qualifying candidates.

### Median wall time by algorithm

| algorithm   |   median_wall_s |   median_sigma_corr_vs_halko |
|:------------|----------------:|-----------------------------:|
| halko       |         17.2331 |                   nan        |
| feng        |         18.0143 |                     1        |
| irlb        |        125.739  |                     0.999999 |

