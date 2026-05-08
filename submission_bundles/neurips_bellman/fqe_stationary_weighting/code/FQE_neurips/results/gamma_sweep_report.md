# Gamma-Sweep Weighting Report

- source stage: `gamma_paper`
- `value_gamma` is the FQE Bellman/value discount.
- `ratio_gamma` is the weighting target; `ratio_gamma=1.0` is stationary weighting, not undiscounted FQE.
- The clean mechanism benchmark samples from the behavior reference distribution matched to `ratio_gamma`.

## Main Linear-FQE Slice

| ratio_gamma | shift | estimator | target-ref Q MSE | value MSE | ESS frac | q99 | max w | log-ratio RMSE | calibration | unstable |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.950 | 0.000 | linear_estimated_clipped_fqe | 3.209 | 0.107 | 0.995 | 1.183 | 1.183 | 0.078 | 5.119e-03 | stable |
| 0.950 | 0.000 | linear_neural_weighted_clipped_fqe | 2.935 | 0.262 | 0.997 | 1.126 | 1.126 | 0.050 | 0.013 | stable |
| 0.950 | 0.000 | linear_oracle_clipped_fqe | 3.271 | 0.166 | 1.000 | 1.000 | 1.000 | 0.000 | 7.079e-03 | stable |
| 0.950 | 0.000 | linear_standard_fqe | 3.271 | 0.166 | 1.000 | 1.000 | 1.000 | 0.000 | 7.079e-03 | stable |
| 0.950 | 1.100 | linear_estimated_clipped_fqe | 16.172 | 66.414 | 0.400 | 4.962 | 4.962 | 13.445 | 0.185 | stable |
| 0.950 | 1.100 | linear_neural_weighted_clipped_fqe | 22.730 | 115.257 | 0.400 | 5.525 | 5.525 | 14.134 | 0.159 | stable |
| 0.950 | 1.100 | linear_oracle_clipped_fqe | 71.000 | 7.688 | 0.400 | 7.754 | 7.755 | 14.545 | 0.304 | stable |
| 0.950 | 1.100 | linear_standard_fqe | 504.843 | 131.753 | 1.000 | 1.000 | 1.000 | 15.033 | 0.561 | stable |
| 0.950 | 1.350 | linear_estimated_clipped_fqe | 1.832e+03 | 2.463e+03 | 0.400 | 7.303 | 7.303 | 16.995 | 1.127 | stable |
| 0.950 | 1.350 | linear_neural_weighted_clipped_fqe | 1.354e+03 | 884.352 | 0.400 | 7.001 | 7.001 | 16.817 | 0.864 | stable |
| 0.950 | 1.350 | linear_oracle_clipped_fqe | 1.372e+03 | 1.043e+03 | 0.142 | 21.753 | 21.758 | 17.000 | 1.123 | overlap_collapse |
| 0.950 | 1.350 | linear_standard_fqe | 1.339e+04 | 9.871e+03 | 1.000 | 1.000 | 1.000 | 17.427 | 1.755 | stable |
| 0.990 | 0.000 | linear_estimated_clipped_fqe | 0.866 | 1.773 | 0.993 | 1.183 | 1.183 | 0.100 | 5.466e-03 | stable |
| 0.990 | 0.000 | linear_neural_weighted_clipped_fqe | 0.813 | 1.659 | 0.997 | 1.141 | 1.141 | 0.059 | 7.424e-03 | stable |
| 0.990 | 0.000 | linear_oracle_clipped_fqe | 0.889 | 2.147 | 1.000 | 1.000 | 1.000 | 0.000 | 4.814e-03 | stable |
| 0.990 | 0.000 | linear_standard_fqe | 0.889 | 2.147 | 1.000 | 1.000 | 1.000 | 0.000 | 4.814e-03 | stable |
| 0.990 | 1.100 | linear_estimated_clipped_fqe | 7.379 | 141.321 | 0.625 | 2.359 | 2.359 | 6.013 | 0.041 | stable |
| 0.990 | 1.100 | linear_neural_weighted_clipped_fqe | 2.247 | 115.080 | 0.649 | 3.039 | 3.039 | 7.662 | 0.050 | stable |
| 0.990 | 1.100 | linear_oracle_clipped_fqe | 2.034 | 8.887 | 0.400 | 6.809 | 6.809 | 8.916 | 0.035 | stable |
| 0.990 | 1.100 | linear_standard_fqe | 15.432 | 176.656 | 1.000 | 1.000 | 1.000 | 9.818 | 0.175 | stable |
| 0.990 | 1.350 | linear_estimated_clipped_fqe | 12.236 | 213.897 | 0.532 | 2.675 | 2.675 | 10.743 | 0.140 | stable |
| 0.990 | 1.350 | linear_neural_weighted_clipped_fqe | 11.004 | 40.633 | 0.539 | 2.617 | 2.617 | 11.690 | 0.043 | stable |
| 0.990 | 1.350 | linear_oracle_clipped_fqe | 61.821 | 13.585 | 0.400 | 7.498 | 7.499 | 14.272 | 0.329 | stable |
| 0.990 | 1.350 | linear_standard_fqe | 283.619 | 28.528 | 1.000 | 1.000 | 1.000 | 14.780 | 0.599 | stable |
| 1.000 | 0.000 | linear_estimated_clipped_fqe | 6.803e-03 | 131.330 | 0.995 | 1.173 | 1.173 | 0.072 | 3.860e-03 | stable |
| 1.000 | 0.000 | linear_neural_weighted_clipped_fqe | 5.001e-03 | 130.599 | 0.995 | 1.224 | 1.224 | 0.065 | 7.115e-03 | stable |
| 1.000 | 0.000 | linear_oracle_clipped_fqe | 6.559e-03 | 131.281 | 1.000 | 1.000 | 1.000 | 0.000 | 3.674e-03 | stable |
| 1.000 | 0.000 | linear_standard_fqe | 6.559e-03 | 131.281 | 1.000 | 1.000 | 1.000 | 0.000 | 3.674e-03 | stable |
| 1.000 | 1.100 | linear_estimated_clipped_fqe | 0.048 | 126.408 | 0.640 | 2.597 | 2.598 | 5.366 | 0.018 | stable |
| 1.000 | 1.100 | linear_neural_weighted_clipped_fqe | 0.046 | 124.287 | 0.540 | 4.104 | 4.105 | 3.909 | 0.026 | stable |
| 1.000 | 1.100 | linear_oracle_clipped_fqe | 1.680e-03 | 129.946 | 0.400 | 6.690 | 6.691 | 4.986 | 5.377e-03 | stable |
| 1.000 | 1.100 | linear_standard_fqe | 0.594 | 112.674 | 1.000 | 1.000 | 1.000 | 6.202 | 0.068 | stable |
| 1.000 | 1.350 | linear_estimated_clipped_fqe | 1.709 | 102.204 | 0.557 | 2.884 | 2.884 | 7.140 | 0.046 | stable |
| 1.000 | 1.350 | linear_neural_weighted_clipped_fqe | 0.650 | 112.628 | 0.534 | 3.532 | 3.532 | 9.830 | 0.050 | stable |
| 1.000 | 1.350 | linear_oracle_clipped_fqe | 2.435 | 96.456 | 0.400 | 7.215 | 7.216 | 11.311 | 0.054 | stable |
| 1.000 | 1.350 | linear_standard_fqe | 14.918 | 57.173 | 1.000 | 1.000 | 1.000 | 11.974 | 0.135 | stable |

## Interpretation Guardrails

- If stationary weighting improves less than discounted weighting, report that as a regularization tradeoff rather than a failure of the method.
- If neural weights have better calibration but worse value error, diagnose WLS/FQE instability and ESS rather than claiming ratio accuracy alone is sufficient.
- If `ratio_gamma=1.0` has very low ESS or high q99/max weights, frame it as stationary-overlap stress.