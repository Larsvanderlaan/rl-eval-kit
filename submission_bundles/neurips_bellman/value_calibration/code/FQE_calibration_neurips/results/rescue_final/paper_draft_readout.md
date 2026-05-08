# Paper Draft Readout

- Validation gate passed: `True`.
- Results directory: `<project-root>/FQE_calibration_neurips/results/rescue_final`.
- Figures directory: `<project-root>/FQE_calibration_neurips/figures/rescue_final`.

## Mechanism Checks

- Replications: `100` (`pass`).
- Affine mechanism: matched variant `random_feature_fqe_affine_distorted`, linear relative true-V MSE `0.43`, best nonlinear `isotonic` `0.428`: `pass`.
- Monotone mechanism: matched variant `random_feature_fqe_monotone_saturation_distorted`, best isotonic/hybrid `isotonic` relative true-V MSE `0.375`, linear `0.44`: `pass`.

## Split-Stability Checks

- Split-stability diagnostics: missing.

## Failure Diagnostics

- Method groups with nonzero failure rate: `0`.

## Candidate Main Figures

- `mse_vs_sample_size` and `relative_mse_vs_sample_size` if quick-paper replications are present.
- `coverage_stratified_error` as the failure/limited-overlap diagnostic.
- `calibrator_comparison` / mechanism table: mechanism checks passed.
- `split_stability_diagnostics` only if stable split groups are listed above.
