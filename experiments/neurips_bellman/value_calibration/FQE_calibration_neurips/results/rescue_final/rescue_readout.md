# Rescue Promotion Audit

Results directory: `<project-root>/FQE_calibration_neurips/results/rescue_final`

## Gate

Rows are promotable only with strict cross-calibration or recent-heldout temporal calibration, positive raw value/oracle correlation, raw miscalibration, true-V MSE improvement, Bellman calibration-error improvement, value/calibration win rates >= 0.60, zero failures, finite diagnostics, and no test/oracle/no-split leakage.

## Label Counts

- `limitation`: 9
- `mechanism_only`: 4
- `promote_main`: 12
- `validation_control`: 3

## Promoted Main Rows

| run_mode | suite_name | learner_variant | calibrator | misspecification_setting | relative_true_v_mse_vs_uncalibrated_all_data | relative_calibration_error_plugin_vs_uncalibrated_all_data | relative_true_v_mse_vs_current_retrain_small | relative_calibration_error_plugin_vs_current_retrain_small | true_v_mse_win_rate_vs_uncalibrated_all_data | calibration_error_plugin_win_rate_vs_uncalibrated_all_data | raw_value_oracle_spearman | raw_value_calibration_slope | rescue_audit_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | model_misspecification_sweep | linear_fqe_affine_ridge_light | isotonic | affine | 0.7671 | 0.834 |  |  | 0.91 | 0.72 | 0.6377 | 1.144 | promote_main |
| final | model_misspecification_sweep | linear_fqe_affine_ridge_light | linear | affine | 0.7849 | 0.661 |  |  | 0.91 | 0.9 | 0.6377 | 1.144 | promote_main |
| final | model_misspecification_sweep | linear_fqe_misspecified | isotonic | affine | 0.7336 | 0.7645 |  |  | 0.94 | 0.81 | 0.6378 | 1.144 | promote_main |
| final | model_misspecification_sweep | linear_fqe_misspecified | linear | affine | 0.7497 | 0.6311 |  |  | 0.94 | 0.91 | 0.6378 | 1.144 | promote_main |
| final | temporal_reward_shift_sweep | temporal_rf_fqe | isotonic | temporal_reward_shift | 0.6266 | 0.9161 | 0.3299 | 0.00372 | 0.98 | 0.68 | 0.4295 | 2.015 | promote_main |
| final | temporal_reward_shift_sweep | temporal_rf_fqe | linear | temporal_reward_shift | 0.6301 | 0.7171 | 0.3317 | 0.002912 | 0.97 | 0.81 | 0.4295 | 2.015 | promote_main |
| final | undertraining_sweep | random_feature_fqe_iter2 | isotonic | finite_iteration_scale | 0.7775 | 0.695 |  |  | 0.95 | 0.9 | 0.85 | 5.98 | promote_main |
| final | undertraining_sweep | random_feature_fqe_iter2 | linear | finite_iteration_scale | 0.781 | 0.6642 |  |  | 0.95 | 0.92 | 0.85 | 5.98 | promote_main |
| final | undertraining_sweep | random_feature_fqe_iter3 | isotonic | finite_iteration_scale | 0.7767 | 0.7154 |  |  | 0.95 | 0.9 | 0.8545 | 4.672 | promote_main |
| final | undertraining_sweep | random_feature_fqe_iter3 | linear | finite_iteration_scale | 0.7803 | 0.6843 |  |  | 0.94 | 0.92 | 0.8545 | 4.672 | promote_main |
| final | undertraining_sweep | random_feature_fqe_iter4 | isotonic | finite_iteration_scale | 0.7759 | 0.7363 |  |  | 0.94 | 0.9 | 0.8587 | 3.89 | promote_main |
| final | undertraining_sweep | random_feature_fqe_iter4 | linear | finite_iteration_scale | 0.7795 | 0.7032 |  |  | 0.94 | 0.92 | 0.8587 | 3.89 | promote_main |

## Appendix Support Rows

_None._

## Do-Not-Claim Rows

| run_mode | suite_name | learner_variant | calibrator | misspecification_setting | relative_true_v_mse_vs_uncalibrated_all_data | relative_calibration_error_plugin_vs_uncalibrated_all_data | relative_true_v_mse_vs_current_retrain_small | relative_calibration_error_plugin_vs_current_retrain_small | true_v_mse_win_rate_vs_uncalibrated_all_data | calibration_error_plugin_win_rate_vs_uncalibrated_all_data | raw_value_oracle_spearman | raw_value_calibration_slope | rescue_audit_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | mechanism_distortion_sweep | random_feature_fqe_affine_distorted | none | well_specified_linear | 1 | 1 |  |  | 0 | 0 | 0.8392 | 2.223 | limitation |
| final | mechanism_distortion_sweep | random_feature_fqe_affine_distorted | isotonic | well_specified_linear | 0.4276 | 0.1812 |  |  | 1 | 1 | 0.8392 | 2.222 | mechanism_only |
| final | mechanism_distortion_sweep | random_feature_fqe_affine_distorted | linear | well_specified_linear | 0.4301 | 0.1689 |  |  | 1 | 1 | 0.8392 | 2.222 | mechanism_only |
| final | mechanism_distortion_sweep | random_feature_fqe_monotone_saturation_distorted | none | well_specified_linear | 1 | 1 |  |  | 0 | 0 | 0.8344 | 26.17 | limitation |
| final | mechanism_distortion_sweep | random_feature_fqe_monotone_saturation_distorted | isotonic | well_specified_linear | 0.3749 | 0.02211 |  |  | 1 | 1 | 0.8343 | 25.61 | mechanism_only |
| final | mechanism_distortion_sweep | random_feature_fqe_monotone_saturation_distorted | linear | well_specified_linear | 0.4404 | 0.6373 |  |  | 0.85 | 1 | 0.8343 | 25.61 | mechanism_only |
| final | model_misspecification_sweep | linear_fqe_affine_ridge_light | none | affine | 1 | 1 |  |  | 0 | 0 | 0.6383 | 1.145 | limitation |
| final | model_misspecification_sweep | linear_fqe_misspecified | none | affine | 1 | 1 |  |  | 0 | 0 | 0.6383 | 1.145 | limitation |
| final | temporal_reward_shift_sweep | temporal_rf_fqe | none | temporal_reward_shift | 1.9 | 246.2 | 1 | 1 | 0.08 | 0 | -0.001313 | -0.0004888 | limitation |
| final | temporal_reward_shift_sweep | temporal_rf_fqe | none | temporal_reward_shift | 1 | 1 | 0.5264 | 0.004061 | 0 | 0 | 0.4295 | 2.015 | limitation |
| final | undertraining_sweep | random_feature_fqe_iter2 | none | finite_iteration_scale | 1 | 1 |  |  | 0 | 0 | 0.8501 | 5.978 | limitation |
| final | undertraining_sweep | random_feature_fqe_iter3 | none | finite_iteration_scale | 1 | 1 |  |  | 0 | 0 | 0.8545 | 4.671 | limitation |
| final | undertraining_sweep | random_feature_fqe_iter4 | none | finite_iteration_scale | 1 | 1 |  |  | 0 | 0 | 0.8587 | 3.889 | limitation |
| final | well_specified_debug | linear_fqe | none | well_specified_linear | 1 | 1 |  |  | 0 | 0 | 0.8765 | 1.002 | validation_control |
| final | well_specified_debug | linear_fqe | isotonic | well_specified_linear | 1.077 | 3.672 |  |  | 0.2 | 0 | 0.8762 | 1.001 | validation_control |
| final | well_specified_debug | linear_fqe | linear | well_specified_linear | 0.9698 | 1.005 |  |  | 0.4 | 0.4 | 0.8762 | 1.001 | validation_control |
