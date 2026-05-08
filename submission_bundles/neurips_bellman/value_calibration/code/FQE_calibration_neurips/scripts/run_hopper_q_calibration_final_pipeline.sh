#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

SMOKE_DIR="FQE_calibration_neurips/results/hopper_q_calibration_preflight_smoke"
BASE_DIR="FQE_calibration_neurips/results/hopper_q_calibration_tune_base"
CAL_DIR="FQE_calibration_neurips/results/hopper_q_calibration_tune_calibrator"
FINAL_DIR="FQE_calibration_neurips/results/hopper_q_calibration_final"

date

"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"

"${PY}" "${BENCH}" \
  --stage smoke \
  --critic_families linear_fqe rf_fqe neural_fqe \
  --seeds 0 \
  --policy_indices 0 \
  --max_trajectories 32 \
  --fqe_updates 10 \
  --rf_components 16 \
  --action_samples 1 \
  --fit_action_samples 1 \
  --initial_action_samples 2 \
  --n_bins 5 \
  --min_bin_size 20 \
  --metric_bins 5 \
  --metric_min_bin_size 20 \
  --bootstrap_reps 20 \
  --output_dir "${SMOKE_DIR}"

"${PY}" "${VALIDATE}" \
  --output_dir "${SMOKE_DIR}" \
  --expected_units 3 \
  --expected_rows 15 \
  --expected_summary_rows 15 \
  --require_weighting_none \
  --require_all_methods

"${PY}" "${BENCH}" \
  --stage tune \
  --tuning_mode base \
  --output_dir "${BASE_DIR}"

"${PY}" "${VALIDATE}" \
  --output_dir "${BASE_DIR}" \
  --expected_units 1440 \
  --expected_rows 7200 \
  --require_selected_configs \
  --expect_base_selection \
  --require_weighting_none \
  --require_all_methods

"${PY}" "${BENCH}" \
  --stage tune \
  --tuning_mode calibrator \
  --tuned_config_path "${BASE_DIR}/tuned_configs.json" \
  --output_dir "${CAL_DIR}"

"${PY}" "${VALIDATE}" \
  --output_dir "${CAL_DIR}" \
  --expected_units 1440 \
  --expected_rows 7200 \
  --require_selected_configs \
  --expect_calibrator_selection \
  --require_credible_calibrator \
  --require_weighting_none \
  --require_all_methods

"${PY}" "${BENCH}" \
  --stage final \
  --critic_families linear_fqe \
  --tuned_config_path "${CAL_DIR}/tuned_configs.json" \
  --output_dir "${FINAL_DIR}" \
  --bootstrap_reps 200

"${PY}" "${BENCH}" \
  --stage final \
  --critic_families rf_fqe \
  --tuned_config_path "${CAL_DIR}/tuned_configs.json" \
  --output_dir "${FINAL_DIR}" \
  --bootstrap_reps 200

for start in 0 10 20 30 40 50 60 70 80 90; do
  stop=$((start + 10))
  "${PY}" "${BENCH}" \
    --stage final \
    --critic_families neural_fqe \
    --seed_start "${start}" \
    --seed_stop "${stop}" \
    --tuned_config_path "${CAL_DIR}/tuned_configs.json" \
    --output_dir "${FINAL_DIR}" \
    --bootstrap_reps 200
done

"${PY}" "${BENCH}" \
  --stage final \
  --tuned_config_path "${CAL_DIR}/tuned_configs.json" \
  --output_dir "${FINAL_DIR}" \
  --bootstrap_reps 5000

"${PY}" "${VALIDATE}" \
  --output_dir "${FINAL_DIR}" \
  --expected_units 3300 \
  --expected_rows 16500 \
  --expected_summary_rows 15 \
  --require_weighting_none \
  --require_all_methods \
  --require_debiased_metrics \
  --require_bootstrap_ci

date
