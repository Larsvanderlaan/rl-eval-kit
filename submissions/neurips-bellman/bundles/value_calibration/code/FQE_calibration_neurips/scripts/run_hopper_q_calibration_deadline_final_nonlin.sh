#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

OUT_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline_final_nonlin_hopper192_s20"
LOG_DIR="FQE_calibration_neurips/results/logs/deadline_final_nonlin_hopper192_s20"
mkdir -p "${LOG_DIR}"
LAST_PID=""

common_args=(
  --stage final
  --critic_families rf_fqe neural_fqe
  --allow_default_final_config
  --max_trajectories 192
  --rf_components 256
  --ridge 1e-3
  --fit_action_samples 2
  --fqe_updates 100
  --critic_lr 3e-4
  --target_tau 0.005
  --batch_size 256
  --device mps
  --torch_threads 1
  --calibrators none linear isotonic histogram isotonic_histogram
  --calibration_iterations 3
  --n_bins 10
  --min_bin_size 50
  --metric_bins 10
  --metric_min_bin_size 50
  --action_samples 2
  --initial_action_samples 4
  --output_dir "${OUT_DIR}"
)

run_shard_bg() {
  local label="$1"
  local start="$2"
  local stop="$3"
  echo "[deadline-final] start ${label} $(date)" >&2
  "${PY}" "${BENCH}" "${common_args[@]}" \
    --seed_start "${start}" \
    --seed_stop "${stop}" \
    --bootstrap_reps 20 \
    >"${LOG_DIR}/${label}.out.log" \
    2>"${LOG_DIR}/${label}.err.log" &
  LAST_PID="$!"
}

wait_pair() {
  local first="$1"
  local second="$2"
  local failed=0
  wait "${first}" || failed=1
  wait "${second}" || failed=1
  if [ "${failed}" -ne 0 ]; then
    echo "[deadline-final] shard failure; see ${LOG_DIR}" >&2
    exit 1
  fi
}

echo "[deadline-final] launched $(date)"
"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"

run_shard_bg seeds_00_05 0 5
pid1="${LAST_PID}"
run_shard_bg seeds_05_10 5 10
pid2="${LAST_PID}"
wait_pair "${pid1}" "${pid2}"

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 10 \
  --bootstrap_reps 500

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 220 \
  --expected_rows 1100 \
  --expected_summary_rows 10 \
  --critic_families rf_fqe neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

run_shard_bg seeds_10_15 10 15
pid3="${LAST_PID}"
run_shard_bg seeds_15_20 15 20
pid4="${LAST_PID}"
wait_pair "${pid3}" "${pid4}"

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 20 \
  --bootstrap_reps 2000

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 440 \
  --expected_rows 2200 \
  --expected_summary_rows 10 \
  --critic_families rf_fqe neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

echo "[deadline-final] completed $(date)"
