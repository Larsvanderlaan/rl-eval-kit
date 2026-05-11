#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

OUT_DIR="FQE_calibration_neurips/results/hopper_q_calibration_hour_nonlin_hopper128_s5"
LOG_DIR="FQE_calibration_neurips/results/logs/hour_nonlin_hopper128_s5"
mkdir -p "${LOG_DIR}"
LAST_PID=""

common_args=(
  --stage final
  --critic_families rf_fqe neural_fqe
  --allow_default_final_config
  --max_trajectories 128
  --rf_components 128
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
  echo "[hour-nonlin] start ${label} $(date)" >&2
  "${PY}" "${BENCH}" "${common_args[@]}" \
    --seed_start "${start}" \
    --seed_stop "${stop}" \
    --bootstrap_reps 20 \
    >"${LOG_DIR}/${label}.out.log" \
    2>"${LOG_DIR}/${label}.err.log" &
  LAST_PID="$!"
}

echo "[hour-nonlin] launched $(date)"
"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"

run_shard_bg seeds_00_03 0 3
pid1="${LAST_PID}"
run_shard_bg seeds_03_05 3 5
pid2="${LAST_PID}"

failed=0
wait "${pid1}" || failed=1
wait "${pid2}" || failed=1
if [ "${failed}" -ne 0 ]; then
  echo "[hour-nonlin] shard failure; see ${LOG_DIR}" >&2
  exit 1
fi

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 5 \
  --bootstrap_reps 1000

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 110 \
  --expected_rows 550 \
  --expected_summary_rows 10 \
  --critic_families rf_fqe neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

echo "[hour-nonlin] completed $(date)"
