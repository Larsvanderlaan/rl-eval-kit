#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

OUT_DIR="FQE_calibration_neurips/results/hopper_q_calibration_papernow_neural20_mps_u1000"
LOG_DIR="FQE_calibration_neurips/results/logs/papernow_neural20_mps_u1000"
mkdir -p "${LOG_DIR}"
LAST_PID=""

common_args=(
  --stage final
  --critic_families neural_fqe
  --allow_default_final_config
  --fqe_updates 1000
  --critic_lr 3e-4
  --target_tau 0.005
  --batch_size 256
  --device mps
  --torch_threads 1
  --calibrators none linear isotonic histogram isotonic_histogram
  --calibration_iterations 4
  --n_bins 20
  --min_bin_size 100
  --metric_bins 30
  --metric_min_bin_size 300
  --action_samples 4
  --initial_action_samples 16
  --output_dir "${OUT_DIR}"
)

run_shard_bg() {
  local label="$1"
  local start="$2"
  local stop="$3"
  echo "[papernow-neural20] start ${label} $(date)" >&2
  "${PY}" "${BENCH}" "${common_args[@]}" \
    --seed_start "${start}" \
    --seed_stop "${stop}" \
    --bootstrap_reps 20 \
    >"${LOG_DIR}/${label}.out.log" \
    2>"${LOG_DIR}/${label}.err.log" &
  LAST_PID="$!"
}

echo "[papernow-neural20] launched $(date)"
"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"

run_shard_bg seeds_00_05 0 5
pid1="${LAST_PID}"
run_shard_bg seeds_05_10 5 10
pid2="${LAST_PID}"

failed=0
wait "${pid1}" || failed=1
wait "${pid2}" || failed=1
if [ "${failed}" -ne 0 ]; then
  echo "[papernow-neural20] shard failure; see ${LOG_DIR}" >&2
  exit 1
fi

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 10 \
  --bootstrap_reps 1000

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 110 \
  --expected_rows 550 \
  --expected_summary_rows 5 \
  --critic_families neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

run_shard_bg seeds_10_15 10 15
pid3="${LAST_PID}"
run_shard_bg seeds_15_20 15 20
pid4="${LAST_PID}"

failed=0
wait "${pid3}" || failed=1
wait "${pid4}" || failed=1
if [ "${failed}" -ne 0 ]; then
  echo "[papernow-neural20] shard failure; see ${LOG_DIR}" >&2
  exit 1
fi

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 20 \
  --bootstrap_reps 2000

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 220 \
  --expected_rows 1100 \
  --expected_summary_rows 5 \
  --critic_families neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

echo "[papernow-neural20] completed $(date)"
