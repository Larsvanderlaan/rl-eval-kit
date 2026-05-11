#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

BOOTSTRAP_SHARD="${BOOTSTRAP_SHARD:-20}"
BOOTSTRAP_INTERIM="${BOOTSTRAP_INTERIM:-1000}"
BOOTSTRAP_FINAL="${BOOTSTRAP_FINAL:-5000}"

BASE_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline2_tune_base"
CAL_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline2_tune_calibrator"
FINAL_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline2_final"
LOG_DIR="FQE_calibration_neurips/results/logs/deadline_parallel2"
mkdir -p "${LOG_DIR}"

LAST_PID=""

run_shard_bg() {
  local label="$1"
  shift
  echo "[deadline] start ${label} $(date)"
  {
    echo "[deadline] command ${label}: ${PY} ${BENCH} $*"
    "${PY}" "${BENCH}" "$@"
  } >"${LOG_DIR}/${label}.out.log" 2>"${LOG_DIR}/${label}.err.log" &
  LAST_PID="$!"
}

wait_pid_group() {
  local failed=0
  local pid
  for pid in "$@"; do
    if [ -n "${pid}" ] && ! wait "${pid}"; then
      failed=1
    fi
  done
  if [ "${failed}" -ne 0 ]; then
    echo "[deadline] one or more shards failed; see ${LOG_DIR}" >&2
    exit 1
  fi
}

run_seed_pairs() {
  local prefix="$1"
  local first="$2"
  local last="$3"
  local step="$4"
  shift 4
  local starts=()
  local start
  for start in $(seq "${first}" "${step}" $((last - step))); do
    starts+=("${start}")
  done
  local i=1
  while [ "${i}" -le "${#starts[@]}" ]; do
    local pids=()
    local j
    for j in 0 1; do
      local idx=$((i + j))
      if [ "${idx}" -le "${#starts[@]}" ]; then
        start="${starts[${idx}]}"
        local stop=$((start + step))
        run_shard_bg "${prefix}_${start}_${stop}" "$@" --seed_start "${start}" --seed_stop "${stop}"
        pids+=("${LAST_PID}")
      fi
    done
    wait_pid_group "${pids[@]}"
    i=$((i + 2))
  done
}

base_args=(
  --stage tune
  --tuning_mode base
  --critic_families linear_fqe rf_fqe neural_fqe
  --policy_indices 0 3 6 10
  --tune_ridges 1e-3
  --tune_fit_action_samples 4
  --tune_rf_components 256
  --tune_neural_updates 5000
  --tune_neural_lrs 3e-4
  --tune_neural_taus 0.005
  --torch_threads 1
  --bootstrap_reps "${BOOTSTRAP_SHARD}"
  --output_dir "${BASE_DIR}"
)

calibrator_args=(
  --stage tune
  --tuning_mode calibrator
  --critic_families linear_fqe rf_fqe neural_fqe
  --policy_indices 0 3 6 10
  --tuned_config_path "${BASE_DIR}/tuned_configs.json"
  --tune_calibrator_bins "10 20"
  --tune_calibrator_min_bin_sizes 100
  --tune_calibration_iterations 4
  --torch_threads 1
  --bootstrap_reps "${BOOTSTRAP_SHARD}"
  --output_dir "${CAL_DIR}"
)

final_args=(
  --stage final
  --critic_families linear_fqe rf_fqe neural_fqe
  --tuned_config_path "${CAL_DIR}/tuned_configs.json"
  --torch_threads 1
  --output_dir "${FINAL_DIR}"
)

echo "[deadline] launched $(date)"
echo "[deadline] explicit pairwise shard parallelism"

"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"

run_seed_pairs "base" 1000 1010 2 "${base_args[@]}"

"${PY}" "${BENCH}" "${base_args[@]}" --seed_start 1000 --seed_stop 1010 --bootstrap_reps "${BOOTSTRAP_INTERIM}"
"${PY}" "${VALIDATE}" \
  --output_dir "${BASE_DIR}" \
  --expected_units 120 \
  --expected_rows 600 \
  --require_selected_configs \
  --expect_base_selection \
  --require_weighting_none \
  --require_all_methods

run_seed_pairs "cal" 1000 1010 2 "${calibrator_args[@]}"

"${PY}" "${BENCH}" "${calibrator_args[@]}" --seed_start 1000 --seed_stop 1010 --bootstrap_reps "${BOOTSTRAP_INTERIM}"
"${PY}" "${VALIDATE}" \
  --output_dir "${CAL_DIR}" \
  --expected_units 240 \
  --expected_rows 1200 \
  --require_selected_configs \
  --expect_calibrator_selection \
  --require_weighting_none \
  --require_all_methods

run_seed_pairs "final" 0 30 10 "${final_args[@]}" --bootstrap_reps "${BOOTSTRAP_SHARD}"

"${PY}" "${BENCH}" "${final_args[@]}" --seed_start 0 --seed_stop 30 --bootstrap_reps "${BOOTSTRAP_INTERIM}"
"${PY}" "${VALIDATE}" \
  --output_dir "${FINAL_DIR}" \
  --expected_units 990 \
  --expected_rows 4950 \
  --expected_summary_rows 15 \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

run_seed_pairs "final" 30 100 10 "${final_args[@]}" --bootstrap_reps "${BOOTSTRAP_SHARD}"

"${PY}" "${BENCH}" "${final_args[@]}" --seed_start 0 --seed_stop 100 --bootstrap_reps "${BOOTSTRAP_FINAL}"
"${PY}" "${VALIDATE}" \
  --output_dir "${FINAL_DIR}" \
  --expected_units 3300 \
  --expected_rows 16500 \
  --expected_summary_rows 15 \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

echo "[deadline] completed $(date)"
