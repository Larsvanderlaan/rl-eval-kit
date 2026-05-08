#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

SOURCE_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline_final_nonlin_hopper192_s20"
OUT_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline_final_nonlin_hopper192_s50"
LOG_DIR="FQE_calibration_neurips/results/logs/deadline_final_nonlin_hopper192_s50"
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

unit_count() {
  find "$1/units" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' '
}

wait_for_source20() {
  echo "[extend50] waiting for 20-seed source aggregate $(date)"
  while true; do
    local n
    n="$(unit_count "${SOURCE_DIR}")"
    if [ "${n}" -ge 440 ]; then
      if [ -f "${SOURCE_DIR}/hopper_q_calibration_manifest.json" ]; then
        local expected completed failed
        expected="$("${PY}" -c "import json; print(json.load(open('${SOURCE_DIR}/hopper_q_calibration_manifest.json')).get('n_expected_units'))")"
        completed="$("${PY}" -c "import json; print(json.load(open('${SOURCE_DIR}/hopper_q_calibration_manifest.json')).get('n_completed_unit_files'))")"
        failed="$("${PY}" -c "import json; print(json.load(open('${SOURCE_DIR}/hopper_q_calibration_manifest.json')).get('n_failed_units'))")"
        if [ "${expected}" = "440" ] && [ "${completed}" = "440" ] && [ "${failed}" = "0" ]; then
          break
        fi
      fi
    fi
    echo "[extend50] source units=${n}; sleeping"
    sleep 60
  done
  "${PY}" "${VALIDATE}" \
    --output_dir "${SOURCE_DIR}" \
    --expected_units 440 \
    --expected_rows 2200 \
    --expected_summary_rows 10 \
    --critic_families rf_fqe neural_fqe \
    --require_weighting_none \
    --require_all_methods \
    --require_bootstrap_ci
}

seed_output_from_source() {
  mkdir -p "${OUT_DIR}/units"
  find "${SOURCE_DIR}/units" -maxdepth 1 -type f -name '*.csv' -exec cp {} "${OUT_DIR}/units/" \;
  echo "[extend50] copied $(unit_count "${OUT_DIR}") source units into ${OUT_DIR}"
}

run_shard_bg() {
  local label="$1"
  local start="$2"
  local stop="$3"
  echo "[extend50] start ${label} $(date)" >&2
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
    echo "[extend50] shard failure; see ${LOG_DIR}" >&2
    exit 1
  fi
}

run_pair() {
  local label1="$1"
  local start1="$2"
  local stop1="$3"
  local label2="$4"
  local start2="$5"
  local stop2="$6"
  run_shard_bg "${label1}" "${start1}" "${stop1}"
  local pid1="${LAST_PID}"
  run_shard_bg "${label2}" "${start2}" "${stop2}"
  local pid2="${LAST_PID}"
  wait_pair "${pid1}" "${pid2}"
}

echo "[extend50] launched $(date)"
"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"
wait_for_source20
seed_output_from_source

run_pair seeds_20_25 20 25 seeds_25_30 25 30
run_pair seeds_30_35 30 35 seeds_35_40 35 40
run_pair seeds_40_45 40 45 seeds_45_50 45 50

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 50 \
  --bootstrap_reps 3000

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 1100 \
  --expected_rows 5500 \
  --expected_summary_rows 10 \
  --critic_families rf_fqe neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

echo "[extend50] completed $(date)"
