#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PY="./.venv/bin/python"
BENCH="FQE_calibration_neurips/scripts/run_hopper_q_calibration_benchmark.py"
VALIDATE="FQE_calibration_neurips/scripts/validate_hopper_q_calibration_run.py"

SOURCE_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline_final_nonlin_hopper192_s20"
OUT_DIR="FQE_calibration_neurips/results/hopper_q_calibration_deadline_sample512_nonlin_s10"
LOG_DIR="FQE_calibration_neurips/results/logs/deadline_sample512_nonlin_s10"
mkdir -p "${LOG_DIR}"

common_args=(
  --stage final
  --critic_families rf_fqe neural_fqe
  --allow_default_final_config
  --max_trajectories 512
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
  echo "[sample512] waiting for 20-seed source aggregate $(date)"
  while true; do
    local n
    n="$(unit_count "${SOURCE_DIR}")"
    if [ "${n}" -ge 440 ] && [ -f "${SOURCE_DIR}/hopper_q_calibration_manifest.json" ]; then
      local expected completed failed
      expected="$("${PY}" -c "import json; print(json.load(open('${SOURCE_DIR}/hopper_q_calibration_manifest.json')).get('n_expected_units'))")"
      completed="$("${PY}" -c "import json; print(json.load(open('${SOURCE_DIR}/hopper_q_calibration_manifest.json')).get('n_completed_unit_files'))")"
      failed="$("${PY}" -c "import json; print(json.load(open('${SOURCE_DIR}/hopper_q_calibration_manifest.json')).get('n_failed_units'))")"
      if [ "${expected}" = "440" ] && [ "${completed}" = "440" ] && [ "${failed}" = "0" ]; then
        break
      fi
    fi
    echo "[sample512] source units=${n}; sleeping"
    sleep 60
  done
}

echo "[sample512] launched $(date)"
"${PY}" -m py_compile "${BENCH}" "${VALIDATE}"
wait_for_source20

"${PY}" "${BENCH}" "${common_args[@]}" \
  --seed_start 0 \
  --seed_stop 10 \
  --bootstrap_reps 2000 \
  >"${LOG_DIR}/seeds_00_10.out.log" \
  2>"${LOG_DIR}/seeds_00_10.err.log"

"${PY}" "${VALIDATE}" \
  --output_dir "${OUT_DIR}" \
  --expected_units 220 \
  --expected_rows 1100 \
  --expected_summary_rows 10 \
  --critic_families rf_fqe neural_fqe \
  --require_weighting_none \
  --require_all_methods \
  --require_bootstrap_ci

echo "[sample512] completed $(date)"
