#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export MPLCONFIGDIR="${ROOT}/.mplconfig"
export XDG_CACHE_HOME="${ROOT}/.cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" "${ROOT}/scripts/run_experiment.py" --config "${ROOT}/configs/debug.yaml" --overwrite
"${PYTHON}" "${ROOT}/scripts/aggregate_results.py" --results-dir "${ROOT}/results/debug"
"${PYTHON}" "${ROOT}/scripts/make_plots.py" --results-dir "${ROOT}/results/debug"
"${PYTHON}" "${ROOT}/scripts/make_tables.py" --results-dir "${ROOT}/results/debug"
"${PYTHON}" "${ROOT}/scripts/assess_story.py" --results-dir "${ROOT}/results/debug"
