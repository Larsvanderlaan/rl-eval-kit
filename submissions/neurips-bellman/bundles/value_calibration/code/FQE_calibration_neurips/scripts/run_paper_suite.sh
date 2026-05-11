#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-FQE_calibration_neurips/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-FQE_calibration_neurips/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

"${PYTHON_BIN}" FQE_calibration_neurips/scripts/run_suite.py \
  --suite_config FQE_calibration_neurips/configs/paper_suite.yaml \
  --mode paper

"${PYTHON_BIN}" FQE_calibration_neurips/scripts/aggregate_results.py \
  --results_dir FQE_calibration_neurips/results/paper

"${PYTHON_BIN}" FQE_calibration_neurips/scripts/make_plots.py \
  --results_dir FQE_calibration_neurips/results/paper \
  --figures_dir FQE_calibration_neurips/figures/paper
