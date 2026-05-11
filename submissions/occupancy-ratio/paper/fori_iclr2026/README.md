# Fitted Occupancy-Ratio Iteration Paper

This directory is the canonical source for the ICLR 2026 FORI manuscript. The files were copied from `/Users/larsvanderlaan/Downloads/Fitted_Occupancy_Ratio_Iteration.zip`; generated LaTeX artifacts and preliminary experiment outputs are intentionally not checked in here.

## Build

```bash
cd submissions/occupancy-ratio/paper/fori_iclr2026
latexmk -pdf main.tex
```

## Package Tests

```bash
cd /path/to/rl-eval-kit
PYTHONPATH=packages/occupancy-ratio .venv/bin/python -m pytest \
  packages/occupancy-ratio/occupancy_ratio_benchmark/tests -q
```

## Paper-Specific Exact Finite-MDP Harness

```bash
cd /path/to/rl-eval-kit
PYTHONPATH=packages/occupancy-ratio:submissions/occupancy-ratio .venv/bin/python -m experiments.occupancy_ratio.fori_eval.runner \
  --profile smoke \
  --run-id fori_paper_smoke \
  --estimators oracle population_fori boosted_tree_stable neural_network_stable \
  --output-root outputs/fori_eval
```

## Package Benchmark Configs

```bash
cd /path/to/rl-eval-kit
PYTHONPATH=packages/occupancy-ratio .venv/bin/python -m occupancy_ratio_benchmark.run \
  --config packages/occupancy-ratio/occupancy_ratio_benchmark/configs/fori_paper_smoke.json \
  --output-root outputs/occupancy_ratio_paper
```

Use `fori_paper_core.json` for controlled learned-estimator comparisons and `fori_paper_dice.json` when the Google Research DualDICE checkout is available. The manuscript should not claim empirical results until generated outputs are reviewed.
