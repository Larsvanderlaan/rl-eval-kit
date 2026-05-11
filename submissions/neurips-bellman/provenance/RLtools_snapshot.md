# RLtools Snapshot

Copied from `/Users/larsvanderlaan/repos/RLtools`.

- Branch: `main`
- Commit: `4fc1c77fac1139c028077facececf5f1ff5f868f`
- Copy time: `2026-05-06T13:48:29`

## Dirty / Untracked Status At Copy Time

```text
M FQE/fqe_boosted.py
 M IRL/__init__.py
 M experiments/figures/baird_fqi_comparison.pdf
 M experiments/figures/baird_fqi_comparison_behavior_norm_0.95.pdf
 M experiments/figures/baird_fqi_two_panel.pdf
?? FQE_calibration_neurips/
?? FQE_neurips/
?? IRL/IRL_journal/
?? IRL/jrssb_simulation.py
?? IRL/output_final/
?? IRL/outputs/
?? IRL/run_jrssb_simulation.py
?? IRL_neurips/
?? experiments/crm_nn_snapshot_50000_nosplit_ptrain_1p0_20260325_173318.pkl
?? experiments/crm_nn_snapshot_50000_nosplit_ptrain_1p0_20260325_173318.txt
?? experiments/crm_nn_snapshot_50000_split_ptrain_0p5_20260325_173318.pkl
?? experiments/crm_nn_snapshot_50000_split_ptrain_0p5_20260325_173318.txt
?? experiments/crm_nn_snapshot_50000_split_ptrain_0p75_20260325_173318.pkl
?? experiments/crm_nn_snapshot_50000_split_ptrain_0p75_20260325_173318.txt
?? experiments/crm_nn_snapshot_50000_split_ptrain_0p9_20260325_173318.pkl
?? experiments/crm_nn_snapshot_50000_split_ptrain_0p9_20260325_173318.txt
?? experiments/figures/baird_FQE_comparison.pdf
?? experiments/figures/baird_FQE_comparison_behavior_norm_0.95.pdf
?? experiments/figures/baird_FQE_two_panel.pdf
?? hopper_fqe_benchmark/
?? main_FQE_neurips_29_experiments_updated.aux
?? main_FQE_neurips_29_experiments_updated.fdb_latexmk
?? main_FQE_neurips_29_experiments_updated.fls
```

## Copied Experiment Payloads

- `experiments/fqe/FQE_neurips`: top-level Python modules, `controlled_discounted_benchmark/`, selected top-level CSV/MD summaries from `results/`, and `paper_experiment_bundle/`.
- `experiments/calibration/FQE_calibration_neurips`: `src/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`, `README.md`, `paper_import_bundle/`, top-level result protocol/readme files, `results/rescue_final/`, and `figures/rescue_final/`.
- `experiments/calibration/hopper_fqe_benchmark`: support package required by calibration Hopper tests, including code and the small `artifacts/benchmark/dope/` pickle subset.

## Exclusions

Caches, Python bytecode, LaTeX build artifacts, notebook checkpoint folders, broad exploratory/raw result directories outside the audited or selected summary outputs, and large Hopper datasets/model artifacts were intentionally excluded.
