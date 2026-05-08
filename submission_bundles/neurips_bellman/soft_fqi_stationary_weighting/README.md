# Anonymous Reproducibility Bundle: Stationary-Weighted Soft FQI

This bundle contains the source code, configs, final draft artifacts, and tests needed to reproduce the soft-FQI stationary-weighting simulations. It omits manuscript files, caches, local provenance, exploratory raw directories, and broad generated tuning grids.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke Test

```bash
PYTHONPATH=code/soft_fqi_stationary_weighting python -m pytest code/soft_fqi_stationary_weighting/tests -q
PYTHONPATH=code/soft_fqi_stationary_weighting python code/soft_fqi_stationary_weighting/scripts/run_experiment.py   --config code/soft_fqi_stationary_weighting/configs/debug.yaml   --overwrite
```

## Full Reproduction

The 200-replication paper run and artifacts can be regenerated with:

```bash
PYTHONPATH=code/soft_fqi_stationary_weighting python code/soft_fqi_stationary_weighting/scripts/run_experiment.py   --config code/soft_fqi_stationary_weighting/configs/paper_main_200_stabilized.yaml   --overwrite
PYTHONPATH=code/soft_fqi_stationary_weighting python code/soft_fqi_stationary_weighting/scripts/aggregate_results.py   --results-dir code/soft_fqi_stationary_weighting/results/paper_main_200_stabilized
PYTHONPATH=code/soft_fqi_stationary_weighting python code/soft_fqi_stationary_weighting/scripts/make_plots.py   --results-dir code/soft_fqi_stationary_weighting/results/paper_main_200_stabilized
PYTHONPATH=code/soft_fqi_stationary_weighting python code/soft_fqi_stationary_weighting/scripts/make_tables.py   --results-dir code/soft_fqi_stationary_weighting/results/paper_main_200_stabilized
```

## Included Final Artifacts

- `code/soft_fqi_stationary_weighting/final_draft_bundle/`: final figure/table/snippet bundle used for the submitted paper.
- `paper_artifacts/figures/`: figures referenced by the submitted paper.
- `paper_artifacts/tables/`: final table inputs referenced by the submitted paper.

The full simulation is seeded but can show small numerical drift across BLAS, hardware, and dependency versions.
