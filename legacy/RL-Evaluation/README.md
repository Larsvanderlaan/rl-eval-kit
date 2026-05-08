# RL Evaluation Packages

This folder contains two production Python distributions:

- `FQE`: installs the `fqe` package for fitted Q evaluation and fitted value iteration.
- `occupancy-ratio`: installs the `occupancy_ratio` package and optional `occupancy_ratio_benchmark` tools.

Install both packages for local development:

```bash
cd RL-Evaluation
python -m pip install -r requirements-dev.txt
```

or install them individually:

```bash
python -m pip install -e "FQE[neural,benchmark]"
python -m pip install -e "occupancy-ratio[neural,benchmark]"
```

Supported imports are `fqe`, `fqe_benchmark`, `occupancy_ratio`, and
`occupancy_ratio_benchmark`. Legacy repository namespaces such as `FQE.*` and
`IRL.fit_occupancy_ratio*` remain as transitional shims and emit deprecation
warnings where practical.

## Commands

```bash
fqe-benchmark --stage smoke --no-plots
occupancy-ratio-benchmark --profile smoke --estimators oracle boosted_tree neural_network --no-google-dualdice --no-plots
occupancy-ratio-gridwalk --help
```

## Development Checks

From the repository root after editable install:

```bash
python -m pytest
python -m ruff check RL-Evaluation/FQE RL-Evaluation/occupancy-ratio
```
