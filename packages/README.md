# RL Evaluation Packages

This folder contains the production Python distributions:

- `fqe`: fitted Q evaluation and fitted value iteration tools.
- `occupancy-ratio`: occupancy-ratio estimators and benchmark tools.
- `bellman-trees`: target-weighted Bellman aggregation trees and forests.

Install both packages for local development:

```bash
python -m pip install -r packages/requirements-dev.txt
```

or install them individually:

```bash
python -m pip install -e "packages/fqe[neural,benchmark]"
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
python -m pip install -e "packages/bellman-trees[test]"
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
python -m ruff check packages/fqe packages/occupancy-ratio packages/bellman-trees
```
