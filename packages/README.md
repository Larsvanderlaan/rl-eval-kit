# RLEvalKit Packages

This folder contains the production Python distributions:

- `fqe`: fitted Q evaluation and fitted value iteration tools.
- `occupancy-ratio`: occupancy-ratio estimators and benchmark tools.
- `genpqr`: generalized policy-to-Q-to-reward tools for inverse RL.
- `causal-ope-benchmark`: causal inference and industry OPE benchmark simulators.

Install all packages for local development:

```bash
python -m pip install -r packages/requirements-dev.txt
```

or install them individually:

```bash
python -m pip install -e "packages/fqe[neural,benchmark]"
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
python -m pip install -e "packages/genpqr[dev]"
python -m pip install -e "packages/causal-ope-benchmark[dev]"
```

Supported imports are `fqe`, `fqe_benchmark`, `occupancy_ratio`,
`occupancy_ratio_benchmark`, `genpqr`, and `causal_ope_benchmark`.
Submission-specific and retired root namespaces live outside the release
surface.

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
python -m ruff check packages/fqe packages/occupancy-ratio packages/genpqr packages/causal-ope-benchmark
```
