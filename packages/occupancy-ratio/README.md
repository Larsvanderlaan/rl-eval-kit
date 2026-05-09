# occupancy-ratio

Discounted occupancy-ratio estimators for off-policy reinforcement-learning
evaluation.

The importable Python package is `occupancy_ratio`. It provides boosted and
neural fitted occupancy-ratio iteration (FORI), conservative product AutoML
tuning, diagnostics/calibration helpers, benchmark tooling, and an optional
Google DualDICE comparator.

## Install

```bash
python -m pip install -e "packages/occupancy-ratio"
```

Common extras:

```bash
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
python -m pip install -e "packages/occupancy-ratio[docs]"
```

Build the documentation site:

```bash
python -m mkdocs build --strict -f packages/occupancy-ratio/mkdocs.yml
```

## Quickstart

```python
from occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
)

model = fit_discounted_occupancy_ratio(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    gamma=0.99,
    occupancy=OccupancyRegressionConfig.stable_defaults(seed=123),
    action_ratio=ActionRatioConfig.stable_defaults(show_progress=False),
    transition_ratio=TransitionRatioConfig.stable_defaults(show_progress=False),
)

weights = model.predict_state_action_ratio(states, actions)
state_ratios = model.predict_state_ratio(states, actions)
```

For production use, start with the boosted stable defaults for a single fit or
the neural-default AutoML entrypoint for candidate tuning:

```python
from occupancy_ratio import OccupancyTuningConfig, tune_occupancy_ratio_auto

tuned = tune_occupancy_ratio_auto(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    gamma=0.99,
    rewards=rewards,
    config=OccupancyTuningConfig(budget="balanced"),
)

model = tuned.model
candidate_rows = tuned.candidate_rows()
fold_rows = tuned.fold_rows()
```

AutoML selection uses proxy diagnostics only; it never selects using oracle
ratios, benchmark truth, or target-policy Monte Carlo values.

## Documentation Map

The MkDocs site under `docs/` contains:

- installation and optional extras;
- package architecture and stable import paths;
- boosted and neural quickstarts;
- data-shape contracts;
- initial-source correction semantics;
- AutoML tuning behavior and telemetry;
- diagnostics, calibration, and benchmark usage;
- generated API reference pages for public functions, configs, and models.

## Benchmarks

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators oracle boosted_tree neural_network \
  --no-google-dualdice \
  --no-plots
```

The module entrypoints are `python -m occupancy_ratio_benchmark.run`,
`python -m occupancy_ratio_benchmark.dualdice_grid`, and
`python -m occupancy_ratio_benchmark.defaults_report`.
