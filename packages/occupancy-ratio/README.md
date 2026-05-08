# occupancy-ratio

Discounted occupancy-ratio estimation tools for off-policy RL evaluation.

The importable Python package is `occupancy_ratio`; install it from this
directory:

```bash
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
```

## Boosted Occupancy Ratio

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
    target_actions=actions_under_target_policy,
    gamma=0.9,
    occupancy=OccupancyRegressionConfig.stable_defaults(seed=123),
    action_ratio=ActionRatioConfig(density_ratio_loss="lsif"),
    transition_ratio=TransitionRatioConfig(density_ratio_loss="lsif"),
)

weights = model.predict_state_action_ratio(states, actions)
state_ratios = model.predict_state_ratio(states, actions)
```

LSIF is the default density-ratio nuisance loss. Logistic action and transition
nuisances are available with `density_ratio_loss="logistic"` and remain opt-in.

## Neural Occupancy Ratio

```python
from occupancy_ratio import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralTransitionRatioConfig,
    fit_discounted_occupancy_ratio_neural,
)

model = fit_discounted_occupancy_ratio_neural(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=actions_under_target_policy,
    gamma=0.9,
    occupancy=NeuralOccupancyRegressionConfig(seed=123),
    action_ratio=NeuralActionRatioConfig(density_ratio_loss="lsif"),
    transition_ratio=NeuralTransitionRatioConfig(density_ratio_loss="lsif"),
)
```

The neural estimator uses mini-batch gradient updates for action, transition,
and occupancy components. Cross-fitting uses held-out nuisance predictors in
the fixed-point target builder while retaining full-data predictors for public
prediction helpers.

## Benchmarks

Install the benchmark extra and run:

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators oracle boosted_tree neural_network \
  --no-google-dualdice \
  --no-plots
```

The module entrypoints are `python -m occupancy_ratio_benchmark.run`,
`python -m occupancy_ratio_benchmark.dualdice_grid`, and
`python -m occupancy_ratio_benchmark.defaults_report`. Use
`--profile overnight` for the hybrid controlled plus Gymnasium/MuJoCo defaults
sweep against official Google DualDICE.
