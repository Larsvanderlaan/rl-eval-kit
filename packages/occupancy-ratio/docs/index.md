# occupancy-ratio

`occupancy-ratio` estimates discounted occupancy density ratios for off-policy
reinforcement-learning evaluation. It is built for users who have logged
transitions from a behavior policy and actions sampled from an evaluation
policy, and want stable weights for OPE, diagnostics, or method comparison.

The package exposes three practical layers:

- **Boosted FORI**: LightGBM fitted occupancy-ratio iteration, useful for
  stable single-model fits and tabular or well-covered settings.
- **Neural FORI**: PyTorch fitted occupancy-ratio iteration for larger
  continuous-control workloads.
- **AutoML tuning**: conservative cross-validation that selects among FORI
  candidates using proxy diagnostics only.

It also includes optional Google DualDICE integration for apples-to-apples
comparison with the Google Research implementation and a benchmark CLI for
controlled and realistic evaluation suites.

For DICE-style and minimax comparators, `fit_minimax_weight(...)` exposes one
common interface over the official Google `policy_eval` DualDICE wrapper,
Google DICE-RL NeuralDice variants, and SCOPE-RL minimax weight learners.

## The Object Being Estimated

The main fitted model predicts the discounted state-action occupancy ratio

```text
rho_pi,gamma(s) * pi(a | s) / [rho_ref(s) * pi0(a | s)]
```

on state-action rows. In ordinary logged-transition use, `rho_ref` is the
state distribution represented by the estimator's reference rows. The fitted
model also exposes an action-ratio nuisance estimate and a state-ratio helper.

## Recommended Starting Point

Use `fit_discounted_occupancy_ratio(...)` with stable defaults for a single
model fit, or `tune_occupancy_ratio_auto(...)` for the neural-default AutoML
search.

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
    target_actions=target_actions,
    gamma=0.99,
    occupancy=OccupancyRegressionConfig.stable_defaults(seed=123),
    action_ratio=ActionRatioConfig.stable_defaults(show_progress=False),
    transition_ratio=TransitionRatioConfig.stable_defaults(show_progress=False),
)

omega = model.predict_state_action_ratio(states, actions)
```

See the [quickstart](quickstart.md) and [data-shape guide](data-shapes.md) for
the exact array contract.
