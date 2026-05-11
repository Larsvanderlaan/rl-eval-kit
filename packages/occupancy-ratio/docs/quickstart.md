# Quickstart

This example fits a boosted occupancy-ratio model and uses it for a simple
weighted OPE estimate.

```python
import numpy as np
from occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
    weight_summary,
)

# Logged behavior-policy transitions.
# Each array has one row per transition.
states = dataset.states
actions = dataset.actions
next_states = dataset.next_states
rewards = dataset.rewards

# Actions sampled from the target policy at each logged current state.
# For continuous policies this is usually one Monte Carlo action per row.
target_actions = dataset.target_actions

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

# Discounted state-action occupancy weights on observed behavior rows.
omega = model.predict_state_action_ratio(states, actions)

# A simple normalized weighted reward diagnostic.
# Production OPE should report uncertainty and sensitivity diagnostics too.
value_estimate = np.sum(omega * rewards) / np.sum(omega)
print(value_estimate)
print(weight_summary(omega, cap=model.occupancy_ratio_max))
```

## What To Inspect First

After fitting, check:

- `model.diagnostics`: first-stage and final-ratio diagnostics.
- `model.history`: fixed-point iteration history.
- `weight_summary(omega)`: ESS, tails, nonfinite fraction, and clipping.
- `model.predict_action_ratio(states, actions)`: whether the action nuisance is
  learning a meaningful behavior-target shift.

## When To Use AutoML

Use `tune_occupancy_ratio_auto(...)` when a single stable fit is too sensitive
or when you want a conservative candidate search that still avoids oracle truth.

```python
from occupancy_ratio import OccupancyTuningConfig, tune_occupancy_ratio_auto

tuned = tune_occupancy_ratio_auto(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions,
    gamma=0.99,
    rewards=rewards,
    config=OccupancyTuningConfig(budget="balanced"),
)

model = tuned.model
candidate_rows = tuned.candidate_rows()
fold_rows = tuned.fold_rows()
```

By default, AutoML searches neural candidates only. Opt into boosted candidates
with `families=("boosted",)` or `families=("boosted", "neural")`.
