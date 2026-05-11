---
title: Discounted Occupancy Ratios Quickstart
description: First workflows for FORI and tuning.
---

## Stable boosted fit

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
```

## Automatic tuning

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

The default automatic tuner uses the neural family. Callers can opt into boosted
or mixed search with `families=("boosted",)` or
`families=("boosted", "neural")`.

## Target-validation assisted tuning

```python
from occupancy_ratio import tune_occupancy_ratio_with_target_validation

tuned = tune_occupancy_ratio_with_target_validation(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    target_next_actions=target_next_actions_under_pi,
    rewards=rewards,
    gamma=0.99,
    initial_states=initial_states,
    initial_actions=initial_actions_under_pi,
    validation_states=target_states,
    validation_actions=target_actions,
    validation_rewards=target_rewards,
    validation_episode_ids=target_episode_ids,
    validation_timestep=target_timesteps,
    validation_continuation=target_continuation,
)
```

The default `score_mode="discounted_moments"` compares candidate
reference-weighted moments against empirical discounted occupancy moments from
target-policy validation rollouts.
