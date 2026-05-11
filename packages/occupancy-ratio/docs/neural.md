# Neural Estimator

`fit_discounted_occupancy_ratio_neural(...)` implements the same fitted
occupancy-ratio iteration as the boosted estimator, but uses PyTorch MLPs for
the nuisance and occupancy stages.

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
    target_actions=target_actions,
    target_next_actions=target_next_actions,
    gamma=0.99,
    occupancy=NeuralOccupancyRegressionConfig.stable_defaults(seed=123),
    action_ratio=NeuralActionRatioConfig.stable_defaults(seed=123),
    transition_ratio=NeuralTransitionRatioConfig.stable_defaults(seed=123),
)
```

## When To Use It

Use the neural estimator when:

- the state-action space is continuous and smooth;
- boosted trees struggle with representation capacity;
- you want a closer architectural comparison to neural DICE methods.

Use the boosted estimator first for tabular, small, or well-covered settings;
it is often easier to diagnose even though AutoML defaults to neural candidates.

## Stage Budgets

The neural pipeline has independent budgets for:

- action-ratio nuisance fitting;
- initial/source nuisance fitting;
- transition or direct one-step nuisance fitting;
- occupancy fixed-point regression.

For Gym-style workloads, spend enough capacity on the nuisance stages before
only widening the occupancy network. Underfitted nuisance ratios can make a
large occupancy network look stable while recovering nearly uniform weights.

## Device

Set `device="cuda"` in the config objects when PyTorch and the environment
support GPU execution. Keep seeds explicit for reproducible comparisons.
