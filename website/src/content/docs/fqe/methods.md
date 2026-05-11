---
title: FQE Methods
description: Boosted FQE, neural FQE, stationary weighting, target validation, and SBV.
---

## Boosted and neural FQE

Boosted FQE uses LightGBM with conservative stable defaults: Huber loss, early
stopping, regularized tree settings, value-bound inference, deterministic
single-thread execution, and final refit in tuning workflows.

Neural FQE mirrors the boosted API and adds target networks, Polyak updates,
gradient clipping, Huber loss by default, and input standardization fitted on
the training split only.

## Target-validation assisted tuning

`tune_fqe_with_target_validation(...)` is an opt-in path for independent
target-policy rollouts or simulator labels. The default `score_mode="n_step_td"`
scores fitted candidates on finite target-policy rollout prefixes plus the
candidate continuation value at the prefix tail.

Target validation is separate from the default automatic tuner. Use it when you
have target-policy validation rollouts or labels and want candidate scores tied
to that validation set.

## Stationary weighted FQE

Stationary weighted FQE estimates discounted occupancy-ratio weights and passes
them into the same weighted FQE fitter.

```python
from fqe import GoogleDualDICEConfig, StationaryWeightedFQEConfig, fit_stationary_weighted_fqe

result = fit_stationary_weighted_fqe(
    states=states,
    actions=actions,
    target_actions=target_actions_under_eval_policy,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    gamma_ratio=0.99,
    initial_states=initial_states,
    initial_actions=initial_actions_under_eval_policy,
    config=StationaryWeightedFQEConfig(
        google_dualdice_config=GoogleDualDICEConfig(
            google_research_path="/tmp/google-research",
            num_updates=1000,
        ),
    ),
)
```

Stationary weighting can delegate to the Google DualDICE integration when its
optional dependencies and a Google Research checkout are available. The package
native FORI backend is selected explicitly with `ratio_backend="occupancy_ratio"`.

## Bellman calibration

Bellman calibration is post-hoc and separate from fitting. It checks whether
prediction bins have Bellman targets with matching means and can recommend
`apply`, `neutral`, or `do_not_apply`.

## Low-rank operator SBV

Low-Rank Operator Supervised Bellman Validation is a post-hoc selector for many
Q candidates under one target policy. It learns a shared conditional operator
for the candidate next-value matrix instead of fitting one Bellman regression
per candidate.

Use trajectory-clean splits: `D_B_train`, `D_B_val`, and `D_score`.
