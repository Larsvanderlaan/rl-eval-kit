---
title: FQE Quickstart
description: First FQE workflows for boosted, neural, automatic tuning, and policy samplers.
---

## Boosted Q-FQE

Start with boosted stable defaults unless you already know you need neural FQE.

```python
from fqe import BoostedFQEConfig, fit_fqe_lgbm

model = fit_fqe_lgbm(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    sample_weight=row_weights,
    config=BoostedFQEConfig.stable_defaults(seed=123),
)
```

## Automatic tuning

Use automatic tuning when you want the package to compare a capped candidate
set and refit the selected configuration on the full dataset. The report keeps
the candidate, fold, runtime, and diagnostic rows used for the decision.

```python
from fqe import tune_fqe_auto

tuned = tune_fqe_auto(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    sample_weight=row_weights,
    initial_states=initial_states,
    initial_actions=initial_actions,
    budget="balanced",
)

model = tuned.model
candidate_rows = tuned.candidate_rows()
fold_rows = tuned.fold_rows()
```

## Neural FQE

Install the neural extra and opt in when you need a PyTorch function class.

```python
from fqe import NeuralFQEConfig, fit_fqe_neural

model = fit_fqe_neural(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    config=NeuralFQEConfig.stable_defaults(
        hidden_dims=(256, 256),
        gradient_steps_per_iteration=20,
        device="cpu",
    ),
)
```

## Policy sampler wrapper

Use `fit_fqe_from_policy` when you want the fitter to request target-policy
samples instead of precomputing `next_actions`.

```python
from fqe import fit_fqe_from_policy

def sample_next_actions(next_states, rng, n_samples):
    return policy.sample(next_states, rng=rng, n_samples=n_samples)

model = fit_fqe_from_policy(
    states=states,
    actions=actions,
    next_states=next_states,
    rewards=rewards,
    gamma=0.99,
    next_action_sampler=sample_next_actions,
    n_next_action_samples=8,
)
```

Precomputed `next_actions` remain the deterministic low-level path.
