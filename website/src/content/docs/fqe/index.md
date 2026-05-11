---
title: FQE
description: Fitted Q evaluation, tuning, stationary weighting, calibration, and validation.
---

`fqe` estimates a fixed target policy's Q-function and initial-state policy
value from logged transitions, target-policy next actions, rewards, and
bootstrap masks. It returns fitted Q/value models, value estimates, and Bellman
diagnostics.

## What is estimated?

The main object is the target-policy Q-function:

```text
Q^pi(s, a) = E_pi[sum_t gamma^t R_t | S_0 = s, A_0 = a]
```

The fitted model can predict row-level Q values and estimate an initial-state
policy value by averaging `Q(initial_states, initial_actions)`.

## Install

```bash
python -m pip install -e "packages/fqe[neural,benchmark]"
```

Use the narrower package if you only need boosted FQE:

```bash
python -m pip install -e "packages/fqe"
```

## Minimal example

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

q_values = model.predict_q(states, actions)
policy_value = model.estimate_policy_value(initial_states, initial_actions)
```

## Data contract

| Field | Required | Shape intent |
| --- | --- | --- |
| `states`, `actions` | Yes | Logged transition rows |
| `next_states` | Yes | One next state per row |
| `next_actions` | Yes in Q-mode | One or many sampled evaluation-policy actions per next state |
| `rewards` | Yes | One reward per row |
| `gamma` | Yes | Value-estimation discount |
| `terminals` | Recommended | Terminal mask for Bellman targets |
| `sample_weight` | Optional | User row weights propagated through fitting and validation |
| `initial_states`, `initial_actions` | For value estimates and tuning | Evaluation-policy initial rows |

`next_actions` can be shape `(n, action_dim)` or `(n, n_action_samples,
action_dim)`. Multiple actions are averaged in the Bellman target.

## When to use it

- You have logged transitions and actions sampled from an evaluation policy.
- You need a direct-method OPE estimate or reusable Q-function.
- You want stable boosted defaults first, then neural FQE when the function
  class or workload calls for it.
- You need target-validation assisted selection, Bellman calibration, or
  post-hoc model selection among many Q candidates.

## Limitations

- Evaluation-policy actions are unsupported in the logged state space.
- Held-out Bellman risk is small only because the target distribution is poorly
  represented.
- Target-validation rollouts have large truncation tail mass.
- Neural FQE was undertrained or selected using validation signals that do not
  match the target-policy use case.

## Method surface

| Method | Entry point | Use case |
| --- | --- | --- |
| Boosted FQE | `fit_fqe_lgbm` | Stable default, tabular or structured arrays |
| Neural FQE | `fit_fqe_neural` | Larger continuous-control workloads |
| Value-only FVI | `fit_value_lgbm`, `fit_value_neural` | Bellman operator already expressed over states |
| Automatic tuning | `tune_fqe_auto` | Candidate search and final refit |
| Target validation | `tune_fqe_with_target_validation` | Independent target-policy rollouts or labels |
| Stationary weighting | `fit_stationary_weighted_fqe` | Reweighted Bellman regression under distribution shift |
| Bellman calibration | `fit_bellman_calibrator` | Post-hoc calibration diagnostics and correction |
| Low-rank SBV | `LowRankOperatorSBVValidator` | Efficient selection among many Q candidates |

## Papers

- [Fitted Q Evaluation Without Bellman Completeness via Stationary Weighting](../papers/)
- [Bellman Calibration for V-Learning in Offline Reinforcement Learning](../papers/)
- [Stationary Reweighting Yields Local Convergence of Soft Fitted Q-Iteration](../papers/)

## API links

- [Package README](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/fqe/README.md)
- [Top-level exports](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/fqe/fqe/__init__.py)
- [Low-rank SBV docs](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/fqe/docs/low_rank_operator_sbv.md)
