---
title: Discounted Occupancy Ratios
description: Discounted occupancy-ratio estimation with FORI, diagnostics, tuning, and benchmarks.
---

The `occupancy-ratio` package estimates discounted occupancy ratios from logged
reference transitions and target-policy action samples. The fitted model outputs
state-action weights for target-policy reweighting, plus diagnostics for
support, tails, clipping, ESS, and source correction. The importable package is
`occupancy_ratio`.

## What is estimated?

The main fitted model predicts:

```text
rho_pi,gamma(s) * pi(a | s) / [rho_ref(s) * pi0(a | s)]
```

This ratio reweights reference rows toward the discounted state-action
distribution induced by the target policy.

Notation:

| Symbol | Meaning |
| --- | --- |
| `rho_pi,gamma(s)` | Target policy's normalized discounted state occupancy |
| `rho_ref(s)` | Reference or behavior state distribution represented by the rows |
| `pi(a | s)` | Target policy action density or probability |
| `pi0(a | s)` | Behavior/reference policy action density or probability |
| Denominator rows | The logged state-action rows used as the reference distribution |

## Install

```bash
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
```

Use docs extras when building the package-local MkDocs reference:

```bash
python -m pip install -e "packages/occupancy-ratio[docs]"
```

## Minimal example

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

## Data contract

| Field | Required | Shape intent |
| --- | --- | --- |
| `states`, `actions` | Yes | Behavior/reference rows |
| `next_states` | Yes | One next state per reference row |
| `target_actions` | Yes | Target-policy current actions for reference states |
| `target_next_actions` | For target validation and some workflows | Target-policy actions at next states |
| `initial_states` | For source correction | Initial target-state numerator rows |
| `initial_actions` | For joint source correction | Target-policy initial actions |
| `rewards` | For OPE-aware tuning | Reward proxy and validation diagnostics |
| `sample_weight` | Optional | User row weights |

## Source correction

| Available data | Source path | Operational meaning |
| --- | --- | --- |
| No `initial_states` | Source ratio is 1 | Backward-compatible fit against the reference rows |
| `initial_states` only | Factored state-source correction | Fit `rho_initial / rho_ref`, then multiply by the target-action ratio |
| `initial_states` and `initial_actions` | Joint initial state-action correction | Fit the initial state-action source directly using target-policy initial actions |

## When to use it

- You need discounted density ratios for OPE, weighted FQE, or diagnostics.
- You want a fitted-regression alternative to coupled minimax or DICE-style
  saddle optimization.
- You need automatic tuning and detailed diagnostics.
- You want Google DualDICE as an optional external comparator or backend.

## Limitations

- ESS is near one and coefficient of variation is near zero under meaningful
  behavior-target mismatch.
- Clipping or tail diagnostics show that a few rows dominate the estimate.
- Source correction is active but initial-state or initial-action support is
  poor.
- Optional Google DualDICE dependencies are missing and the run does not report
  a clear skip or error.

## Method surface

| Method | Entry point | Use case |
| --- | --- | --- |
| Boosted FORI | `fit_discounted_occupancy_ratio` | Stable single fit with LightGBM |
| Neural FORI | `fit_discounted_occupancy_ratio_neural` | Larger continuous-control workloads |
| Automatic tuning | `tune_occupancy_ratio_auto` | Deterministic candidate search and final refit |
| Target validation | `tune_occupancy_ratio_with_target_validation` | Independent target-policy moment or scalar validation |
| Google DualDICE | `fit_google_dualdice_occupancy_ratio` | Optional external comparator/backend |
| Benchmarks | `occupancy-ratio-benchmark` | Controlled and realistic OPE screens |

## Papers

- [Fitted Occupancy-Ratio Iteration for Offline Reinforcement Learning](../papers/)
- [Fitted Q Evaluation Without Bellman Completeness via Stationary Weighting](../papers/)

## API links

- [Package README](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/occupancy-ratio/README.md)
- [Package docs source](https://github.com/Larsvanderlaan/rl-eval-kit/tree/main/packages/occupancy-ratio/docs)
- [Top-level exports](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/occupancy-ratio/occupancy_ratio/__init__.py)
