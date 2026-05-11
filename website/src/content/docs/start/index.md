---
title: Start
description: Choose and install the right RLEvalKit package.
---

RLEvalKit is a Python ecosystem for offline reinforcement-learning evaluation
and normalized reward estimation. Start with the relevant estimand, then add
optional extras only when a workflow needs them.

## Install

From the repository root:

```bash
python -m pip install -e "packages/fqe[neural,benchmark]"
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
python -m pip install -e "packages/genpqr[fqe,torch]"
```

The core packages keep optional dependencies lazy. Importing `genpqr` does not
load Torch, Gymnasium, Stable-Baselines3, imitation, d3rlpy, or SCOPE-RL unless
you select an adapter that needs them.

## Which package should I use?

| Goal | Package | First API | Output |
| --- | --- | --- | --- |
| Estimate a fixed target policy's Q-function or policy value | `fqe` | `fit_fqe_lgbm`, `tune_fqe_auto` | Q model, value estimate, Bellman diagnostics |
| Estimate discounted target-to-reference occupancy ratios | `occupancy_ratio` | `fit_discounted_occupancy_ratio`, `tune_occupancy_ratio_auto` | State-action weights and ratio diagnostics |
| Estimate a normalized reward representation from behavior | `genpqr` | `fit_genpqr`, `fit_deep_genpqr` | Reward function and IRL diagnostics |
| Reweight FQE toward target-policy stationarity | `fqe` plus `occupancy-ratio` or Google DualDICE | `fit_stationary_weighted_fqe` | Weighted FQE model and weight diagnostics |

## Minimal examples

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
    config=BoostedFQEConfig.stable_defaults(seed=123),
)
```

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

```python
from genpqr import GenPQRConfig, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_boosted"),
)
```

## Data contracts

Most workflows use NumPy-like arrays with explicit row alignment:

| Field | Meaning |
| --- | --- |
| `states`, `actions` | Logged behavior-policy state-action rows |
| `next_states` | One next state per transition row |
| `target_actions` | Current target-policy actions for ratio estimation |
| `next_actions` or `target_next_actions` | Evaluation-policy actions at `next_states` |
| `initial_states`, `initial_actions` | Initial target-policy anchor rows for policy values or source correction |
| `terminals`, `timeouts` | Episode termination and continuation conventions |
| `sample_weight` | User row weights for fitting, validation, and refits |

Validate shape errors at the package boundary rather than debugging silent
broadcasting after a long run.

## Reading diagnostics

RLEvalKit reports diagnostic summaries that help users decide whether an
estimate is usable for their data. The public report should answer three
questions: why was this candidate selected, what data support does it rely on,
and what should be reviewed before using the estimate downstream?

- ESS is a diagnostic, not a target. Under meaningful policy shift, near-uniform
  weights can indicate a nearly constant ratio rather than a successful fit.
- Target-validation rollouts are finite samples. Tail-mass diagnostics report
  how much discount mass remains after the observed prefix.
- Optional integrations such as Google DualDICE, Torch, TensorFlow, Gymnasium,
  MuJoCo, imitation, d3rlpy, and SCOPE-RL should report adapter status clearly.

## Next steps

- Read the [FQE guide](../fqe/) for policy-value and Q-function evaluation.
- Read the [discounted occupancy ratios guide](../occupancy-ratio/) for density
  ratios and FORI.
- Read the [genPQR guide](../genpqr/) for inverse RL and normalized reward estimation.
- Open the [paper library](../papers/) when you need assumptions, theorem
  statements, or citation links.
