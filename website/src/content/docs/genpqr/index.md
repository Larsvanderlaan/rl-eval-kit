---
title: genPQR
description: Modular generalized policy-to-Q-to-reward tools for inverse reinforcement learning.
---

`genpqr` estimates a normalized reward representation from logged behavior by
combining a fitted behavior policy, Q evaluation, and an explicit normalization
policy or anchor.

## What is estimated?

GenPQR estimates rewards under a maximum-entropy or Gumbel-shock style
behavioral model. A reward is not identified without a normalization policy or
anchor constraint, so the normalization choice is part of the estimand.

```text
Given a fitted behavior policy pi_hat, normalization policy mu, and anchor g(s),
genPQR fits Q_u for u(s,a)=log pi_hat(a|s)-g(s), then reports

r_{mu,g}(s,a)=Q_u(s,a)-E_{A~mu(.|s)}Q_u(s,A)+g(s).
```

This does not identify a unique environment reward without identifying
restrictions.

## Install

Core import only requires NumPy:

```bash
python -m pip install -e "packages/genpqr"
```

Install optional learners only when needed:

```bash
python -m pip install -e "packages/genpqr[fqe,torch]"
python -m pip install -e "packages/genpqr[imitation]"
python -m pip install -e "packages/genpqr[d3rlpy,scope-rl]"
```

## Minimal example

```python
from genpqr import GenPQRConfig, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_boosted"),
)

rewards = result.reward_function.predict_reward(states, actions)
```

## Data contract

| Object | Role |
| --- | --- |
| `TransitionDataset` | Validated row-wise workflow |
| `EpisodeDataset` | Trajectory-preserving workflow |
| `ActionSpaceSpec` | Explicit discrete or continuous action contract |
| `normalization_policy` | Defines the reward normalization target |
| Policy estimator | Supplies action probabilities or log densities |
| Q estimator | Supplies continuation values for reward estimation |
| `GenPQRResult` | Reward function, typed diagnostics, serialization |

## When to use it

- You have demonstrations or logged behavior and want a reward explanation.
- You need to compare BC/FQE, AIRL/FQE, GAIL/FQE, SCOPE-RL diagnostics, or
  DeepPQR anchor workflows behind one public interface.
- You want custom policy and Q estimators through public protocols.

## Diagnostics and limitations

- The chosen reward normalization is not meaningful for the domain.
- Anchor-action support is weak but the workflow requires anchor DeepPQR.
- Continuous-action normalization uses too few Monte Carlo samples.
- Behavior-policy log probabilities or densities are unreliable for observed,
  queried, normalization, or anchor actions.
- Optional adapters should fail during preflight or raise missing-dependency
  errors. Treat any configured fallback as a separate method choice.

## Method surface

| Workflow | Entry point | Use case |
| --- | --- | --- |
| Lightweight BC + FQE | `fit_genpqr` with `behavior_cloning_native` and `fqe_boosted` | Dependency-light first pass |
| AIRL/GAIL + FQE | `fit_genpqr` with imitation adapters | Adversarial imitation workflows |
| DeepGenPQR | `fit_deep_genpqr` | Neural workflow with AIRL and action-aware neural FQE |
| DeepPQR anchor | `DeepPQRAnchorQEstimator` or `q_mode="anchor_deeppqr"` | Anchor-action reward identification |
| Cross-fitting | `fit_genpqr_crossfit` | Out-of-fold reward predictions |
| Serialization | `result.save`, `load_genpqr_result` | Portable fitted artifacts |

## Papers

- [Modular Inverse Reinforcement Learning via Policy Estimation and Q-Evaluation](../papers/)
- [Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models](../papers/)

## API links

- [Package README](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/genpqr/README.md)
- [Docs source](https://github.com/Larsvanderlaan/rl-eval-kit/tree/main/packages/genpqr/docs)
- [Example gallery](https://github.com/Larsvanderlaan/rl-eval-kit/tree/main/packages/genpqr/examples)
