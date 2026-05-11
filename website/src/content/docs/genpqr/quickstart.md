---
title: genPQR Quickstart
description: First workflows for behavior cloning, AIRL, DeepGenPQR, and continuous actions.
---

## Dependency-light BC + FQE

```python
from genpqr import GenPQRConfig, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_boosted"),
)
```

## Default AIRL + neural FQE

AIRL requires an environment because adversarial imitation training needs
generator rollouts. If `env` or the optional imitation stack is unavailable,
the default fails with a clear error instead of silently switching algorithms.

```python
from genpqr import ActionSpaceSpec, DiscreteNormalizationPolicy, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.99,
    env=env,
    action_space=ActionSpaceSpec.discrete(n_actions),
    normalization_policy=DiscreteNormalizationPolicy.uniform(n_actions),
)
```

## DeepGenPQR

```python
from genpqr import DeepGenPQRConfig, fit_deep_genpqr

result = fit_deep_genpqr(
    dataset=dataset,
    gamma=0.99,
    env=env,
    config=DeepGenPQRConfig.from_preset("deepgenpqr_airl_fqe_balanced"),
)
```

For finite actions, DeepGenPQR uses action-aware neural FQE. Inspect per-action
coverage and reward diagnostics before relying on results from sparse-action
regions.

## Continuous actions

```python
from genpqr import ActionSpaceSpec, ContinuousNormalizationPolicy, GenPQRConfig, fit_genpqr

mu = ContinuousNormalizationPolicy(
    action_dim=action_dim,
    sampler=lambda states, rng, n: rng.normal(size=(states.shape[0], n, action_dim)),
)

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    action_space=ActionSpaceSpec.continuous(action_dim),
    normalization_policy=mu,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_neural"),
)
```
