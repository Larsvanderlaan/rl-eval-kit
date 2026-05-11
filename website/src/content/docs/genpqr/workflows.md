---
title: genPQR Workflows
description: AIRL, GAIL, minimax diagnostics, cross-fitting, and examples.
---

## AIRL and GAIL

```python
from genpqr import GenPQRConfig, fit_genpqr

airl_result = fit_genpqr(
    states=states,
    actions=actions,
    next_states=next_states,
    terminals=dones,
    gamma=0.99,
    env=env,
    config=GenPQRConfig(
        policy="imitation_airl",
        q="fqe_neural",
        policy_config={"total_timesteps": 100_000, "demo_batch_size": 512},
    ),
)
```

Switch to GAIL by changing `policy="imitation_gail"`.

## SCOPE-RL minimax diagnostics

The dataset-bound SCOPE-RL adapter is for fitted-row diagnostics only. SCOPE-RL's
public OPE input path returns prediction arrays for the fitted rows, not a
reusable action-aware Q-function.

For reusable out-of-sample reward prediction, use `ReusableScopeRLQEstimator`
with a model that exposes `fit(...)` and `predict_q(...)`.

```python
from genpqr import GenPQRConfig, ReusableScopeRLQEstimator, fit_genpqr

scope_q = ReusableScopeRLQEstimator(model_or_factory=scope_model_factory)

result = fit_genpqr(
    dataset=dataset,
    gamma=0.99,
    config=GenPQRConfig(policy="behavior_cloning_native", q=scope_q),
)
```

## Cross-fitting

```python
from genpqr import DiscreteNormalizationPolicy, GenPQRConfig, fit_genpqr_crossfit

crossfit = fit_genpqr_crossfit(
    dataset=dataset,
    gamma=0.95,
    n_folds=5,
    normalization_policy=DiscreteNormalizationPolicy.uniform(n_actions),
    config=GenPQRConfig.from_preset("deeppqr_linear"),
)
```

Use cross-fitting when you need out-of-fold reward predictions or fold-level
instability diagnostics.
