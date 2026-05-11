---
title: genPQR Methods
description: Reward identification, normalization, DeepGenPQR, and DeepPQR anchors.
---

## Reward identification

Inverse RL rewards are generally only partially identified. GenPQR makes the
normalization explicit: the reported reward is the normalized representation
consistent with the estimated policy, estimated Q-function, and chosen
normalization policy or anchor.

The normalization policy is therefore part of the estimand, not a configuration
detail.

## Policy-to-Q-to-reward

The core workflow is modular:

1. Estimate or supply a behavior policy.
2. Estimate or supply a Q-function under that policy and normalization.
3. Recover the normalized reward through the Bellman identity.
4. Report diagnostics and serialize the fitted result.

Policy and Q estimators can be named lazy adapters or user-supplied protocol
objects.

## DeepGenPQR

DeepGenPQR is the neural workflow. The default path runs Deep AIRL through the
lazy imitation adapter and then fits pooled neural FQE over logged action rows.

Use pooled FQE when anchor actions are rare, when an all-action Q model is
useful, or when the reward surface is smooth enough for shared statistical
strength.

## DeepPQR anchor mode

Use anchor DeepPQR when your estimand uses an anchor-action normalization or
state-only anchor Q parameterization, and the logged data contain enough
positive-weight anchor rows across the states where rewards will be queried.

```python
from genpqr import DeepGenPQRConfig, fit_deep_genpqr

config = DeepGenPQRConfig.from_preset(
    "deepgenpqr_airl_anchor_balanced",
    anchor_action=0,
    min_anchor_count=25,
)
result = fit_deep_genpqr(dataset=dataset, gamma=0.99, env=env, config=config)
```

Keep `anchor_fallback="error"` for analysis runs. Use
`anchor_fallback="pooled_fqe"` only when you want the run to record a warning
and refit pooled FQE when anchor support is too sparse for the configured
anchor workflow.
