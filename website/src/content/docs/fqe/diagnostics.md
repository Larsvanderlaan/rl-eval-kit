---
title: FQE Diagnostics
description: How to read FQE tuning, validation, calibration, and stationary-weighting diagnostics.
---

## Core signals

| Signal | What it tells you | Suspicious pattern |
| --- | --- | --- |
| Held-out weighted Bellman risk | Whether fitted Bellman consistency transfers to held-out rows | Low risk only on behavior-heavy regions |
| Policy-value stability | Whether candidate value estimates agree across folds or seeds | Strong value swings under small config changes |
| Calibration residuals | Whether Bellman target means match prediction bins | Local residual improvement without value improvement |
| Target-validation tail mass | How much discounted rollout remains unobserved | Large tail mass with confident selection |
| Stationary weight source | Which ratio fallback produced FQE row weights | Uniform-looking weights under strong policy shift |

## Calibration example

```python
from fqe import (
    bellman_calibration_diagnostics,
    fit_bellman_calibrator,
    plot_bellman_calibration_diagnostics,
)

pred = model.predict_q(states, actions)
next_pred = model.predict_q(next_states, next_actions)

calibrator = fit_bellman_calibrator(
    pred,
    next_pred,
    rewards,
    gamma=0.99,
    terminals=dones,
)

diagnostics = bellman_calibration_diagnostics(
    pred,
    next_pred,
    rewards,
    gamma=0.99,
    terminals=dones,
    calibrator=calibrator,
)
plot_bellman_calibration_diagnostics(diagnostics, path="bellman_calibration.png")
```

## Guardrails

- `sample_weight` is the user row-weight interface and should propagate through
  fitting, validation, tuning folds, final refits, and calibration helpers.
- `initial_weights` are only for policy-value averaging over initial rows.
- Stationary weighted FQE separates `gamma` for value estimation from
  `gamma_ratio` for occupancy weighting.
- Tuning rows should include candidate id, family, budget stage,
  selected/promoted flags, score components, runtime, fold rows, errors, and
  final-refit diagnostics.
