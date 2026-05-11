---
title: FQE Diagnostics
description: How to read FQE tuning, validation, calibration, and stationary-weighting diagnostics.
---

## Core signals

These are user-facing checks: they help decide whether a fitted Q model is
ready to use, not just whether a training run completed.

| Signal | What it tells you | What to review |
| --- | --- | --- |
| Held-out weighted Bellman risk | Whether Bellman consistency transfers to held-out rows | Whether the held-out rows cover target-policy regions |
| Policy-value stability | Whether value estimates agree across folds or seeds | Strong value swings under small config changes |
| Calibration residuals | Whether Bellman target means match prediction bins | Whether calibration changes the value estimate in a useful direction |
| Target-validation tail mass | How much discounted rollout remains unobserved | Large tail mass with otherwise confident selection |
| Stationary weight source | Which ratio backend produced FQE row weights | Whether weights are plausible under the policy shift |

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

## What belongs in a report

- `sample_weight` is the user row-weight interface and should propagate through
  fitting, validation, tuning folds, final refits, and calibration helpers.
- `initial_weights` are only for policy-value averaging over initial rows.
- Stationary weighted FQE separates `gamma` for value estimation from
  `gamma_ratio` for occupancy weighting.
- Tuning reports should include the selected method family, score components,
  runtime, fold-level scores, user-visible errors, and final-refit diagnostics.
