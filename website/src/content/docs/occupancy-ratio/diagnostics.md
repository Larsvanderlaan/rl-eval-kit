---
title: Discounted Occupancy Ratios Diagnostics
description: Reading ratio, source, tuning, and benchmark diagnostics.
---

## Ratio quality signals

| Diagnostic | Meaning | Red flag |
| --- | --- | --- |
| `source_state_ratio_enabled` | Factored source correction path used | Enabled with poor initial-state support |
| `initial_joint_ratio_enabled` | Joint initial ratio path used | Enabled without credible initial actions |
| `*_ess_fraction` | Effective sample size fraction | High ESS plus near-zero CV under real policy shift |
| `*_clipped_fraction` | How much ratio mass was clipped | Large clipping fraction or unstable tails |
| `source_state_ratio_loss` | Source ratio nuisance fit loss | Degenerate nuisance fit or fallback |
| `initial_joint_ratio_loss` | Joint initial ratio nuisance fit loss | Poor numerator/denominator separation |

## Collapse and tail risk

Near-uniform weights are not automatically good. If behavior and target
policies are meaningfully different, a near-one ESS with near-zero weight CV can
mean the estimator collapsed toward a constant ratio.

Investigate collapse, over-regularization, underfitting, or tabular tuning
failure before calling the run successful.

## What to check next

| Pattern | What to inspect |
| --- | --- |
| High ESS and near-zero CV under policy shift | Candidate complexity, regularization, and fold-level ratio variation |
| Large clipping fraction | Tail rows, support mismatch, and stabilized objective settings |
| Large source-ratio loss or max | Initial-state support and whether joint source correction is appropriate |
| Weak support warnings | State-action coverage for target-policy action samples |
| Unstable OPE values across folds | Ratio tails, reward scale, and target-validation data quality |

## Optional dependency diagnostics

Google DualDICE, Torch, TensorFlow, Gymnasium, MuJoCo, Minari, OpenML, and
plotting libraries are optional. Missing extras should appear as structured
skip or error rows in smoke runs, not as unexplained crashes.
