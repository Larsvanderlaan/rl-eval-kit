---
title: Discounted Occupancy Ratios Diagnostics
description: Reading ratio, source, tuning, and benchmark diagnostics.
---

## Ratio quality signals

These checks are user-facing because they determine whether the fitted weights
are plausible for OPE, weighted FQE, or downstream reporting.

| Signal | Meaning | What to review |
| --- | --- | --- |
| Source-correction path | Whether the fit used no source, factored state-source correction, or joint initial state-action correction | Whether the supplied initial states/actions have credible support |
| Effective sample size and weight variation | How concentrated or uniform the fitted weights are | High ESS plus near-zero variation under real policy shift |
| Clipping and tail summaries | Whether a few rows dominate the weighted estimate | Large clipped fraction or very large max weights |
| Source-ratio fit quality | Whether the initial-source nuisance fit looks stable | Large source losses, extreme source ratios, or fallback status |
| Fold-level OPE stability | Whether weighted value estimates agree across folds | Strong swings across folds or candidate families |

## Uniform weights and tail risk

Near-uniform weights are not automatically good. If behavior and target
policies are meaningfully different, a near-one ESS with near-zero weight CV can
mean the estimator learned a nearly constant ratio.

Review candidate complexity, regularization, fit quality, and fold-level ratio
variation before treating the run as successful.

## Exported fields for audits

The exported JSON keeps implementation-level names so reports can be traced back
to code. The most useful fields to surface in product summaries are:

| Field family | User-facing interpretation |
| --- | --- |
| `source_state_ratio_enabled`, `initial_joint_ratio_enabled` | Which source-correction path was used |
| `*_ess_fraction` | Effective sample size fraction for fitted weights |
| `*_clipped_fraction` | Fraction of mass affected by clipping |
| `source_state_ratio_loss`, `initial_joint_ratio_loss` | Nuisance source-ratio fit quality |

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
