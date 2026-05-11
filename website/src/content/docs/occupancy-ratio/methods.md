---
title: Discounted Occupancy Ratios Methods
description: FORI, source correction, tuning, target validation, and Google DualDICE.
---

## FORI

Fitted Occupancy-Ratio Iteration estimates two preliminary density ratios and
then solves an adjoint Bellman equation through supervised one-step adjoint
regressions. The method turns discounted occupancy-ratio estimation into a
sequence of fitted prediction problems plus explicit stabilization.

## Source correction

When `initial_states` and target-policy `initial_actions` are supplied,
`initial_ratio_mode="auto"` resolves to the joint initial state-action ratio:

```text
rho_initial(s) * pi(a | s) / [rho_ref(s) * pi0(a | s)]
```

When only `initial_states` are available, auto falls back to the factored source
term:

```text
rho_initial(s) / rho_ref(s) * pi(a | s) / pi0(a | s)
```

When `initial_states` is omitted, the source ratio remains backward-compatible
and exactly equal to one.

## Automatic tuning

The tuning suite scores candidates by proxy risk, OPE/reward stability, ratio
quality, and runtime. It penalizes catastrophic low ESS, tail blowups, clipping,
and near-uniform collapse under meaningful behavior-target mismatch.

It does not reward a candidate merely because ESS is closer to one.

## Google DualDICE integration

Google DualDICE is an optional dependency path:

- TensorFlow and TensorFlow Addons load lazily.
- The Google Research checkout is preflighted.
- `fit_google_dualdice_occupancy_ratio(...)` aligns with the public FORI
  signature where possible.
- DualDICE is an optional external comparator or backend.

## Target-validation modes

Use target validation only when independent target-policy rollouts or simulator
labels are available. Scalar OPE validation can compare a value estimate, but it
does not validate the full ratio function.
