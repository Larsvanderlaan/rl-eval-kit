# FQE Package Notes

The FQE package is the value-estimation companion to `occupancy-ratio`. Keep it
production-oriented: stable defaults first, tuning with real-user proxy metrics,
and clear compatibility for existing fitters.

## Default Estimator Policy

- `BoostedFQEConfig.stable_defaults()` is the ordinary off-the-shelf default.
  It should remain conservative: Huber loss, early stopping, value-bound
  inference, final refit, regularized LightGBM settings, and deterministic
  single-thread execution.
- `NeuralFQEConfig.stable_defaults()` is opt-in for users who want neural FQE.
  It should keep Huber loss, target networks, Polyak updates, gradient clipping,
  input standardization, and CPU as the safe default.
- Do not promote a raw squared-loss or fragile high-capacity variant from a
  narrow benchmark. Stable boosted FQE should remain the product default unless
  broad controlled and realistic screens justify a change.

## User Weights

`sample_weight` is the official user row-weight interface. It must propagate to
boosted and neural training, held-out Bellman risk, tuning folds, final refits,
calibration diagnostics, and benchmark adapters. Do not rename it or introduce a
parallel `user_weights` API.

`initial_weights` are only for policy-value averaging over initial evaluation
states/actions. They do not replace transition-row `sample_weight`.

Stationary weighted FQE multiplies user `sample_weight` by estimated discounted
occupancy-ratio weights and normalizes the combined weights to mean one by
default. Keep the value-estimation discount `gamma` separate from
`gamma_ratio`; the latter approximates stationary weighting and must remain
strictly below one. The default `gamma_ratio` is `0.99`, based on the initial
Gym smoke sweep showing it is a better near-stationary default than `0.95`
without making boosted weights as variable as a nearly undiscounted fit.

The stationary harness currently defaults to the official Google DualDICE
backend for weighting. This is a temporary product choice from the May 2026 Gym
screens. DualDICE requires `initial_states`, target-policy `initial_actions`,
`occupancy-ratio[google-dualdice]`, and a Google Research checkout. Keep the
package-native FORI backend available with `ratio_backend="occupancy_ratio"` and
benchmark aliases such as `stationary_weighted_fori_fqe` and
`stationary_weighted_fori_neural_fqe`.

When the explicit FORI backend is used, pair ratio learners with the FQE family:
boosted FQE uses boosted occupancy-ratio weights, neural FQE uses neural
occupancy-ratio weights. If an occupancy fixed point degenerates to uniform
weights, keep the guarded action-ratio fallback and surface the chosen weight
source in diagnostics. Its default initial-source correction should stay
`initial_ratio_mode="factored"` so source correction is state-only unless a
caller deliberately asks for a joint state-action source ratio. Keep the source
fit fallback enabled so degenerate nuisance fits retry without initial-source
correction and surface `occupancy_fit_fallback_used` in diagnostics.

Do not undertrain neural stationary weights in benchmarks. If a benchmark needs
custom neural occupancy-ratio configs for speed, keep them strong enough to move
the action and state-action ratios, prefer the factored one-step path unless a
direct neural one-step ratio is explicitly being evaluated, and limit CPU torch
threads for deterministic local runs. Direct calls to the public harness should
continue to inherit the stronger `occupancy-ratio` defaults unless the caller
passes explicit configs.

The Google DualDICE stationary backend should call the official
`occupancy-ratio` wrapper (`fit_google_dualdice_occupancy_ratio`) rather than
duplicating Google Research integration inside FQE. Keep it optional and
preflighted; missing TensorFlow/Addons or missing Google Research source should
produce clear missing-dependency rows in benchmarks.

## Product Tuning

The product tuning harness lives in `fqe.tuning` and is exported from top-level
`fqe`.

Public entrypoints:

- `tune_fqe(...)`: configurable CV/search harness.
- `tune_fqe_auto(...)`: recommended AutoML entrypoint.
- Result/config dataclasses: `FQETuningConfig`, `FQESearchSpace`,
  `FQETuningResult`, `FQECandidateResult`, and `FQEFoldResult`.

Default product behavior:

- Boosted-only unless the caller explicitly includes neural with
  `families=("neural",)` or `families=("boosted", "neural")`.
- Row-wise 3-fold CV by default; caller-supplied `groups` must keep groups
  intact.
- Selection must be proxy-only. Never select using true Q-functions, target
  policy Monte Carlo values, oracle ratios, or benchmark truth.
- Final refit on all rows is the default, and the returned `model` should be
  the refit model for the selected candidate.
- No optimizer-service dependency such as Optuna or Ray for the product
  harness.

Budgets:

- `budget="fast"` is for interactive checks and CI smoke coverage.
- `budget="balanced"` is the recommended user-facing AutoML preset.

Telemetry should remain useful for product debugging: candidate id, family,
budget stage, selected/promoted flags, score components, runtime, fold rows,
errors, and final-refit diagnostics.

Compatibility wrappers such as `tune_fqe_cv` and `tune_fqe_neural_cv` must keep
their existing return shape unless a task explicitly asks for a breaking change.
