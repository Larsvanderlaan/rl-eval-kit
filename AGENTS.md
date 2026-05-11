# Repository Agent Notes

## Occupancy-Ratio Code Quality Guidelines

For production work in `packages/occupancy-ratio`, follow the detailed package
guidance in `packages/occupancy-ratio/AGENTS.md`. In short: keep public
facades stable, put reusable logic in focused internal modules, validate array
shapes at API boundaries, keep optional dependencies lazy, document public
functions with NumPy-style docstrings, and add targeted tests for any change to
defaults, source correction, target construction, stabilization, tuning,
diagnostics, or serialization. Treat `_boosted_impl.py` and `_neural_impl.py`
as compatibility/orchestration shims, not as the default place for new logic.

## Occupancy-Ratio Initial-Source Correction

The occupancy-ratio package supports an initial-source correction in both the
boosted and neural fixed-point estimators. The preferred algorithmic source
object is the joint initial state-action ratio
`rho_initial(s) * pi(a | s) / (rho_ref(s) * pi0(a | s))`. When
`initial_states` and target-policy `initial_actions` are provided to the
high-level fitters, `initial_ratio_mode="auto"` should resolve to the joint
ratio path. When only `initial_states` are available, auto falls back to the
factored source term
`rho_initial(s) / rho_ref(s) * pi(a | s) / pi0(a | s)`. When
`initial_states` is omitted, behavior is backward-compatible and the source
ratio is exactly `1`.

Implementation details:

- Boosted code uses `SourceStateRatioConfig`; neural code uses
  `NeuralSourceStateRatioConfig`. These configs are used for both joint initial
  ratios and factored state-source ratios, despite the historical
  "source-state" name.
- Prefer fitting the joint initial ratio with behavior/reference state-action
  rows `(S, A)` as the denominator and target initial state-action rows
  `(initial_states, initial_actions)` as the numerator.
- Use the state-only source density ratio only as a fallback. In that path, fit
  `rho_initial / rho_ref` with the estimator reference states `S` as the
  denominator and `initial_states` as the numerator.
- `initial_weights`, when supplied, are normalized within the numerator block so
  they do not rescale the density-ratio objective.
- In the joint path, pass the fitted joint initial ratio directly as the source
  weight on each query state-action row.
- In the factored fallback path, multiply the source state ratio by the existing
  action ratio for every query state-action row:
  `source_state_ratio_query * action_ratio`.
- The boosted and neural target builders expose both source-weight paths:
  `w_source_query` for direct joint source weights and
  `source_state_ratio_query` for factored fallback source correction.
- Do not add a `-zeta^2 / 2` term to these iterative regressions. That term is
  part of the DualDICE saddle objective, not the fitted fixed-point estimator.

Benchmark impact and defaults:

- Occupancy benchmarks use `source_state_correction_mode="auto"` by default.
  Auto mode keeps correction off for controlled truth settings whose reference
  distribution is already the initial distribution (`discrete_chain`,
  `discrete_grid`, `linear_gaussian`, `nonlinear_monte_carlo`) and turns it on
  for Gym, logged, behavior-discounted, Minari, and similar datasets.
- Bare benchmark estimator names should remain safe off-the-shelf choices:
  `boosted_tree` expands to `boosted_tree_stable`, and `neural_network` expands
  to `neural_network_stable`.
- LSIF is the default nuisance density-ratio loss for boosted and neural
  estimators. Logistic nuisance variants are opt-in or auto-candidates, not the
  default.
- "Stable" must still mean a useful ratio estimator. Do not treat high ESS as a
  success metric by itself. If behavior and target policies are meaningfully
  different but the oracle/diagnostic ratios should be nonconstant, a near-1 ESS
  with near-zero weight CV is suspicious and should be investigated as collapse,
  over-regularization, or underfitting.
- Boosted-tree stable defaults should recover nonconstant ratios on tabular and
  other well-covered settings. Tree methods may degrade under genuine coverage
  gaps or extrapolation, but poor tabular ratio recovery is a bug or tuning
  failure until proven otherwise.
- Estimates may shift in Gym, nonlinear, Gaussian, logged-bandit, Minari, or any
  setting where `rho_initial != rho_ref`. This is intended and moves the
  iterative estimators closer to Google DualDICE's initial-state anchor.
- Benchmark diagnostics include `source_state_ratio_enabled`,
  `source_state_ratio_mean`, `source_state_ratio_max`,
  `source_state_ratio_ess_fraction`, `source_state_ratio_loss`,
  `source_state_ratio_density_ratio_loss`, and
  `source_state_ratio_clipped_fraction` for the factored fallback. Joint-source
  diagnostics use `initial_joint_ratio_enabled`, `initial_joint_ratio_mean`,
  `initial_joint_ratio_max`, `initial_joint_ratio_ess_fraction`,
  `initial_joint_ratio_loss`, `initial_joint_ratio_density_ratio_loss`, and
  `initial_joint_ratio_clipped_fraction`.

Boosted implementation invariant:

- In iterative boosted occupancy fitting, keep the LightGBM convention
  consistent: the public raw prediction is `w_init + booster.predict(...)`, so
  fixed-point labels are centered by `w_init` and the iterative occupancy stage
  must not combine absolute labels with LightGBM `init_score` plus `init_model`.
  That combination can learn large constant offsets that later normalize into
  near-uniform ratios. Any refactor touching labels, offsets, `init_score`, or
  `init_model` needs a tiny tabular noncollapse regression test.

## Occupancy-Ratio Product CV/AutoML Tuning

The product tuning suite lives in `packages/occupancy-ratio/occupancy_ratio/tuning.py`
and is exported from top-level `occupancy_ratio`. See
`packages/occupancy-ratio/AGENTS.md` for the detailed package policy.

Maintainer rules:

- Use `tune_occupancy_ratio_auto(...)` as the user-facing AutoML entrypoint and
  `tune_occupancy_ratio(...)` for configurable CV/search.
- Keep default AutoML neural, row-wise 3-fold CV, proxy-only selection,
  deterministic capped candidate expansion, and final refit on all data.
- Support `budget="fast"` and `budget="balanced"`; benchmark
  `--automl-tuning` accepts `off`, `fast`, and `balanced`, while `--tune-cv`
  defaults to balanced unless an explicit mode is supplied.
- Never select using oracle ratios, target-policy Monte Carlo values, or any
  benchmark truth. Truth is for reporting only.
- Always full-evaluate the stable baseline candidate and keep the final-refit
  stable fallback guardrail on by default.
- Score tuning candidates by proxy risk, OPE/reward stability, ratio quality,
  and runtime. ESS is only a diagnostic: penalize catastrophic low ESS, tail
  blowups, and clipping, but also penalize near-uniform collapse under
  meaningful behavior-target mismatch. Do not reward a candidate merely because
  its ESS is closer to 1.
- Keep tuning telemetry rich enough for product debugging:
  `candidate_id`, `family`, `budget_stage`, selected/promoted flags, score
  components, runtime, fold rows, and final refit diagnostics.
- Source tuning is active only when `initial_states` is supplied and must follow
  the initial-source correction semantics above: joint initial ratio when
  `initial_actions` are available, factored state-source fallback otherwise.

## Occupancy-Ratio Gym Simulation/Oracle-Tuning Notes

For Gym simulation, ablation, and oracle-tuning work, see the detailed
"Gym Simulation And Oracle-Tuning Notes" section in
`packages/occupancy-ratio/AGENTS.md`. Those parameter choices are useful for
experiment design, but they are not product defaults and should not be promoted
without broader controlled and realistic screens.

Short version:

- Use the common neural FORI grid with `tiny_underfit_4`,
  `small_underfit_8`, `medium_stable_32x32`, `large_stable_64x64`, and
  `big_loose_128x128` for Pendulum and HalfCheetah-style screens.
- For MountainCarContinuous oracle tuning, add longer expressive FORI
  candidates such as `mc_medium_long_64x64` and `mc_tight_long_64x64`; the
  common grid and low-rank ABE one-SE selector can under-correct there.
- For Hopper-style reward-sensitive screens, compare low-rank ABE one-SE with
  reward/OPE-focused Bellman-GMM selection and inspect both raw and constrained
  GMM rows, because hard near-uniform-collapse safety can be too aggressive.
- Never use target-policy Monte Carlo values or oracle ratios for product
  selection. They are acceptable only in explicitly labeled oracle-selection or
  upper-bound experiment tables.
