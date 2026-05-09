# Occupancy Ratio Package Notes

When working on the discounted occupancy-ratio estimators in this package,
consult `../../papers/occupancy_ratio/fori_iclr2026/main.tex`. The package-local `references/fitted_occupancy_ratio_iteration_for_offline_rl.tex` is retained as a snapshot, not the canonical source.

That paper gives the high-level algorithm and theory for fitted occupancy-ratio
iteration (FORI). In particular, it is relevant when changing:

- preliminary initial and one-step density-ratio estimation;
- the adjoint Bellman fixed-point updates in boosted or neural occupancy fits;
- stabilization choices such as clipping, normalization, damping, and
  projection;
- benchmark diagnostics that compare Bellman residuals, ratio quality, or
  finite-iteration behavior.

Use the paper as design context for implementation choices, but keep code
changes aligned with the existing public API and benchmark conventions unless a
task explicitly asks for a larger redesign.

## Default Estimator Policy

Off-the-shelf benchmark defaults should favor robust OPE behavior over winning a
single easy ratio-recovery cell.

- Bare `boosted_tree` expands to `boosted_tree_stable`.
- Bare `neural_network` expands to `neural_network_stable`.
- Both stable defaults use LSIF nuisance ratios unless a caller explicitly asks
  for logistic.
- Raw `squared` and raw `huber` presets are ablations, not recommendations.
- Do not change these defaults from a small benchmark. Require controlled and
  realistic screens with OPE, ratio quality, ESS, clipping, timeout, and
  nonfinite diagnostics before promoting a variant.
- Stable defaults are not allowed to mean "nearly uniform weights." High ESS is
  a diagnostic, not a target. Under meaningful policy shift, near-1 ESS with
  near-zero weight CV is suspicious unless the oracle ratio is also nearly
  constant.
- Boosted trees may struggle when coverage is genuinely poor because trees do
  not extrapolate well. On tabular and well-covered settings, however,
  `boosted_tree_stable` should recover useful nonconstant ratios and competitive
  OPE estimates. Treat tabular near-uniform collapse as a bug or default-tuning
  failure until a theory/debug audit proves otherwise.

`*_stable` means Huber occupancy regression with projection/normalization,
pseudo-outcome clipping, finite ratio caps, and fixed-point damping. The stable
path can lose a little accuracy on very easy low-tail cells, but it avoided the
large finite-iteration/high-gamma collapses seen with raw neural Huber and is
therefore the correct default.

## Boosted LightGBM Fixed-Point Invariants

The iterative boosted occupancy stage uses a centered prediction convention:

- public raw prediction is `w_init + booster.predict(...)`;
- fixed-point labels passed to LightGBM are centered by `w_init`;
- the iterative occupancy stage should not combine absolute labels with
  LightGBM `init_score` and `init_model`;
- offsets must be applied exactly once in predictions and diagnostics.

Mixing absolute labels, `init_score`, and `init_model` in this loop can make
later trees learn huge constant offsets that normalize back into nearly uniform
ratios. Any change to labels, offsets, `init_score`, `init_model`, or custom
occupancy objectives needs regression coverage on a tiny tabular nonconstant
ratio case.

Nuisance LSIF/logistic fits have their own offset and output-transform
conventions. Audit them separately before reusing an occupancy-stage offset
pattern there.

## Competitive Variants

Use these names intentionally:

- `*_calibrated`: stable-style occupancy with scalar nuisance moment
  calibration. Useful when first-stage nuisance normalization is slightly off.
- `*_stable_logistic_nuisance`: stable occupancy with logistic action and
  transition density-ratio nuisances. It can help some OPE cells, but current
  screens do not show it as uniformly better than LSIF; keep LSIF default.
- `*_transition_norm`: normalizes the transition cache. Treat as opt-in or
  CV/auto-selected; it is not the ordinary default.
- `*_auto`: compares a small set of stable candidates using internal diagnostics
  only. It must not peek at oracle ratios or target-policy Monte Carlo values.
- `neural_network_google_parity`: architecture/budget comparison against
  Google DualDICE, not the standard user default. It remains FORI, not the
  DualDICE saddle objective.

For high-stakes OPE, do not silently pick one estimator. Report at least the
stable boosted, stable neural, and Google DualDICE rows, then recommend a value
only when guardrails pass. The high-stakes profile should keep LSIF, adaptive
Huber (`huber_delta=None`, `huber_delta_scale=1.345`), damping `0.5`, ratio caps
around `50`, scalar calibration where configured, and longer neural/boosted
  budgets.

## Google DualDICE Wrapper

The optional official Google DualDICE backend is part of the public ecosystem
via `fit_google_dualdice_occupancy_ratio(...)`, `GoogleDualDICEConfig`, and
`preflight_google_dualdice(...)`. Keep its user-facing signature aligned with
`fit_discounted_occupancy_ratio(...)` where possible: states, actions,
next-states, target current actions, target next actions, initial
state-actions, gamma, and optional row weights. The wrapper should stay
dependency-light at import time: TensorFlow, TensorFlow Addons, and the external
Google Research checkout are loaded only when the preflight or fitter is called.

Google DualDICE directly estimates `zeta(s, a)`. It does not fit separate
action or transition nuisances, so its model may expose `predict_action_ratio`
as ones for API compatibility, but diagnostics and docs must make that clear.
Do not use Google DualDICE as truth for tuning or selection; it is a deployable
comparator and optional backend.

## Product CV And AutoML Tuning Suite

The product tuning harness lives in `occupancy_ratio.tuning` and is exported
from top-level `occupancy_ratio`. Treat it as the user-facing tuning API, not as
benchmark-only glue.

Public entrypoints:

- `tune_occupancy_ratio(...)`: configurable cross-validation/search harness.
- `tune_occupancy_ratio_auto(...)`: recommended AutoML entrypoint.
- Result/config dataclasses: `OccupancyTuningConfig`,
  `OccupancySearchSpace`, `OccupancyTuningResult`, `CandidateResult`, and
  `FoldResult`.

Default product behavior should remain conservative and off-the-shelf:

- Boosted-only unless the caller explicitly includes neural via
  `families=("boosted", "neural")` or `families=("neural",)`.
- Row-wise 3-fold CV by default; caller-supplied `groups` switches to grouped
  folds and must keep groups intact.
- Proxy-only scoring. Never use oracle ratios, target-policy Monte Carlo
  values, or benchmark truth for selection, even when those fields are present.
- Final refit on all data is the default, and the returned `model` should be
  the refit model for the selected candidate.
- No new optimizer dependency such as Optuna/Ray for this harness.

Budgets:

- `budget="fast"` caps the default search at 8 candidates, promotes up to 2,
  and uses a smaller screening fraction. Use it for interactive checks and CI
  smoke coverage.
- `budget="balanced"` caps the default search at 16 candidates, promotes up to
  4, and is the recommended user-facing AutoML preset.

Scoring is a weighted rank over fold-level proxy metrics: held-out fixed-point
risk, optional reward-weighted OPE stability, weight quality, and runtime. The
current product default intentionally gives substantial weight to safety/weight
quality so AutoML does not win by choosing brittle high-tail ratios. Safety does
not mean maximizing ESS: penalize catastrophic low ESS, tail blowups, and
clipping, but also penalize near-uniform collapse when behavior-target mismatch
is meaningful. Do not rank a candidate higher merely because its ESS is closer
to 1. Keep telemetry complete: candidate rows, fold rows, selected/promoted
flags, runtime, score components, final refit diagnostics, and errors.

The default candidate list is curated, deterministic, and ordered with the
stable baseline first. Always full-evaluate that baseline candidate during
promotion. During final refit, the stable-baseline fallback should remain on by
default: if the tuned winner does not show enough proxy safety/runtime benefit,
fall back to the baseline instead of shipping a fragile "winner." This is a
product guardrail, not a benchmark hack.

Google DualDICE must not enter product AutoML selection by default. Keep it as
an explicit neural-family opt-in via `OccupancyTuningConfig(include_google_dualdice=True)`
or an explicit custom candidate with `backend={"name": "google_dualdice"}`. It
requires joint initial state-action rows (`initial_states` and
`initial_actions`) and uses `OccupancySearchSpace.google_dualdice` for its
`GoogleDualDICEConfig`.

Benchmark `google_parity` should not hide nuisance-budget coupling. Keep the
neural action, source/initial, transition, and direct one-step nuisance stages
independently configurable. Backward-compatible defaults may let source inherit
the action budget and direct one-step inherit the transition budget, but code
should not assume these stages have the same complexity.

The current strong neural Gym default is intentionally 64x64 SiLU rather than a
wider Google-parity MLP: spend complexity on per-stage ratio and adjoint fits
first. Benchmark defaults use 800 action/source nuisance steps, 1000
transition/direct-one-step nuisance steps, and 128 neural direct-adjoint steps.
The benchmark config also exposes per-stage hidden dimensions; keep the default
shared unless a sweep shows an asymmetric architecture improves Gym OPE and tail
diagnostics.
Revisit these only with Gym OPE sweeps against `google_dualdice_neural` and
tail diagnostics, not just validation loss.

Source tuning is included only when `initial_states` is supplied. It must follow
the initial-source correction semantics: prefer the joint initial state-action
ratio when `initial_actions` are available; use the factored state-source
fallback only when joint initial actions are unavailable. `initial_weights` are
normalized inside the numerator block.

Benchmark integration:

- `--tune-cv` should use the product harness.
- `--tune-cv` maps to `--automl-tuning balanced` unless an explicit mode is
  supplied.
- `--automl-tuning` accepts `off`, `fast`, or `balanced`.
- `tuning_results.csv` should preserve old compatibility rows where relevant
  and include product telemetry columns such as `candidate_id`, `family`,
  `budget_stage`, score components, `runtime_sec`, `selected`, and `promoted`.

Compatibility wrappers such as `tune_discounted_occupancy_ratio_cv` and
`tune_discounted_occupancy_ratio_neural_cv` should keep working. They may
delegate to the product harness or remain legacy wrappers, but do not remove or
break them.

## Source And Direct Ratios

Use direct one-step ratios when `target_next_actions` are available. That avoids
unnecessary transition-ratio Monte Carlo variance and better matches the
one-step ratio needed by the adjoint update.

Use joint initial/source ratios when `initial_actions` are available. The
algorithmic source object is the joint ratio
`rho_initial(s) * pi(a | s) / (rho_ref(s) * pi0(a | s))`, so
`initial_ratio_mode="auto"` should optimize that object directly. The
state-only source ratio multiplied by the action ratio is a fallback for
datasets that have `initial_states` but no target-policy initial actions.

Use `source_state_correction_mode="auto"` in benchmarks:

- off for controlled truth settings where the reference distribution is already
  the initial distribution;
- on for Gym, logged, behavior-discounted, Minari, and similar datasets where
  the fixed-point source should be anchored to the evaluation initial-state
  law.

When source correction is on and `initial_actions` are available, the source
term uses the direct joint initial ratio on query state-action rows. Otherwise,
the fallback source term is
`(1 - gamma) * rho_initial(s) / rho_ref(s) * pi(a | s) / pi0(a | s)`.

## Diagnostics Subtleties

`clipping_fraction` should mean actual safety clipping of invalid, negative, or
cap-violating weights. Do not treat normalization/projection changes as
clipping; use `postprocessing_changed_fraction` for that. High-stakes guardrails
should combine final projection clipping with safety clipping, not fail every
stable estimator merely because normalization touched every weight.

Prefer ratio L1/TV/log-RMSE diagnostics on controlled settings and OPE absolute
error or MC-standard-error-normalized error on realistic settings. Google
DualDICE is a strong comparator, especially for OPE, but it is not ground truth
when its ratio diagnostics or guardrails are poor.

When controlled truth exists, always inspect ratio quality alongside OPE. A
method with excellent ESS and poor ratio L1/log-RMSE is not a good default, even
if its OPE happens to look acceptable on one reward function.
