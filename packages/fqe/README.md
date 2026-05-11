# FQE

Production-oriented LightGBM fitted Q evaluation and fitted value iteration
tools for offline RL evaluation.

The importable package is `fqe`; install it from this directory:

```bash
python -m pip install -e "packages/fqe[neural,benchmark]"
```

## Q-FQE With Precomputed Target Actions

```python
from fqe import BoostedFQEConfig, fit_fqe_lgbm

model = fit_fqe_lgbm(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    sample_weight=row_weights,
    config=BoostedFQEConfig.stable_defaults(seed=123),
)

q_values = model.predict_q(states, actions)
policy_value = model.estimate_policy_value(initial_states, initial_actions)
```

`next_actions` may be either one action per transition with shape
`(n, action_dim)` or multiple sampled evaluation-policy actions with shape
`(n, n_action_samples, action_dim)`. Multiple actions are averaged in the
Bellman target.

`sample_weight` is the user row-weight interface. It is used in the regression
objective, validation Bellman risk, tuning folds, final refits, and Bellman
calibration helpers. Omit it for uniform weighting.

## Value-Only FVI

```python
from fqe import fit_value_lgbm

value_model = fit_value_lgbm(
    states=states,
    next_states=next_states,
    rewards=rewards,
    gamma=0.95,
    terminals=dones,
)

values = value_model.predict_value(states)
```

Use value mode when the Bellman operator is already expressed over states and
there is no action input.

## AutoML / Tuning

The product tuning API mirrors the occupancy-ratio package while scoring only
real-user proxy quantities such as held-out weighted Bellman risk, calibration
residuals, policy-value stability, and runtime:

```python
from fqe import tune_fqe_auto

tuned = tune_fqe_auto(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    sample_weight=row_weights,
    initial_states=initial_states,
    initial_actions=initial_actions,
    families=("boosted",),  # default; include "neural" only when requested
)

model = tuned.model
candidate_rows = tuned.candidate_rows()
fold_rows = tuned.fold_rows()
```

`budget="fast"` is useful for interactive checks and smoke tests. The default
balanced budget evaluates a deterministic, capped candidate set and refits the
selected stable-guarded candidate on all rows. The tuner never uses benchmark
truth, true Q-functions, or Monte Carlo policy values for selection.

## Target-Validation Assisted Tuning

When you have independent target-policy validation rollouts or simulator
labels, use the opt-in target-validation tuner. This path is separate from
proxy-only AutoML and may use target-policy labels for model selection:

```python
from fqe import tune_fqe_with_target_validation

tuned = tune_fqe_with_target_validation(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    initial_states=initial_states,
    initial_actions=initial_actions,
    validation_states=target_states,
    validation_actions=target_actions,
    validation_rewards=target_rewards,
    validation_next_states=target_next_states,
    validation_episode_ids=target_episode_ids,
    validation_timestep=target_timesteps,
    validation_continuation=target_continuation,
    validation_tail_actions=target_tail_actions,
)

model = tuned.model
rows = tuned.validation_rows()
diagnostics = tuned.validation_diagnostics
```

The default `score_mode="n_step_td"` scores fitted candidates on finite
target-policy rollout prefixes plus the candidate's own continuation value at
the prefix tail. These finite rollouts are validation samples, not exact
infinite-horizon truth; truncation-tail diagnostics report the remaining
discount mass. In Q-mode, pass `validation_tail_actions` whenever a validation
prefix may continue past its last observed row, so the tail value is evaluated
under the target policy.

The default `selection_rule="min_score"` picks the minimum target-validation
score. Pass `selection_rule="one_se"` for a more conservative
one-standard-error selector. Diagnostics always report both
`selected_min_score_candidate_id` and `selected_one_se_candidate_id`.

If you only have a scalar target-policy Monte Carlo value, use
`score_mode="scalar_value"` with `target_value` and optionally
`target_value_se`. Scalar mode validates the initial-state policy value only;
it does not validate the whole Q-function.

## Stationary Weighted FQE

For stationary or near-stationary policy evaluation, FQE can first estimate
discounted occupancy-ratio weights with a separate near-stationary discount and
then pass those weights into the same weighted FQE fitter. The default
stationary backend is currently the official Google DualDICE wrapper; pass
joint target-policy initial state-actions and a Google Research checkout:

```python
from fqe import GoogleDualDICEConfig, StationaryWeightedFQEConfig, fit_stationary_weighted_fqe

result = fit_stationary_weighted_fqe(
    states=states,
    actions=actions,
    target_actions=target_actions_under_eval_policy,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,        # value-estimation discount
    gamma_ratio=0.99,  # occupancy weighting discount, must be < 1
    initial_states=initial_states,
    initial_actions=initial_actions_under_eval_policy,
    sample_weight=row_weights,
    config=StationaryWeightedFQEConfig(
        google_dualdice_config=GoogleDualDICEConfig(
            google_research_path="/tmp/google-research",
            num_updates=1000,
        ),
    ),
)

model = result.fqe_model
weights = result.sample_weight
diagnostics = result.diagnostics
```

Install the optional stationary extra, or install `occupancy-ratio` with its
Google DualDICE extra, before using the default harness. The combined FQE
weights are normalized to mean one by default so occupancy ratios do not rescale
the regression objective.

The package-native FORI backend remains available explicitly:

```python
from fqe import StationaryWeightedFQEConfig

result = fit_stationary_weighted_fqe(
    states=states,
    actions=actions,
    target_actions=target_actions_under_eval_policy,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    gamma_ratio=0.99,
    sample_weight=row_weights,
    config=StationaryWeightedFQEConfig(ratio_backend="occupancy_ratio"),
)
```

For FORI, `ratio_family="auto"` pairs the ratio learner with the FQE family:
boosted FQE uses boosted occupancy-ratio weights, and `family="neural"` uses
the neural occupancy-ratio fitter and neural ratio configs. If the occupancy
fixed point degenerates to effectively uniform row weights, the harness falls
back to the learned action-ratio weights and reports that in diagnostics. The
initial-source correction defaults to the factored, state-only form; pass
`initial_ratio_mode="joint"` only when a joint initial state-action correction
is deliberately intended. If the optional source-ratio nuisance fit fails on a
degenerate small dataset, the harness retries without that source correction and
marks `occupancy_fit_fallback_used=True`.

Benchmark aliases follow the same policy for now: bare
`stationary_weighted_fqe` and `stationary_weighted_neural_fqe` use DualDICE,
while `stationary_weighted_fori_fqe` and `stationary_weighted_fori_neural_fqe`
keep the package-native FORI comparator. The neural FORI comparator uses the
64x64 SiLU stage-budget occupancy settings from `occupancy-ratio`, not the
wider Google-parity diagnostic.

DualDICE remains optional and deploys only when `occupancy-ratio[google-dualdice]`
and a Google Research checkout are available. If those are unavailable, choose
the explicit FORI backend.

The shared minimax-weight facade in `occupancy-ratio` is also available through
stationary FQE:

```python
from fqe import StationaryWeightedFQEConfig, fit_stationary_weighted_fqe
from occupancy_ratio import MinimaxWeightConfig

result = fit_stationary_weighted_fqe(
    states=states,
    actions=actions,
    target_actions=target_actions_under_eval_policy,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    gamma_ratio=0.95,
    initial_states=initial_states,
    initial_actions=initial_actions_under_eval_policy,
    config=StationaryWeightedFQEConfig(
        ratio_backend="minimax_weight",
        minimax_weight_method="google_dice_rl_recommended",
        minimax_weight_config=MinimaxWeightConfig(method="google_dice_rl_recommended"),
    ),
)
```

SCOPE-RL minimax adapters should receive ordered trajectory metadata via
`step_per_trajectory`, or via aligned `episode_ids` and `timesteps`; the
state-marginal SCOPE-RL method also requires `behavior_action_pscore`.

## Policy Sampler Convenience Wrapper

```python
from fqe import fit_fqe_from_policy

def sample_next_actions(next_states, rng, n_samples):
    # Return shape (n, action_dim) for n_samples=1 or
    # shape (n, n_samples, action_dim) for n_samples > 1.
    return policy.sample(next_states, rng=rng, n_samples=n_samples)

model = fit_fqe_from_policy(
    states=states,
    actions=actions,
    next_states=next_states,
    rewards=rewards,
    gamma=0.99,
    next_action_sampler=sample_next_actions,
    n_next_action_samples=8,
)
```

The low-level `fit_fqe_lgbm` call stays deterministic when precomputed
`next_actions` are supplied; the wrapper centralizes sampling when that is more
ergonomic.

## Bellman Calibration Diagnostics

Post-hoc Bellman calibration lives in standalone functions, separate from FQE
fitting:

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

The default `histogram_rescale` calibrator corrects bin-level Bellman target
means while preserving within-bin prediction differences. Diagnostics include
plug-in, fixed-bin debiased, and cross-fitted debiased calibration error,
Bellman residual MSE before/after calibration, bin tables, and a conservative
`apply`, `neutral`, or `do_not_apply` recommendation.

## Low-Rank Operator SBV Candidate Selection

For post-hoc selection among many fitted FQE/Q candidates, the package includes
Low-Rank Operator Supervised Bellman Validation (SBV). It learns one amortized
reward/operator model per rank instead of one Bellman regressor per candidate,
uses trajectory-clean `D_B_train`/`D_B_val`/`D_score` splits, and reports
trajectory-bootstrap standard errors plus a one-standard-error selector.

```python
from fqe import FQECandidate, LowRankOperatorSBVValidator, TransitionDataset, split_by_episode_ids

dataset = TransitionDataset(obs, actions, rewards, next_obs, done, episode_id, timestep)
splits = split_by_episode_ids(dataset, {"D_B": 0.7, "D_score": 0.3}, seed=123)
b_splits = split_by_episode_ids(splits["D_B"], {"D_B_train": 0.8, "D_B_val": 0.2}, seed=124)

candidates = [FQECandidate("checkpoint_010", q10), FQECandidate("checkpoint_020", q20)]
result = LowRankOperatorSBVValidator(gamma=0.99, ranks=[4, 8, 16]).fit_score(
    candidates,
    b_splits["D_B_train"],
    b_splits["D_B_val"],
    splits["D_score"],
    target_policy,
    action_space,
)
```

See `docs/low_rank_operator_sbv.md` for the math, terminal-handling convention,
the conditional generative baseline, the `fqe-compare-validators` report
entrypoint, and the Gym neural-width selector benchmark.

## Neural FQE

The neural API mirrors the LightGBM API:

```python
from fqe import NeuralFQEConfig, fit_fqe_neural

model = fit_fqe_neural(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    config=NeuralFQEConfig.stable_defaults(
        hidden_dims=(256, 256),
        gradient_steps_per_iteration=20,
        device="cpu",
    ),
)

q_values = model.predict_q(states, actions)
```

For value-only neural fitted value iteration:

```python
from fqe import fit_value_neural

value_model = fit_value_neural(
    states=states,
    next_states=next_states,
    rewards=rewards,
    gamma=0.95,
)
```

The neural implementation uses a target network, Polyak updates, gradient
clipping, Huber loss by default, and input standardization fitted on the
training split only. Install PyTorch or the `neural` extra before using neural
fitters.

## Benchmark Suite

The companion benchmark package compares the package estimators against local
legacy baselines and optional external FQE implementations:

```bash
fqe-benchmark \
  --stage smoke \
  --automl-tuning fast \
  --stationary-gamma-ratio 0.99 \
  --output-root outputs/fqe_benchmark \
  --no-plots
```

The module entrypoint is also available as `python -m fqe_benchmark.run`.

Smoke and core stages run built-in tabular and controlled synthetic settings
with exact Q/value truth. Full-stage configuration also emits Hopper/Deep OPE
preflight rows; the heavy Hopper execution remains delegated to the existing
`hopper_fqe_benchmark` pipeline unless its artifacts and external dependencies
are available and wired for that run.

Outputs are written under `<output-root>/<stage>/`:

- `results.csv`
- `summary.csv`
- `diagnostics.json`
- `manifest.json`
- `tuning_results.csv`
- `value_error.png`, `q_mse.png`, and `runtime.png` when plotting is enabled
