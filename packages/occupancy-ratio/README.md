# occupancy-ratio

Discounted occupancy-ratio estimators for off-policy reinforcement-learning
evaluation.

The importable Python package is `occupancy_ratio`. It provides boosted and
neural fitted occupancy-ratio iteration (FORI), conservative product AutoML
tuning, diagnostics/calibration helpers, benchmark tooling, and an optional
Google DualDICE comparator. It also exposes `fit_minimax_weight(...)`, a common
facade for official Google DICE variants and SCOPE-RL minimax weight learners.

## Install

```bash
python -m pip install -e "packages/occupancy-ratio"
```

Common extras:

```bash
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
python -m pip install -e "packages/occupancy-ratio[docs]"
```

Build the documentation site:

```bash
python -m mkdocs build --strict -f packages/occupancy-ratio/mkdocs.yml
```

## Quickstart

```python
from occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
)

model = fit_discounted_occupancy_ratio(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    gamma=0.99,
    occupancy=OccupancyRegressionConfig.stable_defaults(seed=123),
    action_ratio=ActionRatioConfig.stable_defaults(show_progress=False),
    transition_ratio=TransitionRatioConfig.stable_defaults(show_progress=False),
)

weights = model.predict_state_action_ratio(states, actions)
state_ratios = model.predict_state_ratio(states, actions)
```

For production use, start with the boosted stable defaults for a single fit or
the neural-default AutoML entrypoint for candidate tuning:

```python
from occupancy_ratio import OccupancyTuningConfig, tune_occupancy_ratio_auto

tuned = tune_occupancy_ratio_auto(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    gamma=0.99,
    rewards=rewards,
    config=OccupancyTuningConfig(budget="balanced"),
)

model = tuned.model
candidate_rows = tuned.candidate_rows()
fold_rows = tuned.fold_rows()
```

AutoML selection uses proxy diagnostics only; it never selects using oracle
ratios, benchmark truth, or target-policy Monte Carlo values.

## External Minimax Weights

Use `fit_minimax_weight(...)` when you want official Google or SCOPE-RL
estimators behind the same prediction helpers:

```python
from occupancy_ratio import fit_minimax_weight

model = fit_minimax_weight(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    target_next_actions=target_next_actions_under_pi,
    gamma=0.95,
    initial_states=initial_states,
    initial_actions=initial_actions_under_pi,
    method="google_dice_rl_recommended",
)
```

## Target-Validation Assisted Tuning

When independent target-policy validation rollouts or simulator labels are
available, use the opt-in target-validation tuner. Existing proxy-only AutoML
defaults remain unchanged.

```python
from occupancy_ratio import tune_occupancy_ratio_with_target_validation

tuned = tune_occupancy_ratio_with_target_validation(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    target_next_actions=target_next_actions_under_pi,
    rewards=rewards,
    gamma=0.99,
    initial_states=initial_states,
    initial_actions=initial_actions_under_pi,
    validation_states=target_states,
    validation_actions=target_actions,
    validation_rewards=target_rewards,
    validation_episode_ids=target_episode_ids,
    validation_timestep=target_timesteps,
    validation_continuation=target_continuation,
)

model = tuned.model
rows = tuned.validation_rows()
diagnostics = tuned.validation_diagnostics
```

The default `score_mode="discounted_moments"` compares candidate
reference-weighted moments, `E_ref[w f]`, against empirical discounted
occupancy moments from target-policy validation rollouts. Finite rollouts are
validation samples rather than exact infinite-horizon truth; truncation-tail
diagnostics report the remaining discount mass.

The default `selection_rule="min_score"` picks the minimum guarded validation
score. Occupancy guardrails run first, so scalar OPE and moment validation do
not bypass ESS, clipping, normalization, or noncollapse checks. Pass
`selection_rule="one_se"` for a conservative one-standard-error selector.
Diagnostics always report both `selected_min_score_candidate_id` and
`selected_one_se_candidate_id`.

If only a scalar target-policy value is available, use
`score_mode="scalar_ope"` with `target_value` and optionally `target_value_se`.
Under the package's normalized discounted-occupancy convention, scalar mode
compares `mean_ref(w * reward)` to `(1 - gamma) * target_value`. Scalar mode is
value-only and does not validate the full ratio function.

## Documentation Map

The MkDocs site under `docs/` contains:

- installation and optional extras;
- package architecture and stable import paths;
- boosted and neural quickstarts;
- data-shape contracts;
- initial-source correction semantics;
- AutoML tuning behavior and telemetry;
- target-validation assisted tuning;
- diagnostics, calibration, and benchmark usage;
- generated API reference pages for public functions, configs, and models.

## Benchmarks

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators oracle boosted_tree neural_network \
  --no-google-dualdice \
  --no-plots
```

The module entrypoints are `python -m occupancy_ratio_benchmark.run`,
`python -m occupancy_ratio_benchmark.dualdice_grid`, and
`python -m occupancy_ratio_benchmark.defaults_report`.
