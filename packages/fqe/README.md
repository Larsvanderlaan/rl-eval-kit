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
    config=BoostedFQEConfig.stable_defaults(seed=123),
)

q_values = model.predict_q(states, actions)
policy_value = model.estimate_policy_value(initial_states, initial_actions)
```

`next_actions` may be either one action per transition with shape
`(n, action_dim)` or multiple sampled evaluation-policy actions with shape
`(n, n_action_samples, action_dim)`. Multiple actions are averaged in the
Bellman target.

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
- `value_error.png`, `q_mse.png`, and `runtime.png` when plotting is enabled
