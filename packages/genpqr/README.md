# genPQR

`genpqr` is a modular inverse-reinforcement-learning package for generalized
Policy-to-Q-to-Reward recovery. It is designed to sit beside `fqe` and
`occupancy-ratio`: policy estimation and Q estimation are pluggable, while the
reward-recovery identity is kept small, explicit, and testable.

Core import only requires NumPy:

```python
import genpqr
```

Optional learners are lazy. Install extras only for the backends you use:

```bash
python -m pip install -e "packages/genpqr[fqe,torch]"
python -m pip install -e "packages/genpqr[imitation]"
python -m pip install -e "packages/genpqr[d3rlpy,scope-rl]"
```

## Core v1 Features

- `TransitionDataset` and `EpisodeDataset` for validated row-wise and
  trajectory-preserving workflows.
- Public registries for custom named policy and Q estimators.
- Contract checks in `genpqr.testing` for extension authors.
- Typed diagnostics through `GenPQRResult.diagnostics_report`, while the legacy
  flat `diagnostics` dictionary remains available.
- `fit_genpqr_crossfit(...)` for out-of-fold reward predictions.
- `result.save(path)` and `load_genpqr_result(path)` for manifest-backed
  serialization.
- `DeepPQRAnchorQEstimator` and lazy Torch `NeuralDeepPQRAnchorQEstimator`.
- `fit_deep_genpqr(...)` as a neural product workflow for Deep AIRL +
  action-aware neural FQE, with an opt-in DeepPQR anchor/subset Q mode.
- Stable Core v1 public surface for configs, results, datasets, diagnostics,
  registries, presets, and protocols within the `0.1.x` line.

## DeepGenPQR Neural Workflow

DeepGenPQR is the high-level neural front door. By default it runs Deep AIRL
through the lazy `imitation` adapter and then fits pooled neural FQE over all
logged action rows. For finite actions, the default pooled backend is
`auto_neural_fqe`, which routes to an action-head neural FQE architecture
instead of a generic `[state, one_hot(action)]` MLP. Continuous-action data
still route to the generic neural FQE backend.

```python
from genpqr import DeepGenPQRConfig, fit_deep_genpqr

result = fit_deep_genpqr(
    dataset=dataset,
    gamma=0.99,
    env=env,
    config=DeepGenPQRConfig.from_preset("deepgenpqr_airl_fqe_balanced"),
)

rewards = result.predict_reward(states, actions)
```

Use pooled FQE when anchor actions are rare, when you want a reusable
all-action Q model, or when the reward surface is smooth enough that sharing
statistical strength across states and actions is credible. For finite actions,
the action-head pooled backend uses a shared state trunk, a state-only baseline,
balanced action mini-batches, policy-log-probability skip features, and
regularized action residual heads. This makes it substantially safer than a
plain concatenated action MLP when some actions are sparse.

Use anchor DeepPQR when the paper's state-only anchor parameterization is the
intended inductive bias and anchor support is healthy. It can be extremely
accurate in well-covered anchor settings, but it intentionally fails closed when
anchor support is weak. The default anchor behavior is strict; set
`anchor_fallback="pooled_fqe"` only when deployment should prefer a pooled
refit over an anchor-support failure.

To use the DeepPQR-style anchor/subset parameterization instead of pooled FQE,
switch the Q mode:

```python
config = DeepGenPQRConfig.from_preset(
    "deepgenpqr_airl_anchor_balanced",
    anchor_action=0,
    min_anchor_count=25,
)
result = fit_deep_genpqr(dataset=dataset, gamma=0.99, env=env, config=config)
```

To force the legacy generic neural FQE path for finite actions, opt out
explicitly:

```python
config = DeepGenPQRConfig(
    q_mode="pooled_fqe",
    q_backend="fqe_neural",  # generic state-action MLP
)
```

To tune the action-head pooled backend, pass `q_config` overrides:

```python
config = DeepGenPQRConfig(
    q_mode="pooled_fqe",
    q_backend="auto_neural_fqe",
    q_config={
        "config_overrides": {
            "residual_l2": 1e-3,
            "hidden_dims": (256, 256, 128),
        },
        "n_next_action_samples": 16,
    },
)
```

## Quickstart: default AIRL + action-aware neural FQE

The one-line DeepGenPQR default is AIRL policy estimation followed by
action-aware neural FQE. AIRL requires an environment because adversarial
imitation training needs generator rollouts.

```python
from genpqr import DeepGenPQRConfig, fit_deep_genpqr

result = fit_deep_genpqr(
    dataset=dataset,
    gamma=0.99,
    env=env,
    config=DeepGenPQRConfig(q_mode="pooled_fqe"),
)

rewards = result.predict_reward(states, actions)
```

If `env` or the optional `imitation` stack is unavailable, the default fails
with a clear configuration or missing-dependency error. It does not silently
switch algorithms.

## AIRL, GAIL, And Minimax Workflows

`policy` and `q` can be named lazy adapters or user-supplied protocol objects.
This makes the common IRL workflows one configuration change apart:

```python
from genpqr import GenPQRConfig, fit_genpqr

airl_result = fit_genpqr(
    states=states,
    actions=actions,
    next_states=next_states,
    terminals=dones,
    gamma=0.99,
    env=env,
    config=GenPQRConfig(
        policy="imitation_airl",
        q="fqe_neural",
        policy_config={"total_timesteps": 100_000, "demo_batch_size": 512},
    ),
)

gail_result = fit_genpqr(
    states=states,
    actions=actions,
    next_states=next_states,
    terminals=dones,
    gamma=0.99,
    env=env,
    config=GenPQRConfig(
        policy="imitation_gail",
        q="fqe_neural",
        policy_config={"total_timesteps": 100_000},
    ),
)
```

SCOPE-RL minimax Q/value learning can be selected through the lazy adapter. Its
documented OPE input path returns fitted-row prediction arrays, so the built-in
adapter is guarded by default; for reusable out-of-sample reward prediction,
wrap the reusable SCOPE-RL value model in the `QEstimator` protocol.

```python
from genpqr import GenPQRConfig, ScopeRLQEstimator, fit_genpqr

scope_mql = ScopeRLQEstimator(
    method="mql",
    env=env,
    evaluation_policies=evaluation_policies,
    allow_dataset_bound_predictions=True,  # fitted-row diagnostics only
)

diagnostic_result = fit_genpqr(
    states=states,
    actions=actions,
    next_states=next_states,
    terminals=dones,
    gamma=0.99,
    config=GenPQRConfig(policy="behavior_cloning_native", q=scope_mql),
)
```

## Lightweight BC + FQE

For a dependency-light first pass, use native behavior cloning and the existing
`fqe` package:

```python
from genpqr import GenPQRConfig, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_boosted"),
)
```

## Continuous Actions

Continuous GenPQR requires a policy estimator that exposes log densities and a
normalization policy that can sample actions.

```python
from genpqr import ActionSpaceSpec, ContinuousNormalizationPolicy, GenPQRConfig

mu = ContinuousNormalizationPolicy(
    action_dim=action_dim,
    sampler=lambda states, rng, n: rng.normal(size=(states.shape[0], n, action_dim)),
)

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    action_space=ActionSpaceSpec.continuous(action_dim),
    normalization_policy=mu,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_neural"),
)
```

## DeepPQR Anchor-Q Backend

DeepPQR is available as a Q backend for finite actions:

```python
from genpqr import DeepPQRAnchorQEstimator, GenPQRConfig

config = GenPQRConfig(
    policy="behavior_cloning_native",
    q=DeepPQRAnchorQEstimator(anchor_action=0),
)
result = fit_genpqr(..., config=config)
```

This backend estimates the state-only anchor value on the anchor-action subset
and reconstructs the full action-stratified Q with policy log-ratios.

## Cross-Fitting And Serialization

```python
from genpqr import DiscreteNormalizationPolicy, GenPQRConfig, fit_genpqr_crossfit

crossfit = fit_genpqr_crossfit(
    dataset=dataset,
    gamma=0.95,
    n_folds=5,
    normalization_policy=DiscreteNormalizationPolicy.uniform(n_actions),
    config=GenPQRConfig.from_preset("deeppqr_linear"),
)

crossfit.final_result.save("genpqr-result")
```

DeepGenPQR results have the same manifest-backed save path:

```python
from genpqr import load_deep_genpqr_result

result.save("deepgenpqr-result")
loaded = load_deep_genpqr_result("deepgenpqr-result")
```

The default finite-action action-head FQE backend is portable when the fitted
policy and normalization policy are portable. If a custom policy is not
portable but is pickleable, saving falls back to a labeled unsafe artifact;
loading that artifact requires `allow_pickle=True` and should be limited to
trusted local runs.

Run the tiny release-validation suite before changing defaults:

```python
from genpqr.benchmarks import run_tiny_production_validation

report = run_tiny_production_validation()
```

See `docs/` and `examples/` for custom estimators, adapters, DeepPQR, continuous
actions, deployment, API reference, and SCOPE-RL minimax diagnostic workflows.
