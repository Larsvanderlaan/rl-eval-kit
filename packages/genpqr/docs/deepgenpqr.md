# DeepGenPQR

DeepGenPQR is the neural product workflow for GenPQR. It keeps the modular
GenPQR identity, but chooses strong neural defaults: Deep AIRL for policy
estimation and action-aware neural FQE for Q estimation.

```python
from genpqr import DeepGenPQRConfig, fit_deep_genpqr

result = fit_deep_genpqr(
    dataset=dataset,
    gamma=0.99,
    env=env,
    config=DeepGenPQRConfig.from_preset("deepgenpqr_airl_fqe_balanced"),
)
```

The default `q_mode="pooled_fqe"` pools all observed actions in neural FQE. Its
default `q_backend="auto_neural_fqe"` chooses the backend from the action space:

- Finite actions use `ActionHeadNeuralFQEstimator`. This model has a shared
  state trunk, a state-only baseline, centered target-policy log-probability
  skip features, and regularized action residual heads. It is designed to avoid
  the over-smoothing across sparse one-hot action features that can happen with
  a generic state-action MLP.
- Continuous actions use the generic neural FQE adapter, because the action-head
  architecture is finite-action specific.

Use `q_mode="anchor_deeppqr"` when the DeepPQR state-only anchor
parameterization is the desired inductive bias:

```python
config = DeepGenPQRConfig(
    q_mode="anchor_deeppqr",
    anchor_action=0,
    min_anchor_count=25,
)
```

## Practical Choice Of Q Mode

Use pooled FQE when:

- anchor actions are rare or operationally hard to collect;
- action values should share strength across related states and actions;
- the action-reward surface is expected to be smooth enough for pooling;
- you need a reusable all-action Q model even when anchor support is weak.

Use anchor DeepPQR when:

- the reward normalization is explicitly tied to an anchor action;
- that anchor action has enough logged support across the relevant state space;
- failing closed on weak anchor support is preferable to extrapolating the
  anchor value.

In rare-anchor smooth settings, action-head pooled FQE can be a good fallback
because it pools information through the shared trunk and the policy-log-prob
skip. In rough action-reward settings, or when the anchor action is nearly
unobserved, pooled FQE may still extrapolate poorly. Inspect per-action and
anchor-support diagnostics before treating a recovered reward as production
quality.

Keep `anchor_fallback="error"` for analysis runs; use
`anchor_fallback="pooled_fqe"` when production jobs should record a warning and
refit pooled FQE instead of failing on weak anchor support.

## Tuning The Pooled Action-Head Backend

The stable action-head defaults use moderate residual shrinkage:

```python
config = DeepGenPQRConfig(
    q_mode="pooled_fqe",
    q_backend="auto_neural_fqe",
    q_config={
        "config_overrides": {
            "hidden_dims": (256, 256, 128),
            "residual_l2": 1e-3,
            "policy_log_prob_skip": True,
            "balanced_batches": True,
        },
        "n_next_action_samples": 16,
    },
)
```

Increase `residual_l2` when actions are sparse and you want to shrink toward a
pooled state-only baseline. Decrease it when action-specific structure is
well-covered and important. If you need the legacy generic neural FQE backend,
set `q_backend="fqe_neural"` explicitly.

For continuous actions, pass a fixed anchor action or a callable
`anchor_action(states)` for state-dependent anchors. `anchor_selector` only
chooses which logged rows are used for the anchor fit; selected rows must still
match `anchor_action` within `anchor_tolerance`. Do not use `anchor_selector`
to relabel arbitrary non-anchor rows as anchor rows. The fitted result reports
anchor count, weighted anchor count, anchor fraction, policy density at the
anchor, weak-support flags, reward quantiles, Q quantiles, and Monte Carlo
normalization standard errors when applicable.

`normalization_config` can build simple normalization policies inside the
config, for example `{"kind": "uniform"}` or
`{"kind": "anchor", "anchor_action": 0}` for finite actions.

Use `result.save(path)` and `load_deep_genpqr_result(path)` for safe portable
serialization when the fitted policy/Q/normalization objects support it.
Action-head neural FQE supports portable save/load when the fitted policy and
normalization policy are portable. Nonportable but pickleable policies can be
saved through the labeled unsafe fallback; load those artifacts only with
`allow_pickle=True` and only from trusted local runs.
