# Optional Adapters

AIRL/GAIL use HumanCompatibleAI `imitation`. d3rlpy and SCOPE-RL adapters are
lazy and guarded. SCOPE-RL's public OPE input path is dataset-bound; use
`ReusableScopeRLQEstimator` when you have a reusable fitted value model.

`ScopeRLDatasetBoundQEstimator` is the explicit name for fitted-row diagnostics.
The compatibility alias `ScopeRLQEstimator` remains available.

## Neural FQE Backends

`auto_neural_fqe` is the recommended neural Q backend for pooled GenPQR and the
default for DeepGenPQR pooled mode. It routes by action space:

- finite actions: `ActionHeadNeuralFQEstimator`;
- continuous actions: the generic `FQEQEstimator(family="neural")`.

The finite-action action-head backend is designed for offline IRL settings
where action coverage is uneven. Instead of concatenating a one-hot action to
the state and asking one MLP to learn everything, it uses a shared state trunk,
a pooled state baseline, centered policy-log-probability skip features, and
regularized action residual heads. Balanced action mini-batches reduce the
chance that common actions dominate the fitted Q function.

Use `fqe_neural` explicitly when you want the generic state-action MLP. Use
`fqe_action_head_neural`, `action_head_neural_fqe`, or
`stratified_neural_fqe` explicitly when you want the finite-action action-head
backend and do not need auto-routing.

Important tradeoffs:

- Action-head FQE can handle rare anchors better than anchor DeepPQR when the
  action-reward surface is smooth enough to pool.
- It is still extrapolating when an action has little or no coverage. Inspect
  per-action error diagnostics in benchmarks and action-count diagnostics in
  fitted models.
- Anchor DeepPQR is usually sharper when anchor support is healthy and the
  anchor normalization is central to the estimand.
