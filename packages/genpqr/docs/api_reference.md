# API Reference

Core v1 public objects:

- Fitters: `fit_genpqr`, `fit_genpqr_auto`, `fit_genpqr_crossfit`, `fit_deep_genpqr`
- Config/results: `GenPQRConfig`, `GenPQRResult`, `GenPQRCrossFitResult`, `DeepGenPQRConfig`, `DeepGenPQRResult`
- Datasets: `TransitionDataset`, `EpisodeDataset`
- Diagnostics: `GenPQRDiagnostics`
- Action/normalization: `ActionSpaceSpec`, `DiscreteNormalizationPolicy`, `ContinuousNormalizationPolicy`
- Policy/Q protocols: `PolicyEstimator`, `EstimatedPolicy`, `QEstimator`, `FittedQFunction`, `RewardFunction`
- Built-ins: `BehaviorCloningPolicyEstimator`, `DeepPQRAnchorQEstimator`,
  `NeuralDeepPQRAnchorQEstimator`, `FQEQEstimator`, `AutoNeuralFQEstimator`,
  `ActionHeadNeuralFQEstimator`, `ActionHeadNeuralFQEConfig`,
  `ConstantFittedQFunction`
- Optional adapters: `D3RLPYFQEstimator`, `ScopeRLDatasetBoundQEstimator`, `ReusableScopeRLQEstimator`
- Registry: `register_policy_estimator`, `register_q_estimator`, `available_policy_estimators`, `available_q_estimators`
- Serialization: `save_genpqr_result`, `load_genpqr_result`, `save_deep_genpqr_result`, `load_deep_genpqr_result`
- DeepGenPQR: `DeepGenPQRQMode`, `DeepGenPQRAnchorFallback`, `list_deepgenpqr_presets`
- Presets/version: `list_presets`, `__version__`

Compatibility policy: Core v1 public objects keep backward-compatible call
signatures within the `0.1.x` line except for bug fixes that reject previously
ambiguous or unsafe inputs with explicit GenPQR errors.

Default neural Q routing: `DeepGenPQRConfig.q_backend` defaults to
`"auto_neural_fqe"`. For finite actions this resolves to
`ActionHeadNeuralFQEstimator`; for continuous actions it resolves to generic
neural FQE through `FQEQEstimator(family="neural")`. The legacy generic
finite-action neural FQE path remains available as `q_backend="fqe_neural"`.
Portable serialization supports the resolved action-head finite-action Q
function when the fitted policy and normalization policy are also portable.
