# API Reference

## FQE

Public imports live in `fqe`.
Facade modules are also available as `fqe.boosted` and `fqe.neural`.

- `BoostedFQEConfig`, `FQEModel`, `fit_fqe_lgbm`, `fit_value_lgbm`, `fit_fqe_from_policy`, `tune_fqe_cv`
- `NeuralFQEConfig`, `NeuralFQEModel`, `fit_fqe_neural`, `fit_value_neural`, `fit_fqe_neural_from_policy`, `tune_fqe_neural_cv`

`FQEModel` and `NeuralFQEModel` expose `predict`, `predict_q`,
`predict_value`, `estimate_policy_value`, and `to_legacy_dict`.

## Occupancy Ratio

Public imports live in `occupancy_ratio`.
Facade modules are also available as `occupancy_ratio.boosted`,
`occupancy_ratio.neural`, and `occupancy_ratio.nuisance`.

- Boosted configs and fitters: `ActionRatioConfig`, `TransitionRatioConfig`, `OccupancyRegressionConfig`, `fit_discounted_occupancy_ratio`, `tune_discounted_occupancy_ratio_cv`
- Neural configs and fitters: `NeuralActionRatioConfig`, `NeuralTransitionRatioConfig`, `NeuralOccupancyRegressionConfig`, `fit_discounted_occupancy_ratio_neural`, `tune_discounted_occupancy_ratio_neural_cv`
- Models: `DiscountedOccupancyRatioModel`, `NeuralDiscountedOccupancyRatioModel`

Both occupancy models expose `predict_state_action_ratio`,
`predict_action_ratio`, `predict_state_ratio`, `predict_for_target_actions`,
and `to_legacy_dict`.

LSIF is the default action and transition density-ratio nuisance loss for both
boosted and neural implementations. Logistic nuisance ratios are opt-in through
`density_ratio_loss="logistic"`.
