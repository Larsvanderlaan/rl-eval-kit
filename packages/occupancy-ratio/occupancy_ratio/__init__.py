from occupancy_ratio.calibration import (
    calibrate_occupancy_bellman_binning,
    estimate_ope_bellman_control_variate,
    occupancy_bellman_calibration_diagnostics,
    plot_occupancy_bellman_calibration_diagnostics,
    recommend_occupancy_bellman_calibration,
)
from occupancy_ratio.fit_importance_and_transition_ratios import (
    fit_importance_ratio_lgbm,
    fit_transition_ratio_lgbm,
)
from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    DiscountedOccupancyRatioModel,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
    fit_occupancy_ratio_lgbm,
    make_forward_occupancy_dataset,
    tune_discounted_occupancy_ratio_cv,
)

_NEURAL_EXPORTS = {
    "NeuralDiscountedOccupancyRatioModel",
    "DiscountedOccupancyRatioNeuralModel",
    "NeuralActionRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralTransitionRatioConfig",
    "fit_discounted_occupancy_ratio_neural",
}


def __getattr__(name: str):
    if name in _NEURAL_EXPORTS:
        from occupancy_ratio import fit_occupancy_ratio_neural as neural

        if name == "DiscountedOccupancyRatioNeuralModel":
            value = getattr(neural, "NeuralDiscountedOccupancyRatioModel")
        else:
            value = getattr(neural, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "calibrate_occupancy_bellman_binning",
    "occupancy_bellman_calibration_diagnostics",
    "recommend_occupancy_bellman_calibration",
    "plot_occupancy_bellman_calibration_diagnostics",
    "estimate_ope_bellman_control_variate",
    "ActionRatioConfig",
    "TransitionRatioConfig",
    "OccupancyRegressionConfig",
    "DiscountedOccupancyRatioModel",
    "fit_discounted_occupancy_ratio",
    "tune_discounted_occupancy_ratio_cv",
    "fit_occupancy_ratio_lgbm",
    "make_forward_occupancy_dataset",
    "fit_importance_ratio_lgbm",
    "fit_transition_ratio_lgbm",
    "NeuralActionRatioConfig",
    "NeuralTransitionRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralDiscountedOccupancyRatioModel",
    "DiscountedOccupancyRatioNeuralModel",
    "fit_discounted_occupancy_ratio_neural",
]
