"""Neural discounted occupancy-ratio public facade."""

from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralDiscountedOccupancyRatioModel,
    NeuralOccupancyRegressionConfig,
    NeuralTransitionRatioConfig,
    fit_action_ratio_neural,
    fit_discounted_occupancy_ratio_neural,
    fit_transition_ratio_neural,
    tune_discounted_occupancy_ratio_neural_cv,
)

DiscountedOccupancyRatioNeuralModel = NeuralDiscountedOccupancyRatioModel

__all__ = [
    "NeuralActionRatioConfig",
    "NeuralTransitionRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralDiscountedOccupancyRatioModel",
    "DiscountedOccupancyRatioNeuralModel",
    "fit_action_ratio_neural",
    "fit_transition_ratio_neural",
    "fit_discounted_occupancy_ratio_neural",
    "tune_discounted_occupancy_ratio_neural_cv",
]
