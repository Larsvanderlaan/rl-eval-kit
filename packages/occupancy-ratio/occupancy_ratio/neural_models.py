"""Fitted neural occupancy-ratio model objects."""

from occupancy_ratio._neural_impl import NeuralDiscountedOccupancyRatioModel

DiscountedOccupancyRatioNeuralModel = NeuralDiscountedOccupancyRatioModel

__all__ = [
    "NeuralDiscountedOccupancyRatioModel",
    "DiscountedOccupancyRatioNeuralModel",
]
