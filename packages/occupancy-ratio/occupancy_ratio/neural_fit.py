"""Neural estimator fit and CV entrypoints."""

from occupancy_ratio._neural_impl import (
    fit_discounted_occupancy_ratio_neural,
    tune_discounted_occupancy_ratio_neural_cv,
)

__all__ = [
    "fit_discounted_occupancy_ratio_neural",
    "tune_discounted_occupancy_ratio_neural_cv",
]
