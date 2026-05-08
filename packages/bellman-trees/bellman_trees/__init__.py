"""Target-weighted Bellman aggregation trees and tree-ensemble features."""

from ._adapters import SklearnLeafAdapter, XGBoostLeafAdapter
from ._data import BellmanTransitionData
from ._ensemble import BellmanAggregationForest, BellmanLeafEnsembleRegressor
from ._hist_gbt import BellmanHistogramGradientBoostingRegressor
from ._native_tree import BellmanAggregationTree
from ._occupancy import (
    DiscountedOccupancyRatioTree,
    DiscountedOccupancySolveResult,
    discounted_flow_moment,
    discounted_flow_moment_from_ratio,
    solve_discounted_occupancy_ratio,
)
from ._occupancy_hist_gbt import DiscountedOccupancyHistogramGradientBoostingRatioEstimator
from ._solver import BellmanSolveResult, solve_projected_bellman
from ._weights import StabilizedWeights, effective_sample_size, stabilize_weights

__all__ = [
    "BellmanAggregationForest",
    "BellmanAggregationTree",
    "BellmanHistogramGradientBoostingRegressor",
    "BellmanLeafEnsembleRegressor",
    "BellmanSolveResult",
    "BellmanTransitionData",
    "DiscountedOccupancyRatioTree",
    "DiscountedOccupancyHistogramGradientBoostingRatioEstimator",
    "DiscountedOccupancySolveResult",
    "SklearnLeafAdapter",
    "StabilizedWeights",
    "XGBoostLeafAdapter",
    "discounted_flow_moment",
    "discounted_flow_moment_from_ratio",
    "effective_sample_size",
    "solve_discounted_occupancy_ratio",
    "solve_projected_bellman",
    "stabilize_weights",
]
