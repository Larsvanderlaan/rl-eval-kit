"""Controlled discounted-occupancy FQE benchmark."""

from .configs import (
    EvaluationConfig,
    FQESolverConfig,
    RatioFeatureConfig,
    WeightEstimatorConfig,
)
from .envs import LinearGaussianEnv, LinearGaussianEnvConfig

__all__ = [
    "EvaluationConfig",
    "FQESolverConfig",
    "LinearGaussianEnv",
    "LinearGaussianEnvConfig",
    "RatioFeatureConfig",
    "WeightEstimatorConfig",
    "run_main",
]


def __getattr__(name: str):
    if name == "run_main":
        from .run_experiment import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
