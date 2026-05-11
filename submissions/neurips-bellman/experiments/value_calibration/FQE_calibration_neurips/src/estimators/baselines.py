from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy
from .ensemble_fqe import EnsembleFQEConfig, fit_ensemble_fqe
from .model_wrappers import maybe_wrap_prediction_distortion
from .neural_fqe import NeuralFQEConfig, fit_neural_fqe
from .random_feature_fqe import RandomFeatureFQEConfig, fit_random_feature_fqe
from .regularized_bellman import RegularizedBellmanConfig, fit_regularized_bellman
from .saddle_point_bellman import (
    IterativeSaddlePointBellmanConfig,
    SaddlePointBellmanConfig,
    fit_iterative_saddle_point_bellman,
    fit_saddle_point_bellman,
)


@dataclass
class EstimatorResult:
    model: Any
    learner: str
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def _with_gamma(params: dict[str, Any], gamma: float) -> dict[str, Any]:
    out = dict(params or {})
    out.setdefault("gamma", gamma)
    return out


def fit_estimator(
    learner: str,
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    gamma: float,
    params: dict[str, Any] | None,
    seed: int,
) -> EstimatorResult:
    learner = str(learner)
    cfg_params = _with_gamma(params or {}, gamma)
    distortion_params = cfg_params.pop("prediction_distortion", None)
    if learner == "neural_fqe":
        model = fit_neural_fqe(batch, n_actions, policy, NeuralFQEConfig(**cfg_params), seed)
    elif learner in {"random_feature_fqe", "linear_fqe"}:
        model = fit_random_feature_fqe(batch, n_actions, policy, RandomFeatureFQEConfig(**cfg_params), seed)
    elif learner == "regularized_bellman":
        model = fit_regularized_bellman(batch, n_actions, policy, RegularizedBellmanConfig(**cfg_params), seed)
    elif learner == "saddle_point_bellman":
        model = fit_saddle_point_bellman(batch, n_actions, policy, SaddlePointBellmanConfig(**cfg_params), seed)
    elif learner == "saddle_point_iterative":
        model = fit_iterative_saddle_point_bellman(
            batch, n_actions, policy, IterativeSaddlePointBellmanConfig(**cfg_params), seed
        )
    elif learner == "ensemble_fqe":
        model = fit_ensemble_fqe(batch, n_actions, policy, EnsembleFQEConfig(**cfg_params), seed)
    else:
        raise ValueError(f"Unknown learner '{learner}'.")
    model = maybe_wrap_prediction_distortion(model, {"prediction_distortion": distortion_params})
    diagnostics = dict(getattr(model, "diagnostics", {}))
    diagnostics.setdefault("failure_flag", 0.0)
    diagnostics.setdefault("failure_reason", "")
    return EstimatorResult(model=model, learner=learner, diagnostics=diagnostics)
