from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from .fqe import FQEConfig, FQEResult, fit_weighted_fqe_nn
from .neural_rkhs_weights import NeuralRKHSWeightsConfig, estimate_ratio_neural_rkhs
from .ratio_estimation import (
    NeuralRatioConfig,
    RatioEstimationResult,
    estimate_ratio_closed_form_linear,
    estimate_ratio_saddle_linear,
    estimate_ratio_saddle_neural,
)
from .utils import TransitionBatch, stabilize_weights, state_action_one_hot


WeightModel = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SWFQEResult:
    """End-to-end stationary-weighted FQE output."""

    fqe_result: FQEResult
    weights: np.ndarray
    ratio_result: Optional[RatioEstimationResult]
    metadata: dict


def _default_tabular_feature_map(
    states: np.ndarray,
    actions: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    return state_action_one_hot(states, actions, n_states=n_states, n_actions=n_actions)


def _weights_from_model(
    weight_model,
    states: np.ndarray,
    actions: np.ndarray,
    n_states: int,
    n_actions: int,
    device: str = "cpu",
) -> np.ndarray:
    if isinstance(weight_model, nn.Module):
        weight_model.eval()
        x = np.zeros((len(states), n_states + n_actions), dtype=np.float32)
        rows = np.arange(len(states))
        x[rows, np.asarray(states, dtype=np.int64)] = 1.0
        x[rows, n_states + np.asarray(actions, dtype=np.int64)] = 1.0
        with torch.no_grad():
            weights = weight_model(torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        return np.asarray(weights, dtype=np.float64).reshape(-1)
    if callable(weight_model):
        return np.asarray(weight_model(states, actions), dtype=np.float64).reshape(-1)
    if hasattr(weight_model, "predict"):
        x = np.zeros((len(states), n_states + n_actions), dtype=np.float32)
        rows = np.arange(len(states))
        x[rows, np.asarray(states, dtype=np.int64)] = 1.0
        x[rows, n_states + np.asarray(actions, dtype=np.int64)] = 1.0
        return np.asarray(weight_model.predict(x), dtype=np.float64).reshape(-1)
    raise TypeError("weight_model must be a torch module, callable, or object with predict(...).")


def resolve_sample_weights(
    batch: TransitionBatch,
    n_states: int,
    n_actions: int,
    sample_weights: np.ndarray | None = None,
    weight_model=None,
    ratio_model: str | None = None,
    ratio_solver: str = "closed_form",
    ratio_feature_map: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    gamma_ratio: float = 1.0,
    min_weight: float = 1e-8,
    max_weight: float | None = 20.0,
    device: str = "cpu",
    ratio_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, Optional[RatioEstimationResult], dict]:
    """
    Resolve SW-FQE sample weights from direct, linear-estimated, or neural/callable sources.
    """

    if ratio_kwargs is None:
        ratio_kwargs = {}

    if sample_weights is not None:
        weights, _ = stabilize_weights(sample_weights, min_weight=min_weight, max_weight=max_weight)
        return weights, None, {"weight_source": "precomputed"}

    if weight_model is not None:
        raw_weights = _weights_from_model(
            weight_model=weight_model,
            states=batch.states,
            actions=batch.actions,
            n_states=n_states,
            n_actions=n_actions,
            device=device,
        )
        weights, _ = stabilize_weights(raw_weights, min_weight=min_weight, max_weight=max_weight)
        source = "neural_weight_model" if isinstance(weight_model, nn.Module) else "callable_weight_model"
        return weights, None, {"weight_source": source}

    if ratio_model is None:
        weights = np.ones(len(batch), dtype=np.float64)
        return weights, None, {"weight_source": "uniform"}

    if ratio_feature_map is None:
        ratio_feature_map = lambda s, a: _default_tabular_feature_map(s, a, n_states=n_states, n_actions=n_actions)

    weight_features = ratio_feature_map(batch.states, batch.actions)
    critic_features = ratio_feature_map(batch.states, batch.actions)
    next_critic_features = ratio_feature_map(batch.next_states, batch.next_actions)

    common_kwargs = dict(
        weight_features=weight_features,
        critic_features=critic_features,
        next_critic_features=next_critic_features,
        gamma_ratio=gamma_ratio,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    common_kwargs.update(ratio_kwargs)

    if ratio_model == "linear":
        if ratio_solver == "closed_form":
            ratio_result = estimate_ratio_closed_form_linear(**common_kwargs)
        elif ratio_solver in {"saddle", "extragradient"}:
            ratio_result = estimate_ratio_saddle_linear(**common_kwargs)
        else:
            raise ValueError(f"Unknown linear ratio_solver '{ratio_solver}'.")
    elif ratio_model == "neural":
        neural_cfg = ratio_kwargs.get("config") if ratio_kwargs is not None else None
        if neural_cfg is not None and not isinstance(neural_cfg, NeuralRatioConfig):
            raise TypeError("ratio_kwargs['config'] must be a NeuralRatioConfig for ratio_model='neural'.")
        ratio_result = estimate_ratio_saddle_neural(**common_kwargs)
    elif ratio_model == "neural_rkhs":
        rkhs_cfg = ratio_kwargs.get("config") if ratio_kwargs is not None else None
        if rkhs_cfg is not None and not isinstance(rkhs_cfg, NeuralRKHSWeightsConfig):
            raise TypeError("ratio_kwargs['config'] must be a NeuralRKHSWeightsConfig for ratio_model='neural_rkhs'.")
        ratio_result = estimate_ratio_neural_rkhs(**common_kwargs)
    else:
        raise ValueError(f"Unknown ratio_model '{ratio_model}'.")

    if ratio_model == "linear":
        source = f"{ratio_model}_{ratio_solver}"
    elif ratio_model == "neural":
        source = "neural_saddle"
    else:
        source = "neural_rkhs"
    return ratio_result.weights, ratio_result, {"weight_source": source}


def fit_stationary_weighted_fqe(
    batch: TransitionBatch,
    n_states: int,
    n_actions: int,
    sample_weights: np.ndarray | None = None,
    weight_model=None,
    ratio_model: str | None = "linear",
    ratio_solver: str = "closed_form",
    ratio_feature_map: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    gamma_ratio: float = 1.0,
    fqe_config: FQEConfig | None = None,
    ratio_kwargs: Optional[dict] = None,
    min_weight: float = 1e-8,
    max_weight: float | None = 20.0,
    seed: int = 0,
) -> SWFQEResult:
    """
    User-friendly end-to-end SW-FQE wrapper with simple defaults.

    The weighted FQE core remains modular: it accepts sample weights directly.
    This wrapper just makes it easy to obtain those weights from a linear ratio
    estimator, a neural saddle estimator, a precomputed vector, or a callable/neural weight model.
    """

    device = fqe_config.device if fqe_config is not None else "cpu"
    weights, ratio_result, metadata = resolve_sample_weights(
        batch=batch,
        n_states=n_states,
        n_actions=n_actions,
        sample_weights=sample_weights,
        weight_model=weight_model,
        ratio_model=ratio_model,
        ratio_solver=ratio_solver,
        ratio_feature_map=ratio_feature_map,
        gamma_ratio=gamma_ratio,
        min_weight=min_weight,
        max_weight=max_weight,
        device=device,
        ratio_kwargs=ratio_kwargs,
    )
    fqe_result = fit_weighted_fqe_nn(
        batch=batch,
        n_states=n_states,
        n_actions=n_actions,
        weights=weights,
        config=fqe_config,
        seed=seed,
    )
    metadata.update(
        {
            "gamma_ratio": float(gamma_ratio),
            "weight_mean": float(np.mean(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
        }
    )
    return SWFQEResult(fqe_result=fqe_result, weights=weights, ratio_result=ratio_result, metadata=metadata)
