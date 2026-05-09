"""Neural estimator configuration objects and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from occupancy_ratio.validation import (
    _validate_occupancy_stabilization_config,
    _validate_ratio_prediction_config,
)

__all__ = [
    "NeuralActionRatioConfig",
    "NeuralSourceStateRatioConfig",
    "NeuralTransitionRatioConfig",
    "NeuralOccupancyRegressionConfig",
]


def _validate_neural_common(
    *,
    hidden_dims: Sequence[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_steps: int,
    validation_fraction: float,
    patience: int,
    min_improvement: float,
    grad_clip_norm: Optional[float],
) -> None:
    if not tuple(hidden_dims) or any(int(dim) <= 0 for dim in hidden_dims):
        raise ValueError("hidden_dims must contain positive layer widths.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1).")
    if patience < 0:
        raise ValueError("patience must be nonnegative.")
    if min_improvement < 0.0:
        raise ValueError("min_improvement must be nonnegative.")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive when supplied.")


def _normalized_density_ratio_loss(loss: str) -> str:
    out = str(loss).lower()
    if out not in {"lsif", "logistic"}:
        raise ValueError("density_ratio_loss must be 'lsif' or 'logistic'.")
    return out


def _validate_density_ratio_loss(loss: str, logistic_logit_clip: Optional[float]) -> None:
    _normalized_density_ratio_loss(loss)
    if logistic_logit_clip is not None and float(logistic_logit_clip) <= 0.0:
        raise ValueError("logistic_logit_clip must be positive when supplied.")


@dataclass
class NeuralActionRatioConfig:
    """Tuning for neural action importance-ratio nuisance fits."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_steps: int = 400
    validation_fraction: float = 0.2
    patience: int = 20
    min_improvement: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    normalization_penalty: float = 10.0
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.normalization_penalty < 0.0:
            raise ValueError("normalization_penalty must be nonnegative.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
        )
        _validate_density_ratio_loss(self.density_ratio_loss, self.logistic_logit_clip)

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralActionRatioConfig":
        params = dict(
            normalization_penalty=10.0,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=400,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralActionRatioConfig":
        params = dict(
            normalization_penalty=1.0,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=800,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralActionRatioConfig":
        params = dict(
            normalization_penalty=0.0,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=1200,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralSourceStateRatioConfig:
    """Tuning for neural initial/source density-ratio nuisance fits."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_steps: int = 400
    validation_fraction: float = 0.2
    patience: int = 20
    min_improvement: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    normalization_penalty: float = 10.0
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.normalization_penalty < 0.0:
            raise ValueError("normalization_penalty must be nonnegative.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
        )
        _validate_density_ratio_loss(self.density_ratio_loss, self.logistic_logit_clip)

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralSourceStateRatioConfig":
        params = dict(
            normalization_penalty=10.0,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=400,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralSourceStateRatioConfig":
        params = dict(
            normalization_penalty=1.0,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=800,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralSourceStateRatioConfig":
        params = dict(
            normalization_penalty=0.0,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=1200,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralTransitionRatioConfig:
    """Tuning for neural transition density-ratio nuisance fits."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_steps: int = 600
    permutation_samples: int = 4
    validation_fraction: float = 0.2
    patience: int = 20
    min_improvement: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    normalization_penalty: float = 10.0
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    device: str = "cpu"
    seed: int = 123

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.permutation_samples <= 0:
            raise ValueError("permutation_samples must be positive.")
        if self.normalization_penalty < 0.0:
            raise ValueError("normalization_penalty must be nonnegative.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
        )
        _validate_density_ratio_loss(self.density_ratio_loss, self.logistic_logit_clip)

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralTransitionRatioConfig":
        params = dict(
            normalization_penalty=10.0,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=600,
            permutation_samples=4,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralTransitionRatioConfig":
        params = dict(
            normalization_penalty=1.0,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=800,
            permutation_samples=16,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralTransitionRatioConfig":
        params = dict(
            normalization_penalty=0.0,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            max_steps=1200,
            permutation_samples=32,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class NeuralOccupancyRegressionConfig:
    """Tuning for neural fixed-point occupancy regression."""

    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "silu"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    num_iterations: int = 30
    gradient_steps_per_iteration: int = 4
    mcmc_samples: int = 24
    initial_ratio: float = 1.0
    loss: str = "huber"
    huber_delta: Optional[float] = None
    huber_delta_scale: float = 1.345
    huber_delta_quantile_power: Optional[float] = 0.25
    huber_delta_min_quantile: float = 0.80
    fixed_point_damping: float = 0.5
    normalize_occupancy: bool = True
    occupancy_ratio_max: Optional[float] = 50.0
    occupancy_projection_eps: float = 1e-12
    clip_pseudo_outcomes: bool = True
    pseudo_outcome_max: Optional[float] = None
    pseudo_outcome_upper_quantile: float = 0.995
    pseudo_outcome_min: float = 0.0
    normalize_transition_cache: bool = False
    transition_cache_norm_eps: float = 1e-12
    occupancy_sample_weight_mode: str = "uniform"
    occupancy_sample_weight_max: Optional[float] = 20.0
    fixed_point_tol: Optional[float] = None
    fixed_point_patience: int = 3
    min_outer_iterations: int = 3
    target_min: Optional[float] = 0.0
    target_max: Optional[float] = None
    early_stopping: bool = True
    validation_fraction: float = 0.2
    min_improvement: float = 1e-6
    patience: int = 10
    validation_warmup_iterations: int = 1
    refresh_on_plateau: bool = True
    refresh_after_plateaus: int = 1
    eval_mcmc_multiplier: int = 5
    eval_seed_offset: int = 777_777
    direct_adjoint_steps: Optional[int] = 128
    direct_adjoint_learning_rate: Optional[float] = 1e-3
    direct_adjoint_weight_decay: Optional[float] = 0.0
    direct_one_step_density_ratio_loss: str = "lsif"
    direct_one_step_prediction_max: Optional[float] = 10.0
    direct_one_step_logistic_logit_clip: Optional[float] = 10.0
    direct_one_step_moment_calibration: str = "scalar"
    direct_one_step_max_steps: Optional[int] = None
    direct_one_step_hidden_dims: Optional[Sequence[int]] = None
    grad_clip_norm: Optional[float] = 5.0
    device: str = "cpu"
    seed: int = 123
    show_progress: bool = False

    def __post_init__(self) -> None:
        _validate_neural_common(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_steps=self.num_iterations,
            validation_fraction=self.validation_fraction,
            patience=self.patience,
            min_improvement=self.min_improvement,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.gradient_steps_per_iteration <= 0:
            raise ValueError("gradient_steps_per_iteration must be positive.")
        if self.mcmc_samples <= 0:
            raise ValueError("mcmc_samples must be positive.")
        if self.initial_ratio <= 0.0:
            raise ValueError("initial_ratio must be positive.")
        self._normalized_loss()
        if self.huber_delta is not None and self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when supplied.")
        if self.huber_delta_scale <= 0.0:
            raise ValueError("huber_delta_scale must be positive.")
        if self.huber_delta_quantile_power is not None and self.huber_delta_quantile_power <= 0.0:
            raise ValueError("huber_delta_quantile_power must be positive when supplied.")
        if not (0.0 < self.huber_delta_min_quantile < 1.0):
            raise ValueError("huber_delta_min_quantile must be in (0, 1).")
        if self.validation_warmup_iterations < 0:
            raise ValueError("validation_warmup_iterations must be nonnegative.")
        if self.direct_adjoint_steps is not None and self.direct_adjoint_steps <= 0:
            raise ValueError("direct_adjoint_steps must be positive when supplied.")
        if self.direct_adjoint_learning_rate is not None and self.direct_adjoint_learning_rate <= 0.0:
            raise ValueError("direct_adjoint_learning_rate must be positive when supplied.")
        if self.direct_adjoint_weight_decay is not None and self.direct_adjoint_weight_decay < 0.0:
            raise ValueError("direct_adjoint_weight_decay must be nonnegative when supplied.")
        _validate_density_ratio_loss(self.direct_one_step_density_ratio_loss, self.direct_one_step_logistic_logit_clip)
        if self.direct_one_step_prediction_max is not None and self.direct_one_step_prediction_max <= 0.0:
            raise ValueError("direct_one_step_prediction_max must be positive when supplied.")
        if str(self.direct_one_step_moment_calibration) not in {"none", "scalar"}:
            raise ValueError("direct_one_step_moment_calibration must be 'none' or 'scalar'.")
        if self.direct_one_step_max_steps is not None and self.direct_one_step_max_steps <= 0:
            raise ValueError("direct_one_step_max_steps must be positive when supplied.")
        if self.direct_one_step_hidden_dims is not None and (
            not tuple(self.direct_one_step_hidden_dims)
            or any(int(width) <= 0 for width in self.direct_one_step_hidden_dims)
        ):
            raise ValueError("direct_one_step_hidden_dims must contain positive widths when supplied.")
        _validate_occupancy_stabilization_config(
            fixed_point_damping=self.fixed_point_damping,
            occupancy_ratio_max=self.occupancy_ratio_max,
            occupancy_projection_eps=self.occupancy_projection_eps,
            pseudo_outcome_max=self.pseudo_outcome_max,
            pseudo_outcome_upper_quantile=self.pseudo_outcome_upper_quantile,
            pseudo_outcome_min=self.pseudo_outcome_min,
            transition_cache_norm_eps=self.transition_cache_norm_eps,
            occupancy_sample_weight_mode=self.occupancy_sample_weight_mode,
            occupancy_sample_weight_max=self.occupancy_sample_weight_max,
            fixed_point_tol=self.fixed_point_tol,
            fixed_point_patience=self.fixed_point_patience,
            min_outer_iterations=self.min_outer_iterations,
        )

    def _normalized_loss(self) -> str:
        aliases = {
            "l2": "squared",
            "mse": "squared",
            "squared_error": "squared",
            "squared": "squared",
            "huber": "huber",
            "robust": "huber",
        }
        normalized = str(self.loss).strip().lower()
        if normalized not in aliases:
            raise ValueError("loss must be 'squared' or 'huber'.")
        return aliases[normalized]

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "NeuralOccupancyRegressionConfig":
        params = dict(
            loss="huber",
            fixed_point_damping=0.5,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            clip_pseudo_outcomes=True,
            pseudo_outcome_upper_quantile=0.995,
            occupancy_sample_weight_mode="uniform",
            direct_adjoint_steps=128,
            direct_one_step_prediction_max=10.0,
            direct_one_step_moment_calibration="scalar",
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "NeuralOccupancyRegressionConfig":
        params = dict(
            num_iterations=80,
            gradient_steps_per_iteration=8,
            mcmc_samples=64,
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=True,
            occupancy_ratio_max=200.0,
            clip_pseudo_outcomes=False,
            occupancy_sample_weight_mode="sqrt_target",
            direct_adjoint_steps=128,
            direct_one_step_prediction_max=200.0,
            direct_one_step_moment_calibration="none",
            learning_rate=5e-4,
            weight_decay=1e-5,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "NeuralOccupancyRegressionConfig":
        params = dict(
            num_iterations=120,
            gradient_steps_per_iteration=12,
            mcmc_samples=96,
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
            occupancy_sample_weight_mode="sqrt_target",
            occupancy_sample_weight_max=None,
            direct_adjoint_steps=256,
            direct_one_step_prediction_max=None,
            direct_one_step_moment_calibration="none",
            learning_rate=5e-4,
            weight_decay=0.0,
        )
        params.update(overrides)
        return cls(**params)


for _cls in (
    NeuralActionRatioConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
    NeuralOccupancyRegressionConfig,
):
    _cls.__module__ = "occupancy_ratio.fit_occupancy_ratio_neural"

del _cls
