"""Boosted estimator configuration dataclasses and presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from occupancy_ratio.stabilization import _normalize_occupancy_loss
from occupancy_ratio.validation import (
    _validate_occupancy_stabilization_config,
    _validate_ratio_prediction_config,
)

__all__ = [
    "ActionRatioConfig",
    "SourceStateRatioConfig",
    "TransitionRatioConfig",
    "OccupancyRegressionConfig",
]


@dataclass
class ActionRatioConfig:
    """Tuning for the first-stage action importance ratio.

    Fits ``pi(a | s) / pi0(a | s)`` from observed behavior actions and sampled
    target-policy actions. Advanced LightGBM options go in ``lgb_params``;
    less common estimator options can still be supplied through ``extra_kwargs``.
    """

    lgb_params: Dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 100
    validation_fraction: float = 0.2
    early_stopping_rounds: int = 10
    refit_on_all_data: bool = False
    clip_nonnegative: bool = True
    hessian_floor: float = 1e-3
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    show_progress: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
            density_ratio_loss=self.density_ratio_loss,
            logistic_logit_clip=self.logistic_logit_clip,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(
            clip_nonneg=self.clip_nonnegative,
            num_boost_round=self.num_boost_round,
            lgb_params=dict(self.lgb_params),
            eps_hess=self.hessian_floor,
            test_size=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
            refit_on_all_data=self.refit_on_all_data,
            show_tqdm=self.show_progress,
            init_score_value=self.initial_ratio,
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            normalize_predictions=self.normalize_predictions,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
            crossfit_seed=self.crossfit_seed,
            density_ratio_loss=self.density_ratio_loss,
            logistic_logit_clip=self.logistic_logit_clip,
        )
        kwargs.update(self.extra_kwargs)
        return kwargs

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "ActionRatioConfig":
        """Construct the conservative nuisance-ratio preset."""
        params = dict(
            refit_on_all_data=False,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "ActionRatioConfig":
        """Construct a less conservative nuisance-ratio preset."""
        params = dict(
            refit_on_all_data=True,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            num_boost_round=300,
            early_stopping_rounds=20,
            lgb_params={
                "num_leaves": 127,
                "min_data_in_leaf": 50,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "ActionRatioConfig":
        """Construct a minimally capped nuisance-ratio preset for DICE comparisons."""
        params = dict(
            refit_on_all_data=True,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            num_boost_round=500,
            early_stopping_rounds=0,
            lgb_params={
                "num_leaves": 255,
                "min_data_in_leaf": 20,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class SourceStateRatioConfig:
    """Tuning for initial/source density-ratio fits.

    In ``initial_ratio_mode="joint"`` this config fits an initial state-action
    ratio on ``(s, a)`` rows. In ``"factored"`` mode it fits the state-only
    source ratio and multiplies by the action ratio separately.
    """

    lgb_params: Dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 100
    validation_fraction: float = 0.2
    early_stopping_rounds: int = 10
    refit_on_all_data: bool = False
    clip_nonnegative: bool = True
    hessian_floor: float = 1e-3
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    show_progress: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
            density_ratio_loss=self.density_ratio_loss,
            logistic_logit_clip=self.logistic_logit_clip,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(
            clip_nonneg=self.clip_nonnegative,
            num_boost_round=self.num_boost_round,
            lgb_params=dict(self.lgb_params),
            eps_hess=self.hessian_floor,
            test_size=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
            refit_on_all_data=self.refit_on_all_data,
            show_tqdm=self.show_progress,
            init_score_value=self.initial_ratio,
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            normalize_predictions=self.normalize_predictions,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
            crossfit_seed=self.crossfit_seed,
            density_ratio_loss=self.density_ratio_loss,
            logistic_logit_clip=self.logistic_logit_clip,
        )
        kwargs.update(self.extra_kwargs)
        return kwargs

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "SourceStateRatioConfig":
        params = dict(
            refit_on_all_data=False,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "SourceStateRatioConfig":
        params = dict(
            refit_on_all_data=True,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            num_boost_round=300,
            early_stopping_rounds=20,
            lgb_params={
                "num_leaves": 127,
                "min_data_in_leaf": 50,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "SourceStateRatioConfig":
        params = dict(
            refit_on_all_data=True,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            num_boost_round=500,
            early_stopping_rounds=0,
            lgb_params={
                "num_leaves": 255,
                "min_data_in_leaf": 20,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class TransitionRatioConfig:
    """Tuning for the first-stage transition density ratio.

    Fits ``P(s_next | s,a) / rho0(s_next)`` with permuted reference states.
    ``permutation_samples`` controls how many reference states are paired with
    each observed transition.
    """

    lgb_params: Dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 300
    permutation_samples: int = 20
    validation_fraction: float = 0.2
    early_stopping_rounds: int = 10
    refit_on_all_data: bool = False
    clip_nonnegative: bool = True
    hessian_floor: float = 1e-3
    initial_ratio: float = 1.0
    prediction_max: Optional[float] = 50.0
    prediction_power: float = 1.0
    normalize_predictions: bool = False
    moment_calibration: str = "none"
    crossfit_folds: int = 1
    crossfit_seed: Optional[int] = None
    density_ratio_loss: str = "lsif"
    logistic_logit_clip: Optional[float] = 20.0
    show_progress: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_ratio_prediction_config(
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
            density_ratio_loss=self.density_ratio_loss,
            logistic_logit_clip=self.logistic_logit_clip,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(
            K_perm=self.permutation_samples,
            clip_nonneg=self.clip_nonnegative,
            num_boost_round=self.num_boost_round,
            lgb_params=dict(self.lgb_params),
            eps_hess=self.hessian_floor,
            test_size=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
            refit_on_all_data=self.refit_on_all_data,
            show_tqdm=self.show_progress,
            init_score_value=self.initial_ratio,
            prediction_max=self.prediction_max,
            prediction_power=self.prediction_power,
            normalize_predictions=self.normalize_predictions,
            moment_calibration=self.moment_calibration,
            crossfit_folds=self.crossfit_folds,
            crossfit_seed=self.crossfit_seed,
            density_ratio_loss=self.density_ratio_loss,
            logistic_logit_clip=self.logistic_logit_clip,
        )
        kwargs.update(self.extra_kwargs)
        return kwargs

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "TransitionRatioConfig":
        params = dict(
            refit_on_all_data=False,
            prediction_max=50.0,
            normalize_predictions=False,
            moment_calibration="none",
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "TransitionRatioConfig":
        params = dict(
            refit_on_all_data=True,
            prediction_max=200.0,
            normalize_predictions=False,
            moment_calibration="none",
            num_boost_round=300,
            early_stopping_rounds=20,
            permutation_samples=50,
            lgb_params={
                "num_leaves": 127,
                "min_data_in_leaf": 50,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "TransitionRatioConfig":
        params = dict(
            refit_on_all_data=True,
            prediction_max=None,
            normalize_predictions=False,
            moment_calibration="none",
            num_boost_round=500,
            early_stopping_rounds=0,
            permutation_samples=100,
            lgb_params={
                "num_leaves": 255,
                "min_data_in_leaf": 20,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class OccupancyRegressionConfig:
    """Tuning for the second-stage occupancy fixed-point regression."""

    lgb_params: Dict[str, Any] = field(default_factory=dict)
    num_iterations: int = 200
    trees_per_iteration: int = 1
    mcmc_samples: int = 80
    batch_size: int = 1000
    initial_ratio: float = 1.0
    loss: str = "huber"
    huber_delta: Optional[float] = None
    huber_delta_scale: float = 1.345
    huber_delta_quantile_power: Optional[float] = 0.25
    huber_delta_min_quantile: float = 0.80
    huber_hessian_floor: float = 1e-3
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
    refresh_on_plateau: bool = True
    refresh_after_plateaus: int = 1
    eval_mcmc_multiplier: int = 5
    eval_seed_offset: int = 777_777
    direct_adjoint_num_boost_round: int = 32
    direct_adjoint_lgb_params: Dict[str, Any] = field(default_factory=dict)
    direct_adjoint_loss: str = "squared"
    direct_adjoint_validation_fraction: float = 0.2
    direct_adjoint_early_stopping_rounds: int = 0
    direct_adjoint_sample_weight_mode: str = "uniform"
    direct_adjoint_sample_weight_max: Optional[float] = 50.0
    seed: int = 123
    show_progress: bool = True

    def __post_init__(self) -> None:
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
        _normalize_occupancy_loss(self.loss)
        if self.huber_delta is not None and self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when supplied.")
        if self.huber_delta_scale <= 0.0:
            raise ValueError("huber_delta_scale must be positive.")
        if self.huber_delta_quantile_power is not None and self.huber_delta_quantile_power <= 0.0:
            raise ValueError("huber_delta_quantile_power must be positive when supplied.")
        if not (0.0 < self.huber_delta_min_quantile < 1.0):
            raise ValueError("huber_delta_min_quantile must be in (0, 1).")
        if self.huber_hessian_floor < 0.0:
            raise ValueError("huber_hessian_floor must be nonnegative.")
        if self.direct_adjoint_num_boost_round <= 0:
            raise ValueError("direct_adjoint_num_boost_round must be positive.")
        _normalize_occupancy_loss(self.direct_adjoint_loss)
        if not (0.0 <= float(self.direct_adjoint_validation_fraction) < 1.0):
            raise ValueError("direct_adjoint_validation_fraction must be in [0, 1).")
        if self.direct_adjoint_early_stopping_rounds < 0:
            raise ValueError("direct_adjoint_early_stopping_rounds must be >= 0.")
        if str(self.direct_adjoint_sample_weight_mode) not in {"uniform", "sqrt_target", "target"}:
            raise ValueError("direct_adjoint_sample_weight_mode must be 'uniform', 'sqrt_target', or 'target'.")
        if self.direct_adjoint_sample_weight_max is not None and self.direct_adjoint_sample_weight_max <= 0.0:
            raise ValueError("direct_adjoint_sample_weight_max must be positive when supplied.")

    @classmethod
    def stable_defaults(cls, **overrides: Any) -> "OccupancyRegressionConfig":
        """Construct the practical stabilization preset for heavy-tailed targets."""
        params = dict(
            loss="huber",
            fixed_point_damping=0.5,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            clip_pseudo_outcomes=True,
            pseudo_outcome_upper_quantile=0.995,
            occupancy_sample_weight_mode="uniform",
            direct_adjoint_num_boost_round=32,
            early_stopping=True,
            patience=10,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def balanced_defaults(cls, **overrides: Any) -> "OccupancyRegressionConfig":
        """Construct a practical less-conservative preset for real-world OPE."""
        params = dict(
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=True,
            occupancy_ratio_max=200.0,
            clip_pseudo_outcomes=False,
            pseudo_outcome_upper_quantile=0.9995,
            occupancy_sample_weight_mode="sqrt_target",
            occupancy_sample_weight_max=50.0,
            direct_adjoint_num_boost_round=128,
            direct_adjoint_early_stopping_rounds=20,
            direct_adjoint_sample_weight_mode="sqrt_target",
            direct_adjoint_sample_weight_max=50.0,
            early_stopping=True,
            patience=20,
            lgb_params={
                "num_leaves": 127,
                "min_data_in_leaf": 50,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
            },
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def dualdice_comparable_defaults(cls, **overrides: Any) -> "OccupancyRegressionConfig":
        """Construct a minimally stabilized preset for apples-to-apples DICE comparisons."""
        params = dict(
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
            occupancy_sample_weight_mode="sqrt_target",
            occupancy_sample_weight_max=None,
            direct_adjoint_num_boost_round=256,
            direct_adjoint_early_stopping_rounds=0,
            direct_adjoint_sample_weight_mode="sqrt_target",
            direct_adjoint_sample_weight_max=None,
            early_stopping=False,
            lgb_params={
                "num_leaves": 255,
                "min_data_in_leaf": 20,
                "learning_rate": 0.05,
            },
        )
        params.update(overrides)
        return cls(**params)


for _cls in (ActionRatioConfig, SourceStateRatioConfig, TransitionRatioConfig, OccupancyRegressionConfig):
    _cls.__module__ = "occupancy_ratio.fit_occupancy_ratio"

del _cls
