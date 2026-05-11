from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from occupancy_ratio.fit_importance_and_transition_ratios import (
    fit_importance_ratio_lgbm,
    fit_state_density_ratio_lgbm,
    fit_transition_ratio_lgbm,
    _postprocess_ratio_predictions,
    _predict_ratio_from_booster,
)


Array = np.ndarray
TargetBuilder = Callable[..., Dict[str, Any]]

__all__ = [
    "ActionRatioConfig",
    "SourceStateRatioConfig",
    "TransitionRatioConfig",
    "OccupancyRegressionConfig",
    "DiscountedOccupancyRatioModel",
    "fit_discounted_occupancy_ratio",
    "tune_discounted_occupancy_ratio_cv",
    "fit_occupancy_ratio_lgbm",
    "make_forward_occupancy_dataset",
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


@dataclass
class DiscountedOccupancyRatioModel:
    """Fitted discounted occupancy ratio with user-facing prediction helpers."""

    occupancy_booster: Optional[lgb.Booster]
    action_ratio_booster: Optional[lgb.Booster]
    transition_ratio_booster: lgb.Booster
    occupancy_initial_ratio: float
    action_ratio_offset: float
    transition_ratio_offset: float
    gamma: float
    state_dim: int
    action_dim: int
    history: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    legacy_result: Dict[str, Any]
    occupancy_normalize: bool = False
    occupancy_ratio_max: Optional[float] = None
    occupancy_projection_eps: float = 1e-12
    occupancy_prediction_scale: Optional[float] = None
    action_prediction_max: Optional[float] = None
    action_prediction_power: float = 1.0
    action_normalize_predictions: bool = False
    action_prediction_scale: float = 1.0
    action_density_ratio_loss: str = "lsif"
    action_logistic_logit_clip: Optional[float] = 20.0
    action_prior_correction: float = 1.0
    occupancy_training_features: Optional[Array] = None
    occupancy_training_predictions: Optional[Array] = None
    action_ratio_training_features: Optional[Array] = None
    action_ratio_training_predictions: Optional[Array] = None

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model with pickle.

        LightGBM boosters are pickle-compatible; this method preserves the
        fitted damped-state prediction cache used by public ``clip=True``
        predictions.
        """
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "DiscountedOccupancyRatioModel":
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"Serialized object is {type(model).__name__}, not {cls.__name__}.")
        return model

    def predict_state_action_ratio(
        self,
        states: Array,
        actions: Array,
        *,
        clip: bool = True,
    ) -> Array:
        """Predict ``rho_pi,gamma(s) * pi(a | s) / pi0(a | s)``."""
        features = self._state_action_features(states, actions)
        raw = np.full(features.shape[0], float(self.occupancy_initial_ratio), dtype=np.float64)
        if self.occupancy_booster is not None:
            raw += self.occupancy_booster.predict(features).astype(np.float64, copy=False)
        if not clip:
            return raw
        projected = _project_nonnegative_normalized(
            raw,
            max_value=self.occupancy_ratio_max,
            normalize=self.occupancy_normalize,
            eps=self.occupancy_projection_eps,
            normalization_scale=self.occupancy_prediction_scale,
        )
        return _replace_known_training_predictions(
            features,
            projected,
            known_features=self.occupancy_training_features,
            known_predictions=self.occupancy_training_predictions,
        )

    def predict_action_ratio(
        self,
        states: Array,
        actions: Array,
        *,
        clip: bool = True,
    ) -> Array:
        """Predict the first-stage action ratio ``pi(a | s) / pi0(a | s)``."""
        features = self._state_action_features(states, actions)
        if self.action_ratio_booster is None:
            if self.action_ratio_training_features is None or self.action_ratio_training_predictions is None:
                raise ValueError("This model was fit with known action ratios and has no action-ratio predictor.")
            out = _lookup_known_training_predictions(
                features,
                known_features=self.action_ratio_training_features,
                known_predictions=self.action_ratio_training_predictions,
            )
            if out is None:
                raise ValueError("Known action-ratio model can only predict exact fitted state-action rows.")
            return out
        raw = _predict_ratio_from_booster(
            booster=self.action_ratio_booster,
            X=features,
            offset=float(self.action_ratio_offset),
            density_ratio_loss=self.action_density_ratio_loss,
            logistic_logit_clip=self.action_logistic_logit_clip,
            prior_correction=self.action_prior_correction,
        )
        if not clip:
            return raw
        processed, _ = _postprocess_ratio_predictions(
            raw,
            clip_nonneg=True,
            prediction_max=self.action_prediction_max,
            prediction_power=self.action_prediction_power,
            normalize_predictions=self.action_normalize_predictions,
        )
        return processed * float(self.action_prediction_scale)

    def predict_state_ratio(
        self,
        states: Array,
        actions: Array,
        *,
        clip: bool = True,
    ) -> Array:
        """Predict ``rho_pi,gamma(s)`` by dividing state-action ratio by action ratio."""
        state_action = self.predict_state_action_ratio(states, actions, clip=clip)
        action = self.predict_action_ratio(states, actions, clip=clip)
        return _safe_divide(state_action, action)

    def predict_for_target_actions(
        self,
        states: Array,
        target_actions: Array,
        *,
        observed_actions: Optional[Array] = None,
        clip: bool = True,
    ) -> Dict[str, Array]:
        """Predict ratios on target actions, optionally also on observed actions."""
        out = dict(
            target_state_action_ratio=self.predict_state_action_ratio(states, target_actions, clip=clip),
            target_action_ratio=self.predict_action_ratio(states, target_actions, clip=clip),
        )
        out["target_state_ratio"] = _safe_divide(
            out["target_state_action_ratio"],
            out["target_action_ratio"],
        )
        if observed_actions is not None:
            out["observed_state_action_ratio"] = self.predict_state_action_ratio(
                states,
                observed_actions,
                clip=clip,
            )
            out["observed_action_ratio"] = self.predict_action_ratio(states, observed_actions, clip=clip)
            out["observed_state_ratio"] = _safe_divide(
                out["observed_state_action_ratio"],
                out["observed_action_ratio"],
            )
        return out

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return the dictionary payload used by the legacy API."""
        return dict(self.legacy_result)

    @classmethod
    def from_legacy_result(
        cls,
        result: Dict[str, Any],
        *,
        gamma: float,
        state_dim: int,
        action_dim: int,
        occupancy_initial_ratio: float,
    ) -> "DiscountedOccupancyRatioModel":
        diagnostics = dict(
            stopped_early=result.get("stopped_early"),
            stop_iter=result.get("stop_iter"),
            trees_used=result.get("trees_used"),
            refresh_count=result.get("refresh_count"),
            mcmc_samples=result.get("mcmc_samples"),
            eval_mcmc_samples=result.get("eval_mcmc_samples"),
            loss=result.get("loss"),
            huber_delta=result.get("huber_delta"),
            huber_delta_scale=result.get("huber_delta_scale"),
            huber_delta_quantile_power=result.get("huber_delta_quantile_power"),
            huber_delta_min_quantile=result.get("huber_delta_min_quantile"),
            fixed_point_damping=result.get("fixed_point_damping"),
            normalize_occupancy=result.get("normalize_occupancy"),
            occupancy_ratio_max=result.get("occupancy_ratio_max"),
            occupancy_prediction_scale=result.get("occupancy_prediction_scale"),
            direct_adjoint_num_boost_round=result.get("direct_adjoint_num_boost_round"),
            direct_adjoint_loss=result.get("direct_adjoint_loss"),
            direct_adjoint_validation_fraction=result.get("direct_adjoint_validation_fraction"),
            direct_adjoint_early_stopping_rounds=result.get("direct_adjoint_early_stopping_rounds"),
            direct_adjoint_sample_weight_mode=result.get("direct_adjoint_sample_weight_mode"),
            direct_adjoint_sample_weight_max=result.get("direct_adjoint_sample_weight_max"),
            action_prediction_max=result.get("iw_prediction_max"),
            action_prediction_power=result.get("iw_prediction_power"),
            action_prediction_scale=result.get("iw_prediction_scale"),
            action_density_ratio_loss=result.get("iw_density_ratio_loss", "lsif"),
            action_logistic_logit_clip=result.get("iw_logistic_logit_clip", 20.0),
            action_prior_correction=result.get("iw_prior_correction", 1.0),
            transition_density_ratio_loss=result.get("k_density_ratio_loss", "lsif"),
            transition_logistic_logit_clip=result.get("k_logistic_logit_clip", 20.0),
            transition_prior_correction=result.get("k_prior_correction", 1.0),
            initial_ratio_mode=result.get("initial_ratio_mode", "factored"),
            one_step_ratio_mode=result.get("one_step_ratio_mode", "factored"),
            initial_joint_ratio_enabled=result.get("initial_joint_ratio_enabled", False),
            initial_joint_ratio_mean=result.get("initial_joint_ratio_mean", 1.0),
            initial_joint_ratio_max=result.get("initial_joint_ratio_max", 1.0),
            initial_joint_ratio_ess_fraction=result.get("initial_joint_ratio_ess_fraction", 1.0),
            initial_joint_ratio_loss=result.get("initial_joint_ratio_loss"),
            initial_joint_ratio_density_ratio_loss=result.get("initial_joint_ratio_density_ratio_loss", "none"),
            initial_joint_ratio_clipped_fraction=result.get("initial_joint_ratio_clipped_fraction", 0.0),
            initial_joint_ratio_query_clipped_fraction=result.get("initial_joint_ratio_query_clipped_fraction", 0.0),
            initial_joint_ratio_prediction_max=result.get("initial_joint_ratio_prediction_max"),
            initial_joint_ratio_prediction_scale=result.get("initial_joint_ratio_prediction_scale", 1.0),
            one_step_direct_ratio_enabled=result.get("one_step_direct_ratio_enabled", False),
            one_step_direct_ratio_mean=result.get("one_step_direct_ratio_mean", 1.0),
            one_step_direct_ratio_max=result.get("one_step_direct_ratio_max", 1.0),
            one_step_direct_ratio_ess_fraction=result.get("one_step_direct_ratio_ess_fraction", 1.0),
            one_step_direct_ratio_loss=result.get("one_step_direct_ratio_loss"),
            one_step_direct_ratio_density_ratio_loss=result.get("one_step_direct_ratio_density_ratio_loss", "none"),
            one_step_direct_ratio_clipped_fraction=result.get("one_step_direct_ratio_clipped_fraction", 0.0),
            one_step_direct_ratio_query_clipped_fraction=result.get("one_step_direct_ratio_query_clipped_fraction", 0.0),
            one_step_direct_ratio_prediction_max=result.get("one_step_direct_ratio_prediction_max"),
            one_step_direct_ratio_prediction_scale=result.get("one_step_direct_ratio_prediction_scale", 1.0),
            source_state_ratio_enabled=result.get("source_state_ratio_enabled", False),
            source_state_ratio_mean=result.get("source_state_ratio_mean", 1.0),
            source_state_ratio_max=result.get("source_state_ratio_max", 1.0),
            source_state_ratio_ess_fraction=result.get("source_state_ratio_ess_fraction", 1.0),
            source_state_ratio_loss=result.get("source_state_ratio_loss"),
            source_state_ratio_density_ratio_loss=result.get("source_state_ratio_density_ratio_loss", "none"),
            source_state_ratio_clipped_fraction=result.get("source_state_ratio_clipped_fraction", 0.0),
            source_state_ratio_query_clipped_fraction=result.get("source_state_ratio_query_clipped_fraction", 0.0),
            source_state_ratio_prediction_max=result.get("source_state_ratio_prediction_max"),
            source_state_ratio_prediction_scale=result.get("source_state_ratio_prediction_scale", 1.0),
            num_target_action_samples=result.get("num_target_action_samples", 1),
            continuation_mean=result.get("continuation_mean", 1.0),
            continuation_min=result.get("continuation_min", 1.0),
            known_action_ratio=result.get("known_action_ratio", False),
        )
        return cls(
            occupancy_booster=result["bst_w"],
            action_ratio_booster=result["bst_iw"],
            transition_ratio_booster=result["bst_k"],
            occupancy_initial_ratio=float(occupancy_initial_ratio),
            action_ratio_offset=float(result.get("iw_prediction_offset", 0.0)),
            transition_ratio_offset=float(result.get("k_prediction_offset", 0.0)),
            gamma=float(gamma),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            history=list(result.get("history", [])),
            diagnostics=diagnostics,
            legacy_result=result,
            occupancy_normalize=bool(result.get("normalize_occupancy", False)),
            occupancy_ratio_max=result.get("occupancy_ratio_max"),
            occupancy_projection_eps=float(result.get("occupancy_projection_eps", 1e-12)),
            occupancy_prediction_scale=result.get("occupancy_prediction_scale"),
            action_prediction_max=result.get("iw_prediction_max"),
            action_prediction_power=float(result.get("iw_prediction_power", 1.0)),
            action_normalize_predictions=bool(result.get("iw_normalize_predictions", False)),
            action_prediction_scale=float(result.get("iw_prediction_scale", 1.0)),
            action_density_ratio_loss=str(result.get("iw_density_ratio_loss", "lsif")),
            action_logistic_logit_clip=result.get("iw_logistic_logit_clip", 20.0),
            action_prior_correction=float(result.get("iw_prior_correction", 1.0)),
            occupancy_training_features=_legacy_training_prediction_features(result),
            occupancy_training_predictions=_legacy_training_predictions(result),
            action_ratio_training_features=result.get("known_action_ratio_features"),
            action_ratio_training_predictions=result.get("known_action_ratio_predictions"),
        )

    def _state_action_features(self, states: Array, actions: Array) -> Array:
        states = _as_2d(states, "states")
        actions = _as_2d(actions, "actions")
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must have the same number of rows.")
        if states.shape[1] != self.state_dim:
            raise ValueError(f"states must have {self.state_dim} columns.")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have {self.action_dim} columns.")
        return np.concatenate([states, actions], axis=1)


def fit_discounted_occupancy_ratio(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    terminals: Optional[Array] = None,
    timeouts: Optional[Array] = None,
    handle_timeouts: str = "nonterminal",
    absorbing_state: bool = False,
    action_ratio_values: Optional[Array] = None,
    behavior_log_prob: Optional[Array] = None,
    target_log_prob: Optional[Array] = None,
    known_action_ratio_clip_max: Optional[float] = None,
    known_action_ratio_normalize: bool = False,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
    occupancy: Optional[OccupancyRegressionConfig] = None,
    action_ratio: Optional[ActionRatioConfig] = None,
    source_state_ratio: Optional[SourceStateRatioConfig] = None,
    transition_ratio: Optional[TransitionRatioConfig] = None,
    action_ratio_booster: Optional[lgb.Booster] = None,
    transition_ratio_booster: Optional[lgb.Booster] = None,
    action_ratio_offset: float = 0.0,
    transition_ratio_offset: float = 0.0,
    _prefit_nuisance: Optional[Dict[str, Any]] = None,
) -> DiscountedOccupancyRatioModel:
    """Fit a discounted occupancy density-ratio model.

    This is the preferred user-facing API. It keeps the common call compact,
    while exposing full stage-specific tuning through config objects and
    allowing prefit nuisance boosters when you want to reuse first-stage fits.
    """
    occupancy = OccupancyRegressionConfig() if occupancy is None else occupancy
    action_ratio = ActionRatioConfig() if action_ratio is None else action_ratio
    source_state_ratio = SourceStateRatioConfig() if source_state_ratio is None else source_state_ratio
    transition_ratio = TransitionRatioConfig() if transition_ratio is None else transition_ratio

    states_2d = _as_2d(states, "states")
    actions_2d = _as_2d(actions, "actions")
    initial_states_2d = None if initial_states is None else _as_2d(initial_states, "initial_states")
    initial_actions_2d = None if initial_actions is None else _as_2d(initial_actions, "initial_actions")
    target_next_actions_arr = None if target_next_actions is None else np.asarray(target_next_actions)
    result = fit_occupancy_ratio_lgbm(
        S=states_2d,
        A=actions_2d,
        S_next=next_states,
        A_pi=target_actions,
        gamma=gamma,
        S_initial=initial_states_2d,
        A_initial=initial_actions_2d,
        initial_weights=initial_weights,
        A_pi_next=target_next_actions_arr,
        terminals=terminals,
        timeouts=timeouts,
        handle_timeouts=handle_timeouts,
        absorbing_state=absorbing_state,
        action_ratio_values=action_ratio_values,
        behavior_log_prob=behavior_log_prob,
        target_log_prob=target_log_prob,
        known_action_ratio_clip_max=known_action_ratio_clip_max,
        known_action_ratio_normalize=known_action_ratio_normalize,
        initial_ratio_mode=initial_ratio_mode,
        one_step_ratio_mode=one_step_ratio_mode,
        num_outer_iters=occupancy.num_iterations,
        inner_num_boost_round=occupancy.trees_per_iteration,
        mcmc_samples=occupancy.mcmc_samples,
        seed=occupancy.seed,
        batch_query=occupancy.batch_size,
        lgb_params=dict(occupancy.lgb_params),
        clip_y_min=occupancy.target_min,
        clip_y_max=occupancy.target_max,
        k_kwargs=transition_ratio.to_kwargs(),
        iw_kwargs=action_ratio.to_kwargs(),
        source_kwargs=source_state_ratio.to_kwargs(),
        bst_k_init=transition_ratio_booster,
        bst_iw_init=action_ratio_booster,
        bst_k_init_offset=transition_ratio_offset,
        bst_iw_init_offset=action_ratio_offset,
        _prefit_nuisance=_prefit_nuisance,
        w_init=occupancy.initial_ratio,
        loss=occupancy.loss,
        huber_delta=occupancy.huber_delta,
        huber_delta_scale=occupancy.huber_delta_scale,
        huber_delta_quantile_power=occupancy.huber_delta_quantile_power,
        huber_delta_min_quantile=occupancy.huber_delta_min_quantile,
        huber_hessian_floor=occupancy.huber_hessian_floor,
        fixed_point_damping=occupancy.fixed_point_damping,
        normalize_occupancy=occupancy.normalize_occupancy,
        occupancy_ratio_max=occupancy.occupancy_ratio_max,
        occupancy_projection_eps=occupancy.occupancy_projection_eps,
        clip_pseudo_outcomes=occupancy.clip_pseudo_outcomes,
        pseudo_outcome_max=occupancy.pseudo_outcome_max,
        pseudo_outcome_upper_quantile=occupancy.pseudo_outcome_upper_quantile,
        pseudo_outcome_min=occupancy.pseudo_outcome_min,
        normalize_transition_cache=occupancy.normalize_transition_cache,
        transition_cache_norm_eps=occupancy.transition_cache_norm_eps,
        occupancy_sample_weight_mode=occupancy.occupancy_sample_weight_mode,
        occupancy_sample_weight_max=occupancy.occupancy_sample_weight_max,
        fixed_point_tol=occupancy.fixed_point_tol,
        fixed_point_patience=occupancy.fixed_point_patience,
        min_outer_iterations=occupancy.min_outer_iterations,
        early_stopping=occupancy.early_stopping,
        test_frac=occupancy.validation_fraction,
        early_stopping_min_delta=occupancy.min_improvement,
        early_stopping_patience=occupancy.patience,
        refresh_on_plateau=occupancy.refresh_on_plateau,
        refresh_after_n_plateau=occupancy.refresh_after_plateaus,
        eval_mcmc_multiplier=occupancy.eval_mcmc_multiplier,
        eval_seed_offset=occupancy.eval_seed_offset,
        direct_adjoint_num_boost_round=occupancy.direct_adjoint_num_boost_round,
        direct_adjoint_lgb_params=dict(occupancy.direct_adjoint_lgb_params),
        direct_adjoint_loss=occupancy.direct_adjoint_loss,
        direct_adjoint_validation_fraction=occupancy.direct_adjoint_validation_fraction,
        direct_adjoint_early_stopping_rounds=occupancy.direct_adjoint_early_stopping_rounds,
        direct_adjoint_sample_weight_mode=occupancy.direct_adjoint_sample_weight_mode,
        direct_adjoint_sample_weight_max=occupancy.direct_adjoint_sample_weight_max,
        show_progress=occupancy.show_progress,
    )
    return DiscountedOccupancyRatioModel.from_legacy_result(
        result,
        gamma=gamma,
        state_dim=states_2d.shape[1],
        action_dim=actions_2d.shape[1],
        occupancy_initial_ratio=occupancy.initial_ratio,
    )


def tune_discounted_occupancy_ratio_cv(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
    occupancy: Optional[OccupancyRegressionConfig] = None,
    action_ratio: Optional[ActionRatioConfig] = None,
    source_state_ratio: Optional[SourceStateRatioConfig] = None,
    transition_ratio: Optional[TransitionRatioConfig] = None,
    occupancy_grid: Optional[Sequence[Dict[str, Any]]] = None,
    action_ratio_grid: Optional[Sequence[Dict[str, Any]]] = None,
    transition_ratio_grid: Optional[Sequence[Dict[str, Any]]] = None,
    cv_folds: int = 3,
    scoring: str = "composite",
    lambda_norm: float = 0.1,
    lambda_tail: float = 0.01,
    seed: int = 123,
    fit_final: bool = True,
) -> Dict[str, Any]:
    """Tune nuisance and occupancy configs with lightweight cross-validation.

    Nuisance stages use held-out LSIF losses. Occupancy candidates run their
    normal inner early stopping on each training fold, then score held-out
    predictions with the selected validation loss plus normalization and tail
    penalties when ``scoring='composite'``.
    """
    if int(cv_folds) < 2:
        raise ValueError("cv_folds must be >= 2.")
    if str(scoring) not in {"composite", "loss"}:
        raise ValueError("scoring must be 'composite' or 'loss'.")
    if lambda_norm < 0.0 or lambda_tail < 0.0:
        raise ValueError("lambda_norm and lambda_tail must be nonnegative.")

    S = _as_2d(states, "states")
    A = _as_2d(actions, "actions")
    S_next = _as_2d(next_states, "next_states")
    A_pi = _as_2d(target_actions, "target_actions")
    S_initial = None if initial_states is None else _as_2d(initial_states, "initial_states")
    A_initial = None if initial_actions is None else _as_2d(initial_actions, "initial_actions")
    A_pi_next = None if target_next_actions is None else _as_2d(target_next_actions, "target_next_actions")
    _validate_aligned_inputs(S=S, A=A, S_next=S_next, A_pi=A_pi)
    _validate_initial_state_inputs(S=S, S_initial=S_initial, initial_weights=initial_weights)
    _validate_initial_action_inputs(A=A, S_initial=S_initial, A_initial=A_initial)
    _validate_next_target_actions(A=A, S=S, A_pi_next=A_pi_next)
    resolved_initial_mode = _resolve_initial_ratio_mode(initial_ratio_mode, S_initial=S_initial, A_initial=A_initial)
    resolved_one_step_mode = _resolve_one_step_ratio_mode(one_step_ratio_mode, A_pi_next=A_pi_next)
    folds = _make_fold_indices(S.shape[0], int(cv_folds), int(seed))
    base_occ = OccupancyRegressionConfig() if occupancy is None else occupancy
    base_iw = ActionRatioConfig() if action_ratio is None else action_ratio
    base_source = SourceStateRatioConfig() if source_state_ratio is None else source_state_ratio
    base_k = TransitionRatioConfig() if transition_ratio is None else transition_ratio

    action_candidates = _candidate_configs(base_iw, action_ratio_grid)
    transition_candidates = _candidate_configs(base_k, transition_ratio_grid)
    occupancy_candidates = _candidate_configs(base_occ, occupancy_grid)

    action_scores = [
        _score_action_config_cv(S=S, A=A, A_pi=A_pi, folds=folds, config=cfg, seed=seed)
        for cfg in action_candidates
    ]
    best_action_idx = int(np.argmin([row["score"] for row in action_scores]))
    selected_action = action_candidates[best_action_idx]

    transition_scores = [
        _score_transition_config_cv(S=S, A=A, S_next=S_next, folds=folds, config=cfg, seed=seed)
        for cfg in transition_candidates
    ]
    best_transition_idx = int(np.argmin([row["score"] for row in transition_scores]))
    selected_transition = transition_candidates[best_transition_idx]

    occupancy_scores = [
        _score_occupancy_config_cv(
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            gamma=gamma,
            folds=folds,
            occupancy=cfg,
            action_ratio=selected_action,
            source_state_ratio=base_source,
            transition_ratio=selected_transition,
            scoring=scoring,
            lambda_norm=lambda_norm,
            lambda_tail=lambda_tail,
            seed=seed,
            initial_states=S_initial,
            initial_actions=A_initial,
            initial_weights=initial_weights,
            target_next_actions=A_pi_next,
            initial_ratio_mode=resolved_initial_mode,
            one_step_ratio_mode=resolved_one_step_mode,
        )
        for cfg in occupancy_candidates
    ]
    best_occupancy_idx = int(np.argmin([row["score"] for row in occupancy_scores]))
    selected_occupancy = occupancy_candidates[best_occupancy_idx]

    model = None
    if fit_final:
        model = fit_discounted_occupancy_ratio(
            states=S,
            actions=A,
            next_states=S_next,
            target_actions=A_pi,
            gamma=gamma,
            initial_states=S_initial,
            initial_actions=A_initial,
            initial_weights=initial_weights,
            target_next_actions=A_pi_next,
            initial_ratio_mode=resolved_initial_mode,
            one_step_ratio_mode=resolved_one_step_mode,
            occupancy=selected_occupancy,
            action_ratio=selected_action,
            source_state_ratio=base_source,
            transition_ratio=selected_transition,
        )

    return dict(
        selected_occupancy=selected_occupancy,
        selected_action_ratio=selected_action,
        selected_source_state_ratio=base_source,
        selected_transition_ratio=selected_transition,
        selected_indices=dict(
            occupancy=best_occupancy_idx,
            action_ratio=best_action_idx,
            transition_ratio=best_transition_idx,
        ),
        action_scores=action_scores,
        transition_scores=transition_scores,
        occupancy_scores=occupancy_scores,
        scoring=str(scoring),
        lambda_norm=float(lambda_norm),
        lambda_tail=float(lambda_tail),
        model=model,
    )


def fit_occupancy_ratio_lgbm(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    gamma: float,
    S_initial: Optional[Array] = None,
    A_initial: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    A_pi_next: Optional[Array] = None,
    terminals: Optional[Array] = None,
    timeouts: Optional[Array] = None,
    handle_timeouts: str = "nonterminal",
    absorbing_state: bool = False,
    action_ratio_values: Optional[Array] = None,
    behavior_log_prob: Optional[Array] = None,
    target_log_prob: Optional[Array] = None,
    known_action_ratio_clip_max: Optional[float] = None,
    known_action_ratio_normalize: bool = False,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
    num_outer_iters: int = 200,
    inner_num_boost_round: int = 1,
    mcmc_samples: int = 80,
    seed: int = 123,
    batch_query: int = 1000,
    lgb_params: Optional[Dict[str, Any]] = None,
    clip_y_min: Optional[float] = 0.0,
    clip_y_max: Optional[float] = None,
    k_lgb_params: Optional[Dict[str, Any]] = None,
    iw_lgb_params: Optional[Dict[str, Any]] = None,
    k_kwargs: Optional[Dict[str, Any]] = None,
    iw_kwargs: Optional[Dict[str, Any]] = None,
    source_kwargs: Optional[Dict[str, Any]] = None,
    bst_k_init: Optional[lgb.Booster] = None,
    bst_iw_init: Optional[lgb.Booster] = None,
    bst_k_init_offset: float = 0.0,
    bst_iw_init_offset: float = 0.0,
    _prefit_nuisance: Optional[Dict[str, Any]] = None,
    w_init: float = 1.0,
    loss: str = "huber",
    huber_delta: Optional[float] = None,
    huber_delta_scale: float = 1.345,
    huber_delta_quantile_power: Optional[float] = 0.25,
    huber_delta_min_quantile: float = 0.80,
    huber_hessian_floor: float = 1e-3,
    fixed_point_damping: float = 0.5,
    normalize_occupancy: bool = True,
    occupancy_ratio_max: Optional[float] = 50.0,
    occupancy_projection_eps: float = 1e-12,
    clip_pseudo_outcomes: bool = True,
    pseudo_outcome_max: Optional[float] = None,
    pseudo_outcome_upper_quantile: float = 0.995,
    pseudo_outcome_min: float = 0.0,
    normalize_transition_cache: bool = False,
    transition_cache_norm_eps: float = 1e-12,
    occupancy_sample_weight_mode: str = "uniform",
    occupancy_sample_weight_max: Optional[float] = 20.0,
    fixed_point_tol: Optional[float] = None,
    fixed_point_patience: int = 3,
    min_outer_iterations: int = 3,
    early_stopping: bool = True,
    test_frac: float = 0.2,
    early_stopping_min_delta: float = 1e-6,
    early_stopping_patience: int = 10,
    refresh_on_plateau: bool = True,
    refresh_after_n_plateau: int = 1,
    eval_mcmc_multiplier: int = 5,
    eval_seed_offset: int = 777_777,
    direct_adjoint_num_boost_round: int = 32,
    direct_adjoint_lgb_params: Optional[Dict[str, Any]] = None,
    direct_adjoint_loss: str = "squared",
    direct_adjoint_validation_fraction: float = 0.2,
    direct_adjoint_early_stopping_rounds: int = 0,
    direct_adjoint_sample_weight_mode: str = "uniform",
    direct_adjoint_sample_weight_max: Optional[float] = 50.0,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Estimate a discounted state-action occupancy density ratio with LightGBM.

    The target is the normalized discounted ratio

        d_pi,gamma(s,a) = rho_pi,gamma(s) * pi(a | s) / pi0(a | s),

    evaluated on both target-policy action rows ``(S, A_pi)`` and observed
    behavior rows ``(S, A)``. The estimator has two stages:

    1. Fit nuisance density ratios with LSIF-style objectives:
       ``k_hat(s,a,s') = P(s' | s,a) / rho0(s')`` and
       ``iota_hat(s,a) = pi(a | s) / pi0(a | s)``.
    2. Regress the fixed-point target
       ``(1-gamma) * source_hat(s,a)
       + gamma * c_hat(s,a) * E[d_old(S',A') | S'=s, A'=a]``
       in direct mode, or the corresponding factored source/action/transition
       target in factored mode, using incremental LightGBM trees. Set
       ``loss="huber"`` to fit this heavy-tailed
       pseudo-outcome target with a robust M-estimation loss. A fixed
       ``huber_delta`` identifies the conditional Huber location, not the exact
       conditional mean; ``huber_delta=None`` uses an adaptive threshold
       proportional to
       ``MAD(residual) * sqrt(n / log(n))``, capped at the
       ``1 - n^{-1/4}`` empirical residual quantile. Both pieces grow with the
       effective training sample size, so the population target approaches the
       squared-loss conditional mean while still limiting extreme finite-sample
       pseudo-outcomes.

    Returned ``pred_beh`` and ``pred_pi`` are raw model predictions for
    compatibility. Use ``pred_state_action_ratio_*`` for nonnegative clipped
    density-ratio estimates intended for downstream weighting.
    """
    _validate_fit_args(
        gamma=gamma,
        inner_num_boost_round=inner_num_boost_round,
        mcmc_samples=mcmc_samples,
        batch_query=batch_query,
        early_stopping=early_stopping,
        test_frac=test_frac,
        early_stopping_patience=early_stopping_patience,
        refresh_after_n_plateau=refresh_after_n_plateau,
        eval_mcmc_multiplier=eval_mcmc_multiplier,
        loss=loss,
        huber_delta=huber_delta,
        huber_delta_scale=huber_delta_scale,
        huber_delta_quantile_power=huber_delta_quantile_power,
        huber_delta_min_quantile=huber_delta_min_quantile,
        huber_hessian_floor=huber_hessian_floor,
        fixed_point_damping=fixed_point_damping,
        occupancy_ratio_max=occupancy_ratio_max,
        occupancy_projection_eps=occupancy_projection_eps,
        pseudo_outcome_max=pseudo_outcome_max,
        pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
        pseudo_outcome_min=pseudo_outcome_min,
        transition_cache_norm_eps=transition_cache_norm_eps,
        occupancy_sample_weight_mode=occupancy_sample_weight_mode,
        occupancy_sample_weight_max=occupancy_sample_weight_max,
        fixed_point_tol=fixed_point_tol,
        fixed_point_patience=fixed_point_patience,
        min_outer_iterations=min_outer_iterations,
        direct_adjoint_num_boost_round=direct_adjoint_num_boost_round,
        direct_adjoint_loss=direct_adjoint_loss,
        direct_adjoint_validation_fraction=direct_adjoint_validation_fraction,
        direct_adjoint_early_stopping_rounds=direct_adjoint_early_stopping_rounds,
        direct_adjoint_sample_weight_mode=direct_adjoint_sample_weight_mode,
        direct_adjoint_sample_weight_max=direct_adjoint_sample_weight_max,
    )
    loss = _normalize_occupancy_loss(loss)

    S = _as_2d(S, "S")
    A = _as_2d(A, "A")
    S_next = _as_2d(S_next, "S_next")
    A_pi_info = _prepare_target_action_samples(S, A, A_pi, name="A_pi")
    A_pi = A_pi_info["actions"]
    S_pi = A_pi_info["states"]
    target_row_index = A_pi_info["row_index"]
    num_target_action_samples = int(A_pi_info["num_samples"])
    S_initial = None if S_initial is None else _as_2d(S_initial, "S_initial")
    A_initial = None if A_initial is None else _as_2d(A_initial, "A_initial")
    A_pi_next_info = None if A_pi_next is None else _prepare_target_action_samples(S_next, A, A_pi_next, name="A_pi_next")
    A_pi_next = None if A_pi_next_info is None else A_pi_next_info["actions"]
    S_next_pi = None if A_pi_next_info is None else A_pi_next_info["states"]
    next_target_row_index = None if A_pi_next_info is None else A_pi_next_info["row_index"]
    _validate_base_transition_inputs(S=S, A=A, S_next=S_next)
    _validate_target_action_rows(A=A, A_pi=A_pi, name="A_pi")
    if A_pi_next is not None:
        _validate_target_action_rows(A=A, A_pi=A_pi_next, name="A_pi_next")
    _validate_initial_state_inputs(S=S, S_initial=S_initial, initial_weights=initial_weights)
    _validate_initial_action_inputs(A=A, S_initial=S_initial, A_initial=A_initial)
    if A_pi_next_info is not None and int(A_pi_next_info["num_samples"]) != num_target_action_samples:
        raise ValueError("target_next_actions must have the same number of target-action samples as target_actions.")
    resolved_initial_mode = _resolve_initial_ratio_mode(
        initial_ratio_mode,
        S_initial=S_initial,
        A_initial=A_initial,
    )
    resolved_one_step_mode = _resolve_one_step_ratio_mode(one_step_ratio_mode, A_pi_next=A_pi_next)

    n = S.shape[0]
    X_sa_beh = np.concatenate([S, A], axis=1)
    X_sa_pi = np.concatenate([S_pi, A_pi], axis=1)
    X_sa_query = np.vstack([X_sa_pi, X_sa_beh])
    X_s_query = np.vstack([S_pi, S])
    X_sa_initial = None if S_initial is None or A_initial is None else np.concatenate([S_initial, A_initial], axis=1)
    X_sa_next_pi = None if A_pi_next is None or S_next_pi is None else np.concatenate([S_next_pi, A_pi_next], axis=1)
    q = X_sa_query.shape[0]
    continuation = _resolve_continuation(
        terminals=terminals,
        timeouts=timeouts,
        handle_timeouts=handle_timeouts,
        absorbing_state=absorbing_state,
        n_rows=n,
    )
    continuation_query = np.concatenate([continuation[target_row_index], continuation])
    known_iw_beh, known_iw_query = _resolve_known_action_ratio_inputs(
        action_ratio_values=action_ratio_values,
        behavior_log_prob=behavior_log_prob,
        target_log_prob=target_log_prob,
        n_rows=n,
        q_rows=q,
        n_target_rows=X_sa_pi.shape[0],
        prediction_max=known_action_ratio_clip_max,
        normalize=known_action_ratio_normalize,
    )

    nuisance_kwargs = _prepare_nuisance_kwargs(
        lgb_params=lgb_params,
        k_lgb_params=k_lgb_params,
        iw_lgb_params=iw_lgb_params,
        k_kwargs=k_kwargs,
        iw_kwargs=iw_kwargs,
        source_kwargs=source_kwargs,
    )
    prefit_nuisance = {} if _prefit_nuisance is None else dict(_prefit_nuisance)
    if prefit_nuisance:
        k_fit = prefit_nuisance["transition_fit"]
        bst_k = k_fit["bst_k"]
        k_prediction_offset = float(k_fit.get("prediction_offset", 0.0))
        iw_fit = prefit_nuisance["action_fit"]
        bst_iw = iw_fit["bst_iw"]
        iw_prediction_offset = float(iw_fit.get("prediction_offset", 0.0))
        iw_hat_beh = _predict_processed_nuisance(fit=iw_fit, X=X_sa_beh, kind="iw")
        source_fit = prefit_nuisance.get("source_fit")
        source_weight_query = _checked_optional_query_vector(
            prefit_nuisance.get("source_weight_query"),
            q,
            "source_weight_query",
        )
        source_state_query = _checked_optional_query_vector(
            prefit_nuisance.get("source_state_query"),
            q,
            "source_state_query",
        )
        source_diagnostics = dict(prefit_nuisance.get("source_diagnostics", _source_state_ratio_diagnostics(None, None)))
    else:
        bst_k, k_fit, k_prediction_offset = _fit_or_use_transition_ratio(
            S=S,
            A=A,
            S_next=S_next,
            seed=seed,
            bst_k_init=bst_k_init,
            bst_k_init_offset=bst_k_init_offset,
            k_kwargs=nuisance_kwargs["k_kwargs"],
        )
        bst_iw, iw_fit, iw_hat_beh, iw_prediction_offset = _fit_or_use_importance_ratio(
            S=S,
            A=A,
            A_pi=A_pi,
            S_pi=S_pi,
            target_row_index=target_row_index,
            X_sa_beh=X_sa_beh,
            X_sa_query=X_sa_query,
            seed=seed,
            bst_iw_init=bst_iw_init,
            bst_iw_init_offset=bst_iw_init_offset,
            known_iw_beh=known_iw_beh,
            known_iw_query=known_iw_query,
            iw_kwargs=nuisance_kwargs["iw_kwargs"],
        )
        source_fit, source_weight_query, source_state_query, source_diagnostics = _fit_initial_ratio(
            S=S,
            X_sa_beh=X_sa_beh,
            S_query=X_s_query,
            X_sa_query=X_sa_query,
            S_initial=S_initial,
            X_sa_initial=X_sa_initial,
            initial_weights=initial_weights,
            seed=seed,
            source_kwargs=nuisance_kwargs["source_kwargs"],
            initial_ratio_mode=resolved_initial_mode,
        )
    if source_weight_query is None:
        source_weight_query = _make_factored_initial_source_weights(
            bst_iw=bst_iw,
            iw_fit=iw_fit,
            iw_kwargs=nuisance_kwargs["iw_kwargs"],
            iw_prediction_offset=iw_prediction_offset,
            X_sa_query=X_sa_query,
            source_state_query=source_state_query,
            known_iw_query=known_iw_query,
        )
    if prefit_nuisance:
        c_fit = prefit_nuisance.get("c_fit")
        c_ratio_query = _checked_optional_query_vector(prefit_nuisance.get("c_ratio_query"), q, "c_ratio_query")
        c_diagnostics = dict(prefit_nuisance.get("c_diagnostics", _one_step_direct_ratio_diagnostics(None, None)))
    else:
        c_fit, c_ratio_query, c_diagnostics = _fit_direct_one_step_ratio(
            X_ref=X_sa_beh,
            X_next_pi=X_sa_next_pi,
            X_query=X_sa_query,
            seed=seed,
            source_kwargs=nuisance_kwargs["source_kwargs"],
            one_step_ratio_mode=resolved_one_step_mode,
        )
    crossfit_context = None
    if not prefit_nuisance and num_target_action_samples == 1 and known_iw_query is None:
        crossfit_context = _fit_crossfit_nuisance_context(
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            X_sa_beh=X_sa_beh,
            seed=seed,
            bst_k_final=bst_k,
            bst_iw_final=bst_iw,
            k_fit_final=k_fit,
            iw_fit_final=iw_fit,
            k_prediction_offset=k_prediction_offset,
            iw_prediction_offset=iw_prediction_offset,
            k_kwargs=nuisance_kwargs["k_kwargs"],
            iw_kwargs=nuisance_kwargs["iw_kwargs"],
            bst_k_init=bst_k_init,
            bst_iw_init=bst_iw_init,
        )

    train_idx, test_idx = _make_train_test_indices(
        n_rows=q,
        early_stopping=early_stopping,
        test_frac=test_frac,
        seed=seed + 9871,
    )
    X_train = X_sa_query[train_idx]

    params_base = _default_occupancy_lgb_params(seed=seed)
    if lgb_params is not None:
        params_base.update(dict(lgb_params))
    learning_rate = float(params_base.get("learning_rate", 0.1))

    refresh_count = 0

    def make_builder(seed_for_builder: int, mcmc_for_builder: int) -> TargetBuilder:
        if resolved_one_step_mode == "direct":
            return make_direct_adjoint_occupancy_dataset(
                X_sa_successor=X_sa_next_pi,
                X_sa_query=X_sa_query,
                c_ratio_query=c_ratio_query,
                w_source_query=source_weight_query,
                continuation_query=continuation_query,
                successor_row_index=next_target_row_index,
                gamma=gamma,
                seed=int(seed_for_builder),
                num_boost_round=max(1, int(direct_adjoint_num_boost_round)),
                lgb_params=dict(params_base) | ({} if direct_adjoint_lgb_params is None else dict(direct_adjoint_lgb_params)),
                loss=direct_adjoint_loss,
                validation_fraction=float(direct_adjoint_validation_fraction),
                early_stopping_rounds=int(direct_adjoint_early_stopping_rounds),
                sample_weight_mode=str(direct_adjoint_sample_weight_mode),
                sample_weight_max=direct_adjoint_sample_weight_max,
            )
        if crossfit_context is not None:
            return make_crossfit_forward_occupancy_dataset(
                crossfit_context=crossfit_context,
                X_sa_kernel=X_sa_beh,
                X_s_query=X_s_query,
                X_sa_iw=X_sa_beh,
                X_sa_query_iw=X_sa_query,
                gamma=gamma,
                mcmc_samples=int(mcmc_for_builder),
                seed=int(seed_for_builder),
                batch_query=int(batch_query),
                clip_w_query_max=nuisance_kwargs["iw_kwargs"].get("prediction_max"),
                action_prediction_power=float(nuisance_kwargs["iw_kwargs"].get("prediction_power", 1.0)),
                action_normalize_predictions=bool(nuisance_kwargs["iw_kwargs"].get("normalize_predictions", False)),
                action_density_ratio_loss=str(nuisance_kwargs["iw_kwargs"].get("density_ratio_loss", "lsif")),
                action_logistic_logit_clip=nuisance_kwargs["iw_kwargs"].get("logistic_logit_clip", 20.0),
                clip_k_max=nuisance_kwargs["k_kwargs"].get("prediction_max"),
                transition_prediction_power=float(nuisance_kwargs["k_kwargs"].get("prediction_power", 1.0)),
                transition_normalize_predictions=bool(nuisance_kwargs["k_kwargs"].get("normalize_predictions", False)),
                transition_density_ratio_loss=str(nuisance_kwargs["k_kwargs"].get("density_ratio_loss", "lsif")),
                transition_logistic_logit_clip=nuisance_kwargs["k_kwargs"].get("logistic_logit_clip", 20.0),
                normalize_transition_cache=bool(normalize_transition_cache),
                transition_cache_norm_eps=float(transition_cache_norm_eps),
                w_source_query=source_weight_query if resolved_initial_mode == "joint" else None,
                source_state_ratio_query=source_state_query if resolved_initial_mode == "factored" else None,
                continuation_query=continuation_query,
                w_query_override=known_iw_query,
            )
        return make_forward_occupancy_dataset(
            bst_k=bst_k,
            bst_iw=bst_iw,
            k_prediction_offset=k_prediction_offset,
            iw_prediction_offset=iw_prediction_offset,
            X_sa_kernel=X_sa_beh,
            X_s_query=X_s_query,
            X_sa_iw=X_sa_beh,
            X_sa_query_iw=X_sa_query,
            gamma=gamma,
            mcmc_samples=int(mcmc_for_builder),
            seed=int(seed_for_builder),
            batch_query=int(batch_query),
            clip_w_query_max=nuisance_kwargs["iw_kwargs"].get("prediction_max"),
            action_prediction_power=float(nuisance_kwargs["iw_kwargs"].get("prediction_power", 1.0)),
            action_normalize_predictions=bool(nuisance_kwargs["iw_kwargs"].get("normalize_predictions", False)),
            action_prediction_scale=_nuisance_prediction_scale(iw_fit),
            action_density_ratio_loss=str((iw_fit or {}).get("density_ratio_loss", "lsif")),
            action_logistic_logit_clip=(iw_fit or {}).get("logistic_logit_clip", 20.0),
            action_prior_correction=float((iw_fit or {}).get("prior_correction", 1.0)),
            clip_k_max=nuisance_kwargs["k_kwargs"].get("prediction_max"),
            transition_prediction_power=float(nuisance_kwargs["k_kwargs"].get("prediction_power", 1.0)),
            transition_normalize_predictions=bool(nuisance_kwargs["k_kwargs"].get("normalize_predictions", False)),
            transition_prediction_scale=_nuisance_prediction_scale(k_fit),
            transition_density_ratio_loss=str((k_fit or {}).get("density_ratio_loss", "lsif")),
            transition_logistic_logit_clip=(k_fit or {}).get("logistic_logit_clip", 20.0),
            transition_prior_correction=float((k_fit or {}).get("prior_correction", 1.0)),
            normalize_transition_cache=bool(normalize_transition_cache),
            transition_cache_norm_eps=float(transition_cache_norm_eps),
            w_source_query=source_weight_query if resolved_initial_mode == "joint" else None,
            source_state_ratio_query=source_state_query if resolved_initial_mode == "factored" else None,
            continuation_query=continuation_query,
            w_query_override=known_iw_query,
        )

    def make_train_builder() -> TargetBuilder:
        nonlocal refresh_count
        refresh_count += 1
        return make_builder(seed_for_builder=seed + 10_000 * refresh_count, mcmc_for_builder=mcmc_samples)

    build_train = make_train_builder()
    eval_mcmc = int(max(1, mcmc_samples * eval_mcmc_multiplier))
    build_eval = (
        make_builder(seed_for_builder=seed + eval_seed_offset, mcmc_for_builder=eval_mcmc)
        if early_stopping
        else None
    )

    pred_query_raw = np.full(q, float(w_init), dtype=np.float64)
    pred_beh_raw = np.full(n, float(w_init), dtype=np.float64)
    pred_query_state = _project_nonnegative_normalized(
        pred_query_raw,
        max_value=occupancy_ratio_max,
        normalize=normalize_occupancy,
        eps=occupancy_projection_eps,
    )
    pred_beh_state = _project_nonnegative_normalized(
        pred_beh_raw,
        max_value=occupancy_ratio_max,
        normalize=normalize_occupancy,
        eps=occupancy_projection_eps,
    )
    current_model: Optional[lgb.Booster] = None
    boost_iteration = 0
    trees_used = 0
    patience = 0
    plateau_streak = 0
    stopped_early = False
    stop_iter: Optional[int] = None
    fixed_point_stop_streak = 0
    stop_reason: Optional[str] = None
    history: List[Dict[str, Any]] = []

    iterator = range(num_outer_iters)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Occupancy-ratio boosting",
            leave=True,
            dynamic_ncols=False,
            ncols=170,
        )

    for iteration in iterator:
        _check_prediction_cache(
            current_model=current_model,
            X_query=X_sa_query,
            X_beh=X_sa_beh,
            pred_query_raw=pred_query_raw,
            pred_beh_raw=pred_beh_raw,
            w_init=w_init,
            iteration=iteration,
        )

        out_train = build_train(
            w_beh=pred_beh_state,
            w_old_query=pred_query_state.astype(np.float32, copy=False),
            eta=1.0,
            clip_y_min=None,
            clip_y_max=None,
        )
        target_train, target_train_diag = _make_stabilized_fixed_point_target(
            raw_target=out_train["y"],
            current=pred_query_state,
            eta=1.0,
            normalize=normalize_occupancy,
            occupancy_ratio_max=occupancy_ratio_max,
            eps=occupancy_projection_eps,
            clip_pseudo_outcomes=clip_pseudo_outcomes,
            pseudo_outcome_max=pseudo_outcome_max,
            pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
            pseudo_outcome_min=pseudo_outcome_min,
            target_min=clip_y_min,
            target_max=clip_y_max,
        )
        y_train = target_train[train_idx]
        sample_weights, sample_weight_diag = _make_occupancy_sample_weights(
            mode=occupancy_sample_weight_mode,
            action_ratio=out_train.get("w_query"),
            target=target_train,
            max_value=occupancy_sample_weight_max,
        )
        train_weight = None if occupancy_sample_weight_mode == "uniform" else sample_weights[train_idx]
        train_resid = pred_query_state[train_idx] - y_train
        loss_delta = _resolve_huber_delta(
            train_resid,
            loss=loss,
            huber_delta=huber_delta,
            huber_delta_scale=huber_delta_scale,
            huber_delta_quantile_power=huber_delta_quantile_power,
            huber_delta_min_quantile=huber_delta_min_quantile,
        )

        if early_stopping:
            out_eval = build_eval(
                w_beh=pred_beh_state,
                w_old_query=pred_query_state.astype(np.float32, copy=False),
                eta=1.0,
                clip_y_min=None,
                clip_y_max=None,
            )
            target_eval, target_eval_diag = _make_stabilized_fixed_point_target(
                raw_target=out_eval["y"],
                current=pred_query_state,
                eta=1.0,
                normalize=normalize_occupancy,
                occupancy_ratio_max=occupancy_ratio_max,
                eps=occupancy_projection_eps,
                clip_pseudo_outcomes=clip_pseudo_outcomes,
                pseudo_outcome_max=pseudo_outcome_max,
                pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
                pseudo_outcome_min=pseudo_outcome_min,
                target_min=clip_y_min,
                target_max=clip_y_max,
            )
            y_test = target_eval[test_idx]
            risk_old = _occupancy_loss_value(
                pred_query_state[test_idx],
                y_test,
                loss=loss,
                huber_delta=loss_delta,
            )
        else:
            out_eval = out_train
            target_eval = target_train
            target_eval_diag = target_train_diag
            y_test = None
            risk_old = float("nan")
        # Fit the tree ensemble on the centered occupancy scale. The public
        # model prediction is ``w_init + booster.predict(...)``; using
        # LightGBM init_score together with init_model and a custom objective
        # can make later trees learn large constant offsets.
        dtrain = lgb.Dataset(X_train, label=y_train - float(w_init), weight=train_weight, free_raw_data=False)

        params_iter = dict(params_base)
        params_iter["learning_rate"] = learning_rate
        params_iter["objective"] = _make_occupancy_objective(
            loss=loss,
            huber_delta=loss_delta,
            huber_hessian_floor=huber_hessian_floor,
        )

        bst_candidate = lgb.train(
            params=params_iter,
            train_set=dtrain,
            num_boost_round=int(inner_num_boost_round),
            init_model=current_model,
            keep_training_booster=True,
        )

        delta_query = _predict_new_trees(
            bst_candidate,
            X_sa_query,
            boost_iteration=boost_iteration,
            num_trees=inner_num_boost_round,
        )
        delta_beh = _predict_new_trees(
            bst_candidate,
            X_sa_beh,
            boost_iteration=boost_iteration,
            num_trees=inner_num_boost_round,
        )

        candidate_query_raw = pred_query_raw + delta_query
        candidate_beh_raw = pred_beh_raw + delta_beh
        candidate_query_projected, query_projection_diag = _project_nonnegative_normalized(
            candidate_query_raw,
            max_value=occupancy_ratio_max,
            normalize=normalize_occupancy,
            eps=occupancy_projection_eps,
            return_info=True,
        )
        candidate_beh_projected, beh_projection_diag = _project_nonnegative_normalized(
            candidate_beh_raw,
            max_value=occupancy_ratio_max,
            normalize=normalize_occupancy,
            eps=occupancy_projection_eps,
            return_info=True,
        )
        candidate_query_state = _damped_update(pred_query_state, candidate_query_projected, fixed_point_damping)
        candidate_beh_state = _damped_update(pred_beh_state, candidate_beh_projected, fixed_point_damping)

        if early_stopping:
            risk_new = _occupancy_loss_value(
                candidate_query_state[test_idx],
                y_test,
                loss=loss,
                huber_delta=loss_delta,
            )
            improved = risk_new <= risk_old - early_stopping_min_delta * learning_rate
        else:
            risk_new = float("nan")
            improved = True

        pat_next = 0 if improved else patience + 1
        if show_progress and hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(
                f"new={risk_new:.3e} old={risk_old:.3e} d={(risk_old-risk_new):+.1e} "
                f"acc={int(improved)} pat={pat_next} tr={trees_used} lr={learning_rate:.2e} "
                f"K={int(inner_num_boost_round)} ref={refresh_count}"
            )

        row = _history_row(
            iteration=iteration,
            risk_old=risk_old,
            risk_new=risk_new,
            improved=improved,
            learning_rate=learning_rate,
            boost_iteration=boost_iteration,
            trees_used=trees_used,
            refresh_count=refresh_count,
            inner_num_boost_round=inner_num_boost_round,
            out_train=out_train,
            out_eval=out_eval,
        )
        row.update(
            _fixed_point_history_diag(
                current_query=pred_query_state,
                next_query=candidate_query_state,
                current_beh=pred_beh_state,
                next_beh=candidate_beh_state,
                train_idx=train_idx,
                test_idx=test_idx,
                raw_update=candidate_query_raw,
                projected_update=candidate_query_projected,
                damped_update=candidate_query_state,
                target=target_train,
                target_diag=target_train_diag,
                eval_target_diag=target_eval_diag,
                query_projection_diag=query_projection_diag,
                beh_projection_diag=beh_projection_diag,
                sample_weight_diag=sample_weight_diag,
                eta=fixed_point_damping,
                occupancy_ratio_max=occupancy_ratio_max,
                eps=occupancy_projection_eps,
            )
        )
        row["loss"] = loss
        if loss_delta is not None:
            row["huber_delta"] = float(loss_delta)

        if improved:
            current_model = bst_candidate
            pred_query_raw = float(w_init) + current_model.predict(X_sa_query).astype(np.float64, copy=False)
            pred_beh_raw = float(w_init) + current_model.predict(X_sa_beh).astype(np.float64, copy=False)
            pred_query_projected = _project_nonnegative_normalized(
                pred_query_raw,
                max_value=occupancy_ratio_max,
                normalize=normalize_occupancy,
                eps=occupancy_projection_eps,
            )
            pred_beh_projected = _project_nonnegative_normalized(
                pred_beh_raw,
                max_value=occupancy_ratio_max,
                normalize=normalize_occupancy,
                eps=occupancy_projection_eps,
            )
            pred_query_state = _damped_update(pred_query_state, pred_query_projected, fixed_point_damping)
            pred_beh_state = _damped_update(pred_beh_state, pred_beh_projected, fixed_point_damping)
            boost_iteration += int(inner_num_boost_round)
            trees_used += int(inner_num_boost_round)
            patience = 0
            plateau_streak = 0
            row["accepted"] = True
            row["did_refresh"] = False

            fp_rel = row.get("fixed_point_rel_change_eval", row.get("fixed_point_rel_change_train"))
            if fixed_point_tol is not None and iteration + 1 >= int(min_outer_iterations):
                if np.isfinite(float(fp_rel)) and float(fp_rel) <= float(fixed_point_tol):
                    fixed_point_stop_streak += 1
                else:
                    fixed_point_stop_streak = 0
                row["fixed_point_stop_streak"] = int(fixed_point_stop_streak)
                if fixed_point_stop_streak >= int(fixed_point_patience):
                    stopped_early = True
                    stop_iter = int(iteration)
                    stop_reason = "fixed_point_tol"
                    history.append(row)
                    break
        else:
            patience += 1
            plateau_streak += 1
            fixed_point_stop_streak = 0
            row["accepted"] = False
            if refresh_on_plateau and plateau_streak >= refresh_after_n_plateau:
                build_train = make_train_builder()
                plateau_streak = 0
                row["did_refresh"] = True
            else:
                row["did_refresh"] = False

            if early_stopping and patience >= early_stopping_patience:
                stopped_early = True
                stop_iter = int(iteration)
                stop_reason = "validation_loss"
                history.append(row)
                break

        history.append(row)

    return _build_occupancy_result(
        bst_w=current_model,
        bst_k=bst_k,
        bst_iw=bst_iw,
        k_fit=k_fit,
        iw_fit=iw_fit,
        source_fit=source_fit,
        source_diagnostics=source_diagnostics,
        c_fit=c_fit,
        c_diagnostics=c_diagnostics,
        initial_ratio_mode=resolved_initial_mode,
        one_step_ratio_mode=resolved_one_step_mode,
        loss=loss,
        huber_delta=huber_delta,
        huber_delta_scale=huber_delta_scale,
        huber_delta_quantile_power=huber_delta_quantile_power,
        huber_delta_min_quantile=huber_delta_min_quantile,
        huber_hessian_floor=huber_hessian_floor,
        k_prediction_offset=k_prediction_offset,
        iw_prediction_offset=iw_prediction_offset,
        X_sa_query=X_sa_query,
        X_s_query=X_s_query,
        pred_query_raw=pred_query_raw,
        pred_beh_raw=pred_beh_raw,
        pred_query_state=pred_query_state,
        pred_beh_state=pred_beh_state,
        iw_hat_beh=iw_hat_beh,
        crossfit_context=crossfit_context,
        n=n,
        history=history,
        stopped_early=stopped_early,
        stop_iter=stop_iter,
        stop_reason=stop_reason,
        trees_used=trees_used,
        refresh_count=refresh_count,
        eval_mcmc=eval_mcmc,
        mcmc_samples=mcmc_samples,
        inner_num_boost_round=inner_num_boost_round,
        fixed_point_damping=fixed_point_damping,
        normalize_occupancy=normalize_occupancy,
        occupancy_ratio_max=occupancy_ratio_max,
        occupancy_projection_eps=occupancy_projection_eps,
        clip_pseudo_outcomes=clip_pseudo_outcomes,
        pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
        occupancy_sample_weight_mode=occupancy_sample_weight_mode,
        direct_adjoint_num_boost_round=direct_adjoint_num_boost_round,
        direct_adjoint_lgb_params=direct_adjoint_lgb_params,
        direct_adjoint_loss=direct_adjoint_loss,
        direct_adjoint_validation_fraction=direct_adjoint_validation_fraction,
        direct_adjoint_early_stopping_rounds=direct_adjoint_early_stopping_rounds,
        direct_adjoint_sample_weight_mode=direct_adjoint_sample_weight_mode,
        direct_adjoint_sample_weight_max=direct_adjoint_sample_weight_max,
        num_target_action_samples=num_target_action_samples,
        continuation_query=continuation_query,
        known_iw_query=known_iw_query,
    )


def _validate_fit_args(
    *,
    gamma: float,
    inner_num_boost_round: int,
    mcmc_samples: int,
    batch_query: int,
    early_stopping: bool,
    test_frac: float,
    early_stopping_patience: int,
    refresh_after_n_plateau: int,
    eval_mcmc_multiplier: int,
    loss: str,
    huber_delta: Optional[float],
    huber_delta_scale: float,
    huber_delta_quantile_power: Optional[float],
    huber_delta_min_quantile: float,
    huber_hessian_floor: float,
    fixed_point_damping: float,
    occupancy_ratio_max: Optional[float],
    occupancy_projection_eps: float,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    transition_cache_norm_eps: float,
    occupancy_sample_weight_mode: str,
    occupancy_sample_weight_max: Optional[float],
    fixed_point_tol: Optional[float],
    fixed_point_patience: int,
    min_outer_iterations: int,
    direct_adjoint_num_boost_round: int,
    direct_adjoint_loss: str,
    direct_adjoint_validation_fraction: float,
    direct_adjoint_early_stopping_rounds: int,
    direct_adjoint_sample_weight_mode: str,
    direct_adjoint_sample_weight_max: Optional[float],
) -> None:
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    if inner_num_boost_round <= 0:
        raise ValueError("inner_num_boost_round must be positive.")
    if mcmc_samples <= 0:
        raise ValueError("mcmc_samples must be positive.")
    if batch_query <= 0:
        raise ValueError("batch_query must be positive.")
    if early_stopping and not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be in (0, 1).")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if refresh_after_n_plateau <= 0:
        raise ValueError("refresh_after_n_plateau must be positive.")
    if eval_mcmc_multiplier <= 0:
        raise ValueError("eval_mcmc_multiplier must be positive.")
    _normalize_occupancy_loss(loss)
    if huber_delta is not None and huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive when supplied.")
    if huber_delta_scale <= 0.0:
        raise ValueError("huber_delta_scale must be positive.")
    if huber_delta_quantile_power is not None and huber_delta_quantile_power <= 0.0:
        raise ValueError("huber_delta_quantile_power must be positive when supplied.")
    if not (0.0 < huber_delta_min_quantile < 1.0):
        raise ValueError("huber_delta_min_quantile must be in (0, 1).")
    if huber_hessian_floor < 0.0:
        raise ValueError("huber_hessian_floor must be nonnegative.")
    if direct_adjoint_num_boost_round <= 0:
        raise ValueError("direct_adjoint_num_boost_round must be positive.")
    _normalize_occupancy_loss(direct_adjoint_loss)
    if not (0.0 <= float(direct_adjoint_validation_fraction) < 1.0):
        raise ValueError("direct_adjoint_validation_fraction must be in [0, 1).")
    if direct_adjoint_early_stopping_rounds < 0:
        raise ValueError("direct_adjoint_early_stopping_rounds must be >= 0.")
    if str(direct_adjoint_sample_weight_mode) not in {"uniform", "sqrt_target", "target"}:
        raise ValueError("direct_adjoint_sample_weight_mode must be 'uniform', 'sqrt_target', or 'target'.")
    if direct_adjoint_sample_weight_max is not None and direct_adjoint_sample_weight_max <= 0.0:
        raise ValueError("direct_adjoint_sample_weight_max must be positive when supplied.")
    _validate_occupancy_stabilization_config(
        fixed_point_damping=fixed_point_damping,
        occupancy_ratio_max=occupancy_ratio_max,
        occupancy_projection_eps=occupancy_projection_eps,
        pseudo_outcome_max=pseudo_outcome_max,
        pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
        pseudo_outcome_min=pseudo_outcome_min,
        transition_cache_norm_eps=transition_cache_norm_eps,
        occupancy_sample_weight_mode=occupancy_sample_weight_mode,
        occupancy_sample_weight_max=occupancy_sample_weight_max,
        fixed_point_tol=fixed_point_tol,
        fixed_point_patience=fixed_point_patience,
        min_outer_iterations=min_outer_iterations,
    )


def _validate_ratio_prediction_config(
    *,
    prediction_max: Optional[float],
    prediction_power: float,
    moment_calibration: str = "none",
    crossfit_folds: int = 1,
    density_ratio_loss: str = "lsif",
    logistic_logit_clip: Optional[float] = 20.0,
) -> None:
    if prediction_max is not None and prediction_max <= 0.0:
        raise ValueError("prediction_max must be positive when supplied.")
    if not (0.0 < float(prediction_power) <= 1.0):
        raise ValueError("prediction_power must be in (0, 1].")
    if str(moment_calibration) not in {"none", "scalar"}:
        raise ValueError("moment_calibration must be 'none' or 'scalar'.")
    if int(crossfit_folds) < 1:
        raise ValueError("crossfit_folds must be >= 1.")
    if str(density_ratio_loss).strip().lower() not in {"lsif", "logistic"}:
        raise ValueError("density_ratio_loss must be 'lsif' or 'logistic'.")
    if logistic_logit_clip is not None and float(logistic_logit_clip) <= 0.0:
        raise ValueError("logistic_logit_clip must be positive when supplied.")


def _validate_occupancy_stabilization_config(
    *,
    fixed_point_damping: float,
    occupancy_ratio_max: Optional[float],
    occupancy_projection_eps: float,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    transition_cache_norm_eps: float,
    occupancy_sample_weight_mode: str,
    occupancy_sample_weight_max: Optional[float],
    fixed_point_tol: Optional[float],
    fixed_point_patience: int,
    min_outer_iterations: int,
) -> None:
    if not (0.0 < float(fixed_point_damping) <= 1.0):
        raise ValueError("fixed_point_damping must be in (0, 1].")
    if occupancy_ratio_max is not None and occupancy_ratio_max <= 0.0:
        raise ValueError("occupancy_ratio_max must be positive when supplied.")
    if occupancy_projection_eps <= 0.0:
        raise ValueError("occupancy_projection_eps must be positive.")
    if pseudo_outcome_max is not None and pseudo_outcome_max <= 0.0:
        raise ValueError("pseudo_outcome_max must be positive when supplied.")
    if not (0.0 < float(pseudo_outcome_upper_quantile) < 1.0):
        raise ValueError("pseudo_outcome_upper_quantile must be in (0, 1).")
    if pseudo_outcome_min < 0.0:
        raise ValueError("pseudo_outcome_min must be nonnegative.")
    if transition_cache_norm_eps <= 0.0:
        raise ValueError("transition_cache_norm_eps must be positive.")
    allowed_weight_modes = {"uniform", "sqrt_action_ratio", "action_ratio", "sqrt_target", "target"}
    if str(occupancy_sample_weight_mode) not in allowed_weight_modes:
        raise ValueError(f"occupancy_sample_weight_mode must be one of {sorted(allowed_weight_modes)}.")
    if occupancy_sample_weight_max is not None and occupancy_sample_weight_max <= 0.0:
        raise ValueError("occupancy_sample_weight_max must be positive when supplied.")
    if fixed_point_tol is not None and fixed_point_tol <= 0.0:
        raise ValueError("fixed_point_tol must be positive when supplied.")
    if fixed_point_patience <= 0:
        raise ValueError("fixed_point_patience must be positive.")
    if min_outer_iterations < 0:
        raise ValueError("min_outer_iterations must be nonnegative.")


def _as_2d(x: Array, name: str) -> Array:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must be 1D or 2D.")


def _prepare_target_action_samples(S: Array, A: Array, actions: Array, *, name: str) -> Dict[str, Array | int]:
    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape[0] != S.shape[0]:
            raise ValueError(f"{name} must match states rows.")
        return {
            "actions": arr,
            "states": S,
            "row_index": np.arange(S.shape[0], dtype=np.int64),
            "num_samples": 1,
        }
    if arr.ndim == 3:
        if arr.shape[0] != S.shape[0]:
            raise ValueError(f"{name} must match states rows.")
        if arr.shape[2] != A.shape[1]:
            raise ValueError(f"{name} must have the same feature dimension as actions.")
        n, m, d = arr.shape
        row_index = np.repeat(np.arange(n, dtype=np.int64), m)
        return {
            "actions": arr.reshape(n * m, d),
            "states": np.repeat(S, m, axis=0),
            "row_index": row_index,
            "num_samples": int(m),
        }
    raise ValueError(f"{name} must be 1D, 2D, or 3D.")


def _validate_base_transition_inputs(*, S: Array, A: Array, S_next: Array) -> None:
    n = S.shape[0]
    if A.shape[0] != n or S_next.shape[0] != n:
        raise ValueError("S, A, and S_next must all have the same number of rows.")
    if S_next.shape[1] != S.shape[1]:
        raise ValueError("S_next must have the same feature dimension as S.")


def _validate_target_action_rows(*, A: Array, A_pi: Array, name: str) -> None:
    if A_pi.shape[1] != A.shape[1]:
        raise ValueError(f"{name} must have the same feature dimension as actions.")


def _validate_aligned_inputs(*, S: Array, A: Array, S_next: Array, A_pi: Array) -> None:
    n = S.shape[0]
    if A.shape[0] != n or S_next.shape[0] != n or A_pi.shape[0] != n:
        raise ValueError("S, A, S_next, A_pi must all have the same number of rows.")
    if S_next.shape[1] != S.shape[1]:
        raise ValueError("S_next must have the same feature dimension as S.")
    if A_pi.shape[1] != A.shape[1]:
        raise ValueError("A_pi must have the same feature dimension as A.")


def _validate_initial_state_inputs(
    *,
    S: Array,
    S_initial: Optional[Array],
    initial_weights: Optional[Array],
) -> None:
    if S_initial is None:
        if initial_weights is not None:
            raise ValueError("initial_weights requires initial_states.")
        return
    if S_initial.shape[1] != S.shape[1]:
        raise ValueError("initial_states must have the same feature dimension as states.")
    if S_initial.shape[0] == 0:
        raise ValueError("initial_states must contain at least one row.")
    if initial_weights is not None:
        weights = np.asarray(initial_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != S_initial.shape[0]:
            raise ValueError("initial_weights must match initial_states rows.")
        if not np.any(np.isfinite(weights) & (weights > 0.0)):
            raise ValueError("initial_weights must contain at least one positive finite value.")


def _validate_initial_action_inputs(*, A: Array, S_initial: Optional[Array], A_initial: Optional[Array]) -> None:
    if A_initial is None:
        return
    if S_initial is None:
        raise ValueError("initial_actions requires initial_states.")
    if A_initial.shape[0] != S_initial.shape[0]:
        raise ValueError("initial_actions must match initial_states rows.")
    if A_initial.shape[1] != A.shape[1]:
        raise ValueError("initial_actions must have the same feature dimension as actions.")


def _validate_next_target_actions(*, A: Array, S: Array, A_pi_next: Optional[Array]) -> None:
    if A_pi_next is None:
        return
    if A_pi_next.shape[0] != S.shape[0]:
        raise ValueError("target_next_actions must match states rows.")
    if A_pi_next.shape[1] != A.shape[1]:
        raise ValueError("target_next_actions must have the same feature dimension as actions.")


def _resolve_initial_ratio_mode(
    mode: str,
    *,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "joint", "factored"}:
        raise ValueError("initial_ratio_mode must be 'auto', 'joint', or 'factored'.")
    if normalized == "auto":
        return "joint" if S_initial is not None and A_initial is not None else "factored"
    if normalized == "joint" and (S_initial is None or A_initial is None):
        raise ValueError("initial_ratio_mode='joint' requires initial_states and initial_actions.")
    return normalized


def _resolve_one_step_ratio_mode(mode: str, *, A_pi_next: Optional[Array]) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "direct", "factored"}:
        raise ValueError("one_step_ratio_mode must be 'auto', 'direct', or 'factored'.")
    if normalized == "auto":
        return "direct" if A_pi_next is not None else "factored"
    if normalized == "direct" and A_pi_next is None:
        raise ValueError("one_step_ratio_mode='direct' requires target_next_actions.")
    return normalized


def _resolve_continuation(
    *,
    terminals: Optional[Array],
    timeouts: Optional[Array],
    handle_timeouts: str,
    absorbing_state: bool,
    n_rows: int,
) -> Array:
    normalized = str(handle_timeouts).strip().lower()
    if normalized not in {"nonterminal", "terminal", "error"}:
        raise ValueError("handle_timeouts must be 'nonterminal', 'terminal', or 'error'.")
    terminal_arr = _optional_binary_vector(terminals, n_rows, "terminals")
    timeout_arr = _optional_binary_vector(timeouts, n_rows, "timeouts")
    if normalized == "error" and np.any(timeout_arr > 0.0):
        raise ValueError("timeouts were supplied and handle_timeouts='error'.")
    if bool(absorbing_state):
        return np.ones(int(n_rows), dtype=np.float64)
    continuation = 1.0 - terminal_arr
    if normalized == "terminal":
        continuation = continuation * (1.0 - timeout_arr)
    return np.clip(continuation, 0.0, 1.0).astype(np.float64, copy=False)


def _optional_binary_vector(values: Optional[Array], n_rows: int, name: str) -> Array:
    if values is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must be in [0, 1].")
    return arr.astype(np.float64, copy=False)


def _resolve_known_action_ratio_inputs(
    *,
    action_ratio_values: Optional[Array],
    behavior_log_prob: Optional[Array],
    target_log_prob: Optional[Array],
    n_rows: int,
    q_rows: int,
    n_target_rows: int,
    prediction_max: Optional[float],
    normalize: bool,
) -> tuple[Optional[Array], Optional[Array]]:
    supplied = action_ratio_values is not None
    supplied_logs = behavior_log_prob is not None or target_log_prob is not None
    if supplied and supplied_logs:
        raise ValueError("Provide either action_ratio_values or behavior/target log-probs, not both.")
    if behavior_log_prob is None and target_log_prob is None and action_ratio_values is None:
        return None, None
    if (behavior_log_prob is None) != (target_log_prob is None):
        raise ValueError("behavior_log_prob and target_log_prob must be supplied together.")
    if action_ratio_values is not None:
        raw = np.asarray(action_ratio_values, dtype=np.float64).reshape(-1)
    else:
        raw = np.exp(
            np.asarray(target_log_prob, dtype=np.float64).reshape(-1)
            - np.asarray(behavior_log_prob, dtype=np.float64).reshape(-1)
        )
    if raw.shape[0] not in {int(n_rows), int(q_rows)}:
        raise ValueError("known action ratios must have either behavior rows or all query rows.")
    values = _postprocess_known_action_ratios(raw, prediction_max=prediction_max, normalize=normalize)
    if raw.shape[0] == int(q_rows):
        return values[-int(n_rows) :], values
    query = np.concatenate([np.ones(int(n_target_rows), dtype=np.float64), values])
    return values, query


def _postprocess_known_action_ratios(values: Array, *, prediction_max: Optional[float], normalize: bool) -> Array:
    out = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite_pos = float(prediction_max) if prediction_max is not None else np.finfo(np.float64).max / 16.0
    out = np.nan_to_num(out, nan=0.0, posinf=finite_pos, neginf=0.0)
    np.maximum(out, 0.0, out=out)
    if prediction_max is not None:
        np.minimum(out, float(prediction_max), out=out)
    if normalize:
        mean = float(np.mean(out)) if out.size else 0.0
        if np.isfinite(mean) and mean > 1e-12:
            out = out / mean
    return out.astype(np.float64, copy=False)


def _candidate_configs(base: Any, grid: Optional[Sequence[Dict[str, Any]]]) -> List[Any]:
    if not grid:
        return [base]
    return [replace(base, **dict(overrides)) for overrides in grid]


def _score_action_config_cv(
    *,
    S: Array,
    A: Array,
    A_pi: Array,
    folds: Sequence[Array],
    config: ActionRatioConfig,
    seed: int,
) -> Dict[str, Any]:
    scores = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        kwargs = config.to_kwargs()
        kwargs["crossfit_folds"] = 1
        kwargs["show_tqdm"] = False
        fit = fit_importance_ratio_lgbm(
            S=S[train_idx],
            A=A[train_idx],
            A_pi=A_pi[train_idx],
            seed=seed + 17_001 * (fold_id + 1),
            **kwargs,
        )
        X_beh = np.concatenate([S[valid_idx], A[valid_idx]], axis=1)
        X_pi = np.concatenate([S[valid_idx], A_pi[valid_idx]], axis=1)
        pred_beh = _predict_processed_nuisance(fit=fit, X=X_beh, kind="iw")
        pred_pi = _predict_processed_nuisance(fit=fit, X=X_pi, kind="iw")
        scores.append(float(np.mean(pred_beh**2) - 2.0 * np.mean(pred_pi)))
    return dict(score=float(np.mean(scores)), fold_scores=[float(x) for x in scores], config_overrides={})


def _score_transition_config_cv(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    folds: Sequence[Array],
    config: TransitionRatioConfig,
    seed: int,
) -> Dict[str, Any]:
    scores = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        kwargs = config.to_kwargs()
        kwargs["crossfit_folds"] = 1
        kwargs["show_tqdm"] = False
        fit = fit_transition_ratio_lgbm(
            S=S[train_idx],
            A=A[train_idx],
            S_next=S_next[train_idx],
            seed=seed + 19_001 * (fold_id + 1),
            **kwargs,
        )
        X_sa_valid = np.concatenate([S[valid_idx], A[valid_idx]], axis=1)
        X_beh = np.hstack([X_sa_valid, S_next[valid_idx]])
        pred_beh = _predict_processed_nuisance(fit=fit, X=X_beh, kind="k")
        X_ref = _make_transition_reference_features(
            X_sa=X_sa_valid,
            S_ref=S[valid_idx],
            K=max(1, int(config.permutation_samples)),
            seed=seed + 23_001 * (fold_id + 1),
        )
        pred_ref = _predict_processed_nuisance(fit=fit, X=X_ref, kind="k")
        score = float(np.mean(pred_ref**2) - 2.0 * np.mean(pred_beh) + (np.mean(pred_ref) - 1.0) ** 2)
        scores.append(score)
    return dict(score=float(np.mean(scores)), fold_scores=[float(x) for x in scores], config_overrides={})


def _score_occupancy_config_cv(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    gamma: float,
    folds: Sequence[Array],
    occupancy: OccupancyRegressionConfig,
    action_ratio: ActionRatioConfig,
    source_state_ratio: SourceStateRatioConfig,
    transition_ratio: TransitionRatioConfig,
    scoring: str,
    lambda_norm: float,
    lambda_tail: float,
    seed: int,
    initial_states: Optional[Array],
    initial_actions: Optional[Array],
    initial_weights: Optional[Array],
    target_next_actions: Optional[Array],
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
) -> Dict[str, Any]:
    fold_rows = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        occ = replace(occupancy, show_progress=False, seed=seed + 29_001 * (fold_id + 1))
        act = replace(action_ratio, show_progress=False)
        trn = replace(transition_ratio, show_progress=False)
        model = fit_discounted_occupancy_ratio(
            states=S[train_idx],
            actions=A[train_idx],
            next_states=S_next[train_idx],
            target_actions=A_pi[train_idx],
            gamma=gamma,
            initial_states=_fold_initial_states(initial_states, train_idx, S.shape[0]),
            initial_actions=_fold_initial_states(initial_actions, train_idx, S.shape[0]),
            initial_weights=_fold_initial_weights(initial_weights, initial_states, train_idx, S.shape[0]),
            target_next_actions=None if target_next_actions is None else target_next_actions[train_idx],
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
            occupancy=occ,
            action_ratio=act,
            source_state_ratio=source_state_ratio,
            transition_ratio=trn,
        )
        weights = model.predict_state_action_ratio(S[valid_idx], A[valid_idx], clip=True)
        base_loss = _best_history_loss(model.history)
        norm_error = abs(float(np.mean(weights)) - 1.0)
        ess = _ess(weights)
        p99 = _quantile_or_nan(weights, 0.99)
        tail_penalty = float(np.log1p(p99 / max(ess, 1e-12)))
        score = float(base_loss)
        if scoring == "composite":
            score += float(lambda_norm) * norm_error + float(lambda_tail) * tail_penalty
        fold_rows.append(
            dict(
                score=score,
                validation_loss=float(base_loss),
                norm_error=float(norm_error),
                ess=float(ess),
                ess_fraction=float(ess / max(weights.size, 1)),
                p99=float(p99),
                max_weight=float(np.max(weights)) if weights.size else float("nan"),
                selected_iteration=int(model.diagnostics.get("stop_iter") or len(model.history)),
            )
        )
    return dict(
        score=float(np.mean([row["score"] for row in fold_rows])),
        validation_loss=float(np.mean([row["validation_loss"] for row in fold_rows])),
        norm_error=float(np.mean([row["norm_error"] for row in fold_rows])),
        ess_fraction=float(np.mean([row["ess_fraction"] for row in fold_rows])),
        p99=float(np.mean([row["p99"] for row in fold_rows])),
        fold_scores=fold_rows,
    )


def _best_history_loss(history: Sequence[Dict[str, Any]]) -> float:
    losses = [float(row["risk_new"]) for row in history if "risk_new" in row and np.isfinite(float(row["risk_new"]))]
    return float(np.min(losses)) if losses else 0.0


def _complement_indices(n: int, valid_idx: Array) -> Array:
    mask = np.ones(int(n), dtype=bool)
    mask[np.asarray(valid_idx, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def _fold_initial_states(initial_states: Optional[Array], train_idx: Array, full_n: int) -> Optional[Array]:
    if initial_states is None:
        return None
    arr = np.asarray(initial_states)
    if arr.shape[0] == int(full_n):
        return arr[np.asarray(train_idx, dtype=np.int64)]
    return arr


def _fold_initial_weights(
    initial_weights: Optional[Array],
    initial_states: Optional[Array],
    train_idx: Array,
    full_n: int,
) -> Optional[Array]:
    if initial_weights is None:
        return None
    weights = np.asarray(initial_weights)
    if initial_states is not None and np.asarray(initial_states).shape[0] == int(full_n):
        return weights[np.asarray(train_idx, dtype=np.int64)]
    return weights


def _predict_processed_nuisance(*, fit: Dict[str, Any], X: Array, kind: str) -> Array:
    booster_key = "bst_iw" if kind == "iw" else "bst_k"
    raw = _predict_ratio_from_booster(
        booster=fit[booster_key],
        X=X,
        offset=float(fit.get("prediction_offset", 0.0)),
        density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")),
        logistic_logit_clip=fit.get("logistic_logit_clip", 20.0),
        prior_correction=float(fit.get("prior_correction", 1.0)),
    )
    pred, _ = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=True,
        prediction_max=fit.get("prediction_max"),
        prediction_power=float(fit.get("prediction_power", 1.0)),
        normalize_predictions=bool(fit.get("normalize_predictions", False)),
    )
    return pred * float(fit.get("prediction_scale", 1.0))


def _make_transition_reference_features(*, X_sa: Array, S_ref: Array, K: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    X_sa = np.asarray(X_sa, dtype=np.float32)
    S_ref = _as_2d(np.asarray(S_ref, dtype=np.float32), "S_ref")
    blocks = []
    for _ in range(int(K)):
        blocks.append(np.hstack([X_sa, S_ref[rng.permutation(S_ref.shape[0])]]))
    return np.vstack(blocks)


def _prepare_nuisance_kwargs(
    *,
    lgb_params: Optional[Dict[str, Any]],
    k_lgb_params: Optional[Dict[str, Any]],
    iw_lgb_params: Optional[Dict[str, Any]],
    k_kwargs: Optional[Dict[str, Any]],
    iw_kwargs: Optional[Dict[str, Any]],
    source_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    base = {} if lgb_params is None else dict(lgb_params)
    k_options = {} if k_kwargs is None else dict(k_kwargs)
    iw_options = {} if iw_kwargs is None else dict(iw_kwargs)
    source_options = {} if source_kwargs is None else dict(source_kwargs)
    k_options.setdefault("refit_on_all_data", False)
    iw_options.setdefault("refit_on_all_data", False)
    source_options.setdefault("refit_on_all_data", False)
    k_options.setdefault("lgb_params", dict(base) | ({} if k_lgb_params is None else dict(k_lgb_params)))
    iw_options.setdefault("lgb_params", dict(base) | ({} if iw_lgb_params is None else dict(iw_lgb_params)))
    source_options.setdefault("lgb_params", dict(base))
    return {"k_kwargs": k_options, "iw_kwargs": iw_options, "source_kwargs": source_options}


def _fit_or_use_transition_ratio(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    seed: int,
    bst_k_init: Optional[lgb.Booster],
    bst_k_init_offset: float,
    k_kwargs: Dict[str, Any],
) -> tuple[lgb.Booster, Optional[Dict[str, Any]], float]:
    if bst_k_init is not None:
        return bst_k_init, None, float(bst_k_init_offset)
    fit = fit_transition_ratio_lgbm(S=S, A=A, S_next=S_next, seed=seed, **k_kwargs)
    return fit["bst_k"], fit, float(fit.get("prediction_offset", 0.0))


def _fit_or_use_importance_ratio(
    *,
    S: Array,
    A: Array,
    A_pi: Array,
    S_pi: Array,
    target_row_index: Array,
    X_sa_beh: Array,
    X_sa_query: Array,
    seed: int,
    bst_iw_init: Optional[lgb.Booster],
    bst_iw_init_offset: float,
    known_iw_beh: Optional[Array],
    known_iw_query: Optional[Array],
    iw_kwargs: Dict[str, Any],
) -> tuple[Optional[lgb.Booster], Optional[Dict[str, Any]], Array, float]:
    if known_iw_beh is not None:
        known_beh = np.asarray(known_iw_beh, dtype=np.float64).reshape(-1)
        known_query = (
            None
            if known_iw_query is None
            else np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
        )
        fit = dict(
            w_hat=known_beh,
            w_hat_raw=known_beh,
            w_hat_summary=_summarize_vector(known_beh),
            known_action_ratio=True,
            prediction_max=iw_kwargs.get("prediction_max"),
            prediction_power=float(iw_kwargs.get("prediction_power", 1.0)),
            normalize_predictions=bool(iw_kwargs.get("normalize_predictions", False)),
            known_action_ratio_features=X_sa_query if known_query is not None else X_sa_beh,
            known_action_ratio_predictions=known_query if known_query is not None else known_beh,
        )
        return None, fit, known_beh, 0.0
    if bst_iw_init is not None:
        offset = float(bst_iw_init_offset)
        density_ratio_loss = str(iw_kwargs.get("density_ratio_loss", "lsif"))
        logistic_logit_clip = iw_kwargs.get("logistic_logit_clip", 20.0)
        prior_correction = float(iw_kwargs.get("prior_correction", 1.0))
        iw_hat_raw = _predict_ratio_from_booster(
            booster=bst_iw_init,
            X=X_sa_beh,
            offset=offset,
            density_ratio_loss=density_ratio_loss,
            logistic_logit_clip=logistic_logit_clip,
            prior_correction=prior_correction,
        )
        iw_hat, iw_summary = _postprocess_ratio_predictions(
            iw_hat_raw,
            clip_nonneg=bool(iw_kwargs.get("clip_nonneg", True)),
            prediction_max=iw_kwargs.get("prediction_max"),
            prediction_power=float(iw_kwargs.get("prediction_power", 1.0)),
            normalize_predictions=bool(iw_kwargs.get("normalize_predictions", False)),
        )
        fit = dict(
            w_hat=iw_hat,
            w_hat_raw=iw_hat_raw,
            w_hat_summary=iw_summary,
            prefit=True,
            prediction_max=iw_kwargs.get("prediction_max"),
            prediction_power=float(iw_kwargs.get("prediction_power", 1.0)),
            normalize_predictions=bool(iw_kwargs.get("normalize_predictions", False)),
        )
        return bst_iw_init, fit, iw_hat, offset
    if A_pi.shape[0] == S.shape[0]:
        fit = fit_importance_ratio_lgbm(S=S, A=A, A_pi=A_pi, seed=seed, **iw_kwargs)
    else:
        row_index = np.asarray(target_row_index, dtype=np.int64).reshape(-1)
        fit = fit_importance_ratio_lgbm(
            S=S_pi,
            A=A[row_index],
            A_pi=A_pi,
            seed=seed,
            **iw_kwargs,
        )
    offset = float(fit.get("prediction_offset", 0.0))
    iw_hat_beh = _predict_processed_nuisance(fit=fit, X=X_sa_beh, kind="iw")
    return fit["bst_iw"], fit, _nonnegative(iw_hat_beh), offset


def _fit_source_state_ratio(
    *,
    S: Array,
    S_query: Array,
    S_initial: Optional[Array],
    initial_weights: Optional[Array],
    seed: int,
    source_kwargs: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Dict[str, Any]]:
    if S_initial is None:
        return None, None, _source_state_ratio_diagnostics(None, None)
    fit = fit_state_density_ratio_lgbm(
        S_ref=S,
        S_num=S_initial,
        numerator_weights=initial_weights,
        seed=seed + 53_001,
        **source_kwargs,
    )
    source_query = _predict_processed_source_state_ratio(fit=fit, X=S_query)
    return fit, source_query, _source_state_ratio_diagnostics(source_query, fit)


def _fit_initial_ratio(
    *,
    S: Array,
    X_sa_beh: Array,
    S_query: Array,
    X_sa_query: Array,
    S_initial: Optional[Array],
    X_sa_initial: Optional[Array],
    initial_weights: Optional[Array],
    seed: int,
    source_kwargs: Dict[str, Any],
    initial_ratio_mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Optional[Array], Dict[str, Any]]:
    if initial_ratio_mode == "joint":
        if X_sa_initial is None:
            raise ValueError("initial_ratio_mode='joint' requires initial state-action rows.")
        fit = fit_state_density_ratio_lgbm(
            S_ref=X_sa_beh,
            S_num=X_sa_initial,
            numerator_weights=initial_weights,
            seed=seed + 53_001,
            **source_kwargs,
        )
        source_query = _predict_processed_source_state_ratio(fit=fit, X=X_sa_query)
        joint_diagnostics = _source_state_ratio_diagnostics(source_query, fit)
        diagnostics = _source_state_ratio_diagnostics(None, None)
        diagnostics.update(
            initial_joint_ratio_enabled=True,
            initial_joint_ratio_mean=joint_diagnostics["source_state_ratio_mean"],
            initial_joint_ratio_max=joint_diagnostics["source_state_ratio_max"],
            initial_joint_ratio_ess_fraction=joint_diagnostics["source_state_ratio_ess_fraction"],
            initial_joint_ratio_loss=joint_diagnostics["source_state_ratio_loss"],
            initial_joint_ratio_density_ratio_loss=joint_diagnostics["source_state_ratio_density_ratio_loss"],
            initial_joint_ratio_clipped_fraction=joint_diagnostics["source_state_ratio_clipped_fraction"],
            initial_joint_ratio_query_clipped_fraction=joint_diagnostics[
                "source_state_ratio_query_clipped_fraction"
            ],
            initial_joint_ratio_prediction_max=joint_diagnostics["source_state_ratio_prediction_max"],
            initial_joint_ratio_prediction_scale=joint_diagnostics["source_state_ratio_prediction_scale"],
        )
        return fit, source_query, None, diagnostics

    fit, source_state_query, diagnostics = _fit_source_state_ratio(
        S=S,
        S_query=S_query,
        S_initial=S_initial,
        initial_weights=initial_weights,
        seed=seed,
        source_kwargs=source_kwargs,
    )
    diagnostics.update(
        initial_joint_ratio_enabled=False,
        initial_joint_ratio_mean=1.0,
        initial_joint_ratio_max=1.0,
        initial_joint_ratio_ess_fraction=1.0,
        initial_joint_ratio_loss=float("nan"),
        initial_joint_ratio_density_ratio_loss="none",
        initial_joint_ratio_clipped_fraction=0.0,
        initial_joint_ratio_query_clipped_fraction=0.0,
        initial_joint_ratio_prediction_max=float("nan"),
        initial_joint_ratio_prediction_scale=1.0,
    )
    return fit, None, source_state_query, diagnostics


def _make_factored_initial_source_weights(
    *,
    bst_iw: Optional[lgb.Booster],
    iw_fit: Optional[Dict[str, Any]],
    iw_kwargs: Dict[str, Any],
    iw_prediction_offset: float,
    X_sa_query: Array,
    source_state_query: Optional[Array],
    known_iw_query: Optional[Array] = None,
) -> Array:
    if known_iw_query is not None:
        out = np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
    else:
        if bst_iw is None:
            raise ValueError("Action-ratio booster is required when known query ratios are unavailable.")
        raw = _predict_ratio_from_booster(
            booster=bst_iw,
            X=X_sa_query,
            offset=float(iw_prediction_offset),
            density_ratio_loss=str((iw_fit or {}).get("density_ratio_loss", iw_kwargs.get("density_ratio_loss", "lsif"))),
            logistic_logit_clip=(iw_fit or {}).get("logistic_logit_clip", iw_kwargs.get("logistic_logit_clip", 20.0)),
            prior_correction=float((iw_fit or {}).get("prior_correction", iw_kwargs.get("prior_correction", 1.0))),
        )
        pred, _ = _postprocess_ratio_predictions(
            raw,
            clip_nonneg=True,
            prediction_max=(iw_fit or {}).get("prediction_max", iw_kwargs.get("prediction_max")),
            prediction_power=float((iw_fit or {}).get("prediction_power", iw_kwargs.get("prediction_power", 1.0))),
            normalize_predictions=bool((iw_fit or {}).get("normalize_predictions", iw_kwargs.get("normalize_predictions", False))),
        )
        out = pred * _nuisance_prediction_scale(iw_fit)
    if source_state_query is not None:
        out = out * np.asarray(source_state_query, dtype=np.float64).reshape(-1)
    return np.maximum(out, 0.0)


def _fit_direct_one_step_ratio(
    *,
    X_ref: Array,
    X_next_pi: Optional[Array],
    X_query: Array,
    seed: int,
    source_kwargs: Dict[str, Any],
    one_step_ratio_mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Dict[str, Any]]:
    if one_step_ratio_mode != "direct":
        return None, None, _one_step_direct_ratio_diagnostics(None, None)
    if X_next_pi is None:
        raise ValueError("one_step_ratio_mode='direct' requires target_next_actions.")
    fit = fit_state_density_ratio_lgbm(
        S_ref=X_ref,
        S_num=X_next_pi,
        numerator_weights=None,
        seed=seed + 61_001,
        **source_kwargs,
    )
    c_query = _predict_processed_source_state_ratio(fit=fit, X=X_query)
    return fit, c_query, _one_step_direct_ratio_diagnostics(c_query, fit)


def _one_step_direct_ratio_diagnostics(c_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if c_query is None:
        return dict(
            one_step_direct_ratio_enabled=False,
            one_step_direct_ratio_mean=1.0,
            one_step_direct_ratio_max=1.0,
            one_step_direct_ratio_ess_fraction=1.0,
            one_step_direct_ratio_loss=float("nan"),
            one_step_direct_ratio_clipped_fraction=0.0,
            one_step_direct_ratio_query_clipped_fraction=0.0,
            one_step_direct_ratio_density_ratio_loss="none",
            one_step_direct_ratio_prediction_max=float("nan"),
            one_step_direct_ratio_prediction_scale=1.0,
        )
    values = np.asarray(c_query, dtype=np.float64).reshape(-1)
    summary = fit.get("source_hat_summary", {}) if isinstance(fit, dict) else {}
    query_cap_fraction = _ratio_query_cap_fraction(values, fit)
    return dict(
        one_step_direct_ratio_enabled=True,
        one_step_direct_ratio_mean=float(np.mean(values)) if values.size else float("nan"),
        one_step_direct_ratio_max=float(np.max(values)) if values.size else float("nan"),
        one_step_direct_ratio_ess_fraction=float(_ess(values) / max(values.size, 1)),
        one_step_direct_ratio_loss=_fit_eval_loss(fit),
        one_step_direct_ratio_clipped_fraction=float(summary.get("clipped_fraction", 0.0)),
        one_step_direct_ratio_query_clipped_fraction=float(query_cap_fraction),
        one_step_direct_ratio_density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")) if isinstance(fit, dict) else "",
        one_step_direct_ratio_prediction_max=_fit_prediction_max(fit),
        one_step_direct_ratio_prediction_scale=_nuisance_prediction_scale(fit),
    )


def _predict_processed_source_state_ratio(*, fit: Dict[str, Any], X: Array) -> Array:
    raw = _predict_ratio_from_booster(
        booster=fit["bst_source"],
        X=X,
        offset=float(fit.get("prediction_offset", 0.0)),
        density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")),
        logistic_logit_clip=fit.get("logistic_logit_clip", 20.0),
        prior_correction=float(fit.get("prior_correction", 1.0)),
    )
    pred, _ = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=True,
        prediction_max=fit.get("prediction_max"),
        prediction_power=float(fit.get("prediction_power", 1.0)),
        normalize_predictions=bool(fit.get("normalize_predictions", False)),
    )
    return pred * _nuisance_prediction_scale(fit)


def _source_state_ratio_diagnostics(source_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if source_query is None:
        return dict(
            source_state_ratio_enabled=False,
            source_state_ratio_mean=1.0,
            source_state_ratio_max=1.0,
            source_state_ratio_ess_fraction=1.0,
            source_state_ratio_loss=float("nan"),
            source_state_ratio_density_ratio_loss="none",
            source_state_ratio_clipped_fraction=0.0,
            source_state_ratio_query_clipped_fraction=0.0,
            source_state_ratio_prediction_max=float("nan"),
            source_state_ratio_prediction_scale=1.0,
        )
    values = np.asarray(source_query, dtype=np.float64).reshape(-1)
    summary = fit.get("source_hat_summary", {}) if isinstance(fit, dict) else {}
    query_cap_fraction = _ratio_query_cap_fraction(values, fit)
    return dict(
        source_state_ratio_enabled=True,
        source_state_ratio_mean=float(np.mean(values)) if values.size else float("nan"),
        source_state_ratio_max=float(np.max(values)) if values.size else float("nan"),
        source_state_ratio_ess_fraction=float(_ess(values) / max(values.size, 1)),
        source_state_ratio_loss=_fit_eval_loss(fit),
        source_state_ratio_density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")) if isinstance(fit, dict) else "",
        source_state_ratio_clipped_fraction=float(summary.get("clipped_fraction", 0.0)),
        source_state_ratio_query_clipped_fraction=float(query_cap_fraction),
        source_state_ratio_prediction_max=_fit_prediction_max(fit),
        source_state_ratio_prediction_scale=_nuisance_prediction_scale(fit),
    )


def _fit_prediction_max(fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return float("nan")
    value = fit.get("prediction_max")
    if value is None:
        return float("nan")
    return float(value)


def _ratio_query_cap_fraction(values: Array, fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return 0.0
    prediction_max = fit.get("prediction_max")
    if prediction_max is None:
        return 0.0
    cap = float(prediction_max) * float(_nuisance_prediction_scale(fit))
    if not np.isfinite(cap) or cap <= 0.0:
        return 0.0
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.mean(x >= cap * (1.0 - 1e-10)))


def _fit_eval_loss(fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return float("nan")
    evals = fit.get("evals_result")
    if not isinstance(evals, dict):
        return float("nan")
    valid = evals.get("valid")
    if not isinstance(valid, dict) or not valid:
        return float("nan")
    for key in ("loss", "binary_logloss"):
        values = valid.get(key)
        if values:
            finite = [float(v) for v in values if np.isfinite(float(v))]
            return float(np.min(finite)) if finite else float("nan")
    for values in valid.values():
        if values:
            finite = [float(v) for v in values if np.isfinite(float(v))]
            return float(np.min(finite)) if finite else float("nan")
    return float("nan")


def _nuisance_prediction_scale(fit: Optional[Dict[str, Any]]) -> float:
    if isinstance(fit, dict):
        return float(fit.get("prediction_scale", 1.0))
    return 1.0


def _fit_crossfit_nuisance_context(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    X_sa_beh: Array,
    seed: int,
    bst_k_final: lgb.Booster,
    bst_iw_final: lgb.Booster,
    k_fit_final: Optional[Dict[str, Any]],
    iw_fit_final: Optional[Dict[str, Any]],
    k_prediction_offset: float,
    iw_prediction_offset: float,
    k_kwargs: Dict[str, Any],
    iw_kwargs: Dict[str, Any],
    bst_k_init: Optional[lgb.Booster],
    bst_iw_init: Optional[lgb.Booster],
) -> Optional[Dict[str, Any]]:
    k_folds = int(k_kwargs.get("crossfit_folds", 1) or 1)
    iw_folds = int(iw_kwargs.get("crossfit_folds", 1) or 1)
    folds = max(k_folds, iw_folds)
    if folds <= 1:
        return None
    if bst_iw_final is None:
        return None
    if bst_k_init is not None or bst_iw_init is not None:
        return None

    fold_indices = _make_fold_indices(
        S.shape[0],
        folds,
        int(iw_kwargs.get("crossfit_seed") or k_kwargs.get("crossfit_seed") or seed + 31_337),
    )
    k_models = []
    iw_models = []
    iw_oof = np.empty(S.shape[0], dtype=np.float64)
    k_oof = np.empty(S.shape[0], dtype=np.float64)
    for fold_id, valid_idx in enumerate(fold_indices):
        train_mask = np.ones(S.shape[0], dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.flatnonzero(train_mask)

        if iw_folds > 1:
            iw_options = dict(iw_kwargs)
            iw_options["crossfit_folds"] = 1
            iw_options["show_tqdm"] = False
            iw_fit = fit_importance_ratio_lgbm(
                S=S[train_idx],
                A=A[train_idx],
                A_pi=A_pi[train_idx],
                seed=seed + 1_003 * (fold_id + 1),
                **iw_options,
            )
            iw_model = dict(
                booster=iw_fit["bst_iw"],
                offset=float(iw_fit.get("prediction_offset", 0.0)),
                scale=_nuisance_prediction_scale(iw_fit),
                fit=iw_fit,
                density_ratio_loss=str(iw_fit.get("density_ratio_loss", "lsif")),
                logistic_logit_clip=iw_fit.get("logistic_logit_clip", 20.0),
                prior_correction=float(iw_fit.get("prior_correction", 1.0)),
            )
            raw = _predict_ratio_from_booster(
                booster=iw_model["booster"],
                X=X_sa_beh[valid_idx],
                offset=float(iw_model["offset"]),
                density_ratio_loss=str(iw_model["density_ratio_loss"]),
                logistic_logit_clip=iw_model["logistic_logit_clip"],
                prior_correction=float(iw_model["prior_correction"]),
            )
            pred, _ = _postprocess_ratio_predictions(
                raw,
                clip_nonneg=bool(iw_options.get("clip_nonneg", True)),
                prediction_max=iw_options.get("prediction_max"),
                prediction_power=float(iw_options.get("prediction_power", 1.0)),
                normalize_predictions=bool(iw_options.get("normalize_predictions", False)),
            )
            iw_oof[valid_idx] = pred * iw_model["scale"]
        else:
            iw_model = dict(
                booster=bst_iw_final,
                offset=iw_prediction_offset,
                scale=_nuisance_prediction_scale(iw_fit_final),
                fit=iw_fit_final,
                density_ratio_loss=str((iw_fit_final or {}).get("density_ratio_loss", "lsif")),
                logistic_logit_clip=(iw_fit_final or {}).get("logistic_logit_clip", 20.0),
                prior_correction=float((iw_fit_final or {}).get("prior_correction", 1.0)),
            )
            iw_oof[valid_idx] = np.nan
        iw_models.append(iw_model)

        if k_folds > 1:
            k_options = dict(k_kwargs)
            k_options["crossfit_folds"] = 1
            k_options["show_tqdm"] = False
            k_fit = fit_transition_ratio_lgbm(
                S=S[train_idx],
                A=A[train_idx],
                S_next=S_next[train_idx],
                seed=seed + 2_003 * (fold_id + 1),
                **k_options,
            )
            k_model = dict(
                booster=k_fit["bst_k"],
                offset=float(k_fit.get("prediction_offset", 0.0)),
                scale=_nuisance_prediction_scale(k_fit),
                fit=k_fit,
                density_ratio_loss=str(k_fit.get("density_ratio_loss", "lsif")),
                logistic_logit_clip=k_fit.get("logistic_logit_clip", 20.0),
                prior_correction=float(k_fit.get("prior_correction", 1.0)),
            )
            Xk_valid = np.hstack([X_sa_beh[valid_idx], S_next[valid_idx]])
            raw = _predict_ratio_from_booster(
                booster=k_model["booster"],
                X=Xk_valid,
                offset=float(k_model["offset"]),
                density_ratio_loss=str(k_model["density_ratio_loss"]),
                logistic_logit_clip=k_model["logistic_logit_clip"],
                prior_correction=float(k_model["prior_correction"]),
            )
            pred, _ = _postprocess_ratio_predictions(
                raw,
                clip_nonneg=bool(k_options.get("clip_nonneg", True)),
                prediction_max=k_options.get("prediction_max"),
                prediction_power=float(k_options.get("prediction_power", 1.0)),
                normalize_predictions=bool(k_options.get("normalize_predictions", False)),
            )
            k_oof[valid_idx] = pred * k_model["scale"]
        else:
            k_model = dict(
                booster=bst_k_final,
                offset=k_prediction_offset,
                scale=_nuisance_prediction_scale(k_fit_final),
                fit=k_fit_final,
                density_ratio_loss=str((k_fit_final or {}).get("density_ratio_loss", "lsif")),
                logistic_logit_clip=(k_fit_final or {}).get("logistic_logit_clip", 20.0),
                prior_correction=float((k_fit_final or {}).get("prior_correction", 1.0)),
            )
            k_oof[valid_idx] = np.nan
        k_models.append(k_model)

    diagnostics = dict(
        enabled=True,
        folds=int(folds),
        action_crossfit_folds=int(iw_folds),
        transition_crossfit_folds=int(k_folds),
        action_oof_mean=float(np.nanmean(iw_oof)) if np.any(np.isfinite(iw_oof)) else float("nan"),
        transition_oof_mean=float(np.nanmean(k_oof)) if np.any(np.isfinite(k_oof)) else float("nan"),
    )
    return dict(
        folds=fold_indices,
        iw_models=iw_models,
        k_models=k_models,
        iw_oof=iw_oof,
        k_oof=k_oof,
        diagnostics=diagnostics,
    )


def _make_fold_indices(n_rows: int, n_folds: int, seed: int) -> List[Array]:
    if int(n_folds) < 1:
        raise ValueError("crossfit_folds must be >= 1.")
    rng = np.random.default_rng(seed)
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(rng.permutation(n_rows), int(n_folds))]


def _make_train_test_indices(
    *,
    n_rows: int,
    early_stopping: bool,
    test_frac: float,
    seed: int,
) -> tuple[Array, Array]:
    if not early_stopping:
        return np.arange(n_rows, dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_test = max(1, int(np.floor(test_frac * n_rows)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    if train_idx.size == 0:
        raise ValueError("test_frac too large: no training rows left.")
    return train_idx, test_idx


def _default_occupancy_lgb_params(*, seed: int) -> Dict[str, Any]:
    return dict(
        objective="regression",
        learning_rate=0.1,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=0.0,
        verbose=-1,
        seed=seed,
    )


def _normalize_occupancy_loss(loss: str) -> str:
    normalized = str(loss).strip().lower()
    aliases = {
        "l2": "squared",
        "mse": "squared",
        "squared_error": "squared",
        "squared": "squared",
        "huber": "huber",
        "robust": "huber",
    }
    if normalized not in aliases:
        raise ValueError("loss must be 'squared' or 'huber'.")
    return aliases[normalized]


def _resolve_huber_delta(
    residuals: Array,
    *,
    loss: str,
    huber_delta: Optional[float],
    huber_delta_scale: float,
    huber_delta_quantile_power: Optional[float],
    huber_delta_min_quantile: float,
) -> Optional[float]:
    if loss != "huber":
        return None
    if huber_delta is not None:
        # A fixed threshold estimates a conditional Huber location. The
        # adaptive default below lets the threshold diverge with n so robust
        # finite-sample fitting still targets the conditional mean asymptotically.
        return float(huber_delta)

    resid = np.asarray(residuals, dtype=np.float64).reshape(-1)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return 1.0
    centered = resid - float(np.median(resid))
    mad = float(np.median(np.abs(centered)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(resid))
    if not np.isfinite(scale) or scale <= 0.0:
        q75, q25 = np.percentile(resid, [75.0, 25.0])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(float(np.mean(np.abs(resid))), 1.0)
    growth = _adaptive_huber_growth(resid.size)
    delta = float(huber_delta_scale) * scale * growth
    quantile_cap = _adaptive_huber_quantile_cap(
        resid,
        quantile_power=huber_delta_quantile_power,
        min_quantile=huber_delta_min_quantile,
    )
    if quantile_cap is not None:
        delta = min(delta, quantile_cap)
    return max(delta, 1e-8)


def _adaptive_huber_growth(n_eff: int) -> float:
    """Increasing Huber threshold multiplier for mean-consistent robust fitting."""
    n_eff = int(n_eff)
    if n_eff <= 2:
        return 1.0
    return float(np.sqrt(n_eff / np.log(n_eff)))


def _adaptive_huber_quantile_cap(
    residuals: Array,
    *,
    quantile_power: Optional[float],
    min_quantile: float,
) -> Optional[float]:
    """Finite-sample cap whose quantile level tends to one as sample size grows."""
    if quantile_power is None:
        return None
    abs_resid = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    abs_resid = abs_resid[np.isfinite(abs_resid)]
    if abs_resid.size == 0:
        return None
    level = max(float(min_quantile), 1.0 - abs_resid.size ** (-float(quantile_power)))
    level = min(level, 1.0 - 1.0 / max(float(abs_resid.size), 2.0))
    cap = float(np.quantile(abs_resid, level))
    return max(cap, 1e-8) if np.isfinite(cap) else None


def _make_occupancy_objective(
    *,
    loss: str,
    huber_delta: Optional[float],
    huber_hessian_floor: float,
) -> Callable[[Array, lgb.Dataset], tuple[Array, Array]]:
    if loss == "squared":
        return _squared_error_objective

    if huber_delta is None:
        raise ValueError("huber_delta is required for Huber occupancy loss.")
    delta = float(huber_delta)
    hessian_floor = float(huber_hessian_floor)

    def huber_objective(preds: Array, train_data: lgb.Dataset) -> tuple[Array, Array]:
        resid = preds - train_data.get_label()
        abs_resid = np.abs(resid)
        grad = np.clip(resid, -delta, delta)
        hess = np.where(abs_resid <= delta, 1.0, hessian_floor)
        return grad, hess

    return huber_objective


def _squared_error_objective(preds: Array, train_data: lgb.Dataset) -> tuple[Array, Array]:
    resid = preds - train_data.get_label()
    return resid, np.ones_like(resid)


def _occupancy_loss_value(
    preds: Array,
    labels: Array,
    *,
    loss: str,
    huber_delta: Optional[float],
) -> float:
    resid = np.asarray(preds, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    if loss == "squared":
        return float(0.5 * np.mean(resid**2))
    if huber_delta is None:
        raise ValueError("huber_delta is required for Huber occupancy loss.")
    abs_resid = np.abs(resid)
    quadratic = abs_resid <= float(huber_delta)
    values = np.empty_like(abs_resid, dtype=np.float64)
    values[quadratic] = 0.5 * resid[quadratic] ** 2
    values[~quadratic] = float(huber_delta) * (abs_resid[~quadratic] - 0.5 * float(huber_delta))
    return float(np.mean(values))


def _predict_new_trees(
    model: lgb.Booster,
    X: Array,
    *,
    boost_iteration: int,
    num_trees: int,
) -> Array:
    return model.predict(
        X,
        start_iteration=int(boost_iteration),
        num_iteration=int(num_trees),
    ).astype(np.float64, copy=False)


def _check_prediction_cache(
    *,
    current_model: Optional[lgb.Booster],
    X_query: Array,
    X_beh: Array,
    pred_query_raw: Array,
    pred_beh_raw: Array,
    w_init: float,
    iteration: int,
    tolerance: float = 1e-8,
) -> None:
    if current_model is None or iteration >= 10:
        return
    query_check = float(w_init) + current_model.predict(X_query).astype(np.float64, copy=False)
    beh_check = float(w_init) + current_model.predict(X_beh).astype(np.float64, copy=False)
    if not np.allclose(query_check, pred_query_raw, atol=tolerance, rtol=1e-6):
        raise ValueError("pred_query cache does not match w_init + model.predict(X_sa_query).")
    if not np.allclose(beh_check, pred_beh_raw, atol=tolerance, rtol=1e-6):
        raise ValueError("pred_beh cache does not match w_init + model.predict(X_sa_iw).")


def _history_row(
    *,
    iteration: int,
    risk_old: float,
    risk_new: float,
    improved: bool,
    learning_rate: float,
    boost_iteration: int,
    trees_used: int,
    refresh_count: int,
    inner_num_boost_round: int,
    out_train: Dict[str, Any],
    out_eval: Dict[str, Any],
) -> Dict[str, Any]:
    row = dict(
        iter=int(iteration),
        risk_old=float(risk_old),
        risk_new=float(risk_new),
        improved=bool(improved),
        learning_rate=float(learning_rate),
        boost_iteration=int(boost_iteration),
        trees_used=int(trees_used),
        refresh_count=int(refresh_count),
        inner_num_boost_round=int(inner_num_boost_round),
    )
    row.update(out_train.get("diag", {}))
    row.update({f"eval_{key}": val for key, val in out_eval.get("diag", {}).items()})
    return row


def _build_occupancy_result(
    *,
    bst_w: Optional[lgb.Booster],
    bst_k: lgb.Booster,
    bst_iw: Optional[lgb.Booster],
    k_fit: Optional[Dict[str, Any]],
    iw_fit: Optional[Dict[str, Any]],
    source_fit: Optional[Dict[str, Any]],
    source_diagnostics: Dict[str, Any],
    c_fit: Optional[Dict[str, Any]],
    c_diagnostics: Dict[str, Any],
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
    loss: str,
    huber_delta: Optional[float],
    huber_delta_scale: float,
    huber_delta_quantile_power: Optional[float],
    huber_delta_min_quantile: float,
    huber_hessian_floor: float,
    k_prediction_offset: float,
    iw_prediction_offset: float,
    X_sa_query: Array,
    X_s_query: Array,
    pred_query_raw: Array,
    pred_beh_raw: Array,
    pred_query_state: Array,
    pred_beh_state: Array,
    iw_hat_beh: Array,
    crossfit_context: Optional[Dict[str, Any]],
    n: int,
    history: List[Dict[str, Any]],
    stopped_early: bool,
    stop_iter: Optional[int],
    stop_reason: Optional[str],
    trees_used: int,
    refresh_count: int,
    eval_mcmc: int,
    mcmc_samples: int,
    inner_num_boost_round: int,
    fixed_point_damping: float,
    normalize_occupancy: bool,
    occupancy_ratio_max: Optional[float],
    occupancy_projection_eps: float,
    clip_pseudo_outcomes: bool,
    pseudo_outcome_upper_quantile: float,
    occupancy_sample_weight_mode: str,
    direct_adjoint_num_boost_round: int,
    direct_adjoint_lgb_params: Optional[Dict[str, Any]],
    direct_adjoint_loss: str,
    direct_adjoint_validation_fraction: float,
    direct_adjoint_early_stopping_rounds: int,
    direct_adjoint_sample_weight_mode: str,
    direct_adjoint_sample_weight_max: Optional[float],
    num_target_action_samples: int,
    continuation_query: Array,
    known_iw_query: Optional[Array],
) -> Dict[str, Any]:
    n_target = int(np.asarray(pred_query_raw).shape[0] - int(n))
    pred_pi_raw = pred_query_raw[:n_target]
    pred_beh_in_query_raw = pred_query_raw[n_target:]
    pred_query = np.asarray(pred_query_state, dtype=np.float64).reshape(-1)
    pred_beh = np.asarray(pred_beh_state, dtype=np.float64).reshape(-1)
    pred_pi = pred_query[:n_target]
    pred_beh_in_query = pred_query[n_target:]
    prediction_scale = None
    if normalize_occupancy:
        projected_beh = _project_nonnegative_normalized(
            pred_beh_raw,
            max_value=occupancy_ratio_max,
            normalize=False,
            eps=occupancy_projection_eps,
        )
        mean_projected = float(np.mean(projected_beh)) if projected_beh.size else 1.0
        prediction_scale = mean_projected if np.isfinite(mean_projected) and mean_projected > occupancy_projection_eps else 1.0

    iw_density_ratio_loss = str(iw_fit.get("density_ratio_loss", "lsif")) if isinstance(iw_fit, dict) else "lsif"
    iw_logistic_logit_clip = iw_fit.get("logistic_logit_clip", 20.0) if isinstance(iw_fit, dict) else 20.0
    iw_prior_correction = float(iw_fit.get("prior_correction", 1.0)) if isinstance(iw_fit, dict) else 1.0
    k_density_ratio_loss = str(k_fit.get("density_ratio_loss", "lsif")) if isinstance(k_fit, dict) else "lsif"
    k_logistic_logit_clip = k_fit.get("logistic_logit_clip", 20.0) if isinstance(k_fit, dict) else 20.0
    k_prior_correction = float(k_fit.get("prior_correction", 1.0)) if isinstance(k_fit, dict) else 1.0
    c_density_ratio_loss = str(c_fit.get("density_ratio_loss", "none")) if isinstance(c_fit, dict) else "none"
    iw_prediction_max = iw_fit.get("prediction_max") if isinstance(iw_fit, dict) else None
    iw_prediction_power = float(iw_fit.get("prediction_power", 1.0)) if isinstance(iw_fit, dict) else 1.0
    iw_normalize_predictions = bool(iw_fit.get("normalize_predictions", False)) if isinstance(iw_fit, dict) else False
    if known_iw_query is not None:
        iw_query_hat = np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
    else:
        iw_query_raw = _predict_ratio_from_booster(
            booster=bst_iw,
            X=X_sa_query,
            offset=float(iw_prediction_offset),
            density_ratio_loss=iw_density_ratio_loss,
            logistic_logit_clip=iw_logistic_logit_clip,
            prior_correction=iw_prior_correction,
        )
        iw_query_hat, _ = _postprocess_ratio_predictions(
            iw_query_raw,
            clip_nonneg=True,
            prediction_max=iw_prediction_max,
            prediction_power=iw_prediction_power,
            normalize_predictions=iw_normalize_predictions,
        )
        iw_query_hat = iw_query_hat * _nuisance_prediction_scale(iw_fit)
    iw_pi_hat = iw_query_hat[:n_target]
    iw_beh_in_query_hat = iw_query_hat[n_target:]
    state_ratio_beh = _safe_divide(pred_beh, iw_hat_beh)
    state_ratio_pi = _safe_divide(pred_pi, iw_pi_hat)

    return dict(
        bst_w=bst_w,
        bst_k=bst_k,
        bst_iw=bst_iw,
        k_fit=k_fit,
        iw_fit=iw_fit,
        source_fit=source_fit,
        loss=loss,
        huber_delta=None if huber_delta is None else float(huber_delta),
        huber_delta_scale=float(huber_delta_scale),
        huber_delta_quantile_power=None
        if huber_delta_quantile_power is None
        else float(huber_delta_quantile_power),
        huber_delta_min_quantile=float(huber_delta_min_quantile),
        huber_hessian_floor=float(huber_hessian_floor),
        k_prediction_offset=float(k_prediction_offset),
        iw_prediction_offset=float(iw_prediction_offset),
        pred_query=pred_query_raw,
        pred_query_raw=pred_query_raw,
        pred_query_clipped=pred_query,
        pred_query_stabilized=pred_query,
        pred_beh=pred_beh_raw,
        pred_beh_raw=pred_beh_raw,
        pred_beh_stabilized=pred_beh,
        X_sa_query=X_sa_query,
        X_s_query=X_s_query,
        pred_pi=pred_pi_raw,
        pred_pi_raw=pred_pi_raw,
        pred_pi_clipped=pred_pi,
        pred_iw=iw_hat_beh,
        pred_iw_query=iw_query_hat,
        pred_iw_pi=iw_pi_hat,
        pred_iw_beh_in_query=iw_beh_in_query_hat,
        pred_state_action_ratio_beh=pred_beh,
        pred_state_action_ratio_beh_raw=pred_beh_raw,
        pred_state_action_ratio_pi=pred_pi,
        pred_state_action_ratio_pi_raw=pred_pi_raw,
        pred_state_ratio_beh=state_ratio_beh,
        pred_state_ratio_pi=state_ratio_pi,
        pred_sa_iw_in_query=pred_beh_in_query_raw,
        pred_sa_iw_in_query_raw=pred_beh_in_query_raw,
        pred_sa_iw_in_query_clipped=pred_beh_in_query,
        history=history,
        stopped_early=stopped_early,
        stop_iter=stop_iter,
        stop_reason=stop_reason,
        trees_used=int(trees_used),
        refresh_count=int(refresh_count),
        eval_mcmc_samples=int(eval_mcmc),
        mcmc_samples=int(mcmc_samples),
        inner_num_boost_round=int(inner_num_boost_round),
        fixed_point_damping=float(fixed_point_damping),
        normalize_occupancy=bool(normalize_occupancy),
        occupancy_ratio_max=None if occupancy_ratio_max is None else float(occupancy_ratio_max),
        occupancy_projection_eps=float(occupancy_projection_eps),
        occupancy_prediction_scale=None if prediction_scale is None else float(prediction_scale),
        clip_pseudo_outcomes=bool(clip_pseudo_outcomes),
        pseudo_outcome_upper_quantile=float(pseudo_outcome_upper_quantile),
        occupancy_sample_weight_mode=str(occupancy_sample_weight_mode),
        num_target_action_samples=int(num_target_action_samples),
        continuation_mean=float(np.mean(continuation_query)),
        continuation_min=float(np.min(continuation_query)),
        direct_adjoint_num_boost_round=int(direct_adjoint_num_boost_round),
        direct_adjoint_lgb_params=dict(direct_adjoint_lgb_params or {}),
        direct_adjoint_loss=str(_normalize_occupancy_loss(direct_adjoint_loss)),
        direct_adjoint_validation_fraction=float(direct_adjoint_validation_fraction),
        direct_adjoint_early_stopping_rounds=int(direct_adjoint_early_stopping_rounds),
        direct_adjoint_sample_weight_mode=str(direct_adjoint_sample_weight_mode),
        direct_adjoint_sample_weight_max=None
        if direct_adjoint_sample_weight_max is None
        else float(direct_adjoint_sample_weight_max),
        nuisance_crossfit=None if crossfit_context is None else crossfit_context.get("diagnostics", {}),
        iw_prediction_max=iw_prediction_max,
        iw_prediction_power=float(iw_prediction_power),
        iw_normalize_predictions=bool(iw_normalize_predictions),
        iw_prediction_scale=_nuisance_prediction_scale(iw_fit),
        k_prediction_scale=_nuisance_prediction_scale(k_fit),
        iw_density_ratio_loss=iw_density_ratio_loss,
        iw_logistic_logit_clip=iw_logistic_logit_clip,
        iw_prior_correction=float(iw_prior_correction),
        k_density_ratio_loss=k_density_ratio_loss,
        k_logistic_logit_clip=k_logistic_logit_clip,
        k_prior_correction=float(k_prior_correction),
        c_fit=c_fit,
        c_density_ratio_loss=c_density_ratio_loss,
        known_action_ratio=bool(known_iw_query is not None),
        known_action_ratio_features=X_sa_query if known_iw_query is not None else None,
        known_action_ratio_predictions=known_iw_query,
        initial_ratio_mode=str(initial_ratio_mode),
        one_step_ratio_mode=str(one_step_ratio_mode),
        **source_diagnostics,
        **c_diagnostics,
    )


def _legacy_training_prediction_features(result: Dict[str, Any]) -> Optional[Array]:
    """Return exact training feature rows whose stabilized predictions are known."""
    X_query = result.get("X_sa_query")
    if X_query is None:
        return None
    X_query_arr = np.asarray(X_query, dtype=np.float64)
    if X_query_arr.ndim != 2:
        return None
    pred_beh = result.get("pred_state_action_ratio_beh")
    if pred_beh is None:
        return X_query_arr
    n_beh = np.asarray(pred_beh).reshape(-1).shape[0]
    if n_beh <= 0 or n_beh > X_query_arr.shape[0]:
        return X_query_arr
    return np.vstack([X_query_arr, X_query_arr[-n_beh:]])


def _legacy_training_predictions(result: Dict[str, Any]) -> Optional[Array]:
    """Return stabilized predictions paired with ``_legacy_training_prediction_features``."""
    pred_query = result.get("pred_query_stabilized", result.get("pred_query_clipped"))
    if pred_query is None:
        return None
    pred_query_arr = np.asarray(pred_query, dtype=np.float64).reshape(-1)
    pred_beh = result.get("pred_state_action_ratio_beh")
    if pred_beh is None:
        return pred_query_arr
    pred_beh_arr = np.asarray(pred_beh, dtype=np.float64).reshape(-1)
    return np.concatenate([pred_query_arr, pred_beh_arr])


def _replace_known_training_predictions(
    features: Array,
    predictions: Array,
    *,
    known_features: Optional[Array],
    known_predictions: Optional[Array],
) -> Array:
    """Replace exact training rows with the final damped/projected fitted state."""
    if known_features is None or known_predictions is None:
        return predictions
    x = np.asarray(features, dtype=np.float64)
    known_x = np.asarray(known_features, dtype=np.float64)
    known_y = np.asarray(known_predictions, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or known_x.ndim != 2 or x.shape[1] != known_x.shape[1] or known_x.shape[0] != known_y.shape[0]:
        return predictions
    lookup = {tuple(row.tolist()): float(value) for row, value in zip(known_x, known_y)}
    if not lookup:
        return predictions
    out = np.asarray(predictions, dtype=np.float64).reshape(-1).copy()
    for idx, row in enumerate(x):
        value = lookup.get(tuple(row.tolist()))
        if value is not None:
            out[idx] = value
    return out


def _lookup_known_training_predictions(
    features: Array,
    *,
    known_features: Array,
    known_predictions: Array,
) -> Optional[Array]:
    x = np.asarray(features, dtype=np.float64)
    known_x = np.asarray(known_features, dtype=np.float64)
    known_y = np.asarray(known_predictions, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or known_x.ndim != 2 or x.shape[1] != known_x.shape[1] or known_x.shape[0] != known_y.shape[0]:
        return None
    lookup = {tuple(row.tolist()): float(value) for row, value in zip(known_x, known_y)}
    out = np.empty(x.shape[0], dtype=np.float64)
    for idx, row in enumerate(x):
        value = lookup.get(tuple(row.tolist()))
        if value is None:
            return None
        out[idx] = value
    return out


def _nonnegative(x: Array) -> Array:
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def _safe_divide(numerator: Array, denominator: Array, *, eps: float = 1e-12) -> Array:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > eps,
    )


def _project_nonnegative_normalized(
    values: Array,
    reference_weights: Optional[Array] = None,
    max_value: Optional[float] = None,
    normalize: bool = True,
    eps: float = 1e-12,
    *,
    normalization_scale: Optional[float] = None,
    return_info: bool = False,
) -> Array | tuple[Array, Dict[str, float]]:
    """Project ratio estimates onto a nonnegative, optionally bounded scale.

    The occupancy ratio has unit mean under the reference distribution. The
    empirical normalization enforces that moment on the current reference batch;
    passing ``normalization_scale`` lets prediction use a training-time scale
    instead of depending on the arbitrary batch supplied by a caller.
    """
    x_raw = np.asarray(values, dtype=np.float64).reshape(-1)
    cap = None if max_value is None else float(max_value)
    posinf = cap if cap is not None else np.finfo(np.float64).max / 16.0
    negative_fraction = float(np.mean(x_raw < 0.0)) if x_raw.size else 0.0
    nonfinite_fraction = float(np.mean(~np.isfinite(x_raw))) if x_raw.size else 0.0
    x = np.nan_to_num(x_raw, nan=0.0, posinf=posinf, neginf=0.0)
    np.maximum(x, 0.0, out=x)
    clipped_fraction = 0.0
    post_normalization_clipped_fraction = 0.0
    if cap is not None:
        clipped_fraction = float(np.mean(x_raw > cap)) if x.size else 0.0
        np.minimum(x, cap, out=x)

    scale = 1.0
    if normalize:
        if normalization_scale is not None:
            scale = float(normalization_scale)
        else:
            scale = _weighted_mean(x, reference_weights)
        if np.isfinite(scale) and scale > eps:
            x = x / scale
            if cap is not None:
                post_normalization_clipped_fraction = float(np.mean(x > cap)) if x.size else 0.0
                np.minimum(x, cap, out=x)
        else:
            fill = 1.0 if cap is None else min(1.0, cap)
            x = np.full_like(x, fill, dtype=np.float64)
            scale = 0.0

    info = dict(
        projection_clipped_fraction=float(clipped_fraction),
        projection_post_normalization_clipped_fraction=float(post_normalization_clipped_fraction),
        projection_negative_fraction=float(negative_fraction),
        projection_nonfinite_fraction=float(nonfinite_fraction),
        projection_normalization_scale=float(scale),
        projection_max_value=float(cap) if cap is not None else float("nan"),
    )
    if return_info:
        return x.astype(np.float64, copy=False), info
    return x.astype(np.float64, copy=False)


def _weighted_mean(values: Array, weights: Optional[Array]) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    if weights is None:
        return float(np.mean(x))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != x.shape[0]:
        raise ValueError("reference_weights must match values length.")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float(np.mean(x))
    return float(np.sum(w * x) / denom)


def _clip_pseudo_outcomes(
    values: Array,
    *,
    enabled: bool,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    target_min: Optional[float],
    target_max: Optional[float],
) -> tuple[Array, Dict[str, float]]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.nan_to_num(raw, nan=float(pseudo_outcome_min), posinf=0.0, neginf=float(pseudo_outcome_min))
    lower = max(float(pseudo_outcome_min), float(target_min) if target_min is not None else float(pseudo_outcome_min))
    upper_candidates: List[float] = []
    if target_max is not None:
        upper_candidates.append(float(target_max))
    if enabled:
        if pseudo_outcome_max is not None:
            upper_candidates.append(float(pseudo_outcome_max))
        else:
            finite = y[np.isfinite(y)]
            finite = finite[finite >= lower]
            if finite.size:
                upper_candidates.append(float(np.quantile(finite, float(pseudo_outcome_upper_quantile))))
    cap = min(upper_candidates) if upper_candidates else None
    if cap is not None:
        cap = max(float(cap), lower)
    before = y.copy()
    np.maximum(y, lower, out=y)
    if cap is not None:
        np.minimum(y, cap, out=y)
    clipped = before != y
    diag = dict(
        pseudo_outcome_cap=float(cap) if cap is not None else float("nan"),
        pseudo_outcome_min=float(lower),
        pseudo_outcome_clipped_fraction=float(np.mean(clipped)) if clipped.size else 0.0,
        pseudo_outcome_p95=_quantile_or_nan(y, 0.95),
        pseudo_outcome_p99=_quantile_or_nan(y, 0.99),
        pseudo_outcome_max=float(np.max(y)) if y.size else float("nan"),
        pseudo_outcome_mean=float(np.mean(y)) if y.size else float("nan"),
    )
    return y, diag


def _make_stabilized_fixed_point_target(
    *,
    raw_target: Array,
    current: Array,
    eta: float,
    normalize: bool,
    occupancy_ratio_max: Optional[float],
    eps: float,
    clip_pseudo_outcomes: bool,
    pseudo_outcome_max: Optional[float],
    pseudo_outcome_upper_quantile: float,
    pseudo_outcome_min: float,
    target_min: Optional[float],
    target_max: Optional[float],
) -> tuple[Array, Dict[str, Any]]:
    clipped, clip_diag = _clip_pseudo_outcomes(
        raw_target,
        enabled=clip_pseudo_outcomes,
        pseudo_outcome_max=pseudo_outcome_max,
        pseudo_outcome_upper_quantile=pseudo_outcome_upper_quantile,
        pseudo_outcome_min=pseudo_outcome_min,
        target_min=target_min,
        target_max=target_max,
    )
    projected, projection_diag = _project_nonnegative_normalized(
        clipped,
        max_value=occupancy_ratio_max,
        normalize=normalize,
        eps=eps,
        return_info=True,
    )
    damped = _damped_update(current, projected, eta)
    diag: Dict[str, Any] = {}
    diag.update(clip_diag)
    diag.update({f"target_raw_{key}": val for key, val in _summarize_vector(raw_target).items()})
    diag.update({f"target_projected_{key}": val for key, val in _summarize_vector(projected).items()})
    diag.update({f"target_damped_{key}": val for key, val in _summarize_vector(damped).items()})
    diag.update({f"pseudo_{key}": val for key, val in projection_diag.items()})
    return damped, diag


def _damped_update(current: Array, projected_update: Array, eta: float) -> Array:
    current_arr = np.asarray(current, dtype=np.float64)
    update_arr = np.asarray(projected_update, dtype=np.float64)
    return (1.0 - float(eta)) * current_arr + float(eta) * update_arr


def _make_occupancy_sample_weights(
    *,
    mode: str,
    action_ratio: Optional[Array],
    target: Array,
    max_value: Optional[float],
    eps: float = 1e-12,
) -> tuple[Array, Dict[str, float]]:
    mode = str(mode)
    target_arr = _project_nonnegative_normalized(target, max_value=None, normalize=False, eps=eps)
    if mode == "uniform":
        weights = np.ones_like(target_arr, dtype=np.float64)
    elif mode in {"sqrt_action_ratio", "action_ratio"}:
        if action_ratio is None:
            raise ValueError(f"action_ratio is required for occupancy_sample_weight_mode='{mode}'.")
        base = _project_nonnegative_normalized(action_ratio, max_value=max_value, normalize=False, eps=eps)
        weights = np.sqrt(base) if mode == "sqrt_action_ratio" else base
    elif mode in {"sqrt_target", "target"}:
        weights = np.sqrt(target_arr) if mode == "sqrt_target" else target_arr
    else:
        raise ValueError("Unknown occupancy sample-weight mode.")

    weights = np.nan_to_num(weights, nan=1.0, posinf=max_value if max_value is not None else 1.0, neginf=0.0)
    np.maximum(weights, 0.0, out=weights)
    clipped_fraction = 0.0
    if max_value is not None:
        clipped_fraction = float(np.mean(weights > float(max_value))) if weights.size else 0.0
        np.minimum(weights, float(max_value), out=weights)
    mean = float(np.mean(weights)) if weights.size else 0.0
    if np.isfinite(mean) and mean > eps:
        weights = weights / mean
    else:
        weights = np.ones_like(weights, dtype=np.float64)
    if max_value is not None:
        np.minimum(weights, float(max_value), out=weights)
        mean = float(np.mean(weights)) if weights.size else 0.0
        if np.isfinite(mean) and mean > eps:
            weights = weights / mean
            np.minimum(weights, float(max_value), out=weights)
    diag = _summarize_weights(weights)
    diag["sample_weight_clipped_fraction"] = float(clipped_fraction)
    diag["sample_weight_mode"] = mode
    return weights.astype(np.float64, copy=False), diag


def _ess(weights: Array, *, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w)]
    denom = float(np.sum(w**2))
    if w.size == 0 or denom <= eps:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def _summarize_vector(values: Array) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=float("nan"), std=float("nan"), p95=float("nan"), p99=float("nan"), max=float("nan"))
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        max=float(np.max(x)),
    )


def _summarize_ratio_predictions(values: Array) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "clipped_fraction": 0.0,
            "normalization_scale": 1.0,
        }
    return {
        "min": float(np.min(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "clipped_fraction": 0.0,
        "normalization_scale": 1.0,
    }


def _summarize_weights(weights: Array) -> Dict[str, float]:
    summary = _summarize_vector(weights)
    return {
        "sample_weight_mean": summary["mean"],
        "sample_weight_std": summary["std"],
        "sample_weight_p95": summary["p95"],
        "sample_weight_p99": summary["p99"],
        "sample_weight_max": summary["max"],
    }


def _quantile_or_nan(values: Array, q: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def _fixed_point_history_diag(
    *,
    current_query: Array,
    next_query: Array,
    current_beh: Array,
    next_beh: Array,
    train_idx: Array,
    test_idx: Array,
    raw_update: Array,
    projected_update: Array,
    damped_update: Array,
    target: Array,
    target_diag: Dict[str, Any],
    eval_target_diag: Dict[str, Any],
    query_projection_diag: Dict[str, float],
    beh_projection_diag: Dict[str, float],
    sample_weight_diag: Dict[str, float],
    eta: float,
    occupancy_ratio_max: Optional[float],
    eps: float,
) -> Dict[str, Any]:
    diff = np.abs(np.asarray(next_query, dtype=np.float64) - np.asarray(current_query, dtype=np.float64))
    current_abs = np.abs(np.asarray(current_query, dtype=np.float64))
    out: Dict[str, Any] = dict(
        fixed_point_abs_change_train=float(np.mean(diff[train_idx])),
        fixed_point_rel_change_train=float(np.mean(diff[train_idx]) / (np.mean(current_abs[train_idx]) + eps)),
        fixed_point_damping=float(eta),
        occupancy_ratio_max=None if occupancy_ratio_max is None else float(occupancy_ratio_max),
        projection_clipped_fraction=float(query_projection_diag.get("projection_clipped_fraction", 0.0)),
        projection_clipped_fraction_beh=float(beh_projection_diag.get("projection_clipped_fraction", 0.0)),
        projection_post_normalization_clipped_fraction=float(
            query_projection_diag.get("projection_post_normalization_clipped_fraction", 0.0)
        ),
        projection_negative_fraction=float(query_projection_diag.get("projection_negative_fraction", 0.0)),
        projection_negative_fraction_beh=float(beh_projection_diag.get("projection_negative_fraction", 0.0)),
        projection_nonfinite_fraction=float(query_projection_diag.get("projection_nonfinite_fraction", 0.0)),
        projection_normalization_scale=float(query_projection_diag.get("projection_normalization_scale", 1.0)),
    )
    if test_idx.size:
        out["fixed_point_abs_change_eval"] = float(np.mean(diff[test_idx]))
        out["fixed_point_rel_change_eval"] = float(np.mean(diff[test_idx]) / (np.mean(current_abs[test_idx]) + eps))

    ess = _ess(next_beh, eps=eps)
    out["ess"] = float(ess)
    out["ess_fraction"] = float(ess / max(np.asarray(next_beh).size, 1))
    out["weight_max"] = float(np.max(next_beh)) if np.asarray(next_beh).size else float("nan")
    out["weight_p95"] = _quantile_or_nan(next_beh, 0.95)
    out["weight_p99"] = _quantile_or_nan(next_beh, 0.99)
    out.update({f"raw_update_{key}": val for key, val in _summarize_vector(raw_update).items()})
    out.update({f"projected_update_{key}": val for key, val in _summarize_vector(projected_update).items()})
    out.update({f"damped_update_{key}": val for key, val in _summarize_vector(damped_update).items()})
    out.update({f"target_{key}": val for key, val in _summarize_vector(target).items()})
    out.update(target_diag)
    out.update({f"eval_{key}": val for key, val in eval_target_diag.items()})
    out.update(sample_weight_diag)
    return _json_friendly_dict(out)


def _json_friendly_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, (bool, str, int, float)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _combine_builder_diags(diags: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = sorted({key for diag in diags for key in diag})
    for key in keys:
        vals = []
        for diag in diags:
            value = diag.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                vals.append(float(value))
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _mse(x: Array, y: Array) -> float:
    return float(np.mean((x - y) ** 2))


@dataclass
class _BatchCache:
    j0: int
    j1: int
    mb: int
    n_flat: int
    idx_flat: Array
    k_flat: Array


def make_crossfit_forward_occupancy_dataset(
    *,
    crossfit_context: Dict[str, Any],
    X_sa_kernel: Array,
    X_s_query: Array,
    X_sa_iw: Array,
    X_sa_query_iw: Array,
    gamma: float,
    mcmc_samples: int = 100,
    seed: int = 123,
    batch_query: int = 500,
    clip_w_query_max: Optional[float] = 50.0,
    action_prediction_power: float = 1.0,
    action_normalize_predictions: bool = False,
    action_density_ratio_loss: str = "lsif",
    action_logistic_logit_clip: Optional[float] = 20.0,
    clip_k_max: Optional[float] = 50.0,
    transition_prediction_power: float = 1.0,
    transition_normalize_predictions: bool = False,
    transition_density_ratio_loss: str = "lsif",
    transition_logistic_logit_clip: Optional[float] = 20.0,
    normalize_transition_cache: bool = False,
    transition_cache_norm_eps: float = 1e-12,
    w_source_query: Optional[Array] = None,
    source_state_ratio_query: Optional[Array] = None,
    continuation_query: Optional[Array] = None,
    w_query_override: Optional[Array] = None,
) -> TargetBuilder:
    """Create a fold-aware target builder using held-out nuisance models by query row."""
    n = X_sa_kernel.shape[0]
    q = X_s_query.shape[0]
    y_full = np.empty(q, dtype=np.float64)
    w_query_full = np.empty(q, dtype=np.float64)
    folds: List[Array] = list(crossfit_context["folds"])
    builders = []
    query_indices = []
    for fold_id, fold_idx in enumerate(folds):
        query_idx = np.concatenate([fold_idx, n + fold_idx]).astype(np.int64, copy=False)
        query_indices.append(query_idx)
        iw_model = crossfit_context["iw_models"][fold_id]
        k_model = crossfit_context["k_models"][fold_id]
        builders.append(
            make_forward_occupancy_dataset(
                bst_k=k_model["booster"],
                bst_iw=iw_model["booster"],
                k_prediction_offset=float(k_model["offset"]),
                iw_prediction_offset=float(iw_model["offset"]),
                X_sa_kernel=X_sa_kernel,
                X_s_query=X_s_query[query_idx],
                X_sa_iw=X_sa_iw,
                X_sa_query_iw=X_sa_query_iw[query_idx],
                gamma=gamma,
                mcmc_samples=mcmc_samples,
                seed=seed + 10_007 * (fold_id + 1),
                batch_query=batch_query,
                clip_w_query_max=clip_w_query_max,
                action_prediction_power=action_prediction_power,
                action_normalize_predictions=action_normalize_predictions,
                action_prediction_scale=float(iw_model.get("scale", 1.0)),
                action_density_ratio_loss=str(iw_model.get("density_ratio_loss", action_density_ratio_loss)),
                action_logistic_logit_clip=iw_model.get("logistic_logit_clip", action_logistic_logit_clip),
                action_prior_correction=float(iw_model.get("prior_correction", 1.0)),
                clip_k_max=clip_k_max,
                transition_prediction_power=transition_prediction_power,
                transition_normalize_predictions=transition_normalize_predictions,
                transition_prediction_scale=float(k_model.get("scale", 1.0)),
                transition_density_ratio_loss=str(k_model.get("density_ratio_loss", transition_density_ratio_loss)),
                transition_logistic_logit_clip=k_model.get("logistic_logit_clip", transition_logistic_logit_clip),
                transition_prior_correction=float(k_model.get("prior_correction", 1.0)),
                normalize_transition_cache=normalize_transition_cache,
                transition_cache_norm_eps=transition_cache_norm_eps,
                w_source_query=None
                if w_source_query is None
                else np.asarray(w_source_query, dtype=np.float64)[query_idx],
                source_state_ratio_query=None
                if source_state_ratio_query is None
                else np.asarray(source_state_ratio_query, dtype=np.float64)[query_idx],
                continuation_query=None
                if continuation_query is None
                else np.asarray(continuation_query, dtype=np.float64)[query_idx],
                w_query_override=None
                if w_query_override is None
                else np.asarray(w_query_override, dtype=np.float64)[query_idx],
            )
        )

    def build_iteration_targets(
        *,
        w_beh: Array,
        w_old_query: Array,
        eta: float = 1.0,
        clip_y_min: Optional[float] = 0.0,
        clip_y_max: Optional[float] = None,
    ) -> Dict[str, Any]:
        diag_rows = []
        for query_idx, builder in zip(query_indices, builders):
            out = builder(
                w_beh=w_beh,
                w_old_query=np.asarray(w_old_query)[query_idx],
                eta=eta,
                clip_y_min=clip_y_min,
                clip_y_max=clip_y_max,
            )
            y_full[query_idx] = out["y"]
            w_query_full[query_idx] = out["w_query"]
            diag_rows.append(out.get("diag", {}))
        return dict(
            X=X_sa_query_iw,
            y=y_full.copy(),
            w_query=w_query_full.copy(),
            diag=_combine_builder_diags(diag_rows) | {"crossfit_target_builder": True},
        )

    return build_iteration_targets


def make_forward_occupancy_dataset(
    *,
    bst_k: lgb.Booster,
    bst_iw: Optional[lgb.Booster],
    k_prediction_offset: float = 0.0,
    iw_prediction_offset: float = 0.0,
    X_sa_kernel: Array,
    X_s_query: Array,
    X_sa_iw: Array,
    X_sa_query_iw: Array,
    gamma: float,
    mcmc_samples: int = 100,
    seed: int = 123,
    batch_query: int = 500,
    clip_w_query_max: Optional[float] = 50.0,
    action_prediction_power: float = 1.0,
    action_normalize_predictions: bool = False,
    action_prediction_scale: float = 1.0,
    action_density_ratio_loss: str = "lsif",
    action_logistic_logit_clip: Optional[float] = 20.0,
    action_prior_correction: float = 1.0,
    clip_k_max: Optional[float] = 50.0,
    transition_prediction_power: float = 1.0,
    transition_normalize_predictions: bool = False,
    transition_prediction_scale: float = 1.0,
    transition_density_ratio_loss: str = "lsif",
    transition_logistic_logit_clip: Optional[float] = 20.0,
    transition_prior_correction: float = 1.0,
    normalize_transition_cache: bool = False,
    transition_cache_norm_eps: float = 1e-12,
    w_source_query: Optional[Array] = None,
    source_state_ratio_query: Optional[Array] = None,
    continuation_query: Optional[Array] = None,
    w_query_override: Optional[Array] = None,
    pred_num_threads: Optional[int] = None,
) -> TargetBuilder:
    """Create a reusable Monte Carlo target builder for occupancy boosting.

    The expensive draws and transition-ratio predictions are cached once per
    builder. Each call then only gathers current behavior-row occupancy weights
    and evaluates the forward fixed-point target.
    """
    rng = np.random.default_rng(seed)
    X_sa_kernel = np.asarray(X_sa_kernel, dtype=np.float32, order="C")
    X_s_query = _as_2d(np.asarray(X_s_query, dtype=np.float32, order="C"), "X_s_query")
    X_sa_iw = np.asarray(X_sa_iw, dtype=np.float32, order="C")
    X_sa_query_iw = np.asarray(X_sa_query_iw, dtype=np.float32, order="C")

    n = X_sa_kernel.shape[0]
    q = X_s_query.shape[0]
    if X_sa_iw.shape[0] != n:
        raise ValueError("X_sa_iw must have same number of rows as X_sa_kernel.")
    if X_sa_query_iw.shape[0] != q:
        raise ValueError("X_sa_query_iw must have same number of rows as X_s_query.")

    draws_per_query = int(mcmc_samples)
    if draws_per_query <= 0:
        raise ValueError("mcmc_samples must be positive.")

    if w_query_override is not None:
        w_query = _checked_vector(w_query_override, q, "w_query_override", dtype=np.float64)
        w_query_summary = _summarize_ratio_predictions(w_query)
    else:
        if bst_iw is None:
            raise ValueError("bst_iw is required unless w_query_override is supplied.")
        w_query_raw = _predict_nonnegative(
            bst_iw,
            X_sa_query_iw,
            offset=iw_prediction_offset,
            density_ratio_loss=action_density_ratio_loss,
            logistic_logit_clip=action_logistic_logit_clip,
            prior_correction=action_prior_correction,
            num_threads=pred_num_threads,
        )
        w_query, w_query_summary = _postprocess_ratio_predictions(
            w_query_raw,
            clip_nonneg=True,
            prediction_max=clip_w_query_max,
            prediction_power=action_prediction_power,
            normalize_predictions=action_normalize_predictions,
        )
        w_query = w_query * float(action_prediction_scale)
    w_query = w_query.astype(np.float32)
    continuation = np.ones(q, dtype=np.float32) if continuation_query is None else _checked_vector(
        continuation_query,
        q,
        "continuation_query",
        dtype=np.float32,
    )
    w_source = _prepare_source_weights(
        w_query=w_query,
        w_source_query=w_source_query,
        source_state_ratio_query=source_state_ratio_query,
        q=q,
    )
    source_summary = _source_state_ratio_summary(source_state_ratio_query, q=q)

    caches = _build_transition_caches(
        bst_k=bst_k,
        k_prediction_offset=k_prediction_offset,
        X_sa_kernel=X_sa_kernel,
        X_s_query=X_s_query,
        n=n,
        q=q,
        draws_per_query=draws_per_query,
        batch_query=batch_query,
        rng=rng,
        clip_k_max=clip_k_max,
        transition_prediction_power=transition_prediction_power,
        transition_normalize_predictions=transition_normalize_predictions,
        transition_prediction_scale=transition_prediction_scale,
        transition_density_ratio_loss=transition_density_ratio_loss,
        transition_logistic_logit_clip=transition_logistic_logit_clip,
        transition_prior_correction=transition_prior_correction,
        normalize_transition_cache=normalize_transition_cache,
        transition_cache_norm_eps=transition_cache_norm_eps,
        pred_num_threads=pred_num_threads,
    )

    max_mb = min(int(batch_query), q)
    numer_buf = np.empty(q, dtype=np.float32)
    y_buf = np.empty(q, dtype=np.float32)
    tmp_buf = np.empty(q, dtype=np.float32)
    w_take_buf = np.empty(max_mb * draws_per_query, dtype=np.float32)
    prod_flat_buf = np.empty(max_mb * draws_per_query, dtype=np.float32)
    gamma32 = np.float32(gamma)
    one_minus_gamma32 = np.float32(1.0 - gamma)

    def build_iteration_targets(
        *,
        w_beh: Array,
        w_old_query: Array,
        eta: float = 1.0,
        clip_y_min: Optional[float] = 0.0,
        clip_y_max: Optional[float] = None,
    ) -> Dict[str, Any]:
        w_beh32 = _checked_vector(w_beh, n, "w_beh", dtype=np.float32)
        np.maximum(w_beh32, np.float32(0.0), out=w_beh32)
        w_old_query32 = _checked_vector(w_old_query, q, "w_old_query", dtype=np.float32)

        for cache in caches:
            np.take(w_beh32, cache.idx_flat, out=w_take_buf[: cache.n_flat])
            np.multiply(w_take_buf[: cache.n_flat], cache.k_flat, out=prod_flat_buf[: cache.n_flat])
            numer_buf[cache.j0 : cache.j1] = prod_flat_buf[: cache.n_flat].reshape(
                cache.mb,
                draws_per_query,
            ).mean(axis=1)

        np.multiply(w_query, numer_buf, out=y_buf)
        np.multiply(y_buf, continuation, out=y_buf)
        np.multiply(y_buf, gamma32, out=y_buf)
        np.multiply(w_source, one_minus_gamma32, out=tmp_buf)
        np.add(y_buf, tmp_buf, out=y_buf)

        if eta < 1.0:
            eta32 = np.float32(eta)
            np.multiply(y_buf, eta32, out=y_buf)
            np.multiply(w_old_query32, np.float32(1.0 - eta), out=tmp_buf)
            np.add(y_buf, tmp_buf, out=y_buf)

        if clip_y_min is not None:
            np.maximum(y_buf, np.float32(clip_y_min), out=y_buf)
        if clip_y_max is not None:
            np.minimum(y_buf, np.float32(clip_y_max), out=y_buf)

        y = y_buf.astype(np.float64, copy=True)
        return dict(
            X=X_sa_query_iw,
            y=y,
            w_query=w_query.astype(np.float64, copy=True),
            diag=dict(
                mean_target=float(np.mean(y)),
                min_target=float(np.min(y)),
                max_target=float(np.max(y)),
                target_p95=float(np.quantile(y, 0.95)),
                target_p99=float(np.quantile(y, 0.99)),
                mean_w_query=float(np.mean(w_query)),
                source_state_ratio_enabled=bool(source_state_ratio_query is not None),
                source_state_ratio_mean=float(source_summary["mean"]),
                source_state_ratio_max=float(source_summary["max"]),
                source_state_ratio_ess_fraction=float(source_summary["ess_fraction"]),
                continuation_mean=float(np.mean(continuation)),
                continuation_min=float(np.min(continuation)),
                w_query_min=float(w_query_summary["min"]),
                w_query_p50=float(w_query_summary["p50"]),
                w_query_p90=float(w_query_summary["p90"]),
                w_query_p95=float(w_query_summary["p95"]),
                w_query_p99=float(w_query_summary["p99"]),
                w_query_max=float(w_query_summary["max"]),
                w_query_clipped_fraction=float(w_query_summary["clipped_fraction"]),
                mean_forward_numer=float(np.mean(numer_buf)),
            ),
        )

    return build_iteration_targets


def make_direct_adjoint_occupancy_dataset(
    *,
    X_sa_successor: Array,
    X_sa_query: Array,
    c_ratio_query: Array,
    w_source_query: Array,
    continuation_query: Optional[Array] = None,
    successor_row_index: Optional[Array] = None,
    gamma: float,
    seed: int = 123,
    num_boost_round: int = 1,
    lgb_params: Optional[Dict[str, Any]] = None,
    loss: str = "squared",
    validation_fraction: float = 0.2,
    early_stopping_rounds: int = 0,
    sample_weight_mode: str = "uniform",
    sample_weight_max: Optional[float] = 50.0,
) -> TargetBuilder:
    """Create direct FORI targets using ``c_pi`` and an adjoint regression."""
    X_sa_successor = np.asarray(X_sa_successor, dtype=np.float32, order="C")
    X_sa_query = np.asarray(X_sa_query, dtype=np.float32, order="C")
    if X_sa_successor.ndim != 2 or X_sa_query.ndim != 2:
        raise ValueError("X_sa_successor and X_sa_query must be 2D arrays.")
    if X_sa_successor.shape[1] != X_sa_query.shape[1]:
        raise ValueError("X_sa_successor and X_sa_query must have the same feature dimension.")
    n = X_sa_successor.shape[0]
    q = X_sa_query.shape[0]
    successor_index = (
        np.arange(n, dtype=np.int64)
        if successor_row_index is None
        else np.asarray(successor_row_index, dtype=np.int64).reshape(-1)
    )
    if successor_index.shape[0] != n:
        raise ValueError("successor_row_index must match X_sa_successor rows.")
    c_query = _checked_vector(c_ratio_query, q, "c_ratio_query", dtype=np.float32)
    np.maximum(c_query, np.float32(0.0), out=c_query)
    w_source = _checked_vector(w_source_query, q, "w_source_query", dtype=np.float32)
    np.maximum(w_source, np.float32(0.0), out=w_source)
    continuation = np.ones(q, dtype=np.float32) if continuation_query is None else _checked_vector(
        continuation_query,
        q,
        "continuation_query",
        dtype=np.float32,
    )
    params_base = _default_occupancy_lgb_params(seed=int(seed))
    if lgb_params is not None:
        params_base.update(dict(lgb_params))
    loss_name = _normalize_occupancy_loss(loss)
    params_base["objective"] = "regression"
    params_base["verbose"] = -1
    rounds = max(1, int(num_boost_round))
    use_validation = int(early_stopping_rounds) > 0 and float(validation_fraction) > 0.0 and n > 1
    if use_validation:
        adj_train_idx, adj_valid_idx = _make_train_test_indices(
            n_rows=n,
            early_stopping=True,
            test_frac=float(validation_fraction),
            seed=int(seed) + 71_003,
        )
    else:
        adj_train_idx = np.arange(n, dtype=np.int64)
        adj_valid_idx = np.array([], dtype=np.int64)
    call_count = 0

    def build_iteration_targets(
        *,
        w_beh: Array,
        w_old_query: Optional[Array] = None,
        eta: float = 1.0,
        clip_y_min: Optional[float] = 0.0,
        clip_y_max: Optional[float] = None,
    ) -> Dict[str, Any]:
        del w_old_query, eta
        nonlocal call_count
        call_count += 1
        w_beh_base = np.asarray(w_beh, dtype=np.float32).reshape(-1)
        if np.max(successor_index, initial=-1) >= w_beh_base.shape[0]:
            raise ValueError("successor_row_index references rows outside w_beh.")
        w_beh32 = w_beh_base[successor_index].astype(np.float32, copy=False)
        np.maximum(w_beh32, np.float32(0.0), out=w_beh32)
        adj_weights, adj_weight_diag = _make_occupancy_sample_weights(
            mode=sample_weight_mode,
            action_ratio=None,
            target=w_beh32,
            max_value=sample_weight_max,
        )
        params = dict(params_base)
        params["seed"] = int(seed) + 997 * call_count
        train_weight = None if str(sample_weight_mode) == "uniform" else adj_weights[adj_train_idx]
        dtrain = lgb.Dataset(
            X_sa_successor[adj_train_idx],
            label=w_beh32[adj_train_idx],
            weight=train_weight,
            free_raw_data=False,
        )
        valid_sets = None
        valid_names = None
        callbacks = None
        if loss_name == "huber":
            huber_delta = _resolve_huber_delta(
                np.full(adj_train_idx.shape[0], float(np.mean(w_beh32[adj_train_idx])), dtype=np.float64)
                - w_beh32[adj_train_idx].astype(np.float64),
                loss=loss_name,
                huber_delta=None,
                huber_delta_scale=1.345,
                huber_delta_quantile_power=0.25,
                huber_delta_min_quantile=0.80,
            )
            params["objective"] = _make_occupancy_objective(
                loss=loss_name,
                huber_delta=huber_delta,
                huber_hessian_floor=1e-3,
            )
            params.setdefault("metric", "l2")
        if use_validation:
            dvalid = lgb.Dataset(
                X_sa_successor[adj_valid_idx],
                label=w_beh32[adj_valid_idx],
                free_raw_data=False,
                reference=dtrain,
            )
            valid_sets = [dvalid]
            valid_names = ["valid"]
            callbacks = [lgb.early_stopping(int(early_stopping_rounds), verbose=False)]
        model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        best_iteration = int(getattr(model, "best_iteration", 0) or rounds)
        m_query = model.predict(X_sa_query).astype(np.float32, copy=False)
        np.maximum(m_query, np.float32(0.0), out=m_query)
        y = np.float32(1.0 - gamma) * w_source + np.float32(gamma) * continuation * c_query * m_query
        if clip_y_min is not None:
            np.maximum(y, np.float32(clip_y_min), out=y)
        if clip_y_max is not None:
            np.minimum(y, np.float32(clip_y_max), out=y)
        y64 = y.astype(np.float64, copy=True)
        return dict(
            X=X_sa_query,
            y=y64,
            w_query=c_query.astype(np.float64, copy=True),
            diag=dict(
                one_step_direct_ratio_enabled=True,
                direct_adjoint_num_boost_round=float(rounds),
                direct_adjoint_best_iteration=float(best_iteration),
                direct_adjoint_loss=str(loss_name),
                direct_adjoint_validation_fraction=float(validation_fraction),
                direct_adjoint_early_stopping_rounds=float(early_stopping_rounds),
                direct_adjoint_sample_weight_mode=str(sample_weight_mode),
                direct_adjoint_sample_weight_max=float(sample_weight_max) if sample_weight_max is not None else float("nan"),
                continuation_mean=float(np.mean(continuation)),
                continuation_min=float(np.min(continuation)),
                initial_joint_or_factored_source_mean=float(np.mean(w_source)),
                mean_target=float(np.mean(y64)),
                min_target=float(np.min(y64)),
                max_target=float(np.max(y64)),
                target_p95=float(np.quantile(y64, 0.95)),
                target_p99=float(np.quantile(y64, 0.99)),
                mean_w_query=float(np.mean(c_query)),
                w_query_min=float(np.min(c_query)),
                w_query_p50=float(np.quantile(c_query, 0.50)),
                w_query_p90=float(np.quantile(c_query, 0.90)),
                w_query_p95=float(np.quantile(c_query, 0.95)),
                w_query_p99=float(np.quantile(c_query, 0.99)),
                w_query_max=float(np.max(c_query)),
                w_query_clipped_fraction=0.0,
                mean_forward_numer=float(np.mean(m_query)),
                **adj_weight_diag,
            ),
        )

    return build_iteration_targets


def _predict_nonnegative(
    booster: lgb.Booster,
    X: Array,
    *,
    offset: float = 0.0,
    density_ratio_loss: str = "lsif",
    logistic_logit_clip: Optional[float] = 20.0,
    prior_correction: float = 1.0,
    num_threads: Optional[int] = None,
) -> Array:
    pred = _predict_ratio_from_booster(
        booster=booster,
        X=X,
        offset=float(offset),
        density_ratio_loss=str(density_ratio_loss),
        logistic_logit_clip=logistic_logit_clip,
        prior_correction=float(prior_correction),
        num_threads=num_threads,
    )
    return _nonnegative(pred)


def _prepare_source_weights(
    *,
    w_query: Array,
    w_source_query: Optional[Array],
    source_state_ratio_query: Optional[Array],
    q: int,
) -> Array:
    if w_source_query is not None and source_state_ratio_query is not None:
        raise ValueError("Provide either w_source_query or source_state_ratio_query, not both.")
    if w_source_query is None and source_state_ratio_query is None:
        return w_query
    if source_state_ratio_query is not None:
        source_state = _checked_vector(source_state_ratio_query, q, "source_state_ratio_query", dtype=np.float32)
        np.maximum(source_state, np.float32(0.0), out=source_state)
        return (np.asarray(w_query, dtype=np.float32) * source_state).astype(np.float32, copy=False)
    w_source = _checked_vector(w_source_query, q, "w_source_query", dtype=np.float32)
    np.maximum(w_source, np.float32(0.0), out=w_source)
    return w_source


def _source_state_ratio_summary(source_state_ratio_query: Optional[Array], *, q: int) -> Dict[str, float]:
    if source_state_ratio_query is None:
        return dict(mean=1.0, max=1.0, ess_fraction=1.0)
    values = _checked_vector(source_state_ratio_query, q, "source_state_ratio_query", dtype=np.float64)
    np.maximum(values, 0.0, out=values)
    return dict(
        mean=float(np.mean(values)) if values.size else float("nan"),
        max=float(np.max(values)) if values.size else float("nan"),
        ess_fraction=float(_ess(values) / max(values.size, 1)),
    )


def _build_transition_caches(
    *,
    bst_k: lgb.Booster,
    k_prediction_offset: float,
    X_sa_kernel: Array,
    X_s_query: Array,
    n: int,
    q: int,
    draws_per_query: int,
    batch_query: int,
    rng: np.random.Generator,
    clip_k_max: Optional[float],
    transition_prediction_power: float,
    transition_normalize_predictions: bool,
    transition_prediction_scale: float,
    transition_density_ratio_loss: str,
    transition_logistic_logit_clip: Optional[float],
    transition_prior_correction: float,
    normalize_transition_cache: bool,
    transition_cache_norm_eps: float,
    pred_num_threads: Optional[int],
) -> List[_BatchCache]:
    d_sa = X_sa_kernel.shape[1]
    d_state = X_s_query.shape[1]
    max_mb = min(int(batch_query), q)
    X_sa_flat_buf = np.empty((max_mb * draws_per_query, d_sa), dtype=np.float32)
    Xk_buf = np.empty((max_mb * draws_per_query, d_sa + d_state), dtype=np.float32)
    caches: List[_BatchCache] = []

    for j0 in range(0, q, batch_query):
        j1 = min(q, j0 + batch_query)
        mb = j1 - j0
        n_flat = mb * draws_per_query
        idx_flat = rng.integers(0, n, size=n_flat, endpoint=False).astype(np.int32, copy=False)

        X_sa_flat_buf[:n_flat, :] = X_sa_kernel[idx_flat, :]
        Xk_buf[:n_flat, :d_sa] = X_sa_flat_buf[:n_flat, :]
        s_batch = X_s_query[j0:j1, :]
        for row in range(mb):
            lo = row * draws_per_query
            hi = lo + draws_per_query
            Xk_buf[lo:hi, d_sa:] = s_batch[row, :]

        k_raw = _predict_nonnegative(
            bst_k,
            Xk_buf[:n_flat, :],
            offset=k_prediction_offset,
            density_ratio_loss=transition_density_ratio_loss,
            logistic_logit_clip=transition_logistic_logit_clip,
            prior_correction=transition_prior_correction,
            num_threads=pred_num_threads,
        )
        k_flat, _ = _postprocess_ratio_predictions(
            k_raw,
            clip_nonneg=True,
            prediction_max=clip_k_max,
            prediction_power=transition_prediction_power,
            normalize_predictions=transition_normalize_predictions,
        )
        k_flat = k_flat * float(transition_prediction_scale)
        k_flat = k_flat.astype(np.float32)

        caches.append(
            _BatchCache(
                j0=j0,
                j1=j1,
                mb=mb,
                n_flat=n_flat,
                idx_flat=idx_flat,
                k_flat=k_flat,
            )
        )

    if normalize_transition_cache:
        if transition_cache_norm_eps <= 0.0:
            raise ValueError("transition_cache_norm_eps must be positive.")
        # k(s,a,s') integrates to one under s' ~ rho0 by definition. When the
        # cached query states are sampled from that rho0 reference distribution,
        # normalizing each source row's empirical average enforces this moment
        # in the Monte Carlo operator without changing the fixed-point form.
        sums = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for cache in caches:
            np.add.at(sums, cache.idx_flat, cache.k_flat.astype(np.float64, copy=False))
            np.add.at(counts, cache.idx_flat, 1.0)
        means = np.divide(sums, counts, out=np.ones_like(sums), where=counts > 0.0)
        means = np.maximum(means, float(transition_cache_norm_eps))
        for cache in caches:
            cache.k_flat[:] = (cache.k_flat.astype(np.float64, copy=False) / means[cache.idx_flat]).astype(
                np.float32,
                copy=False,
            )

    return caches


def _checked_vector(x: Array, length: int, name: str, *, dtype: Any) -> Array:
    x = np.asarray(x, dtype=dtype).reshape(-1)
    if x.shape[0] != length:
        raise ValueError(f"{name} must have length {length}.")
    return x.copy()


def _checked_optional_query_vector(x: Optional[Array], length: int, name: str) -> Optional[Array]:
    if x is None:
        return None
    return _checked_vector(x, length, name, dtype=np.float64)
