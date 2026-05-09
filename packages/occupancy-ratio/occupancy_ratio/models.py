"""Boosted fitted model classes and prediction/serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np

from occupancy_ratio.fit_importance_and_transition_ratios import (
    _postprocess_ratio_predictions,
    _predict_ratio_from_booster,
)
from occupancy_ratio.stabilization import _project_nonnegative_normalized, _safe_divide
from occupancy_ratio.validation import _as_2d


Array = np.ndarray

__all__ = [
    "DiscountedOccupancyRatioModel",
]


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


DiscountedOccupancyRatioModel.__module__ = "occupancy_ratio.fit_occupancy_ratio"
