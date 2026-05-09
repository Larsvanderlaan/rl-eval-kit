from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np

from occupancy_ratio.fit_occupancy_ratio import (
    Array,
    _as_2d,
    _ess,
    _safe_divide,
    _validate_aligned_inputs,
    _validate_initial_action_inputs,
    _validate_initial_state_inputs,
    _validate_next_target_actions,
)


__all__ = [
    "GoogleDualDICEConfig",
    "GoogleDualDICEOccupancyRatioModel",
    "GoogleDualDICEPreflight",
    "fit_google_dualdice_occupancy_ratio",
    "preflight_google_dualdice",
]


@dataclass(frozen=True)
class GoogleDualDICEConfig:
    """Configuration for the optional official Google Research DualDICE backend."""

    google_research_path: str | Path = Path("/tmp/google-research")
    num_updates: int = 1000
    batch_size: int = 128
    weight_decay: float = 1e-5
    seed: int = 123
    prediction_max: Optional[float] = None
    normalize_predictions: bool = False
    limit_tf_threads: bool = True

    def __post_init__(self) -> None:
        if int(self.num_updates) <= 0:
            raise ValueError("num_updates must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be nonnegative.")
        if self.prediction_max is not None and float(self.prediction_max) <= 0.0:
            raise ValueError("prediction_max must be positive when supplied.")


@dataclass(frozen=True)
class GoogleDualDICEPreflight:
    available: bool
    reason: str
    repo_path: Path


@dataclass
class GoogleDualDICEOccupancyRatioModel:
    """Official Google DualDICE zeta model with occupancy-ratio-style helpers."""

    model: Any
    gamma: float
    state_dim: int
    action_dim: int
    diagnostics: dict[str, Any]
    history: list[dict[str, float]]
    config: GoogleDualDICEConfig

    def predict_state_action_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        """Predict Google DualDICE ``zeta(s, a)`` on query state-action rows."""
        tf = _load_tensorflow_for_google_dualdice()
        features = self._state_action_features(states, actions)
        states_tf = tf.convert_to_tensor(features[:, : self.state_dim], dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(features[:, self.state_dim :], dtype=tf.float32)
        raw = self.model.zeta(states_tf, actions_tf).numpy().astype(np.float64).reshape(-1)
        if not clip:
            return raw
        return _postprocess_google_zeta(
            raw,
            prediction_max=self.config.prediction_max,
            normalize=self.config.normalize_predictions,
        )

    def predict_action_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
        """Return ones because Google DualDICE does not fit a separate action ratio."""
        features = self._state_action_features(states, actions)
        return np.ones(features.shape[0], dtype=np.float64)

    def predict_state_ratio(self, states: Array, actions: Array, *, clip: bool = True) -> Array:
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
    ) -> dict[str, Array]:
        out = dict(
            target_state_action_ratio=self.predict_state_action_ratio(states, target_actions, clip=clip),
            target_action_ratio=self.predict_action_ratio(states, target_actions, clip=clip),
        )
        out["target_state_ratio"] = _safe_divide(
            out["target_state_action_ratio"],
            out["target_action_ratio"],
        )
        if observed_actions is not None:
            out["observed_state_action_ratio"] = self.predict_state_action_ratio(states, observed_actions, clip=clip)
            out["observed_action_ratio"] = self.predict_action_ratio(states, observed_actions, clip=clip)
            out["observed_state_ratio"] = _safe_divide(
                out["observed_state_action_ratio"],
                out["observed_action_ratio"],
            )
        return out

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(
            google_dualdice_model=self.model,
            gamma=float(self.gamma),
            history=list(self.history),
            diagnostics=dict(self.diagnostics),
            config=self.config,
        )

    def _state_action_features(self, states: Array, actions: Array) -> Array:
        states_2d = _as_2d(states, "states").astype(np.float32, copy=False)
        actions_2d = _as_2d(actions, "actions").astype(np.float32, copy=False)
        if states_2d.shape[0] != actions_2d.shape[0]:
            raise ValueError("states and actions must have the same number of rows.")
        if states_2d.shape[1] != self.state_dim:
            raise ValueError(f"states must have {self.state_dim} columns.")
        if actions_2d.shape[1] != self.action_dim:
            raise ValueError(f"actions must have {self.action_dim} columns.")
        return np.concatenate([states_2d, actions_2d], axis=1)


def preflight_google_dualdice(
    google_research_path: str | Path = Path("/tmp/google-research"),
) -> GoogleDualDICEPreflight:
    """Check whether the optional Google Research DualDICE backend can run."""
    path = Path(google_research_path)
    if not (path / "policy_eval" / "dual_dice.py").exists():
        return GoogleDualDICEPreflight(
            available=False,
            reason=(
                f"Missing Google DualDICE source at {path / 'policy_eval' / 'dual_dice.py'}. "
                "Clone https://github.com/google-research/google-research and pass google_research_path."
            ),
            repo_path=path,
        )
    try:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        _load_tensorflow_for_google_dualdice()
        from policy_eval.dual_dice import DualDICE  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return GoogleDualDICEPreflight(
            available=False,
            reason=f"Google DualDICE import failed: {type(exc).__name__}: {exc}",
            repo_path=path,
        )
    return GoogleDualDICEPreflight(available=True, reason="", repo_path=path)


def fit_google_dualdice_occupancy_ratio(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Array,
    initial_actions: Array,
    target_next_actions: Array,
    terminals: Optional[Array] = None,
    sample_weight: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    config: Optional[GoogleDualDICEConfig] = None,
    google_research_path: str | Path | None = None,
    num_updates: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> GoogleDualDICEOccupancyRatioModel:
    """Fit official Google DualDICE through the occupancy-ratio API.

    The core arguments mirror :func:`fit_discounted_occupancy_ratio`. Unlike the
    iterative estimators, Google DualDICE directly estimates zeta and does not
    expose separate action-ratio or transition-ratio nuisance models.
    """
    cfg = GoogleDualDICEConfig() if config is None else config
    overrides: dict[str, Any] = {}
    if google_research_path is not None:
        overrides["google_research_path"] = google_research_path
    if num_updates is not None:
        overrides["num_updates"] = int(num_updates)
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    if seed is not None:
        overrides["seed"] = int(seed)
    if overrides:
        cfg = GoogleDualDICEConfig(**{**cfg.__dict__, **overrides})

    gamma_value = float(gamma)
    if not (0.0 <= gamma_value < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    S = _as_2d(states, "states").astype(np.float32, copy=False)
    A = _as_2d(actions, "actions").astype(np.float32, copy=False)
    S_next = _as_2d(next_states, "next_states").astype(np.float32, copy=False)
    A_pi = _as_2d(target_actions, "target_actions").astype(np.float32, copy=False)
    A_pi_next = _as_2d(target_next_actions, "target_next_actions").astype(np.float32, copy=False)
    S_initial = _as_2d(initial_states, "initial_states").astype(np.float32, copy=False)
    A_initial = _as_2d(initial_actions, "initial_actions").astype(np.float32, copy=False)
    _validate_aligned_inputs(S=S, A=A, S_next=S_next, A_pi=A_pi)
    _validate_next_target_actions(A=A, S=S, A_pi_next=A_pi_next)
    _validate_initial_state_inputs(S=S, S_initial=S_initial, initial_weights=initial_weights)
    _validate_initial_action_inputs(A=A, S_initial=S_initial, A_initial=A_initial)
    sample_weight_1d = _optional_weights(sample_weight, S.shape[0], "sample_weight")
    initial_weight_1d = _optional_weights(initial_weights, S_initial.shape[0], "initial_weights")
    terminals_1d = _optional_terminals(terminals, S.shape[0])

    preflight = preflight_google_dualdice(cfg.google_research_path)
    if not preflight.available:
        raise ModuleNotFoundError(preflight.reason)
    if str(preflight.repo_path) not in sys.path:
        sys.path.insert(0, str(preflight.repo_path))
    tf = _load_tensorflow_for_google_dualdice()
    if cfg.limit_tf_threads:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
    from policy_eval.dual_dice import DualDICE

    np.random.seed(int(cfg.seed))
    tf.random.set_seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed) + 44_001)
    model = DualDICE(S.shape[1], A.shape[1], weight_decay=float(cfg.weight_decay))
    actual_batch_size = min(int(cfg.batch_size), S.shape[0])

    states_tf = tf.convert_to_tensor(S, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(S_next, dtype=tf.float32)
    next_actions_tf = tf.convert_to_tensor(A_pi_next, dtype=tf.float32)
    masks_tf = tf.convert_to_tensor(1.0 - terminals_1d, dtype=tf.float32)
    weights_tf = tf.convert_to_tensor(sample_weight_1d, dtype=tf.float32)
    initial_states_tf = tf.convert_to_tensor(S_initial, dtype=tf.float32)
    initial_actions_tf = tf.convert_to_tensor(A_initial, dtype=tf.float32)
    initial_weights_tf = tf.convert_to_tensor(initial_weight_1d, dtype=tf.float32)

    start = time.perf_counter()
    history: list[dict[str, float]] = []
    last_loss = float("nan")
    for step in range(int(cfg.num_updates)):
        idx = rng.integers(0, S.shape[0], size=actual_batch_size)
        loss = model.update(
            initial_states_tf,
            initial_actions_tf,
            initial_weights_tf,
            tf.gather(states_tf, idx),
            tf.gather(actions_tf, idx),
            tf.gather(next_states_tf, idx),
            tf.gather(next_actions_tf, idx),
            tf.gather(masks_tf, idx),
            tf.gather(weights_tf, idx),
            gamma_value,
        )
        last_loss = float(loss.numpy())
        if step == 0 or step == int(cfg.num_updates) - 1 or (step + 1) % 250 == 0:
            history.append({"step": float(step), "loss": last_loss})

    raw = model.zeta(states_tf, actions_tf).numpy().astype(np.float64).reshape(-1)
    clipped = _postprocess_google_zeta(
        raw,
        prediction_max=cfg.prediction_max,
        normalize=cfg.normalize_predictions,
    )
    diagnostics = {
        "backend": "google_dualdice",
        "gamma": gamma_value,
        "num_updates": float(cfg.num_updates),
        "batch_size": float(actual_batch_size),
        "weight_decay": float(cfg.weight_decay),
        "runtime_sec": float(time.perf_counter() - start),
        "final_loss": last_loss,
        "prediction_max": cfg.prediction_max,
        "normalize_predictions": bool(cfg.normalize_predictions),
        "raw_weight_mean": float(np.mean(raw)),
        "raw_weight_std": float(np.std(raw)),
        "raw_weight_min": float(np.min(raw)),
        "raw_weight_max": float(np.max(raw)),
        "raw_weight_p99": float(np.quantile(raw, 0.99)),
        "weight_mean": float(np.mean(clipped)),
        "weight_std": float(np.std(clipped)),
        "weight_min": float(np.min(clipped)),
        "weight_max": float(np.max(clipped)),
        "weight_p99": float(np.quantile(clipped, 0.99)),
        "weight_ess_fraction": float(_ess(clipped) / max(clipped.size, 1)),
        "clipped_fraction": float(np.mean(raw < 0.0)),
    }
    return GoogleDualDICEOccupancyRatioModel(
        model=model,
        gamma=gamma_value,
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        diagnostics=diagnostics,
        history=history,
        config=cfg,
    )


def _load_tensorflow_for_google_dualdice():
    try:
        import tensorflow as tf
        import tensorflow_addons  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Google DualDICE requires TensorFlow, TensorFlow Addons, and a "
            "Google Research checkout. Install occupancy-ratio[google] and pass "
            "GoogleDualDICEConfig(google_research_path=...)."
        ) from exc
    return tf


def _optional_weights(weights: Optional[Array], n_rows: int, name: str) -> Array:
    if weights is None:
        return np.ones(int(n_rows), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{name} must contain positive total weight.")
    return arr


def _optional_terminals(terminals: Optional[Array], n_rows: int) -> Array:
    if terminals is None:
        return np.zeros(int(n_rows), dtype=np.float64)
    arr = np.asarray(terminals, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"terminals must have {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("terminals must contain only finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("terminals must be in [0, 1].")
    return arr


def _postprocess_google_zeta(
    raw: Array,
    *,
    prediction_max: Optional[float],
    normalize: bool,
) -> Array:
    values = np.maximum(np.asarray(raw, dtype=np.float64).reshape(-1), 0.0)
    if prediction_max is not None:
        np.minimum(values, float(prediction_max), out=values)
    if normalize:
        mean = float(np.mean(values)) if values.size else 0.0
        if np.isfinite(mean) and mean > 1e-12:
            values = values / mean
    return values
