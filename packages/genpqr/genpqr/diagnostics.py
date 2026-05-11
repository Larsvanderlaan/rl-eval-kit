"""Structured diagnostics for GenPQR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from genpqr.types import Array, EstimatedPolicy, FittedQFunction
from genpqr.validation import as_1d_float


@dataclass
class GenPQRDiagnostics:
    """Structured diagnostics produced by a GenPQR fit."""

    n_rows: int
    action_space: str
    policy_neg_log_likelihood: float
    policy_log_prob_min: float
    policy_log_prob_max: float
    pseudo_reward_mean: float
    pseudo_reward_std: float
    q_backend: str | None = None
    q_prediction_finite_fraction: float | None = None
    expected_q_finite_fraction: float | None = None
    reward_mean: float | None = None
    reward_std: float | None = None
    reward_min: float | None = None
    reward_max: float | None = None
    reward_finite_fraction: float | None = None
    normalization_residual_abs_mean: float | None = None
    continuous_mc_standard_error_mean: float | None = None
    warnings: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    policy_entropy_mean: float | None = None
    policy_probability_floor_hit_rate: float | None = None
    continuous_log_density_finite_fraction: float | None = None
    reward_quantiles: dict[str, float] = field(default_factory=dict)
    q_quantiles: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary compatible with the legacy diagnostics field."""

        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return data


def build_diagnostics(
    *,
    policy: EstimatedPolicy,
    q_function: FittedQFunction,
    reward_function: Any,
    states: Array,
    actions: Array,
    log_policy: Array,
    pseudo_rewards: Array,
) -> GenPQRDiagnostics:
    """Build structured diagnostics from fitted objects."""

    n_rows = np.asarray(states).shape[0]
    logp = as_1d_float(log_policy, "policy.log_prob", n_rows=n_rows)
    pseudo = as_1d_float(pseudo_rewards, "pseudo_rewards", n_rows=n_rows)
    q_diag = dict(getattr(q_function, "diagnostics", {}))
    warnings: list[str] = []
    warning_codes: list[str] = []
    if q_diag.get("weak_anchor_support"):
        warnings.append("DeepPQR anchor-action support is weak.")
        warning_codes.append("weak_anchor_support")

    q_pred = as_1d_float(q_function.predict_q(states, actions), "q_function.predict_q", n_rows=n_rows)
    rng = np.random.default_rng(getattr(reward_function, "seed", 123))
    expected_q = as_1d_float(
        q_function.expected_q(
            states,
            reward_function.normalization_policy,
            n_action_samples=getattr(reward_function, "n_action_samples", 32),
            rng=rng,
        ),
        "q_function.expected_q",
        n_rows=n_rows,
    )
    rewards = as_1d_float(reward_function.predict_reward(states, actions), "reward_function.predict_reward", n_rows=n_rows)
    residual = reward_function.normalization_residual(states)
    policy_entropy_mean = None
    policy_probability_floor_hit_rate = None
    continuous_log_density_finite_fraction = None
    if getattr(reward_function.action_space, "kind", None) == "discrete" and hasattr(policy, "predict_proba"):
        probs = np.asarray(policy.predict_proba(states), dtype=np.float64)  # type: ignore[attr-defined]
        clipped = np.clip(probs, 1e-300, None)
        policy_entropy_mean = float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))
        floor = float(getattr(policy, "prob_clip_min", 0.0))
        policy_probability_floor_hit_rate = None if floor <= 0.0 else float(np.mean(probs <= floor * (1.0 + 1e-8)))
    elif getattr(reward_function.action_space, "kind", None) == "continuous":
        continuous_log_density_finite_fraction = float(np.mean(np.isfinite(logp)))
    mc_se = None
    if getattr(reward_function.action_space, "kind", None) == "continuous":
        mc_se = reward_function.mc_standard_error(states)
        mc_se_value = float(np.nanmean(mc_se)) if not np.all(np.isnan(mc_se)) else None
    else:
        mc_se_value = None

    extra = {f"q_{key}": value for key, value in q_diag.items()}
    return GenPQRDiagnostics(
        n_rows=int(np.asarray(states).shape[0]),
        action_space=reward_function.action_space.kind,
        policy_neg_log_likelihood=float(-np.mean(logp)),
        policy_log_prob_min=float(np.min(logp)),
        policy_log_prob_max=float(np.max(logp)),
        pseudo_reward_mean=float(np.mean(pseudo)),
        pseudo_reward_std=float(np.std(pseudo)),
        q_backend=q_diag.get("backend"),
        q_prediction_finite_fraction=float(np.mean(np.isfinite(q_pred))),
        expected_q_finite_fraction=float(np.mean(np.isfinite(expected_q))),
        reward_mean=float(np.mean(rewards[np.isfinite(rewards)])) if np.any(np.isfinite(rewards)) else None,
        reward_std=float(np.std(rewards[np.isfinite(rewards)])) if np.any(np.isfinite(rewards)) else None,
        reward_min=float(np.min(rewards[np.isfinite(rewards)])) if np.any(np.isfinite(rewards)) else None,
        reward_max=float(np.max(rewards[np.isfinite(rewards)])) if np.any(np.isfinite(rewards)) else None,
        reward_finite_fraction=float(np.mean(np.isfinite(rewards))),
        normalization_residual_abs_mean=None if np.all(np.isnan(residual)) else float(np.nanmean(np.abs(residual))),
        continuous_mc_standard_error_mean=mc_se_value,
        warnings=warnings,
        warning_codes=warning_codes,
        policy_entropy_mean=policy_entropy_mean,
        policy_probability_floor_hit_rate=policy_probability_floor_hit_rate,
        continuous_log_density_finite_fraction=continuous_log_density_finite_fraction,
        reward_quantiles=_quantiles(rewards),
        q_quantiles=_quantiles(q_pred),
        extra=extra,
    )


def _quantiles(values: Array) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }
