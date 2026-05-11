from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from causal_ope_benchmark.baselines import EstimatorResult
from causal_ope_benchmark.types import LongitudinalDataset, TruthBundle


_SCORE_SCHEMA_VERSION = "private_v1"
_UNCERTAINTY_MULTIPLIER = 1.96


@dataclass(frozen=True)
class _ScoreTarget:
    role: str
    weight: float


def score_result(dataset: LongitudinalDataset, truth: TruthBundle, result: EstimatorResult) -> dict[str, Any]:
    """Score an estimator result against sealed truth."""
    row: dict[str, Any] = {}
    targets = _truth_targets(truth)
    required_targets = _required_targets(dataset, truth, result)
    schema = _score_schema(dataset, truth, result, targets, required_targets)
    row.update(_schema_columns(schema))
    row.update(_truth_mc_columns(truth, targets))
    row.update(_constraint_metrics(dataset, truth))
    constraint_violation_count = _constraint_violation_count(row)
    row["constraint_violation_count"] = int(constraint_violation_count)
    for key in targets:
        spec = schema[key]
        mc_se, mc_noise = _truth_mc_uncertainty(truth, key)
        uncertainty = float(np.sqrt(mc_se * mc_se + mc_noise * mc_noise))
        row[f"{key}_score_role"] = spec.role
        row[f"{key}_score_weight"] = float(spec.weight)
        row[f"{key}_truth_mc_se"] = float(mc_se)
        row[f"{key}_truth_mc_noise"] = float(mc_noise)
        row[f"{key}_truth_mc_uncertainty"] = uncertainty
        if key in truth.target_mc_values:
            row[f"{key}_target_mc_value"] = _as_float(truth.target_mc_values.get(key), default="")
    if result.status != "ok":
        row.update(
            _leaderboard_columns(
                truth=truth,
                result=result,
                missing_required=sorted(required_targets),
                score_available=False,
                constraint_violation_count=constraint_violation_count,
            )
        )
        return row
    errors = []
    required_errors = []
    role_errors: dict[str, list[float]] = {"primary": [], "secondary": [], "constraint": []}
    role_calibrated_errors: dict[str, list[float]] = {"primary": [], "secondary": [], "constraint": []}
    role_weights: dict[str, list[float]] = {"primary": [], "secondary": [], "constraint": []}
    covered = []
    interval_lengths = []
    for key, true_value in targets.items():
        spec = schema[key]
        uncertainty = float(row[f"{key}_truth_mc_uncertainty"])
        if key not in result.estimates:
            row[f"estimand_available_{key}"] = 0
            row[f"estimand_missing_{key}"] = 1
            continue
        row[f"estimand_available_{key}"] = 1
        row[f"estimand_missing_{key}"] = 0
        estimate = float(result.estimates[key])
        error = estimate - float(true_value)
        row[f"{key}_estimate"] = estimate
        row[f"{key}_target"] = float(true_value)
        row[f"{key}_error"] = float(error)
        row[f"{key}_abs_error"] = float(abs(error))
        calibrated_error = _calibrated_abs_error(abs(error), uncertainty)
        row[f"{key}_calibrated_abs_error"] = calibrated_error
        errors.append(float(error))
        if key in required_targets:
            required_errors.append(float(error))
        role_errors[spec.role].append(float(abs(error)))
        role_calibrated_errors[spec.role].append(float(calibrated_error))
        role_weights[spec.role].append(float(spec.weight))
        if key in result.intervals:
            low, high = result.intervals[key]
            covered.append(float(low <= true_value <= high))
            interval_lengths.append(float(high - low))
    if errors:
        err = np.asarray(errors, dtype=np.float64)
        row["mae"] = float(np.mean(np.abs(err)))
        row["rmse"] = float(np.sqrt(np.mean(err * err)))
        row["bias"] = float(np.mean(err))
    if required_errors:
        req_err = np.asarray(required_errors, dtype=np.float64)
        row["primary_mae"] = float(np.mean(np.abs(req_err)))
        row["primary_rmse"] = float(np.sqrt(np.mean(req_err * req_err)))
        row["primary_bias"] = float(np.mean(req_err))
    for role in ("primary", "secondary", "constraint"):
        row[f"{role}_available_estimand_count"] = int(len(role_errors[role]))
        row[f"{role}_weighted_mae"] = _weighted_mean(role_errors[role], role_weights[role])
        row[f"{role}_calibrated_weighted_mae"] = _weighted_mean(role_calibrated_errors[role], role_weights[role])
    row["primary_weighted_mae"] = row.get("primary_weighted_mae", "")
    row["calibrated_score"] = row.get("primary_calibrated_weighted_mae", "")
    if _finite_float(row.get("calibrated_score")) is not None and constraint_violation_count:
        row["constraint_penalty"] = float(constraint_violation_count)
        row["calibrated_score"] = float(row["calibrated_score"]) + float(constraint_violation_count)
    else:
        row["constraint_penalty"] = 0.0
    missing_required = sorted(key for key in required_targets if key not in result.estimates)
    missing_primary = sorted(key for key, spec in schema.items() if spec.role == "primary" and key not in result.estimates)
    row["required_estimand_count"] = int(len(required_targets))
    row["missing_required_estimand_count"] = int(len(missing_required))
    row["missing_primary_estimand_count"] = int(len(missing_primary))
    if missing_required:
        row["missing_required_estimands"] = "|".join(missing_required)
        row["estimand_status"] = "incomplete"
        row["status"] = "incomplete"
    else:
        row["estimand_status"] = "ok"
    if missing_primary:
        row["missing_primary_estimands"] = "|".join(missing_primary)
    score_available = _finite_float(row.get("calibrated_score")) is not None and not missing_required
    row.update(
        _leaderboard_columns(
            truth=truth,
            result=result,
            missing_required=missing_required,
            score_available=score_available,
            constraint_violation_count=constraint_violation_count,
        )
    )
    row["interval_available"] = int(bool(result.intervals))
    if covered:
        row["ci_coverage"] = float(np.mean(covered))
        row["ci_mean_length"] = float(np.mean(interval_lengths))
    row.update(_subgroup_metrics(truth, result))
    row.update(_diagnostic_metrics(dataset, result))
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate row-level metrics by family and estimator."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("profile"), row.get("family"), row.get("estimator"))
        groups.setdefault(key, []).append(row)
    out = []
    for (profile, family, estimator), group in groups.items():
        ok = [row for row in group if row.get("status") == "ok"]
        deployable = [row for row in ok if not _truthy(row.get("diagnostic_only"))]
        leaderboard = [row for row in deployable if _truthy(row.get("leaderboard_result_eligible"))]
        out.append(
            {
                "profile": profile,
                "family": family,
                "estimator": estimator,
                "n_rows": len(group),
                "ok_rows": len(ok),
                "deployable_rows": len(deployable),
                "leaderboard_eligible_rows": len(leaderboard),
                "leaderboard_ineligible_rows": len(deployable) - len(leaderboard),
                "failure_rate": 1.0 - len(ok) / max(len(group), 1),
                "mae_mean": _mean(row.get("mae") for row in deployable),
                "rmse_mean": _mean(row.get("rmse") for row in deployable),
                "bias_mean": _mean(row.get("bias") for row in deployable),
                "primary_mae_mean": _mean(row.get("primary_mae") for row in deployable),
                "primary_rmse_mean": _mean(row.get("primary_rmse") for row in deployable),
                "primary_weighted_mae_mean": _mean(row.get("primary_weighted_mae") for row in deployable),
                "calibrated_score_mean": _mean(row.get("calibrated_score") for row in deployable),
                "leaderboard_primary_weighted_mae_mean": _mean(row.get("primary_weighted_mae") for row in leaderboard),
                "leaderboard_calibrated_score_mean": _mean(row.get("calibrated_score") for row in leaderboard),
                "leaderboard_reason_counts": _reason_counts(row.get("leaderboard_ineligible_reason") for row in deployable),
                "truth_mc_uncertainty_rows": sum(int(_truthy(row.get("truth_mc_uncertainty_used"))) for row in deployable),
                "ci_coverage_mean": _mean(row.get("ci_coverage") for row in deployable),
                "runtime_sec_mean": _mean(row.get("runtime_sec") for row in group),
            }
        )
    return out


def _truth_targets(truth: TruthBundle) -> dict[str, float]:
    targets: dict[str, float] = {}
    targets.update({key: float(value) for key, value in truth.values.items()})
    targets.update(
        {
            key: float(value)
            for key, value in truth.effects.items()
            if key.startswith("effect_horizon_") or key.startswith("itt_effect_horizon_")
            or key.startswith("tot_effect_horizon_")
            or key == "surrogate_bias_horizon_long"
        }
    )
    targets.update({key: float(value) for key, value in truth.rmst.items()})
    if "survival_target" in truth.survival_curves:
        targets["survival_horizon"] = float(np.asarray(truth.survival_curves["survival_target"])[-1])
    targets.update({key: float(value) for key, value in truth.subgroup_effects.items()})
    return targets


def _score_schema(
    dataset: LongitudinalDataset,
    truth: TruthBundle,
    result: EstimatorResult,
    targets: dict[str, float],
    required_targets: set[str],
) -> dict[str, _ScoreTarget]:
    schema: dict[str, _ScoreTarget] = {}
    for key in targets:
        role = "secondary"
        if key in required_targets:
            role = "primary"
        elif _is_constraint_estimand(dataset.family, key):
            role = "constraint"
        weight = _score_weight(dataset.family, key, role)
        schema[key] = _ScoreTarget(role=role, weight=weight)
    primary = [key for key, spec in schema.items() if spec.role == "primary"]
    if result.diagnostic_only:
        return schema
    if not primary:
        fallback = _default_primary_key(dataset.family, targets)
        if fallback is not None:
            schema[fallback] = _ScoreTarget(role="primary", weight=_score_weight(dataset.family, fallback, "primary"))
    return schema


def _schema_columns(schema: dict[str, _ScoreTarget]) -> dict[str, Any]:
    out: dict[str, Any] = {"private_score_schema_version": _SCORE_SCHEMA_VERSION}
    for role in ("primary", "secondary", "constraint"):
        keys = sorted(key for key, spec in schema.items() if spec.role == role)
        out[f"{role}_estimand_count"] = int(len(keys))
        out[f"{role}_estimands"] = "|".join(keys)
        out[f"{role}_score_weight_sum"] = float(sum(schema[key].weight for key in keys))
    return out


def _is_constraint_estimand(family: str, key: str) -> bool:
    if family == "streamretain":
        return False
    if family == "clinic_dtr":
        return False
    return False


def _default_primary_key(family: str, targets: dict[str, float]) -> str | None:
    if family in {"streamretain", "clinic_dtr", "epicare"} and "policy_value" in targets:
        return "policy_value"
    streamlift = sorted(key for key in targets if key.startswith("effect_horizon_"))
    return streamlift[0] if streamlift else None


def _score_weight(family: str, key: str, role: str) -> float:
    if role == "constraint":
        return 0.0
    if family == "streamlift" and key.startswith("effect_horizon_"):
        horizon = _trailing_int(key)
        return float(horizon) if horizon is not None else 1.0
    return 1.0


def _required_targets(dataset: LongitudinalDataset, truth: TruthBundle, result: EstimatorResult) -> set[str]:
    if result.diagnostic_only:
        return set()
    if dataset.family == "streamlift":
        return {key for key in truth.effects if key.startswith("effect_horizon_")}
    if dataset.family == "streamretain":
        return {"policy_value"} if "policy_value" in truth.values else set()
    if dataset.family == "clinic_dtr":
        if result.estimator == "ipcw_rmst":
            required = set()
            if "rmst" in truth.rmst:
                required.add("rmst")
            if "survival_target" in truth.survival_curves:
                required.add("survival_horizon")
            return required
        return {"policy_value"} if "policy_value" in truth.values else set()
    if dataset.family == "epicare":
        return {"policy_value"} if "policy_value" in truth.values else set()
    return set()


def _subgroup_metrics(truth: TruthBundle, result: EstimatorResult) -> dict[str, Any]:
    errors = []
    for key, value in truth.subgroup_effects.items():
        if key in result.estimates:
            errors.append(abs(float(result.estimates[key]) - float(value)))
    if not errors:
        return {}
    return {"subgroup_mae": float(np.mean(errors)), "worst_group_abs_error": float(np.max(errors))}


def _diagnostic_metrics(dataset: LongitudinalDataset, result: EstimatorResult) -> dict[str, Any]:
    out = {
        "censoring_rate": float(np.mean(dataset.censoring)),
        "missingness_rate": float(np.mean(dataset.missingness_mask)),
        "terminal_rate": float(np.mean(dataset.terminals)),
    }
    for key in ("ess_fraction", "weight_p95", "diagnostic_only_ess"):
        if key in result.diagnostics:
            out[key] = result.diagnostics[key]
    return out


def _constraint_metrics(dataset: LongitudinalDataset, truth: TruthBundle) -> dict[str, Any]:
    out: dict[str, Any] = {"constraint_schema_available": int(dataset.family in {"streamretain", "clinic_dtr"})}
    actions = np.argmax(np.asarray(dataset.actions), axis=1)
    if dataset.family == "streamretain":
        spend = np.asarray(dataset.outcome_components.get("intervention_cost", np.zeros(dataset.n)), dtype=np.float64)
        fatigue = np.asarray(dataset.outcome_components.get("fatigue", np.zeros(dataset.n)), dtype=np.float64)
        contact = np.isin(actions, [1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float64)
        budget_limit = _private_float(truth, "budget_limit", default=2.0)
        fatigue_limit = _private_float(truth, "fatigue_limit", default=1.25)
        budget_observed = _private_float(
            truth,
            "target_policy_budget_observed",
            default=float(np.mean(spend)) if spend.size else 0.0,
        )
        fatigue_observed = _private_float(
            truth,
            "target_policy_fatigue_observed",
            default=float(np.mean(fatigue)) if fatigue.size else 0.0,
        )
        contact_rate = _private_float(
            truth,
            "target_policy_contact_rate",
            default=float(np.mean(contact)) if contact.size else 0.0,
        )
        out.update(
            {
                "constraint_budget_limit": budget_limit,
                "constraint_budget_observed": budget_observed,
                "constraint_budget_margin": float(budget_limit - budget_observed),
                "constraint_budget_violation": int(budget_observed > budget_limit),
                "constraint_fatigue_limit": fatigue_limit,
                "constraint_fatigue_observed": fatigue_observed,
                "constraint_fatigue_margin": float(fatigue_limit - fatigue_observed),
                "constraint_fatigue_violation": int(fatigue_observed > fatigue_limit),
                "constraint_contact_rate": contact_rate,
            }
        )
    elif dataset.family == "clinic_dtr":
        toxicity = np.asarray(dataset.outcome_components.get("toxicity", np.zeros(dataset.n)), dtype=np.float64)
        intensify = np.isin(actions, [2, 3]).astype(np.float64)
        contraindicated = intensify * ((dataset.states[:, 6] > 0.72) | (dataset.states[:, 8] > 0.68))
        contraindicated_limit = _private_float(truth, "contraindicated_action_rate_limit", default=0.05)
        toxicity_limit = _private_float(truth, "high_toxicity_intensification_rate_limit", default=0.05)
        contraindicated_rate = _private_float(
            truth,
            "target_policy_contraindicated_action_rate",
            default=float(np.mean(contraindicated)) if contraindicated.size else 0.0,
        )
        toxicity_rate = _private_float(
            truth,
            "target_policy_high_toxicity_intensification_rate",
            default=float(np.mean(intensify * (toxicity > 0.55))) if toxicity.size else 0.0,
        )
        monitoring_rate = _private_float(
            truth,
            "target_policy_monitoring_action_rate",
            default=float(np.mean(actions == 0)) if actions.size else 0.0,
        )
        intensity_default = float(np.mean(np.abs(np.asarray(dataset.action_dose)))) if dataset.action_dose is not None else float(np.mean(actions))
        intensity = _private_float(truth, "target_policy_action_intensity", default=intensity_default)
        out.update(
            {
                "constraint_contraindicated_action_rate_limit": contraindicated_limit,
                "constraint_contraindicated_action_rate": contraindicated_rate,
                "constraint_contraindicated_action_rate_margin": float(contraindicated_limit - contraindicated_rate),
                "constraint_contraindicated_action_rate_violation": int(contraindicated_rate > contraindicated_limit),
                "constraint_high_toxicity_intensification_rate_limit": toxicity_limit,
                "constraint_high_toxicity_intensification_rate": toxicity_rate,
                "constraint_high_toxicity_intensification_rate_margin": float(toxicity_limit - toxicity_rate),
                "constraint_high_toxicity_intensification_rate_violation": int(toxicity_rate > toxicity_limit),
                "constraint_monitoring_action_rate": monitoring_rate,
                "constraint_monitoring_action_intensity": intensity,
            }
        )
    return out


def _truth_mc_columns(truth: TruthBundle, targets: dict[str, float]) -> dict[str, Any]:
    uncertainties = []
    out: dict[str, Any] = {}
    for key in targets:
        mc_se, mc_noise = _truth_mc_uncertainty(truth, key)
        uncertainty = float(np.sqrt(mc_se * mc_se + mc_noise * mc_noise))
        uncertainties.append(uncertainty)
    used = [value for value in uncertainties if value > 0.0]
    out["truth_mc_uncertainty_used"] = int(bool(used))
    out["truth_mc_uncertainty_estimand_count"] = int(len(used))
    out["truth_mc_uncertainty_max"] = float(max(used)) if used else 0.0
    return out


def _truth_mc_uncertainty(truth: TruthBundle, key: str) -> tuple[float, float]:
    se = _as_optional_float(truth.target_standard_errors.get(key))
    noise = _as_optional_float(truth.truth_noise_floor.get(key))
    target_payload = truth.target_mc_values.get(key)
    if se is None:
        se = _extract_numeric(target_payload, ("se", "mc_se", "standard_error", "std_error", "stderr"))
    if noise is None:
        noise = _extract_numeric(target_payload, ("noise", "noise_sd", "mc_noise", "mc_noise_sd"))
    metadata = truth.private_metadata
    if se is None:
        se = _lookup_private_uncertainty(
            metadata,
            key,
            ("target_mc_se", "target_mc_ses", "target_mc_standard_errors", "truth_mc_se", "mc_standard_errors", "mc_se"),
            (f"{key}_mc_se", f"{key}_se", f"mc_se_{key}", f"{key}_truth_mc_se"),
        )
    if noise is None:
        noise = _lookup_private_uncertainty(
            metadata,
            key,
            ("target_mc_noise", "target_mc_noise_sd", "truth_mc_noise", "truth_mc_noise_sd", "mc_noise", "mc_noise_sd"),
            (f"{key}_mc_noise", f"{key}_noise", f"mc_noise_{key}", f"{key}_truth_mc_noise"),
        )
    return max(float(se or 0.0), 0.0), max(float(noise or 0.0), 0.0)


def _lookup_private_uncertainty(
    metadata: dict[str, Any],
    key: str,
    mapping_names: tuple[str, ...],
    direct_names: tuple[str, ...],
) -> float | None:
    for name in direct_names:
        value = _as_optional_float(metadata.get(name))
        if value is not None:
            return value
    for name in mapping_names:
        payload = metadata.get(name)
        if isinstance(payload, dict):
            value = _as_optional_float(payload.get(key))
            if value is not None:
                return value
        value = _extract_numeric(payload, (key,))
        if value is not None:
            return value
    return None


def _leaderboard_columns(
    *,
    truth: TruthBundle,
    result: EstimatorResult,
    missing_required: list[str],
    score_available: bool,
    constraint_violation_count: int,
) -> dict[str, Any]:
    reasons = []
    if not truth.leaderboard_eligible:
        reasons.append("scenario_not_leaderboard_eligible")
    if result.status != "ok":
        reasons.append(f"status_{result.status}")
    if result.diagnostic_only:
        reasons.append("diagnostic_only")
    if missing_required:
        reasons.append("missing_required_estimands:" + ",".join(missing_required))
    if not score_available:
        reasons.append("calibrated_score_unavailable")
    if constraint_violation_count:
        reasons.append(f"constraint_violations:{int(constraint_violation_count)}")
    eligible = not reasons
    return {
        "leaderboard_scenario_eligible": int(bool(truth.leaderboard_eligible)),
        "leaderboard_result_eligible": int(eligible),
        "leaderboard_score_available": int(bool(score_available)),
        "leaderboard_ineligible_reason": "|".join(reasons),
    }


def _calibrated_abs_error(abs_error: float, uncertainty: float) -> float:
    return float(max(float(abs_error) - _UNCERTAINTY_MULTIPLIER * float(uncertainty), 0.0))


def _constraint_violation_count(row: dict[str, Any]) -> int:
    total = 0
    for key, value in row.items():
        if key.startswith("constraint_") and key.endswith("_violation"):
            total += int(_truthy(value))
    return total


def _weighted_mean(values: list[float], weights: list[float]) -> float | str:
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != arr.shape or float(np.sum(w)) <= 0.0:
        return float(np.mean(arr))
    return float(np.average(arr, weights=w))


def _mean(values) -> float | str:
    arr = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            arr.append(val)
    if not arr:
        return ""
    return float(np.mean(arr))


def _reason_counts(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        for reason in str(value).split("|"):
            if not reason:
                continue
            base = reason.split(":", maxsplit=1)[0]
            counts[base] = counts.get(base, 0) + 1
    return "|".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _as_float(value: Any, *, default: Any = 0.0) -> Any:
    parsed = _as_optional_float(value)
    return default if parsed is None else float(parsed)


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, dict):
        return _extract_numeric(value, ("value", "mean", "estimate"))
    if isinstance(value, (tuple, list)) and value:
        return _as_optional_float(value[0])
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _extract_numeric(payload: Any, names: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = _as_optional_float(payload.get(name))
        if value is not None:
            return value
    return None


def _private_float(truth: TruthBundle, key: str, *, default: float) -> float:
    value = _as_optional_float(truth.private_metadata.get(key))
    return float(default if value is None else value)


def _trailing_int(key: str) -> int | None:
    try:
        return int(str(key).rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)
