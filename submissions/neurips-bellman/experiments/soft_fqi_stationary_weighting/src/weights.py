from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TransitionBatch
from .features import RatioFeatureMap
from .soft_dp import discounted_state_distribution, state_action_distribution

RICH_TIKHONOV_RIDGE_GRID = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
)


@dataclass
class WeightResult:
    weights: np.ndarray
    raw_weights: np.ndarray
    diagnostics: dict[str, float | str]


@dataclass
class _MomentFit:
    alpha: np.ndarray
    a_mat: np.ndarray
    b_mat: np.ndarray
    mean_features: np.ndarray
    rhs: np.ndarray


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float((np.sum(w) ** 2) / np.maximum(np.sum(w * w), 1e-300))


def stabilize_weights(
    raw_weights: np.ndarray,
    *,
    min_weight: float = 1e-8,
    max_weight: float | None = 25.0,
    clip_quantile: float | None = 0.99,
    target_ess_fraction: float | None = 0.25,
    max_uniform_mix: float = 0.50,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    positive = np.maximum(raw, min_weight)
    clip_level = np.inf
    if clip_quantile is not None:
        clip_level = min(clip_level, float(np.quantile(positive, clip_quantile)))
    if max_weight is not None:
        clip_level = min(clip_level, float(max_weight))
    clipped = np.minimum(positive, clip_level)
    fraction_clipped = float(np.mean(positive > clip_level)) if np.isfinite(clip_level) else 0.0
    clipped /= np.maximum(np.mean(clipped), 1e-300)
    ess_before = effective_sample_size(clipped) / max(clipped.shape[0], 1)
    chosen_mix = 0.0
    processed = clipped
    if target_ess_fraction is not None and ess_before < target_ess_fraction:
        lo, hi = 0.0, float(max_uniform_mix)
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            candidate = (1.0 - mid) * clipped + mid
            candidate /= np.maximum(np.mean(candidate), 1e-300)
            ess_mid = effective_sample_size(candidate) / max(candidate.shape[0], 1)
            if ess_mid >= target_ess_fraction:
                hi = mid
            else:
                lo = mid
        chosen_mix = hi
        processed = (1.0 - chosen_mix) * clipped + chosen_mix
    processed /= np.maximum(np.mean(processed), 1e-300)
    meta = {
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "clip_level": float(clip_level) if np.isfinite(clip_level) else float("nan"),
        "fraction_clipped": fraction_clipped,
        "ess_fraction_before_mix": float(ess_before),
        "chosen_uniform_mix": float(chosen_mix),
        "ess_fraction_after_mix": float(effective_sample_size(processed) / max(processed.shape[0], 1)),
    }
    return processed, meta


def summarize_weights(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "weight_mean": float(np.mean(w)),
        "weight_std": float(np.std(w)),
        "weight_max": float(np.max(w)),
        "weight_q95": float(np.quantile(w, 0.95)),
        "weight_q99": float(np.quantile(w, 0.99)),
        "effective_sample_size": effective_sample_size(w),
        "effective_sample_size_fraction": float(effective_sample_size(w) / max(w.shape[0], 1)),
    }


def ratio_quality(oracle_weights: np.ndarray, candidate_weights: np.ndarray) -> dict[str, float]:
    oracle = np.maximum(np.asarray(oracle_weights, dtype=np.float64).reshape(-1), 1e-12)
    candidate = np.maximum(np.asarray(candidate_weights, dtype=np.float64).reshape(-1), 1e-12)
    oracle = oracle / np.maximum(np.mean(oracle), 1e-300)
    candidate = candidate / np.maximum(np.mean(candidate), 1e-300)
    if np.std(oracle) < 1e-12 or np.std(candidate) < 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(oracle, candidate)[0, 1])
    support_threshold = max(float(np.quantile(oracle, 0.05)), 1e-8)
    support = oracle >= support_threshold
    if np.sum(support) < 5:
        support = np.ones_like(oracle, dtype=bool)
    oracle_clip = float(np.quantile(oracle, 0.99))
    candidate_clip = float(np.quantile(candidate, 0.99))
    clipped_oracle = np.minimum(oracle, max(oracle_clip, 1e-8))
    clipped_candidate = np.minimum(candidate, max(candidate_clip, 1e-8))
    oracle_centered = oracle - np.mean(oracle)
    candidate_centered = candidate - np.mean(candidate)
    denom = float(np.sum(oracle_centered * oracle_centered))
    calibration_slope = float(np.sum(oracle_centered * candidate_centered) / denom) if denom > 1e-12 else float("nan")
    return {
        "oracle_log_ratio_rmse": float(np.sqrt(np.mean((np.log(candidate) - np.log(oracle)) ** 2))),
        "oracle_log_ratio_rmse_support": float(
            np.sqrt(np.mean((np.log(candidate[support]) - np.log(oracle[support])) ** 2))
        ),
        "oracle_estimated_weight_corr": corr,
        "oracle_estimated_weight_mae": float(np.mean(np.abs(candidate - oracle))),
        "oracle_estimated_weight_rel_mse": float(
            np.mean((candidate - oracle) ** 2) / np.maximum(np.mean(oracle**2), 1e-300)
        ),
        "oracle_estimated_weight_rel_mse_clipped": float(
            np.mean((clipped_candidate - clipped_oracle) ** 2) / np.maximum(np.mean(clipped_oracle**2), 1e-300)
        ),
        "oracle_estimated_weight_calibration_slope": calibration_slope,
        "oracle_weight_support_fraction": float(np.mean(support)),
    }


def oracle_sample_weights(
    batch: TransitionBatch,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    ratio_sa = np.asarray(target_sa_dist, dtype=np.float64) / np.maximum(behavior_sa_dist, eps)
    weights = ratio_sa[batch.states, batch.actions]
    return weights / np.maximum(np.mean(weights), 1e-300)


def target_discounted_or_stationary_sa(
    transition: np.ndarray,
    target_policy: np.ndarray,
    rho0: np.ndarray,
    gamma_weight: float,
) -> np.ndarray:
    target_state = discounted_state_distribution(transition, target_policy, rho0, gamma_weight)
    return state_action_distribution(target_state, target_policy)


def _moment_matrices(
    current_features: np.ndarray,
    delta: np.ndarray,
    rhs: np.ndarray,
    *,
    ridge_primal: float,
    ridge_dual: float,
    normalization_penalty: float,
) -> _MomentFit:
    a_mat = (current_features.T @ delta) / current_features.shape[0]
    b_mat = (current_features.T @ current_features) / current_features.shape[0]
    h_mat = b_mat + float(ridge_dual) * np.eye(b_mat.shape[0], dtype=np.float64)
    try:
        h_inv = np.linalg.solve(h_mat, np.eye(h_mat.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError:
        h_inv = np.linalg.lstsq(h_mat, np.eye(h_mat.shape[0], dtype=np.float64), rcond=None)[0]

    mean_features = current_features.mean(axis=0)
    system = (
        a_mat @ h_inv @ a_mat.T
        + 2.0 * float(normalization_penalty) * np.outer(mean_features, mean_features)
        + float(ridge_primal) * np.eye(a_mat.shape[0], dtype=np.float64)
    )
    rhs_alpha = a_mat @ (h_inv @ rhs) + 2.0 * float(normalization_penalty) * mean_features
    try:
        alpha = np.linalg.solve(system, rhs_alpha)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(system, rhs_alpha, rcond=None)[0]
    return _MomentFit(alpha=alpha, a_mat=a_mat, b_mat=b_mat, mean_features=mean_features, rhs=rhs)


def _moment_flow_score(
    alpha: np.ndarray,
    current_features: np.ndarray,
    delta: np.ndarray,
    rhs: np.ndarray,
    *,
    score_ridge_dual: float,
    normalization_penalty: float,
) -> float:
    a_mat = (current_features.T @ delta) / current_features.shape[0]
    b_mat = (current_features.T @ current_features) / current_features.shape[0]
    violation = a_mat.T @ alpha - rhs
    h_mat = b_mat + float(score_ridge_dual) * np.eye(b_mat.shape[0], dtype=np.float64)
    try:
        dual_scaled = np.linalg.solve(h_mat, violation)
    except np.linalg.LinAlgError:
        dual_scaled = np.linalg.lstsq(h_mat, violation, rcond=None)[0]
    mean_features = current_features.mean(axis=0)
    normalization_error = float(mean_features @ alpha - 1.0)
    flow_score = float(violation @ dual_scaled)
    return flow_score + 2.0 * float(normalization_penalty) * normalization_error * normalization_error


def _select_cv_ridge(
    current_features: np.ndarray,
    delta: np.ndarray,
    rhs: np.ndarray,
    *,
    ridge_grid: tuple[float, ...],
    cv_folds: int,
    cv_seed: int,
    normalization_penalty: float,
    score_ridge_dual: float | None,
    selection_rule: str = "min",
) -> tuple[float, dict[str, float | str]]:
    grid = tuple(float(value) for value in ridge_grid if float(value) >= 0.0)
    if len(grid) == 0:
        raise ValueError("ridge_grid must contain at least one nonnegative value")

    n_obs = current_features.shape[0]
    n_folds = max(2, min(int(cv_folds), n_obs))
    rng = np.random.default_rng(int(cv_seed))
    shuffled = rng.permutation(n_obs)
    folds = np.array_split(shuffled, n_folds)
    candidate_scores: list[float] = []
    candidate_ses: list[float] = []
    for ridge_value in grid:
        fold_scores: list[float] = []
        for fold in folds:
            if fold.size == 0:
                continue
            train_mask = np.ones(n_obs, dtype=bool)
            train_mask[fold] = False
            fit = _moment_matrices(
                current_features[train_mask],
                delta[train_mask],
                rhs,
                ridge_primal=ridge_value,
                ridge_dual=ridge_value,
                normalization_penalty=normalization_penalty,
            )
            fold_scores.append(
                _moment_flow_score(
                    fit.alpha,
                    current_features[fold],
                    delta[fold],
                    rhs,
                    score_ridge_dual=ridge_value if score_ridge_dual is None else score_ridge_dual,
                    normalization_penalty=normalization_penalty,
                )
            )
        scores = np.asarray(fold_scores, dtype=np.float64)
        candidate_scores.append(float(np.mean(scores)))
        candidate_ses.append(float(np.std(scores, ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else 0.0)

    score_arr = np.asarray(candidate_scores, dtype=np.float64)
    min_idx = int(np.nanargmin(score_arr))
    min_score = float(score_arr[min_idx])
    one_se_threshold = min_score + float(candidate_ses[min_idx])
    eligible = [
        idx
        for idx, score in enumerate(score_arr)
        if np.isfinite(score) and score <= one_se_threshold
    ]
    one_se_idx = max(eligible, key=lambda idx: grid[idx]) if eligible else min_idx
    selected_idx = one_se_idx if selection_rule == "one_se" else min_idx
    selected = grid[selected_idx]
    diagnostics: dict[str, float | str] = {
        "cv_ridge_selected": 1.0,
        "cv_ridge_grid": ",".join(f"{value:.12g}" for value in grid),
        "cv_ridge_folds": float(n_folds),
        "cv_score_ridge_dual": "candidate" if score_ridge_dual is None else float(score_ridge_dual),
        "cv_selection_rule": selection_rule,
        "cv_selected_ridge": float(selected),
        "cv_selected_score": float(score_arr[selected_idx]),
        "cv_selected_score_se": float(candidate_ses[selected_idx]),
        "cv_min_score": min_score,
        "cv_min_ridge": float(grid[min_idx]),
        "cv_one_se_ridge": float(grid[one_se_idx]),
        "cv_one_se_threshold": float(one_se_threshold),
    }
    for idx, (ridge_value, score) in enumerate(zip(grid, candidate_scores, strict=True)):
        diagnostics[f"cv_score_{idx}_ridge"] = float(ridge_value)
        diagnostics[f"cv_score_{idx}_moment_flow"] = float(score)
    return selected, diagnostics


def estimate_moment_weights(
    batch: TransitionBatch,
    *,
    states_grid: np.ndarray,
    transition: np.ndarray,
    target_policy: np.ndarray,
    behavior_state_dist: np.ndarray,
    ratio_features: RatioFeatureMap,
    gamma_weight: float,
    ridge_primal: float = 1e-4,
    ridge_dual: float = 1e-4,
    normalization_penalty: float = 10.0,
    cv_ridge: bool = False,
    cv_ridge_grid: tuple[float, ...] | None = None,
    cv_folds: int = 3,
    cv_seed: int = 20260501,
    cv_score_ridge_dual: float | None = None,
    cv_selection_rule: str = "min",
    min_weight: float = 1e-8,
    max_weight: float = 25.0,
    clip_quantile: float = 0.99,
    target_ess_fraction: float = 0.25,
) -> WeightResult:
    current_states = states_grid[batch.states]
    current_features = ratio_features.transform(current_states, batch.actions)
    next_expected_features = ratio_features.expected_under_policy(
        states_grid[batch.next_states],
        target_policy[batch.next_states],
    )
    target_initial_features = ratio_features.expected_under_policy(states_grid, target_policy)
    rho0 = np.asarray(behavior_state_dist, dtype=np.float64).reshape(-1)
    rho0 /= np.maximum(rho0.sum(), 1e-300)
    rhs_feature = rho0 @ target_initial_features

    delta = current_features - float(gamma_weight) * next_expected_features
    rhs = (1.0 - float(gamma_weight)) * rhs_feature
    diagnostics: dict[str, float | str] = {"cv_ridge_selected": 0.0}
    if cv_ridge:
        if cv_ridge_grid is None:
            cv_ridge_grid = RICH_TIKHONOV_RIDGE_GRID
        selection_rule = str(cv_selection_rule)
        if selection_rule not in {"min", "one_se"}:
            raise ValueError("cv_selection_rule must be 'min' or 'one_se'")
        selected_ridge, cv_diagnostics = _select_cv_ridge(
            current_features,
            delta,
            rhs,
            ridge_grid=tuple(float(value) for value in cv_ridge_grid),
            cv_folds=cv_folds,
            cv_seed=cv_seed,
            normalization_penalty=normalization_penalty,
            score_ridge_dual=cv_score_ridge_dual,
            selection_rule=selection_rule,
        )
        ridge_primal = selected_ridge
        ridge_dual = selected_ridge
        diagnostics.update(cv_diagnostics)

    fit = _moment_matrices(
        current_features,
        delta,
        rhs,
        ridge_primal=float(ridge_primal),
        ridge_dual=float(ridge_dual),
        normalization_penalty=normalization_penalty,
    )
    raw = current_features @ fit.alpha
    weights, meta = stabilize_weights(
        raw,
        min_weight=min_weight,
        max_weight=max_weight,
        clip_quantile=clip_quantile,
        target_ess_fraction=target_ess_fraction,
    )
    moment_violation = fit.a_mat.T @ fit.alpha - rhs
    diagnostics.update(
        {
            "weight_solver": "closed_form_shared_rbf_moment",
            "gamma_weight": float(gamma_weight),
            "ridge_primal": float(ridge_primal),
            "ridge_dual": float(ridge_dual),
            "normalization_error": float(np.mean(raw) - 1.0),
            "moment_violation_l2": float(np.linalg.norm(moment_violation)),
        }
    )
    diagnostics.update(meta)
    diagnostics.update(summarize_weights(weights))
    return WeightResult(weights=weights, raw_weights=raw, diagnostics=diagnostics)
