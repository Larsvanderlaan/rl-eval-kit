from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .configs import NeuralRatioConfig, WeightEstimatorConfig
from .envs import LinearGaussianEnv
from .features import RatioFeatureMap
from .policies import GaussianLinearPolicy
from .truth import GaussianMixtureDensity


@dataclass
class DiscountedRatioEstimate:
    alpha: np.ndarray
    beta: np.ndarray
    raw_weights: np.ndarray
    processed_weights: np.ndarray
    clipped_weights: np.ndarray
    diagnostics: dict[str, float]


class _PositiveMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] | list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(nn.SiLU())
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(x)).squeeze(-1) + 1e-8


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _apply_uniform_mix(weights: np.ndarray, uniform_mix: float) -> np.ndarray:
    mixed = (1.0 - uniform_mix) * weights + uniform_mix * np.ones_like(weights)
    return mixed / np.maximum(np.mean(mixed), 1e-12)


def effective_sample_size(weights: np.ndarray) -> float:
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float((weights_arr.sum() ** 2) / np.maximum(np.sum(weights_arr**2), 1e-12))


def process_raw_weights(
    raw_weights: np.ndarray,
    *,
    min_weight: float = 1e-8,
    clip_quantile: float | None = None,
    max_weight: float | None = None,
    target_ess_fraction: float | None = None,
    uniform_mix: float = 0.0,
    max_uniform_mix: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    positive = np.maximum(raw, min_weight)
    clip_level = None
    if clip_quantile is not None:
        clip_level = float(np.quantile(positive, clip_quantile))
    if max_weight is not None:
        clip_level = min(clip_level, max_weight) if clip_level is not None else float(max_weight)
    clipped = positive.copy()
    clipped_count = 0
    if clip_level is not None:
        clipped_count = int(np.sum(clipped > clip_level))
        clipped = np.minimum(clipped, clip_level)
    clipped = clipped / np.maximum(np.mean(clipped), 1e-12)
    ess_before = effective_sample_size(clipped) / max(len(clipped), 1)

    chosen_uniform_mix = float(max(uniform_mix, 0.0))
    if target_ess_fraction is not None and ess_before < target_ess_fraction:
        lo = chosen_uniform_mix
        hi = max(lo, max_uniform_mix)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            cand = _apply_uniform_mix(clipped, mid)
            ess_mid = effective_sample_size(cand) / max(len(cand), 1)
            if ess_mid >= target_ess_fraction:
                hi = mid
            else:
                lo = mid
        chosen_uniform_mix = hi
    processed = _apply_uniform_mix(clipped, chosen_uniform_mix)
    diagnostics = {
        "clip_level": np.nan if clip_level is None else float(clip_level),
        "fraction_clipped": float(clipped_count / max(len(clipped), 1)),
        "chosen_uniform_mix": float(chosen_uniform_mix),
        "ess_fraction_before_mix": float(ess_before),
        "ess_fraction_after_mix": float(effective_sample_size(processed) / max(len(processed), 1)),
    }
    return processed, diagnostics


def process_ess_adaptive_winsor_weights(
    raw_weights: np.ndarray,
    *,
    min_weight: float = 1e-8,
    target_ess_fraction: float = 0.40,
    max_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Winsorize at the loosest cap that attains a target ESS fraction.

    The cap is chosen from the empirical weights only.  It does not use rewards,
    Bellman residuals, oracle values, or test targets, so it is a clean
    stabilization rule for the weighted FQE regression.
    """

    raw = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    positive = np.maximum(raw, min_weight)
    n = max(positive.shape[0], 1)
    upper = float(np.max(positive))
    if max_weight is not None:
        upper = min(upper, float(max_weight))

    def _clip_normalize(cap: float) -> np.ndarray:
        clipped = np.minimum(positive, cap)
        return clipped / np.maximum(float(np.mean(clipped)), 1e-12)

    initial = _clip_normalize(upper)
    initial_ess = effective_sample_size(initial) / n
    if initial_ess >= target_ess_fraction:
        selected_cap = upper
        processed = initial
    else:
        lo = float(np.min(positive))
        hi = upper
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            candidate = _clip_normalize(mid)
            ess_mid = effective_sample_size(candidate) / n
            if ess_mid >= target_ess_fraction:
                lo = mid
            else:
                hi = mid
        selected_cap = lo
        processed = _clip_normalize(selected_cap)
    clipped_count = int(np.sum(positive > selected_cap))
    diagnostics = {
        "clip_level": float(selected_cap),
        "fraction_clipped": float(clipped_count / n),
        "chosen_uniform_mix": 0.0,
        "ess_fraction_before_winsor": float(initial_ess),
        "ess_fraction_after_winsor": float(effective_sample_size(processed) / n),
    }
    return processed, diagnostics


def summarize_weights(
    weights: np.ndarray,
    *,
    fraction_clipped: float = 0.0,
    chosen_uniform_mix: float = 0.0,
    clip_level: float | None = None,
) -> dict[str, float]:
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "weight_mean": float(np.mean(weights_arr)),
        "weight_std": float(np.std(weights_arr)),
        "weight_max": float(np.max(weights_arr)),
        "weight_q90": float(np.quantile(weights_arr, 0.90)),
        "weight_q95": float(np.quantile(weights_arr, 0.95)),
        "weight_q99": float(np.quantile(weights_arr, 0.99)),
        "effective_sample_size": float(effective_sample_size(weights_arr)),
        "effective_sample_size_fraction": float(effective_sample_size(weights_arr) / max(len(weights_arr), 1)),
        "fraction_clipped": float(fraction_clipped),
        "chosen_uniform_mix": float(chosen_uniform_mix),
        "clip_level": np.nan if clip_level is None else float(clip_level),
    }


def ratio_quality(oracle_weights: np.ndarray, candidate_weights: np.ndarray) -> dict[str, float]:
    """Finite-sample diagnostics comparing a candidate ratio to oracle weights."""

    oracle = np.asarray(oracle_weights, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_weights, dtype=np.float64).reshape(-1)
    if oracle.shape != candidate.shape:
        raise ValueError("oracle_weights and candidate_weights must have the same shape.")
    oracle_pos = np.maximum(oracle, 1e-12)
    candidate_pos = np.maximum(candidate, 1e-12)
    log_ratio_rmse = float(np.sqrt(np.mean((np.log(candidate_pos) - np.log(oracle_pos)) ** 2)))
    oracle_sd = float(np.std(oracle_pos))
    candidate_sd = float(np.std(candidate_pos))
    if oracle_sd <= 1e-12 and candidate_sd <= 1e-12:
        corr = 1.0 if np.allclose(oracle_pos, candidate_pos) else 0.0
    elif oracle_sd <= 1e-12 or candidate_sd <= 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(oracle_pos, candidate_pos)[0, 1])
    mae = float(np.mean(np.abs(candidate_pos - oracle_pos)))
    rel_mse = float(np.mean((candidate_pos - oracle_pos) ** 2) / np.maximum(np.mean(oracle_pos**2), 1e-12))
    return {
        "oracle_log_ratio_rmse": log_ratio_rmse,
        "oracle_estimated_weight_corr": corr,
        "oracle_estimated_weight_mae": mae,
        "oracle_estimated_weight_rel_mse": rel_mse,
    }


def weighted_design_condition_number(
    features: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float = 0.0,
) -> float:
    """Condition number of the weighted least-squares Gram matrix."""

    feature_arr = np.asarray(features, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if feature_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("features and weights must have the same number of rows.")
    gram = (feature_arr.T @ (weights_arr[:, None] * feature_arr)) / max(feature_arr.shape[0], 1)
    if ridge > 0.0:
        gram = gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    try:
        return float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        return float("inf")


def oracle_density_ratio(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    target_mixture: GaussianMixtureDensity,
    behavior_mixture: GaussianMixtureDensity,
) -> np.ndarray:
    points = np.concatenate(
        [
            np.asarray(states, dtype=np.float64).reshape(-1, 2),
            np.asarray(actions, dtype=np.float64).reshape(-1, 1),
        ],
        axis=1,
    )
    log_ratio = target_mixture.logpdf(points) - behavior_mixture.logpdf(points)
    log_ratio = np.clip(log_ratio, -60.0, 60.0)
    return np.exp(log_ratio)


def _raw_quadratic_log_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    s1 = states_arr[:, [0]]
    s2 = states_arr[:, [1]]
    a = actions_arr
    return np.concatenate(
        [
            np.ones_like(a),
            s1,
            s2,
            a,
            s1**2,
            s1 * s2,
            s1 * a,
            s2**2,
            s2 * a,
            a**2,
        ],
        axis=1,
    )


def _quadratic_log_features(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = _raw_quadratic_log_features(states, actions)
    mean = raw.mean(axis=0, keepdims=True)
    scale = raw.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    mean[:, 0] = 0.0
    scale[:, 0] = 1.0
    return (raw - mean) / scale, {"mean": mean.reshape(-1), "scale": scale.reshape(-1)}


def exponential_quadratic_moment_raw_weights(
    states: np.ndarray,
    actions: np.ndarray,
    ratio: DiscountedRatioEstimate,
) -> np.ndarray:
    if ratio.alpha.size == 0 or ratio.beta.size != 2 * ratio.alpha.size:
        raise ValueError("The supplied ratio does not contain an exponential-quadratic model.")
    raw = _raw_quadratic_log_features(states, actions)
    dim = ratio.alpha.size
    mean = ratio.beta[:dim].reshape(1, -1)
    scale = ratio.beta[dim:].reshape(1, -1)
    psi = (raw - mean) / np.maximum(scale, 1e-12)
    log_w = np.clip(psi @ ratio.alpha.reshape(-1), -14.0, 14.0)
    return np.exp(log_w)


def estimate_exponential_quadratic_moment_ratio(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    ratio_feature_map: RatioFeatureMap,
    seed: int,
    ridge: float = 1e-3,
    critic_ridge: float = 1e-4,
    normalization_penalty: float = 10.0,
    learning_rate: float = 3e-2,
    max_steps: int = 900,
    batch_size: int = 1024,
    valid_fraction: float = 0.20,
    patience: int = 80,
) -> DiscountedRatioEstimate:
    """Positive quadratic log-ratio fit by stationary-flow moment batching.

    Unlike the behavior-centered linear/RBF moment estimator, this model uses
    global quadratic log-weights,
    ``w(s,a)=exp(theta^T psi(s,a))``.  It therefore extrapolates in the same
    functional form as a Gaussian density ratio while still fitting only the
    stationary-flow moments from offline transitions and the target policy.
    """

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    n = states_arr.shape[0]
    log_features_np, standardizer = _quadratic_log_features(states_arr, actions_arr)
    current_features = ratio_feature_map.transform(states_arr, actions_arr)
    next_expected_features = ratio_feature_map.expected_given_state(next_states, target_policy)
    initial_rhs = ratio_feature_map.expectation_under_initial_distribution(
        env.config.initial_mean,
        env.config.initial_cov,
        target_policy,
    )
    delta_np = current_features - float(gamma) * next_expected_features
    rhs_np = np.zeros(current_features.shape[1], dtype=np.float64)
    if float(gamma) < 1.0 - 1e-12:
        rhs_np = (1.0 - float(gamma)) * initial_rhs
    critic_gram = (current_features.T @ current_features) / max(n, 1)
    h_inv_np = _solve_spd(
        critic_gram + float(critic_ridge) * np.eye(critic_gram.shape[0], dtype=np.float64),
        np.eye(critic_gram.shape[0], dtype=np.float64),
    )

    indices = np.arange(n)
    rng.shuffle(indices)
    n_valid = int(round(float(valid_fraction) * n))
    valid_idx_np = indices[:n_valid]
    train_idx_np = indices[n_valid:] if n_valid < n else indices
    if train_idx_np.size == 0:
        train_idx_np = indices
        valid_idx_np = indices[:0]

    psi = torch.as_tensor(log_features_np, dtype=torch.float64)
    delta = torch.as_tensor(delta_np, dtype=torch.float64)
    rhs = torch.as_tensor(rhs_np, dtype=torch.float64)
    h_inv = torch.as_tensor(h_inv_np, dtype=torch.float64)
    theta = torch.zeros(psi.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=float(learning_rate))
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)
    valid_idx = torch.as_tensor(valid_idx_np, dtype=torch.long)
    score_idx = valid_idx if valid_idx.numel() > 0 else train_idx
    best_theta = theta.detach().clone()
    best_score = float("inf")
    stale = 0
    train_scores: list[float] = []
    valid_scores: list[float] = []

    def objective(idx: torch.Tensor) -> torch.Tensor:
        log_w = torch.clamp(psi[idx] @ theta, -14.0, 14.0)
        weights = torch.exp(log_w)
        moment = (delta[idx].T @ weights) / max(int(idx.numel()), 1) - rhs
        norm_error = torch.mean(weights) - 1.0
        penalty = float(ridge) * torch.sum(theta[1:] ** 2)
        return 0.5 * (moment @ (h_inv @ moment)) + float(normalization_penalty) * norm_error**2 + penalty

    for step in range(1, int(max_steps) + 1):
        if int(batch_size) > 0 and train_idx.numel() > int(batch_size):
            batch_np = rng.choice(train_idx_np, size=int(batch_size), replace=False)
            idx = torch.as_tensor(batch_np, dtype=torch.long)
        else:
            idx = train_idx
        optimizer.zero_grad(set_to_none=True)
        loss = objective(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == int(max_steps):
            with torch.no_grad():
                train_score = float(objective(train_idx).item())
                valid_score = float(objective(score_idx).item())
            train_scores.append(train_score)
            valid_scores.append(valid_score)
            if valid_score + 1e-7 < best_score:
                best_score = valid_score
                best_theta = theta.detach().clone()
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    break

    with torch.no_grad():
        theta.copy_(best_theta)
        log_w = torch.clamp(psi @ theta, -14.0, 14.0)
        raw_weights = torch.exp(log_w).detach().cpu().numpy().astype(np.float64)
    theta_np = best_theta.detach().cpu().numpy().astype(np.float64)
    standardizer_packed = np.concatenate([standardizer["mean"], standardizer["scale"]], axis=0)
    processed_weights, processed_meta = process_raw_weights(raw_weights, min_weight=1e-8)
    diagnostics = {
        "solver": "exponential_quadratic_moment",
        "normalization_error": float(np.mean(raw_weights) - 1.0),
        "moment_violation_l2": float(
            np.linalg.norm((delta_np.T @ raw_weights) / max(n, 1) - rhs_np)
        ),
        "raw_min": float(np.min(raw_weights)),
        "raw_max": float(np.max(raw_weights)),
        "processed_ess_fraction": float(
            effective_sample_size(processed_weights) / max(processed_weights.shape[0], 1)
        ),
        "clipped_ess_fraction": float(
            effective_sample_size(processed_weights) / max(processed_weights.shape[0], 1)
        ),
        "processed_clip_level": processed_meta["clip_level"],
        "clipped_clip_level": processed_meta["clip_level"],
        "clipped_fraction_clipped": processed_meta["fraction_clipped"],
        "ridge": float(ridge),
        "best_valid_score": float(best_score),
        "train_objective_last": float(train_scores[-1]) if train_scores else float("nan"),
        "valid_objective_last": float(valid_scores[-1]) if valid_scores else float("nan"),
        "feature_scale_min": float(np.min(standardizer["scale"])),
    }
    return DiscountedRatioEstimate(
        alpha=theta_np,
        beta=standardizer_packed.astype(np.float64),
        raw_weights=raw_weights,
        processed_weights=processed_weights,
        clipped_weights=processed_weights,
        diagnostics=diagnostics,
    )


def estimate_discounted_occupancy_ratio(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    ratio_feature_map: RatioFeatureMap,
    config: WeightEstimatorConfig,
) -> DiscountedRatioEstimate:
    current_features = ratio_feature_map.transform(states, actions)
    next_expected_features = ratio_feature_map.expected_given_state(next_states, target_policy)
    initial_rhs = ratio_feature_map.expectation_under_initial_distribution(
        env.config.initial_mean,
        env.config.initial_cov,
        target_policy,
    )
    delta = current_features - gamma * next_expected_features
    A = (current_features.T @ delta) / current_features.shape[0]
    B = (current_features.T @ current_features) / current_features.shape[0]
    H = B + config.ridge_dual * np.eye(B.shape[0], dtype=np.float64)
    H_inv = _solve_spd(H, np.eye(H.shape[0], dtype=np.float64))
    m = current_features.mean(axis=0)
    rhs = (1.0 - gamma) * initial_rhs
    system = (
        A @ H_inv @ A.T
        + 2.0 * config.normalization_penalty * np.outer(m, m)
        + config.ridge_primal * np.eye(A.shape[0], dtype=np.float64)
    )
    rhs_alpha = A @ (H_inv @ rhs) + 2.0 * config.normalization_penalty * m
    alpha = _solve_spd(system, rhs_alpha)
    beta = H_inv @ (A.T @ alpha - rhs)
    raw_weights = current_features @ alpha

    processed_weights, processed_meta = process_raw_weights(
        raw_weights,
        min_weight=config.min_weight,
    )
    clipped_weights, clipped_meta = process_raw_weights(
        raw_weights,
        min_weight=config.min_weight,
        clip_quantile=config.clipped_clip_quantile,
        max_weight=config.clipped_max_weight,
        target_ess_fraction=config.clipped_target_ess_fraction,
        uniform_mix=config.clipped_uniform_mix,
        max_uniform_mix=config.clipped_max_uniform_mix,
    )
    diagnostics = {
        "solver": "linear_reduced_moment",
        "normalization_error": float(m @ alpha - 1.0),
        "moment_violation_l2": float(np.linalg.norm(A.T @ alpha - rhs)),
        "raw_min": float(np.min(raw_weights)),
        "raw_max": float(np.max(raw_weights)),
        "processed_ess_fraction": float(
            effective_sample_size(processed_weights) / max(processed_weights.shape[0], 1)
        ),
        "clipped_ess_fraction": float(
            effective_sample_size(clipped_weights) / max(clipped_weights.shape[0], 1)
        ),
        "processed_clip_level": processed_meta["clip_level"],
        "clipped_clip_level": clipped_meta["clip_level"],
        "clipped_fraction_clipped": clipped_meta["fraction_clipped"],
    }
    return DiscountedRatioEstimate(
        alpha=alpha,
        beta=beta,
        raw_weights=raw_weights,
        processed_weights=processed_weights,
        clipped_weights=clipped_weights,
        diagnostics=diagnostics,
    )


def _standardize_train_full(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train = x[train_idx]
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, {"mean": mean.reshape(-1), "scale": scale.reshape(-1)}


def estimate_neural_discounted_occupancy_ratio(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    ratio_gamma: float,
    ratio_feature_map: RatioFeatureMap,
    config: NeuralRatioConfig,
    seed: int,
) -> DiscountedRatioEstimate:
    """Estimate discounted or stationary ratios with a positive neural model.

    The critic is the same finite RBF/polynomial basis used by the linear
    estimator. `ratio_gamma == 1` gives the stationary moment equation with
    zero initial-state RHS and a mean-one normalization penalty.
    """

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(config.device)

    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    weight_x = np.concatenate([states_arr, actions_arr], axis=1).astype(np.float32)
    n = weight_x.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    n_valid = int(round(config.valid_fraction * n))
    valid_idx_np = indices[:n_valid]
    train_idx_np = indices[n_valid:]
    if train_idx_np.size == 0:
        train_idx_np = indices
        valid_idx_np = indices[:0]
    weight_x_std, standardizer = _standardize_train_full(weight_x, train_idx_np)

    current_features = ratio_feature_map.transform(states_arr, actions_arr)
    next_expected_features = ratio_feature_map.expected_given_state(next_states, target_policy)
    initial_rhs = ratio_feature_map.expectation_under_initial_distribution(
        env.config.initial_mean,
        env.config.initial_cov,
        target_policy,
    )
    gamma = float(ratio_gamma)
    delta = current_features - gamma * next_expected_features
    rhs_np = np.zeros(current_features.shape[1], dtype=np.float64)
    if gamma < 1.0 - 1e-12:
        rhs_np = (1.0 - gamma) * initial_rhs

    B = (current_features.T @ current_features) / current_features.shape[0]
    H = B + config.critic_ridge * np.eye(B.shape[0], dtype=np.float64)
    H_inv_np = _solve_spd(H, np.eye(H.shape[0], dtype=np.float64))

    x_t = torch.as_tensor(weight_x_std, dtype=torch.float32, device=device)
    delta_t = torch.as_tensor(delta, dtype=torch.float64, device=device)
    rhs_t = torch.as_tensor(rhs_np, dtype=torch.float64, device=device)
    h_inv_t = torch.as_tensor(H_inv_np, dtype=torch.float64, device=device)
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_idx_np, dtype=torch.long, device=device)

    model = _PositiveMLP(input_dim=weight_x.shape[1], hidden_dims=tuple(config.hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("inf")
    patience = 0
    history_train: list[float] = []
    history_valid: list[float] = []

    def objective(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights_t = model(x_t[idx]).to(torch.float64)
        moment = (delta_t[idx].T @ weights_t) / max(int(idx.numel()), 1) - rhs_t
        sup_value = 0.5 * (moment @ (h_inv_t @ moment))
        norm_error = torch.mean(weights_t) - 1.0
        loss = sup_value + config.normalization_penalty * norm_error.pow(2)
        return loss, moment, norm_error

    score_idx = valid_idx if valid_idx.numel() > 0 else train_idx
    for _step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _moment, _norm = objective(train_idx)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        if _step == 1 or _step % 10 == 0 or _step == config.max_steps:
            with torch.no_grad():
                train_loss, _train_moment, _train_norm = objective(train_idx)
                valid_loss, _valid_moment, _valid_norm = objective(score_idx)
            train_score = float(train_loss.item())
            valid_score = float(valid_loss.item())
            history_train.append(train_score)
            history_valid.append(valid_score)
            if valid_score + config.min_improvement < best_score:
                best_score = valid_score
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw_weights_t = model(x_t).to(torch.float64)
        full_loss, full_moment, full_norm = objective(torch.arange(n, dtype=torch.long, device=device))
        raw_weights = raw_weights_t.detach().cpu().numpy().astype(np.float64)

    processed_weights, processed_meta = process_raw_weights(
        raw_weights,
        min_weight=config.min_weight,
    )
    clipped_weights, clipped_meta = process_raw_weights(
        raw_weights,
        min_weight=config.min_weight,
        clip_quantile=config.clip_quantile,
        max_weight=config.max_weight,
        target_ess_fraction=config.target_ess_fraction,
        max_uniform_mix=config.max_uniform_mix,
    )
    diagnostics = {
        "solver": "neural_positive_rkhs_moment",
        "ratio_gamma": float(ratio_gamma),
        "normalization_error": float(np.mean(raw_weights) - 1.0),
        "moment_violation_l2": float(torch.norm(full_moment).item()),
        "full_reduced_objective": float(full_loss.item()),
        "normalization_error_full": float(full_norm.item()),
        "raw_min": float(np.min(raw_weights)),
        "raw_max": float(np.max(raw_weights)),
        "processed_ess_fraction": float(
            effective_sample_size(processed_weights) / max(processed_weights.shape[0], 1)
        ),
        "clipped_ess_fraction": float(
            effective_sample_size(clipped_weights) / max(clipped_weights.shape[0], 1)
        ),
        "processed_clip_level": processed_meta["clip_level"],
        "clipped_clip_level": clipped_meta["clip_level"],
        "clipped_fraction_clipped": clipped_meta["fraction_clipped"],
        "best_valid_score": float(best_score),
        "n_train": float(train_idx_np.size),
        "n_valid": float(valid_idx_np.size),
        "standardizer_mean_l2": float(np.linalg.norm(standardizer["mean"])),
        "standardizer_scale_min": float(np.min(standardizer["scale"])),
        "train_objective_last": float(history_train[-1]) if history_train else float("nan"),
        "valid_objective_last": float(history_valid[-1]) if history_valid else float("nan"),
    }
    return DiscountedRatioEstimate(
        alpha=np.zeros(0, dtype=np.float64),
        beta=np.zeros(0, dtype=np.float64),
        raw_weights=raw_weights,
        processed_weights=processed_weights,
        clipped_weights=clipped_weights,
        diagnostics=diagnostics,
    )


def feature_calibration_l2(
    weights: np.ndarray,
    sample_features: np.ndarray,
    target_features: np.ndarray,
) -> float:
    """Root-mean-squared feature mean error induced by weighted samples."""

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    phi = np.asarray(sample_features, dtype=np.float64)
    target_phi = np.asarray(target_features, dtype=np.float64)
    weighted_mean = np.mean((w / np.maximum(np.mean(w), 1e-12))[:, None] * phi, axis=0)
    target_mean = np.mean(target_phi, axis=0)
    return float(np.sqrt(np.mean((weighted_mean - target_mean) ** 2)))
