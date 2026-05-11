from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TransitionBatch
from .env import GridMDP
from .features import QFeatureMap, RatioFeatureMap, linear_q_features, neural_q_features
from .metrics import compute_q_metrics
from .soft_dp import bellman_operator, soft_value
from .weights import RICH_TIKHONOV_RIDGE_GRID


@dataclass(frozen=True)
class FQIConfig:
    gamma: float
    tau_final: float
    n_iters: int
    ridge: float
    metrics_stride: int = 5
    anneal_taus: tuple[float, ...] = (0.2, 0.05, 0.01, 0.003, 0.001)


def tau_sequence(schedule: str, config: FQIConfig) -> list[float]:
    if schedule == "direct":
        return [float(config.tau_final)] * int(config.n_iters)
    if schedule != "annealed":
        raise ValueError(f"Unknown schedule '{schedule}'.")
    stages = list(config.anneal_taus)
    n_iters = int(config.n_iters)
    base = n_iters // len(stages)
    remainder = n_iters % len(stages)
    seq: list[float] = []
    for idx, tau in enumerate(stages):
        count = base + (1 if idx < remainder else 0)
        seq.extend([float(tau)] * count)
    return seq[:n_iters]


def _weighted_ridge_solution(phi: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
    w = w / np.maximum(np.mean(w), 1e-300)
    gram = (phi.T @ (w[:, None] * phi)) / max(phi.shape[0], 1)
    rhs = (phi.T @ (w * y)) / max(phi.shape[0], 1)
    system = gram + float(ridge) * np.eye(phi.shape[1], dtype=np.float64)
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _q_features(
    states: np.ndarray,
    actions: np.ndarray,
    action_vectors: np.ndarray,
    q_feature_map: QFeatureMap | None,
) -> np.ndarray:
    if q_feature_map is None:
        return linear_q_features(states, actions, action_vectors)
    return q_feature_map.transform(states, actions)


def _grid_feature_matrix(mdp: GridMDP, q_feature_map: QFeatureMap | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_ids, action_ids = np.meshgrid(np.arange(mdp.n_states), np.arange(mdp.n_actions), indexing="ij")
    state_ids = state_ids.reshape(-1)
    action_ids = action_ids.reshape(-1)
    phi_grid = _q_features(mdp.states[state_ids], action_ids, mdp.actions, q_feature_map)
    return phi_grid, state_ids, action_ids


def _append_metrics(
    rows: list[dict[str, float | int | str]],
    q_values: np.ndarray,
    *,
    iteration: int,
    is_final: bool,
    mdp: GridMDP,
    q_star: np.ndarray,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    fqi_config: FQIConfig,
    reference_value: float,
    rho0: np.ndarray,
    base_meta: dict[str, float | int | str],
    q_feature_map: QFeatureMap | None = None,
) -> None:
    compute_value = is_final or iteration == 0 or iteration % max(fqi_config.metrics_stride, 1) == 0
    metrics = compute_q_metrics(
        q_values,
        mdp=mdp,
        q_star=q_star,
        target_sa_dist=target_sa_dist,
        behavior_sa_dist=behavior_sa_dist,
        gamma=fqi_config.gamma,
        tau_final=fqi_config.tau_final,
        reference_value=reference_value,
        rho0=rho0,
        compute_value=compute_value,
        projection_ridge=fqi_config.ridge,
        q_feature_map=q_feature_map,
    )
    row = dict(base_meta)
    row.update(metrics)
    row["iteration"] = int(iteration)
    row["is_final"] = int(is_final)
    rows.append(row)


def run_population_linear_fqi(
    *,
    mdp: GridMDP,
    q_star: np.ndarray,
    projection_sa_dist: np.ndarray,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    schedule: str,
    fqi_config: FQIConfig,
    reference_value: float,
    rho0: np.ndarray,
    base_meta: dict[str, float | int | str],
    q_feature_map: QFeatureMap | None = None,
) -> list[dict[str, float | int | str]]:
    phi_all, _state_ids, _action_ids = _grid_feature_matrix(mdp, q_feature_map)
    weights_all = projection_sa_dist.reshape(-1)
    theta = np.zeros(phi_all.shape[1], dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    seq = tau_sequence(schedule, fqi_config)
    for iteration, tau in enumerate(seq, start=1):
        q_grid = (phi_all @ theta).reshape(mdp.n_states, mdp.n_actions)
        targets = bellman_operator(mdp.transition, mdp.reward, q_grid, fqi_config.gamma, tau).reshape(-1)
        theta = _weighted_ridge_solution(phi_all, targets, weights_all, fqi_config.ridge)
        q_new = (phi_all @ theta).reshape(mdp.n_states, mdp.n_actions)
        is_final = iteration == len(seq)
        if is_final or iteration % max(fqi_config.metrics_stride, 1) == 0 or iteration == 1:
            meta = dict(base_meta)
            meta["tau"] = float(tau)
            _append_metrics(
                rows,
                q_new,
                iteration=iteration,
                is_final=is_final,
                mdp=mdp,
                q_star=q_star,
                target_sa_dist=target_sa_dist,
                behavior_sa_dist=behavior_sa_dist,
                fqi_config=fqi_config,
                reference_value=reference_value,
                rho0=rho0,
                base_meta=meta,
                q_feature_map=q_feature_map,
            )
    return rows


def run_sample_linear_fqi(
    *,
    mdp: GridMDP,
    batch: TransitionBatch,
    sample_weights: np.ndarray,
    q_star: np.ndarray,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    schedule: str,
    fqi_config: FQIConfig,
    reference_value: float,
    rho0: np.ndarray,
    base_meta: dict[str, float | int | str],
    q_feature_map: QFeatureMap | None = None,
) -> list[dict[str, float | int | str]]:
    phi = _q_features(mdp.states[batch.states], batch.actions, mdp.actions, q_feature_map)
    phi_grid, _state_ids, _action_ids = _grid_feature_matrix(mdp, q_feature_map)
    theta = np.zeros(phi.shape[1], dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    seq = tau_sequence(schedule, fqi_config)
    for iteration, tau in enumerate(seq, start=1):
        q_grid = (phi_grid @ theta).reshape(mdp.n_states, mdp.n_actions)
        v_next = soft_value(q_grid[batch.next_states], tau)
        targets = batch.rewards + fqi_config.gamma * v_next
        theta = _weighted_ridge_solution(phi, targets, sample_weights, fqi_config.ridge)
        q_new = (phi_grid @ theta).reshape(mdp.n_states, mdp.n_actions)
        is_final = iteration == len(seq)
        if is_final or iteration % max(fqi_config.metrics_stride, 1) == 0 or iteration == 1:
            meta = dict(base_meta)
            meta["tau"] = float(tau)
            _append_metrics(
                rows,
                q_new,
                iteration=iteration,
                is_final=is_final,
                mdp=mdp,
                q_star=q_star,
                target_sa_dist=target_sa_dist,
                behavior_sa_dist=behavior_sa_dist,
                fqi_config=fqi_config,
                reference_value=reference_value,
                rho0=rho0,
                base_meta=meta,
                q_feature_map=q_feature_map,
            )
    return rows


def _solve_minimax_linear_step(
    phi: np.ndarray,
    critic: np.ndarray,
    targets: np.ndarray,
    *,
    q_ridge: float,
    critic_ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_obs = max(phi.shape[0], 1)
    a_mat = (critic.T @ phi) / n_obs
    b_vec = (critic.T @ targets) / n_obs
    h_mat = (critic.T @ critic) / n_obs + float(critic_ridge) * np.eye(critic.shape[1], dtype=np.float64)
    try:
        h_inv_a = np.linalg.solve(h_mat, a_mat)
        h_inv_b = np.linalg.solve(h_mat, b_vec)
    except np.linalg.LinAlgError:
        h_inv_a = np.linalg.lstsq(h_mat, a_mat, rcond=None)[0]
        h_inv_b = np.linalg.lstsq(h_mat, b_vec, rcond=None)[0]
    system = a_mat.T @ h_inv_a + float(q_ridge) * np.eye(phi.shape[1], dtype=np.float64)
    rhs = a_mat.T @ h_inv_b
    try:
        theta = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(system, rhs, rcond=None)[0]
    moment = a_mat @ theta - b_vec
    try:
        critic_coef = np.linalg.solve(h_mat, moment)
    except np.linalg.LinAlgError:
        critic_coef = np.linalg.lstsq(h_mat, moment, rcond=None)[0]
    diagnostics = {
        "minimax_residual_norm": float(np.sqrt(max(moment @ critic_coef, 0.0))),
        "minimax_moment_l2": float(np.linalg.norm(moment)),
        "minimax_critic_norm": float(np.linalg.norm(critic_coef)),
        "minimax_q_norm": float(np.linalg.norm(theta)),
    }
    return theta, diagnostics


def _score_minimax_fit(
    phi: np.ndarray,
    critic: np.ndarray,
    targets: np.ndarray,
    theta: np.ndarray,
    *,
    critic_ridge: float,
    q_ridge: float,
) -> float:
    n_obs = max(phi.shape[0], 1)
    moment = (critic.T @ (phi @ theta - targets)) / n_obs
    h_mat = (critic.T @ critic) / n_obs + float(critic_ridge) * np.eye(critic.shape[1], dtype=np.float64)
    try:
        scaled = np.linalg.solve(h_mat, moment)
    except np.linalg.LinAlgError:
        scaled = np.linalg.lstsq(h_mat, moment, rcond=None)[0]
    return float(moment @ scaled + float(q_ridge) * (theta @ theta))


def _select_minimax_ridge(
    phi: np.ndarray,
    critic: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_grid: tuple[float, ...],
    cv_folds: int,
    cv_seed: int,
    selection_rule: str,
) -> tuple[float, dict[str, float | str]]:
    grid = tuple(float(value) for value in ridge_grid if float(value) >= 0.0)
    if len(grid) == 0:
        raise ValueError("minimax ridge grid must contain at least one nonnegative value")
    n_obs = phi.shape[0]
    n_folds = max(2, min(int(cv_folds), n_obs))
    rng = np.random.default_rng(int(cv_seed))
    shuffled = rng.permutation(n_obs)
    folds = np.array_split(shuffled, n_folds)
    candidate_scores: list[float] = []
    candidate_ses: list[float] = []
    for ridge in grid:
        fold_scores: list[float] = []
        for fold in folds:
            if fold.size == 0:
                continue
            train_mask = np.ones(n_obs, dtype=bool)
            train_mask[fold] = False
            theta, _diag = _solve_minimax_linear_step(
                phi[train_mask],
                critic[train_mask],
                targets[train_mask],
                q_ridge=ridge,
                critic_ridge=ridge,
            )
            fold_scores.append(
                _score_minimax_fit(
                    phi[fold],
                    critic[fold],
                    targets[fold],
                    theta,
                    q_ridge=ridge,
                    critic_ridge=ridge,
                )
            )
        scores = np.asarray(fold_scores, dtype=np.float64)
        candidate_scores.append(float(np.mean(scores)))
        candidate_ses.append(float(np.std(scores, ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else 0.0)
    score_arr = np.asarray(candidate_scores, dtype=np.float64)
    min_idx = int(np.nanargmin(score_arr))
    min_score = float(score_arr[min_idx])
    one_se_threshold = min_score + float(candidate_ses[min_idx])
    eligible = [idx for idx, score in enumerate(score_arr) if np.isfinite(score) and score <= one_se_threshold]
    one_se_idx = max(eligible, key=lambda idx: grid[idx]) if eligible else min_idx
    if selection_rule not in {"min", "one_se"}:
        raise ValueError("minimax.cv_selection_rule must be 'min' or 'one_se'")
    selected_idx = one_se_idx if selection_rule == "one_se" else min_idx
    diagnostics: dict[str, float | str] = {
        "cv_minimax_ridge_selected": 1.0,
        "cv_minimax_grid": ",".join(f"{value:.12g}" for value in grid),
        "cv_minimax_folds": float(n_folds),
        "cv_minimax_selection_rule": selection_rule,
        "cv_minimax_selected_ridge": float(grid[selected_idx]),
        "cv_minimax_selected_score": float(score_arr[selected_idx]),
        "cv_minimax_selected_score_se": float(candidate_ses[selected_idx]),
        "cv_minimax_min_score": min_score,
        "cv_minimax_min_ridge": float(grid[min_idx]),
        "cv_minimax_one_se_ridge": float(grid[one_se_idx]),
    }
    for idx, (ridge, score) in enumerate(zip(grid, candidate_scores, strict=True)):
        diagnostics[f"cv_minimax_score_{idx}_ridge"] = float(ridge)
        diagnostics[f"cv_minimax_score_{idx}_moment"] = float(score)
    return float(grid[selected_idx]), diagnostics


def run_minimax_soft_q(
    *,
    mdp: GridMDP,
    batch: TransitionBatch,
    q_star: np.ndarray,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    schedule: str,
    fqi_config: FQIConfig,
    minimax_config: dict[str, object],
    ratio_features: RatioFeatureMap,
    q_feature_map: QFeatureMap | None,
    reference_value: float,
    rho0: np.ndarray,
    seed: int,
    base_meta: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    phi = _q_features(mdp.states[batch.states], batch.actions, mdp.actions, q_feature_map)
    critic = ratio_features.transform(mdp.states[batch.states], batch.actions)
    phi_grid, _state_ids, _action_ids = _grid_feature_matrix(mdp, q_feature_map)
    q_ridge = float(minimax_config.get("q_ridge", 1e-4))
    critic_ridge = float(minimax_config.get("critic_ridge", 1e-4))
    damping = float(minimax_config.get("damping", 0.5))
    damping = min(max(damping, 0.0), 1.0)
    max_inner_iter = max(int(minimax_config.get("max_inner_iter", 1)), 1)
    cv_diagnostics: dict[str, float | str] = {"cv_minimax_ridge_selected": 0.0}
    seq = tau_sequence(schedule, fqi_config)
    if bool(minimax_config.get("cv_ridge", False)):
        q_zero = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
        initial_targets = batch.rewards + fqi_config.gamma * soft_value(q_zero[batch.next_states], seq[0])
        selected_ridge, cv_diagnostics = _select_minimax_ridge(
            phi,
            critic,
            initial_targets,
            ridge_grid=tuple(float(x) for x in minimax_config.get("cv_ridge_grid", RICH_TIKHONOV_RIDGE_GRID)),
            cv_folds=int(minimax_config.get("cv_folds", 3)),
            cv_seed=int(minimax_config.get("cv_seed", seed)),
            selection_rule=str(minimax_config.get("cv_selection_rule", "min")),
        )
        q_ridge = selected_ridge
        critic_ridge = selected_ridge
    theta = np.zeros(phi.shape[1], dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    last_diag: dict[str, float] = {
        "minimax_residual_norm": float("nan"),
        "minimax_moment_l2": float("nan"),
        "minimax_critic_norm": float("nan"),
        "minimax_q_norm": 0.0,
    }
    for iteration, tau in enumerate(seq, start=1):
        for _inner in range(max_inner_iter):
            q_grid = (phi_grid @ theta).reshape(mdp.n_states, mdp.n_actions)
            targets = batch.rewards + fqi_config.gamma * soft_value(q_grid[batch.next_states], tau)
            candidate, last_diag = _solve_minimax_linear_step(
                phi,
                critic,
                targets,
                q_ridge=q_ridge,
                critic_ridge=critic_ridge,
            )
            theta = (1.0 - damping) * theta + damping * candidate
        q_new = (phi_grid @ theta).reshape(mdp.n_states, mdp.n_actions)
        is_final = iteration == len(seq)
        if is_final or iteration % max(fqi_config.metrics_stride, 1) == 0 or iteration == 1:
            meta = dict(base_meta)
            meta.update(cv_diagnostics)
            meta.update(last_diag)
            meta["tau"] = float(tau)
            meta["minimax_q_ridge"] = float(q_ridge)
            meta["minimax_critic_ridge"] = float(critic_ridge)
            meta["minimax_damping"] = float(damping)
            meta["minimax_max_inner_iter"] = float(max_inner_iter)
            _append_metrics(
                rows,
                q_new,
                iteration=iteration,
                is_final=is_final,
                mdp=mdp,
                q_star=q_star,
                target_sa_dist=target_sa_dist,
                behavior_sa_dist=behavior_sa_dist,
                fqi_config=fqi_config,
                reference_value=reference_value,
                rho0=rho0,
                base_meta=meta,
                q_feature_map=q_feature_map,
            )
    return rows


def run_neural_fqi(
    *,
    mdp: GridMDP,
    batch: TransitionBatch,
    sample_weights: np.ndarray,
    q_star: np.ndarray,
    target_sa_dist: np.ndarray,
    behavior_sa_dist: np.ndarray,
    schedule: str,
    fqi_config: FQIConfig,
    neural_config: dict[str, object],
    reference_value: float,
    rho0: np.ndarray,
    seed: int,
    base_meta: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    device = torch.device(str(neural_config.get("device", "cpu")))
    hidden_dims = tuple(int(x) for x in neural_config.get("hidden_dims", [48, 48]))
    batch_size = int(neural_config.get("batch_size", 512))
    epochs_per_iter = int(neural_config.get("epochs_per_iter", 4))
    learning_rate = float(neural_config.get("learning_rate", 5e-4))
    weight_decay = float(neural_config.get("weight_decay", 1e-4))
    target_update_tau = float(neural_config.get("target_update_tau", 1.0))
    grad_clip_norm = float(neural_config.get("grad_clip_norm", 5.0))

    class QNet(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for width in hidden_dims:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.SiLU())
                prev = width
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    x_np = neural_q_features(mdp.states[batch.states], batch.actions, mdp.actions)
    state_ids, action_ids = np.meshgrid(np.arange(mdp.n_states), np.arange(mdp.n_actions), indexing="ij")
    state_ids = state_ids.reshape(-1)
    action_ids = action_ids.reshape(-1)
    grid_x_np = neural_q_features(mdp.states[state_ids], action_ids, mdp.actions)
    mean = x_np.mean(axis=0, keepdims=True)
    scale = x_np.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_np = ((x_np - mean) / scale).astype(np.float32)
    grid_x_np = ((grid_x_np - mean) / scale).astype(np.float32)
    next_features = []
    for action in range(mdp.n_actions):
        next_features.append(
            neural_q_features(mdp.states[batch.next_states], np.full(batch.next_states.shape[0], action), mdp.actions)
        )
    next_x_np = ((np.stack(next_features, axis=1) - mean) / scale).astype(np.float32)
    weights = np.maximum(np.asarray(sample_weights, dtype=np.float64), 1e-12)
    weights = (weights / np.maximum(np.mean(weights), 1e-300)).astype(np.float32)

    model = QNet(input_dim=x_np.shape[1]).to(device)
    target_model = QNet(input_dim=x_np.shape[1]).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    reward_t = torch.as_tensor(batch.rewards.astype(np.float32), dtype=torch.float32, device=device)
    weight_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    dataset = TensorDataset(x_t, reward_t, weight_t, torch.arange(x_t.shape[0], device=device))
    grid_x_t = torch.as_tensor(grid_x_np, dtype=torch.float32, device=device)
    next_x_t = torch.as_tensor(next_x_np.reshape(-1, x_np.shape[1]), dtype=torch.float32, device=device)

    rows: list[dict[str, float | int | str]] = []
    seq = tau_sequence(schedule, fqi_config)
    for iteration, tau in enumerate(seq, start=1):
        target_model.eval()
        with torch.no_grad():
            q_next = target_model(next_x_t).reshape(batch.next_states.shape[0], mdp.n_actions)
            scaled = q_next / float(tau)
            max_scaled = torch.max(scaled, dim=1, keepdim=True).values
            v_next = float(tau) * (max_scaled.squeeze(1) + torch.log(torch.sum(torch.exp(scaled - max_scaled), dim=1)))
            targets = reward_t + float(fqi_config.gamma) * v_next
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, generator=torch.Generator().manual_seed(int(rng.integers(1_000_000))))
        model.train()
        last_loss = float("nan")
        for _epoch in range(epochs_per_iter):
            for xb, _rb, wb, idx in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = torch.mean(wb * (pred - targets[idx]) ** 2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.mul_(1.0 - target_update_tau)
                target_param.data.add_(target_update_tau * param.data)
        is_final = iteration == len(seq)
        if is_final or iteration % max(fqi_config.metrics_stride, 1) == 0 or iteration == 1:
            model.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, grid_x_t.shape[0], 8192):
                    preds.append(model(grid_x_t[start : start + 8192]).detach().cpu().numpy())
            q_new = np.concatenate(preds).reshape(mdp.n_states, mdp.n_actions)
            meta = dict(base_meta)
            meta["tau"] = float(tau)
            meta["neural_train_loss"] = last_loss
            _append_metrics(
                rows,
                q_new,
                iteration=iteration,
                is_final=is_final,
                mdp=mdp,
                q_star=q_star,
                target_sa_dist=target_sa_dist,
                behavior_sa_dist=behavior_sa_dist,
                fqi_config=fqi_config,
                reference_value=reference_value,
                rho0=rho0,
                base_meta=meta,
            )
    return rows
