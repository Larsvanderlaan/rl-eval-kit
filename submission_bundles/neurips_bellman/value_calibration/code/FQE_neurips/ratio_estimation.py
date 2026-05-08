from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from .saddle_optim import ExtragradientConfig, extragradient
from .utils import clip_normalize_weights, stabilize_weights, train_valid_split


@dataclass
class RatioEstimationResult:
    """Common output container for ratio estimators."""

    alpha: Optional[np.ndarray]
    beta: Optional[np.ndarray]
    weights: np.ndarray
    diagnostics: dict[str, float | list[float] | str]
    weight_model: Optional[nn.Module] = None
    critic_model: Optional[nn.Module] = None


@dataclass
class NeuralRatioConfig:
    """Configuration for the neural saddle-point ratio estimator."""

    hidden_dims_weight: Sequence[int] = (128, 128)
    hidden_dims_critic: Sequence[int] = (128, 128)
    activation: str = "relu"
    max_steps: int = 5_000
    batch_size: int = 512
    step_size: float = 1e-3
    ridge_weight: float = 1e-4
    ridge_critic: float = 1e-4
    normalization_penalty: float = 10.0
    positivity: str = "softplus"
    grad_clip_norm: float | None = 5.0
    log_every: int = 250
    valid_fraction: float = 0.1
    early_stopping_patience: int = 20
    min_improvement: float = 1e-5
    use_ema: bool = True
    ema_decay: float = 0.995
    clip_quantile: float | None = 0.995
    uniform_mix: float = 0.05
    target_ess_fraction: float | None = 0.4
    max_uniform_mix: float = 0.5
    device: str = "cpu"
    seed: int = 0


def _prepare_features(
    weight_features: np.ndarray,
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(weight_features, dtype=np.float64)
    g = np.asarray(critic_features, dtype=np.float64)
    gp = np.asarray(next_critic_features, dtype=np.float64)
    if d.ndim != 2 or g.ndim != 2 or gp.ndim != 2:
        raise ValueError("All feature matrices must be 2D arrays.")
    if len({d.shape[0], g.shape[0], gp.shape[0]}) != 1:
        raise ValueError("All feature matrices must have the same number of rows.")
    return d, g, gp


def _moment_matrices(
    weight_features: np.ndarray,
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
    gamma_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d, g, gp = _prepare_features(weight_features, critic_features, next_critic_features)
    n = d.shape[0]
    delta_g = g - gamma_ratio * gp
    A = (d.T @ delta_g) / n
    B = (g.T @ g) / n
    c = g.mean(axis=0)
    m = d.mean(axis=0)
    return A, B, c, m


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def positive_linear_ratio_weights(
    alpha: np.ndarray,
    features: np.ndarray,
    min_weight: float = 1e-8,
    max_weight: Optional[float] = 20.0,
    clip_quantile: float | None = 0.995,
    uniform_mix: float = 0.02,
    target_ess_fraction: float | None = 0.4,
    max_uniform_mix: float = 0.5,
) -> np.ndarray:
    """
    Predict linear ratio weights and enforce positivity by clipping at `min_weight`
    before normalizing to empirical mean one.
    """

    raw_weights = np.asarray(features, dtype=np.float64) @ np.asarray(alpha, dtype=np.float64)
    weights, _ = stabilize_weights(
        raw_weights,
        min_weight=min_weight,
        max_weight=max_weight,
        clip_quantile=clip_quantile,
        uniform_mix=uniform_mix,
        target_ess_fraction=target_ess_fraction,
        max_uniform_mix=max_uniform_mix,
    )
    return weights


def _reduced_objective(
    alpha: np.ndarray,
    A: np.ndarray,
    H_inv: np.ndarray,
    b: np.ndarray,
    m: np.ndarray,
    ridge_primal: float,
    normalization_penalty: float,
) -> float:
    residual = A.T @ alpha - b
    value = 0.5 * float(residual @ (H_inv @ residual))
    value += normalization_penalty * float((m @ alpha - 1.0) ** 2)
    value += 0.5 * ridge_primal * float(alpha @ alpha)
    return value


def _fista_elastic_net(
    A: np.ndarray,
    H_inv: np.ndarray,
    b: np.ndarray,
    m: np.ndarray,
    ridge_primal: float,
    normalization_penalty: float,
    l1_reg: float,
    max_iters: int = 3_000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Optional elastic-net refinement for the reduced convex primal problem."""

    n_params = A.shape[0]
    hessian = A @ H_inv @ A.T + 2.0 * normalization_penalty * np.outer(m, m) + ridge_primal * np.eye(n_params)
    lipschitz = max(float(np.linalg.norm(hessian, ord=2)), 1e-8)
    step = 1.0 / lipschitz

    alpha = np.zeros(n_params, dtype=np.float64)
    y = alpha.copy()
    t = 1.0

    for _ in range(max_iters):
        residual = A.T @ y - b
        grad = A @ (H_inv @ residual) + 2.0 * normalization_penalty * (m @ y - 1.0) * m + ridge_primal * y
        z = y - step * grad
        alpha_new = np.sign(z) * np.maximum(np.abs(z) - step * l1_reg, 0.0)
        if np.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = alpha_new + ((t - 1.0) / t_new) * (alpha_new - alpha)
        alpha = alpha_new
        t = t_new

    return alpha


def estimate_ratio_closed_form_linear(
    weight_features: np.ndarray,
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
    gamma_ratio: float = 1.0,
    ridge_primal: float = 1e-4,
    ridge_dual: float = 1e-4,
    normalization_penalty: float = 10.0,
    l1_reg: float = 0.0,
    min_weight: float = 1e-8,
    max_weight: Optional[float] = 20.0,
    clip_quantile: float | None = 0.995,
    uniform_mix: float = 0.02,
    target_ess_fraction: float | None = 0.4,
    max_uniform_mix: float = 0.5,
) -> RatioEstimationResult:
    """
    Closed-form linear minimax ratio estimator.

    With linear classes d_alpha = D alpha and g_beta = G beta, the inner
    maximization over beta has the exact solution

        beta*(alpha) = (B + ridge_dual I)^(-1) [A^T alpha - (1-gamma_ratio) c].

    Substituting this into the minimax objective yields a reduced convex problem
    in alpha with ridge regularization and an empirical normalization penalty.
    """

    A, B, c, m = _moment_matrices(weight_features, critic_features, next_critic_features, gamma_ratio)
    b = (1.0 - gamma_ratio) * c
    H = B + ridge_dual * np.eye(B.shape[0])
    H_inv = _solve_spd(H, np.eye(H.shape[0]))

    system = A @ H_inv @ A.T + 2.0 * normalization_penalty * np.outer(m, m) + ridge_primal * np.eye(A.shape[0])
    rhs = A @ (H_inv @ b) + 2.0 * normalization_penalty * m
    alpha = _solve_spd(system, rhs)

    solver_name = "closed_form_ridge"
    if l1_reg > 0.0:
        alpha = _fista_elastic_net(
            A=A,
            H_inv=H_inv,
            b=b,
            m=m,
            ridge_primal=ridge_primal,
            normalization_penalty=normalization_penalty,
            l1_reg=l1_reg,
        )
        solver_name = "closed_form_inner_plus_fista_elastic_net"

    beta = H_inv @ (A.T @ alpha - b)
    raw_weights = np.asarray(weight_features, dtype=np.float64) @ alpha
    weights, stabilization_meta = stabilize_weights(
        raw_weights,
        min_weight=min_weight,
        max_weight=max_weight,
        clip_quantile=clip_quantile,
        uniform_mix=uniform_mix,
        target_ess_fraction=target_ess_fraction,
        max_uniform_mix=max_uniform_mix,
        return_metadata=True,
    )

    diagnostics = {
        "solver": solver_name,
        "gamma_ratio": float(gamma_ratio),
        "reduced_objective": _reduced_objective(alpha, A, H_inv, b, m, ridge_primal, normalization_penalty),
        "normalization_error": float(np.mean(raw_weights) - 1.0),
        "moment_violation_l2": float(np.linalg.norm(A.T @ alpha - b)),
        "min_weight_raw": float(raw_weights.min()),
        "max_weight_raw": float(raw_weights.max()),
        "min_weight_processed": float(weights.min()),
        "max_weight_processed": float(weights.max()),
        "effective_max_weight": stabilization_meta["effective_max_weight"],
        "chosen_uniform_mix": float(stabilization_meta["chosen_uniform_mix"]),
        "ess_fraction_before_mix": float(stabilization_meta["ess_fraction_before_mix"]),
        "ess_fraction_after_mix": float(stabilization_meta["ess_fraction_after_mix"]),
        "target_ess_fraction": stabilization_meta["target_ess_fraction"],
    }
    return RatioEstimationResult(alpha=alpha, beta=beta, weights=weights, diagnostics=diagnostics)


def _default_extragradient_step_size(
    A: np.ndarray,
    B: np.ndarray,
    m: np.ndarray,
    ridge_primal: float,
    ridge_dual: float,
    normalization_penalty: float,
) -> float:
    """Conservative smoothness-based step size for the linear saddle problem."""

    primal_curvature = ridge_primal + 2.0 * normalization_penalty * float(np.linalg.norm(m) ** 2)
    dual_curvature = ridge_dual + float(np.linalg.norm(B, ord=2))
    coupling = float(np.linalg.norm(A, ord=2))
    lipschitz = max(primal_curvature + coupling, dual_curvature + coupling, 1e-8)
    return 0.5 / lipschitz


def estimate_ratio_saddle_linear(
    weight_features: np.ndarray,
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
    gamma_ratio: float = 1.0,
    ridge_primal: float = 1e-4,
    ridge_dual: float = 1e-4,
    normalization_penalty: float = 10.0,
    step_size: Optional[float] = None,
    max_iters: int = 5_000,
    tol: float = 1e-7,
    min_weight: float = 1e-8,
    max_weight: Optional[float] = 20.0,
    clip_quantile: float | None = 0.995,
    uniform_mix: float = 0.02,
    target_ess_fraction: float | None = 0.4,
    max_uniform_mix: float = 0.5,
) -> RatioEstimationResult:
    """Linear saddle-point solver for the same moment equations as the closed-form estimator."""

    A, B, c, m = _moment_matrices(weight_features, critic_features, next_critic_features, gamma_ratio)
    b = (1.0 - gamma_ratio) * c

    if step_size is None:
        step_size = _default_extragradient_step_size(
            A=A,
            B=B,
            m=m,
            ridge_primal=ridge_primal,
            ridge_dual=ridge_dual,
            normalization_penalty=normalization_penalty,
        )

    def objective(alpha: np.ndarray, beta: np.ndarray) -> float:
        value = float(alpha @ (A @ beta) - b @ beta)
        value -= 0.5 * float(beta @ (B @ beta))
        value += normalization_penalty * float((m @ alpha - 1.0) ** 2)
        value += 0.5 * ridge_primal * float(alpha @ alpha)
        value -= 0.5 * ridge_dual * float(beta @ beta)
        return value

    def grad_alpha(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return A @ beta + 2.0 * normalization_penalty * (m @ alpha - 1.0) * m + ridge_primal * alpha

    def grad_beta(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return A.T @ alpha - b - B @ beta - ridge_dual * beta

    result = extragradient(
        x0=np.zeros(A.shape[0], dtype=np.float64),
        y0=np.zeros(B.shape[0], dtype=np.float64),
        grad_x=grad_alpha,
        grad_y=grad_beta,
        config=ExtragradientConfig(step_size=step_size, max_iters=max_iters, tol=tol),
        objective=objective,
    )

    alpha = result.x
    beta = result.y
    raw_weights = np.asarray(weight_features, dtype=np.float64) @ alpha
    weights, stabilization_meta = stabilize_weights(
        raw_weights,
        min_weight=min_weight,
        max_weight=max_weight,
        clip_quantile=clip_quantile,
        uniform_mix=uniform_mix,
        target_ess_fraction=target_ess_fraction,
        max_uniform_mix=max_uniform_mix,
        return_metadata=True,
    )

    diagnostics = {
        "solver": "linear_extragradient",
        "gamma_ratio": float(gamma_ratio),
        "step_size": float(step_size),
        "n_iters_recorded": float(len(result.history.get("gap_proxy", []))),
        "final_gap_proxy": float(result.history["gap_proxy"][-1]),
        "normalization_error": float(np.mean(raw_weights) - 1.0),
        "moment_violation_l2": float(np.linalg.norm(A.T @ alpha - b)),
        "min_weight_raw": float(raw_weights.min()),
        "max_weight_raw": float(raw_weights.max()),
        "min_weight_processed": float(weights.min()),
        "max_weight_processed": float(weights.max()),
        "effective_max_weight": stabilization_meta["effective_max_weight"],
        "chosen_uniform_mix": float(stabilization_meta["chosen_uniform_mix"]),
        "ess_fraction_before_mix": float(stabilization_meta["ess_fraction_before_mix"]),
        "ess_fraction_after_mix": float(stabilization_meta["ess_fraction_after_mix"]),
        "target_ess_fraction": stabilization_meta["target_ess_fraction"],
    }
    return RatioEstimationResult(alpha=alpha, beta=beta, weights=weights, diagnostics=diagnostics)


class MLP(nn.Module):
    """Simple MLP used for neural ratio and critic networks."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: str = "relu") -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        act = activations.get(activation.lower(), nn.ReLU)
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(act())
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PositiveRatioNet(nn.Module):
    """Neural ratio model with a positive output map."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: str, positivity: str) -> None:
        super().__init__()
        self.backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)
        positivity = positivity.lower()
        if positivity == "softplus":
            self.positive_map = nn.Softplus()
        elif positivity == "exp":
            self.positive_map = torch.exp
        else:
            raise ValueError(f"Unsupported positivity transform '{positivity}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        out = self.positive_map(logits)
        return out + 1e-8


def _parameter_l2_penalty(module: nn.Module) -> torch.Tensor:
    penalty = None
    for param in module.parameters():
        term = torch.sum(param.pow(2))
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        return torch.tensor(0.0)
    return penalty


def _update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay)
            ema_param.data.add_((1.0 - decay) * param.data)


def _extragradient_param_step(
    module: nn.Module,
    ascent: bool,
    step_size: float,
    saved_params: list[torch.Tensor],
) -> None:
    with torch.no_grad():
        for param, saved in zip(module.parameters(), saved_params):
            if param.grad is None:
                param.copy_(saved)
                continue
            direction = 1.0 if ascent else -1.0
            param.copy_(saved + direction * step_size * param.grad)


def estimate_ratio_saddle_neural(
    weight_features: np.ndarray,
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
    gamma_ratio: float = 1.0,
    config: NeuralRatioConfig | None = None,
    min_weight: float = 1e-8,
    max_weight: Optional[float] = 20.0,
) -> RatioEstimationResult:
    """
    Neural saddle-point ratio estimator with a neural weight model and neural critic.

    The objective is the discounted flow-balance analogue of the paper's minimax
    objective:

        min_w max_v
            E[ d_w(S,A) { g_v(S,A) - gamma_ratio g_v(S',A') } ]
          - (1-gamma_ratio) E[g_v(S,A)]
          - 0.5 E[g_v(S,A)^2]
          + eta (E[d_w(S,A)] - 1)^2
          + 0.5 lambda_w ||w||_2^2
          - 0.5 lambda_v ||v||_2^2.

    We optimize it with standard alternating AdamW updates, explicit
    regularization, validation-based early stopping, and optional EMA parameter
    averaging. This is a practical production baseline for neural minimax ratio
    learning even if it is less theoretically clean than the linear closed-form
    solver.
    """

    if config is None:
        config = NeuralRatioConfig()

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    x = torch.as_tensor(np.asarray(weight_features, dtype=np.float32), device=device)
    z = torch.as_tensor(np.asarray(critic_features, dtype=np.float32), device=device)
    zp = torch.as_tensor(np.asarray(next_critic_features, dtype=np.float32), device=device)
    n = x.shape[0]
    if not (z.shape[0] == n and zp.shape[0] == n):
        raise ValueError("All feature matrices must have the same number of rows.")

    weight_net = PositiveRatioNet(
        input_dim=x.shape[1],
        hidden_dims=config.hidden_dims_weight,
        activation=config.activation,
        positivity=config.positivity,
    ).to(device)
    critic_net = MLP(
        input_dim=z.shape[1],
        hidden_dims=config.hidden_dims_critic,
        activation=config.activation,
    ).to(device)
    train_idx_np, valid_idx_np = train_valid_split(n, config.valid_fraction, rng)
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_idx_np, dtype=torch.long, device=device)

    ema_weight_net = deepcopy(weight_net).to(device) if config.use_ema else None
    ema_critic_net = deepcopy(critic_net).to(device) if config.use_ema else None
    if ema_weight_net is not None:
        ema_weight_net.eval()
        ema_critic_net.eval()
        for param in ema_weight_net.parameters():
            param.requires_grad_(False)
        for param in ema_critic_net.parameters():
            param.requires_grad_(False)

    weight_opt = torch.optim.AdamW(weight_net.parameters(), lr=config.step_size, weight_decay=config.ridge_weight)
    critic_opt = torch.optim.AdamW(critic_net.parameters(), lr=config.step_size, weight_decay=config.ridge_critic)

    history: dict[str, list[float]] = {
        "train_objective": [],
        "train_normalization_error": [],
        "valid_score": [],
        "valid_moment_violation": [],
        "valid_normalization_error": [],
    }

    def objective_for_models(
        weight_model: nn.Module,
        critic_model: nn.Module,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_val = weight_model(x[idx])
        g_val = critic_model(z[idx])
        gp_val = critic_model(zp[idx])
        moment_residual = torch.mean(d_val * (g_val - gamma_ratio * gp_val)) - (1.0 - gamma_ratio) * torch.mean(g_val)
        objective = (
            torch.mean(d_val * (g_val - gamma_ratio * gp_val))
            - (1.0 - gamma_ratio) * torch.mean(g_val)
            - 0.5 * torch.mean(g_val.pow(2))
            + config.normalization_penalty * (torch.mean(d_val) - 1.0).pow(2)
        )
        normalization_error = torch.mean(d_val) - 1.0
        return objective, moment_residual, normalization_error

    def validation_score(weight_model: nn.Module, critic_model: nn.Module) -> tuple[float, float, float]:
        idx = valid_idx if len(valid_idx_np) > 0 else train_idx
        with torch.no_grad():
            _objective, moment_residual, normalization_error = objective_for_models(weight_model, critic_model, idx)
            score = float(moment_residual.pow(2).item() + config.normalization_penalty * normalization_error.pow(2).item())
            return score, float(moment_residual.item()), float(normalization_error.item())

    best_valid = float("inf")
    best_step = 0
    patience = 0
    best_weight_state = deepcopy((ema_weight_net or weight_net).state_dict())
    best_critic_state = deepcopy((ema_critic_net or critic_net).state_dict())

    for step in range(1, config.max_steps + 1):
        batch_idx = train_idx[torch.randint(len(train_idx), (min(config.batch_size, len(train_idx)),), device=device)]

        critic_opt.zero_grad(set_to_none=True)
        critic_objective, _, _ = objective_for_models(weight_net, critic_net, batch_idx)
        critic_loss = -critic_objective
        critic_loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(critic_net.parameters(), config.grad_clip_norm)
        critic_opt.step()

        weight_opt.zero_grad(set_to_none=True)
        weight_objective, _, _ = objective_for_models(weight_net, critic_net, batch_idx)
        weight_loss = weight_objective
        weight_loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(weight_net.parameters(), config.grad_clip_norm)
        weight_opt.step()

        if ema_weight_net is not None:
            _update_ema_model(ema_weight_net, weight_net, config.ema_decay)
            _update_ema_model(ema_critic_net, critic_net, config.ema_decay)

        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            eval_weight_net = ema_weight_net or weight_net
            eval_critic_net = ema_critic_net or critic_net
            with torch.no_grad():
                train_objective, _, train_norm = objective_for_models(eval_weight_net, eval_critic_net, train_idx)
                valid_score, valid_moment, valid_norm = validation_score(eval_weight_net, eval_critic_net)
            history["train_objective"].append(float(train_objective.item()))
            history["train_normalization_error"].append(float(train_norm.item()))
            history["valid_score"].append(valid_score)
            history["valid_moment_violation"].append(valid_moment)
            history["valid_normalization_error"].append(valid_norm)

            if valid_score + config.min_improvement < best_valid:
                best_valid = valid_score
                best_step = step
                patience = 0
                best_weight_state = deepcopy(eval_weight_net.state_dict())
                best_critic_state = deepcopy(eval_critic_net.state_dict())
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

    weight_net.load_state_dict(best_weight_state)
    critic_net.load_state_dict(best_critic_state)

    with torch.no_grad():
        raw_weights_t = weight_net(x)
        g_full = critic_net(z)
        gp_full = critic_net(zp)
        raw_weights = raw_weights_t.detach().cpu().numpy().astype(np.float64)
        weights, effective_max_weight = stabilize_weights(
            weights=raw_weights,
            min_weight=min_weight,
            max_weight=max_weight,
            clip_quantile=config.clip_quantile,
            uniform_mix=config.uniform_mix,
        )
        moment_violation = float(
            torch.mean(raw_weights_t * (g_full - gamma_ratio * gp_full)).item()
            - (1.0 - gamma_ratio) * torch.mean(g_full).item()
        )

    diagnostics = {
        "solver": "neural_adamw_saddle",
        "gamma_ratio": float(gamma_ratio),
        "step_size": float(config.step_size),
        "max_steps": float(config.max_steps),
        "selected_step": float(best_step),
        "best_valid_score": float(best_valid),
        "normalization_error": float(np.mean(raw_weights) - 1.0),
        "moment_violation": moment_violation,
        "min_weight_raw": float(raw_weights.min()),
        "max_weight_raw": float(raw_weights.max()),
        "effective_max_weight": None if effective_max_weight is None else float(effective_max_weight),
        "train_objective_history": history["train_objective"],
        "valid_score_history": history["valid_score"],
    }
    return RatioEstimationResult(
        alpha=None,
        beta=None,
        weights=weights,
        diagnostics=diagnostics,
        weight_model=weight_net.cpu(),
        critic_model=critic_net.cpu(),
    )
