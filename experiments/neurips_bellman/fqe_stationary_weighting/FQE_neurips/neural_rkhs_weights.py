from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from .ratio_estimation import RatioEstimationResult
from .utils import stabilize_weights, train_valid_split


@dataclass
class KernelConfig:
    """Kernel and Nyström feature settings for the RKHS critic."""

    kernel: str = "rbf"
    bandwidth: float | str | None = "median"
    max_anchors: int = 512
    anchor_selection: str = "uniform"
    standardize: bool = True
    jitter: float = 1e-6
    median_heuristic_subsample: int = 2048


@dataclass
class NeuralRKHSWeightsConfig:
    """Configuration for neural weights with a closed-form RKHS critic."""

    hidden_dims_weight: Sequence[int] = (128, 128)
    activation: str = "relu"
    positivity: str = "softplus"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    critic_ridge: float = 1e-4
    normalization_penalty: float = 10.0
    max_steps: int = 2_000
    log_every: int = 100
    valid_fraction: float = 0.1
    early_stopping_patience: int = 15
    min_improvement: float = 1e-5
    grad_clip_norm: float | None = 5.0
    use_ema: bool = True
    ema_decay: float = 0.995
    clip_quantile: float | None = 0.995
    max_weight: float | None = 20.0
    min_weight: float = 1e-8
    uniform_mix: float = 0.02
    target_ess_fraction: float | None = 0.4
    max_uniform_mix: float = 0.5
    device: str = "cpu"
    seed: int = 0
    kernel: KernelConfig = field(default_factory=KernelConfig)


class MLP(nn.Module):
    """Simple MLP for the positive weight model."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: str = "relu") -> None:
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        Act = act_map.get(activation.lower(), nn.ReLU)
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(Act())
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PositiveWeightNet(nn.Module):
    """Positive weight model for stationary / discounted ratio estimation."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: str, positivity: str) -> None:
        super().__init__()
        self.backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)
        positivity = positivity.lower()
        if positivity == "softplus":
            self.map = nn.Softplus()
        elif positivity == "exp":
            self.map = torch.exp
        else:
            raise ValueError(f"Unsupported positivity transform '{positivity}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(self.backbone(x)) + 1e-8


def _update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay)
            ema_param.data.add_((1.0 - decay) * param.data)


def _standardize_features(
    train_x: np.ndarray,
    full_x: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not enabled:
        zeros = np.zeros(train_x.shape[1], dtype=np.float64)
        ones = np.ones(train_x.shape[1], dtype=np.float64)
        return train_x, full_x, {"mean": zeros, "scale": ones}
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (train_x - mean) / scale, (full_x - mean) / scale, {"mean": mean, "scale": scale}


def _pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = torch.sum(x * x, dim=1, keepdim=True)
    y_norm = torch.sum(y * y, dim=1).unsqueeze(0)
    return torch.clamp(x_norm + y_norm - 2.0 * (x @ y.T), min=0.0)


def _median_bandwidth(x: np.ndarray, config: KernelConfig, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if x.shape[0] > config.median_heuristic_subsample:
        idx = rng.choice(x.shape[0], size=config.median_heuristic_subsample, replace=False)
        x = x[idx]
    if x.shape[0] < 2:
        return 1.0
    diffs = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diffs**2, axis=-1))
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    return float(np.median(tri))


def _kernel_matrix(x: torch.Tensor, anchors: torch.Tensor, kernel: str, bandwidth: float) -> torch.Tensor:
    kernel = kernel.lower()
    if kernel == "linear":
        return x @ anchors.T
    sqdist = _pairwise_sqdist(x, anchors)
    bw = max(float(bandwidth), 1e-8)
    if kernel == "rbf":
        return torch.exp(-sqdist / (2.0 * bw * bw))
    if kernel == "laplace":
        return torch.exp(-torch.sqrt(sqdist + 1e-12) / bw)
    if kernel == "matern32":
        r = torch.sqrt(3.0 * sqdist + 1e-12) / bw
        return (1.0 + r) * torch.exp(-r)
    raise ValueError(f"Unsupported kernel '{kernel}'.")


def _select_anchors(x: np.ndarray, max_anchors: int, seed: int, method: str) -> np.ndarray:
    if x.shape[0] <= max_anchors:
        return x
    if method != "uniform":
        raise ValueError(f"Unsupported anchor selection method '{method}'.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_anchors, replace=False)
    return x[idx]


def _build_nystrom_features(
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
    train_idx: np.ndarray,
    config: KernelConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    train_x_raw = np.asarray(critic_features[train_idx], dtype=np.float64)
    full_x_raw = np.asarray(critic_features, dtype=np.float64)
    full_xp_raw = np.asarray(next_critic_features, dtype=np.float64)

    train_x_std, full_x_std, stats = _standardize_features(train_x_raw, full_x_raw, config.standardize)
    _, full_xp_std, _ = _standardize_features(train_x_raw, full_xp_raw, config.standardize)

    anchors_np = _select_anchors(train_x_std, config.max_anchors, seed=seed, method=config.anchor_selection)
    bandwidth = config.bandwidth
    if bandwidth is None or bandwidth == "median":
        bandwidth = _median_bandwidth(train_x_std, config=config, seed=seed)
    if config.kernel == "linear":
        bandwidth = 1.0

    x_t = torch.as_tensor(full_x_std, dtype=torch.float64, device=device)
    xp_t = torch.as_tensor(full_xp_std, dtype=torch.float64, device=device)
    anchors_t = torch.as_tensor(anchors_np, dtype=torch.float64, device=device)

    kuu = _kernel_matrix(anchors_t, anchors_t, config.kernel, float(bandwidth))
    kuu = kuu + config.jitter * torch.eye(kuu.shape[0], dtype=kuu.dtype, device=device)
    chol = torch.linalg.cholesky(kuu)

    kxu = _kernel_matrix(x_t, anchors_t, config.kernel, float(bandwidth))
    kxpu = _kernel_matrix(xp_t, anchors_t, config.kernel, float(bandwidth))
    phi = torch.linalg.solve_triangular(chol, kxu.T, upper=False).T
    phi_next = torch.linalg.solve_triangular(chol, kxpu.T, upper=False).T

    meta = {
        "anchors": anchors_np,
        "bandwidth": float(bandwidth),
        "kernel": config.kernel,
        "standardizer": stats,
        "n_anchors": int(anchors_np.shape[0]),
    }
    return phi, phi_next, meta


def estimate_ratio_neural_rkhs(
    weight_features: np.ndarray,
    critic_features: np.ndarray,
    next_critic_features: np.ndarray,
    gamma_ratio: float = 1.0,
    config: NeuralRKHSWeightsConfig | None = None,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
) -> RatioEstimationResult:
    """
    Estimate stationary / discounted ratios with a neural weight model and an RKHS critic.

    The critic is represented with Nyström kernel features and solved in closed form
    via kernel ridge regression inside the minimax objective. The weight model is
    trained by gradient descent on the reduced objective obtained after plugging in
    the optimal critic coefficients.
    """

    if config is None:
        config = NeuralRKHSWeightsConfig()
    else:
        config = deepcopy(config)

    if min_weight is not None:
        config.min_weight = min_weight
    if max_weight is not None:
        config.max_weight = max_weight

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    x_weight = torch.as_tensor(np.asarray(weight_features, dtype=np.float32), device=device)
    n = x_weight.shape[0]
    train_idx_np, valid_idx_np = train_valid_split(n, config.valid_fraction, rng)
    if train_idx_np.size == 0:
        raise ValueError("Training split is empty.")

    phi, phi_next, kernel_meta = _build_nystrom_features(
        critic_features=np.asarray(critic_features, dtype=np.float64),
        next_critic_features=np.asarray(next_critic_features, dtype=np.float64),
        train_idx=train_idx_np,
        config=config.kernel,
        seed=config.seed,
        device=device,
    )

    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_idx_np, dtype=torch.long, device=device)
    input_dim = int(x_weight.shape[1])
    feature_dim = int(phi.shape[1])

    weight_net = PositiveWeightNet(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims_weight,
        activation=config.activation,
        positivity=config.positivity,
    ).to(device)
    ema_weight_net = deepcopy(weight_net).to(device) if config.use_ema else None
    if ema_weight_net is not None:
        ema_weight_net.eval()
        for param in ema_weight_net.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(weight_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    eye = torch.eye(feature_dim, dtype=torch.float64, device=device)
    history: dict[str, list[float]] = {
        "train_objective": [],
        "valid_score": [],
        "valid_normalization_error": [],
    }

    def reduced_objective(model: nn.Module, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_val = model(x_weight[idx]).to(torch.float64)
        phi_idx = phi[idx]
        phi_next_idx = phi_next[idx]
        n_idx = idx.numel()
        delta_phi = phi_idx - gamma_ratio * phi_next_idx
        source = (delta_phi.T @ d_val) / n_idx - (1.0 - gamma_ratio) * torch.mean(phi_idx, dim=0)
        gram = (phi_idx.T @ phi_idx) / n_idx + config.critic_ridge * eye
        beta = torch.linalg.solve(gram, source)
        sup_value = 0.5 * torch.dot(source, beta)
        normalization_error = torch.mean(d_val) - 1.0
        loss = sup_value + config.normalization_penalty * normalization_error.pow(2)
        return loss, beta, normalization_error, sup_value

    best_valid = float("inf")
    best_step = 0
    patience = 0
    best_state = deepcopy((ema_weight_net or weight_net).state_dict())

    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _beta, _norm_error, _sup_value = reduced_objective(weight_net, train_idx)
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(weight_net.parameters(), config.grad_clip_norm)
        optimizer.step()

        if ema_weight_net is not None:
            _update_ema_model(ema_weight_net, weight_net, config.ema_decay)

        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            eval_model = ema_weight_net or weight_net
            with torch.no_grad():
                train_loss, _train_beta, train_norm, _train_sup = reduced_objective(eval_model, train_idx)
                score_idx = valid_idx if len(valid_idx_np) > 0 else train_idx
                valid_loss, _valid_beta, valid_norm, _valid_sup = reduced_objective(eval_model, score_idx)
            valid_score = float(valid_loss.item())
            history["train_objective"].append(float(train_loss.item()))
            history["valid_score"].append(valid_score)
            history["valid_normalization_error"].append(float(valid_norm.item()))

            if valid_score + config.min_improvement < best_valid:
                best_valid = valid_score
                best_step = step
                patience = 0
                best_state = deepcopy(eval_model.state_dict())
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

    weight_net.load_state_dict(best_state)
    weight_net.eval()

    with torch.no_grad():
        raw_weights_t = weight_net(x_weight).to(torch.float64)
        full_loss, beta_full, norm_error, sup_full = reduced_objective(weight_net, torch.arange(n, device=device))
        raw_weights = raw_weights_t.cpu().numpy().astype(np.float64)
        weights, stabilization_meta = stabilize_weights(
            weights=raw_weights,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            clip_quantile=config.clip_quantile,
            uniform_mix=config.uniform_mix,
            target_ess_fraction=config.target_ess_fraction,
            max_uniform_mix=config.max_uniform_mix,
            return_metadata=True,
        )
        delta_phi_full = phi - gamma_ratio * phi_next
        moment_violation = float(
            torch.norm((delta_phi_full.T @ raw_weights_t) / n - (1.0 - gamma_ratio) * torch.mean(phi, dim=0)).item()
        )

    diagnostics = {
        "solver": "neural_rkhs_closed_form_critic",
        "gamma_ratio": float(gamma_ratio),
        "kernel": kernel_meta["kernel"],
        "bandwidth": kernel_meta["bandwidth"],
        "n_anchors": kernel_meta["n_anchors"],
        "selected_step": float(best_step),
        "best_valid_score": float(best_valid),
        "normalization_error": float(np.mean(raw_weights) - 1.0),
        "moment_violation_l2": moment_violation,
        "min_weight_raw": float(raw_weights.min()),
        "max_weight_raw": float(raw_weights.max()),
        "effective_max_weight": stabilization_meta["effective_max_weight"],
        "target_ess_fraction": stabilization_meta["target_ess_fraction"],
        "chosen_uniform_mix": float(stabilization_meta["chosen_uniform_mix"]),
        "ess_fraction_before_mix": float(stabilization_meta["ess_fraction_before_mix"]),
        "ess_fraction_after_mix": float(stabilization_meta["ess_fraction_after_mix"]),
        "train_objective_history": history["train_objective"],
        "valid_score_history": history["valid_score"],
        "full_reduced_objective": float(full_loss.item()),
        "critic_coef_l2": float(torch.norm(beta_full).item()),
        "critic_sup_value": float(sup_full.item()),
        "normalization_error_full": float(norm_error.item()),
    }
    return RatioEstimationResult(
        alpha=None,
        beta=beta_full.cpu().numpy().astype(np.float64),
        weights=weights,
        diagnostics=diagnostics,
        weight_model=weight_net.cpu(),
        critic_model=None,
    )
