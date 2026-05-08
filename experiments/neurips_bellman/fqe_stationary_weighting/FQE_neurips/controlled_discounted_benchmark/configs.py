from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


FEATURE_REGIMES = (
    "well_specified",
    "misspecified_affine",
    "misspecified_diag_quad",
    "flexible_rbf",
)


@dataclass(frozen=True)
class FQESolverConfig:
    ridge: float = 1e-3
    n_outer_iters: int = 35
    target_update_tau: float = 1.0
    valid_fraction: float = 0.10


@dataclass(frozen=True)
class RatioFeatureConfig:
    n_rbf_centers: int = 24
    bandwidth: float | str = "median"
    bandwidth_scale: float = 1.0
    standardize_features: bool = False


@dataclass(frozen=True)
class WeightEstimatorConfig:
    ridge_primal: float = 1e-4
    ridge_dual: float = 1e-4
    normalization_penalty: float = 10.0
    min_weight: float = 1e-8
    clipped_clip_quantile: float = 0.99
    clipped_max_weight: float = 25.0
    clipped_target_ess_fraction: float = 0.25
    clipped_uniform_mix: float = 0.0
    clipped_max_uniform_mix: float = 0.50
    severe_q99_threshold: float = 20.0


@dataclass(frozen=True)
class EvaluationConfig:
    q_eval_draws: int = 8_000
    state_eval_draws: int = 6_000
    initial_eval_draws: int = 6_000


@dataclass(frozen=True)
class NeuralRatioConfig:
    hidden_dims: Sequence[int] = (64, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    critic_ridge: float = 1e-4
    normalization_penalty: float = 10.0
    max_steps: int = 250
    valid_fraction: float = 0.10
    early_stopping_patience: int = 12
    min_improvement: float = 1e-5
    grad_clip_norm: float = 5.0
    device: str = "cpu"
    min_weight: float = 1e-8
    clip_quantile: float = 0.99
    max_weight: float = 25.0
    target_ess_fraction: float = 0.40
    max_uniform_mix: float = 0.65


@dataclass(frozen=True)
class NeuralFQEConfig:
    hidden_dims: Sequence[int] = (64, 64)
    n_outer_iters: int = 18
    epochs_per_iter: int = 8
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    target_update_tau: float = 0.10
    valid_fraction: float = 0.10
    early_stopping_patience: int = 4
    min_improvement: float = 1e-5
    grad_clip_norm: float = 5.0
    action_quadrature_order: int = 7
    state_quadrature_order: int = 5
    device: str = "cpu"


def gamma_smoke_stage_grid() -> dict[str, object]:
    return {
        "value_gammas": [0.95],
        "ratio_gammas": [0.95, 1.0],
        "shifts": [0.0, 1.1, 1.35],
        "sample_sizes": [1_000],
        "process_noise_sds": [0.05],
        "behavior_action_sds": [0.10],
        "feature_regimes": ["misspecified_affine", "flexible_rbf"],
        "seeds": [0, 1],
        "neural_seeds": [0],
    }


def gamma_design_stage_grid() -> dict[str, object]:
    return {
        "value_gammas": [0.95],
        "ratio_gammas": [0.95, 0.99, 1.0],
        "shifts": [0.0, 1.0, 1.1, 1.35],
        "sample_sizes": [1_000, 4_000],
        "process_noise_sds": [0.05],
        "behavior_action_sds": [0.10],
        "feature_regimes": ["misspecified_affine", "flexible_rbf"],
        "seeds": [0, 1, 2, 3, 4],
        "neural_seeds": [0, 1],
    }


def gamma_final_stage_grid() -> dict[str, object]:
    return {
        "value_gammas": [0.95],
        "ratio_gammas": [0.95, 0.99, 1.0],
        "shifts": [0.0, 1.1, 1.35],
        "sample_sizes": [4_000],
        "process_noise_sds": [0.05],
        "behavior_action_sds": [0.10],
        "feature_regimes": ["misspecified_affine", "flexible_rbf"],
        "seeds": list(range(100)),
        "neural_seeds": list(range(10)),
    }


def gamma_paper_stage_grid() -> dict[str, object]:
    return {
        "value_gammas": [0.95],
        "ratio_gammas": [0.95, 0.99, 1.0],
        "shifts": [0.0, 1.1, 1.35],
        "sample_sizes": [4_000],
        "process_noise_sds": [0.05],
        "behavior_action_sds": [0.10],
        "feature_regimes": ["misspecified_affine", "flexible_rbf"],
        "seeds": list(range(20)),
        "neural_seeds": [],
    }


def stationary_shift_paper_stage_grid() -> dict[str, object]:
    return {
        "value_gammas": [0.95],
        "ratio_gammas": [1.0],
        "shifts": [
            0.0,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
            1.1,
            1.2,
            1.25,
            1.35,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            1.95,
            2.0,
            2.025,
            2.05,
            2.055,
            2.06,
        ],
        "sample_sizes": [4_000],
        "process_noise_sds": [0.12],
        "behavior_action_sds": [0.15],
        "feature_regimes": ["misspecified_affine", "flexible_rbf"],
        "seeds": list(range(100)),
        "neural_seeds": [],
        "behavior_shift_direction_scale": 1.0,
        "behavior_shift_direction": [0.495, 0.11],
    }


def smoke_stage_grid() -> dict[str, object]:
    return {
        "shifts": [0.0, 1.1, 1.35],
        "sample_sizes": [4_000],
        "gammas": [0.95],
        "process_noise_sds": [0.05],
        "behavior_action_sds": [0.10],
        "feature_regimes": ["misspecified_affine"],
        "seeds": [0, 1],
    }


def design_search_grid() -> dict[str, object]:
    return {
        "shifts": [0.0, 0.5, 1.0, 1.1, 1.35],
        "sample_sizes": [1_000, 4_000],
        "gammas": [0.95, 0.99],
        "process_noise_sds": [0.05, 0.12],
        "behavior_action_sds": [0.10, 0.20],
        "feature_regimes": ["well_specified", "misspecified_affine"],
        "seeds": [0, 1, 2],
    }


def final_stage_defaults() -> dict[str, object]:
    return {
        "sample_sizes": [1_000, 4_000, 12_000],
        "feature_regimes": list(FEATURE_REGIMES),
        "seeds": list(range(20)),
    }
