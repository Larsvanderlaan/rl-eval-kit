from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np


Array = np.ndarray
BenchmarkStage = Literal["smoke", "core", "full"]
PreflightStatus = Literal["ok", "missing_dependency", "missing_data", "unsupported_setting"]


@dataclass(frozen=True)
class BenchmarkConfig:
    """User-facing configuration for the FQE benchmark suite."""

    stage: BenchmarkStage = "smoke"
    output_root: Path = Path("outputs/fqe_benchmark")
    seeds: tuple[int, ...] = (0,)
    datasets: tuple[str, ...] = ("tabular_chain", "linear_gaussian")
    estimators: tuple[str, ...] = (
        "ours_boosted_fqe",
        "ours_boosted_fqe_tuned",
        "ours_neural_fqe",
        "ours_neural_fqe_tuned",
        "legacy_boosted_fqe",
        "legacy_neural_fqe",
        "controlled_linear_fqe",
        "d3rlpy_fqe",
        "google_policy_eval_fqe_l2",
        "deep_ope_reference_fqe_l2",
    )
    sample_sizes: tuple[int, ...] = (120,)
    gammas: tuple[float, ...] = (0.0, 0.7)
    policy_shifts: tuple[float, ...] = (0.5,)
    n_eval: int = 256
    n_initial_eval: int = 128
    gym_target_value_rollouts: int = 8
    stationary_gamma_ratio: float = 0.99
    output_plots: bool = True
    fail_fast: bool = False
    tune_cv: bool = False
    automl_tuning: Literal["off", "fast", "balanced"] = "off"
    staged_cv: bool = False
    staged_cv_iterations: tuple[int, ...] | None = None
    include_hopper: bool = False
    hopper_artifact_dir: Path = Path("hopper_fqe_benchmark/artifacts")
    google_research_path: Path = Path("/tmp/google-research")
    google_dualdice_num_updates: int = 1000
    google_dualdice_batch_size: int = 128
    dice_rl_repo_path: Path = Path("/tmp/dice_rl")
    dice_rl_num_steps: int = 1000
    dice_rl_batch_size: int = 128
    dice_rl_learning_rate: float = 1e-4
    dice_rl_hidden_dims: tuple[int, ...] = (64, 64)
    scope_rl_repo_path: Path | None = None
    scope_rl_n_steps: int = 10000
    scope_rl_n_steps_per_epoch: int = 10000
    scope_rl_batch_size: int = 128
    scope_rl_learning_rate: float = 1e-4
    scope_rl_hidden_dim: int = 100
    scope_rl_bandwidth: float = 1.0

    boosted_num_iterations: int = 16
    boosted_tune_num_iterations: int = 12
    neural_num_iterations: int = 8
    neural_gradient_steps_per_iteration: int = 10
    neural_tune_num_iterations: int = 5
    neural_tune_gradient_steps_per_iteration: int = 6

    def __post_init__(self) -> None:
        if self.stage not in {"smoke", "core", "full"}:
            raise ValueError("stage must be 'smoke', 'core', or 'full'.")
        if not self.seeds:
            raise ValueError("seeds must be nonempty.")
        if not self.datasets:
            raise ValueError("datasets must be nonempty.")
        if not self.estimators:
            raise ValueError("estimators must be nonempty.")
        for sample_size in self.sample_sizes:
            if int(sample_size) <= 0:
                raise ValueError("sample_sizes must be positive.")
        for gamma in self.gammas:
            if not (0.0 <= float(gamma) < 1.0):
                raise ValueError("gammas must be in [0, 1).")
        if self.n_eval <= 0 or self.n_initial_eval <= 0:
            raise ValueError("evaluation sizes must be positive.")
        if int(self.gym_target_value_rollouts) <= 0:
            raise ValueError("gym_target_value_rollouts must be positive.")
        if not (0.0 <= float(self.stationary_gamma_ratio) < 1.0):
            raise ValueError("stationary_gamma_ratio must be in [0, 1).")
        if self.automl_tuning not in {"off", "fast", "balanced"}:
            raise ValueError("automl_tuning must be 'off', 'fast', or 'balanced'.")
        if self.staged_cv_iterations is not None:
            if not self.staged_cv_iterations:
                raise ValueError("staged_cv_iterations must be nonempty when supplied.")
            if any(int(value) <= 0 for value in self.staged_cv_iterations):
                raise ValueError("staged_cv_iterations entries must be positive.")
        if int(self.google_dualdice_num_updates) <= 0:
            raise ValueError("google_dualdice_num_updates must be positive.")
        if int(self.google_dualdice_batch_size) <= 0:
            raise ValueError("google_dualdice_batch_size must be positive.")
        if int(self.dice_rl_num_steps) <= 0:
            raise ValueError("dice_rl_num_steps must be positive.")
        if int(self.dice_rl_batch_size) <= 0:
            raise ValueError("dice_rl_batch_size must be positive.")
        if float(self.dice_rl_learning_rate) <= 0.0:
            raise ValueError("dice_rl_learning_rate must be positive.")
        if not self.dice_rl_hidden_dims or any(int(width) <= 0 for width in self.dice_rl_hidden_dims):
            raise ValueError("dice_rl_hidden_dims must contain positive widths.")
        if int(self.scope_rl_n_steps) <= 0:
            raise ValueError("scope_rl_n_steps must be positive.")
        if int(self.scope_rl_n_steps_per_epoch) <= 0:
            raise ValueError("scope_rl_n_steps_per_epoch must be positive.")
        if int(self.scope_rl_batch_size) <= 0:
            raise ValueError("scope_rl_batch_size must be positive.")
        if float(self.scope_rl_learning_rate) <= 0.0:
            raise ValueError("scope_rl_learning_rate must be positive.")
        if int(self.scope_rl_hidden_dim) <= 0:
            raise ValueError("scope_rl_hidden_dim must be positive.")
        if float(self.scope_rl_bandwidth) <= 0.0:
            raise ValueError("scope_rl_bandwidth must be positive.")

    @classmethod
    def for_stage(
        cls,
        stage: BenchmarkStage,
        *,
        output_root: str | Path = Path("outputs/fqe_benchmark"),
    ) -> "BenchmarkConfig":
        if stage == "smoke":
            return cls(
                stage="smoke",
                output_root=Path(output_root),
                seeds=(0,),
                datasets=("tabular_chain", "linear_gaussian"),
                sample_sizes=(120,),
                gammas=(0.0, 0.7),
                policy_shifts=(0.5,),
                n_eval=256,
                n_initial_eval=128,
                boosted_num_iterations=16,
                boosted_tune_num_iterations=10,
                neural_num_iterations=6,
                neural_gradient_steps_per_iteration=8,
                neural_tune_num_iterations=4,
                neural_tune_gradient_steps_per_iteration=5,
                gym_target_value_rollouts=4,
                google_dualdice_num_updates=50,
                google_dualdice_batch_size=64,
                dice_rl_num_steps=50,
                dice_rl_batch_size=64,
                scope_rl_n_steps=50,
                scope_rl_n_steps_per_epoch=50,
                scope_rl_batch_size=32,
            )
        if stage == "core":
            return cls(
                stage="core",
                output_root=Path(output_root),
                seeds=(0, 1, 2),
                datasets=("tabular_chain", "tabular_grid", "linear_gaussian", "hopper_medium"),
                sample_sizes=(500, 1500),
                gammas=(0.5, 0.9),
                policy_shifts=(0.0, 0.7, 1.2),
                n_eval=2000,
                n_initial_eval=1000,
                boosted_num_iterations=80,
                boosted_tune_num_iterations=60,
                neural_num_iterations=30,
                neural_gradient_steps_per_iteration=20,
                neural_tune_num_iterations=20,
                neural_tune_gradient_steps_per_iteration=15,
                gym_target_value_rollouts=16,
                google_dualdice_num_updates=1000,
                google_dualdice_batch_size=128,
                dice_rl_num_steps=1000,
                dice_rl_batch_size=128,
                scope_rl_n_steps=1000,
                scope_rl_n_steps_per_epoch=1000,
                scope_rl_batch_size=128,
            )
        if stage == "full":
            return cls(
                stage="full",
                output_root=Path(output_root),
                seeds=tuple(range(10)),
                datasets=("tabular_chain", "tabular_grid", "linear_gaussian"),
                sample_sizes=(500, 2000, 8000),
                gammas=(0.5, 0.9, 0.95),
                policy_shifts=(0.0, 0.7, 1.2, 1.6),
                n_eval=10000,
                n_initial_eval=5000,
                include_hopper=True,
                boosted_num_iterations=200,
                boosted_tune_num_iterations=120,
                neural_num_iterations=80,
                neural_gradient_steps_per_iteration=30,
                neural_tune_num_iterations=40,
                neural_tune_gradient_steps_per_iteration=20,
                gym_target_value_rollouts=32,
                google_dualdice_num_updates=5000,
                google_dualdice_batch_size=256,
                dice_rl_num_steps=5000,
                dice_rl_batch_size=256,
                scope_rl_n_steps=5000,
                scope_rl_n_steps_per_epoch=5000,
                scope_rl_batch_size=256,
            )
        raise ValueError("stage must be 'smoke', 'core', or 'full'.")

    def output_dir(self) -> Path:
        return self.output_root / self.stage


@dataclass
class BenchmarkDataset:
    """One benchmark dataset plus evaluation truth when available."""

    name: str
    domain: str
    states: Array
    actions: Array
    next_states: Array
    next_actions: Array
    rewards: Array
    terminals: Array
    gamma: float
    seed: int
    initial_states: Array
    initial_actions: Array
    target_eval_states: Array
    target_eval_actions: Array
    behavior_eval_states: Array
    behavior_eval_actions: Array
    true_q_fn: Callable[[Array, Array], Array] | None = None
    true_policy_value: float | None = None
    sample_weight: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    target_actions: Array | None = None
    episode_ids: Array | None = None
    timesteps: Array | None = None
    step_per_trajectory: int | None = None
    behavior_action_pscore: Array | None = None

    def __post_init__(self) -> None:
        if self.target_actions is None:
            self.target_actions = np.asarray(self.actions)
        n = int(np.asarray(self.rewards).reshape(-1).shape[0])
        for name in ("states", "actions", "target_actions", "next_states", "next_actions", "terminals"):
            if np.asarray(getattr(self, name)).shape[0] != n:
                raise ValueError(f"{name} must have {n} rows.")
        for name in ("episode_ids", "timesteps", "behavior_action_pscore"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape[0] != n:
                raise ValueError(f"{name} must have {n} rows.")
        if self.step_per_trajectory is not None and int(self.step_per_trajectory) <= 0:
            raise ValueError("step_per_trajectory must be positive when supplied.")

    @property
    def n(self) -> int:
        return int(np.asarray(self.rewards).reshape(-1).shape[0])

    @property
    def state_dim(self) -> int:
        return int(np.asarray(self.states).reshape(self.n, -1).shape[1])

    @property
    def action_dim(self) -> int:
        return int(np.asarray(self.actions).reshape(self.n, -1).shape[1])


@dataclass(frozen=True)
class EstimatorPreflight:
    status: PreflightStatus
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.status == "ok"


@dataclass
class FittedEstimator:
    estimator: str
    model: Any
    runtime_sec: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    tuning_runtime_sec: float = 0.0
    tuning_rows: list[dict[str, Any]] = field(default_factory=list)

    def predict_q(self, states: Array, actions: Array) -> Array:
        if hasattr(self.model, "predict_q"):
            return np.asarray(self.model.predict_q(states, actions), dtype=np.float64).reshape(-1)
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(states, actions), dtype=np.float64).reshape(-1)
        raise TypeError(f"{self.estimator} model does not expose predict_q or predict.")

    def estimate_policy_value(self, initial_states: Array, initial_actions: Array) -> float:
        if hasattr(self.model, "estimate_policy_value"):
            return float(self.model.estimate_policy_value(initial_states, initial_actions))
        return float(np.mean(self.predict_q(initial_states, initial_actions)))


class EstimatorAdapter(Protocol):
    name: str

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        ...

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        ...


@dataclass
class BenchmarkRunResult:
    output_dir: Path
    results_path: Path
    summary_path: Path
    diagnostics_path: Path
    manifest_path: Path
    tuning_results_path: Path
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    tuning_rows: list[dict[str, Any]]
