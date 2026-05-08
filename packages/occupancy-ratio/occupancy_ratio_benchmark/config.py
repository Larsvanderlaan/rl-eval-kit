from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


BenchmarkProfile = Literal["smoke", "medium", "full", "overnight", "dualdice-paper"]
BenchmarkStage = BenchmarkProfile

BOOSTED_ESTIMATOR_PRESETS = (
    "squared",
    "huber",
    "stable",
    "logistic_nuisance",
    "stable_logistic_nuisance",
    "transition_norm",
    "crossfit2",
    "calibrated",
    "crossfit2_calibrated",
    "bellman_moment_calibrated",
    "auto",
)
NEURAL_ESTIMATOR_PRESETS = (
    "squared",
    "huber",
    "stable",
    "logistic_nuisance",
    "stable_logistic_nuisance",
    "google_parity",
    "transition_norm",
    "crossfit2",
    "calibrated",
    "crossfit2_calibrated",
    "bellman_moment_calibrated",
    "auto",
)
DIRECT_ESTIMATORS = {
    "oracle",
    "boosted_tree",
    "neural_network",
    "google_dualdice",
    "google_dualdice_neural",
    "google_tabular_dualdice_gridwalk",
    *(f"boosted_tree_{preset}" for preset in BOOSTED_ESTIMATOR_PRESETS),
    *(f"neural_network_{preset}" for preset in NEURAL_ESTIMATOR_PRESETS),
    "boosted_tree_huber_projection",
    "boosted_tree_huber_projection_damping",
    "boosted_tree_huber_projection_damping_transition_norm",
    "boosted_tree_huber_projection_damping_weighted",
    "neural_network_huber_projection",
    "neural_network_huber_projection_damping",
    "neural_network_huber_projection_damping_transition_norm",
    "neural_network_huber_projection_damping_weighted",
    "neural_network_google_parity",
}


@dataclass(frozen=True)
class OccupancyRatioBenchmarkConfig:
    """User-facing configuration for the occupancy-ratio benchmark suite."""

    stage: BenchmarkStage = "smoke"
    profile: BenchmarkProfile | None = None
    output_root: Path = Path("outputs/occupancy_ratio_benchmark")
    seeds: Sequence[int] = field(default_factory=lambda: (0, 1))
    sample_sizes: Sequence[int] = field(default_factory=lambda: (500,))
    gammas: Sequence[float] = field(default_factory=lambda: (0.5, 0.9))
    linear_gaussian_policy_shifts: Sequence[float] = field(default_factory=lambda: (1.0,))
    settings: Sequence[str] = field(
        default_factory=lambda: ("discrete_chain", "linear_gaussian", "nonlinear_monte_carlo")
    )
    estimators: Sequence[str] = field(
        default_factory=lambda: ("oracle", "boosted_tree", "neural_network", "google_dualdice_neural")
    )
    boosted_estimator_presets: Sequence[str] = field(default_factory=lambda: ("squared", "huber", "stable"))
    neural_estimator_presets: Sequence[str] = field(default_factory=lambda: ("stable",))
    include_google_dual_dice: bool = True
    include_dualdice_gridwalk: bool = False
    gridwalk_alphas: Sequence[float] = field(default_factory=lambda: (0.0, 0.5))
    external_repo_path: Path = Path("/tmp/google-research")
    n_jobs: int = 1

    boosted_num_iterations: int = 30
    boosted_trees_per_iteration: int = 1
    boosted_mcmc_samples: int = 24
    boosted_batch_size: int = 512
    boosted_losses: Sequence[str] = field(default_factory=lambda: ("squared", "huber"))
    boosted_stabilization_presets: Sequence[str] = field(default_factory=tuple)
    boosted_fixed_point_damping: float = 0.5
    boosted_occupancy_ratio_max: float | None = 50.0
    boosted_pseudo_outcome_upper_quantile: float = 0.995
    boosted_sample_weight_mode: str = "uniform"
    boosted_sample_weight_max: float | None = 20.0
    boosted_nuisance_prediction_max: float | None = 50.0
    boosted_density_ratio_loss: str = "lsif"
    boosted_logistic_logit_clip: float | None = 20.0
    boosted_normalize_transition_cache: bool = False
    boosted_crossfit_folds: int = 1
    boosted_moment_calibration: str = "none"
    huber_delta: float | None = None
    huber_delta_scale: float = 1.345
    huber_hessian_floor: float = 1e-3
    neural_num_iterations: int = 60
    neural_gradient_steps_per_iteration: int = 6
    neural_mcmc_samples: int = 24
    neural_batch_size: int = 512
    neural_hidden_dims: Sequence[int] = field(default_factory=lambda: (64, 64))
    neural_activation: str = "silu"
    neural_learning_rate: float = 5e-4
    neural_nuisance_learning_rate: float = 1e-3
    neural_weight_decay: float = 1e-4
    neural_action_steps: int = 800
    neural_transition_steps: int = 1_000
    neural_transition_permutation_samples: int = 4
    neural_density_ratio_loss: str = "lsif"
    neural_logistic_logit_clip: float | None = 20.0
    neural_losses: Sequence[str] = field(default_factory=lambda: ("squared", "huber"))
    neural_stabilization_presets: Sequence[str] = field(default_factory=tuple)
    neural_fixed_point_damping: float = 0.75
    neural_validation_warmup_iterations: int = 1
    neural_occupancy_ratio_max: float | None = 50.0
    neural_pseudo_outcome_upper_quantile: float = 0.995
    neural_sample_weight_mode: str = "uniform"
    neural_sample_weight_max: float | None = 20.0
    neural_nuisance_prediction_max: float | None = 50.0
    neural_grad_clip_norm: float | None = 5.0
    neural_crossfit_folds: int = 1
    neural_moment_calibration: str = "scalar"
    neural_device: str = "cpu"
    google_num_updates: int = 50
    google_batch_size: int = 128
    google_learning_rates: Sequence[float] = field(default_factory=lambda: (1e-4, 3e-4, 1e-3))
    google_weight_decays: Sequence[float] = field(default_factory=lambda: (1e-6, 1e-5, 1e-4))
    google_hidden_dims: Sequence[int] = field(default_factory=lambda: (64, 128))
    google_update_grid: Sequence[int] = field(default_factory=lambda: (250, 1_000, 5_000))
    tune_cv: bool = False
    cv_folds: int = 3
    cv_scoring: str = "composite"
    cv_lambda_norm: float = 0.1
    cv_lambda_tail: float = 0.01
    cv_fixed_point_dampings: Sequence[float] = field(default_factory=lambda: (0.25, 0.5, 0.75))
    cv_occupancy_ratio_max_values: Sequence[float | None] = field(default_factory=lambda: (25.0, 50.0, None))
    cv_nuisance_prediction_max_values: Sequence[float | None] = field(default_factory=lambda: (25.0, 50.0, None))
    cv_moment_calibrations: Sequence[str] = field(default_factory=lambda: ("none", "scalar"))
    mc_truth_samples: int = 8_000
    gym_target_value_rollouts: int = 24
    openml_task_ids: Sequence[int] = field(default_factory=lambda: (31, 37, 54, 1464))
    openml_max_tasks: int | None = 2
    tabular_state_cap: int | None = None
    obp_campaigns: Sequence[str] = field(default_factory=lambda: ("all",))
    minari_dataset_ids: Sequence[str] = field(
        default_factory=lambda: (
            "D4RL/pointmaze/umaze-v2",
            "D4RL/pointmaze/medium-v2",
            "D4RL/minigrid/fourrooms-random-v0",
            "D4RL/minigrid/fourrooms-v0",
        )
    )
    estimator_timeout_sec: float | None = None
    resume: bool = True
    write_plots: bool = True

    def __post_init__(self) -> None:
        profile = self.stage if self.profile is None else self.profile
        if profile not in {"smoke", "medium", "full", "overnight", "dualdice-paper"}:
            raise ValueError("profile must be 'smoke', 'medium', 'full', 'overnight', or 'dualdice-paper'.")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "stage", profile)
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive.")
        for gamma in self.gammas:
            if not (0.0 <= float(gamma) < 1.0):
                raise ValueError("all gammas must be in [0, 1).")
        for shift in self.linear_gaussian_policy_shifts:
            if float(shift) < 0.0:
                raise ValueError("linear_gaussian_policy_shifts must be nonnegative.")
        for alpha in self.gridwalk_alphas:
            if not (0.0 <= float(alpha) <= 1.0):
                raise ValueError("gridwalk_alphas must be in [0, 1].")
        for sample_size in self.sample_sizes:
            if int(sample_size) <= 0:
                raise ValueError("sample_sizes must be positive.")
        for estimator in self.estimators:
            if str(estimator) not in DIRECT_ESTIMATORS:
                raise ValueError(f"estimators entries must be one of {sorted(DIRECT_ESTIMATORS)}.")
        for preset in self.boosted_estimator_presets:
            if str(preset) not in BOOSTED_ESTIMATOR_PRESETS:
                raise ValueError(
                    f"boosted_estimator_presets entries must be one of {list(BOOSTED_ESTIMATOR_PRESETS)}."
                )
        for preset in self.neural_estimator_presets:
            if str(preset) not in NEURAL_ESTIMATOR_PRESETS:
                raise ValueError(
                    f"neural_estimator_presets entries must be one of {list(NEURAL_ESTIMATOR_PRESETS)}."
                )
        for loss in self.boosted_losses:
            if str(loss) not in {"squared", "huber"}:
                raise ValueError("boosted_losses entries must be 'squared' or 'huber'.")
        for loss in self.neural_losses:
            if str(loss) not in {"squared", "huber"}:
                raise ValueError("neural_losses entries must be 'squared' or 'huber'.")
        boosted_allowed_presets = {
            "squared",
            "huber",
            "stable",
            "logistic_nuisance",
            "stable_logistic_nuisance",
            "transition_norm",
            "crossfit2",
            "calibrated",
            "crossfit2_calibrated",
            "bellman_moment_calibrated",
            "auto",
            "huber_projection",
            "huber_projection_damping",
            "huber_projection_damping_transition_norm",
            "huber_projection_damping_weighted",
        }
        neural_allowed_presets = {*boosted_allowed_presets, "google_parity"}
        for preset in self.boosted_stabilization_presets:
            if str(preset) not in boosted_allowed_presets:
                raise ValueError(
                    f"boosted_stabilization_presets entries must be one of {sorted(boosted_allowed_presets)}."
                )
        for preset in self.neural_stabilization_presets:
            if str(preset) not in neural_allowed_presets:
                raise ValueError(f"neural_stabilization_presets entries must be one of {sorted(neural_allowed_presets)}.")
        if not (0.0 < self.boosted_fixed_point_damping <= 1.0):
            raise ValueError("boosted_fixed_point_damping must be in (0, 1].")
        if not (0.0 < self.neural_fixed_point_damping <= 1.0):
            raise ValueError("neural_fixed_point_damping must be in (0, 1].")
        if self.boosted_occupancy_ratio_max is not None and self.boosted_occupancy_ratio_max <= 0.0:
            raise ValueError("boosted_occupancy_ratio_max must be positive when supplied.")
        if self.neural_occupancy_ratio_max is not None and self.neural_occupancy_ratio_max <= 0.0:
            raise ValueError("neural_occupancy_ratio_max must be positive when supplied.")
        if not (0.0 < self.boosted_pseudo_outcome_upper_quantile < 1.0):
            raise ValueError("boosted_pseudo_outcome_upper_quantile must be in (0, 1).")
        if int(self.boosted_crossfit_folds) < 1:
            raise ValueError("boosted_crossfit_folds must be >= 1.")
        if str(self.boosted_moment_calibration) not in {"none", "scalar"}:
            raise ValueError("boosted_moment_calibration must be 'none' or 'scalar'.")
        if str(self.boosted_density_ratio_loss) not in {"lsif", "logistic"}:
            raise ValueError("boosted_density_ratio_loss must be 'lsif' or 'logistic'.")
        if self.boosted_logistic_logit_clip is not None and float(self.boosted_logistic_logit_clip) <= 0.0:
            raise ValueError("boosted_logistic_logit_clip must be positive when supplied.")
        if not (0.0 < self.neural_pseudo_outcome_upper_quantile < 1.0):
            raise ValueError("neural_pseudo_outcome_upper_quantile must be in (0, 1).")
        if self.neural_num_iterations <= 0:
            raise ValueError("neural_num_iterations must be positive.")
        if self.neural_gradient_steps_per_iteration <= 0:
            raise ValueError("neural_gradient_steps_per_iteration must be positive.")
        if self.neural_mcmc_samples <= 0:
            raise ValueError("neural_mcmc_samples must be positive.")
        if self.neural_batch_size <= 0:
            raise ValueError("neural_batch_size must be positive.")
        if not tuple(self.neural_hidden_dims) or any(int(width) <= 0 for width in self.neural_hidden_dims):
            raise ValueError("neural_hidden_dims must contain positive widths.")
        if int(self.neural_validation_warmup_iterations) < 0:
            raise ValueError("neural_validation_warmup_iterations must be nonnegative.")
        if self.neural_learning_rate <= 0.0 or self.neural_nuisance_learning_rate <= 0.0:
            raise ValueError("neural learning rates must be positive.")
        if self.neural_weight_decay < 0.0:
            raise ValueError("neural_weight_decay must be nonnegative.")
        if self.neural_action_steps <= 0 or self.neural_transition_steps <= 0:
            raise ValueError("neural nuisance step counts must be positive.")
        if self.neural_transition_permutation_samples <= 0:
            raise ValueError("neural_transition_permutation_samples must be positive.")
        if str(self.neural_density_ratio_loss) not in {"lsif", "logistic"}:
            raise ValueError("neural_density_ratio_loss must be 'lsif' or 'logistic'.")
        if self.neural_logistic_logit_clip is not None and float(self.neural_logistic_logit_clip) <= 0.0:
            raise ValueError("neural_logistic_logit_clip must be positive when supplied.")
        if self.neural_grad_clip_norm is not None and self.neural_grad_clip_norm <= 0.0:
            raise ValueError("neural_grad_clip_norm must be positive when supplied.")
        if int(self.neural_crossfit_folds) < 1:
            raise ValueError("neural_crossfit_folds must be >= 1.")
        if str(self.neural_moment_calibration) not in {"none", "scalar"}:
            raise ValueError("neural_moment_calibration must be 'none' or 'scalar'.")
        if int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be >= 2.")
        if str(self.cv_scoring) not in {"composite", "loss"}:
            raise ValueError("cv_scoring must be 'composite' or 'loss'.")
        if self.cv_lambda_norm < 0.0 or self.cv_lambda_tail < 0.0:
            raise ValueError("cv_lambda_norm and cv_lambda_tail must be nonnegative.")
        for damping in self.cv_fixed_point_dampings:
            if not (0.0 < float(damping) <= 1.0):
                raise ValueError("cv_fixed_point_dampings must be in (0, 1].")
        for cap in self.cv_occupancy_ratio_max_values:
            if cap is not None and float(cap) <= 0.0:
                raise ValueError("cv_occupancy_ratio_max_values must be positive or None.")
        for cap in self.cv_nuisance_prediction_max_values:
            if cap is not None and float(cap) <= 0.0:
                raise ValueError("cv_nuisance_prediction_max_values must be positive or None.")
        for method in self.cv_moment_calibrations:
            if str(method) not in {"none", "scalar"}:
                raise ValueError("cv_moment_calibrations entries must be 'none' or 'scalar'.")
        for value in self.google_learning_rates:
            if float(value) <= 0.0:
                raise ValueError("google_learning_rates must be positive.")
        for value in self.google_weight_decays:
            if float(value) < 0.0:
                raise ValueError("google_weight_decays must be nonnegative.")
        for value in self.google_hidden_dims:
            if int(value) <= 0:
                raise ValueError("google_hidden_dims must be positive.")
        for value in self.google_update_grid:
            if int(value) <= 0:
                raise ValueError("google_update_grid must be positive.")
        if int(self.gym_target_value_rollouts) <= 0:
            raise ValueError("gym_target_value_rollouts must be positive.")
        for task_id in self.openml_task_ids:
            if int(task_id) <= 0:
                raise ValueError("openml_task_ids must contain positive task ids.")
        if self.openml_max_tasks is not None and int(self.openml_max_tasks) <= 0:
            raise ValueError("openml_max_tasks must be positive when supplied.")
        if self.tabular_state_cap is not None and int(self.tabular_state_cap) <= 1:
            raise ValueError("tabular_state_cap must be greater than 1 when supplied.")
        for campaign in self.obp_campaigns:
            if not str(campaign):
                raise ValueError("obp_campaigns entries must be nonempty.")
        for dataset_id in self.minari_dataset_ids:
            if not str(dataset_id):
                raise ValueError("minari_dataset_ids entries must be nonempty.")
        if self.huber_delta is not None and self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when supplied.")
        if self.huber_delta_scale <= 0.0:
            raise ValueError("huber_delta_scale must be positive.")
        if self.huber_hessian_floor < 0.0:
            raise ValueError("huber_hessian_floor must be nonnegative.")
        if self.estimator_timeout_sec is None:
            if profile != "overnight":
                default_timeout = 120.0 if profile == "smoke" else 600.0
                object.__setattr__(self, "estimator_timeout_sec", default_timeout)
        elif float(self.estimator_timeout_sec) <= 0.0:
            raise ValueError("estimator_timeout_sec must be positive when supplied.")

    @classmethod
    def for_stage(
        cls,
        stage: BenchmarkStage,
        *,
        output_root: str | Path = Path("outputs/occupancy_ratio_benchmark"),
        external_repo_path: str | Path = Path("/tmp/google-research"),
        include_google_dual_dice: bool = True,
    ) -> "OccupancyRatioBenchmarkConfig":
        """Construct the recommended benchmark configuration.

        ``for_stage`` is retained for compatibility; profiles now include
        ``medium`` and ``dualdice-paper`` in addition to smoke/full.
        """
        if stage == "smoke":
            return cls(
                stage="smoke",
                profile="smoke",
                output_root=Path(output_root),
                seeds=(0, 1),
                sample_sizes=(500,),
                gammas=(0.5, 0.9),
                include_google_dual_dice=include_google_dual_dice,
                external_repo_path=Path(external_repo_path),
                boosted_num_iterations=30,
                boosted_mcmc_samples=24,
                boosted_losses=("squared", "huber"),
                neural_num_iterations=20,
                neural_gradient_steps_per_iteration=3,
                neural_mcmc_samples=12,
                neural_action_steps=120,
                neural_transition_steps=160,
                neural_losses=("squared", "huber"),
                neural_stabilization_presets=(),
                google_num_updates=50,
                mc_truth_samples=8_000,
            )
        if stage == "medium":
            return cls(
                stage="medium",
                profile="medium",
                output_root=Path(output_root),
                seeds=tuple(range(5)),
                sample_sizes=(500, 2_000),
                gammas=(0.5, 0.9, 0.95, 0.99),
                linear_gaussian_policy_shifts=(0.25, 0.5, 1.0, 2.0),
                settings=(
                    "discrete_chain",
                    "linear_gaussian",
                    "nonlinear_monte_carlo",
                    "openml_contextual_bandit",
                    "openml_finite_mdp",
                    "obp_logged_bandit",
                    "minari_pointmaze",
                    "minari_minigrid",
                ),
                estimators=("oracle", "boosted_tree", "neural_network", "google_dualdice_neural"),
                boosted_estimator_presets=("squared", "huber", "stable", "transition_norm", "calibrated"),
                neural_estimator_presets=("squared", "huber", "stable", "transition_norm", "calibrated"),
                boosted_stabilization_presets=(),
                neural_stabilization_presets=(),
                include_google_dual_dice=include_google_dual_dice,
                external_repo_path=Path(external_repo_path),
                boosted_num_iterations=80,
                boosted_mcmc_samples=48,
                neural_num_iterations=60,
                neural_gradient_steps_per_iteration=5,
                neural_mcmc_samples=32,
                neural_action_steps=700,
                neural_transition_steps=900,
                google_num_updates=1_000,
                mc_truth_samples=50_000,
            )
        if stage == "full":
            return cls(
                stage="full",
                profile="full",
                output_root=Path(output_root),
                seeds=tuple(range(10)),
                sample_sizes=(500, 2_000, 10_000),
                gammas=(0.5, 0.9, 0.95, 0.99),
                linear_gaussian_policy_shifts=(0.25, 0.5, 1.0, 2.0),
                settings=(
                    "discrete_chain",
                    "linear_gaussian",
                    "nonlinear_monte_carlo",
                    "openml_contextual_bandit",
                    "openml_finite_mdp",
                    "obp_logged_bandit",
                    "minari_pointmaze",
                    "minari_minigrid",
                ),
                estimators=("oracle", "boosted_tree", "neural_network", "google_dualdice_neural"),
                boosted_estimator_presets=BOOSTED_ESTIMATOR_PRESETS,
                neural_estimator_presets=NEURAL_ESTIMATOR_PRESETS,
                boosted_stabilization_presets=(),
                neural_stabilization_presets=(),
                include_google_dual_dice=include_google_dual_dice,
                external_repo_path=Path(external_repo_path),
                boosted_num_iterations=120,
                boosted_mcmc_samples=80,
                boosted_losses=("squared", "huber"),
                neural_num_iterations=100,
                neural_gradient_steps_per_iteration=8,
                neural_mcmc_samples=64,
                neural_action_steps=1_500,
                neural_transition_steps=2_000,
                neural_losses=("squared", "huber"),
                google_num_updates=5_000,
                mc_truth_samples=200_000,
            )
        if stage == "overnight":
            return cls(
                stage="overnight",
                profile="overnight",
                output_root=Path(output_root),
                seeds=tuple(range(3)),
                sample_sizes=(1_000, 2_000),
                gammas=(0.9, 0.99),
                linear_gaussian_policy_shifts=(0.5, 1.0, 2.0),
                settings=(
                    "discrete_chain",
                    "discrete_grid",
                    "linear_gaussian",
                    "nonlinear_monte_carlo",
                    "gym_pendulum",
                    "gym_mountain_car_continuous",
                    "gym_halfcheetah",
                    "gym_hopper",
                    "openml_contextual_bandit",
                    "openml_finite_mdp",
                    "obp_logged_bandit",
                    "minari_pointmaze",
                    "minari_minigrid",
                ),
                estimators=(
                    "oracle",
                    "boosted_tree_stable",
                    "boosted_tree_stable_logistic_nuisance",
                    "boosted_tree_auto",
                    "neural_network_stable",
                    "google_dualdice_neural",
                ),
                boosted_estimator_presets=("stable", "stable_logistic_nuisance", "auto"),
                neural_estimator_presets=("stable",),
                include_google_dual_dice=include_google_dual_dice,
                include_dualdice_gridwalk=False,
                external_repo_path=Path(external_repo_path),
                boosted_num_iterations=60,
                boosted_mcmc_samples=32,
                neural_num_iterations=80,
                neural_gradient_steps_per_iteration=8,
                neural_mcmc_samples=24,
                neural_action_steps=1_000,
                neural_transition_steps=1_400,
                neural_fixed_point_damping=0.75,
                google_num_updates=1_000,
                mc_truth_samples=50_000,
                gym_target_value_rollouts=32,
                estimator_timeout_sec=None,
            )
        if stage == "dualdice-paper":
            return cls(
                stage="dualdice-paper",
                profile="dualdice-paper",
                output_root=Path(output_root),
                seeds=(0, 1, 2),
                sample_sizes=(500,),
                gammas=(0.9, 0.995),
                settings=("discrete_grid",),
                estimators=("google_tabular_dualdice_gridwalk", "boosted_tree", "neural_network"),
                boosted_estimator_presets=("stable",),
                neural_estimator_presets=("stable",),
                boosted_stabilization_presets=(),
                neural_stabilization_presets=(),
                include_google_dual_dice=include_google_dual_dice,
                include_dualdice_gridwalk=True,
                external_repo_path=Path(external_repo_path),
                boosted_num_iterations=40,
                boosted_mcmc_samples=24,
                neural_num_iterations=20,
                neural_mcmc_samples=16,
                neural_gradient_steps_per_iteration=4,
                google_num_updates=1_000,
            )
        raise ValueError("profile must be 'smoke', 'medium', 'full', 'overnight', or 'dualdice-paper'.")

    @classmethod
    def for_profile(
        cls,
        profile: BenchmarkProfile,
        *,
        output_root: str | Path = Path("outputs/occupancy_ratio_benchmark"),
        external_repo_path: str | Path = Path("/tmp/google-research"),
        include_google_dual_dice: bool = True,
    ) -> "OccupancyRatioBenchmarkConfig":
        return cls.for_stage(
            profile,
            output_root=output_root,
            external_repo_path=external_repo_path,
            include_google_dual_dice=include_google_dual_dice,
        )

    def output_dir(self) -> Path:
        return self.output_root / str(self.profile)

    def resolved_estimators(self) -> tuple[str, ...]:
        estimators = tuple(self.estimators)
        if not self.include_google_dual_dice:
            estimators = tuple(name for name in estimators if "dualdice" not in str(name))
        return estimators
