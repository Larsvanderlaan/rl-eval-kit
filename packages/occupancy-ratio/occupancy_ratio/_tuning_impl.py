from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    DiscountedOccupancyRatioModel,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
)
from occupancy_ratio.google_dualdice import (
    GoogleDualDICEConfig,
    fit_google_dualdice_occupancy_ratio,
)
from occupancy_ratio._tuning_first_stage import (
    fit_first_stage_for_family,
    public_first_stage_telemetry,
)
from occupancy_ratio._tuning_staged import (
    StagedCVCandidateRow,
    StagedCVFoldRow,
    StagedCVResult,
    monotone_one_se_prune,
    run_staged_bootstrap_cv,
)

if TYPE_CHECKING:
    from occupancy_ratio.fit_occupancy_ratio_neural import NeuralDiscountedOccupancyRatioModel


Array = np.ndarray
MomentBlock = Tuple[Array, Array, Array]
MomentBlockCacheKey = Tuple[Any, ...]
MomentBlockCache = Dict[MomentBlockCacheKey, Dict[str, MomentBlock]]

__all__ = [
    "CandidateResult",
    "FoldResult",
    "OccupancySearchSpace",
    "OccupancyTargetValidationCandidateResult",
    "OccupancyTargetValidationResult",
    "OccupancyTuningConfig",
    "OccupancyTuningResult",
    "StagedCVCandidateRow",
    "StagedCVFoldRow",
    "StagedCVResult",
    "tune_occupancy_ratio",
    "tune_occupancy_ratio_auto",
    "tune_occupancy_ratio_with_target_validation",
]


def _default_neural_occupancy_config() -> Any:
    from occupancy_ratio.fit_occupancy_ratio_neural import NeuralOccupancyRegressionConfig

    return NeuralOccupancyRegressionConfig()


def _default_neural_action_config() -> Any:
    from occupancy_ratio.fit_occupancy_ratio_neural import NeuralActionRatioConfig

    return NeuralActionRatioConfig()


def _default_neural_source_config() -> Any:
    from occupancy_ratio.fit_occupancy_ratio_neural import NeuralSourceStateRatioConfig

    return NeuralSourceStateRatioConfig()


def _default_neural_transition_config() -> Any:
    from occupancy_ratio.fit_occupancy_ratio_neural import NeuralTransitionRatioConfig

    return NeuralTransitionRatioConfig()


def fit_discounted_occupancy_ratio_neural(**kwargs: Any) -> Any:
    from occupancy_ratio.fit_occupancy_ratio_neural import fit_discounted_occupancy_ratio_neural as _fit

    return _fit(**kwargs)


@dataclass(frozen=True)
class OccupancyTuningConfig:
    """Product-level CV/AutoML controls for occupancy-ratio tuning.

    The default selector is held-out discounted occupancy moment balance, with
    generated regression loss used as a convergence diagnostic rather than the
    primary target. The stable fallback follows a one-standard-error style rule:
    prefer the stable baseline only when its moment balance is statistically and
    practically tied with the selected candidate and the selected candidate has
    materially worse ESS/CV diagnostics.

    Google DualDICE is intentionally excluded from the default candidate set.
    Set ``include_google_dualdice=True`` to add it as an optional neural-family
    candidate when joint initial state-action rows are available.
    """

    families: Sequence[str] = ("neural",)
    cv_folds: int = 3
    seed: int = 123
    budget: str = "balanced"
    max_candidates: int = 16
    promotion_candidates: int = 4
    refit: bool = True
    screen_fraction: float = 0.4
    score_moment_balance_weight: float = 0.65
    score_validation_weight: float = 0.10
    score_reward_stability_weight: float = 0.0
    score_weight_quality_weight: float = 0.20
    score_runtime_weight: float = 0.05
    score_method: Literal["legacy_rank", "bellman_gmm", "validation_loss"] = "legacy_rank"
    gmm_objective: Literal["ratio", "ope"] = "ratio"
    gmm_cov_ridge: float = 0.10
    gmm_complexity_weight: float = 0.05
    gmm_ope_broad_weight: float = 0.10
    gmm_refit_top_candidates: int = 1
    gmm_refit_fraction: float = 1.0
    gmm_use_safety_constraints: bool = True
    moment_rff_features: int = 16
    moment_geometry_features: int = 8
    moment_value_iterations: int = 30
    moment_value_patience: int = 5
    moment_max_group_weight: float = 0.25
    moment_extra_blocks: Sequence[str] = ("multiscale_rff",)
    moment_multiscale_rff_scales: Sequence[float] = (0.5, 2.0)
    moment_strata_quantiles: Sequence[float] = (0.25, 0.50, 0.75)
    stable_fallback: bool = True
    fallback_score_tolerance: float = 0.08
    fallback_quality_margin: float = 0.02
    fallback_runtime_ratio: float = 2.5
    fallback_moment_balance_tolerance: float = 0.05
    fallback_ess_ratio: float = 0.50
    fallback_ess_abs_drop: float = 0.05
    fallback_cv_ratio: float = 1.40
    fallback_cv_abs_increase: float = 0.50
    include_google_dualdice: bool = False
    stagewise: bool = True
    first_stage_cv_folds: Optional[int] = None
    initial_ratio_mode_candidates: Sequence[str] = ("auto", "factored")
    one_step_ratio_mode_candidates: Sequence[str] = ("auto", "factored")
    staged_bootstrap_cv: bool = False
    staged_cv_iterations: int = 3
    staged_cv_n_bootstrap: int = 200
    staged_cv_always_evaluate_baseline: bool = True
    staged_cv_min_survivors: int = 1
    staged_cv_one_se_multiplier: float = 1.0
    staged_cv_loss_metric: Literal["validation_loss", "selection_risk", "moment_balance"] = "validation_loss"

    def __post_init__(self) -> None:
        families = tuple(str(family) for family in self.families)
        if not families:
            raise ValueError("families must contain at least one estimator family.")
        if any(family not in {"boosted", "neural"} for family in families):
            raise ValueError("families entries must be 'boosted' or 'neural'.")
        if int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be >= 2.")
        if str(self.budget) not in {"fast", "balanced"}:
            raise ValueError("budget must be 'fast' or 'balanced'.")
        if int(self.max_candidates) <= 0:
            raise ValueError("max_candidates must be positive.")
        if int(self.promotion_candidates) <= 0:
            raise ValueError("promotion_candidates must be positive.")
        if not (0.0 < float(self.screen_fraction) <= 1.0):
            raise ValueError("screen_fraction must be in (0, 1].")
        weights = (
            self.score_moment_balance_weight,
            self.score_validation_weight,
            self.score_reward_stability_weight,
            self.score_weight_quality_weight,
            self.score_runtime_weight,
        )
        if any(float(weight) < 0.0 for weight in weights):
            raise ValueError("score weights must be nonnegative.")
        if sum(float(weight) for weight in weights) <= 0.0:
            raise ValueError("at least one score weight must be positive.")
        if str(self.score_method) not in {"legacy_rank", "bellman_gmm", "validation_loss"}:
            raise ValueError("score_method must be 'legacy_rank', 'bellman_gmm', or 'validation_loss'.")
        if str(self.gmm_objective) not in {"ratio", "ope"}:
            raise ValueError("gmm_objective must be 'ratio' or 'ope'.")
        if float(self.gmm_cov_ridge) < 0.0 or not np.isfinite(float(self.gmm_cov_ridge)):
            raise ValueError("gmm_cov_ridge must be finite and nonnegative.")
        if float(self.gmm_complexity_weight) < 0.0 or not np.isfinite(float(self.gmm_complexity_weight)):
            raise ValueError("gmm_complexity_weight must be finite and nonnegative.")
        if not (0.0 <= float(self.gmm_ope_broad_weight) <= 1.0) or not np.isfinite(float(self.gmm_ope_broad_weight)):
            raise ValueError("gmm_ope_broad_weight must be finite and in [0, 1].")
        if int(self.gmm_refit_top_candidates) <= 0:
            raise ValueError("gmm_refit_top_candidates must be positive.")
        if not (0.0 < float(self.gmm_refit_fraction) <= 1.0) or not np.isfinite(float(self.gmm_refit_fraction)):
            raise ValueError("gmm_refit_fraction must be finite and in (0, 1].")
        if int(self.moment_rff_features) < 0:
            raise ValueError("moment_rff_features must be nonnegative.")
        if int(self.moment_geometry_features) < 0:
            raise ValueError("moment_geometry_features must be nonnegative.")
        if int(self.moment_value_iterations) <= 0:
            raise ValueError("moment_value_iterations must be positive.")
        if int(self.moment_value_patience) < 0:
            raise ValueError("moment_value_patience must be nonnegative.")
        if float(self.moment_max_group_weight) < 0.0:
            raise ValueError("moment_max_group_weight must be nonnegative.")
        allowed_extra_blocks = {"second_order", "multiscale_rff", "support", "policy_shift"}
        unknown_blocks = sorted(str(block) for block in self.moment_extra_blocks if str(block) not in allowed_extra_blocks)
        if unknown_blocks:
            raise ValueError(f"moment_extra_blocks entries must be one of {sorted(allowed_extra_blocks)}.")
        for scale in self.moment_multiscale_rff_scales:
            if float(scale) <= 0.0 or not np.isfinite(float(scale)):
                raise ValueError("moment_multiscale_rff_scales entries must be positive finite values.")
        for quantile in self.moment_strata_quantiles:
            if not (0.0 < float(quantile) < 1.0):
                raise ValueError("moment_strata_quantiles entries must be in (0, 1).")
        if float(self.fallback_score_tolerance) < 0.0:
            raise ValueError("fallback_score_tolerance must be nonnegative.")
        if float(self.fallback_quality_margin) < 0.0:
            raise ValueError("fallback_quality_margin must be nonnegative.")
        if float(self.fallback_runtime_ratio) <= 1.0:
            raise ValueError("fallback_runtime_ratio must be > 1.")
        if float(self.fallback_moment_balance_tolerance) < 0.0:
            raise ValueError("fallback_moment_balance_tolerance must be nonnegative.")
        if not (0.0 < float(self.fallback_ess_ratio) <= 1.0):
            raise ValueError("fallback_ess_ratio must be in (0, 1].")
        if float(self.fallback_ess_abs_drop) < 0.0:
            raise ValueError("fallback_ess_abs_drop must be nonnegative.")
        if float(self.fallback_cv_ratio) < 1.0:
            raise ValueError("fallback_cv_ratio must be >= 1.")
        if float(self.fallback_cv_abs_increase) < 0.0:
            raise ValueError("fallback_cv_abs_increase must be nonnegative.")
        if self.first_stage_cv_folds is not None and int(self.first_stage_cv_folds) < 2:
            raise ValueError("first_stage_cv_folds must be >= 2 when supplied.")
        if not (1 <= int(self.staged_cv_iterations) <= 5):
            raise ValueError("staged_cv_iterations must be in [1, 5].")
        if int(self.staged_cv_n_bootstrap) < 0:
            raise ValueError("staged_cv_n_bootstrap must be nonnegative.")
        if int(self.staged_cv_min_survivors) <= 0:
            raise ValueError("staged_cv_min_survivors must be positive.")
        if float(self.staged_cv_one_se_multiplier) < 0.0 or not np.isfinite(float(self.staged_cv_one_se_multiplier)):
            raise ValueError("staged_cv_one_se_multiplier must be finite and nonnegative.")
        if str(self.staged_cv_loss_metric) not in {"validation_loss", "selection_risk", "moment_balance"}:
            raise ValueError("staged_cv_loss_metric must be 'validation_loss', 'selection_risk', or 'moment_balance'.")
        for mode in self.initial_ratio_mode_candidates:
            if str(mode).strip().lower() not in {"auto", "joint", "factored"}:
                raise ValueError("initial_ratio_mode_candidates entries must be 'auto', 'joint', or 'factored'.")
        for mode in self.one_step_ratio_mode_candidates:
            if str(mode).strip().lower() not in {"auto", "direct", "factored"}:
                raise ValueError("one_step_ratio_mode_candidates entries must be 'auto', 'direct', or 'factored'.")


@dataclass(frozen=True)
class OccupancySearchSpace:
    """Base configs and candidate overrides for the tuning harness."""

    boosted_occupancy: OccupancyRegressionConfig = field(default_factory=OccupancyRegressionConfig)
    boosted_action_ratio: ActionRatioConfig = field(default_factory=ActionRatioConfig)
    boosted_source_state_ratio: SourceStateRatioConfig = field(default_factory=SourceStateRatioConfig)
    boosted_transition_ratio: TransitionRatioConfig = field(default_factory=TransitionRatioConfig)
    neural_occupancy: Any = field(default_factory=_default_neural_occupancy_config)
    neural_action_ratio: Any = field(default_factory=_default_neural_action_config)
    neural_source_state_ratio: Any = field(default_factory=_default_neural_source_config)
    neural_transition_ratio: Any = field(default_factory=_default_neural_transition_config)
    google_dualdice: GoogleDualDICEConfig = field(default_factory=GoogleDualDICEConfig)
    boosted_candidates: Optional[Sequence[Dict[str, Dict[str, Any]]]] = None
    neural_candidates: Optional[Sequence[Dict[str, Dict[str, Any]]]] = None
    boosted_first_stage_grids: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None
    neural_first_stage_grids: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None


@dataclass
class FoldResult:
    candidate_id: str
    family: str
    budget_stage: str
    fold: int
    runtime_sec: float
    moment_balance: float
    moment_balance_max_group: float
    validation_loss: float
    norm_error: float
    ess_fraction: float
    p99: float
    max_weight: float
    clipped_fraction: float
    action_shift: float = 0.0
    weight_cv: float = 0.0
    reward_value: float | None = None
    moment_balance_targeted: float = float("nan")
    moment_balance_broad: float = float("nan")
    moment_balance_targeted_max_group: float = float("nan")
    moment_balance_broad_max_group: float = float("nan")
    moment_balance_mass: float = float("nan")
    moment_balance_reward: float = float("nan")
    moment_balance_value: float = float("nan")
    moment_balance_value_strata: float = float("nan")
    moment_balance_geometry: float = float("nan")
    moment_balance_rff: float = float("nan")
    moment_balance_rff_multiscale: float = float("nan")
    selection_risk: float = float("inf")
    selection_risk_raw: float = float("inf")
    selection_effective_dim: float = 0.0
    selection_complexity_penalty: float = 0.0


@dataclass
class CandidateResult:
    candidate_id: str
    family: str
    budget_stage: str
    overrides: Dict[str, Dict[str, Any]]
    fold_results: List[FoldResult]
    metrics: Dict[str, float]
    score: float = float("inf")
    runtime_sec: float = 0.0
    promoted: bool = False
    selected: bool = False
    error: str = ""
    candidate_label: str = ""


@dataclass
class OccupancyTuningResult:
    selected_family: str
    selected_candidate_id: str
    selected_overrides: Dict[str, Dict[str, Any]]
    selected_configs: Dict[str, Any]
    candidates: List[CandidateResult]
    folds: List[FoldResult]
    model: Any
    config: OccupancyTuningConfig
    first_stage: Dict[str, Any] = field(default_factory=dict)
    staged_cv: Optional[StagedCVResult] = None

    def candidate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for candidate in self.candidates:
            row = {
                "candidate_id": candidate.candidate_id,
                "candidate_label": candidate.candidate_label or candidate.candidate_id,
                "family": candidate.family,
                "budget_stage": candidate.budget_stage,
                "score": float(candidate.score),
                "runtime_sec": float(candidate.runtime_sec),
                "promoted": float(candidate.promoted),
                "selected": float(candidate.selected),
                "error": candidate.error,
            }
            row.update({f"metric_{key}": value for key, value in candidate.metrics.items()})
            rows.append(row)
        return rows

    def fold_rows(self) -> List[Dict[str, Any]]:
        return [asdict(fold) for fold in self.folds]

    def first_stage_candidate_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for family, telemetry in self.first_stage.items():
            for row in telemetry.get("candidate_rows", []):
                out = dict(row)
                out.setdefault("family", family)
                rows.append(out)
        return rows

    def first_stage_fold_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for family, telemetry in self.first_stage.items():
            for row in telemetry.get("fold_rows", []):
                out = dict(row)
                out.setdefault("family", family)
                rows.append(out)
        return rows

    def first_stage_skipped_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for family, telemetry in self.first_stage.items():
            for row in telemetry.get("skipped", []):
                out = dict(row)
                out.setdefault("family", family)
                rows.append(out)
        return rows

    def staged_cv_candidate_rows(self) -> List[Dict[str, Any]]:
        if self.staged_cv is None:
            return []
        return self.staged_cv.candidate_dicts()

    def staged_cv_fold_rows(self) -> List[Dict[str, Any]]:
        if self.staged_cv is None:
            return []
        return self.staged_cv.fold_dicts()


@dataclass
class OccupancyTargetValidationCandidateResult:
    candidate_id: str
    candidate_label: str
    family: str
    overrides: Dict[str, Dict[str, Any]]
    metrics: Dict[str, float]
    score: float = float("inf")
    score_se: float = 0.0
    runtime_sec: float = 0.0
    guardrail_passed: bool = True
    selected_min_score: bool = False
    selected: bool = False
    error: str = ""


@dataclass
class OccupancyTargetValidationResult:
    selected_family: str
    selected_candidate_id: str
    selected_overrides: Dict[str, Dict[str, Any]]
    selected_configs: Dict[str, Any]
    candidates: List[OccupancyTargetValidationCandidateResult]
    model: Any
    config: OccupancyTuningConfig
    score_mode: str
    validation_diagnostics: Dict[str, float | str]
    selection_rule: str = "min_score"

    def candidate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for candidate in self.candidates:
            row = {
                "candidate_id": candidate.candidate_id,
                "candidate_label": candidate.candidate_label or candidate.candidate_id,
                "family": candidate.family,
                "score": float(candidate.score),
                "score_se": float(candidate.score_se),
                "runtime_sec": float(candidate.runtime_sec),
                "guardrail_passed": float(candidate.guardrail_passed),
                "selected_min_score": float(candidate.selected_min_score),
                "selected": float(candidate.selected),
                "error": candidate.error,
            }
            row.update({f"metric_{key}": value for key, value in candidate.metrics.items()})
            rows.append(row)
        return rows

    def validation_rows(self) -> List[Dict[str, Any]]:
        return self.candidate_rows()


@dataclass(frozen=True)
class _OccupancyTargetTrajectory:
    states: Array
    actions: Array
    rewards: Optional[Array]
    episode_ids: Array
    timesteps: Array
    continuation: Optional[Array]


def tune_occupancy_ratio_auto(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    rewards: Optional[Array] = None,
    groups: Optional[Array] = None,
    families: Sequence[str] = ("neural",),
    search_space: Optional[OccupancySearchSpace] = None,
    config: Optional[OccupancyTuningConfig] = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
) -> OccupancyTuningResult:
    """Run the balanced product AutoML preset."""
    cfg = config if config is not None else OccupancyTuningConfig(families=tuple(families))
    if config is not None and tuple(families) != ("neural",):
        cfg = replace(cfg, families=tuple(families))
    return tune_occupancy_ratio(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        gamma=gamma,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        target_next_actions=target_next_actions,
        rewards=rewards,
        groups=groups,
        search_space=search_space,
        config=cfg,
        initial_ratio_mode=initial_ratio_mode,
        one_step_ratio_mode=one_step_ratio_mode,
    )


def tune_occupancy_ratio_with_target_validation(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    rewards: Optional[Array] = None,
    families: Sequence[str] = ("neural",),
    search_space: Optional[OccupancySearchSpace] = None,
    config: Optional[OccupancyTuningConfig] = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
    score_mode: Literal["discounted_moments", "scalar_ope"] = "discounted_moments",
    selection_rule: Literal["min_score", "one_se"] = "min_score",
    validation_states: Optional[Array] = None,
    validation_actions: Optional[Array] = None,
    validation_rewards: Optional[Array] = None,
    validation_episode_ids: Optional[Array] = None,
    validation_timestep: Optional[Array] = None,
    validation_terminals: Optional[Array] = None,
    validation_continuation: Optional[Array] = None,
    target_value: Optional[float] = None,
    target_value_se: Optional[float] = None,
) -> OccupancyTargetValidationResult:
    """Tune occupancy-ratio candidates with target-policy validation data.

    ``discounted_moments`` validates ratio-implied reference moments against
    finite target-policy discounted-occupancy moments. ``scalar_ope`` compares
    only the value moment and is reported as value-only selection. Candidates
    are selected by minimum guarded validation score by default; pass
    ``selection_rule="one_se"`` for a conservative one-standard-error selector.
    """

    cfg = config if config is not None else OccupancyTuningConfig(families=tuple(families))
    if config is not None and tuple(families) != ("neural",):
        cfg = replace(cfg, families=tuple(families))
    space = search_space if search_space is not None else OccupancySearchSpace()
    S = _as_2d(states, "states")
    A = _as_2d(actions, "actions")
    S_next = _as_2d(next_states, "next_states")
    A_pi = _as_2d(target_actions, "target_actions")
    if not (S.shape[0] == A.shape[0] == S_next.shape[0] == A_pi.shape[0]):
        raise ValueError("states, actions, next_states, and target_actions must have aligned rows.")
    S_initial = None if initial_states is None else _as_2d(initial_states, "initial_states")
    A_initial = None if initial_actions is None else _as_2d(initial_actions, "initial_actions")
    A_pi_next = None if target_next_actions is None else _as_2d(target_next_actions, "target_next_actions")
    rewards_arr = None if rewards is None else np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards_arr is not None and rewards_arr.shape[0] != S.shape[0]:
        raise ValueError("rewards must have the same number of rows as states.")
    if str(score_mode) not in {"discounted_moments", "scalar_ope"}:
        raise ValueError("score_mode must be 'discounted_moments' or 'scalar_ope'.")
    selection_rule_value = str(selection_rule)
    if selection_rule_value not in {"min_score", "one_se"}:
        raise ValueError("selection_rule must be 'min_score' or 'one_se'.")
    target_trajectory = None
    diagnostics: Dict[str, float | str] = {
        "score_mode": str(score_mode),
        "validation_selection_rule": f"guardrails_then_{selection_rule_value}",
    }
    if str(score_mode) == "discounted_moments":
        target_trajectory = _validate_occupancy_target_trajectory(
            validation_states=validation_states,
            validation_actions=validation_actions,
            validation_rewards=validation_rewards,
            validation_episode_ids=validation_episode_ids,
            validation_timestep=validation_timestep,
            validation_terminals=validation_terminals,
            validation_continuation=validation_continuation,
        )
        diagnostics.update(_occupancy_target_trajectory_diagnostics(target_trajectory, gamma=float(gamma), seed=int(cfg.seed) + 33_019))
    else:
        if rewards_arr is None:
            raise ValueError("rewards are required for score_mode='scalar_ope'.")
        if target_value is None or not np.isfinite(float(target_value)):
            raise ValueError("target_value must be supplied and finite for score_mode='scalar_ope'.")
        diagnostics.update(
            {
                "target_value": float(target_value),
                "target_value_se": 0.0 if target_value_se is None else float(target_value_se),
                "target_discounted_reward_moment": (1.0 - float(gamma)) * float(target_value),
                "validation_label_scope": "scalar_ope_only",
            }
        )

    candidates = _make_candidates(
        space,
        cfg,
        has_initial_states=S_initial is not None,
        has_initial_actions=A_initial is not None,
    )
    if bool(cfg.stagewise):
        candidates = _stagewise_occupancy_candidates(candidates)
    candidate_results: List[OccupancyTargetValidationCandidateResult] = []
    models: Dict[str, Any] = {}
    configs_by_id: Dict[str, Dict[str, Any]] = {}
    for rank, candidate in enumerate(candidates):
        start = time.perf_counter()
        family = str(candidate["family"])
        overrides = dict(candidate["overrides"])
        error = ""
        metrics: Dict[str, float] = {}
        score = float("inf")
        score_se = 0.0
        guardrail_passed = False
        try:
            configs = _build_configs(
                family=family,
                overrides=overrides,
                space=space,
                screen_fraction=1.0,
                seed=int(cfg.seed) + 707_707 + 10_001 * rank,
            )
            model = _fit_family(
                family=family,
                configs=configs,
                states=S,
                actions=A,
                next_states=S_next,
                target_actions=A_pi,
                gamma=float(gamma),
                initial_states=S_initial,
                initial_actions=A_initial,
                initial_weights=initial_weights,
                target_next_actions=A_pi_next,
                initial_ratio_mode=_candidate_mode(overrides, "initial_ratio_mode", initial_ratio_mode),
                one_step_ratio_mode=_candidate_mode(overrides, "one_step_ratio_mode", one_step_ratio_mode),
            )
            weights = np.asarray(model.predict_state_action_ratio(S, A, clip=True), dtype=np.float64).reshape(-1)
            weight_metrics = _final_weight_metrics(weights, _scoring_weight_config(configs), action_shift=_action_shift(A, A_pi))
            guardrail_passed = _target_validation_guardrails_pass(weight_metrics)
            if str(score_mode) == "discounted_moments":
                assert target_trajectory is not None
                metrics = _score_occupancy_discounted_moments(
                    weights=weights,
                    reference_states=S,
                    reference_actions=A,
                    reference_rewards=rewards_arr,
                    target=target_trajectory,
                    gamma=float(gamma),
                    seed=int(cfg.seed) + 919_001 + rank,
                )
            else:
                assert rewards_arr is not None and target_value is not None
                estimate = float(np.mean(weights * rewards_arr))
                target_moment = (1.0 - float(gamma)) * float(target_value)
                score = abs(estimate - target_moment)
                score_se = (1.0 - float(gamma)) * (0.0 if target_value_se is None else max(float(target_value_se), 0.0))
                metrics = {
                    "validation_score": score,
                    "validation_score_se": score_se,
                    "scalar_ope_estimate_discounted_reward": estimate,
                    "scalar_ope_target_discounted_reward": target_moment,
                    "scalar_ope_only": 1.0,
                }
            if str(score_mode) == "discounted_moments":
                score = float(metrics["validation_score"])
                score_se = float(metrics["validation_score_se"])
            metrics.update({f"weight_{key}": value for key, value in weight_metrics.items()})
            metrics["guardrail_passed"] = float(guardrail_passed)
            if not guardrail_passed:
                score = float("inf")
            models[str(candidate["candidate_id"])] = model
            configs_by_id[str(candidate["candidate_id"])] = configs
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        candidate_results.append(
            OccupancyTargetValidationCandidateResult(
                candidate_id=str(candidate["candidate_id"]),
                candidate_label=str(candidate.get("candidate_label", candidate["candidate_id"])),
                family=family,
                overrides=overrides,
                metrics=metrics,
                score=float(score),
                score_se=float(score_se),
                runtime_sec=float(time.perf_counter() - start),
                guardrail_passed=bool(guardrail_passed),
                error=error,
            )
        )
    selected, min_score_selected, one_se_selected = _select_occupancy_target_validation_candidate(
        candidate_results,
        selection_rule=selection_rule_value,
    )
    if selected is None:
        errors = "; ".join(candidate.error for candidate in candidate_results if candidate.error)
        raise RuntimeError(f"No occupancy target-validation candidates completed successfully. {errors}".strip())
    selected.selected = True
    selected_configs = configs_by_id.get(selected.candidate_id)
    assert selected_configs is not None
    diagnostics.update(
        {
            "selected_min_score_candidate_id": "" if min_score_selected is None else min_score_selected.candidate_id,
            "selected_one_se_candidate_id": "" if one_se_selected is None else one_se_selected.candidate_id,
            "all_guardrails_failed": float(not any(row.guardrail_passed for row in candidate_results if not row.error)),
        }
    )
    return OccupancyTargetValidationResult(
        selected_family=selected.family,
        selected_candidate_id=selected.candidate_id,
        selected_overrides=selected.overrides,
        selected_configs=selected_configs,
        candidates=candidate_results,
        model=models.get(selected.candidate_id),
        config=cfg,
        score_mode=str(score_mode),
        selection_rule=selection_rule_value,
        validation_diagnostics=diagnostics,
    )


def tune_occupancy_ratio(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array] = None,
    initial_actions: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    target_next_actions: Optional[Array] = None,
    rewards: Optional[Array] = None,
    groups: Optional[Array] = None,
    search_space: Optional[OccupancySearchSpace] = None,
    config: Optional[OccupancyTuningConfig] = None,
    initial_ratio_mode: str = "auto",
    one_step_ratio_mode: str = "auto",
) -> OccupancyTuningResult:
    cfg = config if config is not None else OccupancyTuningConfig()
    space = search_space if search_space is not None else OccupancySearchSpace()
    S = _as_2d(states, "states")
    A = _as_2d(actions, "actions")
    S_next = _as_2d(next_states, "next_states")
    A_pi = _as_2d(target_actions, "target_actions")
    if not (S.shape[0] == A.shape[0] == S_next.shape[0] == A_pi.shape[0]):
        raise ValueError("states, actions, next_states, and target_actions must have aligned rows.")
    S_initial = None if initial_states is None else _as_2d(initial_states, "initial_states")
    A_initial = None if initial_actions is None else _as_2d(initial_actions, "initial_actions")
    A_pi_next = None if target_next_actions is None else _as_2d(target_next_actions, "target_next_actions")
    rewards_arr = None if rewards is None else np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards_arr is not None and rewards_arr.shape[0] != S.shape[0]:
        raise ValueError("rewards must have the same number of rows as states.")

    folds = _make_folds(S.shape[0], int(cfg.cv_folds), int(cfg.seed), groups=groups)
    candidates = _make_candidates(
        space,
        cfg,
        has_initial_states=S_initial is not None,
        has_initial_actions=A_initial is not None,
    )
    first_stage_by_family: Dict[str, Optional[Dict[str, Any]]] = {}
    if bool(cfg.stagewise):
        first_stage_fold_count = int(cfg.cv_folds if cfg.first_stage_cv_folds is None else cfg.first_stage_cv_folds)
        score_folds = (
            folds
            if first_stage_fold_count == int(cfg.cv_folds)
            else _make_folds(S.shape[0], first_stage_fold_count, int(cfg.seed) + 404_001, groups=groups)
        )
        for family in tuple(str(family) for family in cfg.families):
            first_stage_by_family[family] = fit_first_stage_for_family(
                family=family,
                space=space,
                cfg=cfg,
                candidates=candidates,
                score_folds=score_folds,
                reuse_folds=folds,
                S=S,
                A=A,
                S_next=S_next,
                A_pi=A_pi,
                S_initial=S_initial,
                A_initial=A_initial,
                initial_weights=initial_weights,
                A_pi_next=A_pi_next,
                initial_ratio_mode=initial_ratio_mode,
                one_step_ratio_mode=one_step_ratio_mode,
                seed=int(cfg.seed) + (0 if family == "boosted" else 250_001),
    )
    evaluation_candidates = _stagewise_occupancy_candidates(candidates) if bool(cfg.stagewise) else list(candidates)
    moment_block_cache: MomentBlockCache = {}
    if bool(cfg.staged_bootstrap_cv):
        staged_cv_result, staged_candidates, staged_final_losses = _run_occupancy_staged_cv(
            candidates=evaluation_candidates,
            folds=folds,
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            gamma=gamma,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weights,
            A_pi_next=A_pi_next,
            rewards=rewards_arr,
            space=space,
            cfg=cfg,
            seed=int(cfg.seed) + 311_707,
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
            first_stage_by_family=first_stage_by_family,
            moment_block_cache=moment_block_cache,
        )
        kept_ids = staged_cv_result.kept_candidate_ids
        baseline_ids = {
            str(candidate["candidate_id"])
            for candidate in evaluation_candidates
            if str(candidate["candidate_id"]).endswith("_000")
        }
        full_eval_ids = set(kept_ids)
        if bool(cfg.staged_cv_always_evaluate_baseline):
            full_eval_ids.update(baseline_ids)
        full_candidates = [_evaluate_candidate(
            candidate=candidate,
            budget_stage="full",
            screen_fraction=1.0,
            folds=folds,
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            gamma=gamma,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weights,
            A_pi_next=A_pi_next,
            rewards=rewards_arr,
            space=space,
            cfg=cfg,
            seed=int(cfg.seed) + 97_531,
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
            first_stage=first_stage_by_family.get(str(candidate["family"])),
            moment_block_cache=moment_block_cache,
        ) for candidate in evaluation_candidates if str(candidate["candidate_id"]) in full_eval_ids]
        _score_candidates(full_candidates, cfg)
        for candidate in full_candidates:
            candidate.promoted = candidate.candidate_id in kept_ids
            candidate.metrics["staged_selection_eligible"] = float(candidate.candidate_id in kept_ids)
            if candidate.candidate_id in staged_final_losses:
                candidate.metrics["proxy_score_before_staged_cv"] = float(candidate.score)
                candidate.metrics["staged_cv_final_loss"] = float(staged_final_losses[candidate.candidate_id])
                if candidate.candidate_id in kept_ids:
                    candidate.score = float(staged_final_losses[candidate.candidate_id])
        selection_pool = [
            candidate
            for candidate in full_candidates
            if candidate.candidate_id in kept_ids
            and not candidate.error
            and candidate.fold_results
            and np.isfinite(float(candidate.score))
        ]
        if not selection_pool:
            selection_pool = [
                candidate
                for candidate in full_candidates
                if not candidate.error and candidate.fold_results and np.isfinite(float(candidate.score))
            ]
        if not selection_pool:
            errors = "; ".join(candidate.error for candidate in full_candidates + staged_candidates if candidate.error)
            raise RuntimeError(f"No staged tuning candidates completed successfully. {errors}".strip())
        selected = min(selection_pool, key=lambda row: row.score)
        selected_configs = _build_configs(
            family=selected.family,
            overrides=selected.overrides,
            space=space,
            screen_fraction=1.0,
            seed=int(cfg.seed) + 707_707,
        )
        selected_first_stage = first_stage_by_family.get(selected.family)
        if (
            selected_first_stage is not None
            and str(selected.overrides.get("backend", {}).get("name", "")) != "google_dualdice"
            and not bool(selected.overrides.get("modes"))
        ):
            selected_configs = _configs_with_first_stage(selected_configs, selected_first_stage)
        model = None
        if cfg.refit:
            selected, selected_configs, model = _select_refit_candidate(
                candidates=selection_pool,
                space=space,
                cfg=cfg,
                S=S,
                A=A,
                S_next=S_next,
                A_pi=A_pi,
                gamma=gamma,
                S_initial=S_initial,
                A_initial=A_initial,
                initial_weights=initial_weights,
                A_pi_next=A_pi_next,
                initial_ratio_mode=initial_ratio_mode,
                one_step_ratio_mode=one_step_ratio_mode,
                first_stage_by_family=first_stage_by_family,
            )
        selected.selected = True
        all_candidates = staged_candidates + full_candidates
        all_folds = [fold for candidate in all_candidates for fold in candidate.fold_results]
        return OccupancyTuningResult(
            selected_family=selected.family,
            selected_candidate_id=selected.candidate_id,
            selected_overrides=selected.overrides,
            selected_configs=selected_configs,
            candidates=all_candidates,
            folds=all_folds,
            model=model,
            config=cfg,
            first_stage=public_first_stage_telemetry(first_stage_by_family),
            staged_cv=staged_cv_result,
        )
    screen_candidates = [_evaluate_candidate(
        candidate=candidate,
        budget_stage="screen",
        screen_fraction=_budget_screen_fraction(cfg),
        folds=folds,
        S=S,
        A=A,
        S_next=S_next,
        A_pi=A_pi,
        gamma=gamma,
        S_initial=S_initial,
        A_initial=A_initial,
        initial_weights=initial_weights,
        A_pi_next=A_pi_next,
        rewards=rewards_arr,
        space=space,
        cfg=cfg,
        seed=int(cfg.seed),
        initial_ratio_mode=initial_ratio_mode,
        one_step_ratio_mode=one_step_ratio_mode,
        first_stage=first_stage_by_family.get(str(candidate["family"])),
        moment_block_cache=moment_block_cache,
    ) for candidate in evaluation_candidates]
    _score_candidates(screen_candidates, cfg)
    promoted_limit = _budget_promotion_limit(cfg)
    promoted_ids = {
        candidate.candidate_id
        for candidate in sorted(screen_candidates, key=lambda row: row.score)[: min(promoted_limit, len(screen_candidates))]
    }
    promoted_ids.update(
        candidate.candidate_id
        for candidate in screen_candidates
        if _is_baseline_candidate(candidate) and not candidate.error and candidate.fold_results
    )
    promoted_ids.update(
        candidate.candidate_id
        for candidate in screen_candidates
        if _is_mode_variant_candidate(candidate) and not candidate.error and candidate.fold_results
    )
    for candidate in screen_candidates:
        candidate.promoted = candidate.candidate_id in promoted_ids
    full_candidates = [_evaluate_candidate(
        candidate=candidate,
        budget_stage="full",
        screen_fraction=1.0,
        folds=folds,
        S=S,
        A=A,
        S_next=S_next,
        A_pi=A_pi,
        gamma=gamma,
        S_initial=S_initial,
        A_initial=A_initial,
        initial_weights=initial_weights,
        A_pi_next=A_pi_next,
        rewards=rewards_arr,
        space=space,
        cfg=cfg,
        seed=int(cfg.seed) + 97_531,
        initial_ratio_mode=initial_ratio_mode,
        one_step_ratio_mode=one_step_ratio_mode,
        first_stage=first_stage_by_family.get(str(candidate["family"])),
        moment_block_cache=moment_block_cache,
    ) for candidate in evaluation_candidates if candidate["candidate_id"] in promoted_ids]
    _score_candidates(full_candidates, cfg)
    for candidate in full_candidates:
        candidate.promoted = True
    staged_cv_result = None
    staged_kept_ids: Optional[set[str]] = None
    if bool(cfg.staged_bootstrap_cv) and full_candidates:
        staged_cv_result = run_staged_bootstrap_cv(
            full_candidates,
            cfg,
            seed=int(cfg.seed) + 808_801,
        )
        staged_kept_ids = staged_cv_result.kept_candidate_ids
    selection_pool = [
        candidate
        for candidate in (full_candidates or screen_candidates)
        if not candidate.error and candidate.fold_results and np.isfinite(float(candidate.score))
        and (staged_kept_ids is None or candidate.candidate_id in staged_kept_ids)
    ]
    if not selection_pool and staged_kept_ids is not None:
        selection_pool = [
            candidate
            for candidate in (full_candidates or screen_candidates)
            if not candidate.error and candidate.fold_results and np.isfinite(float(candidate.score))
        ]
    if not selection_pool:
        errors = "; ".join(candidate.error for candidate in (full_candidates or screen_candidates) if candidate.error)
        raise RuntimeError(f"No tuning candidates completed successfully. {errors}".strip())
    model = None
    selected = min(selection_pool, key=lambda row: row.score)
    selected_configs = _build_configs(
        family=selected.family,
        overrides=selected.overrides,
        space=space,
        screen_fraction=1.0,
        seed=int(cfg.seed) + 707_707,
    )
    selected_first_stage = first_stage_by_family.get(selected.family)
    if (
        selected_first_stage is not None
        and str(selected.overrides.get("backend", {}).get("name", "")) != "google_dualdice"
        and not bool(selected.overrides.get("modes"))
    ):
        selected_configs = _configs_with_first_stage(selected_configs, selected_first_stage)
    if cfg.refit:
        selected, selected_configs, model = _select_refit_candidate(
            candidates=selection_pool,
            space=space,
            cfg=cfg,
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            gamma=gamma,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weights,
            A_pi_next=A_pi_next,
            initial_ratio_mode=initial_ratio_mode,
            one_step_ratio_mode=one_step_ratio_mode,
            first_stage_by_family=first_stage_by_family,
        )
    selected.selected = True
    all_candidates = screen_candidates + full_candidates
    all_folds = [fold for candidate in all_candidates for fold in candidate.fold_results]
    return OccupancyTuningResult(
        selected_family=selected.family,
        selected_candidate_id=selected.candidate_id,
        selected_overrides=selected.overrides,
        selected_configs=selected_configs,
        candidates=all_candidates,
        folds=all_folds,
        model=model,
        config=cfg,
        first_stage=public_first_stage_telemetry(first_stage_by_family),
        staged_cv=staged_cv_result,
    )


def _run_occupancy_staged_cv(
    *,
    candidates: Sequence[Dict[str, Any]],
    folds: Sequence[Array],
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    gamma: float,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    A_pi_next: Optional[Array],
    rewards: Optional[Array],
    space: OccupancySearchSpace,
    cfg: OccupancyTuningConfig,
    seed: int,
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
    first_stage_by_family: Dict[str, Optional[Dict[str, Any]]],
    moment_block_cache: Optional[MomentBlockCache],
) -> Tuple[StagedCVResult, List[CandidateResult], Dict[str, float]]:
    metric = str(cfg.staged_cv_loss_metric)
    n_bootstrap = int(cfg.staged_cv_n_bootstrap)
    active_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    baseline_ids = {str(candidate["candidate_id"]) for candidate in candidates if str(candidate["candidate_id"]).endswith("_000")}
    input_dim = int(S.shape[1] + A.shape[1])
    complexity = _occupancy_candidate_complexity_map(candidates, space, input_dim=input_dim)
    all_stage_candidates: List[CandidateResult] = []
    candidate_rows: List[StagedCVCandidateRow] = []
    fold_rows: List[StagedCVFoldRow] = []
    final_losses: Dict[str, float] = {}
    final_threshold = float("inf")
    selected_candidate_id = ""

    for stage in range(1, int(cfg.staged_cv_iterations) + 1):
        evaluation_ids = set(active_ids)
        if bool(cfg.staged_cv_always_evaluate_baseline):
            evaluation_ids.update(baseline_ids)
        stage_results: List[CandidateResult] = []
        stage_losses: Dict[str, float] = {}
        stage_ses: Dict[str, float] = {}
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id not in evaluation_ids:
                continue
            forced_baseline = bool(candidate_id in baseline_ids and candidate_id not in active_ids)
            staged_candidate = _candidate_with_occupancy_iterations(candidate, stage)
            result = _evaluate_candidate(
                candidate=staged_candidate,
                budget_stage=f"staged_{stage}",
                screen_fraction=1.0,
                folds=folds,
                S=S,
                A=A,
                S_next=S_next,
                A_pi=A_pi,
                gamma=gamma,
                S_initial=S_initial,
                A_initial=A_initial,
                initial_weights=initial_weights,
                A_pi_next=A_pi_next,
                rewards=rewards,
                space=space,
                cfg=cfg,
                seed=int(seed) + 10_007 * stage,
                initial_ratio_mode=initial_ratio_mode,
                one_step_ratio_mode=one_step_ratio_mode,
                first_stage=first_stage_by_family.get(str(candidate["family"])),
                moment_block_cache=moment_block_cache,
            )
            losses = np.asarray([_staged_fold_loss(fold, metric) for fold in result.fold_results], dtype=np.float64)
            finite_losses = losses[np.isfinite(losses)]
            loss = float(np.mean(finite_losses)) if finite_losses.size else float("inf")
            loss_se = _bootstrap_mean_se(finite_losses, iterations=n_bootstrap, seed=int(seed) + 77_021 * stage)
            result.score = loss
            result.metrics["staged_cv_stage"] = float(stage)
            result.metrics["staged_cv_loss"] = loss
            result.metrics["staged_cv_loss_se"] = loss_se
            result.metrics["staged_cv_baseline_forced_eval"] = float(forced_baseline)
            stage_results.append(result)
            all_stage_candidates.append(result)
            stage_losses[candidate_id] = loss
            stage_ses[candidate_id] = loss_se
            for fold, fold_loss in zip(result.fold_results, losses):
                fold_rows.append(
                    StagedCVFoldRow(
                        candidate_id=candidate_id,
                        family=str(result.family),
                        budget_stage=f"staged_{stage}",
                        fold=int(fold.fold),
                        loss=float(fold_loss),
                        metric=metric,
                        stage=int(stage),
                        stage_loss=float(fold_loss),
                        stage_loss_se=float(loss_se),
                        active=candidate_id in active_ids,
                        baseline_forced_eval=forced_baseline,
                    )
                )

        stage_rows: List[StagedCVCandidateRow] = []
        for result in stage_results:
            desc = complexity.get(result.candidate_id, {})
            stage_rows.append(
                StagedCVCandidateRow(
                    candidate_id=result.candidate_id,
                    candidate_label=result.candidate_label or result.candidate_id,
                    family=result.family,
                    budget_stage=f"staged_{stage}",
                    loss_mean=float(stage_losses.get(result.candidate_id, float("inf"))),
                    loss_se=float(stage_ses.get(result.candidate_id, float("inf"))),
                    bootstrap_iterations=n_bootstrap,
                    selected_min_loss=False,
                    kept=True,
                    pruned=False,
                    threshold=float("inf"),
                    reason="pending",
                    stage=int(stage),
                    stage_loss=float(stage_losses.get(result.candidate_id, float("inf"))),
                    stage_loss_se=float(stage_ses.get(result.candidate_id, float("inf"))),
                    active=result.candidate_id in active_ids,
                    baseline_forced_eval=bool(result.metrics.get("staged_cv_baseline_forced_eval", 0.0) > 0.5),
                    final_stage=stage == int(cfg.staged_cv_iterations),
                    complexity_group=str(desc.get("group", "")),
                    complexity_rank=str(desc.get("rank_repr", "")),
                    complexity_source=str(desc.get("source", "none")),
                )
            )
        if not stage_rows:
            break
        if not any(row.candidate_id in active_ids and np.isfinite(float(row.loss_mean)) for row in stage_rows):
            break
        active_ids, _, threshold = monotone_one_se_prune(
            stage_rows,
            active_ids,
            complexity,
            float(cfg.staged_cv_one_se_multiplier),
            max(1, int(cfg.staged_cv_min_survivors)),
        )
        final_threshold = threshold
        stage_rows_by_id = {row.candidate_id: row for row in stage_rows}
        for result in stage_results:
            row = stage_rows_by_id[result.candidate_id]
            row.threshold = float(threshold)
            row.reason = row.prune_reason or row.reason
            result.metrics["staged_cv_selected_min_loss"] = float(row.selected_min_loss)
            result.metrics["staged_cv_kept"] = float(row.kept)
            result.metrics["staged_cv_pruned"] = float(row.pruned)
            result.metrics["staged_cv_threshold"] = float(threshold)
            result.metrics["staged_cv_complexity_group"] = row.complexity_group
            result.metrics["staged_cv_complexity_rank"] = row.complexity_rank
            result.metrics["staged_cv_complexity_source"] = row.complexity_source
            result.metrics["staged_cv_prune_reason"] = row.prune_reason
        for fold_row in fold_rows:
            if int(fold_row.stage) != int(stage) or fold_row.candidate_id not in stage_rows_by_id:
                continue
            row = stage_rows_by_id[fold_row.candidate_id]
            fold_row.active = bool(row.active)
            fold_row.pruned = bool(row.pruned)
            fold_row.complexity_group = row.complexity_group
            fold_row.complexity_rank = row.complexity_rank
            fold_row.complexity_source = row.complexity_source
            fold_row.stage_best_candidate_id = row.stage_best_candidate_id
            fold_row.outside_one_se = bool(row.outside_one_se)
            fold_row.strictly_simpler_than_stage_best = bool(row.strictly_simpler_than_stage_best)
            fold_row.prune_reason = row.prune_reason
        candidate_rows.extend(stage_rows)
        if stage == int(cfg.staged_cv_iterations):
            final_losses = {candidate_id: float(stage_losses[candidate_id]) for candidate_id in active_ids if candidate_id in stage_losses}
            if final_losses:
                selected_candidate_id = min(final_losses, key=final_losses.get)

    if not selected_candidate_id:
        finite_rows = [row for row in candidate_rows if np.isfinite(float(row.loss_mean)) and row.kept]
        if finite_rows:
            selected_row = min(finite_rows, key=lambda row: row.loss_mean)
            selected_candidate_id = selected_row.candidate_id
            final_losses = {selected_row.candidate_id: float(selected_row.loss_mean)}
        else:
            selected_candidate_id = ""
    for row in candidate_rows:
        if row.candidate_id == selected_candidate_id and bool(row.final_stage or not any(r.final_stage for r in candidate_rows)):
            row.selected = True
    for row in fold_rows:
        row.selected = bool(row.candidate_id == selected_candidate_id and (row.stage == int(cfg.staged_cv_iterations) or not any(r.final_stage for r in candidate_rows)))
    return (
        StagedCVResult(
            candidate_rows=candidate_rows,
            fold_rows=fold_rows,
            selected_candidate_id=selected_candidate_id,
            threshold=float(final_threshold),
            loss_metric=metric,
        ),
        all_stage_candidates,
        final_losses,
    )


def _candidate_with_occupancy_iterations(candidate: Dict[str, Any], stage: int) -> Dict[str, Any]:
    out = dict(candidate)
    overrides = {str(key): dict(value) for key, value in dict(candidate.get("overrides", {})).items()}
    occupancy = dict(overrides.get("occupancy", {}))
    occupancy["num_iterations"] = int(stage)
    overrides["occupancy"] = occupancy
    out["overrides"] = overrides
    return out


def _occupancy_candidate_complexity_map(
    candidates: Sequence[Dict[str, Any]],
    space: OccupancySearchSpace,
    *,
    input_dim: int,
) -> Dict[str, Dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): _occupancy_candidate_complexity(candidate, space, input_dim=input_dim)
        for candidate in candidates
    }


def _occupancy_candidate_complexity(
    candidate: Dict[str, Any],
    space: OccupancySearchSpace,
    *,
    input_dim: int,
) -> Dict[str, Any]:
    family = str(candidate.get("family", ""))
    overrides = dict(candidate.get("overrides", {}) or {})
    explicit = _explicit_candidate_complexity(family, overrides)
    if explicit is not None:
        return explicit
    try:
        if family == "neural":
            occupancy = replace(space.neural_occupancy, **dict(overrides.get("occupancy", {}) or {}))
            dims = tuple(int(width) for width in occupancy.hidden_dims)
            return _complexity_descriptor(
                group=f"{family}:occupancy_mlp_parameter_count",
                rank=_mlp_parameter_count(int(input_dim), dims, output_dim=1),
                source="inferred",
            )
        if family == "boosted":
            occupancy = replace(space.boosted_occupancy, **dict(overrides.get("occupancy", {}) or {}))
            params = dict(getattr(occupancy, "lgb_params", {}) or {})
            return _complexity_descriptor(
                group=f"{family}:occupancy_tree_capacity",
                rank=_boosted_capacity_rank(params, trees_per_iteration=int(occupancy.trees_per_iteration)),
                source="inferred",
            )
    except Exception:
        pass
    return _complexity_descriptor(group="", rank=None, source="none")


def _explicit_candidate_complexity(family: str, overrides: Dict[str, Any]) -> Dict[str, Any] | None:
    meta = overrides.get("_meta")
    if not isinstance(meta, dict) or "complexity_rank" not in meta:
        return None
    return _complexity_descriptor(
        group=str(meta.get("complexity_group", f"{family}:explicit")),
        rank=meta.get("complexity_rank"),
        source="explicit",
    )


def _complexity_descriptor(*, group: str, rank: Any, source: str) -> Dict[str, Any]:
    rank_tuple = _complexity_rank_tuple(rank)
    return {
        "group": str(group),
        "rank": rank_tuple,
        "rank_repr": "" if rank_tuple is None else "x".join(f"{value:g}" for value in rank_tuple),
        "source": str(source),
    }


def _complexity_rank_tuple(value: Any) -> Tuple[float, ...] | None:
    if value is None:
        return None
    values = value if isinstance(value, (list, tuple)) else (value,)
    try:
        out = tuple(float(item) for item in values)
    except (TypeError, ValueError):
        return None
    if not out or not all(np.isfinite(item) for item in out):
        return None
    return out


def _boosted_capacity_rank(params: Dict[str, Any], *, trees_per_iteration: int) -> Tuple[float, ...]:
    leaves = float(params.get("num_leaves", 31))
    raw_depth = params.get("max_depth", 1_000_000)
    if raw_depth is not None and int(raw_depth) > 0:
        leaves = min(leaves, float(2 ** int(raw_depth)))
    return (leaves * float(trees_per_iteration),)


def _mlp_parameter_count(input_dim: int, hidden_dims: Sequence[int], *, output_dim: int) -> float:
    dims = [int(input_dim), *(int(width) for width in hidden_dims), int(output_dim)]
    total = 0
    for left, right in zip(dims[:-1], dims[1:]):
        total += left * right + right
    return float(total)


def _staged_fold_loss(fold: FoldResult, metric: str) -> float:
    value = getattr(fold, metric, None)
    if value is not None and np.isfinite(float(value)):
        return float(value)
    for fallback in ("validation_loss", "selection_risk", "moment_balance"):
        value = getattr(fold, fallback, None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return float("inf")


def _bootstrap_mean_se(values: Array, *, iterations: int, seed: int) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    if int(iterations) <= 0:
        return float(np.std(x, ddof=1) / np.sqrt(x.size))
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, x.size, size=(int(iterations), x.size))
    means = np.mean(x[draws], axis=1)
    return float(np.std(means, ddof=1)) if means.size > 1 else 0.0


def _select_refit_candidate(
    *,
    candidates: Sequence[CandidateResult],
    space: OccupancySearchSpace,
    cfg: OccupancyTuningConfig,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    gamma: float,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    A_pi_next: Optional[Array],
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
    first_stage_by_family: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> Tuple[CandidateResult, Dict[str, Any], Any]:
    scored: List[Tuple[float, CandidateResult, Dict[str, Any], Any]] = []
    ordered = sorted(candidates, key=lambda row: row.score)
    ordered_for_refit = _refit_candidates_for_method(ordered, cfg)
    refit_fraction = float(cfg.gmm_refit_fraction) if str(cfg.score_method) == "bellman_gmm" else 1.0
    refit_ids = {candidate.candidate_id for candidate in ordered_for_refit}
    for candidate in ordered:
        if candidate.candidate_id not in refit_ids:
            candidate.metrics["final_refit_skipped_by_cap"] = 1.0
    for rank, candidate in enumerate(ordered_for_refit):
        configs = _build_configs(
            family=candidate.family,
            overrides=candidate.overrides,
            space=space,
            screen_fraction=refit_fraction,
            seed=_refit_seed(candidate, space, cfg, rank),
        )
        try:
            first_stage = None if first_stage_by_family is None else first_stage_by_family.get(candidate.family)
            explicit_modes = bool(candidate.overrides.get("modes"))
            uses_stagewise = (
                first_stage is not None
                and str(candidate.overrides.get("backend", {}).get("name", "")) != "google_dualdice"
                and not explicit_modes
            )
            if uses_stagewise:
                configs = _configs_with_first_stage(configs, first_stage)
            refit_initial_mode = (
                str(first_stage["selected_initial_ratio_mode"])
                if uses_stagewise
                else _candidate_mode(candidate.overrides, "initial_ratio_mode", initial_ratio_mode)
            )
            refit_one_step_mode = (
                str(first_stage["selected_one_step_ratio_mode"])
                if uses_stagewise
                else _candidate_mode(candidate.overrides, "one_step_ratio_mode", one_step_ratio_mode)
            )
            model = _fit_family(
                family=candidate.family,
                configs=configs,
                states=S,
                actions=A,
                next_states=S_next,
                target_actions=A_pi,
                gamma=gamma,
                initial_states=S_initial,
                initial_actions=A_initial,
                initial_weights=initial_weights,
                target_next_actions=A_pi_next,
                initial_ratio_mode=refit_initial_mode,
                one_step_ratio_mode=refit_one_step_mode,
                prefit_nuisance=None if not uses_stagewise else first_stage["final_bundle"],
            )
            weights = model.predict_state_action_ratio(S, A, clip=True)
            final_metrics = _final_weight_metrics(weights, _scoring_weight_config(configs), action_shift=_action_shift(A, A_pi))
            candidate.metrics.update({f"final_{key}": value for key, value in final_metrics.items()})
            if str(cfg.score_method) == "bellman_gmm":
                final_flags = _gmm_safety_constraint_flags({**candidate.metrics, **final_metrics})
                candidate.metrics["final_constraint_violated"] = float(final_flags["any"]) if bool(cfg.gmm_use_safety_constraints) else 0.0
                candidate.metrics["final_constraint_catastrophic_ess"] = (
                    float(final_flags["catastrophic_ess"]) if bool(cfg.gmm_use_safety_constraints) else 0.0
                )
                candidate.metrics["final_constraint_clipping"] = float(final_flags["clipping"]) if bool(cfg.gmm_use_safety_constraints) else 0.0
                candidate.metrics["final_constraint_normalization"] = (
                    float(final_flags["normalization"]) if bool(cfg.gmm_use_safety_constraints) else 0.0
                )
                candidate.metrics["final_constraint_near_uniform_collapse"] = (
                    float(final_flags["near_uniform_collapse"]) if bool(cfg.gmm_use_safety_constraints) else 0.0
                )
                final_score = float(candidate.score)
            elif str(cfg.score_method) == "validation_loss" or bool(cfg.staged_bootstrap_cv):
                final_score = float(candidate.score)
            else:
                final_score = float(candidate.score) + _final_refit_penalty(final_metrics)
            candidate.metrics["final_selection_score"] = float(final_score)
            candidate.metrics["final_refit_fraction"] = float(refit_fraction)
            scored.append((final_score, candidate, configs, model))
        except Exception as exc:
            candidate.metrics["final_refit_failed"] = 1.0
            candidate.error = candidate.error or f"{type(exc).__name__}: {exc}"
    if not scored:
        errors = "; ".join(candidate.error for candidate in ordered if candidate.error)
        raise RuntimeError(f"No tuning candidates refit successfully. {errors}".strip())
    if str(cfg.score_method) == "bellman_gmm" and bool(cfg.gmm_use_safety_constraints):
        feasible_scored = [row for row in scored if float(row[1].metrics.get("final_constraint_violated", 0.0)) < 0.5]
        if feasible_scored:
            scored_for_selection = feasible_scored
        else:
            scored_for_selection = scored
            for _, candidate, _, _ in scored_for_selection:
                candidate.metrics["final_constraint_all_violated"] = 1.0
    else:
        scored_for_selection = scored
    selected_score, selected, configs, model = min(scored_for_selection, key=lambda row: row[0])
    baseline_rows = [row for row in scored_for_selection if _is_baseline_candidate(row[1]) and row[1].family == selected.family]
    if bool(cfg.stable_fallback) and baseline_rows:
        baseline_score, baseline, baseline_configs, baseline_model = min(baseline_rows, key=lambda row: row[0])
        if baseline.candidate_id != selected.candidate_id and _should_fallback_to_baseline(
            selected=selected,
            baseline=baseline,
            selected_score=float(selected_score),
            baseline_score=float(baseline_score),
            cfg=cfg,
        ):
            selected.metrics["stable_fallback_replaced_by_baseline"] = 1.0
            baseline.metrics["stable_fallback_selected"] = 1.0
            selected, configs, model = baseline, baseline_configs, baseline_model
    return selected, configs, model


def _refit_candidates_for_method(candidates: Sequence[CandidateResult], cfg: OccupancyTuningConfig) -> List[CandidateResult]:
    ordered = list(candidates)
    if str(cfg.score_method) != "bellman_gmm":
        return ordered
    limit = max(1, int(cfg.gmm_refit_top_candidates))
    return list(ordered[:limit])


def _is_baseline_candidate(candidate: CandidateResult) -> bool:
    return str(candidate.candidate_id).endswith("_000")


def _is_mode_variant_candidate(candidate: CandidateResult) -> bool:
    return bool(candidate.overrides.get("modes"))


def _refit_seed(candidate: CandidateResult, space: OccupancySearchSpace, cfg: OccupancyTuningConfig, rank: int) -> int:
    if _is_baseline_candidate(candidate):
        base = space.boosted_occupancy if candidate.family == "boosted" else space.neural_occupancy
        return int(getattr(base, "seed", cfg.seed))
    if str(candidate.overrides.get("backend", {}).get("name", "")) == "google_dualdice":
        return int(space.google_dualdice.seed)
    return int(cfg.seed) + 707_707 + 10_001 * int(rank)


def _should_fallback_to_baseline(
    *,
    selected: CandidateResult,
    baseline: CandidateResult,
    selected_score: float,
    baseline_score: float,
    cfg: OccupancyTuningConfig,
) -> bool:
    if not (np.isfinite(selected_score) and np.isfinite(baseline_score)):
        return False
    if _weak_moment_instability_fallback(selected=selected, baseline=baseline, cfg=cfg):
        selected.metrics["stable_fallback_weak_moment_instability"] = 1.0
        return True
    if baseline_score > selected_score + float(cfg.fallback_score_tolerance):
        return False
    selected_quality = float(selected.metrics.get("final_weight_quality", selected.metrics.get("weight_quality", float("inf"))))
    baseline_quality = float(baseline.metrics.get("final_weight_quality", baseline.metrics.get("weight_quality", float("inf"))))
    selected_ess = float(selected.metrics.get("final_ess_fraction", selected.metrics.get("ess_fraction", 0.0)))
    baseline_ess = float(baseline.metrics.get("final_ess_fraction", baseline.metrics.get("ess_fraction", 0.0)))
    selected_n = selected.metrics.get("final_n_weights", selected.metrics.get("n_weights"))
    baseline_n = baseline.metrics.get("final_n_weights", baseline.metrics.get("n_weights"))
    selected_clipped = float(selected.metrics.get("final_clipped_fraction", selected.metrics.get("clipped_fraction", 1.0)))
    baseline_clipped = float(baseline.metrics.get("final_clipped_fraction", baseline.metrics.get("clipped_fraction", 1.0)))
    selected_ess_bad = _ess_is_catastrophic(selected_ess, n_weights=selected_n)
    baseline_ess_bad = _ess_is_catastrophic(baseline_ess, n_weights=baseline_n)

    quality_margin = float(cfg.fallback_quality_margin)
    selected_has_clear_quality_gain = (
        baseline_quality - selected_quality > quality_margin
        and selected_clipped <= baseline_clipped + 0.01
        and not selected_ess_bad
    )
    baseline_is_comparable = (
        baseline_quality <= selected_quality + quality_margin
        and not baseline_ess_bad
        and baseline_clipped <= selected_clipped + 0.01
    )
    baseline_remedies_safety = (
        (selected_ess_bad and not baseline_ess_bad)
        or selected_clipped > 0.05
        or selected_clipped > baseline_clipped + 0.03
    )
    baseline_has_less_clipping = (
        baseline_clipped + 0.01 < selected_clipped
    )
    runtime_expensive = (
        float(selected.runtime_sec) > float(cfg.fallback_runtime_ratio) * max(float(baseline.runtime_sec), 1e-9)
        and baseline_score <= selected_score + 2.0 * float(cfg.fallback_score_tolerance)
    )
    safety_fallback = baseline_is_comparable and baseline_remedies_safety and not selected_has_clear_quality_gain
    return bool(safety_fallback or baseline_has_less_clipping or runtime_expensive)


def _weak_moment_instability_fallback(
    *,
    selected: CandidateResult,
    baseline: CandidateResult,
    cfg: OccupancyTuningConfig,
) -> bool:
    """One-SE fallback from an unstable candidate to the stable baseline.

    This guard is intentionally narrow: it only fires when the selected
    candidate fails to improve held-out moment balance beyond sampling noise or
    a small practical tolerance, and the selected candidate is clearly less
    stable by ESS and weight CV. A near-uniform stable baseline under policy
    shift is treated as possible collapse and does not trigger the guard.
    """

    balance_metric = "selection_risk" if str(cfg.score_method) == "bellman_gmm" else "moment_balance"
    balance_se_metric = "selection_risk_se" if str(cfg.score_method) == "bellman_gmm" else "moment_balance_se"
    selected_mb = _metric_value(selected, balance_metric)
    baseline_mb = _metric_value(baseline, balance_metric)
    if not (np.isfinite(selected_mb) and np.isfinite(baseline_mb)):
        return False
    improvement = baseline_mb - selected_mb
    selected_se = _metric_value(selected, balance_se_metric)
    baseline_se = _metric_value(baseline, balance_se_metric)
    pooled_se = 0.0
    if np.isfinite(selected_se) and np.isfinite(baseline_se):
        pooled_se = float(np.sqrt(selected_se**2 + baseline_se**2))
    practical_margin = float(cfg.fallback_moment_balance_tolerance) * max(abs(baseline_mb), 1e-8)
    decisive_margin = max(pooled_se, practical_margin)
    if improvement > decisive_margin:
        return False

    selected_ess = _metric_value(selected, "ess_fraction", prefer_final=True)
    baseline_ess = _metric_value(baseline, "ess_fraction", prefer_final=True)
    selected_cv = _metric_value(selected, "weight_cv", prefer_final=True)
    baseline_cv = _metric_value(baseline, "weight_cv", prefer_final=True)
    if not all(np.isfinite(value) for value in (selected_ess, baseline_ess, selected_cv, baseline_cv)):
        return False
    baseline_action_shift = _metric_value(baseline, "action_shift", prefer_final=True)
    baseline_collapse = (
        np.isfinite(baseline_action_shift)
        and baseline_action_shift >= 0.05
        and baseline_ess >= 0.995
        and baseline_cv <= 0.05
    )
    if baseline_collapse:
        return False
    ess_much_worse = (
        baseline_ess - selected_ess >= float(cfg.fallback_ess_abs_drop)
        and selected_ess <= float(cfg.fallback_ess_ratio) * max(baseline_ess, 0.0)
    )
    cv_much_worse = (
        selected_cv - baseline_cv >= float(cfg.fallback_cv_abs_increase)
        and selected_cv >= float(cfg.fallback_cv_ratio) * max(baseline_cv, 1e-8)
    )
    return bool(ess_much_worse and cv_much_worse)


def _metric_value(candidate: CandidateResult, name: str, *, prefer_final: bool = False) -> float:
    if prefer_final:
        value = candidate.metrics.get(f"final_{name}")
        if value is not None and np.isfinite(float(value)):
            return float(value)
    value = candidate.metrics.get(name)
    return float(value) if value is not None else float("nan")


def _final_weight_metrics(weights: Array, occupancy: Any, *, action_shift: float = 0.0) -> Dict[str, float]:
    x = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "weight_quality": _weight_quality_from_values(x, occupancy, action_shift=action_shift),
        "ess_fraction": _ess_fraction(x),
        "norm_error": abs(float(np.mean(x)) - 1.0) if x.size else float("inf"),
        "p99": _quantile_or_nan(x, 0.99),
        "max_weight": float(np.max(x)) if x.size else float("nan"),
        "clipped_fraction": _clipped_fraction(x, occupancy),
        "action_shift": float(action_shift) if np.isfinite(float(action_shift)) else float("nan"),
        "weight_cv": _weight_cv(x),
        "n_weights": float(x.size),
    }


def _final_refit_penalty(metrics: Dict[str, float]) -> float:
    penalty = 2.00 * float(metrics.get("weight_quality", 0.0))
    ess = float(metrics.get("ess_fraction", 1.0))
    clipped = float(metrics.get("clipped_fraction", 0.0))
    if np.isfinite(ess):
        penalty += 2.0 * _ess_catastrophe_penalty(ess, n_weights=metrics.get("n_weights"))
    if np.isfinite(clipped):
        penalty += max(0.0, clipped - 0.05) * 3.0
    return float(penalty)


def _make_candidates(
    space: OccupancySearchSpace,
    cfg: OccupancyTuningConfig,
    *,
    has_initial_states: bool,
    has_initial_actions: bool = False,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    include_mode_variants = True
    for family in tuple(str(family) for family in cfg.families):
        raw = (
            list(space.boosted_candidates)
            if family == "boosted" and space.boosted_candidates is not None
            else list(space.neural_candidates)
            if family == "neural" and space.neural_candidates is not None
            else _default_family_candidates(
                family,
                space,
                has_initial_states=has_initial_states,
                include_mode_variants=include_mode_variants,
            )
        )
        if family == "neural" and bool(cfg.include_google_dualdice) and has_initial_states and has_initial_actions:
            raw = _with_google_dualdice_candidate(raw)
        capped = _cap_candidates(raw, _budget_candidate_limit(cfg), seed=int(cfg.seed) + (0 if family == "boosted" else 4_997))
        for idx, overrides in enumerate(capped):
            normalized = _normalize_overrides(overrides)
            default_idx = idx - sum(
                1
                for previous in capped[:idx]
                if str(_normalize_overrides(previous).get("backend", {}).get("name", "")) != ""
            )
            candidates.append(
                {
                    "candidate_id": f"{family}_{idx:03d}",
                    "candidate_label": _candidate_label(
                        family=family,
                        idx=default_idx,
                        overrides=normalized,
                        has_initial_states=has_initial_states,
                        include_mode_variants=include_mode_variants,
                    ),
                    "family": family,
                    "overrides": normalized,
                }
            )
    return candidates


def _candidate_label(
    *,
    family: str,
    idx: int,
    overrides: Dict[str, Dict[str, Any]],
    has_initial_states: bool,
    include_mode_variants: bool = True,
) -> str:
    backend = str(overrides.get("backend", {}).get("name", ""))
    if backend:
        return backend
    if family == "neural":
        labels = [
            "neural_stable",
            "neural_google_parity",
            "neural_relaxed_tail",
            "neural_logistic_nuisance",
        ]
        if has_initial_states and include_mode_variants:
            labels.append("neural_factored_source")
        labels.extend(
            [
                "neural_small_width",
                "neural_large_width",
                "neural_tight_cap",
                "neural_loose_nuisance_cap",
                "neural_tight_nuisance_cap",
                "neural_tight_cap_logistic_nuisance",
            ]
        )
        return labels[idx] if idx < len(labels) else f"neural_candidate_{idx:03d}"
    if family == "boosted":
        return "boosted_stable" if idx == 0 else f"boosted_candidate_{idx:03d}"
    return f"{family}_candidate_{idx:03d}"


def _stagewise_occupancy_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop occupancy-stage duplicates created only by nuisance/mode overrides."""

    nuisance_keys = {"action_ratio", "transition_ratio", "source_state_ratio", "direct_one_step_ratio"}
    out: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        overrides = dict(candidate.get("overrides", {}))
        backend = str(overrides.get("backend", {}).get("name", ""))
        if backend == "google_dualdice":
            key_payload = overrides
        else:
            key_payload = {key: value for key, value in overrides.items() if key not in nuisance_keys}
        key = (str(candidate.get("family", "")), repr(key_payload))
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _with_google_dualdice_candidate(candidates: Sequence[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Dict[str, Any]]]:
    google_candidate = {
        "backend": {"name": "google_dualdice"},
        "google_dualdice": {
            "prediction_max": 50.0,
            "normalize_predictions": True,
        },
    }
    rows = list(candidates)
    if any(row.get("backend", {}).get("name") == "google_dualdice" for row in rows):
        return rows
    insert_at = min(2, len(rows))
    return rows[:insert_at] + [google_candidate] + rows[insert_at:]


def _default_family_candidates(
    family: str,
    space: OccupancySearchSpace,
    *,
    has_initial_states: bool,
    include_mode_variants: bool,
) -> List[Dict[str, Dict[str, Any]]]:
    if family == "boosted":
        return _boosted_default_candidates(
            has_initial_states=has_initial_states,
            include_mode_variants=include_mode_variants,
        )
    return _neural_default_candidates(
        space,
        has_initial_states=has_initial_states,
        include_mode_variants=include_mode_variants,
    )


def _boosted_default_candidates(
    *,
    has_initial_states: bool,
    include_mode_variants: bool = True,
) -> List[Dict[str, Dict[str, Any]]]:
    base_occ = {
        "loss": "huber",
        "huber_delta_scale": 1.345,
        "fixed_point_damping": 0.5,
        "occupancy_ratio_max": 50.0,
        "normalize_transition_cache": False,
        "normalize_occupancy": True,
        "clip_pseudo_outcomes": True,
    }
    base_nuisance = {
        "density_ratio_loss": "lsif",
        "prediction_max": 50.0,
        "moment_calibration": "none",
    }
    variants = [
        ({}, {}),
        ({"occupancy_ratio_max": 25.0}, {}),
        ({"occupancy_ratio_max": 100.0}, {}),
        ({"fixed_point_damping": 0.35}, {}),
        ({"fixed_point_damping": 0.75}, {}),
        ({"fixed_point_damping": 0.65, "normalize_transition_cache": True}, {}),
        ({"normalize_transition_cache": True}, {}),
        ({"huber_delta_scale": 1.0}, {}),
        ({"huber_delta_scale": 2.5}, {}),
        ({}, {"prediction_max": 25.0}),
        ({}, {"prediction_max": 100.0}),
        ({}, {"density_ratio_loss": "logistic"}),
        ({}, {"moment_calibration": "scalar"}),
        ({}, {"density_ratio_loss": "logistic", "moment_calibration": "scalar"}),
        ({"occupancy_ratio_max": 25.0, "normalize_transition_cache": True}, {}),
        ({"fixed_point_damping": 0.35, "occupancy_ratio_max": 25.0}, {}),
    ]
    candidates = [
        _candidate_from_parts(base_occ, base_nuisance, occ_over, nuisance_over, has_initial_states)
        for occ_over, nuisance_over in variants
    ]
    if has_initial_states and include_mode_variants:
        candidates.append(
            _candidate_from_parts(
                base_occ,
                base_nuisance,
                {},
                {},
                has_initial_states,
                mode_overrides={"initial_ratio_mode": "factored", "one_step_ratio_mode": "factored"},
            )
        )
    return candidates


def _neural_default_candidates(
    space: OccupancySearchSpace,
    *,
    has_initial_states: bool,
    include_mode_variants: bool = True,
) -> List[Dict[str, Dict[str, Any]]]:
    base_dims = tuple(int(width) for width in space.neural_occupancy.hidden_dims)
    width = max(8, int(base_dims[0]))
    small_dims = tuple(max(8, width // 2) for _ in base_dims)
    large_dims = tuple(width * 2 for _ in base_dims)
    google_dims = tuple(max(256, 4 * width) for _ in base_dims)
    base_lr = float(space.neural_occupancy.learning_rate)
    base_occ = {
        "loss": "huber",
        "hidden_dims": base_dims,
        "learning_rate": base_lr,
        "activation": str(space.neural_occupancy.activation),
        "fixed_point_damping": 0.5,
        "occupancy_ratio_max": 50.0,
        "normalize_transition_cache": False,
        "normalize_occupancy": True,
        "clip_pseudo_outcomes": True,
        "pseudo_outcome_upper_quantile": 0.995,
        "direct_one_step_density_ratio_loss": "lsif",
        "direct_one_step_prediction_max": 50.0,
        "direct_one_step_moment_calibration": "scalar",
    }
    base_nuisance = {
        "hidden_dims": base_dims,
        "learning_rate": float(space.neural_action_ratio.learning_rate),
        "activation": str(space.neural_action_ratio.activation),
        "density_ratio_loss": "lsif",
        "prediction_max": 50.0,
        "moment_calibration": "scalar",
    }
    variants = [
        # High-value Gym/FORI presets first so fast AutoML sees them.
        ({}, {}),
        (
            {"hidden_dims": google_dims, "activation": "relu"},
            {"hidden_dims": google_dims, "activation": "relu"},
        ),
        ({"fixed_point_damping": 0.75, "occupancy_ratio_max": 100.0, "pseudo_outcome_upper_quantile": 0.999}, {"prediction_max": 100.0}),
        ({}, {"density_ratio_loss": "logistic"}),
    ]
    if has_initial_states and include_mode_variants:
        variants.append(({}, {}, {"initial_ratio_mode": "factored", "one_step_ratio_mode": "factored"}))
    variants.extend([
        ({"hidden_dims": small_dims}, {"hidden_dims": small_dims}),
        ({"hidden_dims": large_dims}, {"hidden_dims": large_dims}),
        ({"occupancy_ratio_max": 25.0}, {}),
        ({}, {"prediction_max": 100.0}),
        ({}, {"prediction_max": 25.0}),
        ({"occupancy_ratio_max": 25.0}, {"density_ratio_loss": "logistic"}),
    ])
    candidates = []
    for variant in variants:
        occ_over, nuisance_over, *maybe_modes = variant
        candidates.append(
            _candidate_from_parts(
                base_occ,
                base_nuisance,
                occ_over,
                nuisance_over,
                has_initial_states,
                mode_overrides=maybe_modes[0] if maybe_modes else None,
            )
        )
    return candidates


def _candidate_from_parts(
    base_occupancy: Dict[str, Any],
    base_nuisance: Dict[str, Any],
    occupancy_overrides: Dict[str, Any],
    nuisance_overrides: Dict[str, Any],
    has_initial_states: bool,
    mode_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    occupancy = {**base_occupancy, **occupancy_overrides}
    nuisance = {**base_nuisance, **nuisance_overrides}
    candidate = {
        "occupancy": occupancy,
        "action_ratio": nuisance,
        "transition_ratio": nuisance,
    }
    if has_initial_states:
        candidate["source_state_ratio"] = nuisance
    if mode_overrides:
        candidate["modes"] = dict(mode_overrides)
    return candidate


def _cap_candidates(candidates: Sequence[Dict[str, Dict[str, Any]]], max_candidates: int, *, seed: int) -> List[Dict[str, Dict[str, Any]]]:
    if len(candidates) <= max_candidates:
        return list(candidates)
    return list(candidates[: int(max_candidates)])


def _budget_candidate_limit(cfg: OccupancyTuningConfig) -> int:
    limit = int(cfg.max_candidates)
    if str(cfg.budget) == "fast":
        return max(1, min(limit, 8))
    return limit


def _budget_promotion_limit(cfg: OccupancyTuningConfig) -> int:
    limit = int(cfg.promotion_candidates)
    if str(cfg.budget) == "fast":
        return max(1, min(limit, 3))
    return limit


def _budget_screen_fraction(cfg: OccupancyTuningConfig) -> float:
    fraction = float(cfg.screen_fraction)
    if str(cfg.budget) == "fast":
        return min(fraction, 0.30)
    return fraction


def _normalize_overrides(overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in dict(overrides).items():
        if value is None:
            continue
        out[str(key)] = dict(value)
    return out


def _evaluate_candidate(
    *,
    candidate: Dict[str, Any],
    budget_stage: str,
    screen_fraction: float,
    folds: Sequence[Array],
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    gamma: float,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    A_pi_next: Optional[Array],
    rewards: Optional[Array],
    space: OccupancySearchSpace,
    cfg: OccupancyTuningConfig,
    seed: int,
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
    first_stage: Optional[Dict[str, Any]] = None,
    moment_block_cache: Optional[MomentBlockCache] = None,
) -> CandidateResult:
    fold_results: List[FoldResult] = []
    start = time.perf_counter()
    family = str(candidate["family"])
    overrides = dict(candidate["overrides"])
    error = ""
    try:
        explicit_modes = bool(overrides.get("modes"))
        uses_stagewise = (
            first_stage is not None
            and str(overrides.get("backend", {}).get("name", "")) != "google_dualdice"
            and not explicit_modes
        )
        fold_initial_mode = (
            str(first_stage["selected_initial_ratio_mode"])
            if uses_stagewise
            else _candidate_mode(overrides, "initial_ratio_mode", initial_ratio_mode)
        )
        fold_one_step_mode = (
            str(first_stage["selected_one_step_ratio_mode"])
            if uses_stagewise
            else _candidate_mode(overrides, "one_step_ratio_mode", one_step_ratio_mode)
        )
        for fold_id, valid_idx in enumerate(folds):
            train_idx = _complement_indices(S.shape[0], valid_idx)
            configs = _build_configs(
                family=family,
                overrides=overrides,
                space=space,
                screen_fraction=screen_fraction,
                seed=seed + 10_003 * (fold_id + 1),
            )
            if uses_stagewise:
                configs = _configs_with_first_stage(configs, first_stage)
            fold_start = time.perf_counter()
            model = _fit_family(
                family=family,
                configs=configs,
                states=S[train_idx],
                actions=A[train_idx],
                next_states=S_next[train_idx],
                target_actions=A_pi[train_idx],
                gamma=gamma,
                initial_states=_fold_initial_states(S_initial, train_idx, S.shape[0]),
                initial_actions=_fold_initial_states(A_initial, train_idx, S.shape[0]),
                initial_weights=_fold_initial_weights(initial_weights, S_initial, train_idx, S.shape[0]),
                target_next_actions=None if A_pi_next is None else A_pi_next[train_idx],
                initial_ratio_mode=fold_initial_mode,
                one_step_ratio_mode=fold_one_step_mode,
                prefit_nuisance=None if not uses_stagewise else first_stage["fold_bundles"][fold_id],
            )
            weights = model.predict_state_action_ratio(S[valid_idx], A[valid_idx], clip=True)
            moment_metrics = _heldout_moment_balance_metrics(
                weights=weights,
                train_idx=train_idx,
                valid_idx=valid_idx,
                S=S,
                A=A,
                S_next=S_next,
                A_pi=A_pi,
                A_pi_next=A_pi_next,
                S_initial=S_initial,
                A_initial=A_initial,
                rewards=rewards,
                gamma=float(gamma),
                seed=seed + 50_021 * (fold_id + 1),
                cfg=cfg,
                fold_id=fold_id,
                moment_block_cache=moment_block_cache,
            )
            validation_loss = _best_history_loss(getattr(model, "history", []))
            reward_value = None
            if rewards is not None:
                reward_value = float(np.mean(weights * rewards[valid_idx]))
            fold_results.append(
                FoldResult(
                    candidate_id=str(candidate["candidate_id"]),
                    family=family,
                    budget_stage=budget_stage,
                    fold=int(fold_id),
                    runtime_sec=float(time.perf_counter() - fold_start),
                    moment_balance=float(moment_metrics["moment_balance"]),
                    moment_balance_max_group=float(moment_metrics["moment_balance_max_group"]),
                    validation_loss=validation_loss,
                    norm_error=abs(float(np.mean(weights)) - 1.0),
                    ess_fraction=_ess_fraction(weights),
                    p99=_quantile_or_nan(weights, 0.99),
                    max_weight=float(np.max(weights)) if weights.size else float("nan"),
                    clipped_fraction=_clipped_fraction(weights, _scoring_weight_config(configs)),
                    action_shift=_action_shift(A[valid_idx], A_pi[valid_idx]),
                    weight_cv=_weight_cv(weights),
                    reward_value=reward_value,
                    moment_balance_targeted=float(moment_metrics.get("moment_balance_targeted", float("nan"))),
                    moment_balance_broad=float(moment_metrics.get("moment_balance_broad", float("nan"))),
                    moment_balance_targeted_max_group=float(moment_metrics.get("moment_balance_targeted_max_group", float("nan"))),
                    moment_balance_broad_max_group=float(moment_metrics.get("moment_balance_broad_max_group", float("nan"))),
                    moment_balance_mass=float(moment_metrics.get("moment_balance_group_mass", float("nan"))),
                    moment_balance_reward=float(moment_metrics.get("moment_balance_group_reward", float("nan"))),
                    moment_balance_value=float(moment_metrics.get("moment_balance_group_value", float("nan"))),
                    moment_balance_value_strata=float(moment_metrics.get("moment_balance_group_value_strata", float("nan"))),
                    moment_balance_geometry=float(moment_metrics.get("moment_balance_group_geometry", float("nan"))),
                    moment_balance_rff=float(moment_metrics.get("moment_balance_group_rff", float("nan"))),
                    moment_balance_rff_multiscale=float(moment_metrics.get("moment_balance_group_rff_multiscale", float("nan"))),
                    selection_risk=float(moment_metrics["selection_risk"]),
                    selection_risk_raw=float(moment_metrics["selection_risk_raw"]),
                    selection_effective_dim=float(moment_metrics["selection_effective_dim"]),
                    selection_complexity_penalty=float(moment_metrics["selection_complexity_penalty"]),
                )
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime = float(time.perf_counter() - start)
    metrics = _aggregate_fold_metrics(fold_results, runtime_sec=runtime)
    backend = str(overrides.get("backend", {}).get("name", ""))
    if backend:
        metrics[f"backend_{backend}"] = 1.0
    return CandidateResult(
        candidate_id=str(candidate["candidate_id"]),
        candidate_label=str(candidate.get("candidate_label", candidate["candidate_id"])),
        family=family,
        budget_stage=budget_stage,
        overrides=overrides,
        fold_results=fold_results,
        metrics=metrics,
        runtime_sec=runtime,
        error=error,
    )


def _build_configs(
    *,
    family: str,
    overrides: Dict[str, Dict[str, Any]],
    space: OccupancySearchSpace,
    screen_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    if family == "boosted":
        occupancy = replace(space.boosted_occupancy, **overrides.get("occupancy", {}))
        action = replace(space.boosted_action_ratio, **overrides.get("action_ratio", {}))
        source = replace(space.boosted_source_state_ratio, **overrides.get("source_state_ratio", {}))
        transition = replace(space.boosted_transition_ratio, **overrides.get("transition_ratio", {}))
        occupancy = replace(
            occupancy,
            num_iterations=max(1, int(round(float(occupancy.num_iterations) * float(screen_fraction)))),
            mcmc_samples=max(1, int(round(float(occupancy.mcmc_samples) * float(screen_fraction)))),
            seed=int(seed),
            show_progress=False,
        )
        action = replace(
            action,
            num_boost_round=max(1, int(round(float(action.num_boost_round) * float(screen_fraction)))),
            show_progress=False,
        )
        source = replace(
            source,
            num_boost_round=max(1, int(round(float(source.num_boost_round) * float(screen_fraction)))),
            show_progress=False,
        )
        transition = replace(
            transition,
            num_boost_round=max(1, int(round(float(transition.num_boost_round) * float(screen_fraction)))),
            permutation_samples=max(1, int(round(float(transition.permutation_samples) * float(screen_fraction)))),
            show_progress=False,
        )
        return {
            "occupancy": occupancy,
            "action_ratio": action,
            "source_state_ratio": source,
            "transition_ratio": transition,
        }
    occupancy = replace(space.neural_occupancy, **overrides.get("occupancy", {}))
    action = replace(space.neural_action_ratio, **overrides.get("action_ratio", {}))
    source = replace(space.neural_source_state_ratio, **overrides.get("source_state_ratio", {}))
    transition = replace(space.neural_transition_ratio, **overrides.get("transition_ratio", {}))
    google = replace(space.google_dualdice, **overrides.get("google_dualdice", {}))
    occupancy = replace(
        occupancy,
        num_iterations=max(1, int(round(float(occupancy.num_iterations) * float(screen_fraction)))),
        gradient_steps_per_iteration=max(1, int(round(float(occupancy.gradient_steps_per_iteration) * float(screen_fraction)))),
        mcmc_samples=max(1, int(round(float(occupancy.mcmc_samples) * float(screen_fraction)))),
        seed=int(seed),
        show_progress=False,
    )
    action = replace(action, max_steps=max(1, int(round(float(action.max_steps) * float(screen_fraction)))), seed=int(seed) + 101)
    source = replace(source, max_steps=max(1, int(round(float(source.max_steps) * float(screen_fraction)))), seed=int(seed) + 103)
    transition = replace(
        transition,
        max_steps=max(1, int(round(float(transition.max_steps) * float(screen_fraction)))),
        permutation_samples=max(1, int(round(float(transition.permutation_samples) * float(screen_fraction)))),
        seed=int(seed) + 107,
    )
    google = replace(
        google,
        num_updates=max(1, int(round(float(google.num_updates) * float(screen_fraction)))),
        seed=int(seed) + 109,
    )
    return {
        "backend": str(overrides.get("backend", {}).get("name", "iterative_neural")),
        "occupancy": occupancy,
        "action_ratio": action,
        "source_state_ratio": source,
        "transition_ratio": transition,
        "google_dualdice": google,
    }


def _candidate_mode(overrides: Dict[str, Dict[str, Any]], name: str, default: str) -> str:
    modes = overrides.get("modes", {})
    if not modes:
        return str(default)
    return str(modes.get(name, default))


def _configs_with_first_stage(configs: Dict[str, Any], first_stage: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(configs)
    selected = dict(first_stage.get("selected_configs", {}))
    for key in ("action_ratio", "source_state_ratio", "transition_ratio"):
        if key in selected:
            out[key] = selected[key]
    return out


def _fit_family(
    *,
    family: str,
    configs: Dict[str, Any],
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Array,
    gamma: float,
    initial_states: Optional[Array],
    initial_actions: Optional[Array],
    initial_weights: Optional[Array],
    target_next_actions: Optional[Array],
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
    prefit_nuisance: Optional[Dict[str, Any]] = None,
) -> DiscountedOccupancyRatioModel | NeuralDiscountedOccupancyRatioModel | Any:
    if str(configs.get("backend", "")) == "google_dualdice":
        if initial_states is None or initial_actions is None:
            raise ValueError("Google DualDICE tuning candidates require initial_states and initial_actions.")
        google_target_next_actions = target_next_actions if target_next_actions is not None else target_actions
        return fit_google_dualdice_occupancy_ratio(
            states=states,
            actions=actions,
            next_states=next_states,
            target_actions=target_actions,
            target_next_actions=google_target_next_actions,
            gamma=gamma,
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            config=configs["google_dualdice"],
        )
    fit = fit_discounted_occupancy_ratio if family == "boosted" else fit_discounted_occupancy_ratio_neural
    return fit(
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        gamma=gamma,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        target_next_actions=target_next_actions,
        initial_ratio_mode=initial_ratio_mode,
        one_step_ratio_mode=one_step_ratio_mode,
        occupancy=configs["occupancy"],
        action_ratio=configs["action_ratio"],
        source_state_ratio=configs["source_state_ratio"],
        transition_ratio=configs["transition_ratio"],
        _prefit_nuisance=prefit_nuisance,
    )


def _scoring_weight_config(configs: Dict[str, Any]) -> Any:
    if str(configs.get("backend", "")) == "google_dualdice":
        return configs.get("google_dualdice", configs["occupancy"])
    return configs["occupancy"]


def _aggregate_fold_metrics(folds: Sequence[FoldResult], *, runtime_sec: float) -> Dict[str, float]:
    if not folds:
        return {
            "moment_balance": float("inf"),
            "moment_balance_max_group": float("inf"),
            "moment_balance_se": float("inf"),
            "moment_balance_targeted": float("inf"),
            "moment_balance_broad": float("inf"),
            "moment_balance_targeted_max_group": float("inf"),
            "moment_balance_broad_max_group": float("inf"),
            "validation_loss": float("inf"),
            "reward_stability": float("nan"),
            "weight_quality": float("inf"),
            "selection_risk": float("inf"),
            "selection_risk_raw": float("inf"),
            "selection_risk_se": float("inf"),
            "selection_effective_dim": 0.0,
            "selection_complexity_penalty": 0.0,
            "runtime_sec": float(runtime_sec),
        }
    moment_balance = _mean([fold.moment_balance for fold in folds])
    moment_max_group = _mean([fold.moment_balance_max_group for fold in folds])
    moment_balance_se = _se([fold.moment_balance for fold in folds])
    moment_balance_targeted = _mean([fold.moment_balance_targeted for fold in folds])
    moment_balance_broad = _mean([fold.moment_balance_broad for fold in folds])
    moment_targeted_max_group = _mean([fold.moment_balance_targeted_max_group for fold in folds])
    moment_broad_max_group = _mean([fold.moment_balance_broad_max_group for fold in folds])
    validation = _mean([fold.validation_loss for fold in folds])
    ess = _mean([fold.ess_fraction for fold in folds])
    norm = _mean([fold.norm_error for fold in folds])
    p99 = _mean([fold.p99 for fold in folds])
    clipped = _mean([fold.clipped_fraction for fold in folds])
    action_shift = _mean([fold.action_shift for fold in folds])
    weight_cv = _mean([fold.weight_cv for fold in folds])
    reward_values = [fold.reward_value for fold in folds if fold.reward_value is not None and np.isfinite(float(fold.reward_value))]
    reward_stability = float("nan")
    if len(reward_values) >= 2:
        reward_stability = float(np.std(reward_values) / (abs(float(np.mean(reward_values))) + 1e-12))
    selection_risk_values = [fold.selection_risk for fold in folds]
    weight_quality = float(
        norm
        + _ess_catastrophe_penalty(ess)
        + clipped
        + _near_uniform_penalty(ess, action_shift, weight_cv)
    )
    return {
        "moment_balance": moment_balance,
        "moment_balance_max_group": moment_max_group,
        "moment_balance_se": moment_balance_se,
        "moment_balance_targeted": moment_balance_targeted,
        "moment_balance_broad": moment_balance_broad,
        "moment_balance_targeted_max_group": moment_targeted_max_group,
        "moment_balance_broad_max_group": moment_broad_max_group,
        "moment_balance_mass": _mean([fold.moment_balance_mass for fold in folds]),
        "moment_balance_reward": _mean([fold.moment_balance_reward for fold in folds]),
        "moment_balance_value": _mean([fold.moment_balance_value for fold in folds]),
        "moment_balance_value_strata": _mean([fold.moment_balance_value_strata for fold in folds]),
        "moment_balance_geometry": _mean([fold.moment_balance_geometry for fold in folds]),
        "moment_balance_rff": _mean([fold.moment_balance_rff for fold in folds]),
        "moment_balance_rff_multiscale": _mean([fold.moment_balance_rff_multiscale for fold in folds]),
        "validation_loss": validation,
        "reward_stability": reward_stability,
        "selection_risk": _mean(selection_risk_values),
        "selection_risk_raw": _mean([fold.selection_risk_raw for fold in folds]),
        "selection_risk_se": _se(selection_risk_values),
        "selection_effective_dim": _mean([fold.selection_effective_dim for fold in folds]),
        "selection_complexity_penalty": _mean([fold.selection_complexity_penalty for fold in folds]),
        "weight_quality": weight_quality,
        "ess_fraction": ess,
        "norm_error": norm,
        "p99": p99,
        "clipped_fraction": clipped,
        "action_shift": action_shift,
        "weight_cv": weight_cv,
        "runtime_sec": float(runtime_sec),
    }


def _score_candidates(candidates: Sequence[CandidateResult], cfg: OccupancyTuningConfig) -> None:
    if str(cfg.score_method) == "bellman_gmm":
        _score_candidates_bellman_gmm(candidates, cfg)
        return
    if str(cfg.score_method) == "validation_loss":
        _score_candidates_validation_loss(candidates)
        return
    finite_candidates = [candidate for candidate in candidates if not candidate.error and candidate.fold_results]
    if not finite_candidates:
        return
    reward_available = any(np.isfinite(candidate.metrics.get("reward_stability", float("nan"))) for candidate in finite_candidates)
    moment_available = any(np.isfinite(candidate.metrics.get("moment_balance", float("nan"))) for candidate in finite_candidates)
    moment_weight = float(cfg.score_moment_balance_weight) if moment_available else 0.0
    validation_weight = float(cfg.score_validation_weight)
    reward_weight = float(cfg.score_reward_stability_weight) if reward_available else 0.0
    if not reward_available:
        validation_weight += float(cfg.score_reward_stability_weight)
    if not moment_available:
        validation_weight += float(cfg.score_moment_balance_weight)
    weights = {
        "moment_balance": moment_weight,
        "validation_loss": validation_weight,
        "reward_stability": reward_weight,
        "weight_quality": float(cfg.score_weight_quality_weight),
        "runtime_sec": float(cfg.score_runtime_weight),
    }
    ranks = {
        metric: _rank01([candidate.metrics.get(metric, float("inf")) for candidate in finite_candidates])
        for metric, weight in weights.items()
        if weight > 0.0
    }
    total_weight = sum(weights[metric] for metric in ranks)
    for idx, candidate in enumerate(finite_candidates):
        candidate.score = float(sum(weights[metric] * ranks[metric][idx] for metric in ranks) / total_weight)
    for candidate in candidates:
        if candidate not in finite_candidates:
            candidate.score = float("inf")


def _score_candidates_validation_loss(candidates: Sequence[CandidateResult]) -> None:
    for candidate in candidates:
        if candidate.error or not candidate.fold_results:
            candidate.score = float("inf")
            continue
        loss = float(candidate.metrics.get("validation_loss", float("inf")))
        candidate.metrics["score_method_validation_loss"] = 1.0
        candidate.score = loss if np.isfinite(loss) else float("inf")


def _score_candidates_bellman_gmm(candidates: Sequence[CandidateResult], cfg: OccupancyTuningConfig) -> None:
    finite_candidates = [candidate for candidate in candidates if not candidate.error and candidate.fold_results]
    if not finite_candidates:
        return
    constraint_flags = {candidate.candidate_id: _gmm_safety_constraint_flags(candidate.metrics) for candidate in finite_candidates}
    use_constraints = bool(cfg.gmm_use_safety_constraints)
    feasible_ids = [
        candidate.candidate_id
        for candidate in finite_candidates
        if not use_constraints or not constraint_flags[candidate.candidate_id]["any"]
    ]
    all_violated = bool(use_constraints and not feasible_ids)
    for candidate in finite_candidates:
        risk = float(candidate.metrics.get("selection_risk", float("inf")))
        flags = constraint_flags[candidate.candidate_id]
        candidate.metrics["score_method_bellman_gmm"] = 1.0
        candidate.metrics["gmm_objective_ope"] = float(str(cfg.gmm_objective) == "ope")
        candidate.metrics["gmm_ope_broad_weight"] = float(cfg.gmm_ope_broad_weight)
        candidate.metrics["constraint_violated"] = float(flags["any"]) if use_constraints else 0.0
        candidate.metrics["constraint_nonfinite_risk"] = float(flags["nonfinite_risk"]) if use_constraints else 0.0
        candidate.metrics["constraint_catastrophic_ess"] = float(flags["catastrophic_ess"]) if use_constraints else 0.0
        candidate.metrics["constraint_clipping"] = float(flags["clipping"]) if use_constraints else 0.0
        candidate.metrics["constraint_normalization"] = float(flags["normalization"]) if use_constraints else 0.0
        candidate.metrics["constraint_near_uniform_collapse"] = float(flags["near_uniform_collapse"]) if use_constraints else 0.0
        candidate.metrics["constraint_all_violated"] = float(all_violated)
        if not np.isfinite(risk):
            candidate.score = float("inf")
        elif use_constraints and flags["any"] and not all_violated:
            candidate.score = float("inf")
        else:
            candidate.score = risk
    for candidate in candidates:
        if candidate not in finite_candidates:
            candidate.score = float("inf")


def _gmm_safety_constraint_flags(metrics: Dict[str, Any]) -> Dict[str, bool]:
    risk = _metric_from_dict(metrics, "selection_risk")
    ess = _metric_from_dict(metrics, "ess_fraction")
    clipped = _metric_from_dict(metrics, "clipped_fraction")
    norm = _metric_from_dict(metrics, "norm_error")
    action_shift = _metric_from_dict(metrics, "action_shift")
    weight_cv = _metric_from_dict(metrics, "weight_cv")
    n_weights = metrics.get("n_weights")
    flags = {
        "nonfinite_risk": not np.isfinite(risk),
        "catastrophic_ess": _ess_is_catastrophic(ess, n_weights=n_weights),
        "clipping": bool(np.isfinite(clipped) and clipped > 0.25),
        "normalization": bool(np.isfinite(norm) and norm > 0.25),
        "near_uniform_collapse": bool(
            np.isfinite(action_shift)
            and np.isfinite(ess)
            and np.isfinite(weight_cv)
            and action_shift >= 0.10
            and ess >= 0.999
            and weight_cv <= 0.01
        ),
    }
    flags["any"] = any(flags.values())
    return flags


def _metric_from_dict(metrics: Dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rank01(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.inf)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    if arr.shape[0] <= 1:
        ranks[:] = 0.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, arr.shape[0])
    return [float(value) for value in ranks]


def _fit_reward_value_moment_blocks(cfg: OccupancyTuningConfig) -> bool:
    return not (str(cfg.score_method) == "bellman_gmm" and str(cfg.gmm_objective) == "ratio")


def _moment_extra_blocks_for_scoring(cfg: OccupancyTuningConfig) -> Tuple[str, ...]:
    if str(cfg.score_method) == "bellman_gmm":
        return ()
    return tuple(str(block) for block in cfg.moment_extra_blocks)


def _index_cache_signature(indices: Array) -> Tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(indices, dtype=np.int64).reshape(-1))


def _moment_block_cache_key(
    *,
    fold_id: int,
    seed: int,
    train_idx: Array,
    valid_idx: Array,
    cfg: OccupancyTuningConfig,
    reward_value_blocks_enabled: bool,
) -> MomentBlockCacheKey:
    return (
        int(fold_id),
        int(seed),
        _index_cache_signature(train_idx),
        _index_cache_signature(valid_idx),
        str(cfg.score_method),
        str(cfg.gmm_objective),
        int(cfg.moment_geometry_features),
        int(cfg.moment_rff_features),
        int(cfg.moment_value_iterations),
        int(cfg.moment_value_patience),
        _moment_extra_blocks_for_scoring(cfg),
        tuple(float(scale) for scale in cfg.moment_multiscale_rff_scales),
        tuple(float(quantile) for quantile in cfg.moment_strata_quantiles),
        bool(reward_value_blocks_enabled),
    )


def _heldout_moment_balance_metrics(
    *,
    weights: Array,
    train_idx: Array,
    valid_idx: Array,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    A_pi_next: Optional[Array],
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    rewards: Optional[Array],
    gamma: float,
    seed: int,
    cfg: OccupancyTuningConfig,
    fold_id: Optional[int] = None,
    moment_block_cache: Optional[MomentBlockCache] = None,
) -> Dict[str, float]:
    valid_idx = np.asarray(valid_idx, dtype=np.int64).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    next_actions = A_pi[valid_idx] if A_pi_next is None else A_pi_next[valid_idx]
    initial_states, initial_actions = _validation_initial_state_actions(
        S=S,
        A_pi=A_pi,
        valid_idx=valid_idx,
        S_initial=S_initial,
        A_initial=A_initial,
    )
    reward_value_blocks_enabled = rewards is not None and _fit_reward_value_moment_blocks(cfg)
    cache_key = None
    if moment_block_cache is not None and fold_id is not None:
        cache_key = _moment_block_cache_key(
            fold_id=int(fold_id),
            seed=int(seed),
            train_idx=train_idx,
            valid_idx=valid_idx,
            cfg=cfg,
            reward_value_blocks_enabled=reward_value_blocks_enabled,
        )
    if cache_key is not None and cache_key in moment_block_cache:
        blocks = moment_block_cache[cache_key]
    else:
        feature_builder = _FoldFeatureBuilder(
            S_train=S[train_idx],
            A_train=A[train_idx],
            rewards_train=None if not reward_value_blocks_enabled else rewards[train_idx],
            gamma=float(gamma),
            seed=int(seed),
            geometry_features=int(cfg.moment_geometry_features),
            rff_features=int(cfg.moment_rff_features),
            value_iterations=int(cfg.moment_value_iterations),
            value_patience=int(cfg.moment_value_patience),
            S_next_train=S_next[train_idx],
            A_next_train=A_pi[train_idx] if A_pi_next is None else A_pi_next[train_idx],
            A_target_train=A_pi[train_idx],
            extra_blocks=_moment_extra_blocks_for_scoring(cfg),
            multiscale_rff_scales=tuple(float(scale) for scale in cfg.moment_multiscale_rff_scales),
            strata_quantiles=tuple(float(quantile) for quantile in cfg.moment_strata_quantiles),
        )
        blocks = feature_builder.blocks(
            S_eval=S[valid_idx],
            A_eval=A[valid_idx],
            S_next=S_next[valid_idx],
            A_next=next_actions,
            S_initial=initial_states,
            A_initial=initial_actions,
        )
        if cache_key is not None:
            moment_block_cache[cache_key] = blocks
    group_scores = {
        name: _moment_group_score(weights, float(gamma), block)
        for name, block in blocks.items()
        if np.asarray(block[0]).shape[1] > 0
    }
    finite = np.asarray([score for score in group_scores.values() if np.isfinite(score)], dtype=np.float64)
    if finite.size == 0:
        legacy = {"moment_balance": float("inf"), "moment_balance_max_group": float("inf")}
    else:
        legacy = _legacy_moment_balance_metrics_from_scores(finite, cfg)
    gmm = _bellman_gmm_selection_metrics(weights=weights, gamma=float(gamma), blocks=blocks, cfg=cfg)
    return {
        **legacy,
        **{f"moment_balance_group_{name}": float(score) for name, score in group_scores.items()},
        **gmm,
    }


def _bellman_gmm_selection_metrics(
    *,
    weights: Array,
    gamma: float,
    blocks: Dict[str, Tuple[Array, Array, Array]],
    cfg: OccupancyTuningConfig,
) -> Dict[str, float]:
    block_scores: Dict[str, Dict[str, float]] = {}
    objective = str(cfg.gmm_objective)
    block_weights = _gmm_block_weights(objective, ope_broad_weight=float(cfg.gmm_ope_broad_weight))
    if objective == "ope" and not _has_ope_target_blocks(blocks):
        block_weights = _gmm_block_weights("ratio", ope_broad_weight=float(cfg.gmm_ope_broad_weight))
    n_valid = 0
    for name, block_weight in block_weights.items():
        block = blocks.get(name)
        if block is None:
            continue
        if np.asarray(block[0]).shape[1] <= 0:
            continue
        score = _moment_group_gmm_score(weights, float(gamma), block, cov_ridge=float(cfg.gmm_cov_ridge))
        if not np.isfinite(score["risk"]):
            continue
        score = dict(score)
        score["block_weight"] = float(block_weight)
        block_scores[str(name)] = score
        n_valid = max(n_valid, int(score["n_valid"]))
    if not block_scores:
        return {
            "selection_risk": float("inf"),
            "selection_risk_raw": float("inf"),
            "selection_effective_dim": 0.0,
            "selection_complexity_penalty": 0.0,
            "selection_block_count": 0.0,
        }
    total_weight = sum(float(score["block_weight"]) for score in block_scores.values())
    raw_risk = float(
        sum(float(score["block_weight"]) * float(score["risk"]) for score in block_scores.values())
        / max(total_weight, 1e-12)
    )
    total_effective_dim = float(sum(float(score["effective_dim"]) for score in block_scores.values()))
    complexity = float(max(0.0, float(cfg.gmm_complexity_weight)) * total_effective_dim / max(n_valid, 1))
    return {
        "selection_risk": float(raw_risk + complexity),
        "selection_risk_raw": raw_risk,
        "selection_effective_dim": float(total_effective_dim),
        "selection_complexity_penalty": complexity,
        "selection_block_count": float(len(block_scores)),
        "selection_ope_broad_weight": float(cfg.gmm_ope_broad_weight) if objective == "ope" else float("nan"),
        **{f"selection_group_risk_{name}": float(score["risk"]) for name, score in block_scores.items()},
    }


def _has_ope_target_blocks(blocks: Dict[str, Tuple[Array, Array, Array]]) -> bool:
    for name in ("reward", "value", "value_strata"):
        block = blocks.get(name)
        if block is not None and np.asarray(block[0]).ndim == 2 and np.asarray(block[0]).shape[1] > 0:
            return True
    return False


def _gmm_block_weights(objective: str, *, ope_broad_weight: float = 0.10) -> Dict[str, float]:
    if str(objective) == "ope":
        broad_weight = float(np.clip(float(ope_broad_weight), 0.0, 1.0))
        target_weight = 1.0 - broad_weight
        weights = {name: target_weight / 4.0 for name in ("mass", "reward", "value", "value_strata")}
        for name in ("mass", "geometry", "rff"):
            weights[name] = weights.get(name, 0.0) + broad_weight / 3.0
        return {name: weight for name, weight in weights.items() if weight > 0.0}
    return {
        "mass": 1.0,
        "geometry": 1.0,
        "rff": 1.0,
    }


def _moment_group_gmm_score(
    weights: Array,
    gamma: float,
    block: Tuple[Array, Array, Array],
    *,
    cov_ridge: float,
) -> Dict[str, float]:
    eval_features, next_features, initial_features = (np.asarray(part, dtype=np.float64) for part in block)
    if eval_features.ndim != 2 or eval_features.shape[1] == 0:
        return {"risk": float("nan"), "effective_dim": 0.0, "n_valid": 0.0}
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != eval_features.shape[0]:
        return {"risk": float("nan"), "effective_dim": 0.0, "n_valid": 0.0}
    valid_term = w[:, None] * (eval_features - float(gamma) * next_features)
    initial_term = (1.0 - float(gamma)) * initial_features
    if not (np.all(np.isfinite(valid_term)) and np.all(np.isfinite(initial_term))):
        return {"risk": float("inf"), "effective_dim": 0.0, "n_valid": float(valid_term.shape[0])}
    delta = np.mean(valid_term, axis=0) - np.mean(initial_term, axis=0)
    sigma = _mean_covariance(valid_term) / max(valid_term.shape[0], 1)
    sigma += _mean_covariance(initial_term) / max(initial_term.shape[0], 1)
    sigma = 0.5 * (sigma + sigma.T)
    diag = np.diag(sigma)
    diag_mean = float(np.mean(diag[np.isfinite(diag)])) if diag.size else 0.0
    ridge = max(1e-8, float(cov_ridge) * max(diag_mean, 0.0))
    reg = sigma + ridge * np.eye(sigma.shape[0], dtype=np.float64)
    try:
        solved_delta = np.linalg.solve(reg, delta)
        solved_sigma = np.linalg.solve(reg, sigma)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(reg)
        solved_delta = pinv @ delta
        solved_sigma = pinv @ sigma
    effective_dim = float(np.trace(solved_sigma))
    if not np.isfinite(effective_dim):
        effective_dim = 0.0
    effective_dim = max(0.0, effective_dim)
    risk = float(delta @ solved_delta / max(effective_dim, 1.0))
    if not np.isfinite(risk):
        risk = float("inf")
    return {
        "risk": risk,
        "effective_dim": effective_dim,
        "n_valid": float(valid_term.shape[0]),
    }


def _mean_covariance(x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] <= 1:
        return np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float64)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    return (centered.T @ centered) / float(arr.shape[0] - 1)


def _legacy_moment_balance_metrics_from_scores(finite: Array, cfg: OccupancyTuningConfig) -> Dict[str, float]:
    values = np.asarray(finite, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "moment_balance": float("inf"),
            "moment_balance_max_group": float("inf"),
        }
    return {
        "moment_balance": float(np.mean(values) + float(cfg.moment_max_group_weight) * np.max(values)),
        "moment_balance_max_group": float(np.max(values)),
    }


def _validate_occupancy_target_trajectory(
    *,
    validation_states: Optional[Array],
    validation_actions: Optional[Array],
    validation_rewards: Optional[Array],
    validation_episode_ids: Optional[Array],
    validation_timestep: Optional[Array],
    validation_terminals: Optional[Array],
    validation_continuation: Optional[Array],
) -> _OccupancyTargetTrajectory:
    if validation_states is None or validation_actions is None:
        raise ValueError("validation_states and validation_actions are required for discounted_moments scoring.")
    states = _as_2d(validation_states, "validation_states")
    actions = _as_2d(validation_actions, "validation_actions")
    if states.shape[0] != actions.shape[0]:
        raise ValueError("validation_states and validation_actions must have aligned rows.")
    rewards = None
    if validation_rewards is not None:
        rewards = np.asarray(validation_rewards, dtype=np.float64).reshape(-1)
        if rewards.shape[0] != states.shape[0]:
            raise ValueError("validation_rewards must have one entry per validation row.")
    episode_ids = np.arange(states.shape[0]) if validation_episode_ids is None else np.asarray(validation_episode_ids).reshape(-1)
    if episode_ids.shape[0] != states.shape[0]:
        raise ValueError("validation_episode_ids must have one entry per validation row.")
    timesteps = np.arange(states.shape[0]) if validation_timestep is None else np.asarray(validation_timestep).reshape(-1)
    if timesteps.shape[0] != states.shape[0]:
        raise ValueError("validation_timestep must have one entry per validation row.")
    continuation = None
    if validation_continuation is not None and validation_terminals is not None:
        raise ValueError("Supply only one of validation_continuation or validation_terminals.")
    if validation_continuation is not None:
        continuation = np.asarray(validation_continuation, dtype=np.float64).reshape(-1)
    elif validation_terminals is not None:
        continuation = 1.0 - np.asarray(validation_terminals, dtype=np.float64).reshape(-1)
    if continuation is not None:
        if continuation.shape[0] != states.shape[0]:
            raise ValueError("validation continuation indicators must have one entry per validation row.")
        if not np.all(np.isfinite(continuation)):
            raise ValueError("validation continuation indicators must be finite.")
        continuation = np.clip(continuation, 0.0, 1.0)
    return _OccupancyTargetTrajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        episode_ids=episode_ids,
        timesteps=timesteps,
        continuation=continuation,
    )


def _occupancy_target_trajectory_diagnostics(
    target: _OccupancyTargetTrajectory,
    *,
    gamma: float,
    seed: int,
) -> Dict[str, float | str]:
    returns = []
    tail_masses = []
    horizons = []
    for _, idx in _occupancy_episode_slices(target.episode_ids, target.timesteps):
        rel_steps = _relative_timesteps(target.timesteps[idx])
        discounts = float(gamma) ** rel_steps
        if target.rewards is not None:
            returns.append(float(np.sum(discounts * target.rewards[idx])))
        horizons.append(float(idx.shape[0]))
        continuation = 1.0 if target.continuation is None else float(target.continuation[idx[-1]])
        tail_masses.append(float((float(gamma) ** (float(np.max(rel_steps)) + 1.0)) * continuation))
    return {
        "direct_target_return_mean": _mean(returns) if returns else float("nan"),
        "direct_target_return_se": _bootstrap_scalar_mean_se(np.asarray(returns, dtype=np.float64), seed=seed) if returns else float("nan"),
        "validation_episode_count": float(len(horizons)),
        "validation_row_count": float(target.states.shape[0]),
        "validation_horizon_mean": _mean(horizons),
        "validation_horizon_max": float(np.max(horizons)) if horizons else 0.0,
        "truncation_tail_mass_mean": _mean(tail_masses),
        "truncation_tail_mass_max": float(np.max(tail_masses)) if tail_masses else 0.0,
    }


def _score_occupancy_discounted_moments(
    *,
    weights: Array,
    reference_states: Array,
    reference_actions: Array,
    reference_rewards: Optional[Array],
    target: _OccupancyTargetTrajectory,
    gamma: float,
    seed: int,
) -> Dict[str, float]:
    reward_available = reference_rewards is not None and target.rewards is not None
    ref_rewards = reference_rewards if reward_available else None
    target_rewards = target.rewards if reward_available else None
    ref_features = _occupancy_moment_features(reference_states, reference_actions, ref_rewards)
    target_features = _occupancy_moment_features(target.states, target.actions, target_rewards)
    if ref_features.shape[1] != target_features.shape[1]:
        raise ValueError("Reference and target validation moment feature dimensions do not match.")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != ref_features.shape[0]:
        raise ValueError("Candidate weights must align with reference rows.")
    reference_moment = np.mean(w[:, None] * ref_features, axis=0)
    target_row_weights = _target_discount_weights(target.episode_ids, target.timesteps, gamma=float(gamma))
    target_moment = np.sum(target_row_weights[:, None] * target_features, axis=0)
    diff = reference_moment - target_moment
    score = float(np.mean(diff * diff))
    episode_losses = []
    for _, idx in _occupancy_episode_slices(target.episode_ids, target.timesteps):
        ep_weights = _target_discount_weights(target.episode_ids[idx], target.timesteps[idx], gamma=float(gamma))
        ep_moment = np.sum(ep_weights[:, None] * target_features[idx], axis=0)
        episode_losses.append(float(np.mean((reference_moment - ep_moment) ** 2)))
    return {
        "validation_score": score,
        "validation_score_se": _bootstrap_scalar_mean_se(np.asarray(episode_losses, dtype=np.float64), seed=seed),
        "validation_moment_l2": float(np.linalg.norm(diff)),
        "validation_moment_max_abs": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "reference_weighted_reward_moment": float(reference_moment[1]) if reward_available else float("nan"),
        "target_reward_moment": float(target_moment[1]) if reward_available else float("nan"),
    }


def _occupancy_moment_features(states: Array, actions: Array, rewards: Optional[Array]) -> Array:
    states_2d = _as_2d(states, "moment_states")
    actions_2d = _as_2d(actions, "moment_actions")
    if states_2d.shape[0] != actions_2d.shape[0]:
        raise ValueError("states and actions must have aligned rows for moment features.")
    cols = [np.ones((states_2d.shape[0], 1), dtype=np.float64)]
    if rewards is not None:
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if r.shape[0] != states_2d.shape[0]:
            raise ValueError("rewards must align with moment feature rows.")
        cols.append(r.reshape(-1, 1))
    cols.extend([states_2d.astype(np.float64, copy=False), actions_2d.astype(np.float64, copy=False)])
    return np.concatenate(cols, axis=1)


def _target_discount_weights(episode_ids: Array, timesteps: Array, *, gamma: float) -> Array:
    out = np.zeros(np.asarray(episode_ids).shape[0], dtype=np.float64)
    for _, idx in _occupancy_episode_slices(episode_ids, timesteps):
        rel_steps = _relative_timesteps(np.asarray(timesteps).reshape(-1)[idx])
        out[idx] = float(gamma) ** rel_steps
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Target validation discounted weights have nonpositive mass.")
    return out / total


def _relative_timesteps(timesteps: Array) -> Array:
    steps = np.asarray(timesteps, dtype=np.float64).reshape(-1)
    if steps.size == 0:
        return steps
    return steps - float(np.min(steps))


def _occupancy_episode_slices(episode_ids: Array, timesteps: Array) -> List[Tuple[Any, Array]]:
    ids = np.asarray(episode_ids).reshape(-1)
    steps = np.asarray(timesteps).reshape(-1)
    out: List[Tuple[Any, Array]] = []
    for episode_id in np.unique(ids):
        idx = np.flatnonzero(ids == episode_id)
        order = np.argsort(steps[idx], kind="mergesort")
        out.append((episode_id, idx[order].astype(np.int64, copy=False)))
    return out


def _bootstrap_scalar_mean_se(values: Array, *, seed: int, n_bootstrap: int = 200) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.shape[0] <= 1 or int(n_bootstrap) <= 0:
        return 0.0
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sample = rng.integers(0, arr.shape[0], size=arr.shape[0])
        boot[idx] = float(np.mean(arr[sample]))
    return float(np.std(boot, ddof=1))


def _target_validation_guardrails_pass(metrics: Dict[str, float]) -> bool:
    ess = float(metrics.get("ess_fraction", float("nan")))
    clipped = float(metrics.get("clipped_fraction", float("nan")))
    quality = float(metrics.get("weight_quality", float("inf")))
    norm = float(metrics.get("norm_error", float("inf")))
    if not all(np.isfinite(value) for value in (ess, clipped, quality, norm)):
        return False
    if _ess_is_catastrophic(ess, n_weights=metrics.get("n_weights")):
        return False
    if clipped > 0.20:
        return False
    if norm > 0.50:
        return False
    if quality > 5.0:
        return False
    return True


def _select_occupancy_target_validation_candidate(
    candidates: Sequence[OccupancyTargetValidationCandidateResult],
    *,
    selection_rule: str,
) -> Tuple[
    Optional[OccupancyTargetValidationCandidateResult],
    Optional[OccupancyTargetValidationCandidateResult],
    Optional[OccupancyTargetValidationCandidateResult],
]:
    finite = [
        candidate
        for candidate in candidates
        if not candidate.error and candidate.guardrail_passed and np.isfinite(float(candidate.score))
    ]
    if not finite:
        return None, None, None
    best = min(finite, key=lambda row: row.score)
    best.selected_min_score = True
    threshold = float(best.score) + max(float(best.score_se), 0.0)
    finite_ids = {id(candidate) for candidate in finite}
    one_se = best
    for candidate in candidates:
        if id(candidate) in finite_ids and float(candidate.score) <= threshold:
            one_se = candidate
            break
    selected = best if str(selection_rule) == "min_score" else one_se
    return selected, best, one_se


def _occupancy_selected_min_candidate_id(candidates: Sequence[OccupancyTargetValidationCandidateResult]) -> str:
    finite = [
        candidate
        for candidate in candidates
        if not candidate.error and candidate.guardrail_passed and np.isfinite(float(candidate.score))
    ]
    if not finite:
        return ""
    return min(finite, key=lambda row: row.score).candidate_id


class _FoldFeatureBuilder:
    def __init__(
        self,
        *,
        S_train: Array,
        A_train: Array,
        rewards_train: Optional[Array],
        gamma: float,
        seed: int,
        geometry_features: int,
        rff_features: int,
        value_iterations: int,
        value_patience: int,
        S_next_train: Array,
        A_next_train: Array,
        A_target_train: Array,
        extra_blocks: Sequence[str],
        multiscale_rff_scales: Sequence[float],
        strata_quantiles: Sequence[float],
    ) -> None:
        S_train_2d = _as_2d(S_train, "S_train")
        A_train_2d = _as_2d(A_train, "A_train")
        self.x_train = np.concatenate([S_train_2d, A_train_2d], axis=1)
        self.mean = np.mean(self.x_train, axis=0)
        scale = np.std(self.x_train, axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        self.state_mean = np.mean(S_train_2d, axis=0)
        state_scale = np.std(S_train_2d, axis=0)
        self.state_scale = np.where(state_scale > 1e-8, state_scale, 1.0)
        self.extra_blocks = {str(block) for block in extra_blocks}
        self.strata_quantiles = np.unique(np.asarray(strata_quantiles, dtype=np.float64))
        z_train = self._std_x(S_train, A_train)
        self.geometry_components = self._geometry_components(z_train, int(geometry_features))
        rng = np.random.default_rng(int(seed))
        self.rff_count = max(0, int(rff_features))
        dim = int(z_train.shape[1])
        self.rff_W = rng.normal(size=(dim, self.rff_count)) / np.sqrt(max(dim, 1)) if self.rff_count else np.empty((dim, 0))
        self.rff_b = rng.uniform(0.0, 2.0 * np.pi, size=self.rff_count) if self.rff_count else np.empty(0)
        self.multiscale_rff_scales = tuple(
            float(scale)
            for scale in multiscale_rff_scales
            if self.rff_count and np.isfinite(float(scale)) and float(scale) > 0.0 and abs(float(scale) - 1.0) > 1e-8
        )
        self.second_order_mean: Optional[Array] = None
        self.second_order_scale: Optional[Array] = None
        self.support_mean: Optional[Array] = None
        self.support_scale: Optional[Array] = None
        self.support_bins: Optional[Array] = None
        self.support_bin_mean: Optional[Array] = None
        self.support_bin_scale: Optional[Array] = None
        self.policy_shift_theta: Optional[Array] = None
        self.policy_shift_mean = 0.0
        self.policy_shift_scale = 1.0
        self.policy_shift_bins: Optional[Array] = None
        self.policy_shift_bin_mean: Optional[Array] = None
        self.policy_shift_bin_scale: Optional[Array] = None
        self.reward_theta: Optional[Array] = None
        self.reward_mean = 0.0
        self.reward_scale = 1.0
        self.value_theta: Optional[Array] = None
        self.value_mean = 0.0
        self.value_scale = 1.0
        self.value_bins: Optional[Array] = None
        self.value_bin_mean: Optional[Array] = None
        self.value_bin_scale: Optional[Array] = None
        if "second_order" in self.extra_blocks:
            second = self._second_order_raw(z_train)
            if second.shape[1]:
                self.second_order_mean = np.mean(second, axis=0)
                scale = np.std(second, axis=0)
                self.second_order_scale = np.where(scale > 1e-8, scale, 1.0)
        if "support" in self.extra_blocks:
            support = self._support_raw(z_train)
            if support.shape[1]:
                self.support_mean = np.mean(support, axis=0)
                scale = np.std(support, axis=0)
                self.support_scale = np.where(scale > 1e-8, scale, 1.0)
                self.support_bins = self._quantile_bins(support[:, 0])
                if self.support_bins is not None:
                    ind = self._indicator_from_bins(support[:, 0], self.support_bins)
                    self.support_bin_mean = np.mean(ind, axis=0)
                    scale = np.std(ind, axis=0)
                    self.support_bin_scale = np.where(scale > 1e-8, scale, 1.0)
        if "policy_shift" in self.extra_blocks:
            A_target = _as_2d(A_target_train, "A_target_train")
            if A_target.shape == A_train_2d.shape and A_train_2d.shape[0] >= 4:
                gap = np.linalg.norm(A_train_2d - A_target, axis=1) / np.sqrt(max(1, A_train_2d.shape[1]))
                psi_train = self._state_basis(S_train_2d)
                self.policy_shift_theta = _ridge_solve(psi_train, gap, ridge=1e-3)
                shift_train = psi_train @ self.policy_shift_theta
                self.policy_shift_mean = float(np.mean(shift_train))
                self.policy_shift_scale = float(np.std(shift_train))
                if not np.isfinite(self.policy_shift_scale) or self.policy_shift_scale <= 1e-8:
                    self.policy_shift_scale = 1.0
                self.policy_shift_bins = self._quantile_bins(shift_train)
                if self.policy_shift_bins is not None:
                    ind = self._indicator_from_bins(shift_train, self.policy_shift_bins)
                    self.policy_shift_bin_mean = np.mean(ind, axis=0)
                    scale = np.std(ind, axis=0)
                    self.policy_shift_bin_scale = np.where(scale > 1e-8, scale, 1.0)
        if rewards_train is not None:
            phi_train = self._basis_from_z(z_train)
            y = np.asarray(rewards_train, dtype=np.float64).reshape(-1)
            if y.shape[0] == phi_train.shape[0] and phi_train.shape[0] >= 4:
                self.reward_theta = _ridge_solve(phi_train, y, ridge=1e-3)
                rhat_train = phi_train @ self.reward_theta
                self.reward_mean = float(np.mean(rhat_train))
                self.reward_scale = float(np.std(rhat_train))
                if not np.isfinite(self.reward_scale) or self.reward_scale <= 1e-8:
                    self.reward_scale = 1.0
                phi_next_train = self._basis_from_z(self._std_x(S_next_train, A_next_train))
                self.value_theta = _fit_ridge_fqe(
                    phi_train=phi_train,
                    phi_next_train=phi_next_train,
                    rewards=y,
                    gamma=float(gamma),
                    iterations=max(1, int(value_iterations)),
                    patience=max(0, int(value_patience)),
                    seed=int(seed) + 17_029,
                )
                q_train = phi_train @ self.value_theta
                self.value_mean = float(np.mean(q_train))
                self.value_scale = float(np.std(q_train))
                if not np.isfinite(self.value_scale) or self.value_scale <= 1e-8:
                    self.value_scale = 1.0
                if np.all(np.isfinite(q_train)) and np.std(q_train) > 1e-8:
                    bins = self._quantile_bins(q_train)
                    if bins.size:
                        self.value_bins = bins
                        ind = self._indicator(q_train)
                        self.value_bin_mean = np.mean(ind, axis=0)
                        scale = np.std(ind, axis=0)
                        self.value_bin_scale = np.where(scale > 1e-8, scale, 1.0)

    def blocks(
        self,
        *,
        S_eval: Array,
        A_eval: Array,
        S_next: Array,
        A_next: Array,
        S_initial: Array,
        A_initial: Array,
    ) -> Dict[str, Tuple[Array, Array, Array]]:
        z_eval = self._std_x(S_eval, A_eval)
        z_next = self._std_x(S_next, A_next)
        z_initial = self._std_x(S_initial, A_initial)
        n = z_eval.shape[0]
        n0 = z_initial.shape[0]
        blocks: Dict[str, Tuple[Array, Array, Array]] = {
            "mass": (
                np.ones((n, 1), dtype=np.float64),
                np.ones((n, 1), dtype=np.float64),
                np.ones((n0, 1), dtype=np.float64),
            )
        }
        geometry = (self._geometry(z_eval), self._geometry(z_next), self._geometry(z_initial))
        if geometry[0].shape[1]:
            blocks["geometry"] = geometry
        rff = (self._rff(z_eval), self._rff(z_next), self._rff(z_initial))
        if rff[0].shape[1]:
            blocks["rff"] = rff
        if "multiscale_rff" in self.extra_blocks and self.multiscale_rff_scales:
            rff_multiscale = (
                self._rff_multiscale(z_eval),
                self._rff_multiscale(z_next),
                self._rff_multiscale(z_initial),
            )
            if rff_multiscale[0].shape[1]:
                blocks["rff_multiscale"] = rff_multiscale
        if "second_order" in self.extra_blocks and self.second_order_mean is not None:
            second_order = (
                self._second_order(z_eval),
                self._second_order(z_next),
                self._second_order(z_initial),
            )
            if second_order[0].shape[1]:
                blocks["second_order"] = second_order
        if "support" in self.extra_blocks and self.support_mean is not None:
            support = (
                self._support(z_eval),
                self._support(z_next),
                self._support(z_initial),
            )
            if support[0].shape[1]:
                blocks["support"] = support
            if self.support_bins is not None and self.support_bin_mean is not None and self.support_bin_scale is not None:
                blocks["support_strata"] = (
                    self._std_indicator_from_bins(self._support_score(z_eval), self.support_bins, self.support_bin_mean, self.support_bin_scale),
                    self._std_indicator_from_bins(self._support_score(z_next), self.support_bins, self.support_bin_mean, self.support_bin_scale),
                    self._std_indicator_from_bins(self._support_score(z_initial), self.support_bins, self.support_bin_mean, self.support_bin_scale),
                )
        if self.reward_theta is not None or self.value_theta is not None:
            phi_eval = self._basis_from_z(z_eval)
            phi_next = self._basis_from_z(z_next)
            phi_initial = self._basis_from_z(z_initial)
            if self.reward_theta is not None:
                blocks["reward"] = (
                    self._std_reward(phi_eval @ self.reward_theta),
                    self._std_reward(phi_next @ self.reward_theta),
                    self._std_reward(phi_initial @ self.reward_theta),
                )
            if self.value_theta is not None:
                q_eval = phi_eval @ self.value_theta
                q_next = phi_next @ self.value_theta
                q_initial = phi_initial @ self.value_theta
                blocks["value"] = (
                    self._std_value(q_eval),
                    self._std_value(q_next),
                    self._std_value(q_initial),
                )
                if self.value_bins is not None and self.value_bin_mean is not None and self.value_bin_scale is not None:
                    blocks["value_strata"] = (
                        self._std_indicator(q_eval),
                        self._std_indicator(q_next),
                        self._std_indicator(q_initial),
                    )
        if "policy_shift" in self.extra_blocks and self.policy_shift_theta is not None:
            shift_eval = self._policy_shift_score(S_eval)
            shift_next = self._policy_shift_score(S_next)
            shift_initial = self._policy_shift_score(S_initial)
            blocks["policy_shift"] = (
                self._std_policy_shift(shift_eval),
                self._std_policy_shift(shift_next),
                self._std_policy_shift(shift_initial),
            )
            if (
                self.policy_shift_bins is not None
                and self.policy_shift_bin_mean is not None
                and self.policy_shift_bin_scale is not None
            ):
                blocks["policy_shift_strata"] = (
                    self._std_indicator_from_bins(
                        shift_eval,
                        self.policy_shift_bins,
                        self.policy_shift_bin_mean,
                        self.policy_shift_bin_scale,
                    ),
                    self._std_indicator_from_bins(
                        shift_next,
                        self.policy_shift_bins,
                        self.policy_shift_bin_mean,
                        self.policy_shift_bin_scale,
                    ),
                    self._std_indicator_from_bins(
                        shift_initial,
                        self.policy_shift_bins,
                        self.policy_shift_bin_mean,
                        self.policy_shift_bin_scale,
                    ),
                )
        return blocks

    def _std_x(self, states: Array, actions: Array) -> Array:
        x = np.concatenate([_as_2d(states, "states"), _as_2d(actions, "actions")], axis=1)
        return (x - self.mean.reshape(1, -1)) / self.scale.reshape(1, -1)

    def _std_state(self, states: Array) -> Array:
        s = _as_2d(states, "states")
        return (s - self.state_mean.reshape(1, -1)) / self.state_scale.reshape(1, -1)

    def _geometry_components(self, z_train: Array, feature_cap: int) -> Array:
        cap = max(0, min(int(feature_cap), z_train.shape[1]))
        if cap == 0:
            return np.empty((z_train.shape[1], 0), dtype=np.float64)
        if cap >= z_train.shape[1]:
            return np.eye(z_train.shape[1], dtype=np.float64)
        _, _, vt = np.linalg.svd(z_train - np.mean(z_train, axis=0, keepdims=True), full_matrices=False)
        return vt[:cap].T

    def _geometry(self, z: Array) -> Array:
        return np.asarray(z, dtype=np.float64) @ self.geometry_components

    def _rff(self, z: Array) -> Array:
        if self.rff_count <= 0:
            return np.empty((z.shape[0], 0), dtype=np.float64)
        return np.sqrt(2.0 / max(self.rff_count, 1)) * np.cos(np.asarray(z, dtype=np.float64) @ self.rff_W + self.rff_b.reshape(1, -1))

    def _rff_multiscale(self, z: Array) -> Array:
        if not self.multiscale_rff_scales:
            return np.empty((z.shape[0], 0), dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.float64)
        return np.concatenate(
            [
                np.sqrt(2.0 / max(self.rff_count, 1))
                * np.cos(float(scale) * (z_arr @ self.rff_W) + self.rff_b.reshape(1, -1))
                for scale in self.multiscale_rff_scales
            ],
            axis=1,
        )

    def _second_order_raw(self, z: Array) -> Array:
        geom = self._geometry(z)
        if geom.shape[1] == 0:
            return np.empty((np.asarray(z).shape[0], 0), dtype=np.float64)
        return np.asarray(geom, dtype=np.float64) ** 2

    def _second_order(self, z: Array) -> Array:
        raw = self._second_order_raw(z)
        if self.second_order_mean is None or self.second_order_scale is None or raw.shape[1] == 0:
            return raw
        return (raw - self.second_order_mean.reshape(1, -1)) / self.second_order_scale.reshape(1, -1)

    def _support_raw(self, z: Array) -> Array:
        z_arr = np.asarray(z, dtype=np.float64)
        radius = np.linalg.norm(z_arr, axis=1) / np.sqrt(max(1, z_arr.shape[1]))
        geom = self._geometry(z_arr)
        if geom.shape[1]:
            geom_radius = np.linalg.norm(geom, axis=1) / np.sqrt(max(1, geom.shape[1]))
            return np.column_stack([radius, geom_radius])
        return radius.reshape(-1, 1)

    def _support(self, z: Array) -> Array:
        raw = self._support_raw(z)
        if self.support_mean is None or self.support_scale is None or raw.shape[1] == 0:
            return raw
        return (raw - self.support_mean.reshape(1, -1)) / self.support_scale.reshape(1, -1)

    def _support_score(self, z: Array) -> Array:
        raw = self._support_raw(z)
        return raw[:, 0] if raw.shape[1] else np.empty(np.asarray(z).shape[0], dtype=np.float64)

    def _basis_from_z(self, z: Array) -> Array:
        return np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), self._geometry(z), self._rff(z)], axis=1)

    def _state_basis(self, states: Array) -> Array:
        z = self._std_state(states)
        return np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z, z**2], axis=1)

    def _std_reward(self, values: Array) -> Array:
        return ((np.asarray(values, dtype=np.float64).reshape(-1) - self.reward_mean) / self.reward_scale).reshape(-1, 1)

    def _std_value(self, values: Array) -> Array:
        return ((np.asarray(values, dtype=np.float64).reshape(-1) - self.value_mean) / self.value_scale).reshape(-1, 1)

    def _policy_shift_score(self, states: Array) -> Array:
        if self.policy_shift_theta is None:
            return np.empty(_as_2d(states, "states").shape[0], dtype=np.float64)
        return self._state_basis(states) @ self.policy_shift_theta

    def _std_policy_shift(self, values: Array) -> Array:
        return ((np.asarray(values, dtype=np.float64).reshape(-1) - self.policy_shift_mean) / self.policy_shift_scale).reshape(-1, 1)

    def _indicator(self, q: Array) -> Array:
        if self.value_bins is None:
            return np.empty((np.asarray(q).shape[0], 0), dtype=np.float64)
        return self._indicator_from_bins(q, self.value_bins)

    def _std_indicator(self, q: Array) -> Array:
        ind = self._indicator(q)
        if self.value_bin_mean is None or self.value_bin_scale is None or ind.shape[1] == 0:
            return ind
        return (ind - self.value_bin_mean.reshape(1, -1)) / self.value_bin_scale.reshape(1, -1)

    def _quantile_bins(self, values: Array) -> Array:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if x.size < max(4, self.strata_quantiles.size + 1) or not np.all(np.isfinite(x)) or np.std(x) <= 1e-8:
            return np.empty(0, dtype=np.float64)
        return np.unique(np.quantile(x, self.strata_quantiles))

    def _indicator_from_bins(self, values: Array, bins: Array) -> Array:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        b = np.asarray(bins, dtype=np.float64).reshape(-1)
        if b.size == 0:
            return np.empty((x.shape[0], 0), dtype=np.float64)
        idx = np.digitize(x, b, right=False)
        return np.eye(int(b.size + 1), dtype=np.float64)[idx]

    def _std_indicator_from_bins(self, values: Array, bins: Array, mean: Array, scale: Array) -> Array:
        ind = self._indicator_from_bins(values, bins)
        if ind.shape[1] == 0:
            return ind
        return (ind - np.asarray(mean, dtype=np.float64).reshape(1, -1)) / np.asarray(scale, dtype=np.float64).reshape(1, -1)


def _validation_initial_state_actions(
    *,
    S: Array,
    A_pi: Array,
    valid_idx: Array,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
) -> Tuple[Array, Array]:
    full_n = S.shape[0]
    if S_initial is not None and np.asarray(S_initial).shape[0] == full_n:
        states = np.asarray(S_initial)[valid_idx]
        actions = np.asarray(A_initial)[valid_idx] if A_initial is not None and np.asarray(A_initial).shape[0] == full_n else A_pi[valid_idx]
        return _as_2d(states, "initial_states"), _as_2d(actions, "initial_actions")
    if S_initial is not None and A_initial is not None:
        return _as_2d(S_initial, "initial_states"), _as_2d(A_initial, "initial_actions")
    return S[valid_idx], A_pi[valid_idx]


def _moment_group_score(weights: Array, gamma: float, block: Tuple[Array, Array, Array]) -> float:
    eval_features, next_features, initial_features = (np.asarray(part, dtype=np.float64) for part in block)
    if eval_features.ndim != 2 or eval_features.shape[1] == 0:
        return float("nan")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid_term = w[:, None] * (eval_features - float(gamma) * next_features)
    initial_term = (1.0 - float(gamma)) * initial_features
    delta = np.mean(valid_term, axis=0) - np.mean(initial_term, axis=0)
    var = np.var(valid_term, axis=0, ddof=1) / max(valid_term.shape[0], 1)
    var += np.var(initial_term, axis=0, ddof=1) / max(initial_term.shape[0], 1)
    z2 = delta**2 / np.maximum(var, 1e-8)
    z2 = z2[np.isfinite(z2)]
    return float(np.mean(z2)) if z2.size else float("nan")


def _ridge_solve(features: Array, target: Array, *, ridge: float) -> Array:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)


def _fit_ridge_fqe(
    *,
    phi_train: Array,
    phi_next_train: Array,
    rewards: Array,
    gamma: float,
    iterations: int,
    patience: int,
    seed: int,
) -> Array:
    phi = np.asarray(phi_train, dtype=np.float64)
    phi_next = np.asarray(phi_next_train, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(phi.shape[0])
    n_valid = max(1, int(round(0.2 * phi.shape[0])))
    valid = perm[:n_valid]
    train = perm[n_valid:] if perm[n_valid:].size else valid
    theta = _ridge_solve(phi[train], r[train], ridge=1e-3)
    best = theta.copy()
    best_loss = float("inf")
    stale = 0
    for _ in range(max(1, int(iterations))):
        target = r[train] + float(gamma) * (phi_next[train] @ theta)
        theta_new = _ridge_solve(phi[train], target, ridge=1e-3)
        pred = phi[valid] @ theta_new
        target_valid = r[valid] + float(gamma) * (phi_next[valid] @ theta)
        loss = float(np.mean((pred - target_valid) ** 2))
        if loss < best_loss - 1e-8:
            best_loss = loss
            best = theta_new.copy()
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
        theta = theta_new
    return best


def _make_folds(n_rows: int, n_folds: int, seed: int, *, groups: Optional[Array]) -> List[Array]:
    rng = np.random.default_rng(seed)
    if groups is None:
        return [fold.astype(np.int64, copy=False) for fold in np.array_split(rng.permutation(int(n_rows)), int(n_folds))]
    group_arr = np.asarray(groups).reshape(-1)
    if group_arr.shape[0] != int(n_rows):
        raise ValueError("groups must have the same number of rows as states.")
    unique_groups = np.unique(group_arr)
    shuffled = unique_groups[rng.permutation(unique_groups.shape[0])]
    group_folds = np.array_split(shuffled, int(n_folds))
    return [np.flatnonzero(np.isin(group_arr, group_fold)).astype(np.int64, copy=False) for group_fold in group_folds]


def _complement_indices(n_rows: int, valid_idx: Array) -> Array:
    mask = np.ones(int(n_rows), dtype=bool)
    mask[np.asarray(valid_idx, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def _fold_initial_states(initial_states: Optional[Array], train_idx: Array, full_n: int) -> Optional[Array]:
    if initial_states is None:
        return None
    arr = np.asarray(initial_states)
    if arr.shape[0] == int(full_n):
        return arr[np.asarray(train_idx, dtype=np.int64)]
    return arr


def _fold_initial_weights(
    initial_weights: Optional[Array],
    initial_states: Optional[Array],
    train_idx: Array,
    full_n: int,
) -> Optional[Array]:
    if initial_weights is None:
        return None
    weights = np.asarray(initial_weights)
    if initial_states is not None and np.asarray(initial_states).shape[0] == int(full_n):
        return weights[np.asarray(train_idx, dtype=np.int64)]
    return weights


def _as_2d(x: Array, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must be a 1D or 2D array.")


def _best_history_loss(history: Sequence[Dict[str, Any]]) -> float:
    losses = []
    for row in history:
        for key in ("risk_new", "loss"):
            if key not in row:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                losses.append(value)
    return float(np.min(losses)) if losses else float("inf")


def _ess_fraction(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return float("nan")
    denom = float(np.sum(w**2))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return float((np.sum(w) ** 2 / denom) / w.size)


def _quantile_or_nan(values: Array, q: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def _clipped_fraction(weights: Array, occupancy: Any) -> float:
    cap = getattr(occupancy, "occupancy_ratio_max", None)
    if cap is None:
        cap = getattr(occupancy, "prediction_max", None)
    if cap is None:
        return 0.0
    x = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x >= float(cap) * (1.0 - 1e-8)))


def _weight_quality_from_values(weights: Array, occupancy: Any, *, action_shift: float = 0.0) -> float:
    x = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("inf")
    ess = _ess_fraction(x)
    norm = abs(float(np.mean(x)) - 1.0)
    clipped = _clipped_fraction(x, occupancy)
    return float(
        norm
        + _ess_catastrophe_penalty(ess, n_weights=x.size)
        + clipped
        + _near_uniform_penalty(ess, action_shift, _weight_cv(x))
    )


def _action_shift(behavior_actions: Array, target_actions: Array) -> float:
    behavior = _as_2d(behavior_actions, "behavior_actions")
    target = _as_2d(target_actions, "target_actions")
    if behavior.shape != target.shape or behavior.shape[0] == 0:
        return 0.0
    diff = behavior - target
    return float(np.mean(np.linalg.norm(diff, axis=1)) / np.sqrt(max(1, behavior.shape[1])))


def _weight_cv(weights: Array) -> float:
    x = np.asarray(weights, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    mean = float(np.mean(x))
    return float(np.std(x) / (abs(mean) + 1e-12))


def _ess_catastrophe_penalty(ess: float, *, n_weights: Any = None) -> float:
    if not np.isfinite(float(ess)):
        return 10.0
    floor = _ess_catastrophe_floor(n_weights=n_weights)
    if float(ess) >= floor:
        return 0.0
    return float(10.0 * (floor - max(float(ess), 0.0)) / floor)


def _ess_is_catastrophic(ess: float, *, n_weights: Any = None) -> bool:
    if not np.isfinite(float(ess)):
        return True
    return bool(float(ess) < _ess_catastrophe_floor(n_weights=n_weights))


def _ess_catastrophe_floor(*, n_weights: Any = None) -> float:
    floor = 1e-3
    try:
        n = float(n_weights)
    except (TypeError, ValueError):
        n = float("nan")
    if np.isfinite(n) and n > 0.0:
        floor = max(floor, 1.0 / n)
    return float(floor)


def _near_uniform_penalty(ess: float, action_shift: float, weight_cv: float) -> float:
    if not (np.isfinite(ess) and np.isfinite(action_shift) and np.isfinite(weight_cv)):
        return 0.0
    if ess < 0.995 or action_shift < 0.05 or weight_cv > 0.05:
        return 0.0
    collapse = min(1.0, max(0.0, (float(ess) - 0.995) / 0.005))
    shift = min(1.0, max(0.0, float(action_shift)))
    uniform = min(1.0, max(0.0, (0.05 - float(weight_cv)) / 0.05))
    return float(collapse * shift * uniform)


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("inf")


def _se(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))
