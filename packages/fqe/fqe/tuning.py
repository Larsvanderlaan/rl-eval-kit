from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from fqe.fit_fqe import (
    Array,
    BoostedFQEConfig,
    FQEModel,
    _as_1d_float,
    _as_2d_float,
    _as_next_actions,
    _bellman_risk,
    _config_with_updates,
    _optional_terminals,
    _optional_weights,
    fit_fqe_lgbm,
    fit_value_lgbm,
)
from fqe.fit_neural_fqe import (
    NeuralFQEConfig,
    NeuralFQEModel,
    fit_fqe_neural,
    fit_value_neural,
)
from fqe.bellman import validate_action_weights, validate_bootstrap_inputs, weighted_action_expectation


__all__ = [
    "FQECandidateResult",
    "FQEFoldResult",
    "FQESearchSpace",
    "FQEStagedCVCandidateTelemetry",
    "FQEStagedCVFoldTelemetry",
    "FQEStagedCVStageTelemetry",
    "FQETargetValidationCandidateResult",
    "FQETargetValidationResult",
    "FQETuningConfig",
    "FQETuningResult",
    "tune_fqe",
    "tune_fqe_auto",
    "tune_fqe_with_target_validation",
]


@dataclass(frozen=True)
class FQETuningConfig:
    """Product-level CV/AutoML controls for FQE tuning."""

    families: Sequence[str] = ("boosted",)
    cv_folds: int = 3
    seed: int = 123
    budget: str = "balanced"
    max_candidates: int = 12
    promotion_candidates: int = 4
    refit: bool = True
    screen_fraction: float = 0.4
    score_bellman_weight: float = 0.60
    score_value_stability_weight: float = 0.15
    score_calibration_weight: float = 0.15
    score_runtime_weight: float = 0.10
    stable_fallback: bool = True
    fallback_score_tolerance: float = 0.05
    fallback_runtime_ratio: float = 2.5
    staged_bootstrap_cv: bool = False
    staged_cv_iterations: int | Sequence[int] | None = 3
    staged_cv_n_bootstrap: int = 200
    staged_cv_bootstrap_samples: int | None = None
    staged_cv_one_se_pruning: bool = True
    staged_cv_one_se_multiplier: float = 1.0
    staged_cv_always_evaluate_baseline: bool = True
    staged_cv_min_survivors: int = 1

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
            self.score_bellman_weight,
            self.score_value_stability_weight,
            self.score_calibration_weight,
            self.score_runtime_weight,
        )
        if any(float(weight) < 0.0 for weight in weights):
            raise ValueError("score weights must be nonnegative.")
        if sum(float(weight) for weight in weights) <= 0.0:
            raise ValueError("at least one score weight must be positive.")
        if float(self.fallback_score_tolerance) < 0.0:
            raise ValueError("fallback_score_tolerance must be nonnegative.")
        if float(self.fallback_runtime_ratio) <= 1.0:
            raise ValueError("fallback_runtime_ratio must be > 1.")
        if self.staged_cv_iterations is not None:
            if isinstance(self.staged_cv_iterations, (int, np.integer)):
                if not (1 <= int(self.staged_cv_iterations) <= 5):
                    raise ValueError("staged_cv_iterations must be in [1, 5].")
            else:
                iterations = tuple(int(value) for value in self.staged_cv_iterations)
                if not iterations:
                    raise ValueError("staged_cv_iterations must be nonempty when supplied.")
                if any(value <= 0 or value > 5 for value in iterations):
                    raise ValueError("staged_cv_iterations entries must be in [1, 5].")
        if int(self.staged_cv_n_bootstrap) < 0:
            raise ValueError("staged_cv_n_bootstrap must be nonnegative.")
        if self.staged_cv_bootstrap_samples is not None and int(self.staged_cv_bootstrap_samples) < 0:
            raise ValueError("staged_cv_bootstrap_samples must be nonnegative.")
        if float(self.staged_cv_one_se_multiplier) < 0.0 or not np.isfinite(float(self.staged_cv_one_se_multiplier)):
            raise ValueError("staged_cv_one_se_multiplier must be finite and nonnegative.")
        if int(self.staged_cv_min_survivors) <= 0:
            raise ValueError("staged_cv_min_survivors must be positive.")


from fqe.staged_cv import (  # noqa: E402
    FQEStagedCVCandidateTelemetry,
    FQEStagedCVFoldTelemetry,
    FQEStagedCVStageTelemetry,
)


@dataclass(frozen=True)
class FQESearchSpace:
    """Base configs and candidate overrides for the FQE tuning harness."""

    boosted: BoostedFQEConfig = field(default_factory=BoostedFQEConfig.stable_defaults)
    neural: NeuralFQEConfig = field(default_factory=NeuralFQEConfig.stable_defaults)
    boosted_candidates: Optional[Sequence[Dict[str, Any]]] = None
    neural_candidates: Optional[Sequence[Dict[str, Any]]] = None


@dataclass
class FQEFoldResult:
    candidate_id: str
    family: str
    budget_stage: str
    fold: int
    runtime_sec: float
    bellman_risk: float
    calibration_error: float
    policy_value: float | None = None
    n_train: int = 0
    n_validation: int = 0
    validation_weight_sum: float = 0.0


@dataclass
class FQECandidateResult:
    candidate_id: str
    family: str
    budget_stage: str
    overrides: Dict[str, Any]
    fold_results: List[FQEFoldResult]
    metrics: Dict[str, float]
    score: float = float("inf")
    runtime_sec: float = 0.0
    promoted: bool = False
    selected: bool = False
    error: str = ""


@dataclass
class FQETuningResult:
    selected_family: str
    selected_candidate_id: str
    selected_overrides: Dict[str, Any]
    selected_config: Any
    candidates: List[FQECandidateResult]
    folds: List[FQEFoldResult]
    model: FQEModel | NeuralFQEModel | None
    config: FQETuningConfig
    staged_cv_rows_data: List[Dict[str, Any]] = field(default_factory=list)

    def candidate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for candidate in self.candidates:
            row = {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "budget_stage": candidate.budget_stage,
                "score": float(candidate.score),
                "runtime_sec": float(candidate.runtime_sec),
                "promoted": bool(candidate.promoted),
                "selected": bool(candidate.selected),
                "error": candidate.error,
            }
            row.update({f"metric_{key}": value for key, value in candidate.metrics.items()})
            rows.append(row)
        return rows

    def fold_rows(self) -> List[Dict[str, Any]]:
        return [asdict(fold) for fold in self.folds]

    def staged_cv_rows(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.staged_cv_rows_data]


@dataclass
class FQETargetValidationCandidateResult:
    candidate_id: str
    family: str
    overrides: Dict[str, Any]
    metrics: Dict[str, float]
    score: float = float("inf")
    score_se: float = 0.0
    runtime_sec: float = 0.0
    selected_min_score: bool = False
    selected: bool = False
    error: str = ""


@dataclass
class FQETargetValidationResult:
    selected_family: str
    selected_candidate_id: str
    selected_overrides: Dict[str, Any]
    selected_config: Any
    candidates: List[FQETargetValidationCandidateResult]
    model: FQEModel | NeuralFQEModel | None
    config: FQETuningConfig
    score_mode: str
    validation_diagnostics: Dict[str, float | str]
    selection_rule: str = "min_score"

    def candidate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for candidate in self.candidates:
            row = {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "score": float(candidate.score),
                "score_se": float(candidate.score_se),
                "runtime_sec": float(candidate.runtime_sec),
                "selected_min_score": bool(candidate.selected_min_score),
                "selected": bool(candidate.selected),
                "error": candidate.error,
            }
            row.update({f"metric_{key}": value for key, value in candidate.metrics.items()})
            rows.append(row)
        return rows

    def validation_rows(self) -> List[Dict[str, Any]]:
        return self.candidate_rows()


@dataclass(frozen=True)
class _FQETargetTrajectory:
    states: Array
    actions: Array | None
    rewards: Array
    next_states: Array
    episode_ids: Array
    timesteps: Array
    continuation: Array
    tail_actions: Array | None


def tune_fqe_auto(
    *,
    states: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    actions: Array | None = None,
    next_actions: Array | None = None,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    initial_weights: Array | None = None,
    groups: Array | None = None,
    families: Sequence[str] = ("boosted",),
    search_space: FQESearchSpace | None = None,
    config: FQETuningConfig | None = None,
    categorical_feature: Sequence[int | str] | None = None,
) -> FQETuningResult:
    """Run the recommended product AutoML preset for FQE."""

    cfg = config if config is not None else FQETuningConfig(families=tuple(families))
    if config is not None and tuple(families) != ("boosted",):
        cfg = replace(cfg, families=tuple(families))
    return tune_fqe(
        states=states,
        actions=actions,
        next_states=next_states,
        next_actions=next_actions,
        rewards=rewards,
        gamma=gamma,
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
        sample_weight=sample_weight,
        next_action_weights=next_action_weights,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        groups=groups,
        search_space=search_space,
        config=cfg,
        categorical_feature=categorical_feature,
    )


def tune_fqe_with_target_validation(
    *,
    states: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    actions: Array | None = None,
    next_actions: Array | None = None,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    initial_weights: Array | None = None,
    families: Sequence[str] = ("boosted",),
    search_space: FQESearchSpace | None = None,
    config: FQETuningConfig | None = None,
    categorical_feature: Sequence[int | str] | None = None,
    score_mode: Literal["n_step_td", "scalar_value"] = "n_step_td",
    selection_rule: Literal["min_score", "one_se"] = "min_score",
    validation_states: Array | None = None,
    validation_actions: Array | None = None,
    validation_rewards: Array | None = None,
    validation_next_states: Array | None = None,
    validation_episode_ids: Array | None = None,
    validation_timestep: Array | None = None,
    validation_terminals: Array | None = None,
    validation_continuation: Array | None = None,
    validation_tail_actions: Array | None = None,
    target_value: float | None = None,
    target_value_se: float | None = None,
) -> FQETargetValidationResult:
    """Tune FQE candidates with independent target-policy validation labels.

    The default ``n_step_td`` mode scores candidates with finite target-policy
    rollout prefixes plus the candidate's own continuation value. This is a
    target-policy Bellman validation criterion, not a claim that finite
    trajectories are exact infinite-horizon returns. Candidates are selected by
    minimum validation score by default; pass ``selection_rule="one_se"`` for a
    conservative one-standard-error selector.
    """

    cfg = config if config is not None else FQETuningConfig(families=tuple(families))
    if config is not None and tuple(families) != ("boosted",):
        cfg = replace(cfg, families=tuple(families))
    space = search_space if search_space is not None else FQESearchSpace()
    rewards_1d = _as_1d_float(rewards, "rewards")
    S = _as_2d_float(states, "states", n_rows=rewards_1d.shape[0])
    S_next = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    bootstrap = validate_bootstrap_inputs(
        n_rows=rewards_1d.shape[0],
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
    )
    terminal_arr = 1.0 - bootstrap.continuation
    weight_arr = _optional_weights(sample_weight, rewards_1d.shape[0], "sample_weight")
    mode = _resolve_mode(actions, next_actions)
    A = None if actions is None else _as_2d_float(actions, "actions", n_rows=rewards_1d.shape[0])
    A_next = None
    if next_actions is not None:
        assert A is not None
        A_next = _as_next_actions(next_actions, n_rows=rewards_1d.shape[0], action_dim=A.shape[1])
    next_action_weight_arr = None
    if A_next is not None:
        next_action_weight_arr = validate_action_weights(
            next_action_weights,
            n_rows=rewards_1d.shape[0],
            n_actions=A_next.shape[1],
            name="next_action_weights",
        )
    S_initial = None if initial_states is None else _as_2d_float(initial_states, "initial_states")
    A_initial = None if initial_actions is None else _as_2d_float(
        initial_actions,
        "initial_actions",
        n_rows=None if S_initial is None else S_initial.shape[0],
    )
    initial_weight_arr = None if initial_weights is None else _optional_weights(
        initial_weights,
        S_initial.shape[0] if S_initial is not None else len(initial_weights),
        "initial_weights",
    )
    if str(score_mode) not in {"n_step_td", "scalar_value"}:
        raise ValueError("score_mode must be 'n_step_td' or 'scalar_value'.")
    selection_rule_value = str(selection_rule)
    if selection_rule_value not in {"min_score", "one_se"}:
        raise ValueError("selection_rule must be 'min_score' or 'one_se'.")
    trajectory = None
    diagnostics: Dict[str, float | str] = {
        "score_mode": str(score_mode),
        "validation_selection_rule": selection_rule_value,
    }
    if str(score_mode) == "n_step_td":
        trajectory = _validate_fqe_target_trajectory(
            mode=mode,
            validation_states=validation_states,
            validation_actions=validation_actions,
            validation_rewards=validation_rewards,
            validation_next_states=validation_next_states,
            validation_episode_ids=validation_episode_ids,
            validation_timestep=validation_timestep,
            validation_terminals=validation_terminals,
            validation_continuation=validation_continuation,
            validation_tail_actions=validation_tail_actions,
        )
        diagnostics.update(_fqe_target_trajectory_diagnostics(trajectory, gamma=float(gamma), seed=int(cfg.seed) + 19_003))
    else:
        if target_value is None or not np.isfinite(float(target_value)):
            raise ValueError("target_value must be supplied and finite for score_mode='scalar_value'.")
        diagnostics.update(
            {
                "target_value": float(target_value),
                "target_value_se": 0.0 if target_value_se is None else float(target_value_se),
                "validation_label_scope": "scalar_value_only",
            }
        )

    candidates = _make_candidates(space, cfg)
    candidate_results: List[FQETargetValidationCandidateResult] = []
    models: Dict[str, FQEModel | NeuralFQEModel] = {}
    configs: Dict[str, BoostedFQEConfig | NeuralFQEConfig] = {}
    for rank, candidate in enumerate(candidates):
        start = time.perf_counter()
        family = str(candidate["family"])
        overrides = dict(candidate["overrides"])
        error = ""
        metrics: Dict[str, float] = {}
        score = float("inf")
        score_se = 0.0
        try:
            candidate_cfg = _build_config(
                family=family,
                overrides=overrides,
                space=space,
                screen_fraction=1.0,
                seed=int(cfg.seed) + 707_707 + 10_001 * rank,
                force_final=True,
            )
            model = _fit_family(
                family=family,
                mode=mode,
                config=candidate_cfg,
                S=S,
                A=A,
                S_next=S_next,
                A_next=A_next,
                rewards=rewards_1d,
                gamma=float(gamma),
                terminals=terminal_arr,
                sample_weight=weight_arr,
                next_action_weights=next_action_weight_arr,
                categorical_feature=categorical_feature,
            )
            if str(score_mode) == "n_step_td":
                assert trajectory is not None
                metrics = _score_fqe_target_trajectory(
                    model=model,
                    mode=mode,
                    trajectory=trajectory,
                    gamma=float(gamma),
                    seed=int(cfg.seed) + 811_003 + rank,
                )
            else:
                policy_value = _estimate_policy_value_or_none(model, S_initial, A_initial, initial_weight_arr)
                if policy_value is None or not np.isfinite(float(policy_value)):
                    raise ValueError("candidate cannot estimate the requested initial-state policy value.")
                target = float(target_value)
                score = abs(float(policy_value) - target)
                score_se = 0.0 if target_value_se is None else max(float(target_value_se), 0.0)
                metrics = {
                    "validation_score": score,
                    "validation_score_se": score_se,
                    "policy_value": float(policy_value),
                    "target_value": target,
                    "scalar_value_only": 1.0,
                }
            if str(score_mode) == "n_step_td":
                score = float(metrics["validation_score"])
                score_se = float(metrics["validation_score_se"])
            models[str(candidate["candidate_id"])] = model
            configs[str(candidate["candidate_id"])] = candidate_cfg
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        candidate_results.append(
            FQETargetValidationCandidateResult(
                candidate_id=str(candidate["candidate_id"]),
                family=family,
                overrides=overrides,
                metrics=metrics,
                score=float(score),
                score_se=float(score_se),
                runtime_sec=float(time.perf_counter() - start),
                error=error,
            )
        )

    selected, min_score_selected, one_se_selected = _select_target_validation_candidate(
        candidate_results,
        selection_rule=selection_rule_value,
    )
    if selected is None:
        errors = "; ".join(candidate.error for candidate in candidate_results if candidate.error)
        raise RuntimeError(f"No FQE target-validation candidates completed successfully. {errors}".strip())
    selected.selected = True
    model = models.get(selected.candidate_id)
    selected_config = configs.get(selected.candidate_id)
    assert selected_config is not None
    diagnostics.update(
        {
            "selected_min_score_candidate_id": "" if min_score_selected is None else min_score_selected.candidate_id,
            "selected_one_se_candidate_id": "" if one_se_selected is None else one_se_selected.candidate_id,
        }
    )
    return FQETargetValidationResult(
        selected_family=selected.family,
        selected_candidate_id=selected.candidate_id,
        selected_overrides=selected.overrides,
        selected_config=selected_config,
        candidates=candidate_results,
        model=model,
        config=cfg,
        score_mode=str(score_mode),
        selection_rule=selection_rule_value,
        validation_diagnostics=diagnostics,
    )


def tune_fqe(
    *,
    states: Array,
    next_states: Array,
    rewards: Array,
    gamma: float,
    actions: Array | None = None,
    next_actions: Array | None = None,
    terminals: Array | None = None,
    timeouts: Array | None = None,
    continuation: Array | None = None,
    sample_weight: Array | None = None,
    next_action_weights: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    initial_weights: Array | None = None,
    groups: Array | None = None,
    search_space: FQESearchSpace | None = None,
    config: FQETuningConfig | None = None,
    categorical_feature: Sequence[int | str] | None = None,
) -> FQETuningResult:
    """Tune boosted or neural FQE with proxy-only cross-validation."""

    cfg = config if config is not None else FQETuningConfig()
    space = search_space if search_space is not None else FQESearchSpace()
    rewards_1d = _as_1d_float(rewards, "rewards")
    S = _as_2d_float(states, "states", n_rows=rewards_1d.shape[0])
    S_next = _as_2d_float(next_states, "next_states", n_rows=rewards_1d.shape[0])
    bootstrap = validate_bootstrap_inputs(
        n_rows=rewards_1d.shape[0],
        terminals=terminals,
        timeouts=timeouts,
        continuation=continuation,
    )
    terminal_arr = 1.0 - bootstrap.continuation
    weight_arr = _optional_weights(sample_weight, rewards_1d.shape[0], "sample_weight")
    mode = _resolve_mode(actions, next_actions)
    A = None if actions is None else _as_2d_float(actions, "actions", n_rows=rewards_1d.shape[0])
    A_next = None
    if next_actions is not None:
        assert A is not None
        A_next = _as_next_actions(next_actions, n_rows=rewards_1d.shape[0], action_dim=A.shape[1])
    next_action_weight_arr = None
    if A_next is not None:
        next_action_weight_arr = validate_action_weights(
            next_action_weights,
            n_rows=rewards_1d.shape[0],
            n_actions=A_next.shape[1],
            name="next_action_weights",
        )
    S_initial = None if initial_states is None else _as_2d_float(initial_states, "initial_states")
    A_initial = None if initial_actions is None else _as_2d_float(initial_actions, "initial_actions", n_rows=None if S_initial is None else S_initial.shape[0])
    initial_weight_arr = None if initial_weights is None else _optional_weights(initial_weights, S_initial.shape[0] if S_initial is not None else len(initial_weights), "initial_weights")

    if bool(cfg.staged_bootstrap_cv):
        from fqe.staged_cv import tune_fqe_staged_bootstrap_cv

        return tune_fqe_staged_bootstrap_cv(
            S=S,
            A=A,
            S_next=S_next,
            A_next=A_next,
            rewards=rewards_1d,
            gamma=float(gamma),
            terminals=terminal_arr,
            sample_weight=weight_arr,
            next_action_weights=next_action_weight_arr,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weight_arr,
            groups=groups,
            search_space=space,
            config=cfg,
            mode=mode,
            categorical_feature=categorical_feature,
        )

    folds = _make_folds(S.shape[0], int(cfg.cv_folds), int(cfg.seed), groups=groups)
    candidates = _make_candidates(space, cfg)
    screen_candidates = [
        _evaluate_candidate(
            candidate=candidate,
            budget_stage="screen",
            screen_fraction=_budget_screen_fraction(cfg),
            folds=folds,
            mode=mode,
            S=S,
            A=A,
            S_next=S_next,
            A_next=A_next,
            rewards=rewards_1d,
            gamma=float(gamma),
            terminals=terminal_arr,
            sample_weight=weight_arr,
            next_action_weights=next_action_weight_arr,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weight_arr,
            space=space,
            seed=int(cfg.seed),
            categorical_feature=categorical_feature,
        )
        for candidate in candidates
    ]
    _score_candidates(screen_candidates, cfg)
    promoted_ids = {
        candidate.candidate_id
        for candidate in sorted(screen_candidates, key=lambda row: row.score)[: min(_budget_promotion_limit(cfg), len(screen_candidates))]
    }
    promoted_ids.update(candidate.candidate_id for candidate in screen_candidates if _is_baseline_candidate(candidate))
    for candidate in screen_candidates:
        candidate.promoted = candidate.candidate_id in promoted_ids
    full_candidates = [
        _evaluate_candidate(
            candidate=candidate,
            budget_stage="full",
            screen_fraction=1.0,
            folds=folds,
            mode=mode,
            S=S,
            A=A,
            S_next=S_next,
            A_next=A_next,
            rewards=rewards_1d,
            gamma=float(gamma),
            terminals=terminal_arr,
            sample_weight=weight_arr,
            next_action_weights=next_action_weight_arr,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weight_arr,
            space=space,
            seed=int(cfg.seed) + 97_531,
            categorical_feature=categorical_feature,
        )
        for candidate in candidates
        if candidate["candidate_id"] in promoted_ids
    ]
    _score_candidates(full_candidates, cfg)
    selection_pool = [
        candidate
        for candidate in (full_candidates or screen_candidates)
        if not candidate.error and candidate.fold_results and np.isfinite(float(candidate.score))
    ]
    if not selection_pool:
        errors = "; ".join(candidate.error for candidate in (full_candidates or screen_candidates) if candidate.error)
        raise RuntimeError(f"No FQE tuning candidates completed successfully. {errors}".strip())

    selected = min(selection_pool, key=lambda row: row.score)
    selected_config = _build_config(
        family=selected.family,
        overrides=selected.overrides,
        space=space,
        screen_fraction=1.0,
        seed=int(cfg.seed) + 707_707,
        force_final=True,
    )
    model = None
    if cfg.refit:
        selected, selected_config, model = _select_refit_candidate(
            candidates=selection_pool,
            space=space,
            cfg=cfg,
            mode=mode,
            S=S,
            A=A,
            S_next=S_next,
            A_next=A_next,
            rewards=rewards_1d,
            gamma=float(gamma),
            terminals=terminal_arr,
            sample_weight=weight_arr,
            next_action_weights=next_action_weight_arr,
            categorical_feature=categorical_feature,
        )
    selected.selected = True
    all_candidates = screen_candidates + full_candidates
    all_folds = [fold for candidate in all_candidates for fold in candidate.fold_results]
    return FQETuningResult(
        selected_family=selected.family,
        selected_candidate_id=selected.candidate_id,
        selected_overrides=selected.overrides,
        selected_config=selected_config,
        candidates=all_candidates,
        folds=all_folds,
        model=model,
        config=cfg,
    )


def _resolve_mode(actions: Array | None, next_actions: Array | None) -> str:
    if actions is None and next_actions is None:
        return "value"
    if actions is not None and next_actions is not None:
        return "q"
    raise ValueError("actions and next_actions must either both be supplied or both be omitted.")


def _make_candidates(space: FQESearchSpace, cfg: FQETuningConfig) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for family in tuple(str(family) for family in cfg.families):
        raw = (
            list(space.boosted_candidates)
            if family == "boosted" and space.boosted_candidates is not None
            else list(space.neural_candidates)
            if family == "neural" and space.neural_candidates is not None
            else _default_family_candidates(family, space)
        )
        capped = _cap_candidates(raw, _budget_candidate_limit(cfg))
        for idx, overrides in enumerate(capped):
            candidates.append({"candidate_id": f"{family}_{idx:03d}", "family": family, "overrides": dict(overrides)})
    return candidates


def _default_family_candidates(family: str, space: FQESearchSpace) -> List[Dict[str, Any]]:
    if family == "boosted":
        return [
            {},
            {"lgb_params": {"num_leaves": 15, "min_data_in_leaf": 30}},
            {"lgb_params": {"learning_rate": 0.03, "lambda_l2": 2.0}},
            {"huber_delta_scale": 1.0},
            {"huber_delta_scale": 2.0},
            {"loss": "squared"},
            {"infer_value_bounds": False},
            {"learning_rate_backoff": 0.75},
            {"lgb_params": {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}},
            {"lgb_params": {"num_leaves": 63, "min_data_in_leaf": 50, "lambda_l2": 3.0}},
            {"patience": max(5, int(space.boosted.patience // 2))},
            {"update_target_frequency": 2},
        ]
    base_dims = tuple(int(width) for width in space.neural.hidden_dims)
    small_dims = tuple(max(16, width // 2) for width in base_dims)
    large_dims = tuple(max(16, width * 2) for width in base_dims)
    return [
        {},
        {"hidden_dims": small_dims},
        {"hidden_dims": large_dims},
        {"learning_rate": float(space.neural.learning_rate) / 3.0},
        {"learning_rate": float(space.neural.learning_rate) * 3.0},
        {"huber_delta_scale": 1.0},
        {"huber_delta_scale": 2.0},
        {"loss": "squared"},
        {"target_update_tau": min(1.0, float(space.neural.target_update_tau) * 2.0)},
        {"weight_decay": float(space.neural.weight_decay) * 3.0},
    ]


def _cap_candidates(candidates: Sequence[Dict[str, Any]], max_candidates: int) -> List[Dict[str, Any]]:
    if len(candidates) <= int(max_candidates):
        return list(candidates)
    return list(candidates[: int(max_candidates)])


def _budget_candidate_limit(cfg: FQETuningConfig) -> int:
    limit = int(cfg.max_candidates)
    if str(cfg.budget) == "fast":
        return max(1, min(limit, 8))
    return limit


def _budget_promotion_limit(cfg: FQETuningConfig) -> int:
    limit = int(cfg.promotion_candidates)
    if str(cfg.budget) == "fast":
        return max(1, min(limit, 2))
    return limit


def _budget_screen_fraction(cfg: FQETuningConfig) -> float:
    fraction = float(cfg.screen_fraction)
    if str(cfg.budget) == "fast":
        return min(fraction, 0.30)
    return fraction


def _evaluate_candidate(
    *,
    candidate: Dict[str, Any],
    budget_stage: str,
    screen_fraction: float,
    folds: Sequence[Array],
    mode: str,
    S: Array,
    A: Array | None,
    S_next: Array,
    A_next: Array | None,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
    next_action_weights: Array | None,
    S_initial: Array | None,
    A_initial: Array | None,
    initial_weights: Array | None,
    space: FQESearchSpace,
    seed: int,
    categorical_feature: Sequence[int | str] | None,
) -> FQECandidateResult:
    fold_results: List[FQEFoldResult] = []
    start = time.perf_counter()
    family = str(candidate["family"])
    overrides = dict(candidate["overrides"])
    error = ""
    try:
        for fold_id, valid_idx in enumerate(folds):
            train_idx = _complement_indices(S.shape[0], valid_idx)
            cfg = _build_config(
                family=family,
                overrides=overrides,
                space=space,
                screen_fraction=screen_fraction,
                seed=seed + 10_003 * (fold_id + 1),
                force_final=False,
            )
            fold_start = time.perf_counter()
            model = _fit_family(
                family=family,
                mode=mode,
                config=cfg,
                S=S[train_idx],
                A=None if A is None else A[train_idx],
                S_next=S_next[train_idx],
                A_next=None if A_next is None else A_next[train_idx],
                rewards=rewards[train_idx],
                gamma=gamma,
                terminals=terminals[train_idx],
                sample_weight=sample_weight[train_idx],
                next_action_weights=None if next_action_weights is None else next_action_weights[train_idx],
                categorical_feature=categorical_feature,
            )
            pred = _predict_current(model, mode, S[valid_idx], None if A is None else A[valid_idx])
            next_pred = _predict_next(
                model,
                mode,
                S_next[valid_idx],
                None if A_next is None else A_next[valid_idx],
                None if next_action_weights is None else next_action_weights[valid_idx],
            )
            risk = _bellman_risk(
                predictions=pred,
                next_predictions=next_pred,
                rewards=rewards[valid_idx],
                gamma=gamma,
                terminals=terminals[valid_idx],
                sample_weight=sample_weight[valid_idx],
            )
            calibration = _calibration_error(
                predictions=pred,
                next_predictions=next_pred,
                rewards=rewards[valid_idx],
                gamma=gamma,
                terminals=terminals[valid_idx],
                sample_weight=sample_weight[valid_idx],
            )
            policy_value = _estimate_policy_value_or_none(model, S_initial, A_initial, initial_weights)
            fold_results.append(
                FQEFoldResult(
                    candidate_id=str(candidate["candidate_id"]),
                    family=family,
                    budget_stage=budget_stage,
                    fold=int(fold_id),
                    runtime_sec=float(time.perf_counter() - fold_start),
                    bellman_risk=float(risk),
                    calibration_error=float(calibration),
                    policy_value=policy_value,
                    n_train=int(train_idx.shape[0]),
                    n_validation=int(valid_idx.shape[0]),
                    validation_weight_sum=float(np.sum(sample_weight[valid_idx])),
                )
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime = float(time.perf_counter() - start)
    metrics = _aggregate_fold_metrics(fold_results, runtime_sec=runtime)
    return FQECandidateResult(
        candidate_id=str(candidate["candidate_id"]),
        family=family,
        budget_stage=budget_stage,
        overrides=overrides,
        fold_results=fold_results,
        metrics=metrics,
        runtime_sec=runtime,
        error=error,
    )


def _select_refit_candidate(
    *,
    candidates: Sequence[FQECandidateResult],
    space: FQESearchSpace,
    cfg: FQETuningConfig,
    mode: str,
    S: Array,
    A: Array | None,
    S_next: Array,
    A_next: Array | None,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
    next_action_weights: Array | None,
    categorical_feature: Sequence[int | str] | None,
) -> tuple[FQECandidateResult, Any, FQEModel | NeuralFQEModel]:
    scored: List[tuple[float, FQECandidateResult, Any, FQEModel | NeuralFQEModel]] = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda row: row.score)):
        candidate_cfg = _build_config(
            family=candidate.family,
            overrides=candidate.overrides,
            space=space,
            screen_fraction=1.0,
            seed=int(cfg.seed) + 707_707 + 10_001 * rank,
            force_final=True,
        )
        try:
            model = _fit_family(
                family=candidate.family,
                mode=mode,
                config=candidate_cfg,
                S=S,
                A=A,
                S_next=S_next,
                A_next=A_next,
                rewards=rewards,
                gamma=gamma,
                terminals=terminals,
                sample_weight=sample_weight,
                next_action_weights=next_action_weights,
                categorical_feature=categorical_feature,
            )
            final_risk = float(getattr(model, "diagnostics", {}).get("best_validation_bellman_risk", candidate.metrics.get("bellman_risk", 0.0)))
            candidate.metrics["final_refit_bellman_risk"] = final_risk
            final_score = float(candidate.score) + 0.05 * np.log1p(max(final_risk, 0.0))
            candidate.metrics["final_selection_score"] = float(final_score)
            scored.append((final_score, candidate, candidate_cfg, model))
        except Exception as exc:
            candidate.metrics["final_refit_failed"] = 1.0
            candidate.error = candidate.error or f"{type(exc).__name__}: {exc}"
    if not scored:
        errors = "; ".join(candidate.error for candidate in candidates if candidate.error)
        raise RuntimeError(f"No FQE tuning candidates refit successfully. {errors}".strip())
    selected_score, selected, selected_cfg, selected_model = min(scored, key=lambda row: row[0])
    baseline_rows = [row for row in scored if _is_baseline_candidate(row[1]) and row[1].family == selected.family]
    if bool(cfg.stable_fallback) and baseline_rows:
        baseline_score, baseline, baseline_cfg, baseline_model = min(baseline_rows, key=lambda row: row[0])
        if baseline.candidate_id != selected.candidate_id and _should_fallback_to_baseline(
            selected=selected,
            baseline=baseline,
            selected_score=float(selected_score),
            baseline_score=float(baseline_score),
            cfg=cfg,
        ):
            selected.metrics["stable_fallback_replaced_by_baseline"] = 1.0
            baseline.metrics["stable_fallback_selected"] = 1.0
            selected, selected_cfg, selected_model = baseline, baseline_cfg, baseline_model
    return selected, selected_cfg, selected_model


def _is_baseline_candidate(candidate: FQECandidateResult) -> bool:
    return str(candidate.candidate_id).endswith("_000")


def _should_fallback_to_baseline(
    *,
    selected: FQECandidateResult,
    baseline: FQECandidateResult,
    selected_score: float,
    baseline_score: float,
    cfg: FQETuningConfig,
) -> bool:
    if not (np.isfinite(selected_score) and np.isfinite(baseline_score)):
        return False
    if baseline_score > selected_score + float(cfg.fallback_score_tolerance):
        return False
    baseline_runtime = max(float(baseline.runtime_sec), 1e-9)
    selected_runtime = float(selected.runtime_sec)
    baseline_cal = float(baseline.metrics.get("calibration_error", float("inf")))
    selected_cal = float(selected.metrics.get("calibration_error", float("inf")))
    runtime_expensive = selected_runtime > float(cfg.fallback_runtime_ratio) * baseline_runtime
    calibration_no_better = baseline_cal <= selected_cal * 1.05
    return bool(runtime_expensive or calibration_no_better)


def _build_config(
    *,
    family: str,
    overrides: Dict[str, Any],
    space: FQESearchSpace,
    screen_fraction: float,
    seed: int,
    force_final: bool,
) -> BoostedFQEConfig | NeuralFQEConfig:
    overrides = {key: value for key, value in dict(overrides).items() if str(key) != "_meta"}
    if family == "boosted":
        cfg = _config_with_updates(space.boosted, overrides)
        cfg = replace(
            cfg,
            num_iterations=max(1, int(round(float(cfg.num_iterations) * float(screen_fraction)))),
            patience=max(1, int(round(float(cfg.patience) * max(float(screen_fraction), 0.5)))),
            refit_on_all_data=bool(force_final and cfg.refit_on_all_data),
            seed=int(seed),
            show_progress=False,
        )
        return cfg
    cfg = replace(space.neural, **dict(overrides))
    return replace(
        cfg,
        num_iterations=max(1, int(round(float(cfg.num_iterations) * float(screen_fraction)))),
        gradient_steps_per_iteration=max(1, int(round(float(cfg.gradient_steps_per_iteration) * float(screen_fraction)))),
        patience=max(1, int(round(float(cfg.patience) * max(float(screen_fraction), 0.5)))),
        seed=int(seed),
        show_progress=False,
    )


def _fit_family(
    *,
    family: str,
    mode: str,
    config: BoostedFQEConfig | NeuralFQEConfig,
    S: Array,
    A: Array | None,
    S_next: Array,
    A_next: Array | None,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
    next_action_weights: Array | None,
    categorical_feature: Sequence[int | str] | None,
) -> FQEModel | NeuralFQEModel:
    if family == "boosted":
        if mode == "value":
            return fit_value_lgbm(
                states=S,
                next_states=S_next,
                rewards=rewards,
                gamma=gamma,
                terminals=terminals,
                sample_weight=sample_weight,
                config=config,
                categorical_feature=categorical_feature,
            )
        assert A is not None and A_next is not None
        return fit_fqe_lgbm(
            states=S,
            actions=A,
            next_states=S_next,
            next_actions=A_next,
            rewards=rewards,
            gamma=gamma,
            terminals=terminals,
            sample_weight=sample_weight,
            next_action_weights=next_action_weights,
            config=config,
            categorical_feature=categorical_feature,
        )
    if mode == "value":
        return fit_value_neural(
            states=S,
            next_states=S_next,
            rewards=rewards,
            gamma=gamma,
            terminals=terminals,
            sample_weight=sample_weight,
            config=config,
        )
    assert A is not None and A_next is not None
    return fit_fqe_neural(
        states=S,
        actions=A,
        next_states=S_next,
        next_actions=A_next,
        rewards=rewards,
        gamma=gamma,
        terminals=terminals,
        sample_weight=sample_weight,
        next_action_weights=next_action_weights,
        config=config,
    )


def _predict_current(model: Any, mode: str, states: Array, actions: Array | None) -> Array:
    if mode == "value":
        return np.asarray(model.predict_value(states), dtype=np.float64).reshape(-1)
    if actions is None:
        raise ValueError("actions are required for Q-mode prediction.")
    return np.asarray(model.predict_q(states, actions), dtype=np.float64).reshape(-1)


def _predict_next(
    model: Any,
    mode: str,
    next_states: Array,
    next_actions: Array | None,
    next_action_weights: Array | None = None,
) -> Array:
    if mode == "value":
        return np.asarray(model.predict_value(next_states), dtype=np.float64).reshape(-1)
    if next_actions is None:
        raise ValueError("next_actions are required for Q-mode prediction.")
    arr = np.asarray(next_actions, dtype=np.float64)
    if arr.ndim == 2:
        if next_action_weights is not None:
            validate_action_weights(next_action_weights, n_rows=arr.shape[0], n_actions=1, name="next_action_weights")
        return np.asarray(model.predict_q(next_states, arr), dtype=np.float64).reshape(-1)
    if arr.ndim != 3:
        raise ValueError("next_actions must be 2D or 3D after validation.")
    preds = [np.asarray(model.predict_q(next_states, arr[:, idx, :]), dtype=np.float64).reshape(-1) for idx in range(arr.shape[1])]
    weights = validate_action_weights(
        next_action_weights,
        n_rows=arr.shape[0],
        n_actions=arr.shape[1],
        name="next_action_weights",
    )
    return weighted_action_expectation(np.stack(preds, axis=1), weights)


def _estimate_policy_value_or_none(
    model: Any,
    initial_states: Array | None,
    initial_actions: Array | None,
    initial_weights: Array | None,
) -> float | None:
    if initial_states is None:
        return None
    try:
        if getattr(model, "mode", "q") == "value":
            return float(model.estimate_policy_value(initial_states, initial_weights=initial_weights))
        if initial_actions is None:
            return None
        return float(model.estimate_policy_value(initial_states, initial_actions, initial_weights=initial_weights))
    except Exception:
        return None


def _calibration_error(
    *,
    predictions: Array,
    next_predictions: Array,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
) -> float:
    target = rewards + float(gamma) * (1.0 - terminals) * next_predictions
    residual = predictions - target
    scale = float(np.sqrt(np.average((target - np.average(target, weights=sample_weight)) ** 2, weights=sample_weight))) + 1e-12
    return float(abs(np.average(residual, weights=sample_weight)) / scale)


def _aggregate_fold_metrics(folds: Sequence[FQEFoldResult], *, runtime_sec: float) -> Dict[str, float]:
    if not folds:
        return {
            "bellman_risk": float("inf"),
            "policy_value_stability": float("nan"),
            "calibration_error": float("inf"),
            "runtime_sec": float(runtime_sec),
        }
    values = [fold.policy_value for fold in folds if fold.policy_value is not None and np.isfinite(float(fold.policy_value))]
    stability = float("nan")
    if len(values) >= 2:
        stability = float(np.std(values) / (abs(float(np.mean(values))) + 1e-12))
    return {
        "bellman_risk": _weighted_fold_mean(folds, "bellman_risk"),
        "policy_value_stability": stability,
        "calibration_error": _weighted_fold_mean(folds, "calibration_error"),
        "runtime_sec": float(runtime_sec),
    }


def _score_candidates(candidates: Sequence[FQECandidateResult], cfg: FQETuningConfig) -> None:
    finite_candidates = [candidate for candidate in candidates if not candidate.error and candidate.fold_results]
    if not finite_candidates:
        return
    stability_available = any(
        np.isfinite(candidate.metrics.get("policy_value_stability", float("nan"))) for candidate in finite_candidates
    )
    bellman_weight = float(cfg.score_bellman_weight)
    stability_weight = float(cfg.score_value_stability_weight) if stability_available else 0.0
    if not stability_available:
        bellman_weight += float(cfg.score_value_stability_weight)
    weights = {
        "bellman_risk": bellman_weight,
        "policy_value_stability": stability_weight,
        "calibration_error": float(cfg.score_calibration_weight),
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


def _validate_fqe_target_trajectory(
    *,
    mode: str,
    validation_states: Array | None,
    validation_actions: Array | None,
    validation_rewards: Array | None,
    validation_next_states: Array | None,
    validation_episode_ids: Array | None,
    validation_timestep: Array | None,
    validation_terminals: Array | None,
    validation_continuation: Array | None,
    validation_tail_actions: Array | None,
) -> _FQETargetTrajectory:
    if validation_states is None or validation_rewards is None or validation_next_states is None:
        raise ValueError("validation_states, validation_rewards, and validation_next_states are required for n_step_td scoring.")
    rewards = _as_1d_float(validation_rewards, "validation_rewards")
    states = _as_2d_float(validation_states, "validation_states", n_rows=rewards.shape[0])
    next_states = _as_2d_float(validation_next_states, "validation_next_states", n_rows=rewards.shape[0])
    actions = None
    tail_actions = None
    if mode == "q":
        if validation_actions is None:
            raise ValueError("validation_actions are required for Q-mode target validation.")
        actions = _as_2d_float(validation_actions, "validation_actions", n_rows=rewards.shape[0])
        if validation_tail_actions is not None:
            tail_actions = _as_2d_float(validation_tail_actions, "validation_tail_actions", n_rows=rewards.shape[0])
    if validation_continuation is not None and validation_terminals is not None:
        raise ValueError("Supply only one of validation_continuation or validation_terminals.")
    if validation_continuation is not None:
        continuation = np.asarray(validation_continuation, dtype=np.float64).reshape(-1)
        if continuation.shape[0] != rewards.shape[0]:
            raise ValueError("validation_continuation must have one entry per validation row.")
    elif validation_terminals is not None:
        continuation = 1.0 - _optional_terminals(validation_terminals, rewards.shape[0])
    else:
        raise ValueError("validation_terminals or validation_continuation is required for finite-prefix tail diagnostics.")
    if not np.all(np.isfinite(continuation)):
        raise ValueError("validation continuation indicators must be finite.")
    episode_ids = np.arange(rewards.shape[0]) if validation_episode_ids is None else np.asarray(validation_episode_ids).reshape(-1)
    if episode_ids.shape[0] != rewards.shape[0]:
        raise ValueError("validation_episode_ids must have one entry per validation row.")
    timesteps = np.arange(rewards.shape[0]) if validation_timestep is None else np.asarray(validation_timestep).reshape(-1)
    if timesteps.shape[0] != rewards.shape[0]:
        raise ValueError("validation_timestep must have one entry per validation row.")
    return _FQETargetTrajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        episode_ids=episode_ids,
        timesteps=timesteps,
        continuation=np.clip(continuation, 0.0, 1.0),
        tail_actions=tail_actions,
    )


def _fqe_target_trajectory_diagnostics(
    trajectory: _FQETargetTrajectory,
    *,
    gamma: float,
    seed: int,
) -> Dict[str, float | str]:
    returns = []
    horizons = []
    tail_masses = []
    for _, idx in _episode_slices(trajectory.episode_ids, trajectory.timesteps):
        rewards = trajectory.rewards[idx]
        discounts = float(gamma) ** np.arange(rewards.shape[0], dtype=np.float64)
        returns.append(float(np.sum(discounts * rewards)))
        horizons.append(float(rewards.shape[0]))
        tail_masses.append(float((float(gamma) ** rewards.shape[0]) * trajectory.continuation[idx[-1]]))
    return {
        "direct_target_return_mean": _mean(returns),
        "direct_target_return_se": _bootstrap_scalar_mean_se(np.asarray(returns, dtype=np.float64), seed=seed),
        "validation_episode_count": float(len(returns)),
        "validation_row_count": float(trajectory.rewards.shape[0]),
        "validation_horizon_mean": _mean(horizons),
        "validation_horizon_max": float(np.max(horizons)) if horizons else 0.0,
        "truncation_tail_mass_mean": _mean(tail_masses),
        "truncation_tail_mass_max": float(np.max(tail_masses)) if tail_masses else 0.0,
    }


def _score_fqe_target_trajectory(
    *,
    model: Any,
    mode: str,
    trajectory: _FQETargetTrajectory,
    gamma: float,
    seed: int,
) -> Dict[str, float]:
    predictions = _predict_current(model, mode, trajectory.states, trajectory.actions)
    targets = np.empty_like(predictions, dtype=np.float64)
    episode_losses = []
    for _, idx in _episode_slices(trajectory.episode_ids, trajectory.timesteps):
        last = int(idx[-1])
        tail_value = 0.0
        if float(trajectory.continuation[last]) > 0.0:
            if mode == "value":
                tail_value = float(np.asarray(model.predict_value(trajectory.next_states[last : last + 1]), dtype=np.float64).reshape(-1)[0])
            else:
                if trajectory.tail_actions is None:
                    raise ValueError("validation_tail_actions are required for nonterminal Q-mode target-validation tails.")
                tail_value = float(np.asarray(model.predict_q(trajectory.next_states[last : last + 1], trajectory.tail_actions[last : last + 1]), dtype=np.float64).reshape(-1)[0])
        running = float(trajectory.continuation[last]) * tail_value
        for row in idx[::-1]:
            running = float(trajectory.rewards[row]) + float(gamma) * running
            targets[row] = running
        residual_sq = (predictions[idx] - targets[idx]) ** 2
        episode_losses.append(float(np.mean(residual_sq)))
    row_losses = (predictions - targets) ** 2
    score = float(np.mean(row_losses)) if row_losses.size else float("inf")
    score_se = _bootstrap_scalar_mean_se(np.asarray(episode_losses, dtype=np.float64), seed=seed)
    return {
        "validation_score": score,
        "validation_score_se": score_se,
        "validation_residual_mean": float(np.mean(predictions - targets)) if targets.size else float("nan"),
        "validation_target_mean": float(np.mean(targets)) if targets.size else float("nan"),
        "validation_prediction_mean": float(np.mean(predictions)) if predictions.size else float("nan"),
        "validation_episode_loss_mean": _mean(episode_losses),
    }


def _episode_slices(episode_ids: Array, timesteps: Array) -> List[tuple[Any, Array]]:
    ids = np.asarray(episode_ids).reshape(-1)
    steps = np.asarray(timesteps).reshape(-1)
    out: List[tuple[Any, Array]] = []
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


def _select_target_validation_candidate(
    candidates: Sequence[FQETargetValidationCandidateResult],
    *,
    selection_rule: str,
) -> tuple[
    FQETargetValidationCandidateResult | None,
    FQETargetValidationCandidateResult | None,
    FQETargetValidationCandidateResult | None,
]:
    finite = [candidate for candidate in candidates if not candidate.error and np.isfinite(float(candidate.score))]
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


def _selected_min_candidate_id(candidates: Sequence[FQETargetValidationCandidateResult]) -> str:
    finite = [candidate for candidate in candidates if not candidate.error and np.isfinite(float(candidate.score))]
    if not finite:
        return ""
    return min(finite, key=lambda row: row.score).candidate_id


def _make_folds(n_rows: int, n_folds: int, seed: int, *, groups: Array | None) -> List[Array]:
    n = int(n_rows)
    k = int(n_folds)
    if k < 2:
        raise ValueError("n_folds must be at least 2.")
    if n < k:
        raise ValueError("n_folds cannot exceed the number of rows.")
    rng = np.random.default_rng(seed)
    if groups is None:
        folds = [fold.astype(np.int64, copy=False) for fold in np.array_split(rng.permutation(n), k)]
        _validate_nonempty_folds(folds)
        return folds
    group_arr = np.asarray(groups).reshape(-1)
    if group_arr.shape[0] != n:
        raise ValueError("groups must have the same number of rows as states.")
    unique_groups = np.unique(group_arr)
    if unique_groups.shape[0] < k:
        raise ValueError("group-aware CV requires at least one distinct group per fold.")
    shuffled = unique_groups[rng.permutation(unique_groups.shape[0])]
    group_folds = np.array_split(shuffled, k)
    folds = [np.flatnonzero(np.isin(group_arr, group_fold)).astype(np.int64, copy=False) for group_fold in group_folds]
    _validate_nonempty_folds(folds)
    return folds


def _complement_indices(n_rows: int, valid_idx: Array) -> Array:
    mask = np.ones(int(n_rows), dtype=bool)
    mask[np.asarray(valid_idx, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("inf")


def _weighted_fold_mean(folds: Sequence[FQEFoldResult], metric: str) -> float:
    values = []
    weights = []
    for fold in folds:
        value = float(getattr(fold, metric))
        weight = float(fold.validation_weight_sum)
        if np.isfinite(value) and np.isfinite(weight) and weight > 0.0:
            values.append(value)
            weights.append(weight)
    if not values:
        return float("inf")
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def _validate_nonempty_folds(folds: Sequence[Array]) -> None:
    if any(np.asarray(fold).shape[0] == 0 for fold in folds):
        raise ValueError("cross-validation produced an empty validation fold.")
