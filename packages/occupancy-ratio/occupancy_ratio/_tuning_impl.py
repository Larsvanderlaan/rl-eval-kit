from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


Array = np.ndarray

__all__ = [
    "CandidateResult",
    "FoldResult",
    "OccupancySearchSpace",
    "OccupancyTuningConfig",
    "OccupancyTuningResult",
    "tune_occupancy_ratio",
    "tune_occupancy_ratio_auto",
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
    score_moment_balance_weight: float = 0.55
    score_validation_weight: float = 0.25
    score_reward_stability_weight: float = 0.0
    score_weight_quality_weight: float = 0.15
    score_runtime_weight: float = 0.05
    moment_rff_features: int = 16
    moment_geometry_features: int = 8
    moment_value_iterations: int = 30
    moment_value_patience: int = 5
    moment_max_group_weight: float = 0.25
    moment_extra_blocks: Sequence[str] = ()
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
    ) for candidate in candidates]
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
    ) for candidate in candidates if candidate["candidate_id"] in promoted_ids]
    _score_candidates(full_candidates, cfg)
    for candidate in full_candidates:
        candidate.promoted = True
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
    )


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
) -> Tuple[CandidateResult, Dict[str, Any], Any]:
    scored: List[Tuple[float, CandidateResult, Dict[str, Any], Any]] = []
    ordered = sorted(candidates, key=lambda row: row.score)
    for rank, candidate in enumerate(ordered):
        configs = _build_configs(
            family=candidate.family,
            overrides=candidate.overrides,
            space=space,
            screen_fraction=1.0,
            seed=_refit_seed(candidate, space, cfg, rank),
        )
        try:
            refit_initial_mode = _candidate_mode(candidate.overrides, "initial_ratio_mode", initial_ratio_mode)
            refit_one_step_mode = _candidate_mode(candidate.overrides, "one_step_ratio_mode", one_step_ratio_mode)
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
            )
            weights = model.predict_state_action_ratio(S, A, clip=True)
            final_metrics = _final_weight_metrics(weights, _scoring_weight_config(configs), action_shift=_action_shift(A, A_pi))
            candidate.metrics.update({f"final_{key}": value for key, value in final_metrics.items()})
            final_score = float(candidate.score) + _final_refit_penalty(final_metrics)
            candidate.metrics["final_selection_score"] = float(final_score)
            scored.append((final_score, candidate, configs, model))
        except Exception as exc:
            candidate.metrics["final_refit_failed"] = 1.0
            candidate.error = candidate.error or f"{type(exc).__name__}: {exc}"
    if not scored:
        errors = "; ".join(candidate.error for candidate in ordered if candidate.error)
        raise RuntimeError(f"No tuning candidates refit successfully. {errors}".strip())
    selected_score, selected, configs, model = min(scored, key=lambda row: row[0])
    baseline_rows = [row for row in scored if _is_baseline_candidate(row[1]) and row[1].family == selected.family]
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


def _is_baseline_candidate(candidate: CandidateResult) -> bool:
    return str(candidate.candidate_id).endswith("_000")


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

    selected_mb = _metric_value(selected, "moment_balance")
    baseline_mb = _metric_value(baseline, "moment_balance")
    if not (np.isfinite(selected_mb) and np.isfinite(baseline_mb)):
        return False
    improvement = baseline_mb - selected_mb
    selected_se = _metric_value(selected, "moment_balance_se")
    baseline_se = _metric_value(baseline, "moment_balance_se")
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
    for family in tuple(str(family) for family in cfg.families):
        raw = (
            list(space.boosted_candidates)
            if family == "boosted" and space.boosted_candidates is not None
            else list(space.neural_candidates)
            if family == "neural" and space.neural_candidates is not None
            else _default_family_candidates(family, space, has_initial_states=has_initial_states)
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
        if has_initial_states:
            labels.append("neural_factored_source")
        labels.extend(
            [
                "neural_small_width",
                "neural_large_width",
                "neural_tight_cap",
                "neural_tight_nuisance_cap",
                "neural_loose_nuisance_cap",
                "neural_tight_cap_logistic_nuisance",
            ]
        )
        return labels[idx] if idx < len(labels) else f"neural_candidate_{idx:03d}"
    if family == "boosted":
        return "boosted_stable" if idx == 0 else f"boosted_candidate_{idx:03d}"
    return f"{family}_candidate_{idx:03d}"


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
) -> List[Dict[str, Dict[str, Any]]]:
    if family == "boosted":
        return _boosted_default_candidates(has_initial_states=has_initial_states)
    return _neural_default_candidates(space, has_initial_states=has_initial_states)


def _boosted_default_candidates(*, has_initial_states: bool) -> List[Dict[str, Dict[str, Any]]]:
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
    if has_initial_states:
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
    if has_initial_states:
        variants.append(({}, {}, {"initial_ratio_mode": "factored", "one_step_ratio_mode": "factored"}))
    variants.extend([
        ({"hidden_dims": small_dims}, {"hidden_dims": small_dims}),
        ({"hidden_dims": large_dims}, {"hidden_dims": large_dims}),
        ({"occupancy_ratio_max": 25.0}, {}),
        ({}, {"prediction_max": 25.0}),
        ({}, {"prediction_max": 100.0}),
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
        return max(1, min(limit, 2))
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
) -> CandidateResult:
    fold_results: List[FoldResult] = []
    start = time.perf_counter()
    family = str(candidate["family"])
    overrides = dict(candidate["overrides"])
    error = ""
    try:
        fold_initial_mode = _candidate_mode(overrides, "initial_ratio_mode", initial_ratio_mode)
        fold_one_step_mode = _candidate_mode(overrides, "one_step_ratio_mode", one_step_ratio_mode)
        for fold_id, valid_idx in enumerate(folds):
            train_idx = _complement_indices(S.shape[0], valid_idx)
            configs = _build_configs(
                family=family,
                overrides=overrides,
                space=space,
                screen_fraction=screen_fraction,
                seed=seed + 10_003 * (fold_id + 1),
            )
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
            "validation_loss": float("inf"),
            "reward_stability": float("nan"),
            "weight_quality": float("inf"),
            "runtime_sec": float(runtime_sec),
        }
    moment_balance = _mean([fold.moment_balance for fold in folds])
    moment_max_group = _mean([fold.moment_balance_max_group for fold in folds])
    moment_balance_se = _se([fold.moment_balance for fold in folds])
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
        "validation_loss": validation,
        "reward_stability": reward_stability,
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
    feature_builder = _FoldFeatureBuilder(
        S_train=S[train_idx],
        A_train=A[train_idx],
        rewards_train=None if rewards is None else rewards[train_idx],
        gamma=float(gamma),
        seed=int(seed),
        geometry_features=int(cfg.moment_geometry_features),
        rff_features=int(cfg.moment_rff_features),
        value_iterations=int(cfg.moment_value_iterations),
        value_patience=int(cfg.moment_value_patience),
        S_next_train=S_next[train_idx],
        A_next_train=A_pi[train_idx] if A_pi_next is None else A_pi_next[train_idx],
        A_target_train=A_pi[train_idx],
        extra_blocks=tuple(str(block) for block in cfg.moment_extra_blocks),
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
    group_scores = {
        name: _moment_group_score(weights, float(gamma), block)
        for name, block in blocks.items()
        if np.asarray(block[0]).shape[1] > 0
    }
    finite = np.asarray([score for score in group_scores.values() if np.isfinite(score)], dtype=np.float64)
    if finite.size == 0:
        return {"moment_balance": float("inf"), "moment_balance_max_group": float("inf")}
    return {
        "moment_balance": float(np.mean(finite) + float(cfg.moment_max_group_weight) * np.max(finite)),
        "moment_balance_max_group": float(np.max(finite)),
    }


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
        phi_train = self._basis_from_z(z_train)
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
