from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Sequence

import numpy as np

from fqe.fit_fqe import Array, FQEModel
from fqe.fit_neural_fqe import NeuralFQEModel


@dataclass
class FQEStagedCVFoldTelemetry:
    candidate_id: str
    family: str
    stage: int
    iteration: int
    fold: int
    runtime_sec: float
    bootstrapped_loss: float
    bootstrapped_loss_se: float
    bellman_risk: float
    n_train: int
    n_validation: int
    active: bool = True
    pruned: bool = False
    selected: bool = False
    baseline_forced_eval: bool = False


@dataclass
class FQEStagedCVStageTelemetry:
    candidate_id: str
    family: str
    stage: int
    iteration: int
    bootstrapped_loss: float
    bootstrapped_loss_se: float
    selected_min_loss: bool = False
    kept_by_one_se: bool = False
    pruned: bool = False
    baseline_forced: bool = False
    active: bool = True
    selected: bool = False
    baseline_forced_eval: bool = False
    complexity_group: str = ""
    complexity_rank: str = ""
    complexity_source: str = "none"
    stage_best_candidate_id: str = ""
    stage_best_complexity_rank: str = ""
    outside_one_se: bool = False
    strictly_simpler_than_stage_best: bool = False
    prune_reason: str = ""


@dataclass
class FQEStagedCVCandidateTelemetry:
    candidate_id: str
    family: str
    stage: int
    iteration: int
    bootstrapped_loss: float
    bootstrapped_loss_se: float
    selected_min_loss: bool
    kept_by_one_se: bool
    pruned: bool
    baseline_forced: bool
    final_stage: bool
    active: bool = True
    selected: bool = False
    baseline_forced_eval: bool = False
    complexity_group: str = ""
    complexity_rank: str = ""
    complexity_source: str = "none"
    stage_best_candidate_id: str = ""
    stage_best_complexity_rank: str = ""
    outside_one_se: bool = False
    strictly_simpler_than_stage_best: bool = False
    prune_reason: str = ""


__all__ = [
    "FQEStagedCVCandidateTelemetry",
    "FQEStagedCVFoldTelemetry",
    "FQEStagedCVStageTelemetry",
    "monotone_one_se_prune",
    "tune_fqe_staged_bootstrap_cv",
]


def tune_fqe_staged_bootstrap_cv(
    *,
    S: Array,
    A: Array | None,
    S_next: Array,
    A_next: Array | None,
    rewards: Array,
    gamma: float,
    terminals: Array,
    sample_weight: Array,
    next_action_weights: Array | None = None,
    S_initial: Array | None,
    A_initial: Array | None,
    initial_weights: Array | None,
    groups: Array | None,
    search_space: Any,
    config: Any,
    mode: str,
    categorical_feature: Sequence[int | str] | None,
) -> Any:
    """Run staged bootstrapped-loss CV for FQE candidates.

    Each candidate is fit at the requested stage iteration counts and scored on
    held-out Bellman residuals using that candidate's own bootstrap target. A
    one-standard-error rule prunes candidates between stages while always
    keeping the family baseline candidate in telemetry.
    """

    from fqe import tuning as tuning

    folds = tuning._make_folds(S.shape[0], int(config.cv_folds), int(config.seed), groups=groups)
    candidates = tuning._make_candidates(search_space, config)
    input_dim = int(S.shape[1] if str(mode) == "value" else S.shape[1] + (0 if A is None else A.shape[1]))
    complexity = _candidate_complexity_map(candidates, search_space, input_dim=input_dim)
    stages = _resolve_stage_iterations(candidates, search_space, config)
    active_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    baseline_ids = {str(candidate["candidate_id"]) for candidate in candidates if str(candidate["candidate_id"]).endswith("_000")}
    candidate_results: dict[str, Any] = {}
    fold_telemetry: List[FQEStagedCVFoldTelemetry] = []
    stage_telemetry: List[FQEStagedCVStageTelemetry] = []

    for stage_idx, iteration in enumerate(stages):
        stage_number = int(iteration)
        stage_rows: List[FQEStagedCVStageTelemetry] = []
        evaluation_ids = set(active_ids)
        if bool(getattr(config, "staged_cv_always_evaluate_baseline", True)):
            evaluation_ids.update(baseline_ids)
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id not in evaluation_ids:
                continue
            forced_baseline = bool(candidate_id in baseline_ids and candidate_id not in active_ids)
            result, stage_folds = _evaluate_staged_candidate(
                candidate=candidate,
                stage=stage_number,
                iteration=iteration,
                folds=folds,
                mode=mode,
                S=S,
                A=A,
                S_next=S_next,
                A_next=A_next,
                rewards=rewards,
                gamma=gamma,
                terminals=terminals,
                sample_weight=sample_weight,
                next_action_weights=next_action_weights,
                S_initial=S_initial,
                A_initial=A_initial,
                initial_weights=initial_weights,
                groups=groups,
                search_space=search_space,
                config=config,
                seed=int(config.seed) + 193_003 + stage_idx * 31_337,
                categorical_feature=categorical_feature,
            )
            candidate_results[candidate_id] = result
            for fold_row in stage_folds:
                fold_row.baseline_forced_eval = forced_baseline
            fold_telemetry.extend(stage_folds)
            stage_row = FQEStagedCVStageTelemetry(
                candidate_id=candidate_id,
                family=str(candidate["family"]),
                stage=stage_number,
                iteration=int(iteration),
                bootstrapped_loss=float(result.metrics.get("staged_bootstrap_loss", float("inf"))),
                bootstrapped_loss_se=float(result.metrics.get("staged_bootstrap_loss_se", 0.0)),
                baseline_forced=forced_baseline,
                active=not forced_baseline,
                baseline_forced_eval=forced_baseline,
                complexity_group=str(complexity.get(candidate_id, {}).get("group", "")),
                complexity_rank=str(complexity.get(candidate_id, {}).get("rank_repr", "")),
                complexity_source=str(complexity.get(candidate_id, {}).get("source", "none")),
            )
            stage_rows.append(stage_row)
            stage_telemetry.append(stage_row)

        if stage_rows:
            one_se_multiplier = (
                float(getattr(config, "staged_cv_one_se_multiplier", 1.0))
                if bool(config.staged_cv_one_se_pruning)
                else 0.0
            )
            active_ids, _, _ = monotone_one_se_prune(
                stage_rows,
                active_ids,
                complexity,
                one_se_multiplier,
                max(1, int(getattr(config, "staged_cv_min_survivors", 1))),
            )

    final_iteration = int(stages[-1])
    final_rows = [
        row
        for row in stage_telemetry
        if int(row.iteration) == final_iteration
        and row.candidate_id in active_ids
        and row.candidate_id in candidate_results
        and np.isfinite(float(row.bootstrapped_loss))
    ]
    if not final_rows:
        final_rows = [
            row
            for row in stage_telemetry
            if int(row.iteration) == final_iteration
            and row.candidate_id in candidate_results
            and np.isfinite(float(row.bootstrapped_loss))
        ]
        if not final_rows:
            errors = "; ".join(result.error for result in candidate_results.values() if result.error)
            raise RuntimeError(f"No staged FQE tuning candidates completed successfully. {errors}".strip())
    selected_stage = min(final_rows, key=lambda row: row.bootstrapped_loss)
    selected_stage.selected = True
    selected = candidate_results[selected_stage.candidate_id]
    selected.score = float(selected_stage.bootstrapped_loss)
    selected.selected = True
    selected_config = _build_final_config(
        candidate=selected,
        search_space=search_space,
        config=config,
        seed=int(config.seed) + 707_707,
    )
    model: FQEModel | NeuralFQEModel | None = None
    if bool(config.refit):
        model = tuning._fit_family(
            family=selected.family,
            mode=mode,
            config=selected_config,
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

    candidates_out = list(candidate_results.values())
    final_ids = {row.candidate_id for row in final_rows}
    for result in candidates_out:
        result.promoted = result.candidate_id in final_ids
        result.metrics["staged_bootstrap_cv"] = 1.0
        result.metrics["staged_cv_final_iteration"] = float(final_iteration)
    tuning_folds = [
        tuning.FQEFoldResult(
            candidate_id=row.candidate_id,
            family=row.family,
            budget_stage=f"staged_{row.stage}",
            fold=row.fold,
            runtime_sec=row.runtime_sec,
            bellman_risk=row.bellman_risk,
            calibration_error=0.0,
            policy_value=None,
            n_train=row.n_train,
            n_validation=row.n_validation,
        )
        for row in fold_telemetry
    ]
    for result in candidates_out:
        result.metrics["staged_cv_stage_count"] = float(len(stages))
    _attach_staged_telemetry(candidates_out, stage_telemetry)
    staged_rows = _staged_rows(stage_telemetry, fold_telemetry)
    return tuning.FQETuningResult(
        selected_family=selected.family,
        selected_candidate_id=selected.candidate_id,
        selected_overrides=selected.overrides,
        selected_config=selected_config,
        candidates=candidates_out,
        folds=tuning_folds,
        model=model,
        config=config,
        staged_cv_rows_data=staged_rows,
    )


def _evaluate_staged_candidate(
    *,
    candidate: Dict[str, Any],
    stage: int,
    iteration: int,
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
    groups: Array | None,
    search_space: Any,
    config: Any,
    seed: int,
    categorical_feature: Sequence[int | str] | None,
) -> tuple[Any, List[FQEStagedCVFoldTelemetry]]:
    from fqe import tuning as tuning

    start = time.perf_counter()
    family = str(candidate["family"])
    overrides = dict(candidate["overrides"])
    fold_rows: List[FQEStagedCVFoldTelemetry] = []
    losses: List[float] = []
    error = ""
    try:
        for fold_id, valid_idx in enumerate(folds):
            train_idx = tuning._complement_indices(S.shape[0], valid_idx)
            cfg = _build_stage_config(
                family=family,
                overrides=overrides,
                search_space=search_space,
                iteration=iteration,
                seed=seed + 10_003 * (fold_id + 1),
            )
            fold_start = time.perf_counter()
            model = tuning._fit_family(
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
            if int(iteration) <= 1:
                prev_next_pred = np.zeros(valid_idx.shape[0], dtype=np.float64)
            else:
                prev_cfg = _build_stage_config(
                    family=family,
                    overrides=overrides,
                    search_space=search_space,
                    iteration=max(1, int(iteration) - 1),
                    seed=seed + 503_021 * (fold_id + 1),
                )
                prev_model = tuning._fit_family(
                    family=family,
                    mode=mode,
                    config=prev_cfg,
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
                prev_next_pred = tuning._predict_next(
                    prev_model,
                    mode,
                    S_next[valid_idx],
                    None if A_next is None else A_next[valid_idx],
                    None if next_action_weights is None else next_action_weights[valid_idx],
                )
            pred = tuning._predict_current(model, mode, S[valid_idx], None if A is None else A[valid_idx])
            target = rewards[valid_idx] + float(gamma) * (1.0 - terminals[valid_idx]) * prev_next_pred
            residual = pred - target
            row_losses = residual * residual
            risk = tuning._bellman_risk(
                predictions=pred,
                next_predictions=prev_next_pred,
                rewards=rewards[valid_idx],
                gamma=gamma,
                terminals=terminals[valid_idx],
                sample_weight=sample_weight[valid_idx],
            )
            loss = float(np.average(row_losses, weights=sample_weight[valid_idx]))
            loss_se = _bootstrap_weighted_mean_se(
                row_losses,
                sample_weight[valid_idx],
                seed=seed + 97_003 + fold_id,
                n_bootstrap=_staged_n_bootstrap(config),
                groups=None if groups is None else np.asarray(groups)[valid_idx],
            )
            losses.extend([float(value) for value in row_losses if np.isfinite(float(value))])
            fold_rows.append(
                FQEStagedCVFoldTelemetry(
                    candidate_id=str(candidate["candidate_id"]),
                    family=family,
                    stage=int(stage),
                    iteration=int(iteration),
                    fold=int(fold_id),
                    runtime_sec=float(time.perf_counter() - fold_start),
                    bootstrapped_loss=loss,
                    bootstrapped_loss_se=loss_se,
                    bellman_risk=float(risk),
                    n_train=int(train_idx.shape[0]),
                    n_validation=int(valid_idx.shape[0]),
                )
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime = float(time.perf_counter() - start)
    stage_loss = _weighted_fold_mean([row.bootstrapped_loss for row in fold_rows], [row.n_validation for row in fold_rows])
    stage_se = _bootstrap_weighted_mean_se(
        np.asarray(losses, dtype=np.float64),
        np.ones(len(losses), dtype=np.float64),
        seed=seed + 881_017,
        n_bootstrap=_staged_n_bootstrap(config),
    )
    metrics = {
        "staged_bootstrap_loss": float(stage_loss),
        "staged_bootstrap_loss_se": float(stage_se),
        "bellman_risk": _weighted_fold_mean([row.bellman_risk for row in fold_rows], [row.n_validation for row in fold_rows]),
        "runtime_sec": runtime,
        "policy_value_stability": float("nan"),
        "calibration_error": float("inf"),
    }
    policy_value = None
    if S_initial is not None:
        policy_value = np.nan
    result = tuning.FQECandidateResult(
        candidate_id=str(candidate["candidate_id"]),
        family=family,
        budget_stage=f"staged_{stage}",
        overrides=overrides,
        fold_results=[],
        metrics=metrics,
        score=float(stage_loss),
        runtime_sec=runtime,
        error=error,
    )
    if policy_value is not None:
        result.metrics["policy_value_stability"] = float("nan")
    return result, fold_rows


def _resolve_stage_iterations(candidates: Sequence[Dict[str, Any]], search_space: Any, config: Any) -> tuple[int, ...]:
    if config.staged_cv_iterations is not None:
        if isinstance(config.staged_cv_iterations, (int, np.integer)):
            return tuple(range(1, int(config.staged_cv_iterations) + 1))
        return tuple(sorted(set(int(value) for value in config.staged_cv_iterations)))
    max_iterations = 1
    for candidate in candidates:
        family = str(candidate["family"])
        base = search_space.boosted if family == "boosted" else search_space.neural
        value = int(dict(candidate["overrides"]).get("num_iterations", int(base.num_iterations)))
        max_iterations = max(max_iterations, value)
    max_iterations = min(max_iterations, 5)
    if max_iterations <= 3:
        return (max_iterations,)
    first = max(1, max_iterations // 3)
    second = max(first + 1, (2 * max_iterations) // 3)
    return tuple(dict.fromkeys((first, second, max_iterations)))


def _build_stage_config(
    *,
    family: str,
    overrides: Dict[str, Any],
    search_space: Any,
    iteration: int,
    seed: int,
) -> Any:
    from dataclasses import replace
    from fqe.fit_fqe import _config_with_updates

    overrides = _strip_candidate_meta(overrides)
    if family == "boosted":
        cfg = _config_with_updates(search_space.boosted, overrides)
        return replace(
            cfg,
            num_iterations=int(iteration),
            patience=max(1, min(int(cfg.patience), int(iteration))),
            refit_on_all_data=False,
            seed=int(seed),
            show_progress=False,
        )
    cfg = replace(search_space.neural, **dict(overrides))
    return replace(
        cfg,
        num_iterations=int(iteration),
        patience=max(1, min(int(cfg.patience), int(iteration))),
        seed=int(seed),
        show_progress=False,
    )


def _build_final_config(*, candidate: Any, search_space: Any, config: Any, seed: int) -> Any:
    from fqe import tuning as tuning

    return tuning._build_config(
        family=candidate.family,
        overrides=candidate.overrides,
        space=search_space,
        screen_fraction=1.0,
        seed=int(seed),
        force_final=True,
    )


def _attach_staged_telemetry(candidates: Sequence[Any], stages: Sequence[FQEStagedCVStageTelemetry]) -> None:
    by_candidate: dict[str, List[FQEStagedCVStageTelemetry]] = {}
    final_stage = max((row.stage for row in stages), default=-1)
    for row in stages:
        by_candidate.setdefault(row.candidate_id, []).append(row)
    for candidate in candidates:
        rows = by_candidate.get(candidate.candidate_id, [])
        candidate.metrics["staged_cv_candidate_telemetry"] = str(
            [
                asdict(
                    FQEStagedCVCandidateTelemetry(
                        candidate_id=row.candidate_id,
                        family=row.family,
                        stage=row.stage,
                        iteration=row.iteration,
                        bootstrapped_loss=row.bootstrapped_loss,
                        bootstrapped_loss_se=row.bootstrapped_loss_se,
                        selected_min_loss=row.selected_min_loss,
                        kept_by_one_se=row.kept_by_one_se,
                        pruned=row.pruned,
                        baseline_forced=row.baseline_forced,
                        final_stage=row.stage == final_stage,
                        active=row.active,
                        selected=row.selected,
                        baseline_forced_eval=row.baseline_forced_eval,
                        complexity_group=row.complexity_group,
                        complexity_rank=row.complexity_rank,
                        complexity_source=row.complexity_source,
                        stage_best_candidate_id=row.stage_best_candidate_id,
                        stage_best_complexity_rank=row.stage_best_complexity_rank,
                        outside_one_se=row.outside_one_se,
                        strictly_simpler_than_stage_best=row.strictly_simpler_than_stage_best,
                        prune_reason=row.prune_reason,
                    )
                )
                for row in rows
            ]
        )


def monotone_one_se_prune(
    stage_rows: Sequence[Any],
    active_ids: set[str],
    complexity: dict[str, dict[str, Any]],
    one_se_multiplier: float,
    min_survivors: int,
) -> tuple[set[str], str, float]:
    """Prune only active candidates strictly simpler than the one-SE stage best."""

    active = {str(candidate_id) for candidate_id in active_ids}
    finite_rows = [
        row
        for row in stage_rows
        if str(getattr(row, "candidate_id", "")) in active and np.isfinite(_stage_row_loss(row))
    ]
    if not finite_rows:
        return active, "", float("inf")
    best = min(finite_rows, key=_stage_row_loss)
    best_id = str(best.candidate_id)
    best_desc = complexity.get(best_id, {})
    best_loss = float(_stage_row_loss(best))
    best_se = max(float(_stage_row_se(best)), 0.0)
    threshold = best_loss + float(one_se_multiplier) * best_se
    kept: set[str] = set()
    decisions: dict[str, str] = {}
    for row in stage_rows:
        candidate_id = str(getattr(row, "candidate_id", ""))
        was_active = candidate_id in active
        loss = float(_stage_row_loss(row))
        outside = (not np.isfinite(loss)) or loss > threshold
        desc = complexity.get(candidate_id, {})
        simpler = bool(outside and _strictly_simpler(desc, best_desc))
        if was_active:
            if candidate_id == best_id:
                kept.add(candidate_id)
                reason = "min_loss"
            elif not outside:
                kept.add(candidate_id)
                reason = "within_one_se"
            elif simpler:
                reason = "outside_one_se_simpler"
            elif _same_complexity_group(desc, best_desc):
                kept.add(candidate_id)
                reason = "protected_larger_or_equal"
            else:
                kept.add(candidate_id)
                reason = "protected_incomparable"
        else:
            reason = "baseline_forced_eval" if bool(getattr(row, "baseline_forced_eval", False)) else "inactive"
        decisions[candidate_id] = reason
        _set_if_present(row, "selected_min_loss", candidate_id == best_id)
        _set_if_present(row, "kept_by_one_se", bool(was_active and not outside))
        _set_if_present(row, "outside_one_se", bool(outside))
        _set_if_present(row, "strictly_simpler_than_stage_best", bool(simpler))
        _set_if_present(row, "stage_best_candidate_id", best_id)
        _set_if_present(row, "stage_best_complexity_rank", str(best_desc.get("rank_repr", "")))
        _set_if_present(row, "prune_reason", reason)
        _set_if_present(row, "complexity_group", str(desc.get("group", "")))
        _set_if_present(row, "complexity_rank", str(desc.get("rank_repr", "")))
        _set_if_present(row, "complexity_source", str(desc.get("source", "none")))
    if not kept:
        kept = {best_id}
    min_survivors = max(1, int(min_survivors))
    if len(kept) < min_survivors:
        for row in sorted(finite_rows, key=_stage_row_loss):
            candidate_id = str(row.candidate_id)
            if candidate_id not in kept:
                kept.add(candidate_id)
                decisions[candidate_id] = "min_survivor_readd"
            if len(kept) >= min_survivors:
                break
    for row in stage_rows:
        candidate_id = str(getattr(row, "candidate_id", ""))
        was_active = candidate_id in active
        is_kept = candidate_id in kept
        _set_if_present(row, "active", bool(is_kept))
        _set_if_present(row, "pruned", bool(was_active and not is_kept))
        if decisions.get(candidate_id) == "min_survivor_readd":
            _set_if_present(row, "prune_reason", "min_survivor_readd")
    return kept, best_id, threshold


def _candidate_complexity_map(
    candidates: Sequence[Dict[str, Any]],
    search_space: Any,
    *,
    input_dim: int,
) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): _candidate_complexity(candidate, search_space, input_dim=input_dim)
        for candidate in candidates
    }


def _candidate_complexity(candidate: Dict[str, Any], search_space: Any, *, input_dim: int) -> dict[str, Any]:
    family = str(candidate.get("family", ""))
    overrides = dict(candidate.get("overrides", {}))
    explicit = _explicit_complexity(family, overrides)
    if explicit is not None:
        return explicit
    clean = _strip_candidate_meta(overrides)
    try:
        if family == "neural":
            from dataclasses import replace

            cfg = replace(search_space.neural, **clean)
            dims = tuple(int(width) for width in cfg.hidden_dims)
            return _descriptor(
                group=f"{family}:mlp_parameter_count",
                rank=_mlp_parameter_count(int(input_dim), dims, output_dim=1),
                source="inferred",
            )
        if family == "boosted":
            from fqe.fit_fqe import _config_with_updates

            cfg = _config_with_updates(search_space.boosted, clean)
            params = dict(getattr(cfg, "lgb_params", {}) or {})
            return _descriptor(
                group=f"{family}:tree_capacity",
                rank=_boosted_capacity_rank(params, trees_per_iteration=int(cfg.trees_per_iteration)),
                source="inferred",
            )
    except Exception:
        pass
    return _descriptor(group="", rank=None, source="none")


def _explicit_complexity(family: str, overrides: Dict[str, Any]) -> dict[str, Any] | None:
    meta = overrides.get("_meta")
    if not isinstance(meta, dict) or "complexity_rank" not in meta:
        return None
    return _descriptor(
        group=str(meta.get("complexity_group", f"{family}:explicit")),
        rank=meta.get("complexity_rank"),
        source="explicit",
    )


def _strip_candidate_meta(overrides: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict(overrides).items() if str(key) != "_meta"}


def _descriptor(*, group: str, rank: Any, source: str) -> dict[str, Any]:
    rank_tuple = _rank_tuple(rank)
    return {
        "group": str(group),
        "rank": rank_tuple,
        "rank_repr": "" if rank_tuple is None else "x".join(f"{value:g}" for value in rank_tuple),
        "source": str(source),
    }


def _boosted_capacity_rank(params: Dict[str, Any], *, trees_per_iteration: int) -> tuple[float, ...]:
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


def _rank_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = (value,)
    try:
        out = tuple(float(item) for item in values)
    except (TypeError, ValueError):
        return None
    if not out or not all(np.isfinite(item) for item in out):
        return None
    return out


def _strictly_simpler(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _same_complexity_group(left, right):
        return False
    left_rank = left.get("rank")
    right_rank = right.get("rank")
    if left_rank is None or right_rank is None or len(left_rank) != len(right_rank):
        return False
    return all(a <= b for a, b in zip(left_rank, right_rank)) and any(a < b for a, b in zip(left_rank, right_rank))


def _same_complexity_group(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(left.get("group")) and left.get("group") == right.get("group")


def _stage_row_loss(row: Any) -> float:
    for name in ("bootstrapped_loss", "loss_mean", "stage_loss", "loss"):
        if hasattr(row, name):
            try:
                return float(getattr(row, name))
            except (TypeError, ValueError):
                return float("inf")
    return float("inf")


def _stage_row_se(row: Any) -> float:
    for name in ("bootstrapped_loss_se", "loss_se", "stage_loss_se"):
        if hasattr(row, name):
            try:
                value = float(getattr(row, name))
            except (TypeError, ValueError):
                return 0.0
            return value if np.isfinite(value) else 0.0
    return 0.0


def _set_if_present(row: Any, name: str, value: Any) -> None:
    if hasattr(row, name):
        setattr(row, name, value)


def _bootstrap_weighted_mean_se(
    values: Array,
    weights: Array,
    *,
    seed: int,
    n_bootstrap: int,
    groups: Array | None = None,
) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    group_arr = None if groups is None else np.asarray(groups).reshape(-1)
    if group_arr is not None and group_arr.shape[0] != arr.shape[0]:
        group_arr = None
    mask = np.isfinite(arr) & np.isfinite(weight_arr) & (weight_arr >= 0.0)
    arr = arr[mask]
    weight_arr = weight_arr[mask]
    if group_arr is not None:
        group_arr = group_arr[mask]
        group_values = []
        group_weights = []
        for group in np.unique(group_arr):
            idx = group_arr == group
            total = float(np.sum(weight_arr[idx]))
            if total <= 0.0:
                group_values.append(float(np.mean(arr[idx])))
                group_weights.append(float(np.sum(idx)))
            else:
                group_values.append(float(np.average(arr[idx], weights=weight_arr[idx])))
                group_weights.append(total)
        arr = np.asarray(group_values, dtype=np.float64)
        weight_arr = np.asarray(group_weights, dtype=np.float64)
    if arr.shape[0] <= 1 or int(n_bootstrap) <= 0:
        return 0.0
    if float(np.sum(weight_arr)) <= 0.0:
        weight_arr = np.ones_like(arr)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sample = rng.integers(0, arr.shape[0], size=arr.shape[0])
        boot[idx] = float(np.average(arr[sample], weights=weight_arr[sample]))
    return float(np.std(boot, ddof=1))


def _weighted_fold_mean(values: Sequence[float], counts: Sequence[int]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.float64)
    mask = np.isfinite(arr) & np.isfinite(count_arr) & (count_arr > 0.0)
    if not np.any(mask):
        return float("inf")
    return float(np.average(arr[mask], weights=count_arr[mask]))


def _staged_n_bootstrap(config: Any) -> int:
    legacy = getattr(config, "staged_cv_bootstrap_samples", None)
    if legacy is not None:
        return int(legacy)
    return int(getattr(config, "staged_cv_n_bootstrap", 200))


def _staged_rows(
    stages: Sequence[FQEStagedCVStageTelemetry],
    folds: Sequence[FQEStagedCVFoldTelemetry],
) -> List[Dict[str, Any]]:
    by_candidate_stage = {(row.candidate_id, int(row.stage)): row for row in stages}
    rows: List[Dict[str, Any]] = []
    for row in stages:
        rows.append(
            {
                "row_type": "candidate_stage",
                "stage": int(row.stage),
                "iteration": int(row.iteration),
                "candidate_id": row.candidate_id,
                "family": row.family,
                "fold": "",
                "stage_loss": float(row.bootstrapped_loss),
                "stage_loss_se": float(row.bootstrapped_loss_se),
                "active": bool(row.active),
                "pruned": bool(row.pruned),
                "selected": bool(row.selected),
                "selected_min_loss": bool(row.selected_min_loss),
                "baseline_forced_eval": bool(row.baseline_forced_eval),
                "kept_by_one_se": bool(row.kept_by_one_se),
                "complexity_group": row.complexity_group,
                "complexity_rank": row.complexity_rank,
                "complexity_source": row.complexity_source,
                "stage_best_candidate_id": row.stage_best_candidate_id,
                "stage_best_complexity_rank": row.stage_best_complexity_rank,
                "outside_one_se": bool(row.outside_one_se),
                "strictly_simpler_than_stage_best": bool(row.strictly_simpler_than_stage_best),
                "prune_reason": row.prune_reason,
            }
        )
    for fold in folds:
        stage_row = by_candidate_stage.get((fold.candidate_id, int(fold.stage)))
        rows.append(
            {
                "row_type": "fold_stage",
                "stage": int(fold.stage),
                "iteration": int(fold.iteration),
                "candidate_id": fold.candidate_id,
                "family": fold.family,
                "fold": int(fold.fold),
                "stage_loss": float(fold.bootstrapped_loss),
                "stage_loss_se": float(fold.bootstrapped_loss_se),
                "active": bool(stage_row.active) if stage_row is not None else bool(fold.active),
                "pruned": bool(stage_row.pruned) if stage_row is not None else bool(fold.pruned),
                "selected": bool(stage_row.selected) if stage_row is not None else bool(fold.selected),
                "baseline_forced_eval": bool(stage_row.baseline_forced_eval)
                if stage_row is not None
                else bool(fold.baseline_forced_eval),
                "kept_by_one_se": bool(stage_row.kept_by_one_se) if stage_row is not None else False,
                "selected_min_loss": bool(stage_row.selected_min_loss) if stage_row is not None else False,
                "complexity_group": stage_row.complexity_group if stage_row is not None else "",
                "complexity_rank": stage_row.complexity_rank if stage_row is not None else "",
                "complexity_source": stage_row.complexity_source if stage_row is not None else "none",
                "stage_best_candidate_id": stage_row.stage_best_candidate_id if stage_row is not None else "",
                "stage_best_complexity_rank": stage_row.stage_best_complexity_rank if stage_row is not None else "",
                "outside_one_se": bool(stage_row.outside_one_se) if stage_row is not None else False,
                "strictly_simpler_than_stage_best": bool(stage_row.strictly_simpler_than_stage_best)
                if stage_row is not None
                else False,
                "prune_reason": stage_row.prune_reason if stage_row is not None else "",
            }
        )
    return rows
