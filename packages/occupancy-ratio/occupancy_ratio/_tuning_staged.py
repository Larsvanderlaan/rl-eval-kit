"""Staged bootstrapped-loss pruning for occupancy-ratio tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class StagedCVFoldRow:
    candidate_id: str
    family: str
    budget_stage: str
    fold: int
    loss: float
    metric: str
    stage: int = 0
    stage_loss: float = float("nan")
    stage_loss_se: float = float("nan")
    active: bool = True
    pruned: bool = False
    selected: bool = False
    baseline_forced_eval: bool = False
    complexity_group: str = ""
    complexity_rank: str = ""
    complexity_source: str = "none"
    stage_best_candidate_id: str = ""
    outside_one_se: bool = False
    strictly_simpler_than_stage_best: bool = False
    prune_reason: str = ""


@dataclass
class StagedCVCandidateRow:
    candidate_id: str
    candidate_label: str
    family: str
    budget_stage: str
    loss_mean: float
    loss_se: float
    bootstrap_iterations: int
    selected_min_loss: bool
    kept: bool
    pruned: bool
    threshold: float
    reason: str = ""
    stage: int = 0
    stage_loss: float = float("nan")
    stage_loss_se: float = float("nan")
    active: bool = True
    selected: bool = False
    baseline_forced_eval: bool = False
    final_stage: bool = False
    complexity_group: str = ""
    complexity_rank: str = ""
    complexity_source: str = "none"
    stage_best_candidate_id: str = ""
    stage_best_complexity_rank: str = ""
    outside_one_se: bool = False
    strictly_simpler_than_stage_best: bool = False
    prune_reason: str = ""


@dataclass
class StagedCVResult:
    """Telemetry and keep/prune decisions from staged bootstrapped CV."""

    candidate_rows: list[StagedCVCandidateRow]
    fold_rows: list[StagedCVFoldRow]
    selected_candidate_id: str
    threshold: float
    loss_metric: str

    def candidate_dicts(self) -> list[dict[str, Any]]:
        return [asdict(row) for row in self.candidate_rows]

    def fold_dicts(self) -> list[dict[str, Any]]:
        return [asdict(row) for row in self.fold_rows]

    @property
    def kept_candidate_ids(self) -> set[str]:
        final_rows = [row for row in self.candidate_rows if bool(getattr(row, "final_stage", False))]
        if final_rows:
            return {row.candidate_id for row in final_rows if row.kept}
        return {row.candidate_id for row in self.candidate_rows if row.kept}


def run_staged_bootstrap_cv(
    candidates: Sequence[Any],
    cfg: Any,
    *,
    seed: int,
) -> StagedCVResult:
    """Prune full-CV occupancy candidates using bootstrapped generated loss.

    The supplied candidates are expected to be the already evaluated full-stage
    FORI/occupancy candidates. Their fold losses therefore come from the
    candidate-specific target construction used by the existing fitters,
    including selected first-stage nuisance bundles and resolved source/one-step
    modes.
    """

    metric = str(getattr(cfg, "staged_cv_loss_metric", "validation_loss"))
    iterations = int(getattr(cfg, "staged_cv_n_bootstrap", 200))
    one_se = float(getattr(cfg, "staged_cv_one_se_multiplier", 1.0))
    iterations = max(0, iterations)
    rng = np.random.default_rng(int(seed))

    eligible = [row for row in candidates if not getattr(row, "error", "") and getattr(row, "fold_results", None)]
    fold_rows: list[StagedCVFoldRow] = []
    candidate_payloads: list[tuple[Any, np.ndarray, float, float]] = []
    for candidate in eligible:
        losses = np.asarray([_fold_loss(fold, metric) for fold in candidate.fold_results], dtype=np.float64)
        finite = losses[np.isfinite(losses)]
        for fold, loss in zip(candidate.fold_results, losses):
            fold_rows.append(
                StagedCVFoldRow(
                    candidate_id=str(candidate.candidate_id),
                    family=str(candidate.family),
                    budget_stage=str(candidate.budget_stage),
                    fold=int(getattr(fold, "fold", -1)),
                    loss=float(loss),
                    metric=metric,
                )
            )
        if finite.size == 0:
            mean = float("inf")
            se = float("inf")
        else:
            mean = float(np.mean(finite))
            se = _bootstrap_mean_se(finite, iterations=iterations, rng=rng)
        candidate.metrics["staged_cv_loss"] = float(mean)
        candidate.metrics["staged_cv_loss_se"] = float(se)
        candidate.metrics["staged_cv_iterations"] = float(iterations)
        candidate_payloads.append((candidate, finite, mean, se))

    if not candidate_payloads:
        return StagedCVResult([], fold_rows, "", float("inf"), metric)

    rows: list[StagedCVCandidateRow] = []
    for candidate, _, mean, se in candidate_payloads:
        desc = _candidate_complexity_from_result(candidate)
        rows.append(
            StagedCVCandidateRow(
                candidate_id=str(candidate.candidate_id),
                candidate_label=str(getattr(candidate, "candidate_label", "") or candidate.candidate_id),
                family=str(candidate.family),
                budget_stage=str(candidate.budget_stage),
                loss_mean=float(mean),
                loss_se=float(se),
                bootstrap_iterations=int(iterations),
                selected_min_loss=False,
                kept=True,
                pruned=False,
                threshold=float("inf"),
                reason="pending",
                stage=int(getattr(cfg, "staged_cv_iterations", 0) or 0),
                stage_loss=float(mean),
                stage_loss_se=float(se),
                complexity_group=str(desc.get("group", "")),
                complexity_rank=str(desc.get("rank_repr", "")),
                complexity_source=str(desc.get("source", "none")),
            )
        )

    complexity = {str(candidate.candidate_id): _candidate_complexity_from_result(candidate) for candidate, *_ in candidate_payloads}
    active_ids = {row.candidate_id for row in rows}
    kept_ids, selected_candidate_id, threshold = monotone_one_se_prune(
        rows,
        active_ids,
        complexity,
        one_se,
        max(1, int(getattr(cfg, "staged_cv_min_survivors", 1))),
    )
    candidate_by_id = {str(candidate.candidate_id): candidate for candidate, *_ in candidate_payloads}
    for row in rows:
        row.threshold = float(threshold)
        row.kept = row.candidate_id in kept_ids
        row.pruned = row.candidate_id not in kept_ids
        row.selected = row.candidate_id == selected_candidate_id
        row.reason = row.prune_reason or row.reason
        candidate = candidate_by_id.get(row.candidate_id)
        if candidate is not None:
            candidate.metrics["staged_cv_selected_min_loss"] = float(row.selected_min_loss)
            candidate.metrics["staged_cv_kept"] = float(row.kept)
            candidate.metrics["staged_cv_pruned"] = float(row.pruned)
            candidate.metrics["staged_cv_threshold"] = float(threshold)
            candidate.metrics["staged_cv_complexity_rank"] = row.complexity_rank
            candidate.metrics["staged_cv_prune_reason"] = row.prune_reason
    rows_by_id = {row.candidate_id: row for row in rows}
    for fold_row in fold_rows:
        row = rows_by_id.get(fold_row.candidate_id)
        if row is None:
            continue
        fold_row.active = bool(row.active)
        fold_row.pruned = bool(row.pruned)
        fold_row.selected = bool(row.selected)
        fold_row.complexity_group = row.complexity_group
        fold_row.complexity_rank = row.complexity_rank
        fold_row.complexity_source = row.complexity_source
        fold_row.stage_best_candidate_id = row.stage_best_candidate_id
        fold_row.outside_one_se = bool(row.outside_one_se)
        fold_row.strictly_simpler_than_stage_best = bool(row.strictly_simpler_than_stage_best)
        fold_row.prune_reason = row.prune_reason
    return StagedCVResult(rows, fold_rows, str(selected_candidate_id), float(threshold), metric)


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
        _set_if_present(row, "kept", bool(is_kept))
        _set_if_present(row, "pruned", bool(was_active and not is_kept))
        if decisions.get(candidate_id) == "min_survivor_readd":
            _set_if_present(row, "prune_reason", "min_survivor_readd")
    return kept, best_id, threshold


def _fold_loss(fold: Any, metric: str) -> float:
    value = getattr(fold, metric, None)
    if value is None or not np.isfinite(float(value)):
        for fallback in ("validation_loss", "selection_risk", "moment_balance"):
            value = getattr(fold, fallback, None)
            if value is not None and np.isfinite(float(value)):
                return float(value)
        return float("inf")
    return float(value)


def _candidate_complexity_from_result(candidate: Any) -> dict[str, Any]:
    explicit = _explicit_complexity(str(getattr(candidate, "family", "")), dict(getattr(candidate, "overrides", {}) or {}))
    if explicit is not None:
        return explicit
    return _descriptor(group="", rank=None, source="none")


def _explicit_complexity(family: str, overrides: dict[str, Any]) -> dict[str, Any] | None:
    meta = overrides.get("_meta")
    if not isinstance(meta, dict) or "complexity_rank" not in meta:
        return None
    return _descriptor(
        group=str(meta.get("complexity_group", f"{family}:explicit")),
        rank=meta.get("complexity_rank"),
        source="explicit",
    )


def _descriptor(*, group: str, rank: Any, source: str) -> dict[str, Any]:
    rank_tuple = _rank_tuple(rank)
    return {
        "group": str(group),
        "rank": rank_tuple,
        "rank_repr": "" if rank_tuple is None else "x".join(f"{value:g}" for value in rank_tuple),
        "source": str(source),
    }


def _rank_tuple(value: Any) -> tuple[float, ...] | None:
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


def _bootstrap_mean_se(values: np.ndarray, *, iterations: int, rng: np.random.Generator) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    if int(iterations) <= 0:
        return float(np.std(x, ddof=1) / np.sqrt(x.size))
    draws = rng.integers(0, x.size, size=(int(iterations), x.size))
    means = np.mean(x[draws], axis=1)
    return float(np.std(means, ddof=1)) if means.size > 1 else 0.0
