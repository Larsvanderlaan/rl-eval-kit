"""Stage-wise nuisance tuning helpers for occupancy-ratio AutoML."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from occupancy_ratio.fit_importance_and_transition_ratios import (
    fit_importance_ratio_lgbm,
    fit_state_density_ratio_lgbm,
    fit_transition_ratio_lgbm,
)
from occupancy_ratio._boosted_impl import (
    _make_transition_reference_features,
    _one_step_direct_ratio_diagnostics as _boosted_direct_diagnostics,
    _predict_processed_nuisance,
    _predict_processed_source_state_ratio,
    _source_state_ratio_diagnostics as _boosted_source_diagnostics,
)


Array = np.ndarray


def fit_first_stage_for_family(
    *,
    family: str,
    space: Any,
    cfg: Any,
    candidates: Sequence[Dict[str, Any]],
    score_folds: Sequence[Array],
    reuse_folds: Sequence[Array],
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    A_pi_next: Optional[Array],
    initial_ratio_mode: str,
    one_step_ratio_mode: str,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Tune first-stage density ratios and prepare reusable fold bundles."""

    family = str(family)
    family_candidates = [
        row
        for row in candidates
        if str(row.get("family", "")) == family
        and str(row.get("overrides", {}).get("backend", {}).get("name", "")) != "google_dualdice"
    ]
    if not family_candidates:
        return None

    start = time.perf_counter()
    telemetry: Dict[str, Any] = dict(candidate_rows=[], fold_rows=[], skipped=[], selected={})
    base = _base_configs(space, family)
    overrides = [dict(row.get("overrides", {})) for row in family_candidates]

    action_configs = _ratio_configs(
        base["action_ratio"],
        _stage_overrides(space, family, "action_ratio", overrides),
    )
    action = _select_action_ratio(
        family=family,
        configs=action_configs,
        folds=score_folds,
        S=S,
        A=A,
        A_pi=A_pi,
        seed=int(seed) + 11_001,
        telemetry=telemetry,
    )

    transition_configs = _ratio_configs(
        base["transition_ratio"],
        _stage_overrides(space, family, "transition_ratio", overrides),
    )
    transition = _select_transition_ratio(
        family=family,
        configs=transition_configs,
        folds=score_folds,
        S=S,
        A=A,
        S_next=S_next,
        seed=int(seed) + 13_001,
        telemetry=telemetry,
    )

    initial_modes = _expand_initial_modes(
        requested=initial_ratio_mode,
        configured=getattr(cfg, "initial_ratio_mode_candidates", ("auto", "factored")),
        has_initial_states=S_initial is not None,
        has_initial_actions=A_initial is not None,
        telemetry=telemetry,
        family=family,
    )
    source = _select_source_ratio(
        family=family,
        configs=_ratio_configs(
            base["source_state_ratio"],
            _stage_overrides(space, family, "source_state_ratio", overrides),
        ),
        modes=initial_modes,
        folds=score_folds,
        S=S,
        A=A,
        A_pi=A_pi,
        S_initial=S_initial,
        A_initial=A_initial,
        initial_weights=initial_weights,
        action_folds=action["fold_fits"],
        seed=int(seed) + 17_001,
        telemetry=telemetry,
    )

    one_step_modes = _expand_one_step_modes(
        requested=one_step_ratio_mode,
        configured=getattr(cfg, "one_step_ratio_mode_candidates", ("auto", "factored")),
        has_target_next_actions=A_pi_next is not None,
        telemetry=telemetry,
        family=family,
    )
    direct = _select_one_step_ratio(
        family=family,
        configs=_ratio_configs(
            base["source_state_ratio"],
            _stage_overrides(space, family, "direct_one_step_ratio", overrides)
            or _stage_overrides(space, family, "source_state_ratio", overrides),
        ),
        modes=one_step_modes,
        folds=score_folds,
        S=S,
        A=A,
        S_next=S_next,
        A_pi=A_pi,
        A_pi_next=A_pi_next,
        factored_score=float(action["score"]) + float(transition["score"]),
        seed=int(seed) + 19_001,
        telemetry=telemetry,
    )

    if not _same_folds(score_folds, reuse_folds):
        action["fold_fits"] = _fit_action_folds(
            family=family,
            config=action["config"],
            folds=reuse_folds,
            S=S,
            A=A,
            A_pi=A_pi,
            seed=int(seed) + 31_001,
        )
        transition["fold_fits"] = _fit_transition_folds(
            family=family,
            config=transition["config"],
            folds=reuse_folds,
            S=S,
            A=A,
            S_next=S_next,
            seed=int(seed) + 37_001,
        )
        source["fold_fits"] = _fit_source_folds(
            family=family,
            config=source.get("config"),
            mode=str(source["mode"]),
            folds=reuse_folds,
            S=S,
            A=A,
            A_pi=A_pi,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weights,
            action_folds=action["fold_fits"],
            seed=int(seed) + 41_001,
        )
        direct["fold_fits"] = _fit_direct_folds(
            family=family,
            config=direct.get("config"),
            mode=str(direct["mode"]),
            folds=reuse_folds,
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            A_pi_next=A_pi_next,
            seed=int(seed) + 43_001,
        )

    fold_bundles = [
        _combine_fold_bundle(
            action_fit=action["fold_fits"][fold_id],
            transition_fit=transition["fold_fits"][fold_id],
            source_piece=source["fold_fits"][fold_id],
            direct_piece=direct["fold_fits"][fold_id],
            initial_mode=str(source["mode"]),
            one_step_mode=str(direct["mode"]),
        )
        for fold_id in range(len(reuse_folds))
    ]
    final_bundle = _fit_final_bundle(
        family=family,
        action_config=action["config"],
        transition_config=transition["config"],
        source_config=source.get("config"),
        direct_config=direct.get("config"),
        initial_mode=str(source["mode"]),
        one_step_mode=str(direct["mode"]),
        S=S,
        A=A,
        S_next=S_next,
        A_pi=A_pi,
        S_initial=S_initial,
        A_initial=A_initial,
        initial_weights=initial_weights,
        A_pi_next=A_pi_next,
        seed=int(seed) + 53_001,
    )

    selected_configs = {
        "action_ratio": action["config"],
        "transition_ratio": transition["config"],
        "source_state_ratio": source.get("config") or base["source_state_ratio"],
    }
    telemetry["selected"] = {
        "action_ratio": _config_dict(action["config"]),
        "transition_ratio": _config_dict(transition["config"]),
        "source_state_ratio": _config_dict(selected_configs["source_state_ratio"]),
        "initial_ratio_mode": str(source["mode"]),
        "one_step_ratio_mode": str(direct["mode"]),
        "direct_one_step_ratio": None if direct.get("config") is None else _config_dict(direct["config"]),
    }
    telemetry["refit_diagnostics"] = _refit_diagnostics(final_bundle)
    telemetry["runtime_sec"] = float(time.perf_counter() - start)

    return dict(
        family=family,
        selected_configs=selected_configs,
        selected_initial_ratio_mode=str(source["mode"]),
        selected_one_step_ratio_mode=str(direct["mode"]),
        fold_bundles=fold_bundles,
        final_bundle=final_bundle,
        telemetry=telemetry,
    )


def public_first_stage_telemetry(first_stage_by_family: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        family: stage["telemetry"]
        for family, stage in first_stage_by_family.items()
        if isinstance(stage, dict) and "telemetry" in stage
    }


def _base_configs(space: Any, family: str) -> Dict[str, Any]:
    prefix = "boosted" if family == "boosted" else "neural"
    return {
        "action_ratio": getattr(space, f"{prefix}_action_ratio"),
        "source_state_ratio": getattr(space, f"{prefix}_source_state_ratio"),
        "transition_ratio": getattr(space, f"{prefix}_transition_ratio"),
    }


def _stage_overrides(
    space: Any,
    family: str,
    task: str,
    product_overrides: Sequence[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    grids = getattr(space, f"{family}_first_stage_grids", None)
    if isinstance(grids, dict) and task in grids:
        return [dict(row) for row in grids[task]]
    rows = [dict(overrides.get(task, {})) for overrides in product_overrides if task in overrides or task != "direct_one_step_ratio"]
    if task == "direct_one_step_ratio" and not rows:
        rows = [dict(overrides.get("source_state_ratio", {})) for overrides in product_overrides]
    return rows or [{}]


def _ratio_configs(base: Any, overrides: Sequence[Dict[str, Any]]) -> List[Any]:
    configs: List[Any] = []
    seen = set()
    for override in overrides or ({},):
        cfg = replace(base, **dict(override))
        key = repr(_config_dict(cfg))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs or [base]


def _select_action_ratio(
    *,
    family: str,
    configs: Sequence[Any],
    folds: Sequence[Array],
    S: Array,
    A: Array,
    A_pi: Array,
    seed: int,
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for idx, config in enumerate(configs):
        row_id = f"action_ratio_{idx:03d}"
        start = time.perf_counter()
        fold_fits = []
        fold_scores = []
        fold_updates = []
        error = ""
        try:
            for fold_id, valid_idx in enumerate(folds):
                train_idx = _complement_indices(S.shape[0], valid_idx)
                fit = _fit_action(
                    family=family,
                    config=config,
                    S=S[train_idx],
                    A=A[train_idx],
                    A_pi=A_pi[train_idx],
                    seed=int(seed) + 1_003 * (fold_id + 1),
                )
                score = _score_action(family, fit, S[valid_idx], A[valid_idx], A_pi[valid_idx])
                fold_fits.append(fit)
                fold_scores.append(float(score))
                updates = _fit_update_count(fit)
                fold_updates.append(updates)
                _append_fold_row(telemetry, family, "action_ratio", row_id, "", fold_id, score, update_count=updates)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        score = _finite_mean(fold_scores)
        candidate = dict(config=config, score=score, fold_fits=fold_fits, candidate_id=row_id)
        _append_candidate_row(
            telemetry,
            family,
            "action_ratio",
            row_id,
            "",
            score,
            start,
            config,
            error,
            update_count_mean=_finite_mean(fold_updates),
        )
        if not error and fold_fits and (best is None or score < float(best["score"])):
            best = candidate
    if best is None:
        raise RuntimeError("No first-stage action-ratio candidate completed successfully.")
    _mark_selected(telemetry, "action_ratio", str(best["candidate_id"]))
    return best


def _select_transition_ratio(
    *,
    family: str,
    configs: Sequence[Any],
    folds: Sequence[Array],
    S: Array,
    A: Array,
    S_next: Array,
    seed: int,
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for idx, config in enumerate(configs):
        row_id = f"transition_ratio_{idx:03d}"
        start = time.perf_counter()
        fold_fits = []
        fold_scores = []
        fold_updates = []
        error = ""
        try:
            for fold_id, valid_idx in enumerate(folds):
                train_idx = _complement_indices(S.shape[0], valid_idx)
                fit = _fit_transition(
                    family=family,
                    config=config,
                    S=S[train_idx],
                    A=A[train_idx],
                    S_next=S_next[train_idx],
                    seed=int(seed) + 1_009 * (fold_id + 1),
                )
                score = _score_transition(family, fit, S[valid_idx], A[valid_idx], S_next[valid_idx], config, seed + fold_id)
                fold_fits.append(fit)
                fold_scores.append(float(score))
                updates = _fit_update_count(fit)
                fold_updates.append(updates)
                _append_fold_row(telemetry, family, "transition_ratio", row_id, "", fold_id, score, update_count=updates)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        score = _finite_mean(fold_scores)
        candidate = dict(config=config, score=score, fold_fits=fold_fits, candidate_id=row_id)
        _append_candidate_row(
            telemetry,
            family,
            "transition_ratio",
            row_id,
            "",
            score,
            start,
            config,
            error,
            update_count_mean=_finite_mean(fold_updates),
        )
        if not error and fold_fits and (best is None or score < float(best["score"])):
            best = candidate
    if best is None:
        raise RuntimeError("No first-stage transition-ratio candidate completed successfully.")
    _mark_selected(telemetry, "transition_ratio", str(best["candidate_id"]))
    return best


def _select_source_ratio(
    *,
    family: str,
    configs: Sequence[Any],
    modes: Sequence[str],
    folds: Sequence[Array],
    S: Array,
    A: Array,
    A_pi: Array,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    action_folds: Sequence[Dict[str, Any]],
    seed: int,
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    if S_initial is None:
        start = time.perf_counter()
        pieces = []
        fold_updates = []
        for fold_id, valid_idx in enumerate(folds):
            train_idx = _complement_indices(S.shape[0], valid_idx)
            piece = _source_piece_without_initial(
                family=family,
                action_fit=action_folds[fold_id],
                S=S[train_idx],
                A=A[train_idx],
                A_pi=A_pi[train_idx],
            )
            pieces.append(piece)
            fold_updates.append(_source_piece_update_count(piece))
        row_id = "source_state_ratio_none"
        _append_candidate_row(
            telemetry,
            family,
            "source_state_ratio",
            row_id,
            "factored",
            0.0,
            start,
            None,
            "",
            update_count_mean=_finite_mean(fold_updates),
        )
        _mark_selected(telemetry, "source_state_ratio", row_id)
        return dict(config=None, mode="factored", score=0.0, fold_fits=pieces, candidate_id=row_id)

    best: Optional[Dict[str, Any]] = None
    for mode in modes:
        for idx, config in enumerate(configs):
            row_id = f"source_state_ratio_{mode}_{idx:03d}"
            start = time.perf_counter()
            pieces = []
            fold_scores = []
            fold_updates = []
            error = ""
            try:
                for fold_id, valid_idx in enumerate(folds):
                    train_idx = _complement_indices(S.shape[0], valid_idx)
                    piece = _fit_source_piece(
                        family=family,
                        mode=mode,
                        config=config,
                        S=S,
                        A=A,
                        A_pi=A_pi,
                        S_initial=S_initial,
                        A_initial=A_initial,
                        initial_weights=initial_weights,
                        train_idx=train_idx,
                        valid_idx=np.asarray(valid_idx, dtype=np.int64),
                        fold_id=fold_id,
                        n_folds=len(folds),
                        action_fit=action_folds[fold_id],
                        seed=int(seed) + 1_013 * (fold_id + 1),
                        numerator_seed=int(seed) + 59_001,
                    )
                    pieces.append(piece)
                    fold_scores.append(float(piece["score"]))
                    updates = _source_piece_update_count(piece)
                    fold_updates.append(updates)
                    _append_fold_row(
                        telemetry,
                        family,
                        "source_state_ratio",
                        row_id,
                        mode,
                        fold_id,
                        piece["score"],
                        update_count=updates,
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            score = _finite_mean(fold_scores)
            candidate = dict(config=config, mode=mode, score=score, fold_fits=pieces, candidate_id=row_id)
            _append_candidate_row(
                telemetry,
                family,
                "source_state_ratio",
                row_id,
                mode,
                score,
                start,
                config,
                error,
                update_count_mean=_finite_mean(fold_updates),
            )
            if not error and pieces and (best is None or score < float(best["score"])):
                best = candidate
    if best is None:
        raise RuntimeError("No first-stage source-ratio candidate completed successfully.")
    _mark_selected(telemetry, "source_state_ratio", str(best["candidate_id"]))
    return best


def _select_one_step_ratio(
    *,
    family: str,
    configs: Sequence[Any],
    modes: Sequence[str],
    folds: Sequence[Array],
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    A_pi_next: Optional[Array],
    factored_score: float,
    seed: int,
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    if "factored" in modes:
        pieces = [dict(c_fit=None, c_ratio_query=None, c_diagnostics=_direct_diagnostics(family, None, None)) for _ in folds]
        best = dict(config=None, mode="factored", score=float(factored_score), fold_fits=pieces, candidate_id="one_step_ratio_factored")
        _append_candidate_row(
            telemetry,
            family,
            "one_step_ratio",
            "one_step_ratio_factored",
            "factored",
            factored_score,
            time.perf_counter(),
            None,
            "",
            update_count_mean=0.0,
        )

    if "direct" in modes and A_pi_next is not None:
        for idx, config in enumerate(configs):
            row_id = f"one_step_ratio_direct_{idx:03d}"
            start = time.perf_counter()
            pieces = []
            fold_scores = []
            fold_updates = []
            error = ""
            try:
                for fold_id, valid_idx in enumerate(folds):
                    train_idx = _complement_indices(S.shape[0], valid_idx)
                    piece = _fit_direct_piece(
                        family=family,
                        config=config,
                        S=S,
                        A=A,
                        S_next=S_next,
                        A_pi=A_pi,
                        A_pi_next=A_pi_next,
                        train_idx=train_idx,
                        valid_idx=np.asarray(valid_idx, dtype=np.int64),
                        seed=int(seed) + 1_019 * (fold_id + 1),
                    )
                    pieces.append(piece)
                    fold_scores.append(float(piece["score"]))
                    updates = _direct_piece_update_count(piece)
                    fold_updates.append(updates)
                    _append_fold_row(
                        telemetry,
                        family,
                        "one_step_ratio",
                        row_id,
                        "direct",
                        fold_id,
                        piece["score"],
                        update_count=updates,
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            score = _finite_mean(fold_scores)
            candidate = dict(config=config, mode="direct", score=score, fold_fits=pieces, candidate_id=row_id)
            _append_candidate_row(
                telemetry,
                family,
                "one_step_ratio",
                row_id,
                "direct",
                score,
                start,
                config,
                error,
                update_count_mean=_finite_mean(fold_updates),
            )
            if not error and pieces and (best is None or score < float(best["score"])):
                best = candidate
    if best is None:
        raise RuntimeError("No first-stage one-step-ratio mode completed successfully.")
    _mark_selected(telemetry, "one_step_ratio", str(best["candidate_id"]))
    return best


def _fit_final_bundle(
    *,
    family: str,
    action_config: Any,
    transition_config: Any,
    source_config: Any,
    direct_config: Any,
    initial_mode: str,
    one_step_mode: str,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    A_pi_next: Optional[Array],
    seed: int,
) -> Dict[str, Any]:
    all_idx = np.arange(S.shape[0], dtype=np.int64)
    action_fit = _fit_action(family=family, config=action_config, S=S, A=A, A_pi=A_pi, seed=int(seed) + 101)
    transition_fit = _fit_transition(family=family, config=transition_config, S=S, A=A, S_next=S_next, seed=int(seed) + 103)
    source_piece = (
        _source_piece_without_initial(family=family, action_fit=action_fit, S=S, A=A, A_pi=A_pi)
        if S_initial is None
        else _fit_source_piece(
            family=family,
            mode=initial_mode,
            config=source_config,
            S=S,
            A=A,
            A_pi=A_pi,
            S_initial=S_initial,
            A_initial=A_initial,
            initial_weights=initial_weights,
            train_idx=all_idx,
            valid_idx=all_idx,
            fold_id=0,
            n_folds=1,
            action_fit=action_fit,
            seed=int(seed) + 107,
            final_fit=True,
        )
    )
    direct_piece = (
        _fit_direct_piece(
            family=family,
            config=direct_config,
            S=S,
            A=A,
            S_next=S_next,
            A_pi=A_pi,
            A_pi_next=A_pi_next,
            train_idx=all_idx,
            valid_idx=all_idx,
            seed=int(seed) + 109,
            final_fit=True,
        )
        if one_step_mode == "direct"
        else dict(c_fit=None, c_ratio_query=None, c_diagnostics=_direct_diagnostics(family, None, None))
    )
    return _combine_fold_bundle(
        action_fit=action_fit,
        transition_fit=transition_fit,
        source_piece=source_piece,
        direct_piece=direct_piece,
        initial_mode=initial_mode,
        one_step_mode=one_step_mode,
    )


def _fit_action_folds(*, family: str, config: Any, folds: Sequence[Array], S: Array, A: Array, A_pi: Array, seed: int) -> List[Dict[str, Any]]:
    fits = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        fits.append(_fit_action(family=family, config=config, S=S[train_idx], A=A[train_idx], A_pi=A_pi[train_idx], seed=int(seed) + fold_id))
    return fits


def _fit_transition_folds(*, family: str, config: Any, folds: Sequence[Array], S: Array, A: Array, S_next: Array, seed: int) -> List[Dict[str, Any]]:
    fits = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        fits.append(_fit_transition(family=family, config=config, S=S[train_idx], A=A[train_idx], S_next=S_next[train_idx], seed=int(seed) + fold_id))
    return fits


def _fit_source_folds(
    *,
    family: str,
    config: Any,
    mode: str,
    folds: Sequence[Array],
    S: Array,
    A: Array,
    A_pi: Array,
    S_initial: Optional[Array],
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    action_folds: Sequence[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    if S_initial is None:
        pieces = []
        for fold_id, valid_idx in enumerate(folds):
            train_idx = _complement_indices(S.shape[0], valid_idx)
            pieces.append(
                _source_piece_without_initial(
                    family=family,
                    action_fit=action_folds[fold_id],
                    S=S[train_idx],
                    A=A[train_idx],
                    A_pi=A_pi[train_idx],
                )
            )
        return pieces
    pieces = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        pieces.append(
            _fit_source_piece(
                family=family,
                mode=mode,
                config=config,
                S=S,
                A=A,
                A_pi=A_pi,
                S_initial=S_initial,
                A_initial=A_initial,
                initial_weights=initial_weights,
                train_idx=train_idx,
                valid_idx=np.asarray(valid_idx, dtype=np.int64),
                fold_id=fold_id,
                n_folds=len(folds),
                action_fit=action_folds[fold_id],
                seed=int(seed) + fold_id,
                numerator_seed=int(seed) + 59_001,
            )
        )
    return pieces


def _fit_direct_folds(
    *,
    family: str,
    config: Any,
    mode: str,
    folds: Sequence[Array],
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    A_pi_next: Optional[Array],
    seed: int,
) -> List[Dict[str, Any]]:
    if mode != "direct":
        return [dict(c_fit=None, c_ratio_query=None, c_diagnostics=_direct_diagnostics(family, None, None)) for _ in folds]
    pieces = []
    for fold_id, valid_idx in enumerate(folds):
        train_idx = _complement_indices(S.shape[0], valid_idx)
        pieces.append(
            _fit_direct_piece(
                family=family,
                config=config,
                S=S,
                A=A,
                S_next=S_next,
                A_pi=A_pi,
                A_pi_next=A_pi_next,
                train_idx=train_idx,
                valid_idx=np.asarray(valid_idx, dtype=np.int64),
                seed=int(seed) + fold_id,
            )
        )
    return pieces


def _fit_action(*, family: str, config: Any, S: Array, A: Array, A_pi: Array, seed: int) -> Dict[str, Any]:
    if family == "boosted":
        kwargs = config.to_kwargs()
        kwargs["crossfit_folds"] = 1
        kwargs["show_tqdm"] = False
        return fit_importance_ratio_lgbm(S=S, A=A, A_pi=A_pi, seed=int(seed), **kwargs)
    from occupancy_ratio.fit_occupancy_ratio_neural import fit_action_ratio_neural

    X_beh = np.concatenate([S, A], axis=1).astype(np.float32, copy=False)
    X_pi = np.concatenate([S, A_pi], axis=1).astype(np.float32, copy=False)
    return fit_action_ratio_neural(X_beh, X_pi, replace(config, crossfit_folds=1, seed=int(seed)))


def _fit_transition(*, family: str, config: Any, S: Array, A: Array, S_next: Array, seed: int) -> Dict[str, Any]:
    if family == "boosted":
        kwargs = config.to_kwargs()
        kwargs["crossfit_folds"] = 1
        kwargs["show_tqdm"] = False
        return fit_transition_ratio_lgbm(S=S, A=A, S_next=S_next, seed=int(seed), **kwargs)
    from occupancy_ratio.fit_occupancy_ratio_neural import fit_transition_ratio_neural

    X_sa = np.concatenate([S, A], axis=1).astype(np.float32, copy=False)
    return fit_transition_ratio_neural(X_sa, S_next.astype(np.float32, copy=False), S.astype(np.float32, copy=False), replace(config, crossfit_folds=1, seed=int(seed)))


def _fit_source_piece(
    *,
    family: str,
    mode: str,
    config: Any,
    S: Array,
    A: Array,
    A_pi: Array,
    S_initial: Array,
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    train_idx: Array,
    valid_idx: Array,
    fold_id: int,
    n_folds: int,
    action_fit: Dict[str, Any],
    seed: int,
    numerator_seed: Optional[int] = None,
    final_fit: bool = False,
) -> Dict[str, Any]:
    mode = str(mode)
    init_train, init_valid, weights_train, weights_valid = _initial_train_valid(
        S_initial=S_initial,
        A_initial=A_initial,
        initial_weights=initial_weights,
        train_idx=train_idx,
        valid_idx=valid_idx,
        full_n=S.shape[0],
        fold_id=fold_id,
        n_folds=n_folds,
        seed=int(seed if numerator_seed is None else numerator_seed) + 29_991,
    )
    if mode == "joint":
        if A_initial is None:
            raise ValueError("joint initial ratio requires initial_actions.")
        if init_train[1] is None or init_valid[1] is None:
            raise ValueError("joint initial ratio requires folded initial actions.")
        X_ref = np.concatenate([S[train_idx], A[train_idx]], axis=1)
        X_num = np.concatenate([init_train[0], init_train[1]], axis=1)
        X_ref_valid = np.concatenate([S[valid_idx], A[valid_idx]], axis=1)
        X_num_valid = np.concatenate([init_valid[0], init_valid[1]], axis=1)
        X_query = np.vstack([
            np.concatenate([S[train_idx], A_pi[train_idx]], axis=1),
            np.concatenate([S[train_idx], A[train_idx]], axis=1),
        ])
    else:
        X_ref = S[train_idx]
        X_num = init_train[0]
        X_ref_valid = S[valid_idx]
        X_num_valid = init_valid[0]
        X_query = np.vstack([S[train_idx], S[train_idx]])
    fit = _fit_generic_density(family=family, config=config, X_ref=X_ref, X_num=X_num, numerator_weights=weights_train, seed=int(seed))
    pred_ref = _predict_generic_density(family, fit, X_ref_valid)
    pred_num = _predict_generic_density(family, fit, X_num_valid)
    score = _density_ratio_score(fit, pred_ref, pred_num, numerator_weights=weights_valid)

    if mode == "joint":
        source_weight_query = _predict_generic_density(family, fit, X_query)
        source_state_query = None
    else:
        source_state_query = _predict_generic_density(family, fit, X_query)
        action_query = _predict_action_query(family, action_fit, np.vstack([
            np.concatenate([S[train_idx], A_pi[train_idx]], axis=1),
            np.concatenate([S[train_idx], A[train_idx]], axis=1),
        ]))
        source_weight_query = np.maximum(action_query * source_state_query, 0.0)
    diagnostics = _initial_diagnostics(family, mode, source_state_query, source_weight_query, fit)
    return dict(
        source_fit=fit,
        source_weight_query=source_weight_query,
        source_state_query=source_state_query,
        source_diagnostics=diagnostics,
        score=0.0 if final_fit else float(score),
    )


def _source_piece_without_initial(*, family: str, action_fit: Dict[str, Any], S: Array, A: Array, A_pi: Array) -> Dict[str, Any]:
    X_query = np.vstack([np.concatenate([S, A_pi], axis=1), np.concatenate([S, A], axis=1)])
    source_weight_query = _predict_action_query(family, action_fit, X_query)
    diagnostics = _initial_diagnostics(family, "factored", None, source_weight_query, None)
    return dict(
        source_fit=None,
        source_weight_query=source_weight_query,
        source_state_query=None,
        source_diagnostics=diagnostics,
        score=0.0,
    )


def _fit_direct_piece(
    *,
    family: str,
    config: Any,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    A_pi_next: Optional[Array],
    train_idx: Array,
    valid_idx: Array,
    seed: int,
    final_fit: bool = False,
) -> Dict[str, Any]:
    if A_pi_next is None:
        raise ValueError("direct one-step ratio requires target_next_actions.")
    X_ref = np.concatenate([S[train_idx], A[train_idx]], axis=1)
    X_num = np.concatenate([S_next[train_idx], A_pi_next[train_idx]], axis=1)
    X_ref_valid = np.concatenate([S[valid_idx], A[valid_idx]], axis=1)
    X_num_valid = np.concatenate([S_next[valid_idx], A_pi_next[valid_idx]], axis=1)
    X_query = np.vstack([
        np.concatenate([S[train_idx], A_pi[train_idx]], axis=1),
        np.concatenate([S[train_idx], A[train_idx]], axis=1),
    ])
    fit = _fit_generic_density(family=family, config=config, X_ref=X_ref, X_num=X_num, numerator_weights=None, seed=int(seed))
    pred_ref = _predict_generic_density(family, fit, X_ref_valid)
    pred_num = _predict_generic_density(family, fit, X_num_valid)
    score = _density_ratio_score(fit, pred_ref, pred_num)
    c_ratio_query = _predict_generic_density(family, fit, X_query)
    return dict(
        c_fit=fit,
        c_ratio_query=c_ratio_query,
        c_diagnostics=_direct_diagnostics(family, c_ratio_query, fit),
        score=0.0 if final_fit else float(score),
    )


def _fit_generic_density(
    *,
    family: str,
    config: Any,
    X_ref: Array,
    X_num: Array,
    numerator_weights: Optional[Array],
    seed: int,
) -> Dict[str, Any]:
    if family == "boosted":
        kwargs = config.to_kwargs()
        kwargs["crossfit_folds"] = 1
        kwargs["show_tqdm"] = False
        return fit_state_density_ratio_lgbm(
            S_ref=X_ref,
            S_num=X_num,
            numerator_weights=numerator_weights,
            seed=int(seed),
            **kwargs,
        )
    from occupancy_ratio.fit_occupancy_ratio_neural import fit_source_state_ratio_neural

    return fit_source_state_ratio_neural(
        np.asarray(X_ref, dtype=np.float32),
        np.asarray(X_num, dtype=np.float32),
        replace(config, crossfit_folds=1, seed=int(seed)),
        numerator_weights=numerator_weights,
    )


def _score_action(family: str, fit: Dict[str, Any], S: Array, A: Array, A_pi: Array) -> float:
    X_beh = np.concatenate([S, A], axis=1)
    X_pi = np.concatenate([S, A_pi], axis=1)
    pred_beh = _predict_action_query(family, fit, X_beh)
    pred_pi = _predict_action_query(family, fit, X_pi)
    return _density_ratio_score(fit, pred_beh, pred_pi)


def _score_transition(family: str, fit: Dict[str, Any], S: Array, A: Array, S_next: Array, config: Any, seed: int) -> float:
    X_sa = np.concatenate([S, A], axis=1)
    X_beh = np.concatenate([X_sa, S_next], axis=1)
    pred_beh = _predict_transition_query(family, fit, X_beh)
    if family == "boosted":
        X_ref = _make_transition_reference_features(
            X_sa=X_sa,
            S_ref=S,
            K=max(1, int(getattr(config, "permutation_samples", 1))),
            seed=int(seed) + 23_001,
        )
    else:
        rng = np.random.default_rng(int(seed) + 23_001)
        reps = max(1, int(getattr(config, "permutation_samples", 1)))
        ref_idx = np.concatenate([rng.permutation(S.shape[0]) for _ in range(reps)])
        X_ref = np.concatenate([np.tile(X_sa, (reps, 1)), S[ref_idx]], axis=1)
    pred_ref = _predict_transition_query(family, fit, X_ref)
    score = _density_ratio_score(fit, pred_ref, pred_beh)
    return float(score + (np.mean(pred_ref) - 1.0) ** 2)


def _predict_action_query(family: str, fit: Dict[str, Any], X: Array) -> Array:
    if family == "boosted":
        return _predict_processed_nuisance(fit=fit, X=X, kind="iw")
    return fit["predictor"].predict(np.asarray(X, dtype=np.float32), postprocess=True).astype(np.float64, copy=False)


def _predict_transition_query(family: str, fit: Dict[str, Any], X: Array) -> Array:
    if family == "boosted":
        return _predict_processed_nuisance(fit=fit, X=X, kind="k")
    return fit["predictor"].predict(np.asarray(X, dtype=np.float32), postprocess=True).astype(np.float64, copy=False)


def _predict_generic_density(family: str, fit: Dict[str, Any], X: Array) -> Array:
    if family == "boosted":
        return _predict_processed_source_state_ratio(fit=fit, X=X)
    return fit["predictor"].predict(np.asarray(X, dtype=np.float32), postprocess=True).astype(np.float64, copy=False)


def _density_ratio_score(
    fit: Dict[str, Any],
    pred_den: Array,
    pred_num: Array,
    numerator_weights: Optional[Array] = None,
) -> float:
    loss = str(fit.get("density_ratio_loss", "lsif"))
    den = np.asarray(pred_den, dtype=np.float64).reshape(-1)
    num = np.asarray(pred_num, dtype=np.float64).reshape(-1)
    if loss == "logistic":
        return _binary_logloss_from_ratio(den, num, numerator_weights=numerator_weights)
    return float(np.mean(den**2) - 2.0 * _weighted_mean(num, numerator_weights))


def _binary_logloss_from_ratio(den: Array, num: Array, *, numerator_weights: Optional[Array]) -> float:
    eps = 1e-12
    r_den = np.maximum(np.asarray(den, dtype=np.float64).reshape(-1), eps)
    r_num = np.maximum(np.asarray(num, dtype=np.float64).reshape(-1), eps)
    den_loss = np.log1p(r_den)
    num_loss = np.log1p(1.0 / r_num)
    return float(np.mean(den_loss) + _weighted_mean(num_loss, numerator_weights))


def _weighted_mean(values: Array, weights: Optional[Array]) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        return float(np.mean(x)) if x.size else float("nan")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != x.shape[0]:
        return float(np.mean(x)) if x.size else float("nan")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(w, 0.0, out=w)
    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return float(np.mean(x)) if x.size else float("nan")
    return float(np.sum(w * x) / total)


def _initial_train_valid(
    *,
    S_initial: Array,
    A_initial: Optional[Array],
    initial_weights: Optional[Array],
    train_idx: Array,
    valid_idx: Array,
    full_n: int,
    fold_id: int,
    n_folds: int,
    seed: int,
) -> Tuple[Tuple[Array, Optional[Array]], Tuple[Array, Optional[Array]], Optional[Array], Optional[Array]]:
    S_init = np.asarray(S_initial)
    A_init = None if A_initial is None else np.asarray(A_initial)
    weights = None if initial_weights is None else np.asarray(initial_weights, dtype=np.float64).reshape(-1)
    if S_init.shape[0] == int(full_n):
        w_train = None if weights is None else weights[train_idx]
        w_valid = None if weights is None else weights[valid_idx]
        return (S_init[train_idx], None if A_init is None else A_init[train_idx]), (S_init[valid_idx], None if A_init is None else A_init[valid_idx]), w_train, w_valid
    folds = _make_simple_folds(S_init.shape[0], max(1, int(n_folds)), int(seed))
    valid_num = folds[int(fold_id) % len(folds)]
    train_num = _complement_indices(S_init.shape[0], valid_num)
    if valid_num.size == 0:
        valid_num = train_num
    if train_num.size == 0:
        train_num = valid_num
    w_train = None if weights is None else weights[train_num]
    w_valid = None if weights is None else weights[valid_num]
    return (S_init[train_num], None if A_init is None else A_init[train_num]), (S_init[valid_num], None if A_init is None else A_init[valid_num]), w_train, w_valid


def _initial_diagnostics(family: str, mode: str, source_state_query: Optional[Array], source_weight_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source_diag = _source_diagnostics(family, source_state_query, fit) if mode == "factored" else _source_diagnostics(family, None, None)
    if mode == "joint":
        joint_diag = _source_diagnostics(family, source_weight_query, fit)
        source_diag.update(
            initial_joint_ratio_enabled=True,
            initial_joint_ratio_mean=joint_diag["source_state_ratio_mean"],
            initial_joint_ratio_max=joint_diag["source_state_ratio_max"],
            initial_joint_ratio_ess_fraction=joint_diag["source_state_ratio_ess_fraction"],
            initial_joint_ratio_loss=joint_diag["source_state_ratio_loss"],
            initial_joint_ratio_density_ratio_loss=joint_diag["source_state_ratio_density_ratio_loss"],
            initial_joint_ratio_clipped_fraction=joint_diag["source_state_ratio_clipped_fraction"],
            initial_joint_ratio_query_clipped_fraction=joint_diag["source_state_ratio_query_clipped_fraction"],
            initial_joint_ratio_prediction_max=joint_diag["source_state_ratio_prediction_max"],
            initial_joint_ratio_prediction_scale=joint_diag["source_state_ratio_prediction_scale"],
        )
        if family == "neural":
            source_diag["initial_joint_ratio_updates"] = joint_diag.get("source_state_ratio_updates", 0.0)
        return source_diag
    source_diag.update(
        initial_joint_ratio_enabled=False,
        initial_joint_ratio_mean=1.0,
        initial_joint_ratio_max=1.0,
        initial_joint_ratio_ess_fraction=1.0,
        initial_joint_ratio_loss=float("nan"),
        initial_joint_ratio_density_ratio_loss="none",
        initial_joint_ratio_clipped_fraction=0.0,
        initial_joint_ratio_query_clipped_fraction=0.0,
        initial_joint_ratio_prediction_max=float("nan"),
        initial_joint_ratio_prediction_scale=1.0,
    )
    if family == "neural":
        source_diag["initial_joint_ratio_updates"] = 0.0
    return source_diag


def _source_diagnostics(family: str, values: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if family == "boosted":
        return _boosted_source_diagnostics(values, fit)
    from occupancy_ratio._neural_impl import _source_state_ratio_diagnostics as neural_source_diagnostics

    return neural_source_diagnostics(values, fit)


def _direct_diagnostics(family: str, values: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if family == "boosted":
        return _boosted_direct_diagnostics(values, fit)
    from occupancy_ratio._neural_impl import _one_step_direct_ratio_diagnostics as neural_direct_diagnostics

    return neural_direct_diagnostics(values, fit)


def _combine_fold_bundle(
    *,
    action_fit: Dict[str, Any],
    transition_fit: Dict[str, Any],
    source_piece: Dict[str, Any],
    direct_piece: Dict[str, Any],
    initial_mode: str,
    one_step_mode: str,
) -> Dict[str, Any]:
    return dict(
        action_fit=action_fit,
        transition_fit=transition_fit,
        source_fit=source_piece.get("source_fit"),
        source_weight_query=source_piece.get("source_weight_query"),
        source_state_query=source_piece.get("source_state_query"),
        source_diagnostics=source_piece.get("source_diagnostics", {}),
        c_fit=direct_piece.get("c_fit"),
        c_ratio_query=direct_piece.get("c_ratio_query"),
        c_diagnostics=direct_piece.get("c_diagnostics", {}),
        initial_ratio_mode=str(initial_mode),
        one_step_ratio_mode=str(one_step_mode),
    )


def _refit_diagnostics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    action_fit = dict(bundle.get("action_fit") or {})
    transition_fit = dict(bundle.get("transition_fit") or {})
    source_diag = dict(bundle.get("source_diagnostics") or {})
    direct_diag = dict(bundle.get("c_diagnostics") or {})
    out: Dict[str, Any] = dict(
        initial_ratio_mode=str(bundle.get("initial_ratio_mode", "")),
        one_step_ratio_mode=str(bundle.get("one_step_ratio_mode", "")),
        action_density_ratio_loss=str(action_fit.get("density_ratio_loss", "lsif")),
        transition_density_ratio_loss=str(transition_fit.get("density_ratio_loss", "lsif")),
        action_ratio_updates=_fit_update_count(action_fit),
        transition_ratio_updates=_fit_update_count(transition_fit),
        source_density_ratio_loss=str(source_diag.get("source_state_ratio_density_ratio_loss", "none")),
        initial_joint_ratio_density_ratio_loss=str(source_diag.get("initial_joint_ratio_density_ratio_loss", "none")),
        one_step_direct_ratio_density_ratio_loss=str(direct_diag.get("one_step_direct_ratio_density_ratio_loss", "none")),
    )
    for key, value in source_diag.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
    for key, value in direct_diag.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
    return out


def _expand_initial_modes(
    *,
    requested: str,
    configured: Sequence[str],
    has_initial_states: bool,
    has_initial_actions: bool,
    telemetry: Dict[str, Any],
    family: str,
) -> List[str]:
    if not has_initial_states:
        raw = (requested,) if str(requested).strip().lower() != "auto" else tuple(configured)
        skipped_modes = []
        for mode in raw:
            normalized = str(mode).strip().lower()
            expanded = "joint" if normalized == "auto" and has_initial_actions else "factored" if normalized == "auto" else normalized
            if expanded in {"joint", "factored"} and expanded not in skipped_modes:
                skipped_modes.append(expanded)
                telemetry["skipped"].append(dict(family=family, task="source_state_ratio", mode=expanded, reason="missing_initial_states"))
        return ["factored"]
    raw = (requested,) if str(requested).strip().lower() != "auto" else tuple(configured)
    out = []
    for mode in raw:
        normalized = str(mode).strip().lower()
        expanded = "joint" if normalized == "auto" and has_initial_actions else "factored" if normalized == "auto" else normalized
        if expanded == "joint" and not has_initial_actions:
            telemetry["skipped"].append(dict(family=family, task="source_state_ratio", mode=expanded, reason="missing_initial_actions"))
            continue
        if expanded not in {"joint", "factored"}:
            telemetry["skipped"].append(dict(family=family, task="source_state_ratio", mode=expanded, reason="invalid_mode"))
            continue
        if expanded not in out:
            out.append(expanded)
    return out or ["factored"]


def _expand_one_step_modes(
    *,
    requested: str,
    configured: Sequence[str],
    has_target_next_actions: bool,
    telemetry: Dict[str, Any],
    family: str,
) -> List[str]:
    raw = (requested,) if str(requested).strip().lower() != "auto" else tuple(configured)
    out = []
    for mode in raw:
        normalized = str(mode).strip().lower()
        expanded = "direct" if normalized == "auto" and has_target_next_actions else "factored" if normalized == "auto" else normalized
        if expanded == "direct" and not has_target_next_actions:
            telemetry["skipped"].append(dict(family=family, task="one_step_ratio", mode=expanded, reason="missing_target_next_actions"))
            continue
        if expanded not in {"direct", "factored"}:
            telemetry["skipped"].append(dict(family=family, task="one_step_ratio", mode=expanded, reason="invalid_mode"))
            continue
        if expanded not in out:
            out.append(expanded)
    return out or ["factored"]


def _fit_update_count(fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return 0.0
    for key in ("updates", "best_iteration", "num_updates"):
        value = fit.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return 0.0


def _source_piece_update_count(piece: Dict[str, Any]) -> float:
    diagnostics = dict(piece.get("source_diagnostics") or {})
    for key in ("initial_joint_ratio_updates", "source_state_ratio_updates"):
        value = diagnostics.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return _fit_update_count(piece.get("source_fit"))


def _direct_piece_update_count(piece: Dict[str, Any]) -> float:
    diagnostics = dict(piece.get("c_diagnostics") or {})
    value = diagnostics.get("one_step_direct_ratio_updates")
    if value is not None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float("nan")
        if np.isfinite(out):
            return out
    return _fit_update_count(piece.get("c_fit"))


def _append_candidate_row(
    telemetry: Dict[str, Any],
    family: str,
    task: str,
    candidate_id: str,
    mode: str,
    score: float,
    start: float,
    config: Any,
    error: str,
    *,
    update_count_mean: float = float("nan"),
) -> None:
    telemetry["candidate_rows"].append(
        dict(
            family=family,
            task=task,
            candidate_id=candidate_id,
            mode=mode,
            score=float(score),
            runtime_sec=float(time.perf_counter() - start),
            selected=0.0,
            error=error,
            update_count_mean=float(update_count_mean),
            config={} if config is None else _config_dict(config),
        )
    )


def _append_fold_row(
    telemetry: Dict[str, Any],
    family: str,
    task: str,
    candidate_id: str,
    mode: str,
    fold: int,
    score: float,
    *,
    update_count: float = float("nan"),
) -> None:
    telemetry["fold_rows"].append(
        dict(
            family=family,
            task=task,
            candidate_id=candidate_id,
            mode=mode,
            fold=int(fold),
            score=float(score),
            update_count=float(update_count),
        )
    )


def _mark_selected(telemetry: Dict[str, Any], task: str, candidate_id: str) -> None:
    for row in telemetry["candidate_rows"]:
        if row["task"] == task and row["candidate_id"] == candidate_id:
            row["selected"] = 1.0


def _config_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("inf")


def _make_simple_folds(n_rows: int, n_folds: int, seed: int) -> List[Array]:
    rng = np.random.default_rng(int(seed))
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(rng.permutation(int(n_rows)), int(n_folds))]


def _same_folds(left: Sequence[Array], right: Sequence[Array]) -> bool:
    if len(left) != len(right):
        return False
    return all(np.array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) for a, b in zip(left, right))


def _complement_indices(n_rows: int, valid_idx: Array) -> Array:
    mask = np.ones(int(n_rows), dtype=bool)
    mask[np.asarray(valid_idx, dtype=np.int64)] = False
    return np.flatnonzero(mask)
