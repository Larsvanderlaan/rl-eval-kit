"""LightGBM nuisance fitting helpers for boosted occupancy ratios."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np

from occupancy_ratio.fit_importance_and_transition_ratios import (
    fit_importance_ratio_lgbm,
    fit_state_density_ratio_lgbm,
    fit_transition_ratio_lgbm,
    _postprocess_ratio_predictions,
    _predict_ratio_from_booster,
)
from occupancy_ratio.stabilization import _ess, _nonnegative, _summarize_vector
from occupancy_ratio.validation import _as_2d


Array = np.ndarray

__all__ = [
    "fit_importance_ratio_lgbm",
    "fit_state_density_ratio_lgbm",
    "fit_transition_ratio_lgbm",
    "_fit_crossfit_nuisance_context",
    "_fit_direct_one_step_ratio",
    "_fit_eval_loss",
    "_fit_initial_ratio",
    "_fit_or_use_importance_ratio",
    "_fit_or_use_transition_ratio",
    "_fit_source_state_ratio",
    "_fit_prediction_max",
    "_make_factored_initial_source_weights",
    "_make_fold_indices",
    "_make_transition_reference_features",
    "_nuisance_prediction_scale",
    "_one_step_direct_ratio_diagnostics",
    "_predict_processed_nuisance",
    "_predict_processed_source_state_ratio",
    "_prepare_nuisance_kwargs",
    "_ratio_query_cap_fraction",
    "_source_state_ratio_diagnostics",
]


def _predict_processed_nuisance(*, fit: Dict[str, Any], X: Array, kind: str) -> Array:
    booster_key = "bst_iw" if kind == "iw" else "bst_k"
    raw = _predict_ratio_from_booster(
        booster=fit[booster_key],
        X=X,
        offset=float(fit.get("prediction_offset", 0.0)),
        density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")),
        logistic_logit_clip=fit.get("logistic_logit_clip", 20.0),
        prior_correction=float(fit.get("prior_correction", 1.0)),
    )
    pred, _ = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=True,
        prediction_max=fit.get("prediction_max"),
        prediction_power=float(fit.get("prediction_power", 1.0)),
        normalize_predictions=bool(fit.get("normalize_predictions", False)),
    )
    return pred * float(fit.get("prediction_scale", 1.0))


def _make_transition_reference_features(*, X_sa: Array, S_ref: Array, K: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    X_sa = np.asarray(X_sa, dtype=np.float32)
    S_ref = _as_2d(np.asarray(S_ref, dtype=np.float32), "S_ref")
    blocks = []
    for _ in range(int(K)):
        blocks.append(np.hstack([X_sa, S_ref[rng.permutation(S_ref.shape[0])]]))
    return np.vstack(blocks)


def _prepare_nuisance_kwargs(
    *,
    lgb_params: Optional[Dict[str, Any]],
    k_lgb_params: Optional[Dict[str, Any]],
    iw_lgb_params: Optional[Dict[str, Any]],
    k_kwargs: Optional[Dict[str, Any]],
    iw_kwargs: Optional[Dict[str, Any]],
    source_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    base = {} if lgb_params is None else dict(lgb_params)
    k_options = {} if k_kwargs is None else dict(k_kwargs)
    iw_options = {} if iw_kwargs is None else dict(iw_kwargs)
    source_options = {} if source_kwargs is None else dict(source_kwargs)
    k_options.setdefault("refit_on_all_data", False)
    iw_options.setdefault("refit_on_all_data", False)
    source_options.setdefault("refit_on_all_data", False)
    k_options.setdefault("lgb_params", dict(base) | ({} if k_lgb_params is None else dict(k_lgb_params)))
    iw_options.setdefault("lgb_params", dict(base) | ({} if iw_lgb_params is None else dict(iw_lgb_params)))
    source_options.setdefault("lgb_params", dict(base))
    return {"k_kwargs": k_options, "iw_kwargs": iw_options, "source_kwargs": source_options}


def _fit_or_use_transition_ratio(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    seed: int,
    bst_k_init: Optional[lgb.Booster],
    bst_k_init_offset: float,
    k_kwargs: Dict[str, Any],
) -> tuple[lgb.Booster, Optional[Dict[str, Any]], float]:
    if bst_k_init is not None:
        return bst_k_init, None, float(bst_k_init_offset)
    fit = fit_transition_ratio_lgbm(S=S, A=A, S_next=S_next, seed=seed, **k_kwargs)
    return fit["bst_k"], fit, float(fit.get("prediction_offset", 0.0))


def _fit_or_use_importance_ratio(
    *,
    S: Array,
    A: Array,
    A_pi: Array,
    S_pi: Array,
    target_row_index: Array,
    X_sa_beh: Array,
    X_sa_query: Array,
    seed: int,
    bst_iw_init: Optional[lgb.Booster],
    bst_iw_init_offset: float,
    known_iw_beh: Optional[Array],
    known_iw_query: Optional[Array],
    iw_kwargs: Dict[str, Any],
) -> tuple[Optional[lgb.Booster], Optional[Dict[str, Any]], Array, float]:
    if known_iw_beh is not None:
        known_beh = np.asarray(known_iw_beh, dtype=np.float64).reshape(-1)
        known_query = (
            None
            if known_iw_query is None
            else np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
        )
        fit = dict(
            w_hat=known_beh,
            w_hat_raw=known_beh,
            w_hat_summary=_summarize_vector(known_beh),
            known_action_ratio=True,
            prediction_max=iw_kwargs.get("prediction_max"),
            prediction_power=float(iw_kwargs.get("prediction_power", 1.0)),
            normalize_predictions=bool(iw_kwargs.get("normalize_predictions", False)),
            known_action_ratio_features=X_sa_query if known_query is not None else X_sa_beh,
            known_action_ratio_predictions=known_query if known_query is not None else known_beh,
        )
        return None, fit, known_beh, 0.0
    if bst_iw_init is not None:
        offset = float(bst_iw_init_offset)
        density_ratio_loss = str(iw_kwargs.get("density_ratio_loss", "lsif"))
        logistic_logit_clip = iw_kwargs.get("logistic_logit_clip", 20.0)
        prior_correction = float(iw_kwargs.get("prior_correction", 1.0))
        iw_hat_raw = _predict_ratio_from_booster(
            booster=bst_iw_init,
            X=X_sa_beh,
            offset=offset,
            density_ratio_loss=density_ratio_loss,
            logistic_logit_clip=logistic_logit_clip,
            prior_correction=prior_correction,
        )
        iw_hat, iw_summary = _postprocess_ratio_predictions(
            iw_hat_raw,
            clip_nonneg=bool(iw_kwargs.get("clip_nonneg", True)),
            prediction_max=iw_kwargs.get("prediction_max"),
            prediction_power=float(iw_kwargs.get("prediction_power", 1.0)),
            normalize_predictions=bool(iw_kwargs.get("normalize_predictions", False)),
        )
        fit = dict(
            w_hat=iw_hat,
            w_hat_raw=iw_hat_raw,
            w_hat_summary=iw_summary,
            prefit=True,
            prediction_max=iw_kwargs.get("prediction_max"),
            prediction_power=float(iw_kwargs.get("prediction_power", 1.0)),
            normalize_predictions=bool(iw_kwargs.get("normalize_predictions", False)),
        )
        return bst_iw_init, fit, iw_hat, offset
    if A_pi.shape[0] == S.shape[0]:
        fit = fit_importance_ratio_lgbm(S=S, A=A, A_pi=A_pi, seed=seed, **iw_kwargs)
    else:
        row_index = np.asarray(target_row_index, dtype=np.int64).reshape(-1)
        fit = fit_importance_ratio_lgbm(
            S=S_pi,
            A=A[row_index],
            A_pi=A_pi,
            seed=seed,
            **iw_kwargs,
        )
    offset = float(fit.get("prediction_offset", 0.0))
    iw_hat_beh = _predict_processed_nuisance(fit=fit, X=X_sa_beh, kind="iw")
    return fit["bst_iw"], fit, _nonnegative(iw_hat_beh), offset


def _fit_source_state_ratio(
    *,
    S: Array,
    S_query: Array,
    S_initial: Optional[Array],
    initial_weights: Optional[Array],
    seed: int,
    source_kwargs: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Dict[str, Any]]:
    if S_initial is None:
        return None, None, _source_state_ratio_diagnostics(None, None)
    fit = fit_state_density_ratio_lgbm(
        S_ref=S,
        S_num=S_initial,
        numerator_weights=initial_weights,
        seed=seed + 53_001,
        **source_kwargs,
    )
    source_query = _predict_processed_source_state_ratio(fit=fit, X=S_query)
    return fit, source_query, _source_state_ratio_diagnostics(source_query, fit)


def _fit_initial_ratio(
    *,
    S: Array,
    X_sa_beh: Array,
    S_query: Array,
    X_sa_query: Array,
    S_initial: Optional[Array],
    X_sa_initial: Optional[Array],
    initial_weights: Optional[Array],
    seed: int,
    source_kwargs: Dict[str, Any],
    initial_ratio_mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Optional[Array], Dict[str, Any]]:
    if initial_ratio_mode == "joint":
        if X_sa_initial is None:
            raise ValueError("initial_ratio_mode='joint' requires initial state-action rows.")
        fit = fit_state_density_ratio_lgbm(
            S_ref=X_sa_beh,
            S_num=X_sa_initial,
            numerator_weights=initial_weights,
            seed=seed + 53_001,
            **source_kwargs,
        )
        source_query = _predict_processed_source_state_ratio(fit=fit, X=X_sa_query)
        joint_diagnostics = _source_state_ratio_diagnostics(source_query, fit)
        diagnostics = _source_state_ratio_diagnostics(None, None)
        diagnostics.update(
            initial_joint_ratio_enabled=True,
            initial_joint_ratio_mean=joint_diagnostics["source_state_ratio_mean"],
            initial_joint_ratio_max=joint_diagnostics["source_state_ratio_max"],
            initial_joint_ratio_ess_fraction=joint_diagnostics["source_state_ratio_ess_fraction"],
            initial_joint_ratio_loss=joint_diagnostics["source_state_ratio_loss"],
            initial_joint_ratio_density_ratio_loss=joint_diagnostics["source_state_ratio_density_ratio_loss"],
            initial_joint_ratio_clipped_fraction=joint_diagnostics["source_state_ratio_clipped_fraction"],
            initial_joint_ratio_query_clipped_fraction=joint_diagnostics[
                "source_state_ratio_query_clipped_fraction"
            ],
            initial_joint_ratio_prediction_max=joint_diagnostics["source_state_ratio_prediction_max"],
            initial_joint_ratio_prediction_scale=joint_diagnostics["source_state_ratio_prediction_scale"],
        )
        return fit, source_query, None, diagnostics

    fit, source_state_query, diagnostics = _fit_source_state_ratio(
        S=S,
        S_query=S_query,
        S_initial=S_initial,
        initial_weights=initial_weights,
        seed=seed,
        source_kwargs=source_kwargs,
    )
    diagnostics.update(
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
    return fit, None, source_state_query, diagnostics


def _make_factored_initial_source_weights(
    *,
    bst_iw: Optional[lgb.Booster],
    iw_fit: Optional[Dict[str, Any]],
    iw_kwargs: Dict[str, Any],
    iw_prediction_offset: float,
    X_sa_query: Array,
    source_state_query: Optional[Array],
    known_iw_query: Optional[Array] = None,
) -> Array:
    if known_iw_query is not None:
        out = np.asarray(known_iw_query, dtype=np.float64).reshape(-1)
    else:
        if bst_iw is None:
            raise ValueError("Action-ratio booster is required when known query ratios are unavailable.")
        raw = _predict_ratio_from_booster(
            booster=bst_iw,
            X=X_sa_query,
            offset=float(iw_prediction_offset),
            density_ratio_loss=str((iw_fit or {}).get("density_ratio_loss", iw_kwargs.get("density_ratio_loss", "lsif"))),
            logistic_logit_clip=(iw_fit or {}).get("logistic_logit_clip", iw_kwargs.get("logistic_logit_clip", 20.0)),
            prior_correction=float((iw_fit or {}).get("prior_correction", iw_kwargs.get("prior_correction", 1.0))),
        )
        pred, _ = _postprocess_ratio_predictions(
            raw,
            clip_nonneg=True,
            prediction_max=(iw_fit or {}).get("prediction_max", iw_kwargs.get("prediction_max")),
            prediction_power=float((iw_fit or {}).get("prediction_power", iw_kwargs.get("prediction_power", 1.0))),
            normalize_predictions=bool((iw_fit or {}).get("normalize_predictions", iw_kwargs.get("normalize_predictions", False))),
        )
        out = pred * _nuisance_prediction_scale(iw_fit)
    if source_state_query is not None:
        out = out * np.asarray(source_state_query, dtype=np.float64).reshape(-1)
    return np.maximum(out, 0.0)


def _fit_direct_one_step_ratio(
    *,
    X_ref: Array,
    X_next_pi: Optional[Array],
    X_query: Array,
    seed: int,
    source_kwargs: Dict[str, Any],
    one_step_ratio_mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Array], Dict[str, Any]]:
    if one_step_ratio_mode != "direct":
        return None, None, _one_step_direct_ratio_diagnostics(None, None)
    if X_next_pi is None:
        raise ValueError("one_step_ratio_mode='direct' requires target_next_actions.")
    fit = fit_state_density_ratio_lgbm(
        S_ref=X_ref,
        S_num=X_next_pi,
        numerator_weights=None,
        seed=seed + 61_001,
        **source_kwargs,
    )
    c_query = _predict_processed_source_state_ratio(fit=fit, X=X_query)
    return fit, c_query, _one_step_direct_ratio_diagnostics(c_query, fit)


def _one_step_direct_ratio_diagnostics(c_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if c_query is None:
        return dict(
            one_step_direct_ratio_enabled=False,
            one_step_direct_ratio_mean=1.0,
            one_step_direct_ratio_max=1.0,
            one_step_direct_ratio_ess_fraction=1.0,
            one_step_direct_ratio_loss=float("nan"),
            one_step_direct_ratio_clipped_fraction=0.0,
            one_step_direct_ratio_query_clipped_fraction=0.0,
            one_step_direct_ratio_density_ratio_loss="none",
            one_step_direct_ratio_prediction_max=float("nan"),
            one_step_direct_ratio_prediction_scale=1.0,
        )
    values = np.asarray(c_query, dtype=np.float64).reshape(-1)
    summary = fit.get("source_hat_summary", {}) if isinstance(fit, dict) else {}
    query_cap_fraction = _ratio_query_cap_fraction(values, fit)
    return dict(
        one_step_direct_ratio_enabled=True,
        one_step_direct_ratio_mean=float(np.mean(values)) if values.size else float("nan"),
        one_step_direct_ratio_max=float(np.max(values)) if values.size else float("nan"),
        one_step_direct_ratio_ess_fraction=float(_ess(values) / max(values.size, 1)),
        one_step_direct_ratio_loss=_fit_eval_loss(fit),
        one_step_direct_ratio_clipped_fraction=float(summary.get("clipped_fraction", 0.0)),
        one_step_direct_ratio_query_clipped_fraction=float(query_cap_fraction),
        one_step_direct_ratio_density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")) if isinstance(fit, dict) else "",
        one_step_direct_ratio_prediction_max=_fit_prediction_max(fit),
        one_step_direct_ratio_prediction_scale=_nuisance_prediction_scale(fit),
    )


def _predict_processed_source_state_ratio(*, fit: Dict[str, Any], X: Array) -> Array:
    raw = _predict_ratio_from_booster(
        booster=fit["bst_source"],
        X=X,
        offset=float(fit.get("prediction_offset", 0.0)),
        density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")),
        logistic_logit_clip=fit.get("logistic_logit_clip", 20.0),
        prior_correction=float(fit.get("prior_correction", 1.0)),
    )
    pred, _ = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=True,
        prediction_max=fit.get("prediction_max"),
        prediction_power=float(fit.get("prediction_power", 1.0)),
        normalize_predictions=bool(fit.get("normalize_predictions", False)),
    )
    return pred * _nuisance_prediction_scale(fit)


def _source_state_ratio_diagnostics(source_query: Optional[Array], fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if source_query is None:
        return dict(
            source_state_ratio_enabled=False,
            source_state_ratio_mean=1.0,
            source_state_ratio_max=1.0,
            source_state_ratio_ess_fraction=1.0,
            source_state_ratio_loss=float("nan"),
            source_state_ratio_density_ratio_loss="none",
            source_state_ratio_clipped_fraction=0.0,
            source_state_ratio_query_clipped_fraction=0.0,
            source_state_ratio_prediction_max=float("nan"),
            source_state_ratio_prediction_scale=1.0,
        )
    values = np.asarray(source_query, dtype=np.float64).reshape(-1)
    summary = fit.get("source_hat_summary", {}) if isinstance(fit, dict) else {}
    query_cap_fraction = _ratio_query_cap_fraction(values, fit)
    return dict(
        source_state_ratio_enabled=True,
        source_state_ratio_mean=float(np.mean(values)) if values.size else float("nan"),
        source_state_ratio_max=float(np.max(values)) if values.size else float("nan"),
        source_state_ratio_ess_fraction=float(_ess(values) / max(values.size, 1)),
        source_state_ratio_loss=_fit_eval_loss(fit),
        source_state_ratio_density_ratio_loss=str(fit.get("density_ratio_loss", "lsif")) if isinstance(fit, dict) else "",
        source_state_ratio_clipped_fraction=float(summary.get("clipped_fraction", 0.0)),
        source_state_ratio_query_clipped_fraction=float(query_cap_fraction),
        source_state_ratio_prediction_max=_fit_prediction_max(fit),
        source_state_ratio_prediction_scale=_nuisance_prediction_scale(fit),
    )


def _fit_prediction_max(fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return float("nan")
    value = fit.get("prediction_max")
    if value is None:
        return float("nan")
    return float(value)


def _ratio_query_cap_fraction(values: Array, fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return 0.0
    prediction_max = fit.get("prediction_max")
    if prediction_max is None:
        return 0.0
    cap = float(prediction_max) * float(_nuisance_prediction_scale(fit))
    if not np.isfinite(cap) or cap <= 0.0:
        return 0.0
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.mean(x >= cap * (1.0 - 1e-10)))


def _fit_eval_loss(fit: Optional[Dict[str, Any]]) -> float:
    if not isinstance(fit, dict):
        return float("nan")
    evals = fit.get("evals_result")
    if not isinstance(evals, dict):
        return float("nan")
    valid = evals.get("valid")
    if not isinstance(valid, dict) or not valid:
        return float("nan")
    for key in ("loss", "binary_logloss"):
        values = valid.get(key)
        if values:
            finite = [float(v) for v in values if np.isfinite(float(v))]
            return float(np.min(finite)) if finite else float("nan")
    for values in valid.values():
        if values:
            finite = [float(v) for v in values if np.isfinite(float(v))]
            return float(np.min(finite)) if finite else float("nan")
    return float("nan")


def _nuisance_prediction_scale(fit: Optional[Dict[str, Any]]) -> float:
    if isinstance(fit, dict):
        return float(fit.get("prediction_scale", 1.0))
    return 1.0


def _fit_crossfit_nuisance_context(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    A_pi: Array,
    X_sa_beh: Array,
    seed: int,
    bst_k_final: lgb.Booster,
    bst_iw_final: lgb.Booster,
    k_fit_final: Optional[Dict[str, Any]],
    iw_fit_final: Optional[Dict[str, Any]],
    k_prediction_offset: float,
    iw_prediction_offset: float,
    k_kwargs: Dict[str, Any],
    iw_kwargs: Dict[str, Any],
    bst_k_init: Optional[lgb.Booster],
    bst_iw_init: Optional[lgb.Booster],
) -> Optional[Dict[str, Any]]:
    k_folds = int(k_kwargs.get("crossfit_folds", 1) or 1)
    iw_folds = int(iw_kwargs.get("crossfit_folds", 1) or 1)
    folds = max(k_folds, iw_folds)
    if folds <= 1:
        return None
    if bst_iw_final is None:
        return None
    if bst_k_init is not None or bst_iw_init is not None:
        return None

    fold_indices = _make_fold_indices(
        S.shape[0],
        folds,
        int(iw_kwargs.get("crossfit_seed") or k_kwargs.get("crossfit_seed") or seed + 31_337),
    )
    k_models = []
    iw_models = []
    iw_oof = np.empty(S.shape[0], dtype=np.float64)
    k_oof = np.empty(S.shape[0], dtype=np.float64)
    for fold_id, valid_idx in enumerate(fold_indices):
        train_mask = np.ones(S.shape[0], dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.flatnonzero(train_mask)

        if iw_folds > 1:
            iw_options = dict(iw_kwargs)
            iw_options["crossfit_folds"] = 1
            iw_options["show_tqdm"] = False
            iw_fit = fit_importance_ratio_lgbm(
                S=S[train_idx],
                A=A[train_idx],
                A_pi=A_pi[train_idx],
                seed=seed + 1_003 * (fold_id + 1),
                **iw_options,
            )
            iw_model = dict(
                booster=iw_fit["bst_iw"],
                offset=float(iw_fit.get("prediction_offset", 0.0)),
                scale=_nuisance_prediction_scale(iw_fit),
                fit=iw_fit,
                density_ratio_loss=str(iw_fit.get("density_ratio_loss", "lsif")),
                logistic_logit_clip=iw_fit.get("logistic_logit_clip", 20.0),
                prior_correction=float(iw_fit.get("prior_correction", 1.0)),
            )
            raw = _predict_ratio_from_booster(
                booster=iw_model["booster"],
                X=X_sa_beh[valid_idx],
                offset=float(iw_model["offset"]),
                density_ratio_loss=str(iw_model["density_ratio_loss"]),
                logistic_logit_clip=iw_model["logistic_logit_clip"],
                prior_correction=float(iw_model["prior_correction"]),
            )
            pred, _ = _postprocess_ratio_predictions(
                raw,
                clip_nonneg=bool(iw_options.get("clip_nonneg", True)),
                prediction_max=iw_options.get("prediction_max"),
                prediction_power=float(iw_options.get("prediction_power", 1.0)),
                normalize_predictions=bool(iw_options.get("normalize_predictions", False)),
            )
            iw_oof[valid_idx] = pred * iw_model["scale"]
        else:
            iw_model = dict(
                booster=bst_iw_final,
                offset=iw_prediction_offset,
                scale=_nuisance_prediction_scale(iw_fit_final),
                fit=iw_fit_final,
                density_ratio_loss=str((iw_fit_final or {}).get("density_ratio_loss", "lsif")),
                logistic_logit_clip=(iw_fit_final or {}).get("logistic_logit_clip", 20.0),
                prior_correction=float((iw_fit_final or {}).get("prior_correction", 1.0)),
            )
            iw_oof[valid_idx] = np.nan
        iw_models.append(iw_model)

        if k_folds > 1:
            k_options = dict(k_kwargs)
            k_options["crossfit_folds"] = 1
            k_options["show_tqdm"] = False
            k_fit = fit_transition_ratio_lgbm(
                S=S[train_idx],
                A=A[train_idx],
                S_next=S_next[train_idx],
                seed=seed + 2_003 * (fold_id + 1),
                **k_options,
            )
            k_model = dict(
                booster=k_fit["bst_k"],
                offset=float(k_fit.get("prediction_offset", 0.0)),
                scale=_nuisance_prediction_scale(k_fit),
                fit=k_fit,
                density_ratio_loss=str(k_fit.get("density_ratio_loss", "lsif")),
                logistic_logit_clip=k_fit.get("logistic_logit_clip", 20.0),
                prior_correction=float(k_fit.get("prior_correction", 1.0)),
            )
            Xk_valid = np.hstack([X_sa_beh[valid_idx], S_next[valid_idx]])
            raw = _predict_ratio_from_booster(
                booster=k_model["booster"],
                X=Xk_valid,
                offset=float(k_model["offset"]),
                density_ratio_loss=str(k_model["density_ratio_loss"]),
                logistic_logit_clip=k_model["logistic_logit_clip"],
                prior_correction=float(k_model["prior_correction"]),
            )
            pred, _ = _postprocess_ratio_predictions(
                raw,
                clip_nonneg=bool(k_options.get("clip_nonneg", True)),
                prediction_max=k_options.get("prediction_max"),
                prediction_power=float(k_options.get("prediction_power", 1.0)),
                normalize_predictions=bool(k_options.get("normalize_predictions", False)),
            )
            k_oof[valid_idx] = pred * k_model["scale"]
        else:
            k_model = dict(
                booster=bst_k_final,
                offset=k_prediction_offset,
                scale=_nuisance_prediction_scale(k_fit_final),
                fit=k_fit_final,
                density_ratio_loss=str((k_fit_final or {}).get("density_ratio_loss", "lsif")),
                logistic_logit_clip=(k_fit_final or {}).get("logistic_logit_clip", 20.0),
                prior_correction=float((k_fit_final or {}).get("prior_correction", 1.0)),
            )
            k_oof[valid_idx] = np.nan
        k_models.append(k_model)

    diagnostics = dict(
        enabled=True,
        folds=int(folds),
        action_crossfit_folds=int(iw_folds),
        transition_crossfit_folds=int(k_folds),
        action_oof_mean=float(np.nanmean(iw_oof)) if np.any(np.isfinite(iw_oof)) else float("nan"),
        transition_oof_mean=float(np.nanmean(k_oof)) if np.any(np.isfinite(k_oof)) else float("nan"),
    )
    return dict(
        folds=fold_indices,
        iw_models=iw_models,
        k_models=k_models,
        iw_oof=iw_oof,
        k_oof=k_oof,
        diagnostics=diagnostics,
    )


def _make_fold_indices(n_rows: int, n_folds: int, seed: int) -> List[Array]:
    if int(n_folds) < 1:
        raise ValueError("crossfit_folds must be >= 1.")
    rng = np.random.default_rng(seed)
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(rng.permutation(n_rows), int(n_folds))]
