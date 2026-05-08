from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
from tqdm import tqdm


Array = np.ndarray


def fit_transition_ratio_lgbm(
    *,
    S: Array,
    A: Array,
    S_next: Array,
    K_perm: int = 20,
    seed: int = 123,
    clip_nonneg: bool = True,
    num_boost_round: int = 300,
    lgb_params: Optional[Dict[str, Any]] = None,
    eps_hess: float = 0.0,
    test_size: float = 0.2,
    early_stopping_rounds: int = 10,
    refit_on_all_data: bool = True,
    show_tqdm: bool = True,
    tqdm_leave: bool = True,
    tqdm_desc_es: str = "boosting transition kernel (early stopping)",
    tqdm_desc_refit: str = "boosting transition kernel (refit)",
    early_stopping_min_delta: float = 0.0,
    init_score_value: float = 1.0,
    prediction_max: Optional[float] = None,
    prediction_power: float = 1.0,
    normalize_predictions: bool = False,
    prediction_norm_eps: float = 1e-12,
    moment_calibration: str = "none",
    crossfit_folds: int = 1,
    crossfit_seed: Optional[int] = None,
    density_ratio_loss: str = "lsif",
    logistic_logit_clip: Optional[float] = 20.0,
) -> Dict[str, Any]:
    """Fit ``P(s_next | s,a) / rho_ref(s_next)`` with an LSIF objective.

    The behavior block contains observed triples ``(S, A, S_next)`` and supplies
    the linear term ``-2 E_P[k]``. The reference block pairs each observed
    ``(S, A)`` with independently permuted states from ``S`` and supplies
    ``E_ref[k^2]``. Using ``S`` as the reference makes the denominator the
    baseline state density ``rho0`` expected by the occupancy fixed point.
    """
    if K_perm <= 0:
        raise ValueError("K_perm must be positive.")
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be nonnegative.")
    _validate_prediction_postprocess(
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        prediction_norm_eps=prediction_norm_eps,
    )
    _validate_extra_nuisance_options(
        moment_calibration=moment_calibration,
        crossfit_folds=crossfit_folds,
    )
    density_ratio_loss = _validate_density_ratio_loss(density_ratio_loss)
    _validate_logistic_options(logistic_logit_clip)

    S_feat = _as_2d_float32(S, "S")
    A_feat = _as_2d_float32(A, "A")
    S_next_feat = _as_2d_float32(S_next, "S_next")
    _validate_same_rows(S=S_feat, A=A_feat, S_next=S_next_feat)
    _validate_test_size(test_size)

    X_sa = np.concatenate([S_feat, A_feat], axis=1)
    train_idx, valid_idx = _split_rows(X_sa.shape[0], test_size=test_size, seed=seed)

    X_train, y_train, w_train = make_transition_ratio_long_arrays(
        X_sa=X_sa[train_idx],
        X_s_next=S_next_feat[train_idx],
        X_s_ref=S_feat[train_idx],
        K=K_perm,
        seed=seed,
    )
    X_valid, y_valid, w_valid = make_transition_ratio_long_arrays(
        X_sa=X_sa[valid_idx],
        X_s_next=S_next_feat[valid_idx],
        X_s_ref=S_feat[valid_idx],
        K=K_perm,
        seed=seed + 1,
    )

    dtrain = make_lgb_transition_ratio_dataset_from_arrays(X_train, y_train, w_train)
    dvalid = make_lgb_transition_ratio_dataset_from_arrays(X_valid, y_valid, w_valid)
    params = _transition_lgb_params(lgb_params)
    num_boost_round = extract_num_boost_round(params, default=num_boost_round)
    feval = transition_ratio_eval
    if density_ratio_loss == "lsif":
        _set_constant_init_score(dtrain, X_train.shape[0], init_score_value)
        _set_constant_init_score(dvalid, X_valid.shape[0], init_score_value)

        def objective(preds: Array, dataset: lgb.Dataset) -> tuple[Array, Array]:
            return transition_ratio_objective(preds, dataset, eps=float(eps_hess))

        params["objective"] = objective
    else:
        params = _binary_ratio_lgb_params(params)
        feval = None
    callbacks = _make_callbacks(
        total=num_boost_round,
        desc=tqdm_desc_es,
        evals_result={},
        early_stopping_rounds=early_stopping_rounds,
        early_stopping_min_delta=early_stopping_min_delta,
        show_tqdm=show_tqdm,
        tqdm_leave=tqdm_leave,
    )
    evals_result = callbacks["evals_result"]
    bst_es = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=feval,
        callbacks=callbacks["callbacks"],
    )
    best_iteration = _best_iteration(bst_es, fallback=num_boost_round)

    if refit_on_all_data:
        X_long, y_long, w_long = make_transition_ratio_long_arrays(
            X_sa=X_sa,
            X_s_next=S_next_feat,
            X_s_ref=S_feat,
            K=K_perm,
            seed=seed,
        )
        dall = make_lgb_transition_ratio_dataset_from_arrays(X_long, y_long, w_long)
        if density_ratio_loss == "lsif":
            _set_constant_init_score(dall, X_long.shape[0], init_score_value)
        bst_k = lgb.train(
            params=params,
            train_set=dall,
            num_boost_round=best_iteration,
            valid_sets=[dall],
            valid_names=["train"],
            feval=feval,
            callbacks=_make_callbacks(
                total=best_iteration,
                desc=tqdm_desc_refit,
                show_tqdm=show_tqdm,
                tqdm_leave=tqdm_leave,
            )["callbacks"],
        )
        prior_correction = _binary_prior_correction(y_long, w_long)
    else:
        bst_k = bst_es
        prior_correction = _binary_prior_correction(y_train, w_train)

    Xk_beh = np.hstack([X_sa, S_next_feat])
    k_hat_raw = _predict_ratio_from_booster(
        booster=bst_k,
        X=Xk_beh,
        offset=float(init_score_value),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=logistic_logit_clip,
        prior_correction=prior_correction,
        num_iteration=best_iteration,
    )
    k_hat, k_summary = _postprocess_ratio_predictions(
        k_hat_raw,
        clip_nonneg=clip_nonneg,
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        normalize_predictions=normalize_predictions,
        eps=prediction_norm_eps,
    )
    k_hat, calibration = _calibrate_transition_predictions(
        booster=bst_k,
        X_sa=X_sa,
        S_ref=S_feat,
        predictions=k_hat,
        offset=float(init_score_value),
        K_perm=K_perm,
        seed=seed + 91_337,
        clip_nonneg=clip_nonneg,
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        normalize_predictions=normalize_predictions,
        eps=prediction_norm_eps,
        moment_calibration=moment_calibration,
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=logistic_logit_clip,
        prior_correction=prior_correction,
    )
    k_summary = _ratio_prediction_summary(
        k_hat,
        clipped_fraction=float(k_summary.get("clipped_fraction", 0.0)),
        normalization_scale=float(k_summary.get("normalization_scale", 1.0)),
    )

    return dict(
        bst_k=bst_k,
        best_iteration=best_iteration,
        k_hat=k_hat,
        k_hat_raw=k_hat_raw,
        k_hat_summary=k_summary,
        Xk_beh=Xk_beh,
        X_sa=X_sa,
        S_feat=S_feat,
        S_next_feat=S_next_feat,
        evals_result=evals_result,
        prediction_offset=float(init_score_value),
        prediction_max=None if prediction_max is None else float(prediction_max),
        prediction_power=float(prediction_power),
        normalize_predictions=bool(normalize_predictions),
        prediction_scale=float(calibration["scale"]),
        moment_calibration=str(moment_calibration),
        calibration=calibration,
        crossfit_folds=int(crossfit_folds),
        crossfit_seed=None if crossfit_seed is None else int(crossfit_seed),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=None if logistic_logit_clip is None else float(logistic_logit_clip),
        prior_correction=float(prior_correction),
    )


def fit_importance_ratio_lgbm(
    *,
    S: Array,
    A: Array,
    A_pi: Array,
    clip_nonneg: bool = True,
    num_boost_round: int = 100,
    lgb_params: Optional[Dict[str, Any]] = None,
    eps_hess: float = 1e-3,
    test_size: float = 0.2,
    early_stopping_rounds: int = 10,
    seed: int = 123,
    refit_on_all_data: bool = True,
    show_tqdm: bool = True,
    tqdm_leave: bool = True,
    tqdm_desc_es: str = "boosting importance weights (early stopping)",
    tqdm_desc_refit: str = "boosting importance weights (refit)",
    init_score_value: float = 1.0,
    prediction_max: Optional[float] = None,
    prediction_power: float = 1.0,
    normalize_predictions: bool = False,
    prediction_norm_eps: float = 1e-12,
    moment_calibration: str = "none",
    crossfit_folds: int = 1,
    crossfit_seed: Optional[int] = None,
    density_ratio_loss: str = "lsif",
    logistic_logit_clip: Optional[float] = 20.0,
) -> Dict[str, Any]:
    """Fit the action importance ratio ``pi(a | s) / pi0(a | s)``.

    The long dataset has a behavior block ``(S, A)`` for the quadratic term and
    a target-policy block ``(S, A_pi)`` for the linear term, giving the LSIF
    population objective ``E_pi0[iota^2] - 2 E_pi[iota]``.
    """
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be nonnegative.")
    _validate_prediction_postprocess(
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        prediction_norm_eps=prediction_norm_eps,
    )
    _validate_extra_nuisance_options(
        moment_calibration=moment_calibration,
        crossfit_folds=crossfit_folds,
    )
    density_ratio_loss = _validate_density_ratio_loss(density_ratio_loss)
    _validate_logistic_options(logistic_logit_clip)

    S_feat = _as_2d_float32(S, "S")
    A_feat = _as_2d_float32(A, "A")
    A_pi_feat = _as_2d_float32(A_pi, "A_pi")
    _validate_same_rows(S=S_feat, A=A_feat, A_pi=A_pi_feat)
    if A_pi_feat.shape[1] != A_feat.shape[1]:
        raise ValueError("A_pi must have the same feature dimension as A.")
    _validate_test_size(test_size)

    X_sa = np.concatenate([S_feat, A_feat], axis=1)
    X_sa_pi = np.concatenate([S_feat, A_pi_feat], axis=1)
    train_idx, valid_idx = _split_rows(X_sa.shape[0], test_size=test_size, seed=seed)

    make_action_arrays = (
        make_importance_ratio_long_arrays_from_sa
        if density_ratio_loss == "lsif"
        else make_importance_ratio_binary_arrays_from_sa
    )
    X_train, y_train, w_train = make_action_arrays(X_sa_beh=X_sa[train_idx], X_sa_pi=X_sa_pi[train_idx])
    X_valid, y_valid, w_valid = make_action_arrays(X_sa_beh=X_sa[valid_idx], X_sa_pi=X_sa_pi[valid_idx])

    dtrain = make_lgb_importance_ratio_dataset(X_train, y_train, w_train)
    dvalid = make_lgb_importance_ratio_dataset(X_valid, y_valid, w_valid)

    params = _importance_lgb_params(lgb_params)
    num_boost_round = extract_num_boost_round(params, default=num_boost_round)
    feval = importance_ratio_eval
    if density_ratio_loss == "lsif":
        _set_constant_init_score(dtrain, X_train.shape[0], init_score_value)
        _set_constant_init_score(dvalid, X_valid.shape[0], init_score_value)

        def objective(preds: Array, dataset: lgb.Dataset) -> tuple[Array, Array]:
            return importance_ratio_objective(preds, dataset, eps=float(eps_hess))

        params["objective"] = objective
    else:
        params = _binary_ratio_lgb_params(params)
        feval = None
    callbacks = _make_callbacks(
        total=num_boost_round,
        desc=tqdm_desc_es,
        evals_result={},
        early_stopping_rounds=early_stopping_rounds,
        show_tqdm=show_tqdm,
        tqdm_leave=tqdm_leave,
    )
    evals_result = callbacks["evals_result"]
    bst_es = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=feval,
        callbacks=callbacks["callbacks"],
    )
    best_iteration = _best_iteration(bst_es, fallback=num_boost_round)

    if refit_on_all_data:
        X_long, y_long, w_long = make_action_arrays(X_sa_beh=X_sa, X_sa_pi=X_sa_pi)
        dall = make_lgb_importance_ratio_dataset(X_long, y_long, w_long)
        if density_ratio_loss == "lsif":
            _set_constant_init_score(dall, X_long.shape[0], init_score_value)
        bst_iw = lgb.train(
            params=params,
            train_set=dall,
            num_boost_round=best_iteration,
            valid_sets=[dall],
            valid_names=["train"],
            feval=feval,
            callbacks=_make_callbacks(
                total=best_iteration,
                desc=tqdm_desc_refit,
                show_tqdm=show_tqdm,
                tqdm_leave=tqdm_leave,
            )["callbacks"],
        )
        prior_correction = _binary_prior_correction(y_long, w_long)
    else:
        bst_iw = bst_es
        prior_correction = _binary_prior_correction(y_train, w_train)

    w_hat_raw = _predict_ratio_from_booster(
        booster=bst_iw,
        X=X_sa,
        offset=float(init_score_value),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=logistic_logit_clip,
        prior_correction=prior_correction,
        num_iteration=best_iteration,
    )
    w_hat, w_summary = _postprocess_ratio_predictions(
        w_hat_raw,
        clip_nonneg=clip_nonneg,
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        normalize_predictions=normalize_predictions,
        eps=prediction_norm_eps,
    )
    w_hat, calibration = _calibrate_action_predictions(
        w_hat,
        moment_calibration=moment_calibration,
        eps=prediction_norm_eps,
    )
    w_summary = _ratio_prediction_summary(
        w_hat,
        clipped_fraction=float(w_summary.get("clipped_fraction", 0.0)),
        normalization_scale=float(w_summary.get("normalization_scale", 1.0)),
    )

    return dict(
        bst_iw=bst_iw,
        best_iteration=best_iteration,
        w_hat=w_hat,
        w_hat_raw=w_hat_raw,
        w_hat_summary=w_summary,
        Xw_beh=X_sa,
        X_sa=X_sa,
        X_sa_pi=X_sa_pi,
        evals_result=evals_result,
        prediction_offset=float(init_score_value),
        prediction_max=None if prediction_max is None else float(prediction_max),
        prediction_power=float(prediction_power),
        normalize_predictions=bool(normalize_predictions),
        prediction_scale=float(calibration["scale"]),
        moment_calibration=str(moment_calibration),
        calibration=calibration,
        crossfit_folds=int(crossfit_folds),
        crossfit_seed=None if crossfit_seed is None else int(crossfit_seed),
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=None if logistic_logit_clip is None else float(logistic_logit_clip),
        prior_correction=float(prior_correction),
    )


def make_importance_ratio_long_arrays_from_sa(X_sa_beh: Array, X_sa_pi: Array) -> tuple[Array, Array, Array]:
    """Build the long LSIF dataset for ``pi(a | s) / pi0(a | s)``."""
    X_sa_beh = np.asarray(X_sa_beh, dtype=np.float32)
    X_sa_pi = np.asarray(X_sa_pi, dtype=np.float32)
    if X_sa_beh.ndim != 2 or X_sa_pi.ndim != 2:
        raise ValueError("X_sa_beh and X_sa_pi must be 2D arrays.")
    if X_sa_beh.shape[1] != X_sa_pi.shape[1]:
        raise ValueError("X_sa_beh and X_sa_pi must have the same number of columns.")

    n_beh = X_sa_beh.shape[0]
    n_pi = X_sa_pi.shape[0]
    X_long = np.vstack([X_sa_beh, X_sa_pi])
    y_long = np.concatenate(
        [
            np.ones(n_beh, dtype=np.int32),
            np.zeros(n_pi, dtype=np.int32),
        ]
    )
    weights = np.ones_like(y_long, dtype=np.float64)
    return X_long, y_long, weights


def make_importance_ratio_binary_arrays_from_sa(X_sa_beh: Array, X_sa_pi: Array) -> tuple[Array, Array, Array]:
    """Build a binary classification dataset for ``pi(a | s) / pi0(a | s)``.

    Label one is the numerator distribution and label zero is the behavior
    denominator. Converting classifier odds back to a density ratio requires the
    weighted class-prior correction computed by ``_binary_prior_correction``.
    """
    X_sa_beh = np.asarray(X_sa_beh, dtype=np.float32)
    X_sa_pi = np.asarray(X_sa_pi, dtype=np.float32)
    if X_sa_beh.ndim != 2 or X_sa_pi.ndim != 2:
        raise ValueError("X_sa_beh and X_sa_pi must be 2D arrays.")
    if X_sa_beh.shape[1] != X_sa_pi.shape[1]:
        raise ValueError("X_sa_beh and X_sa_pi must have the same number of columns.")

    n_beh = X_sa_beh.shape[0]
    n_pi = X_sa_pi.shape[0]
    X_long = np.vstack([X_sa_beh, X_sa_pi])
    y_long = np.concatenate(
        [
            np.zeros(n_beh, dtype=np.int32),
            np.ones(n_pi, dtype=np.int32),
        ]
    )
    weights = np.ones_like(y_long, dtype=np.float64)
    return X_long, y_long, weights


def make_lgb_importance_ratio_dataset(X_long: Array, y_long: Array, w_long: Optional[Array] = None) -> lgb.Dataset:
    return lgb.Dataset(X_long, label=y_long, weight=w_long, free_raw_data=False)


def importance_ratio_objective(y_pred: Array, dataset: lgb.Dataset, eps: float = 1e-3) -> tuple[Array, Array]:
    """Gradient and diagonal Hessian for ``E_b[w^2] - 2 E_pi[w]``."""
    y = dataset.get_label().astype(np.int32)
    sample_weight = _dataset_weight(dataset, y_pred)
    is_beh = y == 1
    is_pi = ~is_beh

    grad = np.zeros_like(y_pred, dtype=np.float64)
    hess = np.zeros_like(y_pred, dtype=np.float64)
    grad[is_beh] = 2.0 * y_pred[is_beh]
    hess[is_beh] = 2.0
    grad[is_pi] = -2.0
    hess[is_pi] = 2.0 * eps
    grad *= sample_weight
    hess *= sample_weight
    return grad, hess


def importance_ratio_eval(y_pred: Array, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """Evaluation metric matching the action-ratio LSIF objective."""
    y = dataset.get_label().astype(np.int32)
    sample_weight = _dataset_weight(dataset, y_pred)
    is_beh = y == 1
    is_pi = ~is_beh
    loss = np.sum(sample_weight[is_beh] * y_pred[is_beh] ** 2)
    loss -= 2.0 * np.sum(sample_weight[is_pi] * y_pred[is_pi])
    loss /= max(float(np.sum(sample_weight)), 1e-12)
    return "loss", float(loss), False


def make_transition_ratio_long_arrays(
    X_sa: Array,
    X_s_next: Array,
    K: int,
    seed: Optional[int] = None,
    X_s_ref: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """Build the long LSIF dataset for ``P(s' | s,a) / rho_ref(s')``."""
    if K <= 0:
        raise ValueError("K must be positive.")
    rng = np.random.default_rng(seed)
    X_sa = np.asarray(X_sa, dtype=np.float32)
    X_s_next = _as_2d_float32(X_s_next, "X_s_next")
    if X_sa.ndim != 2:
        raise ValueError("X_sa must be a 2D array.")
    n = X_sa.shape[0]
    if X_s_next.shape[0] != n:
        raise ValueError("X_sa and X_s_next must have the same number of rows.")

    X_s_ref = X_s_next if X_s_ref is None else _as_2d_float32(X_s_ref, "X_s_ref")
    if X_s_ref.shape[0] != n:
        raise ValueError("X_s_ref must have the same number of rows as X_sa.")

    X_ref = np.empty((n * K, X_s_ref.shape[1]), dtype=np.float32)
    for k in range(K):
        lo = k * n
        hi = lo + n
        X_ref[lo:hi, :] = X_s_ref[rng.permutation(n), :]

    X_s_long = np.vstack([X_s_next, X_ref])
    X_sa_long = np.tile(X_sa, (K + 1, 1))
    X_long = np.hstack([X_sa_long, X_s_long])
    y_long = np.concatenate(
        [
            np.ones(n, dtype=np.int32),
            np.zeros(n * K, dtype=np.int32),
        ]
    )
    weights = np.where(y_long == 1, 1.0, 1.0 / float(K)).astype(np.float64)
    return X_long, y_long, weights


def make_lgb_transition_ratio_dataset_from_arrays(
    X_long: Array,
    y_long: Array,
    w_long: Optional[Array] = None,
) -> lgb.Dataset:
    return lgb.Dataset(X_long, label=y_long, weight=w_long, free_raw_data=False)


def transition_ratio_objective(
    y_pred: Array,
    dataset: lgb.Dataset,
    eps: float = 1e-3,
    lam_norm: float = 1.0,
) -> tuple[Array, Array]:
    """Gradient and diagonal Hessian for transition-ratio LSIF.

    The optimized loss is the sum-scaled version of

        E_ref[k^2] - 2 E_transition[k] + lam_norm * (E_ref[k] - 1)^2.

    Sum scaling keeps tree split gradients numerically large while preserving
    the same minimizer as the expectation-scaled loss.
    """
    y = dataset.get_label().astype(np.int32)
    sample_weight = _dataset_weight(dataset, y_pred)
    y_pred = y_pred.astype(np.float64, copy=False)
    is_beh = y == 1
    is_ref = ~is_beh

    grad = np.zeros_like(y_pred, dtype=np.float64)
    hess = np.zeros_like(y_pred, dtype=np.float64)

    w_ref = sample_weight[is_ref]
    w_beh = sample_weight[is_beh]
    z_ref = float(np.sum(w_ref))

    k_ref = y_pred[is_ref]
    grad[is_ref] += 2.0 * w_ref * k_ref
    hess[is_ref] += 2.0 * w_ref

    grad[is_beh] -= 2.0 * w_beh
    hess[is_beh] += eps * w_beh

    if z_ref > 0.0:
        mean_ref = float(np.sum(w_ref * k_ref) / z_ref)
        delta = mean_ref - 1.0
        grad[is_ref] += 2.0 * lam_norm * delta * w_ref
        hess[is_ref] += 2.0 * lam_norm * (w_ref**2) / z_ref
    else:
        hess += eps

    return grad, hess


def transition_ratio_eval(
    y_pred: Array,
    dataset: lgb.Dataset,
    lam_norm: float = 1.0,
) -> tuple[str, float, bool]:
    """Evaluation metric for the transition-ratio LSIF objective."""
    y = dataset.get_label().astype(np.int32)
    sample_weight = _dataset_weight(dataset, y_pred)
    y_pred = y_pred.astype(np.float64, copy=False)
    is_beh = y == 1
    is_ref = ~is_beh

    w_ref = sample_weight[is_ref]
    w_beh = sample_weight[is_beh]
    z_ref = float(np.sum(w_ref))
    z_beh = float(np.sum(w_beh))
    if z_ref <= 0.0 or z_beh <= 0.0:
        return "loss", float("nan"), False

    k_ref = y_pred[is_ref]
    k_beh = y_pred[is_beh]
    e_ref_k2 = float(np.sum(w_ref * (k_ref**2)) / z_ref)
    e_beh_k = float(np.sum(w_beh * k_beh) / z_beh)
    e_ref_k = float(np.sum(w_ref * k_ref) / z_ref)
    loss = e_ref_k2 - 2.0 * e_beh_k + lam_norm * (e_ref_k - 1.0) ** 2
    return "loss", float(loss), False


def extract_num_boost_round(lgb_params: Dict[str, Any], default: int) -> int:
    """Pop common boosting-round aliases out of a LightGBM parameter dict."""
    for key in ("num_boost_round", "num_iterations", "num_iteration", "n_estimators"):
        if key in lgb_params and lgb_params[key] is not None:
            return int(lgb_params.pop(key))
    return int(default)


def _validate_prediction_postprocess(
    *,
    prediction_max: Optional[float],
    prediction_power: float,
    prediction_norm_eps: float,
) -> None:
    if prediction_max is not None and prediction_max <= 0.0:
        raise ValueError("prediction_max must be positive when supplied.")
    if not (0.0 < float(prediction_power) <= 1.0):
        raise ValueError("prediction_power must be in (0, 1].")
    if prediction_norm_eps <= 0.0:
        raise ValueError("prediction_norm_eps must be positive.")


def _validate_extra_nuisance_options(*, moment_calibration: str, crossfit_folds: int) -> None:
    if str(moment_calibration) not in {"none", "scalar"}:
        raise ValueError("moment_calibration must be 'none' or 'scalar'.")
    if int(crossfit_folds) < 1:
        raise ValueError("crossfit_folds must be >= 1.")


def _postprocess_ratio_predictions(
    values: Array,
    *,
    clip_nonneg: bool = True,
    prediction_max: Optional[float] = None,
    prediction_power: float = 1.0,
    normalize_predictions: bool = False,
    eps: float = 1e-12,
) -> tuple[Array, Dict[str, float]]:
    """Sanitize optional nuisance-ratio prediction stabilization.

    Tempering is a finite-sample guardrail for products inside the occupancy
    fixed point. Defaults on the low-level fit functions leave historical
    predictions unchanged except for requested nonnegative clipping.
    """
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_pos = float(prediction_max) if prediction_max is not None else np.finfo(np.float64).max / 16.0
    finite = np.nan_to_num(raw, nan=0.0, posinf=finite_pos, neginf=0.0)
    processed = finite.copy()
    if clip_nonneg or float(prediction_power) < 1.0:
        np.maximum(processed, 0.0, out=processed)

    clipped_fraction = 0.0
    if prediction_max is not None:
        cap = float(prediction_max)
        clipped_fraction = float(np.mean(processed > cap)) if processed.size else 0.0
        np.minimum(processed, cap, out=processed)

    if float(prediction_power) != 1.0:
        processed = np.power(np.maximum(processed, 0.0), float(prediction_power))

    normalization_scale = 1.0
    if normalize_predictions:
        mean = float(np.mean(processed)) if processed.size else 0.0
        if np.isfinite(mean) and mean > eps:
            normalization_scale = mean
            processed = processed / mean

    summary = _ratio_prediction_summary(
        processed,
        clipped_fraction=clipped_fraction,
        normalization_scale=normalization_scale,
    )
    return processed.astype(np.float64, copy=False), summary


def _calibrate_action_predictions(
    predictions: Array,
    *,
    moment_calibration: str,
    eps: float,
) -> tuple[Array, Dict[str, float | str | bool]]:
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    pre_mean = float(np.mean(values)) if values.size else 0.0
    scale = 1.0
    applied = False
    if str(moment_calibration) == "scalar" and np.isfinite(pre_mean) and pre_mean > eps:
        scale = 1.0 / pre_mean
        values = values * scale
        applied = True
    post_mean = float(np.mean(values)) if values.size else float("nan")
    return values, dict(
        method=str(moment_calibration),
        applied=bool(applied),
        scale=float(scale),
        pre_mean=float(pre_mean),
        post_mean=float(post_mean),
    )


def _calibrate_transition_predictions(
    *,
    booster: lgb.Booster,
    X_sa: Array,
    S_ref: Array,
    predictions: Array,
    offset: float,
    K_perm: int,
    seed: int,
    clip_nonneg: bool,
    prediction_max: Optional[float],
    prediction_power: float,
    normalize_predictions: bool,
    eps: float,
    moment_calibration: str,
    density_ratio_loss: str,
    logistic_logit_clip: Optional[float],
    prior_correction: float,
) -> tuple[Array, Dict[str, float | str | bool]]:
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    pre_mean = _transition_reference_mean(
        booster=booster,
        X_sa=X_sa,
        S_ref=S_ref,
        offset=offset,
        K_perm=K_perm,
        seed=seed,
        clip_nonneg=clip_nonneg,
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        normalize_predictions=normalize_predictions,
        eps=eps,
        scale=1.0,
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=logistic_logit_clip,
        prior_correction=prior_correction,
    )
    scale = 1.0
    applied = False
    if str(moment_calibration) == "scalar" and np.isfinite(pre_mean) and pre_mean > eps:
        scale = 1.0 / pre_mean
        values = values * scale
        applied = True
    post_mean = pre_mean * scale if np.isfinite(pre_mean) else float("nan")
    return values, dict(
        method=str(moment_calibration),
        applied=bool(applied),
        scale=float(scale),
        pre_mean=float(pre_mean),
        post_mean=float(post_mean),
    )


def _transition_reference_mean(
    *,
    booster: lgb.Booster,
    X_sa: Array,
    S_ref: Array,
    offset: float,
    K_perm: int,
    seed: int,
    clip_nonneg: bool,
    prediction_max: Optional[float],
    prediction_power: float,
    normalize_predictions: bool,
    eps: float,
    scale: float,
    density_ratio_loss: str,
    logistic_logit_clip: Optional[float],
    prior_correction: float,
) -> float:
    X_long, _, _ = make_transition_ratio_long_arrays(
        X_sa=X_sa,
        X_s_next=S_ref,
        X_s_ref=S_ref,
        K=max(1, int(K_perm)),
        seed=seed,
    )
    ref_block = X_long[X_sa.shape[0] :, :]
    raw = _predict_ratio_from_booster(
        booster=booster,
        X=ref_block,
        offset=offset,
        density_ratio_loss=density_ratio_loss,
        logistic_logit_clip=logistic_logit_clip,
        prior_correction=prior_correction,
    )
    processed, _ = _postprocess_ratio_predictions(
        raw,
        clip_nonneg=clip_nonneg,
        prediction_max=prediction_max,
        prediction_power=prediction_power,
        normalize_predictions=normalize_predictions,
        eps=eps,
    )
    processed = processed * float(scale)
    return float(np.mean(processed)) if processed.size else float("nan")


def _validate_density_ratio_loss(loss: str) -> str:
    normalized = str(loss).strip().lower()
    if normalized not in {"lsif", "logistic"}:
        raise ValueError("density_ratio_loss must be 'lsif' or 'logistic'.")
    return normalized


def _validate_logistic_options(logistic_logit_clip: Optional[float]) -> None:
    if logistic_logit_clip is not None and float(logistic_logit_clip) <= 0.0:
        raise ValueError("logistic_logit_clip must be positive when supplied.")


def _binary_ratio_lgb_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out["objective"] = "binary"
    out.setdefault("metric", "binary_logloss")
    return out


def _binary_prior_correction(y: Array, weight: Optional[Array]) -> float:
    labels = np.asarray(y, dtype=np.int32).reshape(-1)
    weights = np.ones(labels.shape[0], dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    numerator_weight = float(np.sum(weights[labels == 1]))
    denominator_weight = float(np.sum(weights[labels == 0]))
    if numerator_weight <= 0.0 or denominator_weight <= 0.0:
        return 1.0
    return denominator_weight / numerator_weight


def _predict_ratio_from_booster(
    *,
    booster: lgb.Booster,
    X: Array,
    offset: float,
    density_ratio_loss: str,
    logistic_logit_clip: Optional[float],
    prior_correction: float,
    num_iteration: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> Array:
    predict_kwargs: Dict[str, Any] = {}
    if num_iteration is not None:
        predict_kwargs["num_iteration"] = int(num_iteration)
    if num_threads is not None:
        predict_kwargs["num_threads"] = int(num_threads)
    if str(density_ratio_loss) == "logistic":
        logits = booster.predict(X, raw_score=True, **predict_kwargs).astype(np.float64, copy=False)
        if logistic_logit_clip is not None:
            clip = float(logistic_logit_clip)
            logits = np.clip(logits, -clip, clip)
        return np.exp(logits) * float(prior_correction)
    return float(offset) + booster.predict(X, **predict_kwargs).astype(np.float64, copy=False)


def _ratio_prediction_summary(
    values: Array,
    *,
    clipped_fraction: float,
    normalization_scale: float,
) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(
            min=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            clipped_fraction=float(clipped_fraction),
            normalization_scale=float(normalization_scale),
        )
    return dict(
        min=float(np.min(x)),
        p50=float(np.quantile(x, 0.50)),
        p90=float(np.quantile(x, 0.90)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        max=float(np.max(x)),
        mean=float(np.mean(x)),
        clipped_fraction=float(clipped_fraction),
        normalization_scale=float(normalization_scale),
    )


def _as_2d_float32(x: Array, name: str) -> Array:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must be 1D or 2D.")


def _validate_same_rows(**arrays: Array) -> None:
    lengths = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"All inputs must have the same number of rows ({details}).")


def _validate_test_size(test_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")


def _split_rows(n_rows: int, *, test_size: float, seed: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_valid = max(1, int(np.floor(test_size * n_rows)))
    valid_idx = perm[:n_valid]
    train_idx = perm[n_valid:]
    if train_idx.size == 0:
        raise ValueError("test_size too large: no training rows left.")
    return train_idx, valid_idx


def _transition_lgb_params(lgb_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = {} if lgb_params is None else dict(lgb_params)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("num_leaves", 64)
    params.setdefault("min_data_in_leaf", 200)
    params.setdefault("feature_fraction", 1.0)
    params.setdefault("bagging_fraction", 1.0)
    params.setdefault("verbose", -1)
    params.setdefault("num_threads", 0)
    params["min_sum_hessian_in_leaf"] = 0
    params["lambda_l2"] = 0
    return params


def _importance_lgb_params(lgb_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = _transition_lgb_params(lgb_params)
    params.setdefault("boost_from_average", False)
    return params


def _set_constant_init_score(dataset: lgb.Dataset, n_rows: int, value: float) -> None:
    dataset.set_init_score(np.full(n_rows, float(value), dtype=np.float64))


def _make_callbacks(
    *,
    total: int,
    desc: str,
    show_tqdm: bool,
    tqdm_leave: bool,
    evals_result: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 0,
    early_stopping_min_delta: float = 0.0,
) -> Dict[str, Any]:
    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=int(early_stopping_rounds),
                first_metric_only=True,
                min_delta=float(early_stopping_min_delta),
                verbose=False,
            )
        )
    if evals_result is not None:
        callbacks.append(lgb.record_evaluation(evals_result))
    if show_tqdm:
        callbacks.append(_make_lgb_tqdm_callback(total=total, desc=desc, leave=tqdm_leave))
    return {"callbacks": callbacks, "evals_result": evals_result if evals_result is not None else {}}


def _best_iteration(booster: lgb.Booster, *, fallback: int) -> int:
    best = int(getattr(booster, "best_iteration", 0) or 0)
    return best if best > 0 else int(fallback)


def _dataset_weight(dataset: lgb.Dataset, y_pred: Array) -> Array:
    weight = dataset.get_weight()
    if weight is None:
        return np.ones_like(y_pred, dtype=np.float64)
    return weight.astype(np.float64, copy=False)


def _make_lgb_tqdm_callback(
    *,
    total: int,
    desc: str,
    leave: bool = True,
    postfix_every: int = 1,
    metric_substr: str = "loss",
    ncols: int = 160,
):
    pbar = tqdm(
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=False,
        ncols=ncols,
        file=sys.stderr,
        mininterval=0.1,
        maxinterval=1.0,
    )
    last_i = 0

    def callback(env):
        nonlocal last_i
        i = env.iteration + 1
        delta = i - last_i
        if delta > 0:
            pbar.update(delta)
            last_i = i

        if env.evaluation_result_list and (
            postfix_every <= 1 or i % postfix_every == 0 or env.iteration == env.end_iteration
        ):
            for _, eval_name, result, _ in env.evaluation_result_list:
                if metric_substr in str(eval_name):
                    pbar.set_postfix_str(f"{eval_name}={result:.4g}")
                    break

        if env.iteration == env.end_iteration:
            pbar.close()

    callback.order = 999
    callback.before_iteration = False
    return callback
