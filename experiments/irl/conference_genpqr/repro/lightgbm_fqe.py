"""Self-contained LightGBM boosted FQE adapted from the repo's FQE module."""

from __future__ import annotations

import contextlib
import copy
import os
import sys
from typing import Dict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb
import numpy as np


@contextlib.contextmanager
def suppress_lightgbm_output(verbose: bool = False):
    """Mute LightGBM's native stdout/stderr chatter when desired."""
    if verbose:
        yield
        return
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def create_td_dataset(
    S,
    S_next,
    Y,
    discount_factor,
    model=None,
    V_next=None,
    is_terminal_outcome=None,
    weights=None,
    categorical_feature=None,
    init_score=None,
):
    """Construct a LightGBM dataset for TD regression."""
    S = np.asarray(S).reshape(len(Y), -1)
    S_next = np.asarray(S_next).reshape(len(Y), -1)
    Y = np.asarray(Y).reshape(-1)

    if V_next is None and model is not None:
        V_next = model.predict(S_next)
    if V_next is not None:
        V_next = np.asarray(V_next).reshape(-1)
        if is_terminal_outcome is not None:
            V_next = (1 - is_terminal_outcome) * V_next
        td_targets = Y + discount_factor * V_next
    else:
        td_targets = Y
    if init_score is not None:
        td_targets = td_targets - np.asarray(init_score).reshape(-1)
    return lgb.Dataset(
        S,
        label=td_targets,
        weight=weights,
        free_raw_data=False,
        categorical_feature=categorical_feature,
        init_score=None,
    )


def create_td_objective(Y, V_next, discount_factor, is_terminal_outcome, init_score=None):
    """Create the custom LightGBM TD objective."""
    if V_next is not None:
        V_next = np.asarray(V_next).reshape(-1)
        if is_terminal_outcome is not None:
            V_next = (1 - is_terminal_outcome) * V_next
        td_targets = Y + discount_factor * V_next
    else:
        td_targets = Y
    if init_score is not None:
        td_targets = td_targets - np.asarray(init_score).reshape(-1)

    def custom_obj(preds, train_data):
        weights = train_data.get_weight()
        if weights is None:
            weights = np.ones_like(preds)
        grad = weights * (preds - td_targets)
        hess = weights
        return grad, hess

    return custom_obj


def risk_fitted_value_iteration(V, V_next, Y, discount_factor, is_terminal_outcome=None, weights=None, init_score=None):
    """Compute Bellman-risk diagnostics."""
    if V is None and V_next is None:
        return np.inf
    weights = np.ones_like(Y) if weights is None else weights
    init_score = np.zeros_like(Y) if init_score is None else np.asarray(init_score).reshape(-1)
    V_next = np.zeros_like(V) if V_next is None else V_next
    if is_terminal_outcome is not None:
        V_next = (1 - is_terminal_outcome) * V_next
    targets = Y + discount_factor * V_next - init_score
    errors = targets - V
    return np.average(errors**2, weights=weights)


def _random_train_val_indices(n: int, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(test_size * n)))
    val_indices = np.sort(perm[:n_val])
    train_indices = np.sort(perm[n_val:])
    return train_indices, val_indices


def fit_fqe_boosted(
    S,
    A,
    S_next,
    A_next,
    rewards,
    discount_factor,
    is_terminal_outcome=None,
    weights=None,
    categorical_feature=None,
    lgb_params=None,
    test_size=0.2,
    train_indices=None,
    refit_on_all_data=True,
    fit_control=None,
    seed=42,
    verbose=False,
) -> Dict[str, object]:
    """LightGBM boosted FQE on state-action features."""
    S = np.asarray(S)
    A = np.asarray(A).reshape(len(rewards), -1)
    S_next = np.asarray(S_next)
    A_next = np.asarray(A_next).reshape(len(rewards), -1)
    rewards = np.asarray(rewards).reshape(-1)
    X = np.concatenate([S.reshape(len(rewards), -1), A], axis=1)
    X_next = np.concatenate([S_next.reshape(len(rewards), -1), A_next], axis=1)
    return fit_fvi_boosted(
        S=X,
        S_next=X_next,
        rewards=rewards,
        discount_factor=discount_factor,
        is_terminal_outcome=is_terminal_outcome,
        weights=weights,
        categorical_feature=categorical_feature,
        lgb_params=lgb_params,
        test_size=test_size,
        train_indices=train_indices,
        refit_on_all_data=refit_on_all_data,
        fit_control=fit_control,
        seed=seed,
        verbose=verbose,
    )


def fit_fvi_boosted(
    S,
    S_next,
    rewards,
    discount_factor,
    is_terminal_outcome=None,
    weights=None,
    categorical_feature=None,
    lgb_params=None,
    test_size=0.2,
    train_indices=None,
    refit_on_all_data=False,
    fit_control=None,
    seed=42,
    verbose=False,
):
    """Boosted FVI/FQE with one LightGBM tree added per outer iteration."""
    Y = np.asarray(rewards).reshape(-1)
    S = np.asarray(S).reshape(len(Y), -1)
    S_next = np.asarray(S_next).reshape(len(Y), -1)
    weights = np.ones_like(Y) if weights is None else np.asarray(weights).reshape(-1)
    is_terminal_outcome = np.zeros_like(Y) if is_terminal_outcome is None else np.asarray(is_terminal_outcome).reshape(-1)

    params = {
        "objective": "regression",
        "verbosity": -1,
        "bagging_fraction": 1.0,
        "bagging_freq": 1,
        "seed": seed,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_iterations": 1,
        "num_threads": 1,
        "num_leaves": 32,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_sum_hessian_in_leaf": 1e-3,
    }
    if lgb_params:
        params.update(lgb_params)
    learning_rate = params["learning_rate"]
    control = {
        "num_boost_rounds": 150,
        "early_stopping_rounds": 15,
        "early_stopping_min_delta": 1e-6,
    }
    if fit_control:
        control.update(fit_control)

    n = len(Y)
    if train_indices is None:
        train_indices, val_indices = _random_train_val_indices(n, test_size=test_size, seed=seed)
    else:
        train_indices = np.asarray(train_indices)
        val_indices = np.setdiff1d(np.arange(n), train_indices)

    S_train, S_val = S[train_indices], S[val_indices]
    S_next_train, S_next_val = S_next[train_indices], S_next[val_indices]
    Y_train, Y_val = Y[train_indices], Y[val_indices]
    w_train, w_val = weights[train_indices], weights[val_indices]
    d_train, d_val = is_terminal_outcome[train_indices], is_terminal_outcome[val_indices]

    model = None
    model_all = None
    V_train = np.zeros_like(Y_train, dtype=float)
    V_next_train = np.zeros_like(Y_train, dtype=float)
    V_val = np.zeros_like(Y_val, dtype=float)
    V_next_val = np.zeros_like(Y_val, dtype=float)
    V_next_all = np.zeros_like(Y, dtype=float)
    td_train = Y_train + discount_factor * (1 - d_train) * V_next_train
    td_all = Y + discount_factor * (1 - is_terminal_outcome) * V_next_all
    data_train = lgb.Dataset(
        S_train,
        label=td_train,
        weight=w_train,
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )
    data_all = lgb.Dataset(
        S,
        label=td_all,
        weight=weights,
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )

    best_risk = np.inf
    patience = 0
    boost_iteration = 0
    for iteration in range(control["num_boost_rounds"]):
        td_train = Y_train + discount_factor * (1 - d_train) * V_next_train
        data_train.set_label(td_train)
        params_iter = copy.deepcopy(params)
        with suppress_lightgbm_output(verbose=verbose):
            if model is None:
                model = lgb.train(params_iter, data_train, keep_training_booster=True)
            else:
                model.reset_parameter({"learning_rate": learning_rate})
                model.update(train_set=data_train)

        V_train_candidate = V_train + model.predict(S_train, start_iteration=boost_iteration)
        V_next_train_candidate = V_next_train + model.predict(S_next_train, start_iteration=boost_iteration)
        V_val_candidate = V_val + model.predict(S_val, start_iteration=boost_iteration)
        V_next_val_candidate = V_next_val + model.predict(S_next_val, start_iteration=boost_iteration)
        val_risk = risk_fitted_value_iteration(
            V_val_candidate,
            V_next_val,
            Y_val,
            discount_factor=discount_factor,
            is_terminal_outcome=d_val,
            weights=w_val,
        )

        if val_risk <= best_risk - control["early_stopping_min_delta"]:
            best_risk = val_risk
            patience = 0
            V_train = V_train_candidate
            V_next_train = V_next_train_candidate
            V_val = V_val_candidate
            V_next_val = V_next_val_candidate
            boost_iteration += 1
            if refit_on_all_data:
                td_all = Y + discount_factor * (1 - is_terminal_outcome) * V_next_all
                data_all.set_label(td_all)
                params_all = copy.deepcopy(params_iter)
                with suppress_lightgbm_output(verbose=verbose):
                    if model_all is None:
                        model_all = lgb.train(params_all, data_all, keep_training_booster=True)
                    else:
                        model_all.reset_parameter({"learning_rate": learning_rate})
                        model_all.update(train_set=data_all)
                V_next_all = V_next_all + model_all.predict(S_next, start_iteration=boost_iteration - 1)
        else:
            patience += 1
            if model is not None:
                model.rollback_one_iter()
        if verbose and (iteration + 1) % 10 == 0:
            print(f"[LightGBM FQE] iter={iteration + 1} val_risk={best_risk:.6f}")
        if patience >= control["early_stopping_rounds"]:
            break

    output_model = model_all if refit_on_all_data and model_all is not None else model
    return {"model": output_model, "model_train": model, "train_indices": train_indices}
