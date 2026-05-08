import itertools

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from FQE.fqe_boosted import *
import utils as utils


# assumes fit_fqe_boosted, risk_fitted_value_iteration are in scope
# from LongTermMarkov.fit_value_iteration_lightgbm import fit_fqe_boosted, risk_fitted_value_iteration


def tune_fqe(
    lgb_param_grid,
    S,
    A,
    S_next,
    A_next,
    rewards,
    discount_factor,
    is_terminal_outcome=None,
    weights=None,
    categorical_feature=None,
    init_model=None,
    init_score=None,
    test_size: float = 0.2,
    train_indices=None,
    fit_control: dict = None,
    n_jobs: int = -1,
    seed: int = 42,
    return_model: bool = False,
    refit_on_all_data: bool = True,
    boost: bool = True,
    verbose: bool = False,
):
    """
    Hyperparameter tuning for fitted Q-iteration (FQI) using LightGBM.

    We treat Q(s,a) as a regression function on concatenated (S,A) features.
    For each hyperparameter setting, we:
      - train Q via fit_fqe_boosted on an outer training subset, letting fit_fqe_boosted
        handle its own internal train/validation split & early stopping,
      - evaluate TD risk on a fixed outer validation subset,
      - pick the hyperparameters with the smallest TD risk.

    Parameters
    ----------
    lgb_param_grid : dict
        Grid of LightGBM hyperparameters to search over. Values can be scalars
        or lists; scalars will be wrapped into lists.

    S, A, S_next, A_next : array-like
        Transitions:
          (S_t, A_t, R_t, S_{t+1}, A'_{t+1})

    rewards : array-like, shape (n,)
        One-step rewards R_t.

    discount_factor : float
        Discount factor gamma in [0,1).

    is_terminal_outcome : array-like, shape (n,), optional
        Indicator of terminal transitions (0/1 or bool).

    weights : array-like, shape (n,), optional
        Sample weights.

    categorical_feature, init_model, init_score, fit_control, boost :
        Passed through to fit_fqe_boosted.

    test_size : float
        Fraction of the *overall dataset* used as an outer validation split.

    train_indices : array-like, optional
        Explicit indices for the outer training split. If None, a random
        split using `test_size` and `seed` is used.

    n_jobs : int
        Parallel jobs for the hyperparameter grid.

    seed : int
        Random seed (used for the outer split and passed into fit_fqe_boosted).

    return_model : bool
        If True, refit a final Q-function on the full dataset
        with the best hyperparameters (via fit_fqe_boosted).

    refit_on_all_data : bool
        Passed to the final call to fit_fqe_boosted when return_model=True.

    verbose : bool
        Print basic progress messages.

    Returns
    -------
    dict with keys:
        - "results": list of (params, td_risk) for all grid points
        - "best_params": dict of best hyperparameters
        - "best_score": float, smallest TD risk on outer validation
        - "final_model": output of fit_fqe_boosted on full data (or None if return_model=False)
    """
    if fit_control is None:
        fit_control = {}

    # Make sure every value in the grid is a list
    lgb_param_grid = utils.ensure_list_values(lgb_param_grid)
    param_combinations = [
        dict(zip(lgb_param_grid, v))
        for v in itertools.product(*lgb_param_grid.values())
    ]

    # ----- Outer train/validation split (fixed across hyperparams) -----
    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)
    A_next = np.asarray(A_next)
    rewards = np.asarray(rewards)

    n = rewards.shape[0]
    indices = np.arange(n)

    if train_indices is None:
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=seed
        )
    else:
        val_indices = np.setdiff1d(indices, train_indices)

    S_train, S_val = S[train_indices], S[val_indices]
    A_train, A_val = A[train_indices], A[val_indices]
    S_next_train, S_next_val = S_next[train_indices], S_next[val_indices]
    A_next_train, A_next_val = A_next[train_indices], A_next[val_indices]
    Y_train, Y_val = rewards[train_indices], rewards[val_indices]

    if weights is not None:
        weights_train, weights_val = weights[train_indices], weights[val_indices]
    else:
        weights_train, weights_val = None, None

    if init_score is not None:
        init_score_train, init_score_val = init_score[train_indices], init_score[val_indices]
    else:
        init_score_train, init_score_val = None, None

    if is_terminal_outcome is not None:
        is_term_train = is_terminal_outcome[train_indices]
        is_term_val = is_terminal_outcome[val_indices]
    else:
        is_term_train = None
        is_term_val = None

    data_dict_train = {
        "S": S_train,
        "A": A_train,
        "S_next": S_next_train,
        "A_next": A_next_train,
        "Y": Y_train,
        "weights": weights_train,
        "init_score": init_score_train,
        "is_terminal_outcome": is_term_train,
    }

    data_dict_val = {
        "S": S_val,
        "A": A_val,
        "S_next": S_next_val,
        "A_next": A_next_val,
        "Y": Y_val,
        "weights": weights_val,
        "init_score": init_score_val,
        "is_terminal_outcome": is_term_val,
    }

    # ----- Per-parameter evaluation -----
    def evaluate_params(params, data_dict_train, data_dict_val):
        # Fit Q on outer training subset; fit_fqe_boosted handles its *internal* split
        model_fit = fit_fqe_boosted(
            S=data_dict_train["S"],
            A=data_dict_train["A"],
            S_next=data_dict_train["S_next"],
            A_next=data_dict_train["A_next"],
            rewards=data_dict_train["Y"],
            discount_factor=discount_factor,
            is_terminal_outcome=data_dict_train["is_terminal_outcome"],
            weights=data_dict_train["weights"],
            categorical_feature=categorical_feature,
            lgb_params=params,
            init_model=init_model,
            init_score=data_dict_train["init_score"],
            test_size=test_size,
            train_indices=None,
            data_dict_test=None,
            refit_on_all_data=False,
            boost=boost,
            fit_control=fit_control,
            seed=seed,
            verbose=False,
        )

        booster = model_fit["model"]

        # Build concatenated (S,A) and (S',A') for the *outer* validation set
        S_val = data_dict_val["S"]
        A_val = data_dict_val["A"]
        S_next_val = data_dict_val["S_next"]
        A_next_val = data_dict_val["A_next"]
        Y_val = data_dict_val["Y"]

        n_val = Y_val.shape[0]
        S_val_flat = S_val.reshape(n_val, -1)
        A_val_flat = A_val.reshape(n_val, -1)
        X_val = np.concatenate([S_val_flat, A_val_flat], axis=1)

        S_next_val_flat = S_next_val.reshape(n_val, -1)
        A_next_val_flat = A_next_val.reshape(n_val, -1)
        X_next_val = np.concatenate([S_next_val_flat, A_next_val_flat], axis=1)

        Q_val = booster.predict(X_val)
        Q_next_val = booster.predict(X_next_val)

        td_risk = risk_fitted_value_iteration(
            V=Q_val,
            V_next=Q_next_val,
            Y=Y_val,
            discount_factor=discount_factor,
            is_terminal_outcome=data_dict_val["is_terminal_outcome"],
            weights=data_dict_val["weights"],
            init_score=data_dict_val["init_score"],
        )

        return params, td_risk

    # ----- Run grid search in parallel -----
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params, data_dict_train, data_dict_val)
        for params in tqdm(param_combinations, disable=not verbose)
    )

    # ----- Pick best hyperparameters -----
    best_params, best_score = min(results, key=lambda x: x[1])
    if verbose:
        print("Best parameters found: ", best_params)
        print("Best TD risk: ", best_score)

    # ----- Optional final model on full data -----
    if return_model:
        final_model = fit_fqe_boosted(
            S=S,
            A=A,
            S_next=S_next,
            A_next=A_next,
            rewards=rewards,
            discount_factor=discount_factor,
            is_terminal_outcome=is_terminal_outcome,
            weights=weights,
            categorical_feature=categorical_feature,
            lgb_params=best_params,
            init_model=init_model,
            init_score=init_score,
            test_size=test_size,
            train_indices=train_indices,
            data_dict_test=None,
            refit_on_all_data=refit_on_all_data,
            boost=boost,
            fit_control=fit_control,
            seed=seed,
            verbose=False,
        )
    else:
        final_model = None

    return {
        "results": results,
        "best_params": best_params,
        "best_score": best_score,
        "final_model": final_model,
    }

 
 