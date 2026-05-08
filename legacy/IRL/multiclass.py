import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def fit_multiclass_lgbm(
    X,
    A,
    weights=None,
    categorical_feature=None,
    lgb_params=None,
    test_size: float = 0.2,
    train_indices=None,
    data_dict_test=None,
    refit_on_all_data: bool = False,
    seed: int = 123,
    verbose: bool = False,
):
    """
    Fit a LightGBM multiclass model to learn p(a | x), with early stopping
    on a held-out validation set (implemented via callbacks).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    A : array-like, shape (n_samples,)
        Integer action labels (0, 1, ..., K-1).
    weights : array-like, optional
        Sample weights. If None, uniform weights are used.
    categorical_feature : list or list of str/int, optional
        Categorical feature indices or names passed to LightGBM.
    lgb_params : dict, optional
        Additional LightGBM parameters to override defaults.
    test_size : float, optional
        Fraction of data to use as validation set when `train_indices` is None.
    train_indices : array-like, optional
        Predefined training indices. Validation indices are taken as the complement.
    data_dict_test : dict, optional
        If provided, uses this as the validation data:
            {"X": X_val, "A": A_val, "weights": w_val}
    refit_on_all_data : bool, optional
        If True, refit a final model on all data using best_iteration.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, print LightGBM validation logs.

    Returns
    -------
    model : lgb.Booster
        Trained LightGBM multiclass model (predicts class probabilities).
    """

    # -----------------------------
    # Basic setup
    # -----------------------------
    X = np.asarray(X)
    A = np.asarray(A).reshape(-1)
    n = X.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    unique_actions = np.unique(A)
    num_classes = unique_actions.size

    # -----------------------------
    # Default LightGBM params
    # -----------------------------
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_iterations": 2000,   # acts as an upper bound; early stopping will cut it
        "num_leaves": 40,
        "min_data_in_leaf": 50,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_sum_hessian_in_leaf": 1e-3,
        "seed": seed,
        "num_threads": 1,
        "verbosity": -1,
    }

    if lgb_params is not None:
        params.update(lgb_params)

    # Early stopping rounds (we'll use it via callback)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    # -----------------------------
    # Train / validation split
    # -----------------------------
    if data_dict_test is not None:
        X_train, X_val = X, np.asarray(data_dict_test["X"])
        A_train, A_val = A, np.asarray(data_dict_test["A"])
        w_train = weights
        w_val = np.asarray(data_dict_test.get("weights", np.ones_like(A_val, dtype=float)))
    else:
        indices = np.arange(n)
        if train_indices is None:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
            )
        else:
            train_indices = np.asarray(train_indices)
            val_indices = np.setdiff1d(indices, train_indices)

        X_train, X_val = X[train_indices], X[val_indices]
        A_train, A_val = A[train_indices], A[val_indices]
        w_train, w_val = weights[train_indices], weights[val_indices]

    # -----------------------------
    # Build LightGBM datasets
    # -----------------------------
    dtrain = lgb.Dataset(
        X_train,
        label=A_train,
        weight=w_train,
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )

    dval = lgb.Dataset(
        X_val,
        label=A_val,
        weight=w_val,
        reference=dtrain,
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )

    # -----------------------------
    # Callbacks for early stopping + logging
    # -----------------------------
    callbacks = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        )
    # You can also add log_evaluation if you want pretty logging
    if verbose:
        callbacks.append(lgb.log_evaluation(period=50))

    # -----------------------------
    # Train with early stopping via callbacks
    # -----------------------------
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    # -----------------------------
    # Optional refit on all data
    # -----------------------------
    if refit_on_all_data:
        best_iter = model.best_iteration or params.get("num_iterations", 2000)
        dtrain_all = lgb.Dataset(
            X,
            label=A,
            weight=weights,
            free_raw_data=False,
            categorical_feature=categorical_feature,
        )
        # No early stopping when refitting
        refit_params = params.copy()
        model = lgb.train(
            refit_params,
            dtrain_all,
            num_boost_round=best_iter,
        )

    return model
