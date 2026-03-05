import numpy as np
import lightgbm as lgb
 



def fit_transition_ratio_lgbm(
    *,
    S: np.ndarray,          # (N, d_s) or (N,)   -- initial state features (t=0)
    A: np.ndarray,          # (N, d_a) or (N,)   -- behavior action features (t=0)
    S_next: np.ndarray,     # (N, d_s) or (N,)   -- next-state features (t=1)
    K_perm: int = 20,
    seed: int = 123,
    clip_nonneg: bool = True,
    num_boost_round: int = 300,
    lgb_params: dict | None = None,
    eps_hess: float = 0,  # kept for signature compatibility (unused here)
    test_size: float = 0.2,
    early_stopping_rounds: int = 10,
    refit_on_all_data: bool = True,
    show_tqdm: bool = True,
    tqdm_leave: bool = True,
    tqdm_desc_es: str = "boosting transition kernel (early stopping)",
    tqdm_desc_refit: str = "boosting transition kernel (refit)",
    early_stopping_min_delta: float = 0.0,
    init_score_value: float = 0,
):
    """
    Fit ktilde(s,a,s') = P(s'|s,a) / d0(s') via LSIF-style squared-loss.

    IMPORTANT: reference (denominator) samples for s' are drawn from S (initial states),
    not from S_next. This avoids the extra rho_{b,next}/d0 factor later.
    """
    import numpy as np
    import lightgbm as lgb

    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)

    # ---- coerce to 2D feature matrices ----
    if S.ndim == 1:
        S_feat = S.astype(np.float32).reshape(-1, 1)
    elif S.ndim == 2:
        S_feat = S.astype(np.float32)
    else:
        raise ValueError("S must be 1D or 2D.")

    if A.ndim == 1:
        A_feat = A.astype(np.float32).reshape(-1, 1)
    elif A.ndim == 2:
        A_feat = A.astype(np.float32)
    else:
        raise ValueError("A must be 1D or 2D.")

    if S_next.ndim == 1:
        S_next_feat = S_next.astype(np.float32).reshape(-1, 1)
    elif S_next.ndim == 2:
        S_next_feat = S_next.astype(np.float32)
    else:
        raise ValueError("S_next must be 1D or 2D.")

    N = S_feat.shape[0]
    if A_feat.shape[0] != N or S_next_feat.shape[0] != N:
        raise ValueError("S, A, S_next must have the same number of rows (N).")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1).")

    # ---- build (s,a) features ----
    X_sa = np.concatenate([S_feat, A_feat], axis=1)

    # ---- split on original transitions ----
    rng = np.random.default_rng(seed)
    permN = rng.permutation(N)
    n_validN = int(np.floor(test_size * N))
    n_validN = max(1, n_validN)
    validN = permN[:n_validN]
    trainN = permN[n_validN:]
    if trainN.size == 0:
        raise ValueError("test_size too large: no training rows left.")

    # ---- build long data using ktilde reference = S (NOT S_next) ----
    X_train, y_train, w_train = make_transition_ratio_long_arrays(
        X_sa=X_sa[trainN],
        X_s_next=S_next_feat[trainN],   # behavior block uses true next states
        X_s_ref=S_feat[trainN],         # reference block uses initial states (d0)
        K=K_perm,
        seed=seed,
    )
    X_valid, y_valid, w_valid = make_transition_ratio_long_arrays(
        X_sa=X_sa[validN],
        X_s_next=S_next_feat[validN],
        X_s_ref=S_feat[validN],
        K=K_perm,
        seed=seed + 1,
    )

    dtrain = make_lgb_transition_ratio_dataset_from_arrays(
        X_train, y_train, None if w_train is None else w_train
    )
    dvalid = make_lgb_transition_ratio_dataset_from_arrays(
        X_valid, y_valid, None if w_valid is None else w_valid
    )

    # ---- init_score ----
    dtrain.set_init_score(np.full(X_train.shape[0], float(init_score_value), dtype=np.float64))
    dvalid.set_init_score(np.full(X_valid.shape[0], float(init_score_value), dtype=np.float64))

    # ---- params ----
    if lgb_params is None:
        lgb_params = {}
    else:
        lgb_params = dict(lgb_params)

    lgb_params.setdefault("learning_rate", 0.05)
    lgb_params.setdefault("num_leaves", 64)
    lgb_params.setdefault("min_data_in_leaf", 200)
    lgb_params.setdefault("feature_fraction", 1.0)
    lgb_params.setdefault("bagging_fraction", 1.0)
    lgb_params.setdefault("verbose", -1)
    lgb_params.setdefault("num_threads", 0)
    lgb_params["min_sum_hessian_in_leaf"] = 0
    lgb_params["lambda_l2"] = 0

    num_boost_round = extract_num_boost_round(
        lgb_params=lgb_params,
        default=num_boost_round,
    )

    # custom objective (no fobj)
    lgb_params["objective"] = transition_ratio_objective

    def _mk_callbacks(*, total: int, desc: str, evals_result: dict | None = None, early_stop: bool = False):
        cbs = []
        if early_stop:
            cbs.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    first_metric_only=True,
                    min_delta=float(early_stopping_min_delta),
                    verbose=False,
                )
            )
        if evals_result is not None:
            cbs.append(lgb.record_evaluation(evals_result))
        if show_tqdm:
            cbs.append(_make_lgb_tqdm_callback(total=total, desc=desc, leave=tqdm_leave))
        return cbs

    # ---- phase 1: early stopping ----
    evals_result = {}
    bst_es = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=transition_ratio_eval,
        callbacks=_mk_callbacks(
            total=num_boost_round,
            desc=tqdm_desc_es,
            evals_result=evals_result,
            early_stop=True,
        ),
    )

    best_iteration = int(getattr(bst_es, "best_iteration", 0) or 0)
    if best_iteration <= 0:
        best_iteration = int(num_boost_round)

    # ---- phase 2: refit on all data ----
    if refit_on_all_data:
        X_long, y_long, w_long = make_transition_ratio_long_arrays(
            X_sa=X_sa,
            X_s_next=S_next_feat,
            X_s_ref=S_feat,   # key: reference = initial states
            K=K_perm,
            seed=seed,
        )
        dall = make_lgb_transition_ratio_dataset_from_arrays(
            X_long, y_long, None if w_long is None else w_long
        )
        dall.set_init_score(np.full(X_long.shape[0], float(init_score_value), dtype=np.float64))

        bst_k = lgb.train(
            params=lgb_params,
            train_set=dall,
            num_boost_round=best_iteration,
            valid_sets=[dall],
            valid_names=["train"],
            feval=transition_ratio_eval,
            callbacks=_mk_callbacks(total=best_iteration, desc=tqdm_desc_refit),
        )
    else:
        bst_k = bst_es

    # ---- predictions on observed (s,a,s') ----
    Xk_beh = np.hstack([X_sa, S_next_feat])
    k_hat = bst_k.predict(
        Xk_beh,
        num_iteration=best_iteration if refit_on_all_data else bst_k.best_iteration,
    )
    if clip_nonneg:
        k_hat = np.maximum(k_hat, 0.0)

    return dict(
        bst_k=bst_k,
        best_iteration=best_iteration,
        k_hat=k_hat,
        Xk_beh=Xk_beh,
        X_sa=X_sa,
        S_feat=S_feat,
        S_next_feat=S_next_feat,
        evals_result=evals_result,
    )


def fit_importance_ratio_lgbm(
    *,
    S: np.ndarray,          # (N, d_s) or (N,)
    A: np.ndarray,          # (N, d_a) or (N,)
    A_pi: np.ndarray,       # (N, d_a) or (N,)
    clip_nonneg: bool = True,
    num_boost_round: int = 100,
    lgb_params: dict | None = None,
    eps_hess: float = 1e-3,
    test_size: float = 0.2,
    early_stopping_rounds: int = 10,
    seed: int = 123,
    refit_on_all_data: bool = True,
    show_tqdm: bool = True,
    tqdm_leave: bool = True,
    tqdm_desc_es: str = "boosting importance weights (early stopping)",
    tqdm_desc_refit: str = "boosting importance weights (refit)",
):
    """
    Fit iota(s,a) = pi(a|s) / mu_b(a|s) using squared-loss ratio fitting:

        L(iota) = E_{mu_b}[iota^2] - 2 E_{pi}[iota]

    Inputs are FEATURE MATRICES S, A, A_pi. We form:
        X_sa    = [S, A]
        X_sa_pi = [S, A_pi]

    Leakage fix:
      - Split on ORIGINAL indices i=1..N, then build the long dataset separately
        for train and valid so paired rows (beh i, pi i) never split.
    """
    import numpy as np
    import lightgbm as lgb

    S = np.asarray(S)
    A = np.asarray(A)
    A_pi = np.asarray(A_pi)

    # ---- coerce to 2D ----
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A_pi.ndim == 1:
        A_pi = A_pi.reshape(-1, 1)

    N = S.shape[0]
    if A.shape[0] != N or A_pi.shape[0] != N:
        raise ValueError("S, A, A_pi must all have the same number of rows (N).")
    if A_pi.shape[1] != A.shape[1]:
        raise ValueError("A_pi must have the same feature dimension as A.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1).")

    # ---- build design matrices ----
    X_sa = np.concatenate([S, A], axis=1)
    X_sa_pi = np.concatenate([S, A_pi], axis=1)

    # ---- leakage-free split on ORIGINAL rows ----
    rng = np.random.default_rng(seed)
    permN = rng.permutation(N)
    n_validN = max(1, int(np.floor(test_size * N)))
    validN = permN[:n_validN]
    trainN = permN[n_validN:]
    if trainN.size == 0:
        raise ValueError("test_size too large: no training rows left.")

    # ---- long data separately for train/valid ----
    X_train, y_train, w_train = make_importance_ratio_long_arrays_from_sa(
        X_sa_beh=X_sa[trainN],
        X_sa_pi=X_sa_pi[trainN],
    )
    X_valid, y_valid, w_valid = make_importance_ratio_long_arrays_from_sa(
        X_sa_beh=X_sa[validN],
        X_sa_pi=X_sa_pi[validN],
    )

    dtrain = make_lgb_importance_ratio_dataset(X_train, y_train, w_train)
    dvalid = make_lgb_importance_ratio_dataset(X_valid, y_valid, w_valid)

    # ---- params ----
    if lgb_params is None:
        lgb_params = {}
    else:
        lgb_params = dict(lgb_params)

    lgb_params.setdefault("learning_rate", 0.05)
    lgb_params.setdefault("num_leaves", 64)
    lgb_params.setdefault("min_data_in_leaf", 200)
    lgb_params.setdefault("feature_fraction", 1.0)
    lgb_params.setdefault("bagging_fraction", 1.0)
    lgb_params.setdefault("verbose", -1)
    lgb_params.setdefault("num_threads", 0)
    lgb_params["min_sum_hessian_in_leaf"] = 0
    lgb_params["lambda_l2"] = 0
    lgb_params.setdefault("boost_from_average", False)

    num_boost_round = extract_num_boost_round(lgb_params=lgb_params, default=num_boost_round)

    # objective closure uses eps_hess
    def _objective(preds: np.ndarray, dataset: lgb.Dataset):
        return importance_ratio_objective(preds, dataset, eps=float(eps_hess))

    lgb_params["objective"] = _objective

    def _mk_callbacks(*, total: int, desc: str, evals_result: dict | None = None, early_stop: bool = False):
        cbs = []
        if early_stop:
            cbs.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        if evals_result is not None:
            cbs.append(lgb.record_evaluation(evals_result))
        if show_tqdm:
            cbs.append(_make_lgb_tqdm_callback(total=total, desc=desc, leave=tqdm_leave))
        return cbs

    # ---- phase 1: early stopping ----
    evals_result: dict = {}
    bst_es = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=importance_ratio_eval,
        callbacks=_mk_callbacks(
            total=num_boost_round,
            desc=tqdm_desc_es,
            evals_result=evals_result,
            early_stop=True,
        ),
    )

    best_iteration = int(getattr(bst_es, "best_iteration", 0) or 0)
    if best_iteration <= 0:
        best_iteration = int(num_boost_round)

    # ---- phase 2: refit on all data ----
    if refit_on_all_data:
        X_long, y_long, w_long = make_importance_ratio_long_arrays_from_sa(
            X_sa_beh=X_sa,
            X_sa_pi=X_sa_pi,
        )
        dall = make_lgb_importance_ratio_dataset(X_long, y_long, w_long)

        bst_iw = lgb.train(
            params=lgb_params,
            train_set=dall,
            num_boost_round=best_iteration,
            valid_sets=[dall],
            valid_names=["train"],
            feval=importance_ratio_eval,
            callbacks=_mk_callbacks(total=best_iteration, desc=tqdm_desc_refit),
        )
    else:
        bst_iw = bst_es

    # ---- predict iota on behavior rows ----
    w_hat = bst_iw.predict(
        X_sa,
        num_iteration=best_iteration if refit_on_all_data else bst_iw.best_iteration,
    )
    if clip_nonneg:
        w_hat = np.maximum(w_hat, 0.0)

    return dict(
        bst_iw=bst_iw,
        best_iteration=best_iteration,
        w_hat=w_hat,
        Xw_beh=X_sa,
        X_sa=X_sa,
        X_sa_pi=X_sa_pi,
        evals_result=evals_result,
    )


 
def make_importance_ratio_long_arrays_from_sa(X_sa_beh, X_sa_pi):
    """
    Build long data for ratio w(s,a)=pi/mu_b using only (s,a)-features.

    Objective minimized (population):
        E_{mu_b}[w^2] - 2 E_{pi}[w]
    """
    X_sa_beh = np.asarray(X_sa_beh)
    X_sa_pi  = np.asarray(X_sa_pi)

    if X_sa_beh.ndim != 2 or X_sa_pi.ndim != 2:
        raise ValueError("X_sa_beh and X_sa_pi must be 2D arrays.")
    if X_sa_beh.shape[1] != X_sa_pi.shape[1]:
        raise ValueError("X_sa_beh and X_sa_pi must have same number of columns.")

    n_b = X_sa_beh.shape[0]
    n_p = X_sa_pi.shape[0]

    X_long = np.vstack([X_sa_beh, X_sa_pi])
    y_long = np.concatenate([
        np.ones(n_b, dtype=np.int32),   # 1 = behavior (mu_b)
        np.zeros(n_p, dtype=np.int32),  # 0 = target-policy (pi)
    ])
    w_long = np.ones_like(y_long, dtype=float)
    return X_long, y_long, w_long


def make_lgb_importance_ratio_dataset(X_long, y_long, w_long=None):
    return lgb.Dataset(
        X_long,
        label=y_long,
        weight=w_long,
        free_raw_data=False,
    )


def importance_ratio_objective(y_pred, dataset, eps=1e-3):
    """
    Custom objective for w = pi/mu_b via squared-loss ratio fitting:

      behavior rows (y=1): loss = w^2      -> grad = 2w, hess = 2
      pi rows       (y=0): loss = -2w     -> grad = -2, hess = 2*eps  (stabilize)
    """
    y = dataset.get_label().astype(np.int32)
    wgt = dataset.get_weight()
    if wgt is None:
        wgt = np.ones_like(y_pred, dtype=np.float64)

    is_beh = (y == 1)
    is_pi  = ~is_beh

    grad = np.zeros_like(y_pred, dtype=float)
    hess = np.zeros_like(y_pred, dtype=float)

    grad[is_beh] = 2.0 * y_pred[is_beh]
    hess[is_beh] = 2.0

    grad[is_pi]  = -2.0
    hess[is_pi]  = 2.0 * eps

    grad *= wgt
    hess *= wgt
    return grad, hess


def importance_ratio_eval(y_pred, dataset):
    """
    Evaluation metric matching the squared-loss importance-ratio objective:

        L = E_{mu_b}[w^2] - 2 E_{pi}[w]

    where:
      y = 1  -> behavior row  (mu_b)
      y = 0  -> target-policy row (pi)

    Returns
    -------
    name : str
        Metric name
    value : float
        Empirical loss
    is_higher_better : bool
        False (we minimize)
    """
    y = dataset.get_label().astype(np.int32)
    wgt = dataset.get_weight()
    if wgt is None:
        wgt = np.ones_like(y_pred, dtype=np.float64)

    is_beh = (y == 1)
    is_pi  = ~is_beh

    # Empirical expectations (weighted)
    loss_beh = np.sum(wgt[is_beh] * y_pred[is_beh] ** 2)
    loss_pi  = np.sum(wgt[is_pi]  * y_pred[is_pi])

    loss = loss_beh - 2.0 * loss_pi

    # Normalize by total weight (optional but stabilizes scale)
    loss /= np.sum(wgt)

    return "loss", loss, False



 

def make_transition_ratio_long_arrays(X_sa, X_s_next, K, seed=None, X_s_ref=None):
    """
    Long dataset for squared-loss transition-ratio estimation.

    If X_s_ref is provided, this estimates
        ktilde(s,a,s') = P(s'|s,a) / d_ref(s')
    where d_ref is the marginal of X_s_ref (e.g. d0 if X_s_ref are initial states).

    - Behavior block uses true next states X_s_next (linear term).
    - Reference block uses permuted X_s_ref (square term).

    Backward-compatible:
      If X_s_ref is None, we use X_s_ref = X_s_next (old behavior),
      corresponding to denominator rho_{b,next}.
    """
    rng = np.random.default_rng(seed)

    X_sa = np.asarray(X_sa)
    X_s_next = np.asarray(X_s_next)
    n = X_sa.shape[0]
    if X_s_next.shape[0] != n:
        raise ValueError("X_sa and X_s_next must have the same number of rows")

    # NEW: reference states for the denominator
    if X_s_ref is None:
        X_s_ref = X_s_next
    else:
        X_s_ref = np.asarray(X_s_ref)
        if X_s_ref.shape[0] != n:
            raise ValueError("X_s_ref must have the same number of rows as X_sa")

    # Ensure 2D
    def _to_2d(x):
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        raise ValueError("State features must be 1D or 2D")

    X_s_next_2d = _to_2d(X_s_next)
    X_s_ref_2d  = _to_2d(X_s_ref)

    # Behavior block: true next states (S_i')
    X_beh = X_s_next_2d

    # Reference block: independent draws from denominator distribution (permute X_s_ref)
    X_ref = np.vstack([X_s_ref_2d[rng.permutation(n), :] for _ in range(K)])

    X_s_long = np.vstack([X_beh, X_ref])           # (n*(K+1), d_state)
    X_sa_long = np.tile(X_sa, (K + 1, 1))          # (n*(K+1), d_sa)
    X_long = np.hstack([X_sa_long, X_s_long])      # (n*(K+1), d_sa + d_state)

    y_long = np.concatenate(
        [np.ones(n, dtype=np.int32),
         np.zeros(n * K, dtype=np.int32)],
        axis=0,
    )
    w_long = np.where(y_long == 1, 1.0, 1.0 / float(K))
    return X_long, y_long, w_long




def make_lgb_transition_ratio_dataset_from_arrays(X_long, y_long, w_long=None):
    """
    Create a LightGBM Dataset for transition-ratio estimation
    with a custom squared-loss objective.
    """
    return lgb.Dataset(
        X_long,
        label=y_long,
        weight=w_long,
        free_raw_data=False,
    )

 
import numpy as np

import numpy as np

def transition_ratio_objective(y_pred, dataset, eps: float = 1e-3, lam_norm: float = 1.0):
    """
    Sum-scaled objective (numerically friendly for LightGBM), with the SAME target
    as the expectation loss up to a constant scaling of the base terms.

    Base (sum-scaled):
        L_base = sum_ref w_i k_i^2  - 2 * sum_beh w_i k_i

    Normalization penalty (still an expectation under ref):
        L_pen  = lam_norm * (E_ref[k] - 1)^2,
        E_ref[k] = sum_ref w_i k_i / sum_ref w_i.

    Notes
    -----
    - Sum-scaling removes the 1/n shrinkage in gradients that was preventing splits.
    - Behavior Hessian is 0 for the linear term; we add eps*w as a stabilizer (>=0).
    - Penalty Hessian is the exact diagonal of the true rank-1 Hessian (>=0).
    """
    y = dataset.get_label().astype(np.int32)
    wgt = dataset.get_weight()
    if wgt is None:
        wgt = np.ones_like(y_pred, dtype=np.float64)
    else:
        wgt = wgt.astype(np.float64, copy=False)

    y_pred = y_pred.astype(np.float64, copy=False)

    is_beh = (y == 1)
    is_ref = ~is_beh

    grad = np.zeros_like(y_pred, dtype=np.float64)
    hess = np.zeros_like(y_pred, dtype=np.float64)

    w_ref = wgt[is_ref]
    w_beh = wgt[is_beh]

    Z_ref = float(np.sum(w_ref))

    # -----------------------------
    # Base term: sum_ref w k^2
    # -----------------------------
    k_ref = y_pred[is_ref]
    grad[is_ref] += 2.0 * w_ref * k_ref
    hess[is_ref] += 2.0 * w_ref  # exact diagonal

    # -----------------------------
    # Base term: -2 * sum_beh w k
    # -----------------------------
    grad[is_beh] += -2.0 * w_beh
    hess[is_beh] += eps * w_beh  # stabilizer (keeps curvature >= 0)

    # -----------------------------
    # Normalization penalty: lam * (E_ref[k] - 1)^2
    # -----------------------------
    if Z_ref > 0.0:
        mu = float(np.sum(w_ref * k_ref) / Z_ref)  # E_ref[k]
        delta = mu - 1.0

        # grad on ref rows: 2*lam*delta * (w_i / Z_ref)
        grad[is_ref] += (2.0 * lam_norm * delta / Z_ref) * w_ref

        # exact diagonal Hessian of penalty: 2*lam*(w_i/Z_ref)^2
        hess[is_ref] += 2.0 * lam_norm * (w_ref / Z_ref) ** 2
    else:
        # if no ref rows, just provide tiny curvature
        hess += eps

    # Debug prints (optional)
    #print("grad mean:", float(np.mean(grad)))
    #print("grad min/max:", float(np.min(grad)), float(np.max(grad)))
    #print("hess min/max:", float(np.min(hess)), float(np.max(hess)))

    return grad, hess


def transition_ratio_eval(y_pred, dataset, lam_norm: float = 1.0):
    """
    Consistent evaluation metric for the same loss used by transition_ratio_objective:

        L = E_ref[k^2] - 2 E_beh[k] + lam_norm * (E_ref[k] - 1)^2

    with group-wise weighted expectations:
        E_ref[·] = sum_{ref} w_i · / sum_{ref} w_i
        E_beh[·] = sum_{beh} w_i · / sum_{beh} w_i
    """
    y = dataset.get_label().astype(np.int32)
    wgt = dataset.get_weight()
    if wgt is None:
        wgt = np.ones_like(y_pred, dtype=np.float64)
    else:
        wgt = wgt.astype(np.float64, copy=False)

    y_pred = y_pred.astype(np.float64, copy=False)

    is_beh = (y == 1)
    is_ref = ~is_beh

    w_ref = wgt[is_ref]
    w_beh = wgt[is_beh]

    Z_ref = float(np.sum(w_ref))
    Z_beh = float(np.sum(w_beh))

    if Z_ref <= 0.0 or Z_beh <= 0.0:
        return "loss", float("nan"), False

    k_ref = y_pred[is_ref]
    k_beh = y_pred[is_beh]

    E_ref_k2 = float(np.sum(w_ref * (k_ref ** 2)) / Z_ref)
    E_beh_k  = float(np.sum(w_beh * k_beh) / Z_beh)
    E_ref_k  = float(np.sum(w_ref * k_ref) / Z_ref)

    loss = E_ref_k2 - 2.0 * E_beh_k + lam_norm * (E_ref_k - 1.0) ** 2
    return "loss", loss, False



def extract_num_boost_round(lgb_params: dict, default: int):
    """
    Extract boosting rounds from lgb_params if present, handling common aliases.
    Removes the key from lgb_params if found.

    Supported aliases:
      - num_boost_round
      - num_iterations
      - num_iteration
      - n_estimators

    Returns
    -------
    num_boost_round : int
    """
    aliases = [
        "num_boost_round",
        "num_iterations",
        "num_iteration",
        "n_estimators",
    ]

    for key in aliases:
        if key in lgb_params and lgb_params[key] is not None:
            num_boost_round = int(lgb_params.pop(key))
            return num_boost_round

    return int(default)





def _make_lgb_tqdm_callback(
    *,
    total: int,
    desc: str,
    leave: bool = True,
    postfix_every: int = 1,
    metric_substr: str = "loss",
    ncols: int = 160,          # NEW: force enough width
):
    import sys
    from tqdm import tqdm

    pbar = tqdm(
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=False,    # CHANGED: more reliable in Jupyter
        ncols=ncols,            # NEW
        file=sys.stderr,
        mininterval=0.1,
        maxinterval=1.0,
    )

    last_i = 0

    def _cb(env):
        nonlocal last_i
        i = env.iteration + 1

        # advance bar
        delta = i - last_i
        if delta > 0:
            pbar.update(delta)
            last_i = i

        # show metric (after eval is available)
        if env.evaluation_result_list and (
            postfix_every <= 1 or i % postfix_every == 0 or env.iteration == env.end_iteration
        ):
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                if metric_substr in str(eval_name):
                    # keep it SHORT to avoid truncation
                    pbar.set_postfix_str(f"{eval_name}={result:.4g}")
                    break

        if env.iteration == env.end_iteration:
            pbar.close()

    _cb.order = 999
    _cb.before_iteration = False
    return _cb

