from __future__ import annotations

import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List
from tqdm import tqdm
from IRL.fit_importance_and_transition_ratios import fit_transition_ratio_lgbm, fit_importance_ratio_lgbm
from typing import Any, Dict, Optional

  
 
 
def fit_occupancy_ratio_lgbm(
    *,
    S: np.ndarray,
    A: np.ndarray,
    S_next: np.ndarray,
    A_pi: np.ndarray,
    gamma: float,
    num_outer_iters: int = 200,
    inner_num_boost_round: int = 1,   # NEW: number of trees per outer iteration
    mcmc_samples: int = 80,
    seed: int = 123,
    batch_query: int = 1000,
    lgb_params: Optional[Dict[str, Any]] = None,
    clip_y_min: Optional[float] = 0.0,
    clip_y_max: Optional[float] = None,
    k_lgb_params: Optional[Dict[str, Any]] = None,
    iw_lgb_params: Optional[Dict[str, Any]] = None,
    k_kwargs: Optional[Dict[str, Any]] = None,
    iw_kwargs: Optional[Dict[str, Any]] = None,
    bst_k_init: Optional[lgb.Booster] = None,
    bst_iw_init: Optional[lgb.Booster] = None,
    w_init: float = 0.0,
    early_stopping: bool = True,
    test_frac: float = 0.2,
    early_stopping_min_delta: float = 1e-6,
    early_stopping_patience: int = 10,
    refresh_on_plateau: bool = True,
    refresh_after_n_plateau: int = 1,
    eval_mcmc_multiplier: int = 5,
    eval_seed_offset: int = 777_777,
) -> Dict[str, Any]:
    # -------------------- basic checks --------------------
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    if inner_num_boost_round <= 0:
        raise ValueError("inner_num_boost_round must be positive.")
    if early_stopping and not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be in (0, 1).")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if mcmc_samples <= 0:
        raise ValueError("mcmc_samples must be positive.")
    if batch_query <= 0:
        raise ValueError("batch_query must be positive.")
    if eval_mcmc_multiplier <= 0:
        raise ValueError("eval_mcmc_multiplier must be positive.")
    if refresh_after_n_plateau <= 0:
        raise ValueError("refresh_after_n_plateau must be positive.")

    # -------------------- coerce to 2D feature matrices --------------------
    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)
    A_pi = np.asarray(A_pi)

    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if S_next.ndim == 1:
        S_next = S_next.reshape(-1, 1)
    if A_pi.ndim == 1:
        A_pi = A_pi.reshape(-1, 1)

    N = S.shape[0]
    if A.shape[0] != N or S_next.shape[0] != N or A_pi.shape[0] != N:
        raise ValueError("S, A, S_next, A_pi must all have the same number of rows (N).")
    if S_next.shape[1] != S.shape[1]:
        raise ValueError("S_next must have the same feature dimension as S.")
    if A_pi.shape[1] != A.shape[1]:
        raise ValueError("A_pi must have the same feature dimension as A.")

    # -------------------- build design matrices --------------------
    X_sa_kernel = np.concatenate([S, A], axis=1)
    X_sa_iw = X_sa_kernel
    X_sa_pi_iw = np.concatenate([S, A_pi], axis=1)

    X_sa_query = np.vstack([X_sa_pi_iw, X_sa_iw])
    X_s_query = np.vstack([S, S])
    Q = X_sa_query.shape[0]  # 2N

    # -------------------- merge nuisance params --------------------
    k_kwargs = {} if k_kwargs is None else dict(k_kwargs)
    iw_kwargs = {} if iw_kwargs is None else dict(iw_kwargs)

    default_lgb_params = {} if lgb_params is None else dict(lgb_params)
    k_lgb_params = dict(default_lgb_params) | ({} if k_lgb_params is None else dict(k_lgb_params))
    iw_lgb_params = dict(default_lgb_params) | ({} if iw_lgb_params is None else dict(iw_lgb_params))
    k_kwargs.setdefault("lgb_params", k_lgb_params)
    iw_kwargs.setdefault("lgb_params", iw_lgb_params)

    # -------------------- fit / use nuisances --------------------
    if bst_k_init is None:
        out_k = fit_transition_ratio_lgbm(
            S=S, A=A, S_next=S_next, seed=seed, refit_on_all_data=False, **k_kwargs
        )
        bst_k = out_k["bst_k"]
    else:
        bst_k = bst_k_init

    if bst_iw_init is None:
        out_iw = fit_importance_ratio_lgbm(
            S=S, A=A, A_pi=A_pi, seed=seed, refit_on_all_data=False, **iw_kwargs
        )
        bst_iw = out_iw["bst_iw"]
        iw_hat = out_iw["w_hat"]
    else:
        bst_iw = bst_iw_init
        iw_hat = np.maximum(bst_iw.predict(X_sa_iw), 0.0)

    print("iw_hat min/max:", float(np.min(iw_hat)), float(np.max(iw_hat)))
    print("iw_hat quantiles:", np.quantile(iw_hat, [0.0, 0.5, 0.9, 0.99, 1.0]))

    # -------------------- LightGBM params --------------------
    params_base = dict(
        objective="regression",
        learning_rate=0.1,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=0.0,
        verbose=-1,
        seed=seed,
    )
    if lgb_params is not None:
        params_base.update(dict(lgb_params))

    learning_rate = float(params_base.get("learning_rate", 0.1))

    # -------------------- split query rows into train/test --------------------
    if early_stopping:
        rng_split = np.random.default_rng(seed + 9871)
        perm = rng_split.permutation(Q)
        n_test = max(1, int(np.floor(test_frac * Q)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        if train_idx.size == 0:
            raise ValueError("test_frac too large: no training rows left.")
    else:
        train_idx = np.arange(Q, dtype=np.int64)
        test_idx = np.array([], dtype=np.int64)

    X_train = X_sa_query[train_idx]
    X_test = X_sa_query[test_idx] if early_stopping else None

    # -------------------- builders --------------------
    refresh_count = 0

    def _make_builder(seed_for_builder: int, mcmc_for_builder: int):
        return make_forward_occupancy_dataset(
            bst_k=bst_k,
            bst_iw=bst_iw,
            X_sa_kernel=X_sa_kernel,
            X_s_query=X_s_query,
            X_sa_iw=X_sa_iw,
            X_sa_query_iw=X_sa_query,
            gamma=gamma,
            mcmc_samples=int(mcmc_for_builder),
            seed=int(seed_for_builder),
            batch_query=int(batch_query),
        )

    def _make_train_builder():
        nonlocal refresh_count
        refresh_count += 1
        refresh_seed = int(seed + 10_000 * refresh_count)
        return _make_builder(seed_for_builder=refresh_seed, mcmc_for_builder=int(mcmc_samples))

    build_train = _make_train_builder()  # hook only
    eval_mcmc = int(max(1, int(mcmc_samples) * int(eval_mcmc_multiplier)))
    build_eval = _make_builder(seed_for_builder=int(seed + int(eval_seed_offset)), mcmc_for_builder=eval_mcmc)

    # -------------------- cached predictions --------------------
    pred_query = np.full((Q,), float(w_init), dtype=np.float64)
    pred_beh = np.full((N,), float(w_init), dtype=np.float64)

    current_model: Optional[lgb.Booster] = None

    patience = 0
    plateau_streak = 0
    trees_used = 0
    stopped_early = False
    stop_iter = None
    history: List[Dict[str, Any]] = []

    boost_iteration = 0
    tol = 1e-8

    def objective_in_params(preds: np.ndarray, train_data: lgb.Dataset):
        y = train_data.get_label()
        resid = preds - y
        grad = resid
        hess = np.ones_like(grad)
        return grad, hess

    from tqdm import tqdm
    pbar = tqdm(
        range(num_outer_iters),
        desc="Occupancy-ratio boosting",
        leave=True,
        dynamic_ncols=False,
        ncols=170,
    )

    for it in pbar:
        if current_model is not None and it < 10:
            chk_query = float(w_init) + current_model.predict(X_sa_query).astype(np.float64, copy=False)
            chk_beh = float(w_init) + current_model.predict(X_sa_iw).astype(np.float64, copy=False)
            if not np.allclose(chk_query, pred_query, atol=tol, rtol=1e-6):
                raise ValueError("pred_query cache != w_init + model.predict(X_sa_query).")
            if not np.allclose(chk_beh, pred_beh, atol=tol, rtol=1e-6):
                raise ValueError("pred_beh cache != w_init + model.predict(X_sa_iw).")

        out_base = build_eval(
            w_beh=pred_beh.astype(np.float32, copy=False),
            w_old_query=pred_query.astype(np.float32, copy=False),
            eta=1.0,
            clip_y_min=clip_y_min,
            clip_y_max=clip_y_max,
        )
        y_base_all = out_base["y"]
        y_train = y_base_all[train_idx]
        y_test = y_base_all[test_idx] if early_stopping else None

        if early_stopping:
            risk_old = float(np.mean((pred_query[test_idx] - y_test) ** 2))
        else:
            risk_old = float("nan")

        dtrain = lgb.Dataset(X_train, label=y_train, weight=None, free_raw_data=False)
        dtrain.set_init_score(np.full(train_idx.size, float(w_init), dtype=np.float64))

        params_iter = dict(params_base)
        params_iter["learning_rate"] = float(learning_rate)
        params_iter["objective"] = objective_in_params

        # IMPORTANT: train K trees as one candidate update
        bst_candidate = lgb.train(
            params=params_iter,
            train_set=dtrain,
            num_boost_round=int(inner_num_boost_round),
            init_model=current_model,
            keep_training_booster=True,
        )

        # candidate update = sum of K new trees
        delta_query = bst_candidate.predict(
            X_sa_query, start_iteration=boost_iteration, num_iteration=int(inner_num_boost_round)
        ).astype(np.float64, copy=False)
        delta_beh = bst_candidate.predict(
            X_sa_iw, start_iteration=boost_iteration, num_iteration=int(inner_num_boost_round)
        ).astype(np.float64, copy=False)

        if early_stopping:
            delta_test = bst_candidate.predict(
                X_test, start_iteration=boost_iteration, num_iteration=int(inner_num_boost_round)
            ).astype(np.float64, copy=False)
            pred_new_test = pred_query[test_idx] + delta_test
            risk_new = float(np.mean((pred_new_test - y_test) ** 2))
            improved = (risk_new <= (risk_old - early_stopping_min_delta * learning_rate))
        else:
            risk_new = float("nan")
            improved = True

        pat_next = 0 if improved else (patience + 1)
        pbar.set_postfix_str(
            f"new={risk_new:.3e} old={risk_old:.3e} d={(risk_old-risk_new):+.1e} "
            f"acc={int(improved)} pat={pat_next} tr={trees_used} lr={learning_rate:.2e} "
            f"K={int(inner_num_boost_round)} ref={refresh_count}"
        )

        row = dict(
            iter=int(it),
            risk_old=float(risk_old),
            risk_new=float(risk_new),
            improved=bool(improved),
            learning_rate=float(learning_rate),
            boost_iteration=int(boost_iteration),
            trees_used=int(trees_used),
            refresh_count=int(refresh_count),
            inner_num_boost_round=int(inner_num_boost_round),
            **out_base.get("diag", {}),
        )

        if improved:
            current_model = bst_candidate
            pred_query += delta_query
            pred_beh += delta_beh
            boost_iteration += int(inner_num_boost_round)
            trees_used += int(inner_num_boost_round)
            patience = 0
            plateau_streak = 0
            row["accepted"] = True
            row["did_refresh"] = False
        else:
            patience += 1
            plateau_streak += 1
            row["accepted"] = False

            if refresh_on_plateau and plateau_streak >= int(refresh_after_n_plateau):
                build_train = _make_train_builder()  # hook only
                plateau_streak = 0
                row["did_refresh"] = True
            else:
                row["did_refresh"] = False

            if early_stopping and (patience >= early_stopping_patience):
                stopped_early = True
                stop_iter = int(it)
                history.append(row)
                break

        history.append(row)

    M = N
    pred_pi = pred_query[:M]
    pred_sa_iw_in_query = pred_query[M:]

    return dict(
        bst_w=current_model,
        bst_k=bst_k,
        bst_iw=bst_iw,
        pred_query=pred_query,
        pred_beh=pred_beh,
        X_sa_query=X_sa_query,
        X_s_query=X_s_query,
        pred_pi=pred_pi,
        pred_iw=pred_beh,
        pred_sa_iw_in_query=pred_sa_iw_in_query,
        history=history,
        stopped_early=stopped_early,
        stop_iter=stop_iter,
        trees_used=int(trees_used),
        refresh_count=int(refresh_count),
        eval_mcmc_samples=int(eval_mcmc),
        mcmc_samples=int(mcmc_samples),
        inner_num_boost_round=int(inner_num_boost_round),
    )



from dataclasses import dataclass
import numpy as np

@dataclass
class _BatchCache:
    j0: int
    j1: int
    mb: int
    n_flat: int
    idx_flat: np.ndarray  # (n_flat,) int32
    k_flat: np.ndarray    # (n_flat,) float32


def make_forward_occupancy_dataset(
    *,
    bst_k,
    bst_iw,
    X_sa_kernel: np.ndarray,
    X_s_query: np.ndarray,
    X_sa_iw: np.ndarray,
    X_sa_query_iw: np.ndarray,
    gamma: float,
    mcmc_samples: int = 100,
    seed: int = 123,
    batch_query: int = 500,
    clip_w_query_max: float | None = 50.0,
    clip_k_max: float | None = 50.0,
    w_source_query: np.ndarray | None = None,
    pred_num_threads: int | None = None,  # <-- NEW: LightGBM predict threads
):
    rng = np.random.default_rng(seed)

    X_sa_kernel   = np.asarray(X_sa_kernel,   dtype=np.float32, order="C")
    X_s_query     = np.asarray(X_s_query,     dtype=np.float32, order="C")
    X_sa_iw       = np.asarray(X_sa_iw,       dtype=np.float32, order="C")
    X_sa_query_iw = np.asarray(X_sa_query_iw, dtype=np.float32, order="C")

    N = X_sa_kernel.shape[0]
    Q = X_s_query.shape[0]

    if X_sa_iw.shape[0] != N:
        raise ValueError("X_sa_iw must have same #rows as X_sa_kernel.")
    if X_sa_query_iw.shape[0] != Q:
        raise ValueError("X_sa_query_iw must have same #rows as X_s_query.")

    if X_s_query.ndim == 1:
        X_s_query = X_s_query.reshape(-1, 1)

    d_sa_k = X_sa_kernel.shape[1]
    d_next = X_s_query.shape[1]

    B = int(mcmc_samples)
    if B <= 0:
        raise ValueError("mcmc_samples must be positive.")

    # --- LightGBM predict helper (avoids branching everywhere) ---
    def _predict(bst, X):
        if pred_num_threads is None:
            return bst.predict(X)
        return bst.predict(X, num_threads=int(pred_num_threads))

    # iota(s,a) ≈ pi(a|s)/b(a|s) on all query rows
    w_query = _predict(bst_iw, X_sa_query_iw)
    w_query = np.maximum(w_query, 0.0).astype(np.float32, copy=False)
    if clip_w_query_max is not None:
        np.minimum(w_query, np.float32(clip_w_query_max), out=w_query)

    if w_source_query is None:
        w_source = w_query
    else:
        w_source = np.asarray(w_source_query, dtype=np.float32).reshape(-1)
        if w_source.shape[0] != Q:
            raise ValueError("w_source_query must have length Q.")
        w_source = np.maximum(w_source, 0.0).astype(np.float32, copy=False)

    # cache k draws for all query states
    caches: list[_BatchCache] = []
    max_mb = min(int(batch_query), Q)

    X_sa_flat_buf = np.empty((max_mb * B, d_sa_k), dtype=np.float32)
    Xk_buf        = np.empty((max_mb * B, d_sa_k + d_next), dtype=np.float32)

    for j0 in range(0, Q, batch_query):
        j1 = min(Q, j0 + batch_query)
        mb = j1 - j0
        n_flat = mb * B

        idx_flat = rng.integers(0, N, size=n_flat, endpoint=False).astype(np.int32, copy=False)

        # Fill first block (s,a) features
        X_sa_flat_buf[:n_flat, :] = X_sa_kernel[idx_flat, :]

        # Fill second block (s') features directly into Xk_buf (avoid s_rep_buf)
        Xk_buf[:n_flat, :d_sa_k] = X_sa_flat_buf[:n_flat, :]
        # Repeat each query state B times without allocating a huge intermediate:
        # We assign blocks of length B in a small loop over mb (mb <= batch_query).
        s_batch = X_s_query[j0:j1, :]
        for i in range(mb):
            lo = i * B
            hi = lo + B
            Xk_buf[lo:hi, d_sa_k:] = s_batch[i, :]

        k_flat = _predict(bst_k, Xk_buf[:n_flat, :])
        k_flat = np.maximum(k_flat, 0.0).astype(np.float32, copy=False)
        if clip_k_max is not None:
            np.minimum(k_flat, np.float32(clip_k_max), out=k_flat)

        caches.append(_BatchCache(j0=j0, j1=j1, mb=mb, n_flat=n_flat, idx_flat=idx_flat, k_flat=k_flat))

    numer_buf = np.empty(Q, dtype=np.float32)
    y_buf     = np.empty(Q, dtype=np.float32)
    tmp_buf   = np.empty(Q, dtype=np.float32)

    # --- NEW: preallocated gather + product buffers (avoid per-iteration allocs) ---
    w_take_buf   = np.empty(max_mb * B, dtype=np.float32)
    prod_flat_buf = np.empty(max_mb * B, dtype=np.float32)

    g = np.float32(gamma)
    one_minus_g = np.float32(1.0 - gamma)

    def build_iteration_targets(
        *,
        w_beh: np.ndarray,
        w_old_query: np.ndarray,
        eta: float = 1.0,
        clip_y_min: float | None = 0.0,
        clip_y_max: float | None = None,
    ) -> dict:
        nonlocal numer_buf, y_buf, tmp_buf, w_take_buf, prod_flat_buf

        w_beh = np.asarray(w_beh, dtype=np.float32).reshape(-1)
        if w_beh.shape[0] != N:
            raise ValueError("w_beh must have length N.")

        w_old_query = np.asarray(w_old_query, dtype=np.float32).reshape(-1)
        if w_old_query.shape[0] != Q:
            raise ValueError("w_old_query must have length Q.")

        # numer(s') = E_{(S,A)~nu_b}[ w(S,A) * ktilde(S,A,s') ]
        for bc in caches:
            mb = bc.mb
            n_flat = bc.n_flat

            # Gather w_beh[idx_flat] into preallocated buffer (NO new array)
            np.take(w_beh, bc.idx_flat, out=w_take_buf[:n_flat])

            # Multiply into preallocated product buffer
            np.multiply(w_take_buf[:n_flat], bc.k_flat, out=prod_flat_buf[:n_flat])

            # Reduce: mean over B for each of mb rows
            numer_buf[bc.j0:bc.j1] = prod_flat_buf[:n_flat].reshape(mb, B).mean(axis=1)

        # y = (1-gamma)*w_source + gamma * w_query * numer
        np.multiply(w_query, numer_buf, out=y_buf)
        np.multiply(y_buf, g, out=y_buf)

        tmp_buf[:] = w_source
        np.multiply(tmp_buf, one_minus_g, out=tmp_buf)
        np.add(y_buf, tmp_buf, out=y_buf)

        if eta < 1.0:
            eta32 = np.float32(eta)
            np.multiply(y_buf, eta32, out=y_buf)
            np.multiply(w_old_query, np.float32(1.0 - eta), out=tmp_buf)
            np.add(y_buf, tmp_buf, out=y_buf)

        if clip_y_min is not None:
            np.maximum(y_buf, np.float32(clip_y_min), out=y_buf)
        if clip_y_max is not None:
            np.minimum(y_buf, np.float32(clip_y_max), out=y_buf)

        # Keep your “y is float64” convention for downstream
        y = y_buf.astype(np.float64, copy=True)
        return dict(
            X=X_sa_query_iw,
            y=y,
            diag=dict(
                mean_target=float(np.mean(y)),
                min_target=float(np.min(y)),
                max_target=float(np.max(y)),
                mean_w_query=float(np.mean(w_query)),
            ),
        )

    return build_iteration_targets


