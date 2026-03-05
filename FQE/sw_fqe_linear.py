import numpy as np

from FQE.fqe_linear import *
from dualDICE.dualdice_linear import dualdice_linear


def fit_stationary_weighted_fqe_linear(
    S,
    A,
    S_next,
    A_next,
    rewards,
    feature_fn,
    gamma: float = 0.99,
    n_iters: int = 50,
    reg: float = 1e-6,
    ratio_feature_fn=None,
    dualdice_args: dict | None = None,
):
    """
    Linear stationary-weighted FQE (SW-FQE) via closed-form DualDICE +
    weighted linear FQE.

    The stationary ratio is estimated using the γ = 1, normalized
    closed-form DualDICE-style solution:

        w_θ(s,a) = φ_w(s,a)^T θ

    where φ_w is given by `ratio_feature_fn` (defaults to `feature_fn`).
    """

    # ------------------ Defaults ------------------
    if dualdice_args is None:
        dualdice_args = {}

    ridge_B = dualdice_args.get("ridge_B", 1e-6)
    ridge_C = dualdice_args.get("ridge_C", 1e-6)

    # ------------------ Array safety ------------------
    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)
    A_next = np.asarray(A_next)
    rewards = np.asarray(rewards)

    n = S.shape[0]

    # ------------------ Ratio features ------------------
    if ratio_feature_fn is None:
        ratio_feature_fn = feature_fn

    # Just for diagnostics and later weight computation
    Phi_w = np.asarray(ratio_feature_fn(S, A), dtype=np.float64)
    Phi_w_next = np.asarray(ratio_feature_fn(S_next, A_next), dtype=np.float64)
    assert Phi_w.shape == Phi_w_next.shape

    # ================== DualDICE (closed-form) step ==================
    print("\n==============================")
    print("Fitting stationary density ratios via closed-form DualDICE")
    print("==============================")
    print("Ratio feature dim:", Phi_w.shape[1])
    print("Number of samples:", n)
    print("ridge_B:", ridge_B, "ridge_C:", ridge_C)

    theta_ratio = dualdice_linear(
        S=S,
        A=A,
        S_next=S_next,
        A_next=A_next,
        feature_fn=ratio_feature_fn,
        ridge_B=ridge_B,
        ridge_C=ridge_C,
    )

    print("Closed-form DualDICE finished.")
    print("theta_ratio stats: min", theta_ratio.min(), "max", theta_ratio.max())

    # ------------------ Compute stationary weights ------------------
    w = Phi_w @ theta_ratio  # shape (n,)

    # Clip and nudge to avoid exact zeros / negatives
    w = np.maximum(w, 0.0)
    if np.all(w == 0):
        raise ValueError("All stationary weights are zero after clipping.")
    w = w + 1e-8

    print("\nStationary weight diagnostics:")
    print("  w quantiles:", np.quantile(w, [0, 0.5, 0.9, 0.99, 0.999]))

    # ================== Stationary-weighted FQE ==================
    print("\n==============================")
    print("Fitting stationary-weighted Q-function via linear FQE")
    print("==============================")
    print("Q feature dim:", feature_fn(S[:1], A[:1]).shape[1])
    print("Number of FQE iterations:", n_iters)
    print("Ridge reg:", reg)

    theta_Q, history = fit_fqe_linear(
        S=S,
        A=A,
        S_next=S_next,
        A_next=A_next,
        rewards=rewards,
        feature_fn=feature_fn,
        gamma=gamma,
        n_iters=n_iters,
        reg=reg,
        weights=w,
    )

    print("Weighted FQE finished.")

    # ------------------ Augment history ------------------
    history = dict(history)
    history["stationary_weights"] = w
    history["theta_ratio"] = theta_ratio
    history["dualdice_args"] = {"ridge_B": ridge_B, "ridge_C": ridge_C}

    return theta_Q, history
