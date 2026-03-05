import numpy as np
from tqdm import tqdm

 

def fit_soft_fqi_linear(
    S,
    A,
    S_next,
    rewards,
    feature_fn,
    gamma: float = 0.99,
    tau: float = 1.0,
    tau_min: float | None = None,
    n_iters: int = 50,
    reg: float = 1e-6,
    weights=None,
    actions=None,
    init_theta=None,
):
    """
    Soft FQI with linear function approximation and optional temperature homotopy.

    If tau_min is None, uses a fixed temperature tau.
    If tau_min is not None, uses a linear schedule:
        tau_k = tau + (tau_min - tau) * k / (n_iters - 1).

    Returns theta and history with theta_seq, bellman_error, and tau_seq.
    """

    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)
    rewards = np.asarray(rewards, dtype=np.float64)

    n = S.shape[0]

    # -------------------------------------------------
    # Design matrix for current (s,a)
    # -------------------------------------------------
    Phi = np.asarray(feature_fn(S, A), dtype=np.float64)  # (n, d_phi)
    n, d_phi = Phi.shape

    # -------------------------------------------------
    # Action set and next-state design matrices for ALL actions
    # -------------------------------------------------
    if actions is None:
        actions = np.unique(A)
    actions = np.asarray(actions)
    n_actions = actions.shape[0]

    Phi_next_all = []
    for a_val in actions:
        A_grid = np.full_like(A, fill_value=a_val)
        Phi_next_a = np.asarray(feature_fn(S_next, A_grid), dtype=np.float64)
        if Phi_next_a.shape != (n, d_phi):
            raise ValueError(
                f"feature_fn(S_next, {a_val}) returned shape {Phi_next_a.shape}, "
                f"expected ({n}, {d_phi})"
            )
        Phi_next_all.append(Phi_next_a)
    Phi_next_all = np.stack(Phi_next_all, axis=0)  # (n_actions, n, d_phi)

    # -------------------------------------------------
    # Sample weights
    # -------------------------------------------------
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != n:
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")

    # Normalize weights to mean 1
    w_mean = w.mean()
    if w_mean <= 0:
        raise ValueError("weights must have positive mean")
    w = w / w_mean

    # -------------------------------------------------
    # Precompute weighted Gram matrix (independent of tau)
    # -------------------------------------------------
    Phi_w = w[:, None] * Phi               # (n, d_phi)
    G = (Phi.T @ Phi_w) / n                # (d_phi, d_phi)
    G_reg = G + reg * np.eye(d_phi)
    L = np.linalg.cholesky(G_reg)

    # Initialize parameters
    if init_theta is None:
        theta = np.zeros(d_phi, dtype=np.float64)
    else:
        theta = np.asarray(init_theta, dtype=np.float64)
        if theta.shape != (d_phi,):
            raise ValueError(f"init_theta must have shape ({d_phi},), got {theta.shape}")

    theta_seq = []
    bellman_err = []
    tau_seq = []

    # =========================
    # Main soft FQI loop
    # =========================
    for k in tqdm(range(n_iters), desc="Soft Linear FQI", leave=True):
        # Linear temperature schedule if tau_min is provided
        if tau_min is None or n_iters == 1:
            tau_k = tau
        else:
            alpha = k / (n_iters - 1)  # in [0,1]
            tau_k = tau + (tau_min - tau) * alpha
        tau_seq.append(float(tau_k))

        # Q_next_all: shape (n, n_actions)
        Q_next_all = np.tensordot(Phi_next_all, theta, axes=([2], [0])).T  # (n, n_actions)

        # Soft value: V(s') = tau_k * log sum_a exp(Q(s', a) / tau_k)
        scaled = Q_next_all / tau_k
        max_scaled = np.max(scaled, axis=1, keepdims=True)  # numerical stability
        V_next = tau_k * (
            max_scaled.squeeze()
            + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
        )

        # Soft Bellman targets
        y = rewards + gamma * V_next

        # Weighted ridge regression update
        b = (Phi.T @ (w * y)) / n

        # Solve G_reg * theta_new = b via Cholesky
        z = np.linalg.solve(L, b)
        theta_new = np.linalg.solve(L.T, z)

        # Track metrics
        theta_seq.append(theta_new.copy())
        q_sa = Phi @ theta_new
        bellman_err.append(float(np.average((q_sa - y) ** 2, weights=w)))

        theta = theta_new

    history = {
        "theta_seq": theta_seq,
        "bellman_error": bellman_err,
        "tau_seq": tau_seq,
    }
    return theta, history
