import numpy as np
from tqdm import tqdm


def fit_fqe_linear(
    S,
    A,
    S_next,
    A_next,
    rewards,
    feature_fn,
    gamma: float = 0.99,
    n_iters: int = 50,
    reg: float = 1e-6,
    weights=None,
):
    """
    Efficient linear FQI / FQE with optional sample weights and a fixed next-action A'
    (e.g., drawn from a target policy).
    """

    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)
    A_next = np.asarray(A_next)
    rewards = np.asarray(rewards, dtype=np.float64)

    n = S.shape[0]

    # Design matrices
    Phi = np.asarray(feature_fn(S, A), dtype=np.float64)                 # (n, d_phi)
    Phi_next = np.asarray(feature_fn(S_next, A_next), dtype=np.float64)  # (n, d_phi)

    n, d_phi = Phi.shape
    assert Phi_next.shape == (n, d_phi)

    # Sample weights
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

    # Precompute weighted Gram matrix
    Phi_w = w[:, None] * Phi                      # (n, d_phi)
    G = (Phi.T @ Phi_w) / n                      # (d_phi, d_phi)
    G_reg = G + reg * np.eye(d_phi)
    L = np.linalg.cholesky(G_reg)

    # Initialize parameters
    theta = np.zeros(d_phi, dtype=np.float64)

    theta_seq = []
    bellman_err = []

    # =========================
    # Main FQE loop with tqdm
    # =========================
    for k in tqdm(range(n_iters), desc="Linear FQE", leave=True):
        q_next = Phi_next @ theta
        y = rewards + gamma * q_next

        b = (Phi.T @ (w * y)) / n

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
    }
    return theta, history
