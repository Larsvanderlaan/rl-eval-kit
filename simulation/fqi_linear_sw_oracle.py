
import numpy as np
from tqdm import tqdm


def stationary_dist_policy(
    mdp,
    pi,
    tol: float = 1e-12,
    max_iter: int = 10_000,
):
    """
    Compute the stationary state distribution d_π for the Markov chain
    induced by P and policy π via power iteration:

        d^T = d^T P^π,   ∑_s d(s) = 1.

    Parameters
    ----------
    mdp : GarnetMDP
        The MDP with transition kernel mdp.P of shape (S, A, S).

    pi : np.ndarray, shape (n_states, n_actions)
        Policy matrix π(a|s).

    tol : float
        Convergence tolerance for ||d_{k+1} - d_k||_∞.

    max_iter : int
        Maximum number of power iterations.

    Returns
    -------
    d_pi : np.ndarray, shape (n_states,)
        Stationary state distribution under π.
    """
    S, A = mdp.n_states, mdp.n_actions

    # Build P^π
    P_pi = np.zeros((S, S), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            P_pi[s, :] += pi[s, a] * mdp.P[s, a, :]

    # Power iteration
    d = np.ones(S, dtype=np.float64) / S
    for _ in range(max_iter):
        d_next = d @ P_pi
        if np.max(np.abs(d_next - d)) < tol:
            d = d_next
            break
        d = d_next

    # Normalize to guard against numerical drift
    d = d / d.sum()
    return d



def fit_soft_fqi_linear_oracle(
    mdp,
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
    behavior_pi: np.ndarray | None = None,
    actions=None,
    init_theta=None,
    weight_clip: float | None = None,
    eps: float = 1e-12,
):
    """
    Soft FQI with linear function approximation and *oracle* stationary
    reweighting at each policy iterate.

    At iteration k:

      1) Define soft policy π_{θ_k}(a|s) ∝ exp(Q_{θ_k}(s,a) / τ_k).
      2) Compute stationary distributions:
             d_pi   = stationary_dist_policy(mdp, π_{θ_k})
             d_mu   = stationary_dist_policy(mdp, μ)
      3) Form state–action ratios:
             w(s,a) = [d_pi(s) π_{θ_k}(a|s)] / [d_mu(s) μ(a|s)]
         and use w(S_i, A_i) as sample weights in the regression.

    This uses the *true* Markov chain stationary distributions induced by
    the current policy iterate and the behavior policy.

    Parameters
    ----------
    mdp : GarnetMDP
        Environment (must have n_states, n_actions, P).
    S, A, S_next, rewards : array-like
        One-step transition dataset.
    feature_fn : callable
        Maps (S_batch, A_batch) -> feature matrix of shape (n, d_phi).
    gamma : float
        Discount factor.
    tau : float
        Initial temperature.
    tau_min : float or None
        If not None, linearly anneal τ_k from tau to tau_min over iterations.
    n_iters : int
        Number of FQI iterations.
    reg : float
        Ridge regularization strength.
    behavior_pi : np.ndarray, shape (n_states, n_actions)
        Behavior policy μ(a|s). Required for stationary ratios.
    actions : array-like or None
        Set of actions to use when forming soft values V(s').
        Defaults to all actions {0,...,A-1}.
    init_theta : np.ndarray or None
        Optional initial parameter vector of shape (d_phi,).
    weight_clip : float or None
        If not None, clip w(s,a) to [0, weight_clip].
    eps : float
        Small constant to avoid divides-by-zero.

    Returns
    -------
    theta : np.ndarray, shape (d_phi,)
        Final parameter vector.
    history : dict
        Contains "theta_seq", "bellman_error", "tau_seq".
    """
    if behavior_pi is None:
        raise ValueError("behavior_pi (μ) must be provided for oracle stationary reweighting.")

    # -----------------------------------------
    # Basic arrays
    # -----------------------------------------
    S = np.asarray(S, dtype=int)
    A = np.asarray(A, dtype=int)
    S_next = np.asarray(S_next, dtype=int)
    rewards = np.asarray(rewards, dtype=np.float64)

    n = S.shape[0]

    # Design matrix for current (s,a)
    Phi = np.asarray(feature_fn(S, A), dtype=np.float64)  # (n, d_phi)
    n, d_phi = Phi.shape

    # -----------------------------------------
    # Action set and next-state design matrices
    # -----------------------------------------
    if actions is None:
        actions = np.arange(mdp.n_actions)
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

    # -----------------------------------------
    # Precompute features for *all* (s,a) pairs
    #   to build π_{θ_k} efficiently each iter.
    # -----------------------------------------
    S_all, A_all = np.meshgrid(
        np.arange(mdp.n_states),
        np.arange(mdp.n_actions),
        indexing="ij",
    )
    S_vec_all = S_all.ravel()  # length SA
    A_vec_all = A_all.ravel()

    Phi_all = np.asarray(feature_fn(S_vec_all, A_vec_all), dtype=np.float64)  # (SA, d_phi)

    S_states = mdp.n_states
    A_actions = mdp.n_actions
    SA = S_states * A_actions

    # -----------------------------------------
    # Precompute *behavior* stationary distribution d_mu(s,a)
    # -----------------------------------------
    d_behavior_state = stationary_dist_policy(mdp, behavior_pi)  # (S,)
    d_behavior_state = d_behavior_state + eps
    d_behavior_state /= d_behavior_state.sum()

    d_behavior_sa = d_behavior_state[:, None] * behavior_pi      # (S, A)
    d_behavior_sa = d_behavior_sa + eps

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
    for k in tqdm(range(n_iters), desc="Soft Linear FQI (oracle weights)", leave=True):
        # Temperature schedule
        if tau_min is None or n_iters == 1:
            tau_k = tau
        else:
            alpha = k / (n_iters - 1)
            tau_k = tau + (tau_min - tau) * alpha
        tau_seq.append(float(tau_k))

        # -------------------------------------------------
        # 1) Build current soft policy π_{θ_k}(a|s)
        # -------------------------------------------------
        Q_all = Phi_all @ theta          # (SA,)
        Q_table = Q_all.reshape(S_states, A_actions)  # (S, A)

        # Softmax per state: π(a|s) ∝ exp(Q(s,a) / τ_k)
        scaled = Q_table / tau_k
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        exp_shifted = np.exp(scaled - max_scaled)
        pi_theta = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)  # (S, A)

        # -------------------------------------------------
        # 2) Stationary distribution d_{π_{θ_k}}(s)
        # -------------------------------------------------
        d_target_state = stationary_dist_policy(mdp, pi_theta)  # (S,)
        d_target_state = d_target_state + eps
        d_target_state /= d_target_state.sum()

        d_target_sa = d_target_state[:, None] * pi_theta       # (S, A)

        # -------------------------------------------------
        # 3) State-action ratios w(s,a) = d_pi(s,a)/d_mu(s,a)
        # -------------------------------------------------
        w_sa = d_target_sa / d_behavior_sa                     # (S, A)
        if weight_clip is not None:
            w_sa = np.minimum(w_sa, weight_clip)

        # Sample weights for dataset transitions
        w = w_sa[S, A]                                         # (n,)
        # Normalize weights to mean 1 for numerical stability
        w_mean = w.mean()
        if w_mean <= 0:
            raise ValueError("Encountered nonpositive mean weight.")
        w = w / w_mean

        # -------------------------------------------------
        # 4) Soft Bellman targets using current θ_k, τ_k
        # -------------------------------------------------
        # Q_next_all: shape (n, n_actions)
        Q_next_all = np.tensordot(Phi_next_all, theta, axes=([2], [0])).T  # (n, n_actions)

        scaled_next = Q_next_all / tau_k
        max_scaled_next = np.max(scaled_next, axis=1, keepdims=True)
        V_next = tau_k * (
            max_scaled_next.squeeze()
            + np.log(np.sum(np.exp(scaled_next - max_scaled_next), axis=1))
        )

        y = rewards + gamma * V_next

        # -------------------------------------------------
        # 5) Weighted ridge regression update
        # -------------------------------------------------
        Phi_w = w[:, None] * Phi
        G = (Phi.T @ Phi_w) / n
        G_reg = G + reg * np.eye(d_phi)

        # Cholesky solve
        L = np.linalg.cholesky(G_reg)
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
        "tau_seq": tau_seq,
    }
    return theta, history


