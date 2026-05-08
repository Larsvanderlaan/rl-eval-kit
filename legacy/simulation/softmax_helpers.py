
def soft_value_iteration_tabular(
    mdp,
    gamma: float,
    tau: float,
    tol: float = 1e-10,
    max_iter: int = 10_000,
):
    """
    Tabular soft value iteration for a known MDP.

    Computes the soft-optimal Q-function:

        Q*(s,a) = r(s,a) + γ * E_{s'|s,a}[ V*(s') ]

    where

        V*(s) = τ * log sum_a exp( Q*(s,a) / τ ).

    Parameters
    ----------
    mdp : GarnetMDP
        Environment with attributes:
            - mdp.n_states
            - mdp.n_actions
            - mdp.P  shape (S, A, S)
            - mdp.R  shape (S, A)

    gamma : float
        Discount factor.

    tau : float
        Softmax temperature.

    tol : float
        Convergence tolerance in sup-norm.

    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    Q : np.ndarray, shape (S, A)
        Soft-optimal action-value function.
    """
    S, A = mdp.n_states, mdp.n_actions
    P = mdp.P
    R = mdp.R

    Q = np.zeros((S, A), dtype=np.float64)

    for _ in range(max_iter):
        Q_old = Q.copy()

        # ---------
        # Soft value V(s)
        # ---------
        scaled = Q_old / tau
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        V = tau * (
            max_scaled.squeeze()
            + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
        )  # shape (S,)

        # ---------
        # Bellman update
        # ---------
        for s in range(S):
            for a in range(A):
                Q[s, a] = R[s, a] + gamma * P[s, a, :].dot(V)

        # ---------
        # Convergence check
        # ---------
        if np.max(np.abs(Q - Q_old)) < tol:
            break

    return Q

import numpy as np

def make_boltzmann_policy(q_values: np.ndarray, tau: float) -> np.ndarray:
    """
    Boltzmann (softmax) policy:

        pi(a|s) ∝ exp(Q(s,a) / tau)

    Parameters
    ----------
    q_values : np.ndarray, shape (n_states, n_actions)
        Q(s,a) values.

    tau : float
        Temperature parameter (tau > 0). Smaller tau -> more peaked policy.

    Returns
    -------
    pi : np.ndarray, shape (n_states, n_actions)
        Boltzmann policy with pi[s, a] = π(a|s), rows summing to 1.
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    if tau <= 0:
        raise ValueError("tau must be positive for Boltzmann policy")

    # Scale by temperature
    scaled = q_values / tau  # (S, A)

    # Numerical stability: subtract per-state max
    max_scaled = np.max(scaled, axis=1, keepdims=True)  # (S, 1)
    logits = scaled - max_scaled

    # Exponentiate and normalize
    exp_logits = np.exp(logits)
    pi = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return pi




import numpy as np
import matplotlib.pyplot as plt

 

# ------------------------------------------------------------
# Helper: compute MSE trajectory efficiently from theta_seq
# ------------------------------------------------------------
def compute_mse_trajectory(theta_seq, Phi_all, Q_soft_star):
    """
    Compute MSE trajectory E[(Q_hat - Q_soft_star)^2] over a sequence of parameters.

    Parameters
    ----------
    theta_seq : Sequence[np.ndarray]
        List of length T, each element theta_k of shape (d,).
    Phi_all : np.ndarray
        Design matrix of shape (SA, d) for all (s, a) pairs.
    Q_soft_star : np.ndarray
        Ground-truth soft-optimal Q*, shape (S, A).

    Returns
    -------
    mse : np.ndarray
        Array of shape (T,), MSE at each iteration.
    """
    if len(theta_seq) == 0:
        return np.array([])

    theta_mat = np.stack(theta_seq, axis=0)          # (T, d)
    Q_star_flat = Q_soft_star.reshape(-1)            # (SA,)

    # (SA, d) @ (d, T) = (SA, T)
    Q_hat_all = Phi_all @ theta_mat.T                # (SA, T)
    diff = Q_hat_all - Q_star_flat[:, None]          # (SA, T)

    return np.mean(diff ** 2, axis=0)                # (T,)




