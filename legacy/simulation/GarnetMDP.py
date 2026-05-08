import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict

 

# ============================================================
# 1. Garnet MDP definition
# ============================================================

@dataclass
class GarnetMDP:
    """
    Garnet MDP with:
      - n_states: number of states |S|
      - n_actions: number of actions |A|
      - P: transition kernel of shape (n_states, n_actions, n_states)
      - R: reward function of shape (n_states, n_actions)
    """
    n_states: int
    n_actions: int
    P: np.ndarray  # shape (S, A, S)
    R: np.ndarray  # shape (S, A)

    def step(self, s: int, a: int, rng: np.random.RandomState) -> Tuple[int, float]:
        """
        Sample (s', r) given (s, a) using the transition kernel and reward table.

        Parameters
        ----------
        s : int
            Current state.
        a : int
            Action taken.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        s_next : int
            Next state.
        r : float
            Reward r(s, a).
        """
        p_next = self.P[s, a, :]           # (S,)
        s_next = rng.choice(self.n_states, p=p_next)
        r = self.R[s, a]
        return s_next, r


def make_garnet_mdp(
    n_states: int,
    n_actions: int,
    branching_factor: int,
    reward_std: float = 1.0,
    reward_mean: float = 0.0,
    seed: Optional[int] = None,
) -> GarnetMDP:
    """
    Construct a random Garnet MDP.

    For each (s,a), we:
      - sample `branching_factor` next states uniformly without replacement,
      - draw a Dirichlet over those states to form P(s'|s,a),
      - zeros elsewhere.
    Rewards r(s,a) are drawn i.i.d. from N(reward_mean, reward_std^2).

    Parameters
    ----------
    n_states : int
        Number of states |S|.

    n_actions : int
        Number of actions |A|.

    branching_factor : int
        Number of nonzero transitions per (s,a). Must satisfy 1 <= K <= n_states.

    reward_std : float, default=1.0
        Standard deviation of Gaussian rewards.

    reward_mean : float, default=0.0
        Mean of Gaussian rewards.

    seed : int or None, default=None
        Random seed.

    Returns
    -------
    GarnetMDP
        Randomly generated Garnet MDP instance.
    """
    if branching_factor < 1 or branching_factor > n_states:
        raise ValueError("branching_factor must be in [1, n_states].")

    rng = np.random.RandomState(seed)

    P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R = rng.normal(loc=reward_mean, scale=reward_std, size=(n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            # choose K distinct next-states
            support = rng.choice(n_states, size=branching_factor, replace=False)
            probs = rng.dirichlet(alpha=np.ones(branching_factor))
            P[s, a, support] = probs

    return GarnetMDP(n_states=n_states, n_actions=n_actions, P=P, R=R)


# ============================================================
# 2. Policies
# ============================================================

def make_random_policy(
    n_states: int,
    n_actions: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Sample a random stochastic policy π(a|s) via row-wise Dirichlet.

    Returns
    -------
    pi : np.ndarray, shape (n_states, n_actions)
        Policy matrix with pi[s, a] = π(a|s).
    """
    pi = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        pi[s, :] = rng.dirichlet(alpha=np.ones(n_actions))
    return pi


def make_epsilon_greedy_policy(
    q_values: np.ndarray,
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    Construct an ε-greedy policy from a Q-table.

    Parameters
    ----------
    q_values : np.ndarray, shape (n_states, n_actions)
        Q(s,a) values.

    epsilon : float
        Exploration probability.

    Returns
    -------
    pi : np.ndarray, shape (n_states, n_actions)
        ε-greedy policy.
    """
    n_states, n_actions = q_values.shape
    pi = np.full((n_states, n_actions), fill_value=epsilon / n_actions)

    greedy_actions = np.argmax(q_values, axis=1)
    for s in range(n_states):
        a_star = greedy_actions[s]
        pi[s, a_star] += 1.0 - epsilon
    return pi


def sample_policy_action(
    pi: np.ndarray,
    s: int,
    rng: np.random.RandomState,
) -> int:
    """
    Sample action from a (possibly stochastic) policy π(a|s).

    Parameters
    ----------
    pi : np.ndarray, shape (n_states, n_actions)
        Policy matrix.

    s : int
        Current state.

    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    a : int
        Sampled action.
    """
    return rng.choice(pi.shape[1], p=pi[s, :])


# ============================================================
# 3. Bellman utilities (V^π, Q^π, stationary dist)
# ============================================================

def compute_vq_pi(
    mdp: GarnetMDP,
    pi: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute V^π and Q^π by solving the Bellman equations:
      V = r^π + γ P^π V
      Q(s,a) = r(s,a) + γ ∑_{s'} P(s'|s,a) V(s').

    Parameters
    ----------
    mdp : GarnetMDP
        The MDP.

    pi : np.ndarray, shape (n_states, n_actions)
        Policy π(a|s).

    gamma : float
        Discount factor.

    Returns
    -------
    V_pi : np.ndarray, shape (n_states,)
        State-value function V^π.

    Q_pi : np.ndarray, shape (n_states, n_actions)
        Action-value function Q^π.
    """
    S, A = mdp.n_states, mdp.n_actions
    P, R = mdp.P, mdp.R

    # P^π and r^π
    P_pi = np.zeros((S, S), dtype=np.float64)
    r_pi = np.zeros(S, dtype=np.float64)

    for s in range(S):
        for a in range(A):
            P_pi[s, :] += pi[s, a] * P[s, a, :]
            r_pi[s] += pi[s, a] * R[s, a]

    I = np.eye(S)
    V_pi = np.linalg.solve(I - gamma * P_pi, r_pi)

    Q_pi = np.zeros((S, A), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            Q_pi[s, a] = R[s, a] + gamma * P[s, a, :].dot(V_pi)

    return V_pi, Q_pi


def stationary_dist_policy(
    mdp: GarnetMDP,
    pi: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Compute the stationary state distribution d_π for the Markov chain
    induced by P and policy π via power iteration:

        d^T = d^T P^π,   ∑_s d(s) = 1.

    Parameters
    ----------
    mdp : GarnetMDP
        The MDP.

    pi : np.ndarray, shape (n_states, n_actions)
        Policy matrix.

    tol : float
        Convergence tolerance.

    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    d_pi : np.ndarray, shape (n_states,)
        Stationary state distribution under π.
    """
    S, A = mdp.n_states, mdp.n_actions
    P_pi = np.zeros((S, S), dtype=np.float64)

    for s in range(S):
        for a in range(A):
            P_pi[s, :] += pi[s, a] * mdp.P[s, a, :]

    d = np.ones(S, dtype=np.float64) / S
    for _ in range(max_iter):
        d_next = d @ P_pi
        if np.max(np.abs(d_next - d)) < tol:
            d = d_next
            break
    d = d / d.sum()
    return d
    
def empirical_state_dist(
    S: np.ndarray,
    n_states: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Estimate the empirical state distribution ρ̂_S(s) from samples.

    Returns
    -------
    rho_s : np.ndarray, shape (n_states,)
        Empirical state frequencies, summing to 1.
    """
    S = np.asarray(S, dtype=int)
    counts = np.bincount(S, minlength=n_states).astype(np.float64)
    rho_s = counts + eps
    rho_s /= rho_s.sum()
    return rho_s




# ============================================================
# 4. Linear Q-function model: correct but NOT Bellman-complete
# ============================================================

@dataclass
class LinearQFeatures:
    """
    Linear Q-function model:

        Q_theta(s,a) = phi(s,a)^T theta,

    where phi(s,a) ∈ R^d is a fixed feature map.

    We construct phi so that Q^π (for a chosen target policy π) lies in the
    linear span of {phi(·,·)}, i.e., the model is *correct* for Q^π, while
    the space is generically NOT invariant under the Bellman operator, so
    Bellman completeness fails: T^π F ⊄ F.
    """
    n_states: int
    n_actions: int
    feature_dim: int
    Phi: np.ndarray  # shape (n_states, n_actions, feature_dim)

    def featurize(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Return features phi(S_i, A_i) for a batch of indices.

        Parameters
        ----------
        S : np.ndarray, shape (n_samples,)
            State indices.
        A : np.ndarray, shape (n_samples,)
            Action indices.

        Returns
        -------
        X : np.ndarray, shape (n_samples, feature_dim)
            Feature matrix.
        """
        S = np.asarray(S, dtype=int)
        A = np.asarray(A, dtype=int)
        return self.Phi[S, A, :]


def make_linear_q_features_with_true_Q(
    mdp: GarnetMDP,
    Q_pi_true: np.ndarray,
    feature_dim: int,
    seed: Optional[int] = None,
) -> LinearQFeatures:
    """
    Construct a linear Q-feature map whose span contains Q^π exactly, but
    is low-dimensional and generically NOT Bellman-complete.

    Construction (over (s,a) pairs):
      - Flatten Q^π to q_true ∈ R^{S*A}.
      - Sample (feature_dim - 1) random basis vectors in R^{S*A}.
      - Take these plus q_true as columns of a matrix B ∈ R^{SA×d}.
        Then F = { B θ : θ ∈ R^d } is a d-dim subspace with Q^π ∈ F.
      - For a random Garnet MDP, T^π(F) ≠ F with probability 1, so there
        exists Q ∈ F with T^π Q ∉ F (no Bellman completeness).

    We optionally apply per-(s,a) normalization (row scaling), which does
    not change the span of the columns and hence keeps Q^π ∈ F.

    Parameters
    ----------
    mdp : GarnetMDP
        Environment (for n_states, n_actions).

    Q_pi_true : np.ndarray, shape (n_states, n_actions)
        Ground-truth Q^π for the evaluation policy.

    feature_dim : int
        Dimensionality of the linear Q-function features (d). Should satisfy
        1 <= d <= n_states * n_actions. Typically take d << n_states * n_actions.

    seed : int or None
        Random seed.

    normalize : bool, default=True
        If True, normalize each feature vector phi(s,a) to unit ℓ2 norm.

    Returns
    -------
    LinearQFeatures
        Linear feature map whose span contains Q^π but is not Bellman-complete.
    """
    S, A = mdp.n_states, mdp.n_actions
    SA = S * A

    if feature_dim < 1:
        raise ValueError("feature_dim must be at least 1.")
    if feature_dim > SA:
        raise ValueError("feature_dim cannot exceed n_states * n_actions.")

    rng = np.random.RandomState(seed)

    q_true_flat = Q_pi_true.reshape(-1)  # shape (SA,)

    # Basis matrix B: columns span the function class F over (s,a).
    B = np.zeros((SA, feature_dim), dtype=np.float64)

    # First feature_dim - 1 columns are random.
    for j in range(feature_dim - 1):
        B[:, j] = rng.normal(size=SA)

    # Last column is exactly Q^π flattened, so Q^π ∈ span(B).
    B[:, feature_dim - 1] = q_true_flat

    # Reshape into Phi(s,a,·) by assigning each (s,a) to a row of B.
    Phi_flat = B  # shape (SA, d)
    Phi = Phi_flat.reshape(S, A, feature_dim)

 

    return LinearQFeatures(
        n_states=S,
        n_actions=A,
        feature_dim=feature_dim,
        Phi=Phi,
    )


# ============================================================
# 5. Dataset simulation for FQI/FQE
# ============================================================
def simulate_offpolicy_dataset(
    mdp: GarnetMDP,
    behavior_pi: np.ndarray,
    target_pi: np.ndarray,
    n_samples: int,
    start_state_dist: Optional[np.ndarray] = None,
    rng_seed: Optional[int] = None,
    q_features: Optional[LinearQFeatures] = None,
    sampling_type: str = "reset",   # {"reset", "trajectory"}
) -> Dict[str, np.ndarray]:
    """
    Simulate an off-policy one-step transition dataset:

        (S_i, A_i, R_i, S'_i, A'_i),   i = 1,...,n_samples.

    Sampling modes
    --------------
    sampling_type = "reset":
        Each transition is sampled i.i.d. via
            S_i ~ start_state_dist,
            A_i ~ μ(·|S_i).
        This deliberately destroys stationarity and creates strong norm mismatch.

    sampling_type = "trajectory":
        A single long trajectory is generated under μ:
            S_{i+1} ~ P(·|S_i, A_i).

    Next-state actions A'_i are always sampled from the target policy π.
    """
    if sampling_type not in {"reset", "trajectory"}:
        raise ValueError("sampling_type must be one of {'reset', 'trajectory'}")

    rng = np.random.RandomState(rng_seed)
    S, A = mdp.n_states, mdp.n_actions

    if start_state_dist is None:
        start_state_dist = np.ones(S, dtype=np.float64) / S
    else:
        start_state_dist = np.asarray(start_state_dist, dtype=np.float64)
        start_state_dist = start_state_dist / start_state_dist.sum()

    S_arr = np.zeros(n_samples, dtype=int)
    A_arr = np.zeros(n_samples, dtype=int)
    R_arr = np.zeros(n_samples, dtype=float)
    S_next_arr = np.zeros(n_samples, dtype=int)
    A_next_arr = np.zeros(n_samples, dtype=int)

    # initialize only for trajectory mode
    if sampling_type == "trajectory":
        s = rng.choice(S, p=start_state_dist)

    for i in range(n_samples):

        # ✅ RESET MODE: fresh state each sample
        if sampling_type == "reset":
            s = rng.choice(S, p=start_state_dist)

        a = sample_policy_action(behavior_pi, s, rng)
        s_next, r = mdp.step(s, a, rng)
        a_next = sample_policy_action(target_pi, s_next, rng)

        S_arr[i] = s
        A_arr[i] = a
        R_arr[i] = r
        S_next_arr[i] = s_next
        A_next_arr[i] = a_next

        # ✅ TRAJECTORY MODE: continue chain
        if sampling_type == "trajectory":
            s = s_next

    data: Dict[str, np.ndarray] = {
        "S": S_arr,
        "A": A_arr,
        "R": R_arr,
        "S_next": S_next_arr,
        "A_next": A_next_arr,
    }

    if q_features is not None:
        X = q_features.featurize(S_arr, A_arr)
        X_next = q_features.featurize(S_next_arr, A_next_arr)
        data["X"] = X
        data["X_next"] = X_next

    return data



 