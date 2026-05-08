import numpy as np


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def print_propensity_quantiles(mdp, Q, tau=1.0,
                               quantiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):
    """
    Print quantiles of pi(a|s) for each action a over all states s.
    """
    # Compute softmax policy over all states
    pi = mdp.softmax_policy(Q, tau)    # shape (S, A)
    S, A = pi.shape

    print("Propensity quantiles for each action:")
    print("====================================")
    for a in range(A):
        pa = pi[:, a]
        qs = np.quantile(pa, quantiles)
        q_str = ", ".join([f"{q:.4f}" for q in qs])
        print(f"Action {a}: {q_str}")


# ======================================================================
# Convenience wrapper for creating an IRL-ready dataset
# ======================================================================
def make_soft_optimal_irl_dataset(
    n_states=20,
    n_actions_nonzero=3,   # total actions = 1 + this
    feature_dim=5,
    branching=3,
    gamma=0.95,
    tau=0.5,
    horizon=20,
    n_episodes=50,
    seed_mdp=0,
    seed_traj=1,
    beta_precision=20.0,    # controls reward noise
):
    """
    Builds a GarnetMDP with linear-in-phi rewards (squashed via logistic link
    and sampled from a Beta distribution for bounded rewards), computes
    the soft-optimal policy using soft value iteration, and generates
    expert trajectories under that policy.

    Side-effects on `mdp`:
      - Computes and stores the TRUE soft Q/V for the given tau:
            mdp.Q_true_soft, mdp.V_true_soft, mdp.tau_true
      - Stores the TRUE reward mean matrix E[R|s,a] in:
            mdp.R_mean_sa

    Returns:
        mdp, Q_star, V_star, trajectories
    """

    mdp = GarnetMDP(
        n_states=n_states,
        n_actions_nonzero=n_actions_nonzero,
        feature_dim=feature_dim,
        branching=branching,
        gamma=gamma,
        seed=seed_mdp,
        beta_precision=beta_precision,
    )

    # Soft-optimal Q and V (TRUE soft Q-function for this tau)
    Q_star, V_star = mdp.soft_value_iteration(tau=tau)

    # ------------------------------------------------------------------
    # Store true soft Q/V inside the MDP for later access
    # ------------------------------------------------------------------
    mdp.Q_true_soft = Q_star
    mdp.V_true_soft = V_star
    mdp.tau_true = tau

    # Expert demonstrations
    trajectories = mdp.sample_trajectories(
        Q=Q_star,
        tau=tau,
        n_episodes=n_episodes,
        horizon=horizon,
        seed=seed_traj
    )

    return mdp, Q_star, V_star, trajectories


class GarnetMDP:
    """
    Garnet MDP with linear reward model:
        z = theta^T phi(s,a)
        mu = logistic(z)
        R(s,a) ~ Beta(c*mu, c*(1-mu))

    Action 0 has reward exactly 0.

    If structured_transitions=True, transitions P(s'|s,a) depend on
    state features via a Gaussian kernel in feature space.
    """
    def __init__(
        self,
        n_states,
        n_actions_nonzero,
        feature_dim,
        branching,
        gamma,
        seed,
        beta_precision=20.0,       # c parameter
        structured_transitions=True,
        transition_sigma=1.0,      # controls locality in feature space
    ):
        rng = np.random.RandomState(seed)
        self.rng = rng

        self.n_states = n_states
        self.n_actions = 1 + n_actions_nonzero
        self.feature_dim = feature_dim
        self.branching = branching
        self.gamma = gamma
        self.beta_precision = beta_precision
        self.structured_transitions = structured_transitions
        self.transition_sigma = transition_sigma

        # Placeholders for "true" quantities populated later
        self.Q_true_soft = None    # will be set by make_soft_optimal_irl_dataset
        self.V_true_soft = None
        self.tau_true    = None

        # ------------------------------------------------------------
        # Linear reward features (only for a >= 1)
        # ------------------------------------------------------------
        self.features = np.zeros((n_states, self.n_actions, feature_dim))
        self.features[:, 1:, :] = rng.normal(
            size=(n_states, n_actions_nonzero, feature_dim)
        )

        # ------------------------------------------------------------
        # State embeddings for transitions:
        # use average over nonzero-action features as ψ(s)
        # ------------------------------------------------------------
        # shape: (n_states, feature_dim)
        self.state_embed = self.features[:, 1:, :].mean(axis=1)

        # ------------------------------------------------------------
        # Action-specific offsets in feature space (δ_a)
        # ------------------------------------------------------------
        # shape: (n_actions, feature_dim)
        self.action_offsets = rng.normal(size=(self.n_actions, self.feature_dim))

        # ------------------------------------------------------------
        # Transitions P(s'|s,a)
        #   - If structured_transitions=True: depend on ψ(s) and δ_a
        #   - Else: standard random Garnet transitions
        # ------------------------------------------------------------
        P = np.zeros((n_states, self.n_actions, n_states))

        if structured_transitions:
            sigma2 = transition_sigma ** 2
            for s in range(n_states):
                psi_s = self.state_embed[s]  # (feature_dim,)
                for a in range(self.n_actions):
                    # target point in feature space for this (s,a)
                    center = psi_s + self.action_offsets[a]  # (feature_dim,)

                    # squared distance from all states' embeddings to center
                    diff = self.state_embed - center         # (n_states, feature_dim)
                    dist2 = np.sum(diff**2, axis=1)         # (n_states,)

                    # Gaussian kernel logits
                    logits = -dist2 / (2.0 * sigma2)

                    # numerically stable softmax
                    logits -= logits.max()
                    probs = np.exp(logits)
                    probs /= probs.sum()

                    P[s, a, :] = probs
        else:
            # Original Garnet-style random transitions
            for s in range(n_states):
                for a in range(self.n_actions):
                    nxt = rng.choice(
                        n_states,
                        size=min(branching, n_states),
                        replace=False
                    )
                    probs = rng.dirichlet(np.ones(len(nxt)))
                    P[s, a, nxt] = probs

        self.P = P

        # ------------------------------------------------------------
        # Reward parameter theta
        # ------------------------------------------------------------
        self.theta_true = rng.normal(size=feature_dim)

        # ------------------------------------------------------------
        # Generate rewards from Beta with mean σ(θᵀϕ)
        #   - R_mean_sa: true conditional mean E[R|s,a]
        #   - R_sa: one realized reward matrix (deterministic given seed)
        # ------------------------------------------------------------
        self.R_sa = np.zeros((n_states, self.n_actions))
        self.R_mean_sa = np.zeros((n_states, self.n_actions))  # store mean

        # z, mu only defined for a >= 1
        z = np.tensordot(self.features[:, 1:, :], self.theta_true, axes=(2, 0))
        mu = logistic(z)  # shape (n_states, n_actions_nonzero)

        # Store true conditional mean reward
        self.R_mean_sa[:, 1:] = mu

        # Sample one reward matrix from Beta around that mean
        alpha = beta_precision * mu
        beta = beta_precision * (1.0 - mu)
        R_samples = rng.beta(alpha, beta)
        self.R_sa[:, 1:] = R_samples

    # ==============================================================
    # Soft Value Iteration (stable) using TRUE mean rewards
    # ==============================================================
    def soft_value_iteration(self, tau=1.0, max_iter=2000, tol=1e-8):
        S, A = self.n_states, self.n_actions
        gamma = self.gamma
        tau_internal = tau

        Q = np.zeros((S, A))

        for _ in range(max_iter):
            # soft value from Q
            max_Q = Q.max(axis=1, keepdims=True)
            lse = max_Q + np.log(
                np.exp((Q - max_Q) / tau_internal).sum(axis=1, keepdims=True)
            )
            V = (tau_internal * lse).squeeze()

            # Bellman update using true mean rewards
            Q_new = self.R_mean_sa + gamma * (self.P @ V)

            if np.max(np.abs(Q_new - Q)) < tol:
                break

            Q = Q_new

        # Final V
        max_Q = Q.max(axis=1, keepdims=True)
        lse = max_Q + np.log(
            np.exp((Q - max_Q) / tau_internal).sum(axis=1, keepdims=True)
        )
        V = (tau_internal * lse).squeeze()

        return Q, V

    # ==============================================================
    # Softmax Policy (stable)
    # ==============================================================
    def softmax_policy(self, Q, tau=1.0):
        tau_internal = tau
        Qs = Q / tau_internal
        Qs -= Qs.max(axis=1, keepdims=True)
        expQ = np.exp(Qs)
        return expQ / expQ.sum(axis=1, keepdims=True)

    # ==============================================================
    # Trajectory Generator
    # ==============================================================
    def sample_trajectories(self, Q, tau, n_episodes, horizon, seed):
        rng = np.random.RandomState(seed)

        # Compute policy once (stationary)
        pi = self.softmax_policy(Q, tau)   # shape (S, A)
        S, A = self.n_states, self.n_actions

        trajectories = []

        for _ in range(n_episodes):
            s = rng.randint(S)
            traj_states = []
            traj_actions = []
            traj_rewards = []
            traj_next = []

            for _ in range(horizon):
                a = rng.choice(A, p=pi[s])
                r = self.R_sa[s, a]
                s_next = rng.choice(S, p=self.P[s, a])

                traj_states.append(s)
                traj_actions.append(a)
                traj_rewards.append(r)
                traj_next.append(s_next)

                s = s_next

            trajectories.append({
                "states": np.array(traj_states),
                "actions": np.array(traj_actions),
                "rewards": np.array(traj_rewards),
                "next_states": np.array(traj_next)
            })

        return trajectories

    # ==============================================================
    # Helper accessors for "true" quantities
    # ==============================================================
    def get_true_reward_mean(self):
        return self.R_mean_sa

    def get_true_reward_sample(self):
        return self.R_sa

    def get_true_soft_Q(self):
        return self.Q_true_soft

    def get_true_soft_V(self):
        return self.V_true_soft

    def get_true_tau(self):
        return self.tau_true
