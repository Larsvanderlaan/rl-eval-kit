from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FiniteMDP:
    """A finite discounted MDP with behavior and target policies."""

    transition: Array
    rewards: Array
    behavior_policy: Array
    target_policy: Array
    reference_state_dist: Array
    initial_state_dist: Array
    metadata: dict[str, float | int | str]

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transition.shape[1])

    @property
    def n_state_actions(self) -> int:
        return self.n_states * self.n_actions


@dataclass(frozen=True)
class TabularTruth:
    """Exact state-action occupancy objects for one discount."""

    gamma: float
    p_pi: Array
    nu: Array
    d0: Array
    nu_plus: Array
    omega0: Array
    c_pi: Array
    omega_star: Array
    d_pi_gamma: Array

    def bellman_update(
        self,
        omega: Array,
        *,
        omega0: Array | None = None,
        c_pi: Array | None = None,
    ) -> Array:
        """Evaluate the exact backward occupancy Bellman update."""

        w = np.asarray(omega, dtype=np.float64).reshape(-1)
        init = self.omega0 if omega0 is None else np.asarray(omega0, dtype=np.float64).reshape(-1)
        coverage = self.c_pi if c_pi is None else np.asarray(c_pi, dtype=np.float64).reshape(-1)
        m_omega = self.backward_conditional_mean(w)
        return (1.0 - self.gamma) * init + self.gamma * coverage * m_omega

    def pushforward_density(self, omega: Array) -> Array:
        """Return d((omega nu) P_pi) / dnu."""

        w = np.asarray(omega, dtype=np.float64).reshape(-1)
        pushed = (w * self.nu) @ self.p_pi
        return safe_divide(pushed, self.nu)

    def backward_conditional_mean(self, omega: Array) -> Array:
        """Return E[omega(X) | X^+=x] on the finite grid."""

        w = np.asarray(omega, dtype=np.float64).reshape(-1)
        numerator = (w * self.nu) @ self.p_pi
        return safe_divide(numerator, self.nu_plus)


@dataclass(frozen=True)
class FiniteDataset:
    """Sampled transitions plus exact tabular truth for evaluation."""

    mdp: FiniteMDP
    truth: TabularTruth
    states_idx: Array
    actions_idx: Array
    next_states_idx: Array
    target_actions_idx: Array
    next_target_actions_idx: Array
    rewards: Array
    states: Array
    actions: Array
    next_states: Array
    target_actions: Array
    next_target_actions: Array
    all_states: Array
    all_actions: Array
    all_state_action_indices: Array
    reward_panel: Array
    seed: int
    sample_size: int

    @property
    def setting(self) -> str:
        return "finite_dirichlet_mdp"

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def true_ratio_sample(self) -> Array:
        return self.truth.omega_star[state_action_indices(self.states_idx, self.actions_idx, self.mdp.n_actions)]


def make_random_finite_mdp(
    *,
    n_states: int,
    n_actions: int,
    transition_concentration: float,
    mismatch: float,
    overlap_floor: float,
    seed: int,
) -> FiniteMDP:
    """Generate a full-support random finite MDP for occupancy-ratio tests."""

    rng = np.random.default_rng(seed)
    transition = rng.dirichlet(
        np.full(int(n_states), float(transition_concentration), dtype=np.float64),
        size=(int(n_states), int(n_actions)),
    )
    transition = transition.reshape(int(n_states), int(n_actions), int(n_states))

    behavior_logits = rng.normal(scale=0.7, size=(int(n_states), int(n_actions)))
    shift_logits = rng.normal(scale=1.0, size=(int(n_states), int(n_actions)))
    behavior_policy = floor_probs(softmax(behavior_logits, axis=1), overlap_floor)
    target_policy = floor_probs(softmax(behavior_logits + float(mismatch) * shift_logits, axis=1), overlap_floor)

    reference_state_dist = floor_probs(
        rng.dirichlet(np.ones(int(n_states), dtype=np.float64)),
        overlap_floor / max(int(n_states), 1),
    )
    initial_state_dist = reference_state_dist.copy()
    rewards = rng.uniform(-1.0, 1.0, size=(int(n_states), int(n_actions)))
    return FiniteMDP(
        transition=transition,
        rewards=rewards,
        behavior_policy=behavior_policy,
        target_policy=target_policy,
        reference_state_dist=reference_state_dist,
        initial_state_dist=initial_state_dist,
        metadata={
            "n_states": int(n_states),
            "n_actions": int(n_actions),
            "transition_concentration": float(transition_concentration),
            "mismatch": float(mismatch),
            "overlap_floor": float(overlap_floor),
            "mdp_seed": int(seed),
        },
    )


def exact_tabular_truth(mdp: FiniteMDP, gamma: float) -> TabularTruth:
    """Compute exact discounted occupancy ratios with row-vector convention."""

    p_pi = target_state_action_kernel(mdp)
    nu = (mdp.reference_state_dist[:, None] * mdp.behavior_policy).reshape(-1)
    d0 = (mdp.initial_state_dist[:, None] * mdp.target_policy).reshape(-1)
    nu_plus = nu @ p_pi
    system = np.eye(mdp.n_state_actions, dtype=np.float64) - float(gamma) * p_pi
    d_pi_gamma = (1.0 - float(gamma)) * np.linalg.solve(system.T, d0)
    d_pi_gamma = np.maximum(d_pi_gamma, 0.0)
    d_pi_gamma = d_pi_gamma / np.sum(d_pi_gamma)
    return TabularTruth(
        gamma=float(gamma),
        p_pi=p_pi,
        nu=nu,
        d0=d0,
        nu_plus=nu_plus,
        omega0=safe_divide(d0, nu),
        c_pi=safe_divide(nu_plus, nu),
        omega_star=safe_divide(d_pi_gamma, nu),
        d_pi_gamma=d_pi_gamma,
    )


def sample_finite_dataset(
    *,
    mdp: FiniteMDP,
    gamma: float,
    sample_size: int,
    seed: int,
    n_reward_sweeps: int = 64,
) -> FiniteDataset:
    """Sample offline transitions from nu and attach exact grid truth."""

    rng = np.random.default_rng(seed)
    truth = exact_tabular_truth(mdp, gamma)
    flat_idx = rng.choice(mdp.n_state_actions, size=int(sample_size), p=truth.nu)
    states_idx = flat_idx // mdp.n_actions
    actions_idx = flat_idx % mdp.n_actions
    next_states_idx = sample_next_states(mdp, states_idx, actions_idx, rng)
    target_actions_idx = sample_policy_actions(mdp.target_policy, states_idx, rng)
    next_target_actions_idx = sample_policy_actions(mdp.target_policy, next_states_idx, rng)

    all_states_idx = np.repeat(np.arange(mdp.n_states, dtype=np.int64), mdp.n_actions)
    all_actions_idx = np.tile(np.arange(mdp.n_actions, dtype=np.int64), mdp.n_states)
    reward_rng = np.random.default_rng(seed + 91_337)
    reward_panel = reward_rng.uniform(-1.0, 1.0, size=(int(n_reward_sweeps), mdp.n_state_actions))

    return FiniteDataset(
        mdp=mdp,
        truth=truth,
        states_idx=states_idx,
        actions_idx=actions_idx,
        next_states_idx=next_states_idx,
        target_actions_idx=target_actions_idx,
        next_target_actions_idx=next_target_actions_idx,
        rewards=mdp.rewards[states_idx, actions_idx],
        states=one_hot(states_idx, mdp.n_states),
        actions=one_hot(actions_idx, mdp.n_actions),
        next_states=one_hot(next_states_idx, mdp.n_states),
        target_actions=one_hot(target_actions_idx, mdp.n_actions),
        next_target_actions=one_hot(next_target_actions_idx, mdp.n_actions),
        all_states=one_hot(all_states_idx, mdp.n_states),
        all_actions=one_hot(all_actions_idx, mdp.n_actions),
        all_state_action_indices=state_action_indices(all_states_idx, all_actions_idx, mdp.n_actions),
        reward_panel=reward_panel,
        seed=int(seed),
        sample_size=int(sample_size),
    )


def target_state_action_kernel(mdp: FiniteMDP) -> Array:
    """Build P_pi[(s,a),(s',a')] = P(s'|s,a) pi(a'|s')."""

    kernel = np.zeros((mdp.n_state_actions, mdp.n_state_actions), dtype=np.float64)
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            row = s * mdp.n_actions + a
            for sp in range(mdp.n_states):
                start = sp * mdp.n_actions
                stop = start + mdp.n_actions
                kernel[row, start:stop] = mdp.transition[s, a, sp] * mdp.target_policy[sp]
    return kernel


def sample_policy_actions(policy: Array, states: Array, rng: np.random.Generator) -> Array:
    probs = np.asarray(policy, dtype=np.float64)[np.asarray(states, dtype=np.int64).reshape(-1)]
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(size=probs.shape[0])[:, None]
    return (draws > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def sample_next_states(mdp: FiniteMDP, states: Array, actions: Array, rng: np.random.Generator) -> Array:
    probs = mdp.transition[np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64)]
    cdf = np.cumsum(probs, axis=1)
    draws = rng.random(size=probs.shape[0])[:, None]
    return (draws > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def one_hot(indices: Array, size: int) -> Array:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(size)), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def state_action_indices(states: Array, actions: Array, n_actions: int) -> Array:
    return np.asarray(states, dtype=np.int64).reshape(-1) * int(n_actions) + np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(-1)


def softmax(logits: Array, axis: int = -1) -> Array:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def floor_probs(probs: Array, floor: float) -> Array:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        out = np.maximum(p, float(floor))
        return out / np.sum(out)
    out = np.maximum(p, float(floor))
    return out / np.sum(out, axis=1, keepdims=True)


def safe_divide(numerator: Array, denominator: Array, eps: float = 1e-12) -> Array:
    return np.asarray(numerator, dtype=np.float64) / np.maximum(np.asarray(denominator, dtype=np.float64), eps)
