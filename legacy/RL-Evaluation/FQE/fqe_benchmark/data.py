from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fqe_benchmark.types import BenchmarkConfig, BenchmarkDataset


Array = np.ndarray


def make_datasets(config: BenchmarkConfig) -> list[BenchmarkDataset]:
    datasets: list[BenchmarkDataset] = []
    for dataset_name in config.datasets:
        if dataset_name == "hopper_medium":
            if config.include_hopper:
                for seed in config.seeds:
                    datasets.append(make_hopper_placeholder(seed=int(seed), config=config))
            continue
        for sample_size in config.sample_sizes:
            for gamma in config.gammas:
                for seed in config.seeds:
                    shifts = config.policy_shifts if dataset_name == "linear_gaussian" else (0.0,)
                    for shift in shifts:
                        datasets.append(
                            make_dataset(
                                name=dataset_name,
                                sample_size=int(sample_size),
                                gamma=float(gamma),
                                seed=int(seed),
                                policy_shift=float(shift),
                                n_eval=int(config.n_eval),
                                n_initial_eval=int(config.n_initial_eval),
                            )
                        )
    return datasets


def make_dataset(
    *,
    name: str,
    sample_size: int,
    gamma: float,
    seed: int,
    policy_shift: float = 0.0,
    n_eval: int = 256,
    n_initial_eval: int = 128,
) -> BenchmarkDataset:
    if name == "tabular_chain":
        return make_tabular_chain(sample_size=sample_size, gamma=gamma, seed=seed, n_eval=n_eval, n_initial_eval=n_initial_eval)
    if name == "tabular_grid":
        return make_tabular_grid(sample_size=sample_size, gamma=gamma, seed=seed, n_eval=n_eval, n_initial_eval=n_initial_eval)
    if name == "linear_gaussian":
        return make_linear_gaussian(
            sample_size=sample_size,
            gamma=gamma,
            seed=seed,
            policy_shift=policy_shift,
            n_eval=n_eval,
            n_initial_eval=n_initial_eval,
        )
    raise ValueError(f"Unknown benchmark dataset '{name}'.")


def make_hopper_placeholder(*, seed: int, config: BenchmarkConfig) -> BenchmarkDataset:
    # The real Hopper benchmark is delegated to hopper_fqe_benchmark, whose
    # artifacts and external TensorFlow/Deep OPE dependencies are optional. This
    # placeholder lets estimator preflight produce explicit rows in full-stage
    # manifests without pretending synthetic arrays are Hopper data.
    zeros_s = np.zeros((2, 1), dtype=np.float64)
    zeros_a = np.zeros((2, 1), dtype=np.float64)
    return BenchmarkDataset(
        name="hopper_medium",
        domain="offline_hopper",
        states=zeros_s,
        actions=zeros_a,
        next_states=zeros_s,
        next_actions=zeros_a,
        rewards=np.zeros(2, dtype=np.float64),
        terminals=np.ones(2, dtype=np.float64),
        gamma=0.99,
        seed=seed,
        initial_states=zeros_s,
        initial_actions=zeros_a,
        target_eval_states=zeros_s,
        target_eval_actions=zeros_a,
        behavior_eval_states=zeros_s,
        behavior_eval_actions=zeros_a,
        true_q_fn=None,
        true_policy_value=None,
        metadata={
            "sample_size": 0,
            "offline_artifact_dir": str(config.hopper_artifact_dir),
            "note": "placeholder for optional hopper_fqe_benchmark/Deep OPE preflight",
        },
    )


@dataclass(frozen=True)
class _TabularMDP:
    n_states: int
    n_actions: int
    transition: Array
    rewards: Array
    behavior_policy: Array
    target_policy: Array
    initial_dist: Array


def make_tabular_chain(*, sample_size: int, gamma: float, seed: int, n_eval: int, n_initial_eval: int) -> BenchmarkDataset:
    n_states = 5
    n_actions = 2
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        transition[s, 0, max(0, s - 1)] = 1.0
        transition[s, 1, min(n_states - 1, s + 1)] = 1.0
    rewards = -0.1 * np.arange(n_states, dtype=np.float64)[:, None] + np.array([0.0, 1.0])[None, :]
    behavior = np.tile(np.array([0.65, 0.35]), (n_states, 1))
    target = np.tile(np.array([0.20, 0.80]), (n_states, 1))
    initial = np.zeros(n_states, dtype=np.float64)
    initial[0] = 1.0
    return _sample_tabular(
        name="tabular_chain",
        mdp=_TabularMDP(n_states, n_actions, transition, rewards, behavior, target, initial),
        sample_size=sample_size,
        gamma=gamma,
        seed=seed,
        n_eval=n_eval,
        n_initial_eval=n_initial_eval,
    )


def make_tabular_grid(*, sample_size: int, gamma: float, seed: int, n_eval: int, n_initial_eval: int) -> BenchmarkDataset:
    width = 3
    n_states = width * width
    n_actions = 4
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        r, c = divmod(s, width)
        destinations = (
            (max(0, r - 1), c),
            (min(width - 1, r + 1), c),
            (r, max(0, c - 1)),
            (r, min(width - 1, c + 1)),
        )
        for a, (rr, cc) in enumerate(destinations):
            transition[s, a, rr * width + cc] = 1.0
    goal = n_states - 1
    rewards = -0.05 * np.ones((n_states, n_actions), dtype=np.float64)
    rewards[goal, :] = 1.0
    behavior = np.full((n_states, n_actions), 1.0 / n_actions)
    target = np.full((n_states, n_actions), 0.10 / (n_actions - 1))
    for s in range(n_states):
        r, c = divmod(s, width)
        target[s, 1 if r < width - 1 else 3] = 0.90 if r < width - 1 else 0.90
        if r == width - 1 and c < width - 1:
            target[s, :] = 0.10 / (n_actions - 1)
            target[s, 3] = 0.90
    initial = np.zeros(n_states, dtype=np.float64)
    initial[0] = 1.0
    return _sample_tabular(
        name="tabular_grid",
        mdp=_TabularMDP(n_states, n_actions, transition, rewards, behavior, target, initial),
        sample_size=sample_size,
        gamma=gamma,
        seed=seed,
        n_eval=n_eval,
        n_initial_eval=n_initial_eval,
    )


def _sample_tabular(
    *,
    name: str,
    mdp: _TabularMDP,
    sample_size: int,
    gamma: float,
    seed: int,
    n_eval: int,
    n_initial_eval: int,
) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    q_truth = _solve_tabular_q(mdp, gamma)
    states_idx = rng.choice(mdp.n_states, size=sample_size, p=_stationary_distribution(mdp.transition, mdp.behavior_policy, mdp.initial_dist, gamma))
    actions_idx = np.array([rng.choice(mdp.n_actions, p=mdp.behavior_policy[s]) for s in states_idx], dtype=np.int64)
    next_idx = np.array([rng.choice(mdp.n_states, p=mdp.transition[s, a]) for s, a in zip(states_idx, actions_idx)], dtype=np.int64)
    next_actions_idx = np.array([rng.choice(mdp.n_actions, p=mdp.target_policy[s]) for s in next_idx], dtype=np.int64)
    rewards = mdp.rewards[states_idx, actions_idx]

    target_states_idx = rng.choice(mdp.n_states, size=n_eval, p=_stationary_distribution(mdp.transition, mdp.target_policy, mdp.initial_dist, gamma))
    target_actions_idx = np.array([rng.choice(mdp.n_actions, p=mdp.target_policy[s]) for s in target_states_idx], dtype=np.int64)
    behavior_states_idx = rng.choice(mdp.n_states, size=n_eval, p=_stationary_distribution(mdp.transition, mdp.behavior_policy, mdp.initial_dist, gamma))
    behavior_actions_idx = np.array([rng.choice(mdp.n_actions, p=mdp.behavior_policy[s]) for s in behavior_states_idx], dtype=np.int64)
    initial_states_idx = rng.choice(mdp.n_states, size=n_initial_eval, p=mdp.initial_dist)
    initial_actions_idx = np.array([rng.choice(mdp.n_actions, p=mdp.target_policy[s]) for s in initial_states_idx], dtype=np.int64)

    def encode_states(idx: Array) -> Array:
        out = np.zeros((idx.shape[0], mdp.n_states), dtype=np.float64)
        out[np.arange(idx.shape[0]), idx.astype(np.int64)] = 1.0
        return out

    def encode_actions(idx: Array) -> Array:
        out = np.zeros((idx.shape[0], mdp.n_actions), dtype=np.float64)
        out[np.arange(idx.shape[0]), idx.astype(np.int64)] = 1.0
        return out

    def true_q_fn(states: Array, actions: Array) -> Array:
        s_idx = np.argmax(np.asarray(states), axis=1)
        a_idx = np.argmax(np.asarray(actions), axis=1)
        return q_truth[s_idx, a_idx]

    policy_value = float(np.sum(mdp.initial_dist[:, None] * mdp.target_policy * q_truth))
    return BenchmarkDataset(
        name=name,
        domain="tabular",
        states=encode_states(states_idx),
        actions=encode_actions(actions_idx),
        next_states=encode_states(next_idx),
        next_actions=encode_actions(next_actions_idx),
        rewards=rewards,
        terminals=np.zeros(sample_size, dtype=np.float64),
        gamma=gamma,
        seed=seed,
        initial_states=encode_states(initial_states_idx),
        initial_actions=encode_actions(initial_actions_idx),
        target_eval_states=encode_states(target_states_idx),
        target_eval_actions=encode_actions(target_actions_idx),
        behavior_eval_states=encode_states(behavior_states_idx),
        behavior_eval_actions=encode_actions(behavior_actions_idx),
        true_q_fn=true_q_fn,
        true_policy_value=policy_value,
        metadata={"n_states": mdp.n_states, "n_actions": mdp.n_actions, "sample_size": sample_size},
    )


def _solve_tabular_q(mdp: _TabularMDP, gamma: float) -> Array:
    n_sa = mdp.n_states * mdp.n_actions
    p_pi = np.zeros((n_sa, n_sa), dtype=np.float64)
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            row = s * mdp.n_actions + a
            for sp in range(mdp.n_states):
                for ap in range(mdp.n_actions):
                    col = sp * mdp.n_actions + ap
                    p_pi[row, col] += mdp.transition[s, a, sp] * mdp.target_policy[sp, ap]
    rhs = mdp.rewards.reshape(-1)
    q = np.linalg.solve(np.eye(n_sa) - gamma * p_pi, rhs)
    return q.reshape(mdp.n_states, mdp.n_actions)


def _stationary_distribution(transition: Array, policy: Array, initial_dist: Array, gamma: float) -> Array:
    p_pi = np.einsum("sa,san->sn", policy, transition)
    if gamma == 0.0:
        dist = initial_dist.copy()
    else:
        dist = (1.0 - gamma) * initial_dist @ np.linalg.inv(np.eye(initial_dist.shape[0]) - gamma * p_pi)
    dist = np.maximum(np.asarray(dist, dtype=np.float64).reshape(-1), 0.0)
    return dist / np.sum(dist)


def make_linear_gaussian(
    *,
    sample_size: int,
    gamma: float,
    seed: int,
    policy_shift: float,
    n_eval: int,
    n_initial_eval: int,
) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    b = np.array([[0.75, 0.10], [0.0, 0.65]], dtype=np.float64)
    c = np.array([[0.35], [0.20]], dtype=np.float64)
    q_mat = np.diag([0.4, 0.2])
    r_action = 0.15
    target_k = np.array([[-0.7, 0.25]], dtype=np.float64)
    behavior_k = target_k + np.array([[policy_shift, -0.25 * policy_shift]], dtype=np.float64)
    behavior_sd = 0.4
    target_sd = 0.25
    noise_sd = 0.15
    initial_mean = np.array([0.5, -0.5], dtype=np.float64)
    initial_cov = np.diag([0.8, 0.6])
    states = rng.multivariate_normal(initial_mean, initial_cov + np.eye(2), size=sample_size)
    actions = states @ behavior_k.T + rng.normal(scale=behavior_sd, size=(sample_size, 1))
    next_states = states @ b.T + actions @ c.T + rng.normal(scale=noise_sd, size=(sample_size, 2))
    next_actions = next_states @ target_k.T + rng.normal(scale=target_sd, size=(sample_size, 1))
    rewards = _linear_gaussian_reward(states, actions, q_mat, r_action)

    target_eval_states = rng.multivariate_normal(initial_mean, initial_cov + 2.0 * np.eye(2), size=n_eval)
    target_eval_actions = target_eval_states @ target_k.T + rng.normal(scale=target_sd, size=(n_eval, 1))
    behavior_eval_states = rng.multivariate_normal(initial_mean, initial_cov + 2.0 * np.eye(2), size=n_eval)
    behavior_eval_actions = behavior_eval_states @ behavior_k.T + rng.normal(scale=behavior_sd, size=(n_eval, 1))
    initial_states = rng.multivariate_normal(initial_mean, initial_cov, size=n_initial_eval)
    initial_actions = initial_states @ target_k.T + rng.normal(scale=target_sd, size=(n_initial_eval, 1))
    p, linear, const = _solve_linear_gaussian_q(
        gamma=gamma,
        b=b,
        c=c,
        q_mat=q_mat,
        r_action=r_action,
        target_k=target_k,
        target_sd=target_sd,
        noise_sd=noise_sd,
    )

    def true_q_fn(s: Array, a: Array) -> Array:
        x = np.concatenate([np.asarray(s, dtype=np.float64).reshape(-1, 2), np.asarray(a, dtype=np.float64).reshape(-1, 1)], axis=1)
        return np.einsum("ni,ij,nj->n", x, p, x) + x @ linear + const

    policy_value = float(np.mean(true_q_fn(initial_states, initial_actions)))
    return BenchmarkDataset(
        name="linear_gaussian",
        domain="controlled_synthetic",
        states=states,
        actions=actions,
        next_states=next_states,
        next_actions=next_actions,
        rewards=rewards,
        terminals=np.zeros(sample_size, dtype=np.float64),
        gamma=gamma,
        seed=seed,
        initial_states=initial_states,
        initial_actions=initial_actions,
        target_eval_states=target_eval_states,
        target_eval_actions=target_eval_actions,
        behavior_eval_states=behavior_eval_states,
        behavior_eval_actions=behavior_eval_actions,
        true_q_fn=true_q_fn,
        true_policy_value=policy_value,
        metadata={"sample_size": sample_size, "policy_shift": policy_shift, "target_action_sd": target_sd, "behavior_action_sd": behavior_sd},
    )


def _linear_gaussian_reward(states: Array, actions: Array, q_mat: Array, r_action: float) -> Array:
    return -(np.einsum("ni,ij,nj->n", states, q_mat, states) + r_action * actions.reshape(-1) ** 2)


def _solve_linear_gaussian_q(
    *,
    gamma: float,
    b: Array,
    c: Array,
    q_mat: Array,
    r_action: float,
    target_k: Array,
    target_sd: float,
    noise_sd: float,
) -> tuple[Array, Array, float]:
    # Fixed-point iteration over a quadratic Q form on x=(s,a). This is exact
    # for the linear-Gaussian/quadratic reward benchmark up to convergence.
    p = np.zeros((3, 3), dtype=np.float64)
    linear = np.zeros(3, dtype=np.float64)
    const = 0.0
    reward_p = np.zeros((3, 3), dtype=np.float64)
    reward_p[:2, :2] = -q_mat
    reward_p[2, 2] = -float(r_action)
    trans = np.concatenate([b, c], axis=1)
    action_from_next = target_k
    mean_map = np.vstack([trans, action_from_next @ trans])
    noise_cov = noise_sd**2 * np.eye(2)
    action_noise_var = float(target_sd**2 + (action_from_next @ noise_cov @ action_from_next.T).item())
    noise_full_cov = np.zeros((3, 3), dtype=np.float64)
    noise_full_cov[:2, :2] = noise_cov
    noise_full_cov[2, 2] = action_noise_var
    noise_full_cov[:2, 2:3] = noise_cov @ action_from_next.T
    noise_full_cov[2:3, :2] = action_from_next @ noise_cov
    for _ in range(500):
        p_next = reward_p + gamma * (mean_map.T @ p @ mean_map)
        linear_next = gamma * (mean_map.T @ linear)
        const_next = gamma * (const + float(np.trace(p @ noise_full_cov)))
        if np.max(np.abs(p_next - p)) < 1e-10 and abs(const_next - const) < 1e-10:
            p, linear, const = p_next, linear_next, const_next
            break
        p, linear, const = p_next, linear_next, const_next
    return p, linear, float(const)
