from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fqe import (
    DirectMultiOutputSBVValidator,
    FQECandidate,
    GenerativeBellmanValidator,
    LowRankOperatorSBVValidator,
    TransitionDataset,
    compute_candidate_next_value_matrix,
    expected_q_under_policy,
    select_td_with_sbv_audit,
    split_by_episode_ids,
)
from fqe.sbv import _truncated_vt


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")


class TabularQ:
    def __init__(self, q_table: np.ndarray) -> None:
        self.q_table = np.asarray(q_table, dtype=np.float64)

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        s_idx = np.argmax(np.asarray(states), axis=1)
        a_idx = np.argmax(np.asarray(actions), axis=1)
        return self.q_table[s_idx, a_idx]

    def predict_all_actions(self, states: np.ndarray) -> np.ndarray:
        s_idx = np.argmax(np.asarray(states), axis=1)
        return self.q_table[s_idx]


def _one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx.astype(int)] = 1.0
    return out


def _toy_dataset(n_episodes: int = 30, horizon: int = 4, seed: int = 0) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(n_episodes):
        state = int(rng.integers(0, 3))
        for timestep in range(horizon):
            action = int(rng.integers(0, 2))
            next_state = min(2, max(0, state + (1 if action else -1)))
            reward = float(next_state == 2) - 0.1 * state
            done = timestep == horizon - 1
            rows.append((state, action, reward, next_state, done, episode, timestep))
            state = next_state
    states = _one_hot(np.asarray([row[0] for row in rows]), 3)
    actions = _one_hot(np.asarray([row[1] for row in rows]), 2)
    next_states = _one_hot(np.asarray([row[3] for row in rows]), 3)
    return TransitionDataset(
        states,
        actions,
        np.asarray([row[2] for row in rows], dtype=np.float64),
        next_states,
        np.asarray([row[4] for row in rows], dtype=np.float64),
        np.asarray([row[5] for row in rows]),
        np.asarray([row[6] for row in rows]),
    )


def _split_three(dataset: TransitionDataset) -> tuple[TransitionDataset, TransitionDataset, TransitionDataset]:
    splits = split_by_episode_ids(dataset, {"D_B_train": 0.55, "D_B_val": 0.20, "D_score": 0.25}, seed=19)
    return splits["D_B_train"], splits["D_B_val"], splits["D_score"]


def test_split_by_episode_ids_has_no_overlap() -> None:
    dataset = _toy_dataset(n_episodes=15, horizon=3)
    splits = split_by_episode_ids(dataset, {"D_Q": 0.4, "D_B": 0.3, "D_score": 0.3}, seed=7)
    episode_sets = {name: set(split.episode_id.tolist()) for name, split in splits.items()}
    assert episode_sets["D_Q"].isdisjoint(episode_sets["D_B"])
    assert episode_sets["D_Q"].isdisjoint(episode_sets["D_score"])
    assert episode_sets["D_B"].isdisjoint(episode_sets["D_score"])
    assert sum(split.n for split in splits.values()) == dataset.n


def test_terminal_handling_zeroes_next_value_targets() -> None:
    dataset = _toy_dataset(n_episodes=5, horizon=2)
    terminal_dataset = TransitionDataset(
        dataset.obs,
        dataset.actions,
        dataset.rewards,
        dataset.next_obs,
        np.ones(dataset.n),
        dataset.episode_id,
        dataset.timestep,
    )
    candidates = [FQECandidate("q", TabularQ(np.asarray([[100.0, -50.0], [10.0, 20.0], [5.0, 8.0]])))]
    H = compute_candidate_next_value_matrix(candidates, terminal_dataset, np.asarray([[0.5, 0.5]] * 3), np.eye(2))
    assert np.allclose(H, 0.0)


def test_transition_dataset_rejects_zero_total_weights() -> None:
    dataset = _toy_dataset(n_episodes=2, horizon=2)
    with pytest.raises(ValueError, match="positive total"):
        TransitionDataset(
            dataset.obs,
            dataset.actions,
            dataset.rewards,
            dataset.next_obs,
            dataset.done,
            dataset.episode_id,
            dataset.timestep,
            sample_weight=np.zeros(dataset.n),
        )


def test_discrete_expected_q_under_policy_exact() -> None:
    q = TabularQ(np.asarray([[1.0, 3.0], [2.0, 6.0]]))
    states = _one_hot(np.asarray([0, 1]), 2)
    policy = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    expected = expected_q_under_policy(q, states, policy, np.eye(2))
    assert np.allclose(expected, np.asarray([2.5, 3.6]))


def test_low_rank_svd_reconstructs_known_rank_matrix() -> None:
    rng = np.random.default_rng(4)
    left = rng.normal(size=(30, 2))
    right = rng.normal(size=(2, 7))
    H = left @ right
    H_mean = np.mean(H, axis=0, keepdims=True)
    Hc = H - H_mean
    Vt, _ = _truncated_vt(Hc, max_rank=2, backend="numpy", seed=1)
    reconstructed = H_mean + (Hc @ Vt.T) @ Vt
    assert np.mean((H - reconstructed) ** 2) < 1e-10


def test_td_with_sbv_audit_selects_td_and_flags_disagreement() -> None:
    candidates = [
        FQECandidate("small", object(), complexity_order_key=0),
        FQECandidate("large", object(), complexity_order_key=1),
    ]
    rows = [
        {"candidate_id": "small", "naive_td_score": 1.0, "naive_td_score_se": 0.0, "sbv_score": 10.0, "sbv_score_se": 0.0},
        {"candidate_id": "large", "naive_td_score": 0.5, "naive_td_score_se": 0.0, "sbv_score": 1.0, "sbv_score_se": 0.0},
    ]
    result = select_td_with_sbv_audit(rows, candidates, use_one_se=False)
    assert result.selected_candidate_id == "large"
    assert result.diagnostics["td_sbv_audit_status"] == "green"

    disagree_rows = [
        {"candidate_id": "small", "naive_td_score": 0.5, "naive_td_score_se": 0.0, "sbv_score": 10.0, "sbv_score_se": 0.0},
        {"candidate_id": "large", "naive_td_score": 1.0, "naive_td_score_se": 0.0, "sbv_score": 1.0, "sbv_score_se": 0.0},
    ]
    result = select_td_with_sbv_audit(disagree_rows, candidates, use_one_se=False)
    assert result.selected_candidate_id == "small"
    assert result.diagnostics["td_sbv_audit_status"] == "red"
    assert result.rows[0]["td_sbv_audit_recommendation"] == "select_td"


@pytestmark_torch
def test_low_rank_rank_ge_m_matches_direct_sbv_predictions() -> None:
    dataset = _toy_dataset(n_episodes=50, horizon=3, seed=11)
    train, val, score = _split_three(dataset)
    policy = np.asarray([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])
    candidates = [
        FQECandidate("q0", TabularQ(np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]))),
        FQECandidate("q1", TabularQ(np.asarray([[0.5, 0.8], [0.7, 1.5], [1.1, 2.2]]))),
    ]
    kwargs = dict(hidden_sizes=(32,), lr=5e-3, batch_size=32, max_epochs=35, patience=6, seed=5, n_bootstrap=5)
    lowrank = LowRankOperatorSBVValidator(0.8, ranks=[2], **kwargs).fit(candidates, train, val, policy, np.eye(2))
    direct = DirectMultiOutputSBVValidator(0.8, direct_threshold=4, **kwargs).fit(candidates, train, val, policy, np.eye(2))
    lr_pred = lowrank.predict_backup_matrix(score)
    direct_pred = direct.predict_backup_matrix(score)
    assert np.mean((lr_pred - direct_pred) ** 2) < 2.0


@pytestmark_torch
def test_tabular_mdp_low_rank_sbv_selects_true_or_small_noise_candidate() -> None:
    gamma = 0.7
    dataset = _deterministic_tabular_validation_dataset(repeats=45)
    train, val, score = _split_three(dataset)
    target_policy = np.asarray([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
    q_true = _solve_small_chain_q(gamma, target_policy)
    candidates = [
        FQECandidate("true", TabularQ(q_true), complexity_order_key=0),
        FQECandidate("small_noise", TabularQ(q_true + 0.08), complexity_order_key=1),
        FQECandidate("large_noise", TabularQ(q_true + 1.5), complexity_order_key=2),
        FQECandidate("bad", TabularQ(np.zeros_like(q_true)), complexity_order_key=3),
    ]
    validator = LowRankOperatorSBVValidator(
        gamma,
        ranks=[2, 4],
        hidden_sizes=(64,),
        lr=8e-3,
        batch_size=64,
        max_epochs=80,
        patience=10,
        seed=12,
        n_bootstrap=10,
    )
    result = validator.fit_score(candidates, train, val, score, target_policy, np.eye(2))
    assert result.selected_candidate_id in {"true", "small_noise"}
    analytic = np.asarray([_analytic_msbe(candidate.model.q_table, gamma, target_policy) for candidate in candidates])
    scores = np.asarray([row["sbv_score"] for row in result.rows])
    assert _spearman(analytic, scores) > 0.75


@pytestmark_torch
def test_stochastic_transition_sbv_beats_naive_td_rank_correlation() -> None:
    gamma = 0.9
    target_policy = np.asarray([[1.0], [1.0], [1.0]])
    action_space = np.ones((1, 1), dtype=np.float64)
    correlations_sbv = []
    correlations_naive = []
    for seed in range(3):
        dataset = _stochastic_next_state_dataset(seed=seed)
        train, val, score = _split_three(dataset)
        q_tables = [
            np.asarray([[4.5], [0.0], [10.0]]),
            np.asarray([[3.6], [5.0], [5.0]]),
            np.asarray([[0.0], [0.0], [0.0]]),
        ]
        candidates = [FQECandidate(f"q{idx}", TabularQ(q), complexity_order_key=idx) for idx, q in enumerate(q_tables)]
        validator = LowRankOperatorSBVValidator(
            gamma,
            ranks=[2, 3],
            hidden_sizes=(32,),
            lr=8e-3,
            batch_size=64,
            max_epochs=45,
            patience=8,
            seed=31 + seed,
            n_bootstrap=5,
        )
        result = validator.fit_score(candidates, train, val, score, target_policy, action_space)
        analytic = np.asarray([0.0, (3.6 - 4.5) ** 2, 4.5**2])
        sbv = np.asarray([row["sbv_score"] for row in result.rows])
        naive = np.asarray([row["naive_td_score"] for row in result.rows])
        correlations_sbv.append(_spearman(analytic, sbv))
        correlations_naive.append(_spearman(analytic, naive))
    assert np.mean(correlations_sbv) > np.mean(correlations_naive) + 0.25


@pytestmark_torch
def test_generative_baseline_nll_decreases_and_scores_candidates() -> None:
    dataset = _linear_gaussian_dataset(seed=3)
    train, val, score = _split_three(dataset)

    class LinearQ:
        def __init__(self, bias: float) -> None:
            self.bias = float(bias)

        def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            return states[:, 0] + 0.5 * actions[:, 0] + self.bias

    class DeterministicPolicy:
        def mean_actions(self, states: np.ndarray) -> np.ndarray:
            return 0.25 * states[:, [0]]

    candidates = [FQECandidate("a", LinearQ(0.0)), FQECandidate("b", LinearQ(1.0))]
    validator = GenerativeBellmanValidator(
        0.5,
        hidden_sizes=(32,),
        lr=5e-3,
        batch_size=64,
        max_epochs=30,
        patience=6,
        n_model_samples=4,
        seed=44,
        n_bootstrap=5,
    )
    result = validator.fit_score(candidates, train, val, score, DeterministicPolicy(), None)
    assert validator.diagnostics_["validation_nll"] < validator.diagnostics_["initial_validation_nll"]
    assert len(result.rows) == 2
    assert all(np.isfinite(row["generative_score_mc"]) for row in result.rows)


@pytestmark_torch
def test_low_rank_does_not_train_one_regressor_per_candidate() -> None:
    dataset = _toy_dataset(n_episodes=30, horizon=2, seed=8)
    train, val, _score = _split_three(dataset)
    policy = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    rng = np.random.default_rng(9)
    candidates = [
        FQECandidate(f"q{idx}", TabularQ(rng.normal(size=(3, 2))), complexity_order_key=idx)
        for idx in range(200)
    ]
    validator = LowRankOperatorSBVValidator(
        0.6,
        ranks=8,
        hidden_sizes=(16,),
        max_epochs=1,
        patience=1,
        batch_size=64,
        seed=55,
    )
    validator.fit(candidates, train, val, policy, np.eye(2))
    assert validator.trained_operator_model_count == 1
    assert validator.diagnostics_["operator_model_count"] == 1


def _deterministic_tabular_validation_dataset(repeats: int) -> TransitionDataset:
    rows = []
    episode = 0
    for _ in range(repeats):
        for state in range(3):
            for action in range(2):
                next_state = min(2, max(0, state + (1 if action else -1)))
                reward = float(state == 2) + 0.2 * action
                rows.append((state, action, reward, next_state, False, episode, 0))
                episode += 1
    return TransitionDataset(
        _one_hot(np.asarray([row[0] for row in rows]), 3),
        _one_hot(np.asarray([row[1] for row in rows]), 2),
        np.asarray([row[2] for row in rows], dtype=np.float64),
        _one_hot(np.asarray([row[3] for row in rows]), 3),
        np.zeros(len(rows), dtype=np.float64),
        np.asarray([row[5] for row in rows]),
        np.asarray([row[6] for row in rows]),
    )


def _solve_small_chain_q(gamma: float, target_policy: np.ndarray) -> np.ndarray:
    n_states, n_actions = 3, 2
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    rewards = np.zeros((n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            next_state = min(2, max(0, state + (1 if action else -1)))
            transition[state, action, next_state] = 1.0
            rewards[state, action] = float(state == 2) + 0.2 * action
    p_pi = np.zeros((n_states * n_actions, n_states * n_actions), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            row = state * n_actions + action
            for next_state in range(n_states):
                for next_action in range(n_actions):
                    col = next_state * n_actions + next_action
                    p_pi[row, col] += transition[state, action, next_state] * target_policy[next_state, next_action]
    q = np.linalg.solve(np.eye(n_states * n_actions) - gamma * p_pi, rewards.reshape(-1))
    return q.reshape(n_states, n_actions)


def _analytic_msbe(q_table: np.ndarray, gamma: float, target_policy: np.ndarray) -> float:
    errors = []
    for state in range(3):
        for action in range(2):
            next_state = min(2, max(0, state + (1 if action else -1)))
            reward = float(state == 2) + 0.2 * action
            next_v = float(np.sum(target_policy[next_state] * q_table[next_state]))
            errors.append((q_table[state, action] - reward - gamma * next_v) ** 2)
    return float(np.mean(errors))


def _stochastic_next_state_dataset(seed: int) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(160):
        next_state = int(rng.integers(1, 3))
        rows.append((0, 0, 0.0, next_state, False, episode, 0))
    return TransitionDataset(
        _one_hot(np.asarray([row[0] for row in rows]), 3),
        np.ones((len(rows), 1), dtype=np.float64),
        np.zeros(len(rows), dtype=np.float64),
        _one_hot(np.asarray([row[3] for row in rows]), 3),
        np.zeros(len(rows), dtype=np.float64),
        np.asarray([row[5] for row in rows]),
        np.asarray([row[6] for row in rows]),
    )


def _linear_gaussian_dataset(seed: int) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    rows_per_episode = 3
    n_episodes = 80
    n = rows_per_episode * n_episodes
    states = rng.normal(size=(n, 2))
    actions = 0.3 * states[:, [0]] + rng.normal(scale=0.2, size=(n, 1))
    next_states = states + np.concatenate([actions, -0.5 * actions], axis=1) + rng.normal(scale=0.05, size=(n, 2))
    rewards = states[:, 0] + 0.2 * actions[:, 0] + rng.normal(scale=0.05, size=n)
    done = np.zeros(n, dtype=np.float64)
    done[rows_per_episode - 1 :: rows_per_episode] = 1.0
    episode = np.repeat(np.arange(n_episodes), rows_per_episode)
    timestep = np.tile(np.arange(rows_per_episode), n_episodes)
    return TransitionDataset(states, actions, rewards, next_states, done, episode, timestep)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    yr = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])
