from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from fqe import (
    DirectMultiOutputSBVValidator,
    FQECandidate,
    GenerativeBellmanValidator,
    LowRankOperatorSBVValidator,
    TransitionDataset,
    split_by_episode_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ground-truth SBV model-selection experiments.")
    parser.add_argument("--output-dir", default="outputs/fqe_sbv_experiment")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--n-episodes", type=int, default=900)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--generative-epochs", type=int, default=45)
    parser.add_argument("--include-generative", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use a shorter smoke configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    n_episodes = int(args.n_episodes)
    max_epochs = int(args.max_epochs)
    generative_epochs = int(args.generative_epochs)
    if args.quick:
        seeds = tuple(seeds[:3])
        n_episodes = min(n_episodes, 360)
        max_epochs = min(max_epochs, 25)
        generative_epochs = min(generative_epochs, 20)
    result = run_sbv_model_selection_experiment(
        output_dir=Path(args.output_dir),
        seeds=seeds,
        n_episodes=n_episodes,
        max_epochs=max_epochs,
        generative_epochs=generative_epochs,
        include_generative=bool(args.include_generative),
    )
    print(f"Wrote per-seed rows: {result['rows_path']}")
    print(f"Wrote summary: {result['summary_path']}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def run_sbv_model_selection_experiment(
    *,
    output_dir: Path,
    seeds: tuple[int, ...] = tuple(range(10)),
    n_episodes: int = 900,
    max_epochs: int = 60,
    generative_epochs: int = 45,
    include_generative: bool = False,
) -> dict[str, Any]:
    _stabilize_torch_threads()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for setting_name, setting_builder in (
        ("stochastic_lottery", _make_lottery_problem),
        ("stochastic_chain", _make_chain_problem),
        ("stochastic_lottery_many", _make_many_candidate_lottery_problem),
    ):
        for seed in seeds:
            problem = setting_builder(seed=int(seed), n_episodes=int(n_episodes))
            rows.extend(
                _run_one_problem(
                    setting_name=setting_name,
                    seed=int(seed),
                    problem=problem,
                    max_epochs=int(max_epochs),
                    generative_epochs=int(generative_epochs),
                    include_generative=bool(include_generative),
                )
            )
    summary = _summarize_rows(rows)
    rows_path = output_dir / "sbv_model_selection_rows.csv"
    summary_path = output_dir / "sbv_model_selection_summary.json"
    _write_csv(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {"rows_path": rows_path, "summary_path": summary_path, "rows": rows, "summary": summary}


def _run_one_problem(
    *,
    setting_name: str,
    seed: int,
    problem: dict[str, Any],
    max_epochs: int,
    generative_epochs: int,
    include_generative: bool,
) -> list[dict[str, Any]]:
    dataset: TransitionDataset = problem["dataset"]
    candidates: list[FQECandidate] = problem["candidates"]
    target_policy = problem["target_policy"]
    action_space = problem["action_space"]
    analytic_msbe = np.asarray(problem["analytic_msbe"], dtype=np.float64)
    oracle_idx = int(np.argmin(analytic_msbe))
    splits = split_by_episode_ids(dataset, {"D_B": 0.65, "D_score": 0.35}, seed=seed + 10_001)
    b_splits = split_by_episode_ids(splits["D_B"], {"D_B_train": 0.80, "D_B_val": 0.20}, seed=seed + 20_001)
    rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    lowrank = LowRankOperatorSBVValidator(
        gamma=problem["gamma"],
        ranks=[2, 4, 8],
        hidden_sizes=(64, 64),
        lr=5e-3,
        batch_size=128,
        max_epochs=max_epochs,
        patience=8,
        n_bootstrap=30,
        seed=seed + 30_001,
    )
    lowrank_result = lowrank.fit_score(
        candidates,
        b_splits["D_B_train"],
        b_splits["D_B_val"],
        splits["D_score"],
        target_policy,
        action_space,
    )
    runtime = time.perf_counter() - start
    rows.append(_method_row("low_rank_sbv", setting_name, seed, candidates, analytic_msbe, oracle_idx, lowrank_result.rows, "sbv_score", runtime))
    rows.append(_method_row("naive_td", setting_name, seed, candidates, analytic_msbe, oracle_idx, lowrank_result.rows, "naive_td_score", runtime))

    if len(candidates) <= 64:
        start = time.perf_counter()
        direct = DirectMultiOutputSBVValidator(
            gamma=problem["gamma"],
            hidden_sizes=(64, 64),
            lr=5e-3,
            batch_size=128,
            max_epochs=max_epochs,
            patience=8,
            n_bootstrap=30,
            seed=seed + 40_001,
            direct_threshold=64,
        )
        direct.fit(candidates, b_splits["D_B_train"], b_splits["D_B_val"], target_policy, action_space)
        direct_result = direct.score(candidates, splits["D_score"], target_policy, action_space)
        rows.append(_method_row("direct_sbv", setting_name, seed, candidates, analytic_msbe, oracle_idx, direct_result.rows, "direct_sbv_score", time.perf_counter() - start))

    if include_generative:
        start = time.perf_counter()
        gen = GenerativeBellmanValidator(
            gamma=problem["gamma"],
            hidden_sizes=(64, 64),
            lr=3e-3,
            batch_size=128,
            max_epochs=generative_epochs,
            patience=8,
            n_model_samples=8,
            n_bootstrap=30,
            seed=seed + 50_001,
        )
        gen_result = gen.fit_score(
            candidates,
            b_splits["D_B_train"],
            b_splits["D_B_val"],
            splits["D_score"],
            target_policy,
            action_space,
        )
        rows.append(_method_row("generative_mc", setting_name, seed, candidates, analytic_msbe, oracle_idx, gen_result.rows, "generative_score_mc", time.perf_counter() - start))
    return rows


def _method_row(
    method: str,
    setting: str,
    seed: int,
    candidates: list[FQECandidate],
    analytic_msbe: np.ndarray,
    oracle_idx: int,
    score_rows: list[dict[str, Any]],
    score_key: str,
    runtime_sec: float,
) -> dict[str, Any]:
    scores = np.asarray([float(row[score_key]) for row in score_rows], dtype=np.float64)
    selected_idx = int(np.argmin(scores))
    return {
        "setting": setting,
        "seed": int(seed),
        "method": method,
        "selected_candidate_id": candidates[selected_idx].candidate_id,
        "oracle_candidate_id": candidates[oracle_idx].candidate_id,
        "selected_msbe": float(analytic_msbe[selected_idx]),
        "oracle_msbe": float(analytic_msbe[oracle_idx]),
        "msbe_regret": float(analytic_msbe[selected_idx] - analytic_msbe[oracle_idx]),
        "selected_oracle": bool(selected_idx == oracle_idx),
        "score_msbe_spearman": float(_spearman(scores, analytic_msbe)),
        "score_msbe_pearson": float(_pearson(scores, analytic_msbe)),
        "runtime_sec": float(runtime_sec),
    }


def _make_lottery_problem(*, seed: int, n_episodes: int) -> dict[str, Any]:
    gamma = 0.9
    rng = np.random.default_rng(seed)
    state_probs = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)
    rows = []
    for episode in range(n_episodes):
        state = int(rng.choice(3, p=state_probs))
        if state == 0:
            next_state = int(rng.choice([1, 2]))
            reward = 0.0
            done = False
        elif state == 1:
            next_state = 1
            reward = 10.0
            done = True
        else:
            next_state = 2
            reward = -10.0
            done = True
        rows.append((state, 0, reward, next_state, done, episode, 0))
    dataset = _dataset_from_rows(rows, n_states=3, n_actions=1)
    target_policy = np.ones((3, 1), dtype=np.float64)
    action_space = np.zeros((1, 1), dtype=np.float64)
    q_true = np.asarray([[0.0], [10.0], [-10.0]], dtype=np.float64)
    base_tables = [
        ("true", q_true),
        ("small_noise", q_true + np.asarray([[0.2], [-0.4], [0.4]])),
        ("collapsed", np.zeros_like(q_true)),
        ("constant_mean", np.full_like(q_true, 0.5)),
        ("over_spread", np.asarray([[0.0], [15.0], [-15.0]], dtype=np.float64)),
        ("biased_initial", q_true + np.asarray([[2.5], [0.0], [0.0]])),
    ]
    for idx in range(6):
        noise = rng.normal(scale=1.8 + 0.2 * idx, size=q_true.shape)
        base_tables.append((f"random_{idx}", q_true + noise))
    candidates = [FQECandidate(name, _TabularQModel(table), complexity_order_key=idx) for idx, (name, table) in enumerate(base_tables)]
    analytic_msbe = [_lottery_msbe(candidate.model.q_table, gamma, state_probs) for candidate in candidates]
    return {
        "dataset": dataset,
        "candidates": candidates,
        "gamma": gamma,
        "target_policy": target_policy,
        "action_space": action_space,
        "analytic_msbe": analytic_msbe,
    }


def _make_many_candidate_lottery_problem(*, seed: int, n_episodes: int) -> dict[str, Any]:
    problem = _make_lottery_problem(seed=seed, n_episodes=n_episodes)
    rng = np.random.default_rng(seed + 919)
    q_true = np.asarray([[0.0], [10.0], [-10.0]], dtype=np.float64)
    candidates = [
        FQECandidate("true", _TabularQModel(q_true), complexity_order_key=0),
        FQECandidate("collapsed", _TabularQModel(np.zeros_like(q_true)), complexity_order_key=1),
    ]
    for idx in range(198):
        scale = 0.25 + 0.03 * idx
        noise = rng.normal(scale=scale, size=q_true.shape)
        candidates.append(FQECandidate(f"random_{idx:03d}", _TabularQModel(q_true + noise), complexity_order_key=idx + 2))
    state_probs = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)
    problem["candidates"] = candidates
    problem["analytic_msbe"] = [_lottery_msbe(candidate.model.q_table, float(problem["gamma"]), state_probs) for candidate in candidates]
    return problem


def _make_chain_problem(*, seed: int, n_episodes: int) -> dict[str, Any]:
    gamma = 0.85
    rng = np.random.default_rng(seed)
    target_policy = np.asarray([[0.25, 0.75], [0.35, 0.65], [0.60, 0.40], [0.50, 0.50]], dtype=np.float64)
    behavior_policy = np.asarray([[0.60, 0.40], [0.50, 0.50], [0.50, 0.50], [0.70, 0.30]], dtype=np.float64)
    transition, rewards = _chain_transition_reward()
    rows = []
    for episode in range(n_episodes):
        state = int(rng.choice(4, p=np.asarray([0.45, 0.25, 0.20, 0.10])))
        for timestep in range(3):
            action = int(rng.choice(2, p=behavior_policy[state]))
            next_state = int(rng.choice(4, p=transition[state, action]))
            reward = float(rewards[state, action])
            done = bool(state == 3 and timestep > 0)
            rows.append((state, action, reward, next_state, done, episode, timestep))
            if done:
                break
            state = next_state
    dataset = _dataset_from_rows(rows, n_states=4, n_actions=2)
    q_true = _solve_tabular_q(transition, rewards, target_policy, gamma)
    base_tables = [
        ("true", q_true),
        ("small_noise", q_true + rng.normal(scale=0.08, size=q_true.shape)),
        ("medium_noise", q_true + rng.normal(scale=0.5, size=q_true.shape)),
        ("large_noise", q_true + rng.normal(scale=1.2, size=q_true.shape)),
        ("zero", np.zeros_like(q_true)),
        ("over_regularized", 0.35 * q_true),
        ("biased", q_true + 0.7),
        ("wrong_sign", -0.5 * q_true),
    ]
    for idx in range(4):
        base_tables.append((f"random_{idx}", q_true + rng.normal(scale=0.8 + 0.2 * idx, size=q_true.shape)))
    candidates = [FQECandidate(name, _TabularQModel(table), complexity_order_key=idx) for idx, (name, table) in enumerate(base_tables)]
    occupancy = _dataset_state_action_distribution(dataset, n_states=4, n_actions=2)
    analytic_msbe = [_tabular_msbe(candidate.model.q_table, transition, rewards, target_policy, gamma, occupancy) for candidate in candidates]
    return {
        "dataset": dataset,
        "candidates": candidates,
        "gamma": gamma,
        "target_policy": target_policy,
        "action_space": np.eye(2, dtype=np.float64),
        "analytic_msbe": analytic_msbe,
    }


def _dataset_from_rows(rows: list[tuple[int, int, float, int, bool, int, int]], *, n_states: int, n_actions: int) -> TransitionDataset:
    states = _one_hot(np.asarray([row[0] for row in rows]), n_states)
    actions = _one_hot(np.asarray([row[1] for row in rows]), n_actions) if n_actions > 1 else np.zeros((len(rows), 1), dtype=np.float64)
    next_states = _one_hot(np.asarray([row[3] for row in rows]), n_states)
    return TransitionDataset(
        states,
        actions,
        np.asarray([row[2] for row in rows], dtype=np.float64),
        next_states,
        np.asarray([row[4] for row in rows], dtype=np.float64),
        np.asarray([row[5] for row in rows]),
        np.asarray([row[6] for row in rows]),
    )


def _one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx.astype(int)] = 1.0
    return out


class _TabularQModel:
    def __init__(self, q_table: np.ndarray) -> None:
        self.q_table = np.asarray(q_table, dtype=np.float64)

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        state_idx = np.argmax(np.asarray(states), axis=1)
        action_arr = np.asarray(actions)
        action_idx = np.zeros(state_idx.shape[0], dtype=np.int64) if action_arr.shape[1] == 1 else np.argmax(action_arr, axis=1)
        return self.q_table[state_idx, action_idx]

    def predict_all_actions(self, states: np.ndarray) -> np.ndarray:
        state_idx = np.argmax(np.asarray(states), axis=1)
        return self.q_table[state_idx]


def _lottery_msbe(q_table: np.ndarray, gamma: float, state_probs: np.ndarray) -> float:
    q = np.asarray(q_table, dtype=np.float64).reshape(3, 1)
    residual0 = q[0, 0] - gamma * 0.5 * (q[1, 0] + q[2, 0])
    residual1 = q[1, 0] - 10.0
    residual2 = q[2, 0] + 10.0
    residuals = np.asarray([residual0, residual1, residual2], dtype=np.float64)
    return float(np.sum(state_probs * residuals**2))


def _chain_transition_reward() -> tuple[np.ndarray, np.ndarray]:
    transition = np.zeros((4, 2, 4), dtype=np.float64)
    transition[0, 0] = [0.65, 0.35, 0.00, 0.00]
    transition[0, 1] = [0.10, 0.55, 0.35, 0.00]
    transition[1, 0] = [0.15, 0.55, 0.30, 0.00]
    transition[1, 1] = [0.00, 0.15, 0.55, 0.30]
    transition[2, 0] = [0.00, 0.30, 0.55, 0.15]
    transition[2, 1] = [0.00, 0.05, 0.45, 0.50]
    transition[3, 0] = [0.00, 0.00, 0.20, 0.80]
    transition[3, 1] = [0.00, 0.00, 0.05, 0.95]
    rewards = np.asarray([[0.0, 0.4], [0.1, 0.8], [0.2, 1.4], [1.0, 1.2]], dtype=np.float64)
    return transition, rewards


def _solve_tabular_q(transition: np.ndarray, rewards: np.ndarray, target_policy: np.ndarray, gamma: float) -> np.ndarray:
    n_states, n_actions = rewards.shape
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


def _dataset_state_action_distribution(dataset: TransitionDataset, *, n_states: int, n_actions: int) -> np.ndarray:
    state_idx = np.argmax(dataset.obs, axis=1)
    action_idx = np.argmax(dataset.actions, axis=1)
    out = np.zeros((n_states, n_actions), dtype=np.float64)
    for state, action in zip(state_idx, action_idx):
        out[int(state), int(action)] += 1.0
    out /= max(float(np.sum(out)), 1.0)
    return out


def _tabular_msbe(
    q_table: np.ndarray,
    transition: np.ndarray,
    rewards: np.ndarray,
    target_policy: np.ndarray,
    gamma: float,
    occupancy: np.ndarray,
) -> float:
    q = np.asarray(q_table, dtype=np.float64)
    next_v = np.sum(target_policy * q, axis=1)
    expected_next = np.einsum("san,n->sa", transition, next_v)
    residual = q - rewards - gamma * expected_next
    return float(np.sum(occupancy * residual**2))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = _rankdata(np.asarray(x, dtype=np.float64))
    yr = _rankdata(np.asarray(y, dtype=np.float64))
    return _pearson(xr, yr)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = math.sqrt(float(np.sum(x**2) * np.sum(y**2)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x * y) / denom)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_rows": len(rows), "by_method": {}, "by_setting_method": {}}
    methods = sorted({row["method"] for row in rows})
    settings = sorted({row["setting"] for row in rows})
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summary["by_method"][method] = _aggregate(group)
    for setting in settings:
        for method in methods:
            group = [row for row in rows if row["method"] == method and row["setting"] == setting]
            if group:
                summary["by_setting_method"][f"{setting}/{method}"] = _aggregate(group)
    return summary


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "oracle_selection_rate": float(np.mean([bool(row["selected_oracle"]) for row in rows])),
        "mean_msbe_regret": float(np.mean([float(row["msbe_regret"]) for row in rows])),
        "median_msbe_regret": float(np.median([float(row["msbe_regret"]) for row in rows])),
        "mean_score_msbe_spearman": float(np.mean([float(row["score_msbe_spearman"]) for row in rows])),
        "mean_score_msbe_pearson": float(np.mean([float(row["score_msbe_pearson"]) for row in rows])),
        "mean_runtime_sec": float(np.mean([float(row["runtime_sec"]) for row in rows])),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stabilize_torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
