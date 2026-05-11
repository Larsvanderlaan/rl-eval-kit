from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time
import tracemalloc
from typing import Any

import numpy as np

from fqe import (
    DirectMultiOutputSBVValidator,
    FQECandidate,
    GenerativeBellmanValidator,
    LowRankOperatorSBVValidator,
    TransitionDataset,
    select_td_with_sbv_audit,
    split_by_episode_ids,
)
from fqe_benchmark.io import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FQE validation selectors.")
    parser.add_argument("--config", default=None, help="JSON or YAML config. If omitted, run a tiny synthetic smoke example.")
    parser.add_argument("--output-dir", default=None, help="Override output directory from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config) if args.config else _demo_config()
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    result = run_comparison(config)
    print(f"Wrote validator comparison rows: {result['rows_path']}")
    print(f"Wrote validator comparison diagnostics: {result['diagnostics_path']}")


def run_comparison(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 123))
    gamma = float(config.get("gamma", 0.9))
    output_dir = Path(config.get("output_dir", "outputs/fqe_sbv_compare"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(config)
    target_policy = _load_target_policy(config, dataset)
    action_space = _load_action_space(config, dataset)
    candidates = _load_candidates(config, dataset, target_policy)
    fractions = config.get("split_fractions", {"D_B": 0.7, "D_score": 0.3})
    splits = split_by_episode_ids(dataset, fractions, seed)
    d_b = splits.get("D_B") or splits.get("split_0") or next(iter(splits.values()))
    d_score = splits.get("D_score") or splits.get("split_1") or list(splits.values())[-1]
    b_splits = split_by_episode_ids(d_b, {"D_B_train": 0.8, "D_B_val": 0.2}, seed + 17)
    d_b_train = b_splits["D_B_train"]
    d_b_val = b_splits["D_B_val"]

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"config": _json_safe_config(config), "methods": {}}
    initial_states = config.get("initial_states")
    initial_episode_id = config.get("initial_episode_id")
    initial_states_arr = None if initial_states is None else np.asarray(initial_states, dtype=np.float64)
    initial_episode_arr = None if initial_episode_id is None else np.asarray(initial_episode_id)

    tracemalloc.start()
    lowrank_cfg = dict(config.get("low_rank_sbv", {}))
    start = time.perf_counter()
    lowrank = LowRankOperatorSBVValidator(gamma=gamma, seed=seed, **lowrank_cfg)
    lowrank_result = lowrank.fit_score(
        candidates,
        d_b_train,
        d_b_val,
        d_score,
        target_policy,
        action_space,
        initial_states=initial_states_arr,
        initial_episode_id=initial_episode_arr,
    )
    rows.extend(_annotate_rows(lowrank_result.rows, runtime_sec=time.perf_counter() - start, diagnostics=lowrank_result.diagnostics))
    diagnostics["methods"]["low_rank_sbv"] = lowrank_result.diagnostics
    audit_result = select_td_with_sbv_audit(lowrank_result.rows, candidates)
    rows.extend(_annotate_rows(audit_result.rows, runtime_sec=time.perf_counter() - start, diagnostics=audit_result.diagnostics))
    diagnostics["methods"]["td_sbv_audit"] = audit_result.diagnostics

    gen_cfg = dict(config.get("generative", {}))
    start = time.perf_counter()
    generative = GenerativeBellmanValidator(gamma=gamma, seed=seed + 101, **gen_cfg)
    gen_result = generative.fit_score(
        candidates,
        d_b_train,
        d_b_val,
        d_score,
        target_policy,
        action_space,
        initial_states=initial_states_arr,
        initial_episode_id=initial_episode_arr,
    )
    rows.extend(_annotate_rows(gen_result.rows, runtime_sec=time.perf_counter() - start, diagnostics=gen_result.diagnostics))
    diagnostics["methods"]["generative"] = gen_result.diagnostics

    direct_threshold = int(config.get("direct_threshold", 32))
    if len(candidates) <= direct_threshold:
        start = time.perf_counter()
        direct = DirectMultiOutputSBVValidator(gamma=gamma, seed=seed + 202, direct_threshold=direct_threshold, **dict(config.get("direct_sbv", {})))
        direct.fit(candidates, d_b_train, d_b_val, target_policy, action_space)
        direct_result = direct.score(
            candidates,
            d_score,
            target_policy,
            action_space,
            initial_states=initial_states_arr,
            initial_episode_id=initial_episode_arr,
        )
        rows.extend(_annotate_rows(direct_result.rows, runtime_sec=time.perf_counter() - start, diagnostics=direct_result.diagnostics))
        diagnostics["methods"]["direct_sbv"] = direct_result.diagnostics

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    diagnostics["peak_memory_bytes"] = int(peak)
    diagnostics["current_memory_bytes"] = int(current)
    rows_path = output_dir / "validator_comparison.csv"
    diagnostics_path = output_dir / "validator_comparison_diagnostics.json"
    write_csv(rows_path, rows)
    write_json(diagnostics_path, diagnostics)
    return {"rows_path": rows_path, "diagnostics_path": diagnostics_path, "rows": rows, "diagnostics": diagnostics}


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text()
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyYAML is required for YAML SBV configs. Install fqe[validation].") from exc
        return dict(yaml.safe_load(text))
    return dict(json.loads(text))


def _load_dataset(config: dict[str, Any]) -> TransitionDataset:
    if "dataset_npz" not in config:
        return _demo_dataset()
    with np.load(config["dataset_npz"], allow_pickle=False) as data:
        return TransitionDataset(
            obs=data["obs"],
            actions=data["actions"],
            rewards=data["rewards"],
            next_obs=data["next_obs"],
            done=data["done"],
            episode_id=data["episode_id"],
            timestep=data["timestep"] if "timestep" in data else None,
        )


def _load_candidates(config: dict[str, Any], dataset: TransitionDataset, target_policy: Any) -> list[FQECandidate]:
    raw = config.get("candidates")
    if raw:
        candidates = []
        for idx, row in enumerate(raw):
            with Path(row["path"]).open("rb") as handle:
                model = pickle.load(handle)
            candidates.append(
                FQECandidate(
                    candidate_id=str(row.get("candidate_id", f"candidate_{idx:03d}")),
                    model=model,
                    checkpoint_path=row.get("path"),
                    fqe_iteration=row.get("fqe_iteration"),
                    hyperparams=row.get("hyperparams"),
                    complexity_order_key=row.get("complexity_order_key", idx),
                    trained_on_split_ids=row.get("trained_on_split_ids"),
                )
            )
        return candidates
    q_true = _demo_q_table()
    noise = [0.0, 0.15, 0.75, 1.5]
    return [
        FQECandidate(
            candidate_id=f"demo_noise_{idx}",
            model=_TabularQModel(q_true + amount * _demo_noise(q_true.shape, idx)),
            fqe_iteration=idx,
            complexity_order_key=idx,
        )
        for idx, amount in enumerate(noise)
    ]


def _load_target_policy(config: dict[str, Any], dataset: TransitionDataset) -> Any:
    if "target_policy" in config:
        raw = config["target_policy"]
        if isinstance(raw, dict) and raw.get("type") == "discrete_table":
            return np.asarray(raw["probabilities"], dtype=np.float64)
        if isinstance(raw, list):
            return np.asarray(raw, dtype=np.float64)
    return np.asarray([[0.2, 0.8], [0.3, 0.7], [0.6, 0.4]], dtype=np.float64)


def _load_action_space(config: dict[str, Any], dataset: TransitionDataset) -> Any:
    if "action_space" in config:
        return np.asarray(config["action_space"], dtype=np.float64)
    return np.eye(2, dtype=np.float64)


def _annotate_rows(rows: list[dict[str, Any]], *, runtime_sec: float, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        annotated = dict(row)
        annotated["runtime_sec"] = float(runtime_sec)
        annotated["rank_used"] = diagnostics.get("rank")
        annotated["number_of_learned_regressors"] = diagnostics.get("operator_model_count")
        annotated["generative_validation_nll"] = diagnostics.get("validation_nll")
        out.append(annotated)
    return out


def _demo_config() -> dict[str, Any]:
    return {
        "gamma": 0.8,
        "seed": 123,
        "output_dir": "outputs/fqe_sbv_compare_demo",
        "low_rank_sbv": {"ranks": [1, 2], "hidden_sizes": (32,), "max_epochs": 20, "patience": 4},
        "generative": {"hidden_sizes": (32,), "max_epochs": 20, "patience": 4, "n_model_samples": 4},
        "direct_sbv": {"hidden_sizes": (32,), "max_epochs": 20, "patience": 4},
    }


def _demo_dataset() -> TransitionDataset:
    rng = np.random.default_rng(123)
    n_episodes = 60
    horizon = 4
    obs_rows = []
    action_rows = []
    next_rows = []
    reward_rows = []
    done_rows = []
    episode_rows = []
    timestep_rows = []
    for episode in range(n_episodes):
        state = int(rng.integers(0, 3))
        for t in range(horizon):
            action = int(rng.integers(0, 2))
            next_state = min(2, max(0, state + (1 if action else -1)))
            reward = float(next_state == 2) - 0.1 * state
            done = t == horizon - 1
            obs_rows.append(_one_hot(state, 3))
            action_rows.append(_one_hot(action, 2))
            next_rows.append(_one_hot(next_state, 3))
            reward_rows.append(reward)
            done_rows.append(float(done))
            episode_rows.append(episode)
            timestep_rows.append(t)
            state = next_state
            if done:
                break
    return TransitionDataset(
        obs=np.asarray(obs_rows),
        actions=np.asarray(action_rows),
        rewards=np.asarray(reward_rows),
        next_obs=np.asarray(next_rows),
        done=np.asarray(done_rows),
        episode_id=np.asarray(episode_rows),
        timestep=np.asarray(timestep_rows),
    )


def _demo_q_table() -> np.ndarray:
    return np.asarray([[1.1, 2.0], [1.5, 2.5], [2.1, 3.0]], dtype=np.float64)


def _demo_noise(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 999).normal(size=shape)


def _one_hot(idx: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    out[int(idx)] = 1.0
    return out


class _TabularQModel:
    def __init__(self, q_table: np.ndarray) -> None:
        self.q_table = np.asarray(q_table, dtype=np.float64)

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        state_idx = np.argmax(np.asarray(states), axis=1)
        action_idx = np.argmax(np.asarray(actions), axis=1)
        return self.q_table[state_idx, action_idx]

    def predict_all_actions(self, states: np.ndarray) -> np.ndarray:
        state_idx = np.argmax(np.asarray(states), axis=1)
        return self.q_table[state_idx]


def _json_safe_config(config: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in config.items():
        if key == "candidates":
            safe[key] = "[candidate specs omitted]"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


if __name__ == "__main__":
    main()
