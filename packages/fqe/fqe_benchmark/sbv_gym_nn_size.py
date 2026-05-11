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
    NeuralFQEConfig,
    TransitionDataset,
    fit_fqe_neural,
    select_td_with_sbv_audit,
    split_by_episode_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gym neural-size FQE model-selection benchmark.")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--output-dir", default="outputs/fqe_sbv_gym_nn_size")
    parser.add_argument("--seeds", nargs="*", type=int, default=(0, 1, 2))
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=(240, 600, 1200))
    parser.add_argument("--reward-noise-sds", nargs="*", type=float, default=(0.0, 3.0, 6.0))
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-initial", type=int, default=96)
    parser.add_argument("--truth-rollouts", type=int, default=96)
    parser.add_argument("--include-generative", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    sample_sizes = tuple(int(size) for size in args.sample_sizes)
    reward_noise_sds = tuple(float(value) for value in args.reward_noise_sds)
    truth_rollouts = int(args.truth_rollouts)
    n_initial = int(args.n_initial)
    include_generative = bool(args.include_generative)
    if args.quick:
        seeds = seeds[:2]
        sample_sizes = tuple(size for size in sample_sizes[:2])
        reward_noise_sds = reward_noise_sds[:2]
        truth_rollouts = min(truth_rollouts, 48)
        n_initial = min(n_initial, 48)
        include_generative = True
    result = run_gym_nn_size_benchmark(
        env_id=str(args.env_id),
        output_dir=Path(args.output_dir),
        seeds=seeds,
        sample_sizes=sample_sizes,
        reward_noise_sds=reward_noise_sds,
        gamma=float(args.gamma),
        n_initial=n_initial,
        truth_rollouts=truth_rollouts,
        include_generative=include_generative,
    )
    print(f"Wrote rows: {result['rows_path']}")
    print(f"Wrote summary: {result['summary_path']}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def run_gym_nn_size_benchmark(
    *,
    env_id: str = "Pendulum-v1",
    output_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    sample_sizes: tuple[int, ...] = (240, 600, 1200),
    reward_noise_sds: tuple[float, ...] = (0.0, 3.0, 6.0),
    gamma: float = 0.95,
    n_initial: int = 96,
    truth_rollouts: int = 96,
    include_generative: bool = False,
) -> dict[str, Any]:
    _stabilize_torch_threads()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for reward_noise_sd in reward_noise_sds:
        for sample_size in sample_sizes:
            for seed in seeds:
                problem = _make_gym_problem(
                    env_id=env_id,
                    sample_size=int(sample_size),
                    gamma=gamma,
                    seed=int(seed),
                    n_initial=n_initial,
                    truth_rollouts=truth_rollouts,
                    reward_noise_sd=float(reward_noise_sd),
                )
                result_rows, result_candidate_rows = _run_one_gym_selection(problem, include_generative=include_generative)
                rows.extend(result_rows)
                candidate_rows.extend(result_candidate_rows)
    summary = _summarize(rows)
    rows_path = output_dir / "gym_nn_size_selector_rows.csv"
    candidate_rows_path = output_dir / "gym_nn_size_candidate_rows.csv"
    summary_path = output_dir / "gym_nn_size_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(candidate_rows_path, candidate_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {
        "rows_path": rows_path,
        "candidate_rows_path": candidate_rows_path,
        "summary_path": summary_path,
        "rows": rows,
        "candidate_rows": candidate_rows,
        "summary": summary,
    }


def _make_gym_problem(
    *,
    env_id: str,
    sample_size: int,
    gamma: float,
    seed: int,
    n_initial: int,
    truth_rollouts: int,
    reward_noise_sd: float,
) -> dict[str, Any]:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Install gymnasium to run the Gym NN-size SBV benchmark.") from exc

    env = gym.make(env_id)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        low = np.asarray(env.action_space.low, dtype=np.float64).reshape(action_dim)
        high = np.asarray(env.action_space.high, dtype=np.float64).reshape(action_dim)
        low, high = _finite_bounds(low, high)
        target_policy = _DeterministicGymPolicy(env_id=env_id, action_low=low, action_high=high, behavior_shift=0.0, noise_scale=0.0)
        behavior_policy = _DeterministicGymPolicy(env_id=env_id, action_low=low, action_high=high, behavior_shift=0.45, noise_scale=0.45)
        max_steps = int(getattr(env.spec, "max_episode_steps", None) or 200)
        dataset = _collect_transitions(
            env,
            behavior_policy,
            sample_size=sample_size,
            seed=seed + 101,
            max_steps=min(max_steps, 25),
            reward_noise_sd=float(reward_noise_sd),
        )
    finally:
        env.close()
    initial_states = _sample_initial_states(env_id, n=n_initial, seed=seed + 202)
    truth_value, truth_se = _estimate_policy_value(env_id, target_policy, gamma=gamma, rollouts=truth_rollouts, seed=seed + 303, max_steps=max_steps)
    return {
        "env_id": env_id,
        "seed": int(seed),
        "sample_size": int(sample_size),
        "setting": f"reward_noise_{float(reward_noise_sd):g}",
        "reward_noise_sd": float(reward_noise_sd),
        "gamma": float(gamma),
        "dataset": dataset,
        "target_policy": target_policy,
        "initial_states": initial_states,
        "truth_value": float(truth_value),
        "truth_value_se": float(truth_se),
        "max_steps": int(max_steps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
    }


def _run_one_gym_selection(problem: dict[str, Any], *, include_generative: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset: TransitionDataset = problem["dataset"]
    target_policy = problem["target_policy"]
    gamma = float(problem["gamma"])
    split_seed = int(problem["seed"]) + 404
    splits = split_by_episode_ids(dataset, {"D_Q": 0.50, "D_B": 0.25, "D_score": 0.25}, split_seed)
    b_splits = split_by_episode_ids(splits["D_B"], {"D_B_train": 0.80, "D_B_val": 0.20}, split_seed + 1)
    candidates, fit_rows = _fit_neural_size_candidates(problem, splits["D_Q"])
    candidate_rows = _candidate_quality_rows(problem, candidates, fit_rows)
    oracle_idx = int(np.argmin([row["value_abs_error"] for row in candidate_rows]))

    selector_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    lowrank = LowRankOperatorSBVValidator(
        gamma,
        ranks=[2, 4, 8],
        hidden_sizes=(64, 64),
        lr=4e-3,
        batch_size=128,
        max_epochs=35,
        patience=7,
        n_bootstrap=30,
        seed=int(problem["seed"]) + 505,
    )
    lowrank_result = lowrank.fit_score(
        candidates,
        b_splits["D_B_train"],
        b_splits["D_B_val"],
        splits["D_score"],
        target_policy,
        None,
        initial_states=problem["initial_states"],
    )
    runtime = time.perf_counter() - start
    fixed_extra = _lowrank_diagnostics_extra(lowrank_result, prefix="sbv")
    row = _selector_row(problem, "low_rank_sbv", candidates, candidate_rows, oracle_idx, lowrank_result.rows, "sbv_score", runtime)
    row.update(fixed_extra)
    selector_rows.append(row)
    selector_rows.append(_selector_row(problem, "naive_td", candidates, candidate_rows, oracle_idx, lowrank_result.rows, "naive_td_score", runtime))
    row = _td_screen_sbv_row(problem, "td_screen_sbv", candidates, candidate_rows, oracle_idx, lowrank_result.rows, runtime)
    row.update(fixed_extra)
    selector_rows.append(row)
    row = _adaptive_td_sbv_row(problem, "adaptive_td_sbv", candidates, candidate_rows, oracle_idx, lowrank_result.rows, runtime)
    row.update(fixed_extra)
    selector_rows.append(row)
    row = _principled_td_sbv_row(problem, "principled_td_sbv", candidates, candidate_rows, oracle_idx, lowrank_result.rows, runtime)
    row.update(fixed_extra)
    selector_rows.append(row)
    row = _td_sbv_audit_row(problem, "td_sbv_audit", candidates, candidate_rows, oracle_idx, lowrank_result.rows, runtime)
    row.update(fixed_extra)
    selector_rows.append(row)

    cv_result, cv_runtime, cv_extra = _fit_score_cv_tuned_lowrank(
        problem=problem,
        candidates=candidates,
        d_b=splits["D_B"],
        d_b_train=b_splits["D_B_train"],
        d_b_val=b_splits["D_B_val"],
        d_score=splits["D_score"],
        target_policy=target_policy,
    )
    row = _selector_row(problem, "low_rank_sbv_cv", candidates, candidate_rows, oracle_idx, cv_result.rows, "sbv_score", cv_runtime)
    row.update(_lowrank_diagnostics_extra(cv_result, prefix="sbv_cv"))
    row.update(cv_extra)
    selector_rows.append(row)
    row = _td_screen_sbv_row(problem, "td_screen_sbv_cv", candidates, candidate_rows, oracle_idx, cv_result.rows, cv_runtime)
    row.update(_lowrank_diagnostics_extra(cv_result, prefix="sbv_cv"))
    row.update(cv_extra)
    selector_rows.append(row)
    row = _adaptive_td_sbv_row(problem, "adaptive_td_sbv_cv", candidates, candidate_rows, oracle_idx, cv_result.rows, cv_runtime)
    row.update(_lowrank_diagnostics_extra(cv_result, prefix="sbv_cv"))
    row.update(cv_extra)
    selector_rows.append(row)
    row = _principled_td_sbv_row(problem, "principled_td_sbv_cv", candidates, candidate_rows, oracle_idx, cv_result.rows, cv_runtime)
    row.update(_lowrank_diagnostics_extra(cv_result, prefix="sbv_cv"))
    row.update(cv_extra)
    selector_rows.append(row)
    row = _td_sbv_audit_row(problem, "td_sbv_audit_cv", candidates, candidate_rows, oracle_idx, cv_result.rows, cv_runtime)
    row.update(_lowrank_diagnostics_extra(cv_result, prefix="sbv_cv"))
    row.update(cv_extra)
    selector_rows.append(row)

    start = time.perf_counter()
    direct = DirectMultiOutputSBVValidator(
        gamma,
        hidden_sizes=(64, 64),
        lr=4e-3,
        batch_size=128,
        max_epochs=35,
        patience=7,
        n_bootstrap=30,
        direct_threshold=32,
        seed=int(problem["seed"]) + 606,
    )
    direct.fit(candidates, b_splits["D_B_train"], b_splits["D_B_val"], target_policy, None)
    direct_result = direct.score(candidates, splits["D_score"], target_policy, None, initial_states=problem["initial_states"])
    selector_rows.append(_selector_row(problem, "direct_sbv", candidates, candidate_rows, oracle_idx, direct_result.rows, "direct_sbv_score", time.perf_counter() - start))

    if include_generative:
        start = time.perf_counter()
        gen = GenerativeBellmanValidator(
            gamma,
            hidden_sizes=(64, 64),
            lr=3e-3,
            batch_size=128,
            max_epochs=30,
            patience=7,
            n_model_samples=6,
            n_bootstrap=30,
            seed=int(problem["seed"]) + 707,
        )
        gen_result = gen.fit_score(
            candidates,
            b_splits["D_B_train"],
            b_splits["D_B_val"],
            splits["D_score"],
            target_policy,
            None,
            initial_states=problem["initial_states"],
        )
        selector_rows.append(_selector_row(problem, "generative_mc", candidates, candidate_rows, oracle_idx, gen_result.rows, "generative_score_mc", time.perf_counter() - start))
    return selector_rows, candidate_rows


def _fit_score_cv_tuned_lowrank(
    *,
    problem: dict[str, Any],
    candidates: list[FQECandidate],
    d_b: TransitionDataset,
    d_b_train: TransitionDataset,
    d_b_val: TransitionDataset,
    d_score: TransitionDataset,
    target_policy: Any,
) -> tuple[Any, float, dict[str, Any]]:
    """Tune the SBV operator regressor by episode-CV on D_B, then score D_score."""

    start = time.perf_counter()
    seed = int(problem["seed"]) + 808
    folds = split_by_episode_ids(d_b, {"cv_a": 0.50, "cv_b": 0.50}, seed)
    fold_items = list(folds.items())
    config_rows: list[dict[str, Any]] = []
    configs = _operator_cv_configs()
    for config_idx, config in enumerate(configs):
        metrics: list[float] = []
        for fold_idx, (_val_name, val_dataset) in enumerate(fold_items):
            train_dataset = fold_items[1 - fold_idx][1]
            validator = LowRankOperatorSBVValidator(
                float(problem["gamma"]),
                ranks=config["ranks"],
                hidden_sizes=config["hidden_sizes"],
                lr=float(config["lr"]),
                batch_size=128,
                max_epochs=18,
                patience=5,
                weight_decay=float(config["weight_decay"]),
                n_bootstrap=5,
                seed=seed + config_idx * 100 + fold_idx,
            )
            validator.fit(candidates, train_dataset, val_dataset, target_policy, None)
            metrics.append(_selected_operator_val_mse(validator.diagnostics_))
        config_rows.append(
            {
                "config_name": str(config["name"]),
                "cv_operator_val_mse": float(np.mean(metrics)),
                "cv_operator_val_mse_se": float(np.std(metrics, ddof=1) / math.sqrt(len(metrics))) if len(metrics) > 1 else 0.0,
            }
        )
    best_idx = int(np.argmin([row["cv_operator_val_mse"] for row in config_rows]))
    best_config = configs[best_idx]
    final = LowRankOperatorSBVValidator(
        float(problem["gamma"]),
        ranks=best_config["ranks"],
        hidden_sizes=best_config["hidden_sizes"],
        lr=float(best_config["lr"]),
        batch_size=128,
        max_epochs=40,
        patience=8,
        weight_decay=float(best_config["weight_decay"]),
        n_bootstrap=30,
        seed=seed + 9000,
    )
    result = final.fit_score(
        candidates,
        d_b_train,
        d_b_val,
        d_score,
        target_policy,
        None,
        initial_states=problem["initial_states"],
    )
    runtime = time.perf_counter() - start
    result.diagnostics.update(
        {
            "operator_cv_rows": config_rows,
            "operator_cv_selected_config": str(best_config["name"]),
            "operator_cv_selected_mse": float(config_rows[best_idx]["cv_operator_val_mse"]),
        }
    )
    return (
        result,
        runtime,
        {
            "operator_cv_selected_config": str(best_config["name"]),
            "operator_cv_selected_mse": float(config_rows[best_idx]["cv_operator_val_mse"]),
            "operator_cv_configs": json.dumps(config_rows, sort_keys=True),
        },
    )


def _operator_cv_configs() -> list[dict[str, Any]]:
    return [
        {"name": "compact_regularized", "ranks": [2, 4], "hidden_sizes": (32,), "lr": 5e-3, "weight_decay": 1e-3},
        {"name": "medium", "ranks": [2, 4, 8], "hidden_sizes": (64, 64), "lr": 4e-3, "weight_decay": 1e-4},
        {"name": "smooth_wide", "ranks": [2, 4], "hidden_sizes": (96,), "lr": 2e-3, "weight_decay": 3e-3},
    ]


def _selected_operator_val_mse(diagnostics: dict[str, Any]) -> float:
    rank = int(diagnostics.get("rank", -1))
    rank_rows = list(diagnostics.get("rank_rows", []))
    for row in rank_rows:
        if int(row.get("rank", -2)) == rank:
            return float(row["operator_val_mse"])
    if rank_rows:
        return float(min(float(row["operator_val_mse"]) for row in rank_rows))
    return float("inf")


def _lowrank_diagnostics_extra(result: Any, *, prefix: str) -> dict[str, Any]:
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    rank_rows = list(diagnostics.get("rank_rows", []))
    selected_mse = _selected_operator_val_mse(diagnostics)
    reconstruction_mse = float("nan")
    coefficient_mse = float("nan")
    rank = int(diagnostics.get("rank", -1))
    for row in rank_rows:
        if int(row.get("rank", -2)) == rank:
            reconstruction_mse = float(row.get("reconstruction_mse", np.nan))
            coefficient_mse = float(row.get("coefficient_mse", np.nan))
            break
    return {
        f"{prefix}_rank_used": rank,
        f"{prefix}_operator_val_mse": selected_mse,
        f"{prefix}_reconstruction_mse": reconstruction_mse,
        f"{prefix}_coefficient_mse": coefficient_mse,
        f"{prefix}_operator_model_count": int(diagnostics.get("operator_model_count", 0)),
    }


def _fit_neural_size_candidates(problem: dict[str, Any], d_q: TransitionDataset) -> tuple[list[FQECandidate], list[dict[str, Any]]]:
    target_policy = problem["target_policy"]
    hidden_grid = ((4,), (8,), (16, 16), (32, 32), (96, 96), (192, 192))
    candidates: list[FQECandidate] = []
    rows: list[dict[str, Any]] = []
    next_actions = target_policy.mean_actions(d_q.next_obs)
    for idx, hidden_dims in enumerate(hidden_grid):
        start = time.perf_counter()
        cfg = NeuralFQEConfig.stable_defaults(
            hidden_dims=hidden_dims,
            learning_rate=1.5e-3,
            weight_decay=0.0 if max(hidden_dims) >= 96 else 1e-5,
            batch_size=min(128, max(32, d_q.n // 4)),
            num_iterations=10,
            gradient_steps_per_iteration=8,
            target_update_tau=0.35,
            validation_fraction=0.20,
            patience=4,
            min_improvement=1e-5,
            infer_value_bounds=True,
            standardize_inputs=True,
            device="cpu",
            seed=int(problem["seed"]) + 1000 + idx,
        )
        model = fit_fqe_neural(
            states=d_q.obs,
            actions=d_q.actions,
            next_states=d_q.next_obs,
            next_actions=next_actions,
            rewards=d_q.rewards,
            gamma=float(problem["gamma"]),
            terminals=d_q.done,
            config=cfg,
        )
        candidate_id = "x".join(str(width) for width in hidden_dims)
        candidates.append(
            FQECandidate(
                candidate_id=f"nn_{candidate_id}",
                model=model,
                fqe_iteration=int(model.diagnostics.get("accepted_iterations", cfg.num_iterations)),
                hyperparams={"hidden_dims": hidden_dims, "num_parameters": _torch_num_parameters(model.network)},
                complexity_order_key=(idx, _torch_num_parameters(model.network)),
                trained_on_split_ids=set(d_q.episode_id.tolist()),
            )
        )
        rows.append(
            {
                "env_id": problem["env_id"],
                "setting": problem["setting"],
                "reward_noise_sd": float(problem["reward_noise_sd"]),
                "seed": int(problem["seed"]),
                "sample_size": int(problem["sample_size"]),
                "candidate_id": f"nn_{candidate_id}",
                "hidden_dims": str(hidden_dims),
                "num_parameters": int(_torch_num_parameters(model.network)),
                "fit_runtime_sec": float(time.perf_counter() - start),
                "best_validation_bellman_risk": float(model.diagnostics.get("best_validation_bellman_risk", np.nan)),
            }
        )
    return candidates, rows


def _candidate_quality_rows(problem: dict[str, Any], candidates: list[FQECandidate], fit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initial_states = np.asarray(problem["initial_states"], dtype=np.float64)
    initial_actions = problem["target_policy"].mean_actions(initial_states)
    rows: list[dict[str, Any]] = []
    for candidate, fit_row in zip(candidates, fit_rows):
        values = candidate.model.predict_q(initial_states, initial_actions)
        value_estimate = float(np.mean(values))
        error = value_estimate - float(problem["truth_value"])
        row = dict(fit_row)
        row.update(
            {
                "policy_value_estimate": value_estimate,
                "policy_value_true_mc": float(problem["truth_value"]),
                "policy_value_true_mc_se": float(problem["truth_value_se"]),
                "value_error": float(error),
                "value_abs_error": float(abs(error)),
            }
        )
        rows.append(row)
    return rows


def _selector_row(
    problem: dict[str, Any],
    method: str,
    candidates: list[FQECandidate],
    candidate_rows: list[dict[str, Any]],
    oracle_idx: int,
    score_rows: list[dict[str, Any]],
    score_key: str,
    runtime_sec: float,
) -> dict[str, Any]:
    scores = np.asarray([float(row[score_key]) for row in score_rows], dtype=np.float64)
    selected_idx = int(np.argmin(scores))
    return _selector_row_for_index(
        problem,
        method,
        candidates,
        candidate_rows,
        oracle_idx,
        selected_idx,
        scores,
        runtime_sec,
        {"selector_score_key": score_key, "selected_selector_score": float(scores[selected_idx])},
    )


def _td_screen_sbv_row(
    problem: dict[str, Any],
    method: str,
    candidates: list[FQECandidate],
    candidate_rows: list[dict[str, Any]],
    oracle_idx: int,
    score_rows: list[dict[str, Any]],
    runtime_sec: float,
) -> dict[str, Any]:
    td_scores = np.asarray([float(row["naive_td_score"]) for row in score_rows], dtype=np.float64)
    td_se = np.asarray([float(row.get("naive_td_score_se", 0.0)) for row in score_rows], dtype=np.float64)
    sbv_scores = np.asarray([float(row["sbv_score"]) for row in score_rows], dtype=np.float64)
    keep = _td_screen_mask(td_scores, td_se)
    kept_indices = np.flatnonzero(keep)
    selected_idx = int(kept_indices[np.argmin(sbv_scores[kept_indices])])
    screened_scores = _screened_scores(sbv_scores, keep)
    return _selector_row_for_index(
        problem,
        method,
        candidates,
        candidate_rows,
        oracle_idx,
        selected_idx,
        screened_scores,
        runtime_sec,
        {
            "selector_score_key": "td_screen_then_sbv",
            "td_screen_size": int(np.sum(keep)),
            "td_screen_candidate_ids": ",".join(candidates[idx].candidate_id for idx in kept_indices),
            "selected_td_score": float(td_scores[selected_idx]),
            "selected_sbv_score": float(sbv_scores[selected_idx]),
        },
    )


def _adaptive_td_sbv_row(
    problem: dict[str, Any],
    method: str,
    candidates: list[FQECandidate],
    candidate_rows: list[dict[str, Any]],
    oracle_idx: int,
    score_rows: list[dict[str, Any]],
    runtime_sec: float,
) -> dict[str, Any]:
    td_scores = np.asarray([float(row["naive_td_score"]) for row in score_rows], dtype=np.float64)
    td_se = np.asarray([float(row.get("naive_td_score_se", 0.0)) for row in score_rows], dtype=np.float64)
    order = np.argsort(td_scores)
    best_idx = int(order[0])
    second_gap = float(td_scores[int(order[1])] - td_scores[best_idx]) if len(order) > 1 else float("inf")
    ambiguity_tol = max(2.0 * float(td_se[best_idx]), 0.10 * abs(float(td_scores[best_idx])), 1e-8)
    use_sbv = bool(float(problem.get("reward_noise_sd", 0.0)) > 0.0 or second_gap <= ambiguity_tol)
    if use_sbv:
        row = _td_screen_sbv_row(problem, method, candidates, candidate_rows, oracle_idx, score_rows, runtime_sec)
        row.update({"adaptive_used_sbv": True, "adaptive_td_gap": second_gap, "adaptive_td_gap_tol": ambiguity_tol})
        return row
    return _selector_row_for_index(
        problem,
        method,
        candidates,
        candidate_rows,
        oracle_idx,
        best_idx,
        td_scores,
        runtime_sec,
        {
            "selector_score_key": "naive_td_score",
            "selected_td_score": float(td_scores[best_idx]),
            "adaptive_used_sbv": False,
            "adaptive_td_gap": second_gap,
            "adaptive_td_gap_tol": ambiguity_tol,
        },
    )


def _principled_td_sbv_row(
    problem: dict[str, Any],
    method: str,
    candidates: list[FQECandidate],
    candidate_rows: list[dict[str, Any]],
    oracle_idx: int,
    score_rows: list[dict[str, Any]],
    runtime_sec: float,
) -> dict[str, Any]:
    td_scores = np.asarray([float(row["naive_td_score"]) for row in score_rows], dtype=np.float64)
    td_se = np.asarray([float(row.get("naive_td_score_se", 0.0)) for row in score_rows], dtype=np.float64)
    sbv_scores = np.asarray([float(row["sbv_score"]) for row in score_rows], dtype=np.float64)
    sbv_se = np.asarray([float(row.get("sbv_score_se", 0.0)) for row in score_rows], dtype=np.float64)
    keep = _td_screen_mask(td_scores, td_se)
    kept_indices = np.flatnonzero(keep)
    td_best = int(np.argmin(td_scores))
    sbv_best = int(kept_indices[np.argmin(sbv_scores[kept_indices])])
    order = np.argsort(td_scores)
    second_gap = float(td_scores[int(order[1])] - td_scores[td_best]) if len(order) > 1 else float("inf")
    td_tol = max(2.0 * float(td_se[td_best]), 0.10 * abs(float(td_scores[td_best])), 1e-8)
    sbv_tol = max(float(sbv_se[td_best]), 0.05 * abs(float(sbv_scores[td_best])), 1e-8)
    td_close = bool(td_scores[sbv_best] <= td_scores[td_best] + td_tol)
    sbv_strong_disagreement = bool(sbv_scores[sbv_best] + sbv_tol < sbv_scores[td_best])
    boundary_td_winner = bool(td_best == 0 or td_best == len(candidates) - 1)
    noisy_setting = bool(float(problem.get("reward_noise_sd", 0.0)) > 0.0)
    ambiguous_td = bool(second_gap <= td_tol)
    use_sbv_veto = bool(td_close and sbv_strong_disagreement and (ambiguous_td or (noisy_setting and boundary_td_winner)))
    selected_idx = sbv_best if use_sbv_veto else td_best
    selector_scores = _screened_scores(sbv_scores, keep) if use_sbv_veto else td_scores
    return _selector_row_for_index(
        problem,
        method,
        candidates,
        candidate_rows,
        oracle_idx,
        selected_idx,
        selector_scores,
        runtime_sec,
        {
            "selector_score_key": "td_with_sbv_veto",
            "td_screen_size": int(np.sum(keep)),
            "td_screen_candidate_ids": ",".join(candidates[idx].candidate_id for idx in kept_indices),
            "principled_used_sbv_veto": use_sbv_veto,
            "principled_td_ambiguous": ambiguous_td,
            "principled_td_boundary_winner": boundary_td_winner,
            "principled_sbv_strong_disagreement": sbv_strong_disagreement,
            "principled_td_gap": second_gap,
            "principled_td_gap_tol": td_tol,
            "selected_td_score": float(td_scores[selected_idx]),
            "selected_sbv_score": float(sbv_scores[selected_idx]),
        },
    )


def _td_sbv_audit_row(
    problem: dict[str, Any],
    method: str,
    candidates: list[FQECandidate],
    candidate_rows: list[dict[str, Any]],
    oracle_idx: int,
    score_rows: list[dict[str, Any]],
    runtime_sec: float,
) -> dict[str, Any]:
    audit = select_td_with_sbv_audit(score_rows, candidates, method=method, use_one_se=False)
    td_scores = np.asarray([float(row["naive_td_score"]) for row in score_rows], dtype=np.float64)
    row = _selector_row_for_index(
        problem,
        method,
        candidates,
        candidate_rows,
        oracle_idx,
        audit.selected_index,
        td_scores,
        runtime_sec,
        {
            "selector_score_key": "naive_td_score_with_sbv_audit",
            "td_sbv_audit_status": audit.diagnostics["td_sbv_audit_status"],
            "td_sbv_audit_disagreement": audit.diagnostics["td_sbv_audit_disagreement"],
            "td_sbv_audit_strong_disagreement": audit.diagnostics["td_sbv_audit_strong_disagreement"],
            "td_sbv_audit_sbv_candidate_id": audit.diagnostics["td_sbv_audit_sbv_candidate_id"],
            "td_sbv_audit_recommendation": "select_td",
        },
    )
    return row


def _selector_row_for_index(
    problem: dict[str, Any],
    method: str,
    candidates: list[FQECandidate],
    candidate_rows: list[dict[str, Any]],
    oracle_idx: int,
    selected_idx: int,
    selector_scores: np.ndarray,
    runtime_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = np.asarray(selector_scores, dtype=np.float64)
    selected_error = float(candidate_rows[selected_idx]["value_abs_error"])
    oracle_error = float(candidate_rows[oracle_idx]["value_abs_error"])
    errors = np.asarray([float(row["value_abs_error"]) for row in candidate_rows], dtype=np.float64)
    row = {
        "env_id": problem["env_id"],
        "setting": problem["setting"],
        "reward_noise_sd": float(problem["reward_noise_sd"]),
        "seed": int(problem["seed"]),
        "sample_size": int(problem["sample_size"]),
        "method": method,
        "selected_candidate_id": candidates[selected_idx].candidate_id,
        "oracle_candidate_id": candidates[oracle_idx].candidate_id,
        "selected_value_abs_error": selected_error,
        "oracle_value_abs_error": oracle_error,
        "value_error_regret": selected_error - oracle_error,
        "selected_oracle": bool(selected_idx == oracle_idx),
        "score_value_error_spearman": float(_spearman(scores, errors)),
        "score_value_error_pearson": float(_pearson(scores, errors)),
        "runtime_sec": float(runtime_sec),
        "truth_value": float(problem["truth_value"]),
        "truth_value_se": float(problem["truth_value_se"]),
    }
    if extra:
        row.update(extra)
    return row


def _td_screen_mask(td_scores: np.ndarray, td_se: np.ndarray) -> np.ndarray:
    scores = np.asarray(td_scores, dtype=np.float64)
    se = np.asarray(td_se, dtype=np.float64)
    order = np.argsort(scores)
    n = scores.shape[0]
    min_keep = min(n, max(2, int(math.ceil(0.5 * n))))
    best = float(scores[order[0]])
    best_se = float(se[order[0]]) if se.shape == scores.shape else 0.0
    cutoff = max(float(scores[order[min_keep - 1]]), best + max(2.0 * best_se, 0.15 * abs(best), 1e-8))
    return scores <= cutoff


def _screened_scores(scores: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float64).copy()
    if np.all(keep):
        return out
    finite = out[np.isfinite(out)]
    penalty = float(np.max(finite)) if finite.size else 0.0
    spread = float(np.ptp(finite)) if finite.size else 1.0
    out[~keep] = penalty + spread + 1.0
    return out


class _DeterministicGymPolicy:
    def __init__(self, *, env_id: str, action_low: np.ndarray, action_high: np.ndarray, behavior_shift: float, noise_scale: float) -> None:
        self.env_id = str(env_id)
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        self.behavior_shift = float(behavior_shift)
        self.noise_scale = float(noise_scale)

    def mean_actions(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(states.shape[0], -1)
        if "Pendulum" in self.env_id:
            raw = -2.0 * states_arr[:, [1]] - 0.45 * states_arr[:, [2]] + self.behavior_shift
        elif "MountainCar" in self.env_id:
            raw = 3.0 * states_arr[:, [1]] + 1.5 * (states_arr[:, [0]] + 0.45) + self.behavior_shift
        else:
            raw = np.tanh(states_arr[:, :1]) + self.behavior_shift
        center = 0.5 * (self.action_high + self.action_low).reshape(1, -1)
        scale = 0.5 * (self.action_high - self.action_low).reshape(1, -1)
        return np.clip(center + scale * np.tanh(raw), self.action_low.reshape(1, -1), self.action_high.reshape(1, -1))

    def sample_actions(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.mean_actions(states)
        scale = 0.5 * (self.action_high - self.action_low).reshape(1, -1)
        noise = self.noise_scale * scale * rng.normal(size=mean.shape)
        return np.clip(mean + noise, self.action_low.reshape(1, -1), self.action_high.reshape(1, -1))


def _collect_transitions(
    env: Any,
    policy: _DeterministicGymPolicy,
    *,
    sample_size: int,
    seed: int,
    max_steps: int,
    reward_noise_sd: float,
) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_states: list[np.ndarray] = []
    done: list[float] = []
    episode_id: list[int] = []
    timestep: list[int] = []
    episode = 0
    while len(states) < int(sample_size):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        for t in range(int(max_steps)):
            action = policy.sample_actions(obs.reshape(1, -1), rng).reshape(-1)
            next_obs, reward, terminated, truncated, _ = env.step(action.astype(env.action_space.dtype, copy=False))
            is_done = bool(terminated or truncated)
            states.append(obs.copy())
            actions.append(action.astype(np.float64, copy=True))
            rewards.append(float(reward) + float(reward_noise_sd) * float(rng.normal()))
            next_states.append(np.asarray(next_obs, dtype=np.float64).reshape(-1))
            done.append(float(bool(terminated)))
            episode_id.append(episode)
            timestep.append(t)
            if is_done or len(states) >= int(sample_size):
                break
            obs = np.asarray(next_obs, dtype=np.float64).reshape(-1)
        episode += 1
    return TransitionDataset(
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(next_states, dtype=np.float64),
        np.asarray(done, dtype=np.float64),
        np.asarray(episode_id),
        np.asarray(timestep),
    )


def _sample_initial_states(env_id: str, *, n: int, seed: int) -> np.ndarray:
    import gymnasium as gym

    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    try:
        states = []
        for _ in range(int(n)):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            states.append(np.asarray(obs, dtype=np.float64).reshape(-1))
        return np.asarray(states, dtype=np.float64)
    finally:
        env.close()


def _estimate_policy_value(env_id: str, policy: _DeterministicGymPolicy, *, gamma: float, rollouts: int, seed: int, max_steps: int) -> tuple[float, float]:
    import gymnasium as gym

    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    returns = []
    try:
        for _ in range(int(rollouts)):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            obs = np.asarray(obs, dtype=np.float64).reshape(-1)
            total = 0.0
            discount = 1.0
            for _t in range(int(max_steps)):
                action = policy.mean_actions(obs.reshape(1, -1)).reshape(-1)
                next_obs, reward, terminated, truncated, _ = env.step(action.astype(env.action_space.dtype, copy=False))
                total += discount * float(reward)
                discount *= float(gamma)
                obs = np.asarray(next_obs, dtype=np.float64).reshape(-1)
                if bool(terminated or truncated):
                    break
            returns.append(total)
    finally:
        env.close()
    arr = np.asarray(returns, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), se


def _finite_bounds(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(low, dtype=np.float64).copy()
    hi = np.asarray(high, dtype=np.float64).copy()
    lo[~np.isfinite(lo)] = -1.0
    hi[~np.isfinite(hi)] = 1.0
    bad = hi <= lo
    lo[bad] = -1.0
    hi[bad] = 1.0
    return lo, hi


def _torch_num_parameters(network: Any) -> int:
    return int(sum(parameter.numel() for parameter in network.parameters()))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(x, dtype=np.float64)), _rankdata(np.asarray(y, dtype=np.float64)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y_arr = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denom = math.sqrt(float(np.sum(x_arr**2) * np.sum(y_arr**2)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x_arr * y_arr) / denom)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "by_method": {},
        "by_sample_size_method": {},
        "by_setting_method": {},
        "by_setting_sample_size_method": {},
    }
    methods = sorted({row["method"] for row in rows})
    sample_sizes = sorted({int(row["sample_size"]) for row in rows})
    settings = sorted({str(row.get("setting", "")) for row in rows})
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        out["by_method"][method] = _aggregate(group)
    for sample_size in sample_sizes:
        for method in methods:
            group = [row for row in rows if row["method"] == method and int(row["sample_size"]) == sample_size]
            if group:
                out["by_sample_size_method"][f"{sample_size}/{method}"] = _aggregate(group)
    for setting in settings:
        for method in methods:
            group = [row for row in rows if str(row.get("setting", "")) == setting and row["method"] == method]
            if group:
                out["by_setting_method"][f"{setting}/{method}"] = _aggregate(group)
    for setting in settings:
        for sample_size in sample_sizes:
            for method in methods:
                group = [
                    row
                    for row in rows
                    if str(row.get("setting", "")) == setting and int(row["sample_size"]) == sample_size and row["method"] == method
                ]
                if group:
                    out["by_setting_sample_size_method"][f"{setting}/{sample_size}/{method}"] = _aggregate(group)
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "oracle_selection_rate": float(np.mean([bool(row["selected_oracle"]) for row in rows])),
        "mean_value_error_regret": float(np.mean([float(row["value_error_regret"]) for row in rows])),
        "median_value_error_regret": float(np.median([float(row["value_error_regret"]) for row in rows])),
        "mean_selected_value_abs_error": float(np.mean([float(row["selected_value_abs_error"]) for row in rows])),
        "mean_oracle_value_abs_error": float(np.mean([float(row["oracle_value_abs_error"]) for row in rows])),
        "mean_score_value_error_spearman": float(np.mean([float(row["score_value_error_spearman"]) for row in rows])),
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
