from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np

from occupancy_ratio.calibration import calibrate_occupancy_bellman_binning
from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralTransitionRatioConfig,
    fit_discounted_occupancy_ratio_neural,
)
from occupancy_ratio_benchmark.data import one_hot


@dataclass(frozen=True)
class GridBenchmarkConfig:
    output_root: Path
    google_research_root: Path
    seeds: tuple[int, ...]
    alphas: tuple[float, ...]
    gammas: tuple[float, ...]
    num_trajectories: int
    max_trajectory_length: int
    boosted_losses: tuple[str, ...]
    boosted_num_iterations: int
    boosted_mcmc_samples: int
    include_neural: bool
    neural_num_iterations: int
    neural_mcmc_samples: int
    neural_gradient_steps_per_iteration: int
    neural_action_steps: int
    neural_transition_steps: int
    neural_transition_permutation_samples: int
    neural_batch_size: int
    neural_hidden_dim: int
    huber_delta_scale: float
    estimator_timeout_sec: float
    include_bellman_moment_calibration: bool = False


def run_gridwalk_benchmark(config: GridBenchmarkConfig) -> list[dict[str, Any]]:
    """Run the original Google DualDICE GridWalk benchmark plus boosted trees."""
    _add_google_research(config.google_research_root)
    import dual_dice.algos.dual_dice as google_tabular_dice  # noqa: PLC0415
    import dual_dice.gridworld.environments as gridworld_envs  # noqa: PLC0415
    import dual_dice.gridworld.policies as gridworld_policies  # noqa: PLC0415
    import dual_dice.transition_data as transition_data  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    output_dir = config.output_root / "gridwalk"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    for gamma in config.gammas:
        for alpha in config.alphas:
            for seed in config.seeds:
                print(f"gridwalk gamma={gamma} alpha={alpha} seed={seed}", flush=True)
                np.random.seed(int(seed))
                env = gridworld_envs.GridWalk(length=10, tabular_obs=True)
                behavior_policy = gridworld_policies.get_behavior_gridwalk_policy(env, tabular_obs=True, alpha=alpha)
                target_policy = gridworld_policies.get_target_gridwalk_policy(env, tabular_obs=True)

                try:
                    with _time_limit(config.estimator_timeout_sec):
                        behavior_data, behavior_episode_reward, behavior_step_reward = transition_data.collect_data(
                            env,
                            behavior_policy,
                            config.num_trajectories,
                            config.max_trajectory_length,
                            gamma=gamma,
                        )
                        target_data, target_episode_reward, target_step_reward = transition_data.collect_data(
                            env,
                            target_policy,
                            config.num_trajectories,
                            config.max_trajectory_length,
                            gamma=gamma,
                        )
                        del target_data
                except Exception as exc:
                    common_error = {
                        "benchmark": "google_dualdice_gridwalk",
                        "setting": "gridwalk_tabular",
                        "seed": int(seed),
                        "alpha": float(alpha),
                        "gamma": float(gamma),
                        "num_trajectories": int(config.num_trajectories),
                        "max_trajectory_length": int(config.max_trajectory_length),
                    }
                    rows.append(_error_row(common_error, "dataset", exc, time.perf_counter()))
                    _write_csv(results_path, rows)
                    continue

                common = {
                    "benchmark": "google_dualdice_gridwalk",
                    "setting": "gridwalk_tabular",
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "num_trajectories": int(config.num_trajectories),
                    "max_trajectory_length": int(config.max_trajectory_length),
                    "behavior_episode_reward": float(behavior_episode_reward),
                    "behavior_step_reward": float(behavior_step_reward),
                    "target_episode_reward": float(target_episode_reward),
                    "target_step_reward": float(target_step_reward),
                }
                rows.append(
                    {
                        **common,
                        "estimator": "behavior",
                        "status": "ok",
                        "estimated_step_reward": float(behavior_step_reward),
                        "absolute_error": float(abs(behavior_step_reward - target_step_reward)),
                        "runtime_sec": 0.0,
                    }
                )
                _write_csv(results_path, rows)
                print("  estimator google_tabular_dualdice", flush=True)
                rows.append(_run_google_tabular_dice(google_tabular_dice, env, behavior_data, target_policy, common, config))
                _write_csv(results_path, rows)
                for loss in config.boosted_losses:
                    print(f"  estimator boosted_tree_{loss}", flush=True)
                    rows.extend(_run_boosted_tree_rows(env, behavior_data, target_policy, common, config, loss=loss))
                    _write_csv(results_path, rows)
                    if config.include_neural:
                        print(f"  estimator neural_network_{loss}", flush=True)
                        rows.extend(_run_neural_network_rows(env, behavior_data, target_policy, common, config, loss=loss))
                        _write_csv(results_path, rows)

    _write_csv(results_path, rows)
    _write_summary(output_dir / "summary.csv", rows)
    return rows


def _run_google_tabular_dice(
    google_tabular_dice,
    env,
    behavior_data,
    target_policy,
    common: dict[str, Any],
    config: GridBenchmarkConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        solver = None
        with _time_limit(config.estimator_timeout_sec):
            solver = google_tabular_dice.TabularDualDice(env.num_states, env.num_actions, common["gamma"])
            estimate = float(solver.solve(behavior_data, target_policy))
            solver.close()
        return {
            **common,
            "estimator": "google_tabular_dualdice",
            "status": "ok",
            "estimated_step_reward": estimate,
            "absolute_error": float(abs(estimate - common["target_step_reward"])),
            "runtime_sec": float(time.perf_counter() - start),
        }
    except Exception as exc:
        if solver is not None:
            try:
                solver.close()
            except Exception:
                pass
        return _error_row(common, "google_tabular_dualdice", exc, start)


def _run_boosted_tree_rows(
    env,
    behavior_data,
    target_policy,
    common: dict[str, Any],
    config: GridBenchmarkConfig,
    *,
    loss: str,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    try:
        all_data = behavior_data.get_all()
        states_i = np.asarray(all_data.state, dtype=np.int64).reshape(-1)
        actions_i = np.asarray(all_data.action, dtype=np.int64).reshape(-1)
        next_states_i = np.asarray(all_data.next_state, dtype=np.int64).reshape(-1)
        target_actions_i = target_policy.sample_action(states_i)
        min_leaf = _safe_min_data_in_leaf(states_i.size)

        with _time_limit(config.estimator_timeout_sec):
            model = fit_discounted_occupancy_ratio(
                states=one_hot(states_i, env.num_states),
                actions=one_hot(actions_i, env.num_actions),
                next_states=one_hot(next_states_i, env.num_states),
                target_actions=one_hot(target_actions_i, env.num_actions),
                gamma=float(common["gamma"]),
                occupancy=OccupancyRegressionConfig(
                    num_iterations=int(config.boosted_num_iterations),
                    mcmc_samples=int(config.boosted_mcmc_samples),
                    batch_size=512,
                    loss=loss,
                    huber_delta_scale=float(config.huber_delta_scale),
                    show_progress=False,
                    seed=int(common["seed"]),
                    lgb_params={
                        "learning_rate": 0.08,
                        "num_leaves": 63,
                        "min_data_in_leaf": min_leaf,
                        "verbose": -1,
                        "num_threads": 0,
                    },
                ),
                action_ratio=ActionRatioConfig(
                    num_boost_round=60,
                    validation_fraction=0.2,
                    early_stopping_rounds=5,
                    refit_on_all_data=True,
                    show_progress=False,
                    lgb_params={"num_leaves": 31, "min_data_in_leaf": min_leaf, "verbose": -1, "num_threads": 0},
                ),
                transition_ratio=TransitionRatioConfig(
                    num_boost_round=80,
                    permutation_samples=5,
                    validation_fraction=0.2,
                    early_stopping_rounds=5,
                    refit_on_all_data=True,
                    lgb_params={"num_leaves": 63, "min_data_in_leaf": min_leaf, "verbose": -1, "num_threads": 0},
                    show_progress=False,
                ),
            )
        raw_weights = model.predict_state_action_ratio(
            one_hot(states_i, env.num_states),
            one_hot(actions_i, env.num_actions),
            clip=False,
        )
        weights = np.maximum(raw_weights, 0.0)
        discounted_weights = weights * (float(common["gamma"]) ** np.asarray(all_data.time_step, dtype=np.float64))
        estimate = _safe_weighted_reward(np.asarray(all_data.reward, dtype=np.float64), discounted_weights)
        base_row = {
            **common,
            "estimator": f"boosted_tree_{loss}",
            "status": "ok",
            "estimated_step_reward": estimate,
            "absolute_error": float(abs(estimate - common["target_step_reward"])),
            "runtime_sec": float(time.perf_counter() - start),
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_max": float(np.max(weights)),
            "weight_q99": float(np.quantile(weights, 0.99)),
            "effective_sample_size_fraction": float((weights.sum() ** 2) / np.maximum(np.sum(weights**2), 1e-12) / len(weights)),
            "negative_raw_fraction": float(np.mean(raw_weights < 0.0)),
            "trees_used": float(model.diagnostics.get("trees_used") or 0),
        }
        rows = [base_row]
        if config.include_bellman_moment_calibration:
            rows.append(
                _gridwalk_calibrated_row(
                    env,
                    target_policy,
                    common,
                    all_data,
                    weights,
                    raw_weights,
                    base_row,
                    estimator=f"boosted_tree_{loss}_bellman_moment_calibrated",
                    w_max=50.0,
                )
            )
        return rows
    except Exception as exc:
        return [_error_row(common, f"boosted_tree_{loss}", exc, start)]


def _run_neural_network_rows(
    env,
    behavior_data,
    target_policy,
    common: dict[str, Any],
    config: GridBenchmarkConfig,
    *,
    loss: str,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    try:
        all_data = behavior_data.get_all()
        states_i = np.asarray(all_data.state, dtype=np.int64).reshape(-1)
        actions_i = np.asarray(all_data.action, dtype=np.int64).reshape(-1)
        next_states_i = np.asarray(all_data.next_state, dtype=np.int64).reshape(-1)
        target_actions_i = target_policy.sample_action(states_i)
        states = one_hot(states_i, env.num_states)
        actions = one_hot(actions_i, env.num_actions)

        with _time_limit(config.estimator_timeout_sec):
            model = fit_discounted_occupancy_ratio_neural(
                states=states,
                actions=actions,
                next_states=one_hot(next_states_i, env.num_states),
                target_actions=one_hot(target_actions_i, env.num_actions),
                gamma=float(common["gamma"]),
                occupancy=NeuralOccupancyRegressionConfig(
                    num_iterations=int(config.neural_num_iterations),
                    gradient_steps_per_iteration=int(config.neural_gradient_steps_per_iteration),
                    mcmc_samples=int(config.neural_mcmc_samples),
                    batch_size=int(config.neural_batch_size),
                    hidden_dims=(int(config.neural_hidden_dim),),
                    loss=loss,
                    huber_delta_scale=float(config.huber_delta_scale),
                    fixed_point_damping=0.5,
                    normalize_occupancy=True,
                    occupancy_ratio_max=50.0,
                    clip_pseudo_outcomes=True,
                    normalize_transition_cache=True,
                    seed=int(common["seed"]),
                    show_progress=False,
                ),
                action_ratio=NeuralActionRatioConfig(
                    max_steps=int(config.neural_action_steps),
                    batch_size=int(config.neural_batch_size),
                    hidden_dims=(int(config.neural_hidden_dim),),
                    moment_calibration="scalar",
                    seed=int(common["seed"]) + 101,
                ),
                transition_ratio=NeuralTransitionRatioConfig(
                    max_steps=int(config.neural_transition_steps),
                    batch_size=int(config.neural_batch_size),
                    hidden_dims=(int(config.neural_hidden_dim),),
                    permutation_samples=int(config.neural_transition_permutation_samples),
                    moment_calibration="scalar",
                    seed=int(common["seed"]) + 202,
                ),
            )
        raw_weights = model.predict_state_action_ratio(states, actions, clip=False)
        weights = model.predict_state_action_ratio(states, actions, clip=True)
        discounted_weights = weights * (float(common["gamma"]) ** np.asarray(all_data.time_step, dtype=np.float64))
        estimate = _safe_weighted_reward(np.asarray(all_data.reward, dtype=np.float64), discounted_weights)
        base_row = {
            **common,
            "estimator": f"neural_network_{loss}",
            "status": "ok",
            "estimated_step_reward": estimate,
            "absolute_error": float(abs(estimate - common["target_step_reward"])),
            "runtime_sec": float(time.perf_counter() - start),
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_max": float(np.max(weights)),
            "weight_q99": float(np.quantile(weights, 0.99)),
            "effective_sample_size_fraction": float((weights.sum() ** 2) / np.maximum(np.sum(weights**2), 1e-12) / len(weights)),
            "negative_raw_fraction": float(np.mean(raw_weights < 0.0)),
            "neural_gradient_steps_used": float(model.diagnostics.get("gradient_steps_used") or 0),
        }
        rows = [base_row]
        if config.include_bellman_moment_calibration:
            rows.append(
                _gridwalk_calibrated_row(
                    env,
                    target_policy,
                    common,
                    all_data,
                    weights,
                    raw_weights,
                    base_row,
                    estimator=f"neural_network_{loss}_bellman_moment_calibrated",
                    w_max=50.0,
                )
            )
        return rows
    except Exception as exc:
        return [_error_row(common, f"neural_network_{loss}", exc, start)]


def _gridwalk_calibrated_row(
    env,
    target_policy,
    common: dict[str, Any],
    all_data,
    weights: np.ndarray,
    raw_weights: np.ndarray,
    base_row: dict[str, Any],
    *,
    estimator: str,
    w_max: float,
) -> dict[str, Any]:
    states_i = np.asarray(all_data.state, dtype=np.int64).reshape(-1)
    actions_i = np.asarray(all_data.action, dtype=np.int64).reshape(-1)
    next_states_i = np.asarray(all_data.next_state, dtype=np.int64).reshape(-1)
    next_target_actions_i = target_policy.sample_action(next_states_i)
    time_steps = np.asarray(all_data.time_step, dtype=np.int64).reshape(-1)
    init_states_i = states_i[time_steps == 0]
    if init_states_i.size == 0:
        init_states_i = states_i[:1]
    init_actions_i = target_policy.sample_action(init_states_i)

    h = _gridwalk_features(env, states_i, actions_i)
    h_next = _gridwalk_features(env, next_states_i, next_target_actions_i)
    init_moments = np.mean(_gridwalk_features(env, init_states_i, init_actions_i), axis=0)
    result = calibrate_occupancy_bellman_binning(
        omega_hat=weights,
        h=h,
        h_next=h_next,
        init_moments=init_moments,
        gamma=float(common["gamma"]),
        n_bins=10,
        min_bin_size=30,
        w_max=w_max,
        normalize=True,
        return_diagnostics=True,
    )
    calibrated_weights = np.asarray(result["omega_cal"], dtype=np.float64).reshape(-1)
    diag = result["diagnostics"]
    discounted_weights = calibrated_weights * (float(common["gamma"]) ** np.asarray(all_data.time_step, dtype=np.float64))
    estimate = _safe_weighted_reward(np.asarray(all_data.reward, dtype=np.float64), discounted_weights)
    row = {
        **base_row,
        "estimator": estimator,
        "estimated_step_reward": estimate,
        "absolute_error": float(abs(estimate - common["target_step_reward"])),
        "weight_mean": float(np.mean(calibrated_weights)),
        "weight_std": float(np.std(calibrated_weights)),
        "weight_max": float(np.max(calibrated_weights)),
        "weight_q99": float(np.quantile(calibrated_weights, 0.99)),
        "effective_sample_size_fraction": float(
            (calibrated_weights.sum() ** 2)
            / np.maximum(np.sum(calibrated_weights**2), 1e-12)
            / len(calibrated_weights)
        ),
        "negative_raw_fraction": float(np.mean(raw_weights < 0.0)),
        "bellman_moment_bin_count": float(np.asarray(diag["bin_counts"]).size),
        "bellman_moment_residual_norm_before": float(diag["residual_norm_before"]),
        "bellman_moment_residual_norm_after": float(diag["residual_norm_after"]),
        "bellman_moment_ess_before": float(diag["ess_before"]),
        "bellman_moment_ess_after": float(diag["ess_after"]),
        "bellman_moment_max_weight_after": float(diag["max_weight_after"]),
        "bellman_moment_residual_reduction_fraction": float(diag["residual_reduction_fraction"]),
        "bellman_moment_ess_loss_fraction": float(diag["ess_loss_fraction"]),
        "bellman_moment_q99_increase_fraction": float(diag["q99_increase_fraction"]),
        "bellman_moment_max_weight_increase_fraction": float(diag["max_weight_increase_fraction"]),
        "bellman_moment_recommendation": str(diag["calibration_recommendation"]),
        "bellman_moment_recommendation_reasons": " | ".join(
            str(reason) for reason in diag["calibration_recommendation_reasons"]
        ),
    }
    if "clipped_fraction" in diag:
        row["bellman_moment_clipped_fraction"] = float(diag["clipped_fraction"])
    return row


def _gridwalk_features(env, states_i: np.ndarray, actions_i: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.ones((int(states_i.shape[0]), 1), dtype=np.float64),
            one_hot(states_i, env.num_states),
            one_hot(actions_i, env.num_actions),
        ],
        axis=1,
    )


def _error_row(common: dict[str, Any], estimator: str, exc: Exception, start: float) -> dict[str, Any]:
    return {
        **common,
        "estimator": estimator,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "estimated_step_reward": "",
        "absolute_error": "",
        "runtime_sec": float(time.perf_counter() - start),
    }


def _safe_min_data_in_leaf(n_rows: int) -> int:
    return max(1, min(50, int(n_rows) // 4))


def _safe_weighted_reward(rewards: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float(np.sum(rewards * weights) / denom)


@contextmanager
def _time_limit(seconds: float):
    seconds = float(seconds)
    if seconds <= 0.0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handler(_signum, _frame):
        raise TimeoutError(f"Estimator exceeded {seconds:g} seconds.")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["estimator"], row["alpha"], row["gamma"], row["status"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (estimator, alpha, gamma, status), group in groups.items():
        out: dict[str, Any] = {
            "estimator": estimator,
            "alpha": alpha,
            "gamma": gamma,
            "status": status,
            "n_runs": len(group),
        }
        for metric in ("estimated_step_reward", "absolute_error", "runtime_sec", "effective_sample_size_fraction", "negative_raw_fraction"):
            vals = []
            for row in group:
                value = row.get(metric, "")
                if value == "":
                    continue
                vals.append(float(value))
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[f"{metric}_mean"] = float(np.mean(arr))
                out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        summary.append(out)
    _write_csv(path, summary)


def _add_google_research(path: Path) -> None:
    if not (path / "dual_dice" / "run.py").exists():
        raise FileNotFoundError(f"Missing Google DualDICE source at {path / 'dual_dice'}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the original Google DualDICE GridWalk benchmark.")
    parser.add_argument("--output-root", default="outputs/occupancy_ratio_benchmark_google_paper")
    parser.add_argument("--google-research-root", default="/tmp/google-research")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.5])
    parser.add_argument("--gammas", nargs="*", type=float, default=[0.9, 0.995])
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--max-trajectory-length", type=int, default=100)
    parser.add_argument("--boosted-losses", nargs="*", choices=("squared", "huber"), default=["huber"])
    parser.add_argument("--boosted-num-iterations", type=int, default=40)
    parser.add_argument("--boosted-mcmc-samples", type=int, default=24)
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument("--include-bellman-moment-calibration", action="store_true")
    parser.add_argument("--neural-num-iterations", type=int, default=20)
    parser.add_argument("--neural-mcmc-samples", type=int, default=16)
    parser.add_argument("--neural-gradient-steps-per-iteration", type=int, default=4)
    parser.add_argument("--neural-action-steps", type=int, default=80)
    parser.add_argument("--neural-transition-steps", type=int, default=120)
    parser.add_argument("--neural-transition-permutation-samples", type=int, default=1)
    parser.add_argument("--neural-batch-size", type=int, default=512)
    parser.add_argument("--neural-hidden-dim", type=int, default=8)
    parser.add_argument("--huber-delta-scale", type=float, default=1.345)
    parser.add_argument("--estimator-timeout-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GridBenchmarkConfig(
        output_root=Path(args.output_root),
        google_research_root=Path(args.google_research_root),
        seeds=tuple(args.seeds),
        alphas=tuple(args.alphas),
        gammas=tuple(args.gammas),
        num_trajectories=int(args.num_trajectories),
        max_trajectory_length=int(args.max_trajectory_length),
        boosted_losses=tuple(args.boosted_losses),
        boosted_num_iterations=int(args.boosted_num_iterations),
        boosted_mcmc_samples=int(args.boosted_mcmc_samples),
        include_neural=bool(args.include_neural),
        neural_num_iterations=int(args.neural_num_iterations),
        neural_mcmc_samples=int(args.neural_mcmc_samples),
        neural_gradient_steps_per_iteration=int(args.neural_gradient_steps_per_iteration),
        neural_action_steps=int(args.neural_action_steps),
        neural_transition_steps=int(args.neural_transition_steps),
        neural_transition_permutation_samples=int(args.neural_transition_permutation_samples),
        neural_batch_size=int(args.neural_batch_size),
        neural_hidden_dim=int(args.neural_hidden_dim),
        huber_delta_scale=float(args.huber_delta_scale),
        estimator_timeout_sec=float(args.estimator_timeout_sec),
        include_bellman_moment_calibration=bool(args.include_bellman_moment_calibration),
    )
    rows = run_gridwalk_benchmark(config)
    print(f"Wrote results: {config.output_root / 'gridwalk' / 'results.csv'}")
    print(f"Wrote summary: {config.output_root / 'gridwalk' / 'summary.csv'}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
