from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .hard_benchmark_experiment import evaluate_hard_benchmark_setting


@dataclass
class HardBenchmarkConsistencyConfig:
    """Sample-size sweep for the tuned hard benchmark."""

    behavior_solid_prob: float = 0.30
    gamma_eval: float = 0.97
    n_outer_iters: int = 5
    solver: str = "direct"
    data_mode: str = "stationary_iid"
    dataset_sizes: tuple[int, ...] = (600, 1000, 1500, 2500, 4000)
    seeds: int = 5


def run_hard_benchmark_consistency(
    config: HardBenchmarkConsistencyConfig | None = None,
) -> dict[str, object]:
    if config is None:
        config = HardBenchmarkConsistencyConfig()

    methods = [
        "unweighted",
        "oracle",
        "weighted_policy_ratio",
        "weighted_linear_basic",
        "weighted_linear_flexible",
    ]
    sweep = []
    for dataset_size in config.dataset_sizes:
        per_seed = [
            evaluate_hard_benchmark_setting(
                behavior_solid_prob=config.behavior_solid_prob,
                dataset_size=dataset_size,
                gamma_eval=config.gamma_eval,
                n_outer_iters=config.n_outer_iters,
                seed=seed,
                include_secondary=False,
                include_rkhs=False,
                linear_solver=config.solver,
                data_mode=config.data_mode,
            )
            for seed in range(config.seeds)
        ]
        median_relative_target = {
            method: float(
                np.median(
                    [
                        result["linear_metrics"][method]["target_policy_relative_rmse"]
                        for result in per_seed
                    ]
                )
            )
            for method in methods
        }
        median_relative_initial = {
            method: float(
                np.median(
                    [
                        result["linear_metrics"][method]["initial_policy_value_relative_abs_error"]
                        for result in per_seed
                    ]
                )
            )
            for method in methods
        }
        oracle_curve = {
            key: float(
                np.median(
                    [
                        result["linear_metrics"]["oracle"]["target_curve_diagnostics"][key]
                        for result in per_seed
                    ]
                )
            )
            for key in per_seed[0]["linear_metrics"]["oracle"]["target_curve_diagnostics"].keys()
        }
        sweep.append(
            {
                "dataset_size": int(dataset_size),
                "median_relative_target_rmse": median_relative_target,
                "median_relative_initial_value_error": median_relative_initial,
                "oracle_gain_fraction": float(
                    (median_relative_target["unweighted"] - median_relative_target["oracle"])
                    / max(median_relative_target["unweighted"], 1e-12)
                ),
                "linear_basic_gain_fraction": float(
                    (median_relative_target["unweighted"] - median_relative_target["weighted_linear_basic"])
                    / max(median_relative_target["unweighted"], 1e-12)
                ),
                "oracle_curve": oracle_curve,
            }
        )

    return {
        "config": {
            "behavior_solid_prob": config.behavior_solid_prob,
            "gamma_eval": config.gamma_eval,
            "n_outer_iters": config.n_outer_iters,
            "solver": config.solver,
            "data_mode": config.data_mode,
            "dataset_sizes": list(config.dataset_sizes),
            "seeds": config.seeds,
        },
        "sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-size consistency sweep for the tuned hard benchmark.")
    parser.add_argument("--behavior-solid-prob", type=float, default=0.30)
    parser.add_argument("--gamma-eval", type=float, default=0.97)
    parser.add_argument("--n-outer-iters", type=int, default=5)
    parser.add_argument("--solver", type=str, default="direct", choices=["direct", "iterative"])
    parser.add_argument("--data-mode", type=str, default="stationary_iid", choices=["stationary_iid", "trajectory"])
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=[600, 1000, 1500, 2500, 4000])
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    output = run_hard_benchmark_consistency(
        HardBenchmarkConsistencyConfig(
            behavior_solid_prob=args.behavior_solid_prob,
            gamma_eval=args.gamma_eval,
            n_outer_iters=args.n_outer_iters,
            solver=args.solver,
            data_mode=args.data_mode,
            dataset_sizes=tuple(args.dataset_sizes),
            seeds=args.seeds,
        )
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
