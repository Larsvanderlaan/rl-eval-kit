from __future__ import annotations

import argparse
import json

from .latent_garnet_benchmark import (
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    evaluate_weight_estimators_on_benchmark,
)


def run_coverage_sweep(
    coverages: list[float],
    n_states: int = 100,
    n_actions: int = 4,
    latent_dim: int = 3,
    branching_factor: int = 5,
    dataset_size: int = 10_000,
    gamma_ratio: float = 1.0,
    seed: int = 0,
) -> dict:
    results = []
    for idx, coverage in enumerate(coverages):
        cfg = LatentGarnetConfig(
            n_states=n_states,
            n_actions=n_actions,
            latent_dim=latent_dim,
            branching_factor=branching_factor,
            dataset_size=dataset_size,
            behavior_coverage=coverage,
            seed=seed,
        )
        benchmark = build_latent_garnet_benchmark(cfg)
        evaluation = evaluate_weight_estimators_on_benchmark(benchmark, gamma_ratio=gamma_ratio, seed=seed + idx)
        evaluation["config"] = {
            "behavior_coverage": coverage,
            "n_states": n_states,
            "n_actions": n_actions,
            "dataset_size": dataset_size,
            "gamma_ratio": gamma_ratio,
            "seed": seed,
            "dataset_seed": seed + idx,
        }
        results.append(evaluation)
    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep stationary behavior coverage on the latent Garnet benchmark.")
    parser.add_argument("--coverages", type=float, nargs="+", default=[1.0, 0.7, 0.4, 0.2])
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--gamma-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = run_coverage_sweep(
        coverages=args.coverages,
        n_states=args.n_states,
        n_actions=args.n_actions,
        latent_dim=args.latent_dim,
        branching_factor=args.branching_factor,
        dataset_size=args.dataset_size,
        gamma_ratio=args.gamma_ratio,
        seed=args.seed,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
