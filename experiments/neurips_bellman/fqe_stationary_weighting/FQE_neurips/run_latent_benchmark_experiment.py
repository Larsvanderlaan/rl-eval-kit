from __future__ import annotations

import argparse
import json

from .latent_garnet_benchmark import (
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    evaluate_weight_estimators_on_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one latent Garnet stationary-weight benchmark experiment.")
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--behavior-coverage", type=float, default=0.3)
    parser.add_argument("--observation-mode", type=str, default="compact_nonlinear")
    parser.add_argument("--gamma-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark = build_latent_garnet_benchmark(
        LatentGarnetConfig(
            n_states=args.n_states,
            n_actions=args.n_actions,
            latent_dim=args.latent_dim,
            branching_factor=args.branching_factor,
            dataset_size=args.dataset_size,
            behavior_coverage=args.behavior_coverage,
            observation_mode=args.observation_mode,
            seed=args.seed,
        )
    )
    results = evaluate_weight_estimators_on_benchmark(
        benchmark=benchmark,
        gamma_ratio=args.gamma_ratio,
        seed=args.seed,
    )
    results["config"] = {
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "latent_dim": args.latent_dim,
        "branching_factor": args.branching_factor,
        "dataset_size": args.dataset_size,
        "behavior_coverage": args.behavior_coverage,
        "observation_mode": args.observation_mode,
        "gamma_ratio": args.gamma_ratio,
        "seed": args.seed,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
