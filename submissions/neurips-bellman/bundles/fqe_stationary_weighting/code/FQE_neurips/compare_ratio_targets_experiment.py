from __future__ import annotations

import argparse
import json

from .latent_garnet_benchmark import (
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    evaluate_fqe_methods_on_benchmark,
)


def run_ratio_target_comparison(
    n_states: int = 60,
    n_actions: int = 4,
    latent_dim: int = 3,
    branching_factor: int = 5,
    dataset_size: int = 1000,
    behavior_coverage: float = 0.5,
    observation_mode: str = "compact_nonlinear",
    gamma_eval: float = 0.95,
    gamma_ratio_discounted: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """
    Compare stationary-ratio weighting against discounted-ratio weighting on the same benchmark.

    The output includes both:
    - stationary / target-policy error metrics, where stationary weighting is the most aligned,
    - initial-value metrics under the benchmark start-state distribution, where discounted ratios
      may be more aligned and/or more stable.
    """

    benchmark = build_latent_garnet_benchmark(
        LatentGarnetConfig(
            n_states=n_states,
            n_actions=n_actions,
            latent_dim=latent_dim,
            branching_factor=branching_factor,
            dataset_size=dataset_size,
            behavior_coverage=behavior_coverage,
            observation_mode=observation_mode,
            seed=seed,
        )
    )

    stationary = evaluate_fqe_methods_on_benchmark(
        benchmark=benchmark,
        gamma_eval=gamma_eval,
        gamma_ratio=1.0,
        seed=seed,
    )
    discounted = evaluate_fqe_methods_on_benchmark(
        benchmark=benchmark,
        gamma_eval=gamma_eval,
        gamma_ratio=gamma_ratio_discounted,
        seed=seed,
    )

    return {
        "config": {
            "n_states": n_states,
            "n_actions": n_actions,
            "latent_dim": latent_dim,
            "branching_factor": branching_factor,
            "dataset_size": dataset_size,
            "behavior_coverage": behavior_coverage,
            "observation_mode": observation_mode,
            "gamma_eval": gamma_eval,
            "gamma_ratio_stationary": 1.0,
            "gamma_ratio_discounted": gamma_ratio_discounted,
            "seed": seed,
        },
        "stationary_ratio": stationary,
        "discounted_ratio": discounted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stationary-ratio and discounted-ratio weighting on the latent Garnet PE benchmark."
    )
    parser.add_argument("--n-states", type=int, default=60)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--behavior-coverage", type=float, default=0.5)
    parser.add_argument("--observation-mode", type=str, default="compact_nonlinear")
    parser.add_argument("--gamma-eval", type=float, default=0.95)
    parser.add_argument("--gamma-ratio-discounted", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = run_ratio_target_comparison(
        n_states=args.n_states,
        n_actions=args.n_actions,
        latent_dim=args.latent_dim,
        branching_factor=args.branching_factor,
        dataset_size=args.dataset_size,
        behavior_coverage=args.behavior_coverage,
        observation_mode=args.observation_mode,
        gamma_eval=args.gamma_eval,
        gamma_ratio_discounted=args.gamma_ratio_discounted,
        seed=args.seed,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
