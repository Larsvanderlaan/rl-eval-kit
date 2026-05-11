from __future__ import annotations

import argparse
import json

from .latent_garnet_benchmark import (
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    evaluate_fqe_methods_on_benchmark,
)


def run_fqe_coverage_study(
    coverages: list[float],
    n_states: int = 100,
    n_actions: int = 4,
    latent_dim: int = 3,
    branching_factor: int = 5,
    dataset_size: int = 10_000,
    data_mode: str = "mixed",
    n_trajectories: int = 200,
    iid_fraction: float = 0.5,
    observation_mode: str = "compact_nonlinear",
    gamma_ratio: float = 1.0,
    gamma_eval: float = 0.99,
    seed: int = 0,
) -> dict[str, object]:
    """Run the latent-benchmark FQE study across a sweep of behavior coverage values."""

    results = []
    for idx, coverage in enumerate(coverages):
        cfg = LatentGarnetConfig(
            n_states=n_states,
            n_actions=n_actions,
            latent_dim=latent_dim,
            branching_factor=branching_factor,
            dataset_size=dataset_size,
            data_mode=data_mode,
            n_trajectories=n_trajectories,
            iid_fraction=iid_fraction,
            behavior_coverage=coverage,
            observation_mode=observation_mode,
            seed=seed,
        )
        benchmark = build_latent_garnet_benchmark(cfg)
        evaluation = evaluate_fqe_methods_on_benchmark(
            benchmark=benchmark,
            gamma_eval=gamma_eval,
            gamma_ratio=gamma_ratio,
            seed=seed + idx,
        )
        evaluation["config"] = {
            "behavior_coverage": coverage,
            "n_states": n_states,
            "n_actions": n_actions,
            "latent_dim": latent_dim,
            "branching_factor": branching_factor,
            "dataset_size": dataset_size,
            "data_mode": data_mode,
            "n_trajectories": n_trajectories,
            "iid_fraction": iid_fraction,
            "observation_mode": observation_mode,
            "gamma_ratio": gamma_ratio,
            "gamma_eval": gamma_eval,
            "seed": seed,
            "dataset_seed": seed + idx,
        }
        results.append(evaluation)
    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the latent Garnet stationary-weighted FQE coverage study.")
    parser.add_argument("--coverages", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument("--n-states", type=int, default=80)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=6000)
    parser.add_argument("--data-mode", type=str, default="mixed")
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--iid-fraction", type=float, default=0.5)
    parser.add_argument("--observation-mode", type=str, default="compact_nonlinear")
    parser.add_argument("--gamma-ratio", type=float, default=1.0)
    parser.add_argument("--gamma-eval", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = run_fqe_coverage_study(
        coverages=args.coverages,
        n_states=args.n_states,
        n_actions=args.n_actions,
        latent_dim=args.latent_dim,
        branching_factor=args.branching_factor,
        dataset_size=args.dataset_size,
        data_mode=args.data_mode,
        n_trajectories=args.n_trajectories,
        iid_fraction=args.iid_fraction,
        observation_mode=args.observation_mode,
        gamma_ratio=args.gamma_ratio,
        gamma_eval=args.gamma_eval,
        seed=args.seed,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
