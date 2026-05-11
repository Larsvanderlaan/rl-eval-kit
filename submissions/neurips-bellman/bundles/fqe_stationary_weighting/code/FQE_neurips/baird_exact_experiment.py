from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BairdExactConfig:
    """Exact 7-state Baird-style benchmark for linear FQE contraction experiments."""

    gamma: float = 0.95
    n_samples: int = 50_000
    n_iters: int = 50
    ridge: float = 1e-6
    n_reps: int = 20
    alphas: tuple[float, ...] = (1.0, 0.975, 0.75, 0.1)
    base_seed: int = 123
    init_mode: str = "hub_bias"


def make_baird_features() -> np.ndarray:
    """
    Standard 7D Baird feature map.

    States:
    - 0 = hub
    - 1..6 = spokes
    """

    phi = np.zeros((7, 7), dtype=np.float64)
    for i in range(1, 7):
        phi[i, i - 1] = 2.0
        phi[i, -1] = 1.0
    phi[0, :6] = 2.0
    phi[0, -1] = 2.0
    return phi


PHI_BAIRD = make_baird_features()
MU_TARGET = np.zeros(7, dtype=np.float64)
MU_TARGET[0] = 1.0


def step_baird(state: int, action: int, rng: np.random.Generator) -> int:
    """
    Baird-style hub-and-spokes dynamics.

    Actions:
    - 0 = solid: deterministically go to hub
    - 1 = dashed: go to a random spoke
    """

    if action == 0:
        return 0
    return int(rng.integers(1, 7))


def simulate_baird_dataset(alpha: float, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a behavior-policy trajectory.

    The behavior chooses:
    - solid with probability `alpha`
    - dashed with probability `1 - alpha`
    Rewards are identically zero, so the true value is Q^pi = 0.
    """

    rng = np.random.default_rng(seed)
    states = np.zeros(n_samples, dtype=np.int64)
    next_states = np.zeros(n_samples, dtype=np.int64)
    s = int(rng.integers(0, 7))
    for t in range(n_samples):
        states[t] = s
        action = 0 if rng.random() < alpha else 1
        s_next = step_baird(s, action, rng)
        next_states[t] = s_next
        s = s_next
    return states, next_states


def initial_theta(init_mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if init_mode == "random":
        return rng.normal(size=7)
    if init_mode == "hub_bias":
        theta = np.zeros(7, dtype=np.float64)
        theta[-1] = 10.0
        return theta
    raise ValueError(f"Unsupported init_mode '{init_mode}'.")


def value_from_theta(theta: np.ndarray) -> np.ndarray:
    return PHI_BAIRD @ theta


def fqe_linear_baird(
    states: np.ndarray,
    next_states: np.ndarray,
    gamma: float,
    n_iters: int,
    ridge: float,
    weighted: bool,
    theta0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact linear FQE update used in the Baird notebook experiment.

    Weighted FQE uses the exact stationary ratio, which places all mass on the hub.
    This isolates the norm-mismatch effect rather than density-ratio estimation error.
    """

    theta = np.asarray(theta0, dtype=np.float64).copy()
    errors = []
    x_all = PHI_BAIRD[states]

    counts = np.bincount(states, minlength=7)
    rho_hat = counts / max(counts.sum(), 1)

    if weighted:
        idx = np.where(states == 0)[0]
        if idx.size == 0:
            idx = np.arange(len(states))
        x = x_all[idx]
    else:
        idx = np.arange(len(states))
        x = x_all

    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
    proj = np.linalg.solve(gram, x.T)

    for _ in range(n_iters):
        v = value_from_theta(theta)
        targets = gamma * v[next_states]
        theta = proj @ targets[idx]
        v = value_from_theta(theta)
        errors.append(float(np.sqrt(np.sum(MU_TARGET * v**2))))

    return np.asarray(errors, dtype=np.float64), theta


def evaluate_baird_exact(config: BairdExactConfig | None = None) -> dict[str, object]:
    if config is None:
        config = BairdExactConfig()

    alpha_results: list[dict[str, object]] = []
    for j, alpha in enumerate(config.alphas):
        unweighted_runs = []
        weighted_runs = []
        for m in range(config.n_reps):
            seed = config.base_seed + 1000 * j + m
            states, next_states = simulate_baird_dataset(alpha, config.n_samples, seed)
            theta0 = initial_theta(config.init_mode, seed)
            err_u, theta_u = fqe_linear_baird(
                states, next_states, config.gamma, config.n_iters, config.ridge, False, theta0
            )
            err_w, theta_w = fqe_linear_baird(
                states, next_states, config.gamma, config.n_iters, config.ridge, True, theta0
            )
            unweighted_runs.append(err_u)
            weighted_runs.append(err_w)

        unweighted_arr = np.vstack(unweighted_runs)
        weighted_arr = np.vstack(weighted_runs)
        alpha_results.append(
            {
                "alpha": float(alpha),
                "unweighted_final_mean": float(unweighted_arr[:, -1].mean()),
                "weighted_final_mean": float(weighted_arr[:, -1].mean()),
                "unweighted_final_std": float(unweighted_arr[:, -1].std()),
                "weighted_final_std": float(weighted_arr[:, -1].std()),
                "improvement_ratio": float(unweighted_arr[:, -1].mean() / max(weighted_arr[:, -1].mean(), 1e-12)),
                "unweighted_curve_mean": unweighted_arr.mean(axis=0).tolist(),
                "weighted_curve_mean": weighted_arr.mean(axis=0).tolist(),
            }
        )

    return {
        "config": {
            "gamma": config.gamma,
            "n_samples": config.n_samples,
            "n_iters": config.n_iters,
            "ridge": config.ridge,
            "n_reps": config.n_reps,
            "alphas": list(config.alphas),
            "init_mode": config.init_mode,
        },
        "results": alpha_results,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the exact Baird-style linear FQE comparison.")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.975, 0.75, 0.1])
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--init-mode", type=str, default="hub_bias")
    args = parser.parse_args()

    out = evaluate_baird_exact(
        BairdExactConfig(
            gamma=args.gamma,
            n_samples=args.n_samples,
            n_iters=args.n_iters,
            ridge=args.ridge,
            n_reps=args.n_reps,
            alphas=tuple(args.alphas),
            base_seed=args.base_seed,
            init_mode=args.init_mode,
        )
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
