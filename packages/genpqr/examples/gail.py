"""GAIL workflow example with clear optional-dependency errors."""

from __future__ import annotations

import argparse

from genpqr import GenPQRConfig, GenPQRConfigurationError, fit_genpqr
from genpqr.benchmarks import make_tabular_chain


def run_gail(dataset, env):
    """Fit GAIL + neural FQE when imitation/SB3/Torch are installed."""

    return fit_genpqr(
        dataset=dataset,
        gamma=0.99,
        env=env,
        config=GenPQRConfig.from_preset("gail_balanced"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-error", action="store_true", help="Run the missing-env preflight demonstration.")
    args = parser.parse_args()
    if not args.demo_error:
        print("Provide an environment and call run_gail(dataset, env). Use --demo-error to see preflight behavior.")
        return
    try:
        run_gail(make_tabular_chain(16), env=None)
    except GenPQRConfigurationError as exc:
        print(f"GAIL preflight error: {exc}")


if __name__ == "__main__":
    main()
