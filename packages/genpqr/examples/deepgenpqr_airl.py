"""DeepGenPQR AIRL + neural FQE workflow."""

from __future__ import annotations

import argparse

from genpqr import DeepGenPQRConfig, GenPQRConfigurationError, fit_deep_genpqr
from genpqr.benchmarks import make_tabular_chain


def run_deepgenpqr_airl(dataset, env):
    """Fit the default DeepGenPQR AIRL + neural FQE workflow."""

    return fit_deep_genpqr(
        dataset=dataset,
        gamma=0.99,
        env=env,
        config=DeepGenPQRConfig.from_preset("deepgenpqr_airl_fqe_balanced"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-error", action="store_true", help="Show the missing-env preflight error.")
    args = parser.parse_args()
    if not args.demo_error:
        print("Provide an environment and call run_deepgenpqr_airl(dataset, env).")
        return
    try:
        run_deepgenpqr_airl(make_tabular_chain(16), env=None)
    except GenPQRConfigurationError as exc:
        print(f"DeepGenPQR AIRL preflight error: {exc}")


if __name__ == "__main__":
    main()
