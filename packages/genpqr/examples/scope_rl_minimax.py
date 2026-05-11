"""Guarded SCOPE-RL minimax-Q diagnostic workflow example."""

from __future__ import annotations

import argparse

from genpqr import GenPQRAdapterError, GenPQRConfig, ScopeRLDatasetBoundQEstimator, fit_genpqr
from genpqr.benchmarks import make_tabular_chain


def run_scope_mql_diagnostics(dataset, env, evaluation_policies):
    """Use SCOPE-RL's dataset-bound public path for fitted-row diagnostics."""

    return fit_genpqr(
        dataset=dataset,
        gamma=0.99,
        config=GenPQRConfig(
            policy="behavior_cloning_native",
            q=ScopeRLDatasetBoundQEstimator(
                method="mql",
                env=env,
                evaluation_policies=evaluation_policies,
                allow_dataset_bound_predictions=True,
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-error", action="store_true", help="Run the dataset-bound guard demonstration.")
    args = parser.parse_args()
    if not args.demo_error:
        print("Provide env/evaluation_policies and call run_scope_mql_diagnostics(...).")
        return
    try:
        fit_genpqr(
            dataset=make_tabular_chain(16),
            gamma=0.9,
            config=GenPQRConfig(
                policy="behavior_cloning_native",
                q=ScopeRLDatasetBoundQEstimator(method="mql", env=object(), evaluation_policies=object()),
            ),
        )
    except GenPQRAdapterError as exc:
        print(f"SCOPE-RL guard error: {exc}")


if __name__ == "__main__":
    main()
