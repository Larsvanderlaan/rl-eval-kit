"""Cross-fitted GenPQR example."""

from __future__ import annotations

from genpqr import DiscreteNormalizationPolicy, GenPQRConfig, fit_genpqr_crossfit
from genpqr.benchmarks import make_tabular_chain


def main() -> None:
    dataset = make_tabular_chain(60, seed=4)
    result = fit_genpqr_crossfit(
        dataset=dataset,
        gamma=0.9,
        n_folds=3,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(policy="behavior_cloning_native", q="deeppqr_linear"),
    )
    print(result.diagnostics)


if __name__ == "__main__":
    main()
