"""BC + FQE GenPQR example."""

from __future__ import annotations

from genpqr import DiscreteNormalizationPolicy, GenPQRConfig, fit_genpqr
from genpqr.benchmarks import make_tabular_chain


def main() -> None:
    dataset = make_tabular_chain(40, seed=1)
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.9,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(policy="behavior_cloning_native", q="deeppqr_linear"),
    )
    print(result.predict_reward(dataset.states[:3], dataset.actions[:3]))


if __name__ == "__main__":
    main()
