"""Linear DeepPQR anchor-Q example."""

from __future__ import annotations

from genpqr import DeepPQRAnchorQEstimator, DiscreteNormalizationPolicy, GenPQRConfig, fit_genpqr
from genpqr.benchmarks import make_tabular_chain


def main() -> None:
    dataset = make_tabular_chain(50, seed=2)
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.8,
        normalization_policy=DiscreteNormalizationPolicy.anchor(2, 0),
        config=GenPQRConfig(policy="behavior_cloning_native", q=DeepPQRAnchorQEstimator(anchor_action=0)),
    )
    print(result.diagnostics["q_anchor_count"])


if __name__ == "__main__":
    main()
