"""Neural DeepPQR example using the lazy Torch backend."""

from __future__ import annotations

from genpqr import DiscreteNormalizationPolicy, GenPQRConfig, NeuralDeepPQRAnchorQEstimator, fit_genpqr
from genpqr.benchmarks import make_tabular_chain


def main() -> None:
    dataset = make_tabular_chain(80, seed=3)
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.8,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(
            policy="behavior_cloning_native",
            q=NeuralDeepPQRAnchorQEstimator(anchor_action=0, hidden_dims=(16,), max_epochs=5, patience=2),
        ),
    )
    print(result.diagnostics["q_backend"])


if __name__ == "__main__":
    main()
