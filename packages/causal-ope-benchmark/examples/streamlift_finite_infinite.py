"""Run StreamLift with finite and discounted infinite-horizon estimands."""

from pathlib import Path

from causal_ope_benchmark import CausalOPEBenchmarkConfig, load_results, run_suite


def main() -> None:
    config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root=Path("outputs/causal_ope_benchmark_examples"))
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "families": ("streamlift",),
            "sample_sizes": (60,),
            "observed_horizons": (3,),
            "mc_truth_rollouts": 8,
            "estimators": ("streamlift_stratified_gcomp",),
            "streamlift_include_infinite_horizon": True,
            "streamlift_infinite_horizon_max_steps": 80,
        }
    )
    result = run_suite(config)
    bundle = load_results(result.output_dir)
    first = bundle.results[0]
    print(result.readout_path)
    print("finite effect:", first.get("effect_horizon_36_estimate"))
    print("infinite effect:", first.get("effect_horizon_infinite_estimate"))


if __name__ == "__main__":
    main()
