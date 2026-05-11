"""Run a tiny StreamRetain OPE smoke benchmark."""

from pathlib import Path

from causal_ope_benchmark import CausalOPEBenchmarkConfig, load_results, run_suite


def main() -> None:
    config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root=Path("outputs/causal_ope_benchmark_examples"))
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "families": ("streamretain",),
            "sample_sizes": (80,),
            "target_policies": ("moderate",),
            "mc_truth_rollouts": 8,
            "estimators": ("direct_method", "ipw", "snipw"),
        }
    )
    result = run_suite(config)
    bundle = load_results(result.output_dir)
    print(result.readout_path)
    for row in bundle.results:
        print(row["estimator"], row["status"], row.get("policy_value_estimate"))


if __name__ == "__main__":
    main()
