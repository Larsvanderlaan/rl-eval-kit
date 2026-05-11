"""Run the optional EpiCare adapter when EpiCare/Gym are installed."""

from pathlib import Path

from causal_ope_benchmark import CausalOPEBenchmarkConfig, run_suite


def main() -> None:
    config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root=Path("outputs/causal_ope_benchmark_examples"))
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "families": ("epicare",),
            "sample_sizes": (20,),
            "target_policies": ("moderate",),
            "mc_truth_rollouts": 2,
            "estimators": ("direct_method",),
        }
    )
    result = run_suite(config)
    print(result.readout_path)
    for row in result.rows:
        print(row["family"], row["estimator"], row["status"], row.get("skip_reason", ""))


if __name__ == "__main__":
    main()
