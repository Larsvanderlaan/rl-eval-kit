"""Run the smallest neural calibration workflow."""

from pathlib import Path

from causal_ope_benchmark import CalibrationStudyConfig, load_calibration_results, run_calibration


def main() -> None:
    config = CalibrationStudyConfig.for_preset("smoke", output_root=Path("outputs/causal_ope_benchmark_examples"))
    config = CalibrationStudyConfig(
        **{
            **config.__dict__,
            "families": ("streamretain",),
            "sample_sizes": (30,),
            "target_policies": ("moderate",),
            "mc_truth_rollouts": 4,
            "estimators": ("neural_fqe",),
        }
    )
    result = run_calibration(config)
    bundle = load_calibration_results(result.output_dir)
    print(result.readout_path)
    for row in bundle.results:
        print(row["estimator"], row["tuning_track"], row["status"])


if __name__ == "__main__":
    main()
