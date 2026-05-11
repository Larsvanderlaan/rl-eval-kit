from pathlib import Path

from causal_ope_benchmark import DifficultyStressStudyConfig, run_difficulty_study


def main() -> None:
    config = DifficultyStressStudyConfig.for_scale(
        "ci",
        output_root=Path("outputs/causal_ope_benchmark_examples"),
    )
    result = run_difficulty_study(
        config,
        difficulties=("easy", "medium"),
        families=("streamretain", "clinic_dtr"),
        methods=("direct_method", "neural_fqe_auto", "discounted_occupancy_neural_auto"),
        sample_sizes=(60,),
        mc_truth_rollouts=8,
    )
    print(result.readout_path)


if __name__ == "__main__":
    main()
