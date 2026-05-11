from __future__ import annotations

from experiments.occupancy_ratio.fori_eval.estimators import EstimatorOutput, run_population_fori
from experiments.occupancy_ratio.fori_eval.finite_mdp import FiniteDataset


def run_population_ablations(dataset: FiniteDataset, *, num_iterations: int = 30) -> list[EstimatorOutput]:
    """Run exact-grid factorization and stabilization ablations."""

    truth = dataset.truth
    return [
        run_population_fori(dataset, name="population_fori", num_iterations=num_iterations),
        run_population_fori(
            dataset,
            name="ablation_no_one_step_coverage",
            num_iterations=num_iterations,
            c_pi=truth.c_pi * 0.0 + 1.0,
        ),
        run_population_fori(
            dataset,
            name="ablation_no_backward_propagation",
            num_iterations=1,
            m_override="constant_one",
        ),
        run_population_fori(dataset, name="ablation_one_shot_update", num_iterations=1),
        run_population_fori(dataset, name="ablation_clipping_only", num_iterations=num_iterations, clip_max=50.0),
        run_population_fori(dataset, name="ablation_normalization_only", num_iterations=num_iterations, normalize=True),
        run_population_fori(dataset, name="ablation_damping_only", num_iterations=num_iterations, damping=0.5),
        run_population_fori(
            dataset,
            name="ablation_stabilized_clip_norm_damp",
            num_iterations=num_iterations,
            clip_max=50.0,
            normalize=True,
            damping=0.5,
        ),
    ]
