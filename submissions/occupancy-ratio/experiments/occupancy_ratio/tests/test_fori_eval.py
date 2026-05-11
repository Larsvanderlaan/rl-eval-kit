from __future__ import annotations

import csv

import numpy as np

from experiments.occupancy_ratio.fori_eval.estimators import run_oracle, run_population_fori
from experiments.occupancy_ratio.fori_eval.finite_mdp import (
    exact_tabular_truth,
    make_random_finite_mdp,
    sample_finite_dataset,
)
from experiments.occupancy_ratio.fori_eval.runner import RunnerConfig, run


def test_exact_omega_star_is_bellman_fixed_point() -> None:
    mdp = make_random_finite_mdp(
        n_states=5,
        n_actions=2,
        transition_concentration=1.0,
        mismatch=1.0,
        overlap_floor=0.02,
        seed=11,
    )
    truth = exact_tabular_truth(mdp, gamma=0.9)
    residual = np.sum(truth.nu * np.abs(truth.omega_star - truth.bellman_update(truth.omega_star)))
    assert residual < 1e-9


def test_gamma_zero_ratio_equals_initial_ratio() -> None:
    mdp = make_random_finite_mdp(
        n_states=6,
        n_actions=3,
        transition_concentration=0.5,
        mismatch=1.5,
        overlap_floor=0.02,
        seed=12,
    )
    truth = exact_tabular_truth(mdp, gamma=0.0)
    np.testing.assert_allclose(truth.omega_star, truth.omega0, atol=1e-10, rtol=1e-10)


def test_gamma_zero_no_policy_shift_oracle_and_population_near_one() -> None:
    mdp = make_random_finite_mdp(
        n_states=5,
        n_actions=2,
        transition_concentration=1.0,
        mismatch=0.0,
        overlap_floor=0.02,
        seed=13,
    )
    dataset = sample_finite_dataset(mdp=mdp, gamma=0.0, sample_size=80, seed=14, n_reward_sweeps=4)
    oracle = run_oracle(dataset)
    population = run_population_fori(dataset, num_iterations=50)
    np.testing.assert_allclose(oracle.weights, np.ones_like(dataset.truth.omega_star), atol=1e-8, rtol=1e-8)
    assert population.diagnostics["ratio_l1_nu"] < 1e-3


def test_runner_smoke_writes_outputs(tmp_path) -> None:
    config = RunnerConfig.for_profile(
        "smoke",
        output_root=tmp_path,
        run_id="smoke",
        estimators=("oracle", "population"),
        sample_sizes=(80,),
        n_states=(5,),
        n_actions=(2,),
        gammas=(0.9,),
        transition_concentrations=(1.0,),
        mismatches=(1.0,),
        reward_sweeps=4,
        population_iterations=5,
        write_plots=False,
    )
    result = run(config)
    for key in ("results_path", "summary_path", "diagnostics_path", "manifest_path"):
        assert result[key].exists()
    with result["results_path"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    required = {
        "ratio_l1_nu",
        "bellman_flow_residual_l1",
        "reward_sweep_value_rmse",
        "effective_sample_size_fraction",
        "status",
        "seed",
        "sample_size",
        "gamma",
        "mismatch",
        "runtime_sec",
    }
    assert required.issubset(rows[0])
