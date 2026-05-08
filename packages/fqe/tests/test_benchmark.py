from __future__ import annotations

import importlib.util

import numpy as np

from fqe_benchmark.adapters import estimator_registry
from fqe_benchmark.data import make_dataset
from fqe_benchmark.runner import run_benchmark
from fqe_benchmark.types import BenchmarkConfig


LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_tabular_gamma_zero_truth_matches_rewards() -> None:
    dataset = make_dataset(
        name="tabular_chain",
        sample_size=50,
        gamma=0.0,
        seed=0,
        n_eval=64,
        n_initial_eval=32,
    )
    q_train = dataset.true_q_fn(dataset.states, dataset.actions)
    assert np.allclose(q_train, dataset.rewards)
    assert np.isfinite(dataset.true_policy_value)


def test_linear_gaussian_dataset_shapes_and_truth() -> None:
    dataset = make_dataset(
        name="linear_gaussian",
        sample_size=40,
        gamma=0.5,
        seed=1,
        policy_shift=0.4,
        n_eval=32,
        n_initial_eval=16,
    )
    assert dataset.states.shape == (40, 2)
    assert dataset.actions.shape == (40, 1)
    assert dataset.next_actions.shape == (40, 1)
    q_eval = dataset.true_q_fn(dataset.target_eval_states, dataset.target_eval_actions)
    assert q_eval.shape == (32,)
    assert np.all(np.isfinite(q_eval))


def test_optional_adapter_preflight_reports_missing_or_unsupported() -> None:
    config = BenchmarkConfig.for_stage("smoke")
    registry = estimator_registry()
    preflight = registry["d3rlpy_fqe"].preflight(config, None)
    assert preflight.status in {"missing_dependency", "unsupported_setting", "ok"}
    google = registry["google_policy_eval_fqe_l2"].preflight(config, None)
    assert google.status in {"missing_dependency", "unsupported_setting"}


def test_fqe_benchmark_public_module() -> None:
    import importlib

    assert importlib.import_module("fqe.boosted").fit_fqe_lgbm
    assert importlib.import_module("fqe.neural").fit_fqe_neural
    assert importlib.import_module("fqe_benchmark.run").main


def test_benchmark_smoke_writes_outputs(tmp_path) -> None:
    estimators = ["controlled_linear_fqe"]
    if LIGHTGBM_AVAILABLE:
        estimators.append("ours_boosted_fqe")
    if TORCH_AVAILABLE:
        estimators.append("ours_neural_fqe")
    estimators.append("d3rlpy_fqe")
    config = BenchmarkConfig(
        stage="smoke",
        output_root=tmp_path,
        seeds=(0,),
        datasets=("tabular_chain",),
        estimators=tuple(estimators),
        sample_sizes=(50,),
        gammas=(0.0,),
        output_plots=False,
        boosted_num_iterations=6,
        neural_num_iterations=3,
        neural_gradient_steps_per_iteration=4,
    )
    result = run_benchmark(config)
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.diagnostics_path.exists()
    assert result.manifest_path.exists()
    statuses = {row["estimator"]: row["status"] for row in result.rows}
    assert statuses["controlled_linear_fqe"] == "ok"
    assert statuses["d3rlpy_fqe"] in {"missing_dependency", "unsupported_setting", "ok"}
    ok_rows = [row for row in result.rows if row["status"] == "ok"]
    assert ok_rows
    assert all("policy_value_absolute_error" in row for row in ok_rows)
