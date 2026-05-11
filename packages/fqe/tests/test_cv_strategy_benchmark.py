from __future__ import annotations

import csv
import importlib.util

import pytest

from tools import fqe_cv_strategy_benchmark as bench


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_fqe_cv_strategy_ladder_metadata_orders_large_mlp_by_parameter_count() -> None:
    candidates = bench._candidate_grid(input_dim=4, max_candidates=10)
    by_dims = {tuple(row["hidden_dims"]): row for row in candidates}

    rank_64 = by_dims[(64, 64)]["_meta"]["complexity_rank"]
    rank_256 = by_dims[(256, 256)]["_meta"]["complexity_rank"]

    assert rank_64 < rank_256
    assert by_dims[(64, 64)]["_meta"]["complexity_group"] == "fqe_neural_param_ladder"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
def test_fqe_cv_strategy_runner_smoke_and_oracle_reporting_only(tmp_path) -> None:
    args = bench._parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--discrete-datasets",
            "tabular_chain",
            "--skip-linear",
            "--skip-gym",
            "--synthetic-sample-sizes",
            "48",
            "--synthetic-gammas",
            "0.0",
            "--synthetic-seeds",
            "0",
            "--stage-counts",
            "1",
            "2",
            "--selectors",
            "staged_k1",
            "staged_k3",
            "naive_final_bellman_cv",
            "oracle_best",
            "--max-candidates",
            "2",
            "--cv-folds",
            "2",
            "--bootstrap",
            "0",
            "--final-iterations",
            "1",
            "--gradient-steps-per-iteration",
            "1",
            "--batch-size",
            "32",
            "--n-eval",
            "32",
            "--n-initial-eval",
            "16",
        ]
    )

    result = bench.run_benchmark(args)

    assert result["selector_rows"]
    assert result["candidate_truth_rows"]
    assert result["aggregate_rows"]
    for filename in (
        "selector_rows.csv",
        "candidate_truth_rows.csv",
        "stage_rows.csv",
        "aggregate_by_strategy.csv",
        "paired_deltas.csv",
        "bootstrap_ci.csv",
        "family_guardrails.csv",
        "recommendation.json",
        "recommendation.md",
        "promotion_decision.json",
        "promotion_decision.md",
        "run_summary.json",
    ):
        assert (tmp_path / filename).exists()

    selector_rows = list(csv.DictReader((tmp_path / "selector_rows.csv").open()))
    candidate_rows = list(csv.DictReader((tmp_path / "candidate_truth_rows.csv").open()))
    oracle_rows = [row for row in selector_rows if row["selector"] == "oracle_best"]
    non_oracle = [row for row in selector_rows if row["selector"] != "oracle_best"]

    assert oracle_rows
    assert all(row["selection_source"] == "reporting_only" for row in oracle_rows)
    assert all(row["selection_source"] == "proxy_cv" for row in non_oracle)

    best_candidate = min(candidate_rows, key=lambda row: float(row["policy_value_abs_error"]))
    assert oracle_rows[0]["selected_candidate_id"] == best_candidate["candidate_id"]
    assert float(oracle_rows[0]["oracle_regret"]) == pytest.approx(0.0)
    assert result["run_summary"]["selector_rows_complete"]
    assert result["run_summary"]["candidate_truth_rows_complete"]
    assert result["paired_rows"]
    assert result["bootstrap_ci_rows"]
    assert result["promotion_decision"]["target_selector"] == "staged_k3"


def test_fqe_cv_strategy_aggregate_recommends_non_oracle_selector() -> None:
    rows = [
        {
            "cell_id": "a",
            "dataset_family": "discrete",
            "selector": "staged_k3",
            "policy_value_abs_error": 1.0,
            "oracle_regret": 0.0,
            "runtime_sec": 2.0,
        },
        {
            "cell_id": "a",
            "dataset_family": "discrete",
            "selector": "oracle_best",
            "policy_value_abs_error": 1.0,
            "oracle_regret": 0.0,
            "runtime_sec": 0.0,
        },
        {
            "cell_id": "a",
            "dataset_family": "discrete",
            "selector": "naive_final_bellman_cv",
            "policy_value_abs_error": 2.0,
            "oracle_regret": 1.0,
            "runtime_sec": 1.0,
        },
    ]

    aggregate = bench._aggregate_by_strategy(rows)
    recommendation = bench._recommend_default(rows, aggregate)

    assert recommendation["recommended_selector"] == "staged_k3"
    assert recommendation["recommended_selector"] != "oracle_best"


def test_fqe_cv_strategy_profile_smoke_sets_tiny_defaults() -> None:
    args = bench._parse_args(["--profile", "smoke", "--output-dir", "custom-out"])

    assert args.output_dir == "custom-out"
    assert args.selectors == ("staged_k1", "staged_k3", "naive_final_bellman_cv", "oracle_best")
    assert args.synthetic_sample_sizes == (48,)
    assert args.max_candidates == 2


def test_fqe_cv_strategy_promotion_requires_family_guardrails() -> None:
    aggregate = [
        {
            "dataset_family": "all",
            "selector": "staged_k3",
            "median_oracle_regret": 0.1,
            "median_abs_error": 0.1,
        },
        {
            "dataset_family": "all",
            "selector": "naive_final_bellman_cv",
            "median_oracle_regret": 0.2,
            "median_abs_error": 0.2,
        },
        {
            "dataset_family": "all",
            "selector": "product_composite_cv",
            "median_oracle_regret": 0.3,
            "median_abs_error": 0.3,
        },
    ]
    guardrails = [{"guardrail": "discrete_mean_oracle_regret", "passed": 0.0}]

    decision = bench._promotion_decision(
        [],
        aggregate,
        guardrails,
        target_selector="staged_k3",
        non_stage_selectors=("naive_final_bellman_cv", "product_composite_cv"),
    )

    assert decision["conditions"]["non_stage_improvement_at_least_15pct"]
    assert not decision["conditions"]["family_guardrails_pass"]
    assert not decision["promote"]
