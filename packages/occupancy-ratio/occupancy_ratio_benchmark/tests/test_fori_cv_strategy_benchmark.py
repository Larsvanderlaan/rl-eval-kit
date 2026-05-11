from __future__ import annotations

import csv
import importlib.util

import pytest

from tools import fori_cv_strategy_benchmark as bench


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_fori_cv_strategy_ladder_metadata_orders_large_mlp_by_parameter_count() -> None:
    candidates = bench._candidate_grid(input_dim=4, max_candidates=10)
    by_dims = {tuple(row["occupancy"]["hidden_dims"]): row for row in candidates}

    rank_64 = by_dims[(64, 64)]["_meta"]["complexity_rank"]
    rank_256 = by_dims[(256, 256)]["_meta"]["complexity_rank"]

    assert rank_64 < rank_256
    assert by_dims[(64, 64)]["_meta"]["complexity_group"] == "fori_neural_occupancy_param_ladder"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
def test_fori_cv_strategy_runner_smoke_and_oracle_reporting_only(tmp_path) -> None:
    args = bench._parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--settings",
            "discrete_chain",
            "--controlled-sample-sizes",
            "32",
            "--controlled-gammas",
            "0.0",
            "--seeds",
            "0",
            "--selectors",
            "staged_k1",
            "staged_k3",
            "naive_final_bellman_cv",
            "bellman_gmm_ratio",
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
            "--mcmc-samples",
            "2",
            "--batch-size",
            "32",
            "--nuisance-hidden-dims",
            "8",
            "--action-steps",
            "5",
            "--source-steps",
            "5",
            "--transition-steps",
            "5",
            "--transition-permutation-samples",
            "1",
            "--direct-adjoint-steps",
            "4",
            "--direct-one-step-steps",
            "4",
            "--source-state-correction-mode",
            "always",
            "--analysis-bootstrap",
            "5",
        ]
    )

    result = bench.run_benchmark(args)

    assert result["selector_rows"]
    assert result["candidate_truth_rows"]
    assert result["stage_rows"]
    for filename in (
        "selector_rows.csv",
        "candidate_truth_rows.csv",
        "stage_rows.csv",
        "aggregate_by_strategy.csv",
        "paired_deltas.csv",
        "bootstrap_ci.csv",
        "family_guardrails.csv",
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
    assert all(row["selection_source"] != "reporting_only" for row in non_oracle)

    best_candidate = min(candidate_rows, key=lambda row: float(row["ope_value_abs_error"]))
    assert oracle_rows[0]["selected_candidate_id"] == best_candidate["candidate_id"]
    assert float(oracle_rows[0]["oracle_regret"]) == pytest.approx(0.0)
    assert result["run_summary"]["selector_rows_complete"]
    assert result["run_summary"]["candidate_truth_rows_complete"]
    assert result["promotion_decision"]["target_selector"] == "staged_k3"
    assert any(row.get("initial_ratio_mode") for row in candidate_rows)


def test_fori_cv_strategy_promotion_requires_gmm_and_guardrails() -> None:
    aggregate = [
        {"dataset_family": "all", "selector": "staged_k3", "median_oracle_regret": 0.10},
        {"dataset_family": "all", "selector": "product_composite_cv", "median_oracle_regret": 0.20},
        {"dataset_family": "all", "selector": "bellman_gmm_ratio", "median_oracle_regret": 0.15},
        {"dataset_family": "all", "selector": "bellman_gmm_ope", "median_oracle_regret": 0.18},
    ]
    guardrails = [{"guardrail": "collapse", "passed": 0.0}]

    decision = bench._promotion_decision([], aggregate, guardrails, target_selector="staged_k3")

    assert decision["conditions"]["beats_product_and_gmm_median_regret"]
    assert not decision["conditions"]["guardrails_pass"]
    assert not decision["promote"]


def test_fori_cv_strategy_profile_smoke_sets_tiny_defaults() -> None:
    args = bench._parse_args(["--profile", "smoke", "--output-dir", "custom-out"])

    assert args.output_dir == "custom-out"
    assert args.settings == ("discrete_chain",)
    assert args.selectors == ("staged_k1", "staged_k3", "naive_final_bellman_cv", "bellman_gmm_ratio", "oracle_best")
    assert args.max_candidates == 2
