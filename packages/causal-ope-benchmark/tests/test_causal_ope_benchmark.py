from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import causal_ope_benchmark as cob
from causal_ope_benchmark.baselines import EstimatorResult, run_estimator
from causal_ope_benchmark.calibration import CalibrationStudyConfig, _occupancy_value, run_calibration_study
from causal_ope_benchmark.adapters import (
    to_effect_panel,
    to_fqe_dataset,
    to_occupancy_ratio_dataset,
    to_survival_panel,
)
from causal_ope_benchmark.config import (
    CausalOPEBenchmarkConfig,
    DomainScenario,
    scenarios_for_profile,
    sensitivity_scenarios_for_profile,
)
from causal_ope_benchmark.difficulty import difficulty_cells, scenarios_for_difficulty
from causal_ope_benchmark.gym_envs import FixedPolicyWrapper, make_gym_env
from causal_ope_benchmark.metrics import score_result
from causal_ope_benchmark.policies import get_fixed_policy
from causal_ope_benchmark.runner import run_benchmark
from causal_ope_benchmark.scope_rl import to_scope_rl_logged_dataset
from causal_ope_benchmark.simulators import make_clinic_dtr_problem, make_streamlift_problem, make_streamretain_problem
from causal_ope_benchmark.validation import (
    validate_calibration_output_bundle,
    validate_difficulty_output_bundle,
    validate_output_bundle,
    validate_problem,
    validate_scope_rl_logged_dataset,
)


def test_public_api_facade_and_package_import_are_stable(tmp_path: Path) -> None:
    assert cob.package_version() == "0.2.0"
    families = cob.list_families()
    assert {family.name for family in families} == {"streamlift", "streamretain", "clinic_dtr", "epicare"}
    retain = cob.describe_family("streamretain")
    assert retain.display_name == "StreamRetain"
    assert retain.default_profile_member
    assert "moderate" in cob.list_target_policies("streamretain")
    estimators = cob.list_estimators()
    assert any(estimator.name == "neural_fqe" and estimator.optional_dependency for estimator in estimators)
    difficulties = cob.list_difficulties()
    assert {difficulty.name for difficulty in difficulties} == {"easy", "medium", "hard", "realistic"}
    assert cob.describe_difficulty("hard").overlap == "weak"
    estimator_names = {estimator.name for estimator in estimators}
    assert {
        "boosted_fqe_auto",
        "neural_fqe_auto",
        "discounted_occupancy_boosted",
        "discounted_occupancy_neural",
        "discounted_occupancy_boosted_auto",
        "discounted_occupancy_neural_auto",
        "oracle_selected_fqe_diagnostic",
        "oracle_selected_discounted_occupancy_diagnostic",
    } <= estimator_names
    problem = cob.make_benchmark_problem(
        "streamlift",
        sample_size=20,
        gamma=0.9,
        seed=5,
        observed_horizon=2,
        config=CausalOPEBenchmarkConfig.for_profile("smoke", output_root=tmp_path),
    )
    assert problem.dataset.family == "streamlift"
    validate_problem(problem)
    assert cob.package_version()


def test_package_metadata_is_release_ready() -> None:
    package_root = Path(__file__).resolve().parents[1]
    pyproject = (package_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.2.0"' in pyproject
    assert 'license = { file = "LICENSE" }' in pyproject
    assert "Apache Software License" in pyproject
    assert (package_root / "LICENSE").exists()
    assert (package_root / "scripts" / "packaging_smoke.py").exists()
    assert (package_root / "docs" / "release_checklist.md").exists()


def test_load_results_and_output_bundle_schema(tmp_path: Path) -> None:
    config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root=tmp_path)
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "families": ("streamretain",),
            "sample_sizes": (18,),
            "mc_truth_rollouts": 4,
            "estimators": ("direct_method",),
        }
    )
    result = cob.run_suite(config)
    bundle = cob.load_results(result.output_dir)
    assert bundle.results
    assert bundle.summary
    assert bundle.output_schema_path.exists()
    assert bundle.output_schema["result_schema_version"] == "results_v1"
    assert bundle.manifest["benchmark_schema_version"] == "benchmark_v1"
    assert "optional_dependencies" in bundle.manifest
    assert "scorer_mc_rollouts" in bundle.manifest["config"]
    assert "mc_truth_rollouts" not in bundle.manifest["config"]
    assert bundle.results[0]["benchmark_schema_version"] == "benchmark_v1"


def test_streamlift_schema_leakage_and_adapters() -> None:
    scenario = scenarios_for_profile("smoke", "streamlift")[0]
    problem = make_streamlift_problem(
        sample_size=40,
        gamma=0.9,
        seed=7,
        scenario=scenario,
        observed_horizon=2,
        mc_truth_rollouts=8,
    )
    dataset = problem.dataset
    assert dataset.n > 0
    assert dataset.states.shape == dataset.next_states.shape
    assert dataset.actions.shape == dataset.target_actions.shape
    assert dataset.action_available.shape == dataset.actions.shape
    assert np.all(dataset.behavior_propensity > 0.0)
    assert np.all(dataset.target_propensity > 0.0)
    assert np.all(dataset.target_propensity_observed_action > 0.0)
    assert dataset.target_action_probabilities is not None
    assert dataset.next_target_action_probabilities is not None
    assert dataset.initial_target_action_probabilities is not None
    assert np.allclose(dataset.target_action_probabilities.sum(axis=1), 1.0)
    assert np.allclose(dataset.next_target_action_probabilities.sum(axis=1), 1.0)
    assert np.allclose(dataset.initial_target_action_probabilities.sum(axis=1), 1.0)
    observed_action = np.argmax(dataset.actions, axis=1)
    assert np.allclose(
        dataset.target_propensity_observed_action,
        dataset.target_action_probabilities[np.arange(dataset.n), observed_action],
    )
    assert dataset.assigned_arm is not None
    assert dataset.received_treatment is not None
    assert dataset.assignment_propensity is not None
    assert dataset.metadata_public["campaign_mode"] == "finite_campaign"
    assert dataset.metadata_public["campaign_length"] == 2
    assert dataset.metadata_public["primary_observed_endpoint"] == "revenue"
    assert dataset.metadata_public["decision_endpoint"] == "discounted_ltv"
    state_features = str(dataset.metadata_public["state_features"]).split("|")
    for feature in ("taste_depth", "price_sensitivity", "competitor_pull", "household_complexity"):
        assert feature in state_features
    stable_cols = [state_features.index(feature) for feature in ("taste_depth", "price_sensitivity", "competitor_pull", "household_complexity")]
    assert np.allclose(dataset.states[:, stable_cols], dataset.next_states[:, stable_cols])
    _assert_no_public_truth(dataset.metadata_public)
    assert "surrogate_validity" in problem.truth.private_metadata
    assert "surrogate_bias_horizon_long" in problem.truth.private_metadata
    assert problem.truth.mc_rollouts == 8
    assert problem.truth.target_standard_errors
    assert problem.truth.truth_noise_floor
    occ = to_occupancy_ratio_dataset(dataset)
    fqe = to_fqe_dataset(dataset)
    panel = to_effect_panel(dataset)
    assert occ.states.shape[0] == dataset.n
    assert fqe.next_actions.shape == dataset.next_target_actions.shape
    assert panel.unit_id.shape[0] == np.unique(dataset.unit_id).shape[0]
    assert panel.observed_reward.shape[0] == panel.unit_id.shape[0]
    assert panel.behavior_propensity.shape[0] == panel.unit_id.shape[0]
    assert panel.baseline_state.shape[1] == dataset.state_dim
    first_idx = np.array([np.flatnonzero(dataset.unit_id == unit)[0] for unit in np.unique(dataset.unit_id)], dtype=np.int64)
    assert np.array_equal(panel.arm, dataset.assigned_arm[first_idx].astype(np.int64))
    _assert_no_public_truth(occ.metadata)
    _assert_no_public_truth(fqe.metadata)
    _assert_no_public_truth(panel.metadata)


def test_streamlift_stratified_gcomp_uses_public_stationary_dynamics() -> None:
    scenario = scenarios_for_profile("smoke", "streamlift")[0]
    problem = make_streamlift_problem(
        sample_size=90,
        gamma=0.9,
        seed=17,
        scenario=scenario,
        observed_horizon=3,
        mc_truth_rollouts=8,
    )
    result = run_estimator("streamlift_stratified_gcomp", problem)
    assert result.status == "ok"
    assert result.diagnostics["gcomp_model"] == "arm_stratified_stationary_ridge"
    assert result.diagnostics["gcomp_control_rows"] > 0
    assert result.diagnostics["gcomp_treatment_rows"] > 0
    assert np.isfinite(result.estimates["effect_horizon_36"])
    assert "surrogate_validity" not in result.diagnostics


def test_streamlift_can_score_finite_and_infinite_horizon_effects() -> None:
    scenario = scenarios_for_profile("smoke", "streamlift")[0]
    problem = make_streamlift_problem(
        sample_size=30,
        gamma=0.9,
        seed=19,
        scenario=scenario,
        observed_horizon=3,
        forecast_horizons=(6, 12),
        long_horizon=12,
        include_infinite_horizon=True,
        infinite_horizon_max_steps=40,
        mc_truth_rollouts=4,
    )
    assert problem.dataset.metadata_public["horizon_modes"] == "finite|infinite"
    assert problem.dataset.metadata_public["infinite_horizon"] is True
    assert "effect_horizon_12" in problem.truth.effects
    assert "effect_horizon_infinite" in problem.truth.effects
    assert "value_treatment_horizon_infinite" in problem.truth.values
    result = run_estimator("streamlift_stratified_gcomp", problem)
    assert result.status == "ok"
    assert np.isfinite(result.estimates["effect_horizon_12"])
    assert np.isfinite(result.estimates["effect_horizon_infinite"])
    score = score_result(problem.dataset, problem.truth, result)
    assert score["missing_required_estimand_count"] == 0


def test_generation_is_deterministic_by_seed() -> None:
    scenario = scenarios_for_profile("smoke", "streamretain")[0]
    first = make_streamretain_problem(sample_size=30, gamma=0.9, seed=11, scenario=scenario, mc_truth_rollouts=6)
    second = make_streamretain_problem(sample_size=30, gamma=0.9, seed=11, scenario=scenario, mc_truth_rollouts=6)
    assert np.allclose(first.dataset.states, second.dataset.states)
    assert np.allclose(first.dataset.actions, second.dataset.actions)
    assert first.truth.values["policy_value"] == second.truth.values["policy_value"]


def test_policies_respect_availability_and_propensities() -> None:
    with pytest.raises(ValueError, match="Allowed policies"):
        get_fixed_policy("clinic_dtr", "not_a_policy")
    scenario = scenarios_for_profile("smoke", "clinic_dtr")[0]
    problem = make_clinic_dtr_problem(sample_size=35, gamma=0.9, seed=13, scenario=scenario, mc_truth_rollouts=6)
    dataset = problem.dataset
    assert np.all((dataset.actions * (1.0 - dataset.action_available)) == 0.0)
    assert np.all((dataset.target_actions * (1.0 - dataset.action_available)) == 0.0)
    assert np.all(dataset.behavior_propensity > 0.0)
    assert np.all(dataset.target_propensity > 0.0)
    assert dataset.target_action_probabilities is not None
    assert dataset.next_target_action_probabilities is not None
    assert dataset.initial_target_action_probabilities is not None
    assert np.allclose(dataset.target_action_probabilities.sum(axis=1), 1.0)
    assert np.allclose(dataset.next_target_action_probabilities.sum(axis=1), 1.0)
    target_policy = get_fixed_policy("clinic_dtr", str(dataset.metadata_public["target_policy"]))
    probs = target_policy.probabilities(dataset.states, dataset.action_available, dataset.time)
    observed_action = np.argmax(dataset.actions, axis=1)
    assert np.allclose(dataset.target_propensity_observed_action, probs[np.arange(dataset.n), observed_action])
    survival = to_survival_panel(dataset)
    assert survival.covariates.shape == dataset.states.shape
    assert survival.event.shape[0] == dataset.n


def test_scenario_rates_and_truth_are_present() -> None:
    scenario = sensitivity_scenarios_for_profile("full", "clinic_dtr")[0]
    problem = make_clinic_dtr_problem(sample_size=60, gamma=0.9, seed=17, scenario=scenario, mc_truth_rollouts=8)
    dataset = problem.dataset
    assert np.mean(dataset.missingness_mask) > 0.0
    assert "policy_value" in problem.truth.values
    assert "rmst" in problem.truth.rmst
    assert "survival_target" in problem.truth.survival_curves
    assert not problem.truth.leaderboard_eligible


def test_canonical_profiles_exclude_latent_and_unsupported_stressors() -> None:
    paper = CausalOPEBenchmarkConfig.for_profile("paper")
    assert paper.families == ("streamretain", "clinic_dtr")
    assert paper.target_policies == ("moderate", "safety_constrained")
    for profile in ("smoke", "core", "full", "paper"):
        for family in ("streamretain", "clinic_dtr"):
            scenarios = scenarios_for_profile(profile, family)
            assert scenarios
            for scenario in scenarios:
                assert scenario.confounding != "latent"
                assert scenario.overlap != "structural_gap"
                assert scenario.missingness == "none"
                assert scenario.censoring in {"none", "administrative"}
                assert not scenario.nonstationarity
                assert scenario.leaderboard_eligible


def test_scenario_realism_knobs_change_generation() -> None:
    base = DomainScenario(name="base", overlap="moderate", confounding="observed", target_policy_distance=0.6)
    with pytest.raises(ValueError, match="target_policy_distance"):
        DomainScenario(name="invalid", overlap="moderate", confounding="observed", target_policy_distance=1.3)
    close = DomainScenario(name="close", overlap="moderate", confounding="observed", target_policy_distance=0.2)
    distant = DomainScenario(name="distant", overlap="moderate", confounding="observed", target_policy_distance=0.95)
    close_problem = make_streamretain_problem(sample_size=50, gamma=0.9, seed=18, scenario=close, mc_truth_rollouts=4)
    first = make_streamretain_problem(sample_size=50, gamma=0.9, seed=18, scenario=base, mc_truth_rollouts=4)
    second = make_streamretain_problem(sample_size=50, gamma=0.9, seed=18, scenario=distant, mc_truth_rollouts=4)
    assert not np.isclose(np.mean(first.dataset.behavior_propensity), np.mean(second.dataset.behavior_propensity))
    assert float(close_problem.dataset.metadata_public["target_policy_distance"]) <= float(first.dataset.metadata_public["target_policy_distance"])
    assert float(first.dataset.metadata_public["target_policy_distance"]) <= float(second.dataset.metadata_public["target_policy_distance"])

    stationary = DomainScenario(name="stationary", nonstationarity=False)
    shifted = DomainScenario(name="shifted", nonstationarity=True)
    c_first = make_clinic_dtr_problem(sample_size=40, gamma=0.9, seed=18, scenario=stationary, mc_truth_rollouts=4)
    c_second = make_clinic_dtr_problem(sample_size=40, gamma=0.9, seed=18, scenario=shifted, mc_truth_rollouts=4)
    assert not np.allclose(c_first.truth.values["policy_value"], c_second.truth.values["policy_value"])


def test_streamlift_noncompliance_behavior_propensity() -> None:
    scenario = DomainScenario(
        name="randomized_good_overlap",
        overlap="good",
        confounding="randomized",
        noncompliance_rate=0.20,
    )
    problem = make_streamlift_problem(
        sample_size=50,
        gamma=0.9,
        seed=19,
        scenario=scenario,
        observed_horizon=2,
        mc_truth_rollouts=4,
    )
    dataset = problem.dataset
    assert dataset.assignment_propensity is not None
    action = np.argmax(dataset.actions, axis=1)
    p_received_1 = dataset.assignment_propensity * 0.80 + (1.0 - dataset.assignment_propensity) * 0.20
    expected = np.where(action == 1, p_received_1, 1.0 - p_received_1)
    assert np.allclose(dataset.behavior_propensity, expected)


def test_fqe_initial_actions_are_target_policy_actions() -> None:
    scenario = scenarios_for_profile("smoke", "streamretain")[0]
    problem = make_streamretain_problem(sample_size=50, gamma=0.9, seed=23, scenario=scenario, mc_truth_rollouts=4)
    dataset = problem.dataset
    initial_idx = np.asarray(dataset.splits["initial"], dtype=np.int64)
    assert set(np.unique(np.argmax(dataset.initial_actions, axis=1))).issubset(set(range(dataset.action_dim)))
    assert len(np.unique(np.argmax(dataset.initial_actions, axis=1))) > 1
    fqe = to_fqe_dataset(dataset)
    assert np.allclose(fqe.initial_actions, dataset.initial_actions)
    exact_fqe = to_fqe_dataset(dataset, target_policy_expectation_mode="exact_discrete")
    assert exact_fqe.initial_action_probabilities is not None
    assert exact_fqe.target_policy_expectation_mode == "exact_discrete"
    assert exact_fqe.row_expansion_factor > 1.0
    assert exact_fqe.source_row_index is not None
    assert exact_fqe.source_row_index.shape[0] == exact_fqe.states.shape[0]
    assert int(np.max(exact_fqe.source_row_index)) < dataset.n
    assert dataset.unit_id[exact_fqe.source_row_index].shape[0] == exact_fqe.states.shape[0]
    assert exact_fqe.action_dose is not None
    assert np.all(fqe.terminals >= dataset.terminals)
    assert np.all(fqe.terminals[initial_idx] >= dataset.terminals[initial_idx])


def test_streamlift_campaign_modes_change_long_horizon_truth() -> None:
    common = {
        "name": "campaign",
        "overlap": "good",
        "confounding": "randomized",
        "delay_pattern": "delayed_benefit",
        "surrogate_validity": "valid",
    }
    one_shot = DomainScenario(**common, streamlift_campaign_mode="one_shot")
    finite = DomainScenario(**common, streamlift_campaign_mode="finite_campaign", campaign_length=3)
    persistent = DomainScenario(**common, streamlift_campaign_mode="persistent")
    one = make_streamlift_problem(sample_size=35, gamma=0.9, seed=24, scenario=one_shot, observed_horizon=3, mc_truth_rollouts=4)
    mid = make_streamlift_problem(sample_size=35, gamma=0.9, seed=24, scenario=finite, observed_horizon=3, mc_truth_rollouts=4)
    always = make_streamlift_problem(sample_size=35, gamma=0.9, seed=24, scenario=persistent, observed_horizon=3, mc_truth_rollouts=4)
    assert one.dataset.metadata_public["campaign_mode"] == "one_shot"
    assert mid.dataset.metadata_public["campaign_mode"] == "finite_campaign"
    assert always.dataset.metadata_public["campaign_mode"] == "persistent"
    vals = [problem.truth.effects["effect_horizon_36"] for problem in (one, mid, always)]
    assert len({round(value, 6) for value in vals}) > 1


def test_dose_and_action_constraint_surfaces_are_public_and_active() -> None:
    constrained = DomainScenario(name="constrained", action_constraints=True)
    unconstrained = DomainScenario(name="unconstrained", action_constraints=False)
    first = make_clinic_dtr_problem(sample_size=70, gamma=0.9, seed=25, scenario=constrained, mc_truth_rollouts=4)
    second = make_clinic_dtr_problem(sample_size=70, gamma=0.9, seed=25, scenario=unconstrained, mc_truth_rollouts=4)
    assert first.dataset.action_dose is not None
    assert first.dataset.target_action_dose is not None
    assert first.dataset.dose_available is not None
    assert np.std(first.dataset.action_dose) > 0.0
    assert np.all(first.dataset.actions * (1.0 - first.dataset.action_available) == 0.0)
    assert np.all(second.dataset.action_available == 1.0)
    assert np.any(first.dataset.action_available == 0.0)


def test_ipcw_rmst_uses_censoring_without_treating_it_as_failure() -> None:
    scenario = DomainScenario(name="censored", censoring="informative")
    problem = make_clinic_dtr_problem(sample_size=80, gamma=0.9, seed=27, scenario=scenario, mc_truth_rollouts=6)
    dataset = problem.dataset
    result = run_estimator("ipcw_rmst", problem, config=CausalOPEBenchmarkConfig.for_profile("smoke"))
    assert result.status == "ok"
    assert 0.0 <= result.estimates["survival_horizon"] <= 1.0
    assert result.estimates["rmst"] >= result.estimates["survival_horizon"]
    if np.any(dataset.censoring > 0.5):
        naive_event_survival = np.mean([
            1.0 - np.max(dataset.terminals[np.flatnonzero(dataset.unit_id == unit)])
            for unit in np.unique(dataset.unit_id)
        ])
        assert result.estimates["survival_horizon"] <= naive_event_survival + 0.25


def test_neural_fqe_estimator_is_package_integrated() -> None:
    scenario = scenarios_for_profile("smoke", "clinic_dtr")[0]
    problem = make_clinic_dtr_problem(sample_size=45, gamma=0.9, seed=29, scenario=scenario, mc_truth_rollouts=4)
    config = CausalOPEBenchmarkConfig.for_profile("smoke")
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "fqe_hidden_dims": (8,),
            "fqe_num_iterations": 2,
            "fqe_gradient_steps_per_iteration": 1,
            "fqe_batch_size": 32,
        }
    )
    result = run_estimator("neural_fqe", problem, config=config)
    assert result.status in {"ok", "missing_dependency"}
    if result.status == "ok":
        assert "policy_value" in result.estimates
        assert np.isfinite(result.estimates["policy_value"])


def test_fqe_automl_estimator_writes_proxy_tuning_rows(monkeypatch) -> None:
    scenario = scenarios_for_profile("smoke", "streamretain")[0]
    problem = make_streamretain_problem(sample_size=20, gamma=0.9, seed=30, scenario=scenario, mc_truth_rollouts=4)
    config = CausalOPEBenchmarkConfig.for_profile("smoke")
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "fqe_hidden_dims": (4,),
            "fqe_num_iterations": 2,
            "fqe_gradient_steps_per_iteration": 1,
            "fqe_batch_size": 16,
            "automl_tuning": "fast",
        }
    )

    class FakeModel:
        diagnostics = {"fake_model": 1}

        def predict_q(self, states, actions):
            return np.full(np.asarray(states).shape[0], 1.25, dtype=np.float64)

    class FakeTuned:
        selected_candidate_id = "boosted_fake"
        selected_family = "boosted"
        model = FakeModel()

        def candidate_rows(self):
            return [{"candidate_id": "boosted_fake", "family": "boosted", "score": 0.5}]

        def fold_rows(self):
            return [{"candidate_id": "boosted_fake", "family": "boosted", "fold": 0, "bellman_risk": 0.1}]

    import fqe

    monkeypatch.setattr(fqe, "tune_fqe_auto", lambda **kwargs: FakeTuned())
    result = run_estimator("boosted_fqe_auto", problem, config=config)
    assert result.status == "ok"
    assert result.estimates["policy_value"] == pytest.approx(1.25)
    assert result.tuning_rows
    assert {row["tuning_stage"] for row in result.tuning_rows} == {"automl_candidate", "automl_fold"}
    assert all("truth" not in key and "oracle" not in key for row in result.tuning_rows for key in row)


def test_discounted_occupancy_estimator_uses_package_known_ratios(monkeypatch) -> None:
    scenario = scenarios_for_profile("smoke", "streamretain")[0]
    problem = make_streamretain_problem(sample_size=20, gamma=0.9, seed=31, scenario=scenario, mc_truth_rollouts=4)
    captured: dict[str, object] = {}

    class FakeOccupancyModel:
        diagnostics = {"fake_ratio": 1.0}

        def predict_state_action_ratio(self, states, actions, *, clip=True):
            n = np.asarray(states).shape[0]
            return np.ones(n, dtype=np.float64) if clip else np.full(n, 1.1, dtype=np.float64)

    def fake_fit(**kwargs):
        captured.update(kwargs)
        return FakeOccupancyModel()

    import occupancy_ratio

    monkeypatch.setattr(occupancy_ratio, "fit_discounted_occupancy_ratio", fake_fit)
    result = run_estimator("discounted_occupancy_boosted", problem, config=CausalOPEBenchmarkConfig.for_profile("smoke"))
    assert result.status == "ok"
    assert np.isfinite(result.estimates["policy_value"])
    expected_ratio = problem.dataset.target_propensity_observed_action / problem.dataset.behavior_propensity
    assert np.allclose(captured["action_ratio_values"], expected_ratio)
    assert captured["initial_states"].shape == problem.dataset.initial_states.shape
    assert captured["initial_actions"].shape == problem.dataset.initial_actions.shape
    assert captured["initial_ratio_mode"] == "auto"
    assert result.diagnostics["known_action_ratio_used"] == 1


def test_runner_oracle_selection_and_tuning_rows_are_diagnostic(monkeypatch, tmp_path: Path) -> None:
    config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root=tmp_path)
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "families": ("streamretain",),
            "sample_sizes": (12,),
            "seeds": (0,),
            "mc_truth_rollouts": 4,
            "estimators": (
                "boosted_fqe",
                "neural_fqe",
                "discounted_occupancy_boosted",
                "discounted_occupancy_neural",
            ),
        }
    )

    def fake_run_estimator(estimator: str, problem, config=None):
        truth = float(problem.truth.values["policy_value"])
        offsets = {
            "boosted_fqe": 3.0,
            "neural_fqe": 1.0,
            "discounted_occupancy_boosted": 2.0,
            "discounted_occupancy_neural": 4.0,
        }
        return EstimatorResult(
            estimator=estimator,
            status="ok",
            estimates={"policy_value": truth + offsets[estimator]},
            diagnostics={"interval_available": 0},
            tuning_rows=[{"estimator": estimator, "candidate_id": f"{estimator}_candidate", "metric_proxy": 1.0}],
        )

    import causal_ope_benchmark.runner as runner_module

    monkeypatch.setattr(runner_module, "run_estimator", fake_run_estimator)
    result = run_benchmark(config)
    oracle_rows = [row for row in result.rows if str(row["estimator"]).startswith("oracle_selected_")]
    assert {row["estimator"] for row in oracle_rows} == {
        "oracle_selected_fqe_diagnostic",
        "oracle_selected_discounted_occupancy_diagnostic",
    }
    assert all(row["diagnostic_only"] == 1 for row in oracle_rows)
    assert all(row["leaderboard_result_eligible"] == 0 for row in oracle_rows)
    by_estimator = {row["estimator"]: row for row in oracle_rows}
    assert by_estimator["oracle_selected_fqe_diagnostic"]["diag_oracle_selected_from"] == "neural_fqe"
    assert by_estimator["oracle_selected_discounted_occupancy_diagnostic"]["diag_oracle_selected_from"] == "discounted_occupancy_boosted"
    assert result.tuning_rows
    assert all("truth" not in key and "oracle" not in key for row in result.tuning_rows for key in row)


def test_gym_env_wrappers_reset_step_and_deterministic_seed() -> None:
    scenario = scenarios_for_profile("smoke", "streamretain")[0]
    env = make_gym_env("streamretain", scenario=scenario, target_policy="moderate", seed=31)
    obs_a, info_a = env.reset(seed=123)
    obs_b, info_b = env.reset(seed=123)
    assert np.allclose(obs_a, obs_b)
    assert obs_a.shape == env.observation_space.shape
    assert env.action_space.n == len(info_a["action_names"])
    assert np.allclose(info_a["action_available"], info_b["action_available"])
    action = int(np.flatnonzero(info_a["action_available"] > 0.5)[0])
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_available" in info
    assert "target_action_probabilities" in info

    masked = DomainScenario(name="masked", action_constraints=True)
    clinic = make_gym_env("clinic_dtr", scenario=masked, target_policy="moderate", seed=32)
    clinic._state = np.asarray([0.75, 1.0, 1.0, 0.80, 0.70, 0.80, 0.90, 0.40, 0.90, 1.0], dtype=np.float64)
    clinic._latent = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    clinic._terminated = False
    clinic._time = 0
    with pytest.raises(ValueError, match="unavailable"):
        clinic.step(2)


def test_fixed_policy_wrapper_exposes_scope_rl_policy_methods() -> None:
    policy = FixedPolicyWrapper("streamretain", "moderate", rng=np.random.default_rng(35))
    x = np.asarray([[0.4, 0.2, 0.6, 0.5, 0.2, 0.7, 0.0, 0.1]], dtype=np.float64)
    probs = policy.calc_action_choice_probability(x)
    assert probs.shape == (1, policy.n_actions)
    assert np.allclose(probs.sum(axis=1), 1.0)
    action, pscore = policy.sample_action_and_output_pscore(x)
    assert action.shape == (1,)
    assert pscore.shape == (1,)
    assert np.all(pscore > 0.0)
    assert np.allclose(policy.calc_pscore_given_action(x, action), pscore)
    assert policy.predict(x).shape == (1,)


def test_scope_rl_logged_dataset_export_has_required_public_fields() -> None:
    scenario = scenarios_for_profile("smoke", "streamretain")[0]
    problem = make_streamretain_problem(sample_size=25, gamma=0.9, seed=33, scenario=scenario, mc_truth_rollouts=4)
    exported = to_scope_rl_logged_dataset(problem.dataset, behavior_policy_name="logged", dataset_id=7)
    validate_scope_rl_logged_dataset(exported)
    required = {
        "size",
        "n_trajectories",
        "step_per_trajectory",
        "action_type",
        "n_actions",
        "state_dim",
        "state",
        "action",
        "reward",
        "done",
        "terminal",
        "pscore",
        "info",
        "behavior_policy",
        "dataset_id",
    }
    assert required.issubset(exported)
    assert exported["action_type"] == "discrete"
    assert exported["behavior_policy"] == "logged"
    assert exported["dataset_id"] == 7
    assert exported["state"].shape[:2] == exported["reward"].shape
    assert exported["state"].shape[2] == problem.dataset.state_dim
    assert exported["action"].shape == exported["reward"].shape
    assert np.all(np.isfinite(exported["pscore"]))
    assert np.all(exported["pscore"] > 0.0)
    assert exported["info"]["action_available"].shape[:2] == exported["reward"].shape
    _assert_no_public_truth(exported["info"]["metadata_public"])

    lift = make_streamlift_problem(sample_size=20, gamma=0.9, seed=34, scenario=scenarios_for_profile("smoke", "streamlift")[0], observed_horizon=2, mc_truth_rollouts=4)
    lift_export = to_scope_rl_logged_dataset(lift.dataset)
    assert lift_export["info"]["panel_only"] is True


def test_epicare_family_is_lazy_and_default_profiles_exclude_it(tmp_path: Path) -> None:
    default = CausalOPEBenchmarkConfig.for_profile("smoke")
    assert "epicare" not in default.families
    config = CausalOPEBenchmarkConfig(
        profile="smoke",
        output_root=tmp_path,
        seeds=(0,),
        families=("epicare",),
        sample_sizes=(4,),
        gammas=(0.9,),
        observed_horizons=(2,),
        target_policies=("moderate",),
        estimators=("direct_method",),
        trajectory_horizon=3,
        mc_truth_rollouts=3,
        fqe_hidden_dims=(8,),
        fqe_num_iterations=2,
        fqe_gradient_steps_per_iteration=1,
        fqe_batch_size=8,
    )
    result = run_benchmark(config)
    assert result.rows
    if importlib.util.find_spec("epicare") is None:
        assert all(row["status"] == "missing_dependency" for row in result.rows)
    else:
        assert all(row["status"] in {"ok", "error", "missing_dependency", "incomplete"} for row in result.rows)
    with result.manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for package in ("gym", "gymnasium", "scope_rl", "epicare"):
        assert package in manifest["packages"]


def test_cli_discovery_and_dry_run_are_fast() -> None:
    package_root = Path(__file__).resolve().parents[1]
    families = subprocess.run(
        [sys.executable, "-m", "causal_ope_benchmark.run", "--list-families"],
        check=True,
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    assert "streamretain" in families.stdout
    estimators = subprocess.run(
        [sys.executable, "-m", "causal_ope_benchmark.run", "--list-estimators"],
        check=True,
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    assert "direct_method" in estimators.stdout
    dry_run = subprocess.run(
        [sys.executable, "-m", "causal_ope_benchmark.run", "--profile", "smoke", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    payload = json.loads(dry_run.stdout)
    assert payload["profile"] == "smoke"
    assert payload["output_dir"].endswith("smoke")
    calibration_dry_run = subprocess.run(
        [sys.executable, "-m", "causal_ope_benchmark.run", "calibrate", "--preset", "smoke", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    calibration_payload = json.loads(calibration_dry_run.stdout)
    assert calibration_payload["preset"] == "smoke"
    assert calibration_payload["output_dir"].endswith("calibration/smoke")


def test_cli_invalid_estimator_and_policy_errors_are_clear() -> None:
    package_root = Path(__file__).resolve().parents[1]
    bad_estimator = subprocess.run(
        [sys.executable, "-m", "causal_ope_benchmark.run", "--estimators", "not_real"],
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    assert bad_estimator.returncode == 2
    assert "Unknown estimator" in bad_estimator.stderr
    assert "direct_method" in bad_estimator.stderr
    bad_policy = subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_ope_benchmark.run",
            "--families",
            "streamretain",
            "--target-policies",
            "not_real",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    assert bad_policy.returncode == 2
    assert "Unknown target policy" in bad_policy.stderr
    assert "moderate" in bad_policy.stderr


def test_packaging_smoke_script_builds_and_installs_when_build_is_available() -> None:
    if importlib.util.find_spec("build") is None:
        pytest.skip("python-build package is not installed")
    package_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(package_root / "scripts" / "packaging_smoke.py"),
            "--package-dir",
            str(package_root),
        ],
        check=True,
        cwd=package_root,
    )


def test_runner_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root=tmp_path)
    config = CausalOPEBenchmarkConfig(
        **{
            **config.__dict__,
            "sample_sizes": (30,),
            "observed_horizons": (2,),
            "mc_truth_rollouts": 6,
            "estimators": ("direct_method", "ipw", "oracle_diagnostic", "neural_fqe_streamlift_diagnostic"),
            "fqe_hidden_dims": (8,),
            "fqe_num_iterations": 2,
            "fqe_gradient_steps_per_iteration": 1,
            "fqe_batch_size": 32,
        }
    )
    result = run_benchmark(config)
    validate_output_bundle(result.output_dir)
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.tuning_path.exists()
    assert result.manifest_path.exists()
    assert result.diagnostics_path.exists()
    assert result.readout_path.exists()
    assert result.output_schema_path.exists()
    with result.results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert any(row["family"] == "streamlift" for row in rows)
    assert any(row["diagnostic_only"] == "1" for row in rows)
    assert any(row["estimator"] == "neural_fqe_streamlift_diagnostic" and row["diagnostic_only"] == "1" for row in rows)
    deployable = [row for row in rows if row["diagnostic_only"] == "0" and row["status"] == "ok"]
    assert deployable
    assert all("calibrated_score" in row for row in rows)
    assert all("primary_weighted_mae" in row for row in rows)
    assert any("leaderboard_result_eligible" in row for row in rows)
    assert any("truth_mc_uncertainty_max" in row for row in rows)
    assert any(row.get("constraint_schema_available") == "1" for row in rows)
    assert all("oracle" not in row["scenario_public"].lower() for row in rows)
    assert all(row.get("benchmark_schema_version") == "benchmark_v1" for row in rows)
    with result.output_schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    assert schema["result_schema_version"] == "results_v1"
    readout = result.readout_path.read_text(encoding="utf-8")
    assert "## Family Summary" in readout
    assert "## Estimator Summary" in readout
    assert "## Status Summary" in readout
    with result.manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert "optional_dependencies" in manifest
    _assert_no_private_payload_tokens(manifest)
    _assert_no_private_payload_tokens(schema)


def test_calibration_smoke_writes_outputs_and_marks_oracle_diagnostic(tmp_path: Path) -> None:
    config = CalibrationStudyConfig.for_preset("smoke", output_root=tmp_path)
    config = CalibrationStudyConfig(
        **{
            **config.__dict__,
            "families": ("streamretain",),
            "sample_sizes": (18,),
            "target_policies": ("moderate",),
            "estimators": ("neural_fqe",),
            "tuning_tracks": ("proxy", "oracle"),
            "mc_truth_rollouts": 4,
        }
    )
    result = run_calibration_study(config)
    validate_calibration_output_bundle(result.output_dir)
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.candidates_path.exists()
    assert result.manifest_path.exists()
    assert result.readout_path.exists()
    assert result.rows
    assert {row["tuning_track"] for row in result.rows} == {"proxy", "oracle"}
    assert all(row["leaderboard_result_eligible"] == 0 for row in result.rows)
    assert all(row["diagnostic_only"] == 1 for row in result.rows if row["tuning_track"] == "oracle")
    assert all(row["status"] in {"ok", "missing_dependency", "error"} for row in result.rows)
    if any(row["status"] == "ok" for row in result.rows):
        assert result.candidate_rows
        assert any(row["tuning_track"] == "proxy" for row in result.candidate_rows)
    readout = result.readout_path.read_text(encoding="utf-8")
    assert "## Difficulty Verdicts" in readout
    assert "Oracle-track rows are diagnostic upper bounds only" in readout
    with result.manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["calibration_schema_version"] == "calibration_v1"
    assert "optional_dependencies" in manifest
    _assert_no_private_payload_tokens(manifest)


def test_difficulty_profiles_are_stationary_identifiable_primary_cells() -> None:
    for difficulty in ("easy", "medium", "hard", "realistic"):
        scenarios = scenarios_for_difficulty(difficulty, "streamretain", scale="audit")
        assert scenarios
        primary_cells = difficulty_cells(difficulty, "streamretain", scale="audit")
        assert any(cell.stress_dimension == "target_policy_distance" for cell in primary_cells)
        for cell in primary_cells:
            if not cell.primary:
                continue
            assert not cell.scenario.nonstationarity
            assert cell.scenario.confounding != "latent"
            assert cell.scenario.overlap != "structural_gap"
            assert 0.0 <= cell.scenario.target_policy_distance <= 1.0
    hard = cob.describe_difficulty("hard")
    assert hard.streamlift_include_infinite_horizon
    assert hard.overlap == "weak"


def test_make_benchmark_problem_accepts_difficulty() -> None:
    problem = cob.make_benchmark_problem(
        "streamretain",
        difficulty="easy",
        sample_size=20,
        gamma=0.9,
        seed=47,
        target_policy="moderate",
        config=CausalOPEBenchmarkConfig.for_profile("smoke"),
    )
    validate_problem(problem)
    assert problem.dataset.metadata_public["nonstationarity"] == "none"
    assert problem.dataset.metadata_public["action_constraints"] == "inactive"
    assert float(problem.dataset.metadata_public["target_policy_distance"]) <= 0.25


def test_difficulty_stress_smoke_writes_outputs(tmp_path: Path) -> None:
    config = cob.DifficultyStressStudyConfig.for_scale("ci", output_root=tmp_path)
    config = cob.DifficultyStressStudyConfig(
        **{
            **config.__dict__,
            "difficulties": ("easy",),
            "families": ("streamretain",),
            "methods": ("direct_method",),
            "sample_sizes": (18,),
            "target_policies": ("moderate",),
            "mc_truth_rollouts": 4,
        }
    )
    result = cob.run_difficulty_study(config)
    validate_difficulty_output_bundle(result.output_dir)
    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.candidates_path.exists()
    assert result.manifest_path.exists()
    assert result.readout_path.exists()
    assert result.rows
    assert all(row["time_invariant_mdp"] == 1 for row in result.rows)
    assert all(row["identifiable_primary_cell"] == 1 for row in result.rows)
    readout = result.readout_path.read_text(encoding="utf-8")
    assert "## Hardness Verdicts" in readout
    with result.manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["difficulty_schema_version"] == "difficulty_v1"
    _assert_no_private_payload_tokens(manifest)


def test_difficulty_cli_dry_run(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            path
            for path in (
                str(package_root),
                os.environ.get("PYTHONPATH", ""),
            )
            if path
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_ope_benchmark.run",
            "stress-test",
            "--scale",
            "ci",
            "--difficulty",
            "easy",
            "--families",
            "streamretain",
            "--methods",
            "direct_method",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["scale"] == "ci"
    assert payload["difficulties"] == ["easy"]


def test_occupancy_calibration_value_matches_discounted_value_convention() -> None:
    weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rewards = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)
    unit_id = np.asarray([0, 0, 1], dtype=np.int64)
    time = np.asarray([0, 1, 0], dtype=np.int64)
    expected = np.mean([1.0 * 0.5 + 0.9 * 2.0 * 1.0, 3.0 * 1.5])
    assert np.isclose(_occupancy_value(weights, rewards, 0.9, unit_id=unit_id, time=time), expected)


def _assert_no_public_truth(metadata: dict[str, object]) -> None:
    text = "|".join([*(str(key).lower() for key in metadata), *(str(value).lower() for value in metadata.values())])
    for token in ("truth", "true_", "oracle", "latent", "target_mc", "surrogate_validity"):
        assert token not in text


def _assert_no_private_payload_tokens(payload: dict[str, object]) -> None:
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    for token in ("scenario_private", "target_mc", "surrogate_validity", "latent_parameters"):
        assert token not in text
