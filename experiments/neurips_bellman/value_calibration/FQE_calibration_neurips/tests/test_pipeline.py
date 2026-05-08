from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from FQE_calibration_neurips.scripts.audit_strict_submission import audit_strict_submission
from FQE_calibration_neurips.scripts.audit_rescue_submission import audit_rescue_submission
from FQE_calibration_neurips.scripts.run_experiment import run_config
from FQE_calibration_neurips.scripts.inspect_paper_draft import inspect_paper_draft
from FQE_calibration_neurips.scripts.run_rescue_stage import run_rescue_stage
from FQE_calibration_neurips.scripts.run_suite import run_suite
from FQE_calibration_neurips.src.aggregation import aggregate_results, summarize_raw
from FQE_calibration_neurips.src.plotting import make_plots
from FQE_calibration_neurips.src.tables import write_tables
from FQE_calibration_neurips.src.validation import GateConfig, evaluate_well_specified_gate


def _tiny_config() -> dict:
    return {
        "seed": 515,
        "replications": 1,
        "gamma": 0.9,
        "horizon": 8,
        "n_actions": 2,
        "sample_sizes": [60, 90],
        "state_dimensions": [3],
        "coverage_settings": ["good"],
        "reward_noise_settings": [0.01],
        "transition_noise": 0.0,
        "test_transitions": 50,
        "initial_eval_states": 50,
        "oracle_rollouts": 40,
        "misspecification": "well_specified_linear",
        "policy_shift": {"good": 0.05, "moderate": 0.5, "severe": 1.0, "extrapolation": 1.5},
        "baseline_learners": ["linear_fqe"],
        "calibration_protocols": ["cross", "split", "no_split"],
        "calibrators": ["linear"],
        "calibration_targets": ["value_bellman"],
        "cross_folds": 2,
        "split_fractions": [0.8],
        "split_comparators": ["offset_correction"],
        "calibrator_params": {"n_bins": 3, "min_bin_size": 3},
        "comparator_params": {"n_components": 8, "ridge": 0.01},
        "learner_params": {"linear_fqe": {"feature_type": "linear", "ridge": 0.0001, "n_iters": 5}},
    }


def _tiny_temporal_config() -> dict:
    return {
        "seed": 616,
        "replications": 1,
        "runner": "temporal_reward_shift",
        "gamma": 0.9,
        "horizon": 8,
        "n_actions": 2,
        "sample_sizes": [50],
        "recent_current_sizes": [20],
        "state_dimensions": [3],
        "coverage_settings": ["moderate"],
        "reward_noise_settings": [0.01],
        "transition_noise": 0.02,
        "current_reward_shift_intercept": 0.6,
        "current_reward_shift_scale": 1.25,
        "test_transitions": 35,
        "diagnostic_test_transitions": 45,
        "initial_eval_states": 35,
        "oracle_rollouts": 30,
        "true_v_rollouts_per_state": 1,
        "direct_true_v_rollout": True,
        "calibration_error_bins": 4,
        "calibration_error_min_bin_size": 4,
        "calibration_error_folds": 2,
        "policy_shift": {"moderate": 0.45},
        "baseline_learners": ["temporal_rf_fqe"],
        "calibrators": ["linear", "isotonic"],
        "calibration_targets": ["value_bellman"],
        "importance_weight_scheme": "action_ratio",
        "importance_weight_clip": 20.0,
        "normalize_importance_weights": True,
        "value_calibration_iterations": 2,
        "calibrator_params": {"n_bins": 4, "min_bin_size": 4, "value_calibration_iterations": 2},
        "learner_params": {"random_feature_fqe": {"n_components": 10, "bandwidth": 0.9, "ridge": 0.01, "n_iters": 3}},
        "learner_variants": {
            "temporal_rf_fqe": {
                "base_learner": "random_feature_fqe",
                "learner_quality_regime": "temporally_shifted",
                "calibration_difficulty": "temporal_reward_shift",
                "main_figure_role": "main",
                "params": {"n_components": 10, "bandwidth": 0.9, "ridge": 0.01, "n_iters": 3},
            }
        },
    }


def test_validation_gate_outputs_pass_and_fail(tmp_path: Path) -> None:
    rows = []
    for n, err in [(60, 0.2), (90, 0.05)]:
        rows.append(
            {
                "baseline_learner": "linear_fqe",
                "calibration_protocol": "uncalibrated_all_data",
                "calibrator": "none",
                "calibration_target": "value_bellman",
                "sample_size": n,
                "value_estimate": 1.0 + err,
                "oracle_value": 1.0,
                "value_error": err,
                "squared_error": err**2,
                "failure_flag": False,
                "oracle_value_method": "independent_monte_carlo_rollout",
                "train_data_provenance": "offline_behavior_batch_seed=1;not_test_or_oracle",
            }
        )
    passed, frame = evaluate_well_specified_gate(rows, tmp_path, GateConfig(bias_threshold=1.0, mse_threshold=1.0))
    assert passed
    assert (tmp_path / "well_specified_gate.json").exists()
    rows[-1]["value_estimate"] = float("inf")
    rows[-1]["value_error"] = float("inf")
    rows[-1]["squared_error"] = float("inf")
    failed, _ = evaluate_well_specified_gate(rows, tmp_path / "fail", GateConfig(bias_threshold=1.0, mse_threshold=1.0))
    assert not failed


def test_simulation_study_design_document_has_required_sections() -> None:
    path = Path("FQE_calibration_neurips/results/simulation_study_design.md")
    text = path.read_text()
    required = [
        "Main Scientific Questions",
        "Simulation Environments",
        "Regime 1: Well-Specified Debug",
        "Regime 2: Main Nonlinear Synthetic",
        "Regime 3: Coverage / Policy-Shift Stress",
        "Regime 4: Misspecification / Distortion",
        "Baseline Learners",
        "Calibration Protocols",
        "Calibrators",
        "Metrics",
        "Main Figures",
        "Appendix Figures",
        "Expected Qualitative Outcomes",
        "Known Limitations",
        "Implementation Status",
    ]
    for section in required:
        assert section in text


def test_run_suite_aggregate_tables_plots_and_separation(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "tiny.yaml"
    val_path = tmp_path / "tiny_validation.yaml"
    suite_path = tmp_path / "tiny_suite.yaml"
    tiny = _tiny_config()
    with cfg_path.open("w") as handle:
        yaml.safe_dump(tiny, handle)
    with val_path.open("w") as handle:
        yaml.safe_dump(tiny, handle)
    with suite_path.open("w") as handle:
        yaml.safe_dump(
            {
                "validation": {
                    "name": "well_specified_debug",
                    "config": str(val_path),
                    "gate": {"bias_threshold": 5.0, "mse_threshold": 10.0},
                },
                "suites": [{"name": "main_nonlinear", "config": str(cfg_path)}],
            },
            handle,
        )
    import FQE_calibration_neurips.scripts.run_suite as run_suite_module

    monkeypatch.setattr(run_suite_module, "ROOT", tmp_path)
    out = run_suite(suite_path, mode="debug", continue_on_failure=False)
    assert out == tmp_path / "results" / "debug"
    assert (out / "validation" / "well_specified_gate.json").exists()
    assert (out / "summary.csv").exists()
    assert (out / "eligible_summary.csv").exists()
    assert (out / "tables" / "main_nonlinear_results.csv").exists()
    assert (out / "tables" / "misspecification_sweep_summary.csv").exists()
    made = make_plots(out, tmp_path / "figures" / "debug")
    assert any(path.suffix == ".png" for path in made)
    assert not (tmp_path / "figures" / "paper").exists()
    assert not (tmp_path / "results" / "paper").exists()


def test_run_suite_honors_output_subdir(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "tiny.yaml"
    suite_path = tmp_path / "tiny_suite.yaml"
    tiny = _tiny_config()
    tiny["calibration_protocols"] = ["cross"]
    tiny["split_fractions"] = []
    tiny["split_comparators"] = []
    with cfg_path.open("w") as handle:
        yaml.safe_dump(tiny, handle)
    with suite_path.open("w") as handle:
        yaml.safe_dump(
            {
                "output_subdir": {"debug": "custom_debug_results"},
                "validation": {
                    "name": "well_specified_debug",
                    "config": str(cfg_path),
                    "gate": {"bias_threshold": 5.0, "mse_threshold": 10.0},
                },
                "suites": [],
            },
            handle,
        )
    import FQE_calibration_neurips.scripts.run_suite as run_suite_module

    monkeypatch.setattr(run_suite_module, "ROOT", tmp_path)
    out = run_suite(suite_path, mode="debug", continue_on_failure=False)
    assert out == tmp_path / "results" / "custom_debug_results"
    assert (out / "summary.csv").exists()


def test_focused_neurips_suite_runs_debug_with_strict_provenance_and_diagnostics(tmp_path: Path) -> None:
    suite_path = Path("FQE_calibration_neurips/configs/focused_neurips_suite.yaml")
    out = run_suite(suite_path, mode="debug", results_dir=tmp_path / "focused_debug")
    raw = pd.read_csv(out / "combined_raw_results.csv")
    summary = pd.read_csv(out / "summary.csv")
    assert {"model_misspecification_sweep", "undertraining_sweep", "mechanism_distortion_sweep"}.issubset(
        set(raw["suite_name"])
    )
    calibrated = raw[raw["calibrated"].astype(bool)].copy()
    assert not calibrated.empty
    assert calibrated["calibration_data_provenance"].astype(str).str.contains("out_of_fold|heldout").all()
    assert not calibrated["calibration_data_provenance"].astype(str).str.contains("test|oracle").any()
    assert set(raw["calibration_target"]) == {"value_bellman"}
    diag_cols = [
        "raw_value_oracle_pearson",
        "raw_value_oracle_spearman",
        "raw_value_calibration_slope",
        "raw_value_calibration_intercept",
    ]
    for col in diag_cols:
        assert col in raw.columns
        assert pd.to_numeric(raw[col], errors="coerce").notna().any()
        assert col in summary.columns
    cross = summary[summary["calibration_protocol"].astype(str).eq("cross")]
    assert not cross.empty
    assert pd.to_numeric(cross["relative_mse_vs_uncalibrated_all_data"], errors="coerce").notna().any()
    assert (out / "tables" / "focused_neurips_main_summary.csv").exists()
    made = make_plots(out, tmp_path / "focused_figures")
    assert any(path.name == "focused_neurips_calibration_story.png" for path in made)


def test_temporal_reward_shift_debug_runs_with_recent_provenance_and_retrain_baseline(tmp_path: Path) -> None:
    cfg = _tiny_temporal_config()
    raw_dir = tmp_path / "raw" / "temporal_reward_shift_sweep"
    rows = run_config(cfg, raw_dir, run_mode="debug", suite_name="temporal_reward_shift_sweep")
    raw = pd.DataFrame(rows)
    assert {"uncalibrated_all_data", "recent_heldout", "current_retrain_small"}.issubset(
        set(raw["calibration_protocol"])
    )
    recent = raw[raw["calibration_protocol"].astype(str).eq("recent_heldout")]
    assert not recent.empty
    assert recent["train_data_provenance"].astype(str).str.contains("old_regime_behavior_batch_seed").all()
    assert recent["calibration_data_provenance"].astype(str).str.contains("recent_current_regime_heldout_seed").all()
    cleaned = recent["calibration_data_provenance"].astype(str).str.replace("not_test_or_oracle", "", regex=False)
    assert not cleaned.str.contains("test|oracle", case=False, na=False).any()
    retrain = raw[raw["calibration_protocol"].astype(str).eq("current_retrain_small")]
    assert not retrain.empty
    assert retrain["train_data_provenance"].astype(str).str.contains("recent_current_regime_retrain_seed").all()

    aggregate_results(tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    temporal = summary[summary["suite_name"].astype(str).eq("temporal_reward_shift_sweep")]
    assert not temporal.empty
    assert "relative_true_v_mse_vs_current_retrain_small" in temporal.columns
    assert "relative_calibration_error_plugin_vs_current_retrain_small" in temporal.columns
    made = make_plots(tmp_path, tmp_path / "figures", allow_invalid=True)
    assert any(path.name == "focused_neurips_calibration_story.png" for path in made)


def test_rescue_suite_stages_are_separated_and_seed_disjoint() -> None:
    cfg = yaml.safe_load(Path("FQE_calibration_neurips/configs/rescue_neurips_suite.yaml").read_text())
    assert cfg["output_subdir"]["pilot"] != cfg["output_subdir"]["confirm"]
    assert cfg["output_subdir"]["confirm"] != cfg["output_subdir"]["final"]
    default_reps = {"pilot": 5, "confirm": 10, "final": 100}

    def expanded_seeds(stage_cfg: dict, stage: str) -> set[int]:
        stride = int(stage_cfg.get("replication_seed_stride", 10_000))
        return {int(stage_cfg["seed"]) + rep * stride for rep in range(default_reps[stage])}

    stage_seeds = {
        stage: cfg["validation"]["mode_overrides"][stage]["seed"]
        for stage in ["pilot", "confirm", "final"]
    }
    assert len(set(stage_seeds.values())) == 3
    validation = cfg["validation"]["mode_overrides"]
    assert expanded_seeds(validation["pilot"], "pilot").isdisjoint(expanded_seeds(validation["confirm"], "confirm"))
    assert expanded_seeds(validation["pilot"], "pilot").isdisjoint(expanded_seeds(validation["final"], "final"))
    assert expanded_seeds(validation["confirm"], "confirm").isdisjoint(expanded_seeds(validation["final"], "final"))
    for suite in cfg["suites"]:
        overrides = suite["mode_overrides"]
        seeds = [overrides[stage]["seed"] for stage in ["pilot", "confirm", "final"]]
        assert len(set(seeds)) == 3
        assert expanded_seeds(overrides["pilot"], "pilot").isdisjoint(expanded_seeds(overrides["confirm"], "confirm"))
        assert expanded_seeds(overrides["pilot"], "pilot").isdisjoint(expanded_seeds(overrides["final"], "final"))
        assert expanded_seeds(overrides["confirm"], "confirm").isdisjoint(expanded_seeds(overrides["final"], "final"))


def test_rescue_stage_debug_runs_and_writes_audit(tmp_path: Path) -> None:
    outputs = run_rescue_stage(
        stage="debug",
        suite_config="FQE_calibration_neurips/configs/rescue_neurips_suite.yaml",
        results_dir=tmp_path / "rescue_debug",
        figures_dir=tmp_path / "figures",
        skip_plots=True,
    )
    out = outputs["results_dir"]
    assert (out / "summary.csv").exists()
    assert (out / "rescue_promotion_audit.csv").exists()
    assert (out / "do_not_claim_manifest.csv").exists()
    audit = pd.read_csv(out / "rescue_promotion_audit.csv")
    assert "rescue_audit_label" in audit.columns
    assert "tuning_only" in set(audit["rescue_audit_label"]) or "limitation" in set(audit["rescue_audit_label"])


def test_rescue_audit_rejects_leakage_unstable_and_mse_only(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "rescue"
    raw_dir.mkdir(parents=True)
    common = {
        "run_mode": "final",
        "suite_name": "model_misspecification_sweep",
        "environment_name": "env",
        "sample_size": 100,
        "state_dimension": 3,
        "coverage_setting": "moderate",
        "misspecification_setting": "affine",
        "baseline_learner": "linear_fqe",
        "learner_variant": "linear_fqe_misspecified",
        "calibration_target": "value_bellman",
        "main_figure_role": "main",
        "calibrated": True,
        "calibration_protocol": "cross",
        "calibrator": "linear",
        "base_learner_used_all_data": False,
        "train_data_provenance": "offline_behavior_batch_seed=1;not_test_or_oracle",
        "calibration_data_provenance": "pooled_out_of_fold_training_predictions",
        "test_data_provenance": "independent_test_seed=2",
        "failure_flag": False,
    }
    summary_rows = [
        {
            **common,
            "relative_true_v_mse_vs_uncalibrated_all_data": 0.8,
            "relative_mse_vs_uncalibrated_all_data": 0.8,
            "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.8,
            "relative_brier_score_vs_uncalibrated_all_data": 1.1,
            "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.8,
            "calibration_error_plugin_win_rate_vs_uncalibrated_all_data": 0.8,
            "failure_rate": 0.0,
            "eligible_fraction": 1.0,
            "raw_value_oracle_spearman": 0.5,
            "raw_value_oracle_pearson": 0.5,
            "raw_value_calibration_slope": 0.7,
            "raw_value_calibration_intercept": 0.2,
        },
        {
            **common,
            "learner_variant": "leaky",
            "calibration_data_provenance": "test_oracle_bad",
            "relative_true_v_mse_vs_uncalibrated_all_data": 0.8,
            "relative_mse_vs_uncalibrated_all_data": 0.8,
            "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.8,
            "relative_brier_score_vs_uncalibrated_all_data": 0.8,
            "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.8,
            "calibration_error_plugin_win_rate_vs_uncalibrated_all_data": 0.8,
            "failure_rate": 0.0,
            "eligible_fraction": 1.0,
            "raw_value_oracle_spearman": 0.5,
            "raw_value_oracle_pearson": 0.5,
            "raw_value_calibration_slope": 0.7,
            "raw_value_calibration_intercept": 0.2,
        },
        {
            **common,
            "learner_variant": "mse_only",
            "relative_true_v_mse_vs_uncalibrated_all_data": 0.8,
            "relative_mse_vs_uncalibrated_all_data": 0.8,
            "relative_calibration_error_plugin_vs_uncalibrated_all_data": 1.1,
            "relative_brier_score_vs_uncalibrated_all_data": 1.1,
            "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.8,
            "calibration_error_plugin_win_rate_vs_uncalibrated_all_data": 0.8,
            "failure_rate": 0.0,
            "eligible_fraction": 1.0,
            "raw_value_oracle_spearman": 0.5,
            "raw_value_oracle_pearson": 0.5,
            "raw_value_calibration_slope": 0.7,
            "raw_value_calibration_intercept": 0.2,
        },
        {
            **common,
            "learner_variant": "unstable",
            "relative_true_v_mse_vs_uncalibrated_all_data": 0.8,
            "relative_mse_vs_uncalibrated_all_data": 0.8,
            "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.8,
            "relative_brier_score_vs_uncalibrated_all_data": 0.8,
            "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.8,
            "calibration_error_plugin_win_rate_vs_uncalibrated_all_data": 0.8,
            "failure_rate": 1.0,
            "eligible_fraction": 0.0,
            "raw_value_oracle_spearman": 0.5,
            "raw_value_oracle_pearson": 0.5,
            "raw_value_calibration_slope": 0.7,
            "raw_value_calibration_intercept": 0.2,
        },
        {
            **common,
            "suite_name": "well_specified_debug",
            "misspecification_setting": "well_specified_linear",
            "learner_variant": "validation_control",
            "relative_true_v_mse_vs_uncalibrated_all_data": 0.8,
            "relative_mse_vs_uncalibrated_all_data": 0.8,
            "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.8,
            "relative_brier_score_vs_uncalibrated_all_data": 0.8,
            "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.8,
            "calibration_error_plugin_win_rate_vs_uncalibrated_all_data": 0.8,
            "failure_rate": 0.0,
            "eligible_fraction": 1.0,
            "raw_value_oracle_spearman": 0.5,
            "raw_value_oracle_pearson": 0.5,
            "raw_value_calibration_slope": 0.7,
            "raw_value_calibration_intercept": 0.2,
        },
    ]
    raw = pd.DataFrame(summary_rows)
    raw.to_csv(raw_dir / "raw_results.csv", index=False)
    raw.to_csv(tmp_path / "combined_raw_results.csv", index=False)
    raw.to_csv(tmp_path / "summary.csv", index=False)
    audit_rescue_submission(tmp_path)
    audit = pd.read_csv(tmp_path / "rescue_promotion_audit.csv")
    labels = dict(zip(audit["learner_variant"], audit["rescue_audit_label"]))
    assert labels["linear_fqe_misspecified"] == "promote_main"
    assert labels["leaky"] == "reject_leakage_or_data_use"
    assert labels["mse_only"] == "reject_mse_only"
    assert labels["unstable"] == "reject_unstable"
    assert labels["validation_control"] == "validation_control"


def test_rescue_audit_promotes_only_temporal_recent_heldout_with_retrain_support(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "temporal_reward_shift_sweep"
    raw_dir.mkdir(parents=True)
    common = {
        "run_mode": "final",
        "environment_name": "nonlinear_discrete_action",
        "sample_size": 2000,
        "state_dimension": 5,
        "coverage_setting": "moderate",
        "misspecification_setting": "temporal_reward_shift",
        "baseline_learner": "random_feature_fqe",
        "learner_variant": "temporal_rf_fqe",
        "calibration_target": "value_bellman",
        "main_figure_role": "main",
        "calibrated": True,
        "calibration_protocol": "recent_heldout",
        "calibrator": "linear",
        "base_learner_used_all_data": False,
        "train_data_provenance": "old_regime_behavior_batch_seed=1;not_test_or_oracle",
        "calibration_data_provenance": "recent_current_regime_heldout_seed=2;not_test_or_oracle",
        "test_data_provenance": "current_regime_independent_test_seed=3;independent_oracle_seed=4",
        "failure_flag": False,
        "relative_true_v_mse_vs_uncalibrated_all_data": 0.75,
        "relative_mse_vs_uncalibrated_all_data": 0.75,
        "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.7,
        "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.8,
        "calibration_error_plugin_win_rate_vs_uncalibrated_all_data": 0.8,
        "relative_true_v_mse_vs_current_retrain_small": 0.95,
        "relative_calibration_error_plugin_vs_current_retrain_small": 0.9,
        "true_v_mse_win_rate_vs_current_retrain_small": 0.8,
        "calibration_error_plugin_win_rate_vs_current_retrain_small": 0.8,
        "failure_rate": 0.0,
        "eligible_fraction": 1.0,
        "raw_value_oracle_spearman": 0.5,
        "raw_value_oracle_pearson": 0.5,
        "raw_value_calibration_slope": 0.7,
        "raw_value_calibration_intercept": 0.2,
    }
    rows = [
        {**common, "suite_name": "temporal_reward_shift_sweep"},
        {**common, "suite_name": "model_misspecification_sweep", "learner_variant": "bad_recent_heldout"},
    ]
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_dir / "raw_results.csv", index=False)
    raw.to_csv(tmp_path / "combined_raw_results.csv", index=False)
    raw.to_csv(tmp_path / "summary.csv", index=False)
    audit_rescue_submission(tmp_path)
    audit = pd.read_csv(tmp_path / "rescue_promotion_audit.csv")
    labels = dict(zip(audit["learner_variant"], audit["rescue_audit_label"]))
    assert labels["temporal_rf_fqe"] == "promote_main"
    assert labels["bad_recent_heldout"] == "reject_leakage_or_data_use"


def test_aggregation_excludes_failed_from_eligible_and_matches_relative_mse(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "main_nonlinear"
    raw_dir.mkdir(parents=True)
    rows = []
    base = {
        "run_mode": "debug",
        "suite_name": "main_nonlinear",
        "environment_name": "env",
        "environment_tier": "nonlinear_synthetic",
        "sample_size": 100,
        "state_dimension": 3,
        "coverage_setting": "good",
        "policy_shift_setting": 0.1,
        "reward_noise_setting": 0.0,
        "misspecification_setting": "none",
        "baseline_learner": "linear_fqe",
        "calibration_target": "value_bellman",
        "split_fraction": 1.0,
        "train_fraction": 1.0,
        "replication_seed": 1,
        "value_estimate": 1.0,
        "oracle_value": 0.0,
        "value_error": 1.0,
        "squared_error": 1.0,
        "bellman_residual": 0.1,
        "calibration_error": 0.1,
        "runtime": 0.01,
        "failure_flag": False,
        "main_evidence_eligible": True,
        "coverage_stratum_0_error": 0.2,
        "coverage_stratum_0_count": 10,
        "coverage_stratum_0_mean_ratio": 0.8,
        "coverage_stratum_0_calibration_error": 0.15,
        "coverage_stratum_1_error": 0.4,
        "coverage_stratum_1_count": 10,
        "coverage_stratum_1_mean_ratio": 1.2,
        "coverage_stratum_1_calibration_error": 0.35,
    }
    rows.append({**base, "calibrated": False, "calibration_protocol": "uncalibrated_all_data", "calibrator": "none"})
    rows.append({**base, "calibrated": True, "calibration_protocol": "cross", "calibrator": "linear", "value_error": 0.5, "squared_error": 0.25})
    rows.append({**base, "calibrated": True, "calibration_protocol": "no_split", "calibrator": "linear", "failure_flag": True, "main_evidence_eligible": False})
    pd.DataFrame(rows).to_csv(raw_dir / "raw_results.csv", index=False)
    aggregate_results(tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    cross = summary[summary["calibration_protocol"] == "cross"].iloc[0]
    assert abs(float(cross["relative_mse_vs_uncalibrated_all_data"]) - 0.25) < 1e-8
    coverage = pd.read_csv(tmp_path / "coverage_stratified_error.csv")
    assert {
        "coverage_stratum",
        "mean_coverage_stratum_error",
        "mean_coverage_stratum_calibration_error",
        "mean_coverage_stratum_ratio",
    }.issubset(coverage.columns)
    eligible = pd.read_csv(tmp_path / "eligible_summary.csv")
    assert "no_split" not in set(eligible["calibration_protocol"])
    paths = write_tables(tmp_path)
    assert any(path.suffix == ".tex" for path in paths)
    assert (tmp_path / "tables" / "misspecification_sweep_summary.csv").exists()
    assert (tmp_path / "calibration_evidence_audit.csv").exists()
    assert (tmp_path / "scalar_cancellation_audit.csv").exists()


def test_paper_plots_refuse_failed_validation_gate(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "paper"
    validation = results_dir / "validation"
    validation.mkdir(parents=True)
    with (validation / "well_specified_gate.json").open("w") as handle:
        json.dump({"gate_passed": False}, handle)
    try:
        make_plots(results_dir, tmp_path / "figures" / "paper")
    except RuntimeError as exc:
        assert "validation gate failed" in str(exc)
    else:
        raise AssertionError("paper plotting should refuse a failed validation gate")


def test_calibration_quality_presets_run_tiny_cpu(tmp_path: Path) -> None:
    config_path = Path("FQE_calibration_neurips/configs/calibration_quality_sweep.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    cfg.update(
        {
            "seed": 909,
            "replications": 1,
            "sample_sizes": [45],
            "state_dimensions": [3],
            "coverage_settings": ["moderate"],
            "reward_noise_settings": [0.05],
            "test_transitions": 35,
            "initial_eval_states": 35,
            "oracle_rollouts": 30,
            "cross_folds": 2,
            "calibration_protocols": ["cross"],
            "calibrators": ["linear"],
            "split_fractions": [0.8],
            "split_comparators": [],
            "learner_params": {
                "neural_fqe": {"hidden_dims": [8], "n_iters": 2, "epochs_per_iter": 1, "batch_size": 16, "lr": 0.002, "device": "cpu"},
                "random_feature_fqe": {"n_components": 12, "bandwidth": 0.7, "ridge": 0.01, "n_iters": 3},
                "regularized_bellman": {"n_components": 12, "bandwidth": 0.7, "ridge": 0.01},
                "saddle_point_bellman": {"n_components": 12, "bandwidth": 0.7, "ridge": 0.01, "critic_ridge": 0.01},
                "saddle_point_iterative": {"n_components": 10, "bandwidth": 0.7, "q_ridge": 0.01, "critic_ridge": 0.01, "max_iters": 4, "step_size": 0.002, "gradient_clip": 10.0, "divergence_threshold": 1000.0},
                "ensemble_fqe": {"n_members": 2, "n_components": 10, "bandwidth": 0.7, "ridge": 0.01, "n_iters": 3},
            },
        }
    )
    for variant in cfg["learner_variants"].values():
        params = variant.setdefault("params", {})
        if variant["base_learner"] == "neural_fqe":
            params.update({"hidden_dims": [8], "n_iters": 2, "epochs_per_iter": 1, "batch_size": 16})
        elif variant["base_learner"] in {"random_feature_fqe", "regularized_bellman", "saddle_point_bellman"}:
            params.update({"n_components": 12})
        elif variant["base_learner"] == "saddle_point_iterative":
            params.update({"n_components": 10, "max_iters": 4, "step_size": 0.002, "divergence_threshold": 1000.0})
        elif variant["base_learner"] == "ensemble_fqe":
            params.update({"n_members": 2, "n_components": 10, "n_iters": 2})
    rows = run_config(cfg, tmp_path, run_mode="debug", suite_name="calibration_quality_sweep")
    variants = {row["learner_variant"] for row in rows}
    assert set(cfg["baseline_learners"]).issubset(variants)
    for row in rows:
        assert row["learner_quality_regime"] in {
            "well_tuned",
            "under_iterated",
            "misspecified",
            "underregularized",
            "overregularized",
            "ill_conditioned",
            "weak_ensemble",
        }
        assert row["calibration_difficulty"] in {
            "already_well_calibrated",
            "affine_miscalibrated",
            "monotone_miscalibrated",
            "nonmonotone_error",
            "coverage_limited",
            "bellman_incomplete",
            "finite_iteration_bias",
            "model_misspecified",
        }
        assert row["main_figure_role"] in {"main", "appendix", "diagnostic_only", "mechanism"}


def test_mechanism_distortion_sweep_runs_tiny_cpu(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("FQE_calibration_neurips/configs/mechanism_distortion_sweep.yaml").read_text())
    cfg.update(
        {
            "seed": 919,
            "replications": 1,
            "sample_sizes": [50],
            "state_dimensions": [3],
            "test_transitions": 35,
            "initial_eval_states": 35,
            "oracle_rollouts": 30,
            "cross_folds": 2,
            "calibrator_params": {"n_bins": 4, "min_bin_size": 4},
            "learner_params": {"random_feature_fqe": {"n_components": 12, "bandwidth": 0.7, "ridge": 0.01, "n_iters": 3}},
        }
    )
    rows = run_config(cfg, tmp_path, run_mode="debug", suite_name="mechanism_distortion_sweep")
    assert rows
    variants = {row["learner_variant"] for row in rows}
    assert {
        "random_feature_fqe_affine_distorted",
        "random_feature_fqe_monotone_saturation_distorted",
    }.issubset(variants)
    for row in rows:
        assert row["main_figure_role"] == "mechanism"
        assert row["calibration_difficulty"] in {"affine_miscalibrated", "monotone_miscalibrated"}


def test_quality_metadata_aggregation_tables_and_plots(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "calibration_quality_sweep"
    raw_dir.mkdir(parents=True)
    rows = []
    for variant, quality, base_error, cross_error in [
        ("linear_fqe_well_tuned", "well_tuned", 1.0, 0.9),
        ("linear_fqe_under_iterated", "under_iterated", 4.0, 2.0),
    ]:
        common = {
            "run_mode": "debug",
            "suite_name": "calibration_quality_sweep",
            "environment_name": "env",
            "environment_tier": "nonlinear_synthetic",
            "sample_size": 100,
            "state_dimension": 3,
            "coverage_setting": "moderate",
            "policy_shift_setting": 0.75,
            "reward_noise_setting": 0.1,
            "misspecification_setting": "none",
            "baseline_learner": "linear_fqe",
            "learner_variant": variant,
            "learner_quality_regime": quality,
            "calibration_difficulty": "already_well_calibrated" if quality == "well_tuned" else "affine_miscalibrated",
            "main_figure_role": "main",
            "calibration_target": "value_bellman",
            "split_fraction": 1.0,
            "train_fraction": 1.0,
            "replication_seed": 1,
            "oracle_value": 0.0,
            "bellman_residual": 0.1,
            "calibration_error": 0.1,
            "runtime": 0.01,
            "failure_flag": False,
            "main_evidence_eligible": True,
        }
        rows.append({**common, "calibrated": False, "calibration_protocol": "uncalibrated_all_data", "calibrator": "none", "value_estimate": base_error, "value_error": base_error, "squared_error": base_error**2})
        rows.append({**common, "calibrated": True, "calibration_protocol": "cross", "calibrator": "linear", "value_estimate": cross_error, "value_error": cross_error, "squared_error": cross_error**2})
    rows.append({**rows[-1], "learner_variant": "linear_fqe_diagnostic", "learner_quality_regime": "ill_conditioned", "main_figure_role": "diagnostic_only", "failure_flag": True, "main_evidence_eligible": False})
    pd.DataFrame(rows).to_csv(raw_dir / "raw_results.csv", index=False)
    aggregate_results(tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    under = summary[(summary["learner_variant"] == "linear_fqe_under_iterated") & (summary["calibration_protocol"] == "cross")].iloc[0]
    assert abs(float(under["relative_mse_vs_uncalibrated_all_data"]) - 0.25) < 1e-8
    assert {"well_tuned", "under_iterated", "ill_conditioned"}.issubset(set(summary["learner_quality_regime"]))
    assert (tmp_path / "tables" / "calibration_quality_sweep_summary.csv").exists()
    made = make_plots(tmp_path, tmp_path / "figures")
    made_names = {path.name for path in made}
    assert "mse_by_learner_quality.png" in made_names
    assert "relative_mse_by_calibration_difficulty.png" in made_names
    assert "failure_rate_by_learner_quality.png" in made_names
    assert "mse_vs_calibration_error_improvement.png" in made_names


def test_evidence_status_flags_scalar_cancellation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "main_nonlinear"
    raw_dir.mkdir(parents=True)
    common = {
        "run_mode": "debug",
        "suite_name": "main_nonlinear",
        "environment_name": "env",
        "environment_tier": "nonlinear_synthetic",
        "sample_size": 100,
        "state_dimension": 3,
        "coverage_setting": "moderate",
        "policy_shift_setting": 0.7,
        "reward_noise_setting": 0.1,
        "misspecification_setting": "none",
        "baseline_learner": "linear_fqe",
        "learner_variant": "linear_fqe",
        "learner_quality_regime": "well_tuned",
        "calibration_difficulty": "already_well_calibrated",
        "main_figure_role": "main",
        "calibration_target": "value_bellman",
        "split_fraction": 1.0,
        "train_fraction": 1.0,
        "replication_seed": 1,
        "value_estimate": 0.0,
        "oracle_value": 0.0,
        "value_error": 0.0,
        "bellman_residual": 0.1,
        "calibration_error": 0.1,
        "bellman_calibration_error_plugin": 0.2,
        "bellman_calibration_error_debiased_raw": 0.2,
        "brier_score": 0.2,
        "bellman_outcome_mse": 0.2,
        "true_v_mse": 0.2,
        "runtime": 0.01,
        "failure_flag": False,
        "main_evidence_eligible": True,
    }
    rows = [
        {**common, "calibrated": False, "calibration_protocol": "uncalibrated_all_data", "calibrator": "none", "squared_error": 1.0},
        {
            **common,
            "calibrated": True,
            "calibration_protocol": "cross",
            "calibrator": "linear",
            "squared_error": 0.25,
            "bellman_calibration_error_plugin": 0.1,
            "brier_score": 0.18,
            "bellman_outcome_mse": 0.18,
            "true_v_mse": 0.19,
        },
        {
            **common,
            "calibrated": True,
            "calibration_protocol": "cross",
            "calibrator": "isotonic",
            "squared_error": 0.25,
            "bellman_calibration_error_plugin": 0.3,
            "brier_score": 0.3,
            "bellman_outcome_mse": 0.3,
            "true_v_mse": 0.3,
        },
    ]
    pd.DataFrame(rows).to_csv(raw_dir / "raw_results.csv", index=False)
    aggregate_results(tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    strong = summary[summary["calibrator"].eq("linear")].iloc[0]
    mse_only = summary[summary["calibrator"].eq("isotonic")].iloc[0]
    assert strong["calibration_evidence_status"] == "strong"
    assert mse_only["calibration_evidence_status"] == "mse_only"
    assert bool(mse_only["scalar_cancellation_flag"])


def test_strict_submission_audit_promotes_only_primary_dual_metric_cross_rows(tmp_path: Path) -> None:
    results_dir = tmp_path / "paper_strict_cross_submission_v2"
    results_dir.mkdir()
    summary = pd.DataFrame(
        [
            {
                "run_mode": "paper",
                "suite_name": "model_misspecification_sweep",
                "environment_name": "env",
                "environment_tier": "nonlinear_synthetic",
                "sample_size": 100,
                "state_dimension": 4,
                "coverage_setting": "moderate",
                "policy_shift_setting": 0.7,
                "reward_noise_setting": 0.1,
                "misspecification_setting": "affine",
                "baseline_learner": "random_feature_fqe",
                "learner_variant": "random_feature_fqe_restricted",
                "learner_quality_regime": "misspecified",
                "calibration_difficulty": "model_misspecified",
                "main_figure_role": "main",
                "calibrated": True,
                "calibration_protocol": "cross",
                "calibrator": "isotonic_histogram",
                "calibration_target": "value_bellman",
                "split_fraction": 1.0,
                "train_fraction": 0.8,
                "relative_true_v_mse_vs_uncalibrated_all_data": 0.8,
                "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.9,
                "relative_brier_score_vs_uncalibrated_all_data": 1.01,
                "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.7,
                "failure_rate": 0.0,
                "eligible_fraction": 1.0,
                "calibration_evidence_status": "strong",
            },
            {
                "run_mode": "paper",
                "suite_name": "model_misspecification_sweep",
                "environment_name": "env",
                "environment_tier": "nonlinear_synthetic",
                "sample_size": 100,
                "state_dimension": 4,
                "coverage_setting": "moderate",
                "policy_shift_setting": 0.7,
                "reward_noise_setting": 0.1,
                "misspecification_setting": "affine",
                "baseline_learner": "random_feature_fqe",
                "learner_variant": "random_feature_fqe_restricted",
                "learner_quality_regime": "misspecified",
                "calibration_difficulty": "model_misspecified",
                "main_figure_role": "main",
                "calibrated": True,
                "calibration_protocol": "cross",
                "calibrator": "linear",
                "calibration_target": "value_bellman",
                "split_fraction": 1.0,
                "train_fraction": 0.8,
                "relative_true_v_mse_vs_uncalibrated_all_data": 0.7,
                "relative_calibration_error_plugin_vs_uncalibrated_all_data": 0.7,
                "relative_brier_score_vs_uncalibrated_all_data": 0.7,
                "true_v_mse_win_rate_vs_uncalibrated_all_data": 0.9,
                "failure_rate": 0.0,
                "eligible_fraction": 1.0,
                "calibration_evidence_status": "strong",
            },
        ]
    )
    summary.to_csv(results_dir / "summary.csv", index=False)
    raw = []
    for calibrator in ["isotonic_histogram", "linear"]:
        for seed in [1, 2]:
            raw.append(
                {
                    "suite_name": "model_misspecification_sweep",
                    "baseline_learner": "random_feature_fqe",
                    "learner_variant": "random_feature_fqe_restricted",
                    "calibration_protocol": "cross",
                    "calibrator": calibrator,
                    "calibration_target": "value_bellman",
                    "sample_size": 100,
                    "state_dimension": 4,
                    "coverage_setting": "moderate",
                    "misspecification_setting": "affine",
                    "calibrated": True,
                    "base_learner_used_all_data": False,
                    "replication_seed": seed,
                }
            )
    pd.DataFrame(raw).to_csv(results_dir / "combined_raw_results.csv", index=False)
    outputs = audit_strict_submission(results_dir)
    audit = pd.read_csv(outputs["audit"])
    promoted = audit[audit["strict_cross_promote_to_main"]]
    assert set(promoted["calibrator"]) == {"isotonic_histogram"}
    assert outputs["readout"].exists()


def test_new_failure_mode_configs_run_tiny_cpu(tmp_path: Path) -> None:
    configs = [
        ("undertraining_sweep.yaml", ["random_feature_fqe_iter1", "random_feature_fqe_well_tuned"]),
        ("bellman_incomplete_sweep.yaml", ["linear_fqe_restricted", "random_feature_fqe_low_capacity"]),
        ("model_misspecification_sweep.yaml", ["linear_fqe_misspecified", "random_feature_fqe_restricted"]),
    ]
    for name, learners in configs:
        cfg = yaml.safe_load(Path(f"FQE_calibration_neurips/configs/{name}").read_text())
        cfg.update(
            {
                "seed": 777,
                "replications": 1,
                "sample_sizes": [35],
                "state_dimensions": [3],
                "coverage_settings": ["moderate"],
                "reward_noise_settings": [0.05],
                "test_transitions": 30,
                "diagnostic_test_transitions": 35,
                "initial_eval_states": 30,
                "oracle_rollouts": 25,
                "true_q_rollouts_per_state": 1,
                "cross_folds": 2,
                "calibration_error_bins": 4,
                "calibration_error_min_bin_size": 4,
                "interval_bootstrap_reps": 10,
                "baseline_learners": learners,
                "calibration_protocols": ["cross"],
                "calibrators": ["linear"],
                "split_fractions": [],
                "learner_params": {
                    "linear_fqe": {"feature_type": "linear", "ridge": 0.001, "n_iters": 3},
                    "random_feature_fqe": {"n_components": 10, "bandwidth": 0.7, "ridge": 0.01, "n_iters": 3},
                    "neural_fqe": {"hidden_dims": [8], "n_iters": 2, "epochs_per_iter": 1, "batch_size": 16, "device": "cpu"},
                    "ensemble_fqe": {"n_members": 2, "n_components": 8, "n_iters": 2},
                    "regularized_bellman": {"n_components": 10, "ridge": 0.01},
                },
            }
        )
        rows = run_config(cfg, tmp_path / name, run_mode="debug", suite_name=name.removesuffix(".yaml"))
        assert rows
        assert {row["learner_variant"] for row in rows}.issuperset(set(learners))


def test_split_stability_diagnostics_match_correct_baselines(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "split_fraction_sweep"
    raw_dir.mkdir(parents=True)
    rows = []
    common = {
        "run_mode": "debug",
        "suite_name": "split_fraction_sweep",
        "environment_name": "env",
        "environment_tier": "nonlinear_synthetic",
        "sample_size": 100,
        "state_dimension": 3,
        "coverage_setting": "good",
        "policy_shift_setting": 0.1,
        "reward_noise_setting": 0.0,
        "misspecification_setting": "none",
        "baseline_learner": "linear_fqe",
        "learner_variant": "linear_fqe",
        "learner_quality_regime": "well_tuned",
        "calibration_difficulty": "already_well_calibrated",
        "main_figure_role": "main",
        "calibration_target": "value_bellman",
        "calibrator": "linear",
        "calibrated": True,
        "value_estimate": 0.0,
        "oracle_value": 0.0,
        "value_error": 0.0,
        "bellman_residual": 0.1,
        "calibration_error": 0.1,
        "runtime": 0.01,
        "failure_flag": False,
        "main_evidence_eligible": True,
        "calibration_fraction": 0.2,
    }
    for seed, split_se, all_se, same_se in [(1, 0.25, 1.0, 0.5), (2, 0.75, 1.5, 1.5)]:
        base = {**common, "replication_seed": seed}
        rows.append({**base, "calibration_protocol": "split", "train_fraction": 0.8, "split_fraction": 0.8, "squared_error": split_se})
        rows.append({**base, "calibrated": False, "calibration_protocol": "uncalibrated_all_data", "calibrator": "none", "train_fraction": 1.0, "split_fraction": 1.0, "squared_error": all_se})
        rows.append({**base, "calibrated": False, "calibration_protocol": "uncalibrated_same_fraction", "calibrator": "none", "train_fraction": 0.8, "split_fraction": 0.8, "squared_error": same_se})
    pd.DataFrame(rows).to_csv(raw_dir / "raw_results.csv", index=False)
    aggregate_results(tmp_path)
    split = pd.read_csv(tmp_path / "split_stability_diagnostics.csv")
    row = split.iloc[0]
    assert int(row["n_replications"]) == 2
    assert abs(float(row["mean_relative_mse_vs_all_data"]) - 0.375) < 1e-8
    assert abs(float(row["mean_relative_mse_vs_same_fraction"]) - 0.5) < 1e-8
    assert abs(float(row["win_rate_vs_all_data"]) - 1.0) < 1e-8
    made = make_plots(tmp_path, tmp_path / "figures")
    assert (tmp_path / "figures" / "split_stability_diagnostics.png") in made


def test_inspect_paper_draft_writes_readout_and_flags_mechanism(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "paper"
    validation = results_dir / "validation"
    validation.mkdir(parents=True)
    with (validation / "well_specified_gate.json").open("w") as handle:
        json.dump({"gate_passed": True}, handle)
    rows = []
    base = {
        "run_mode": "paper",
        "suite_name": "mechanism_distortion_sweep",
        "baseline_learner": "random_feature_fqe",
        "learner_variant": "variant",
        "learner_quality_regime": "misspecified",
        "main_figure_role": "mechanism",
        "calibration_protocol": "cross",
        "calibration_target": "value_bellman",
        "sample_size": 100,
        "n_replications": 10,
        "mse": 1.0,
        "mse_mc_se": 0.1,
        "relative_mse_vs_uncalibrated_all_data": 0.9,
        "failure_rate": 0.0,
    }
    rows.append({**base, "learner_variant": "affine_variant", "calibration_difficulty": "affine_miscalibrated", "calibrator": "linear", "relative_mse_vs_uncalibrated_all_data": 0.75})
    rows.append({**base, "learner_variant": "affine_variant", "calibration_difficulty": "affine_miscalibrated", "calibrator": "isotonic", "relative_mse_vs_uncalibrated_all_data": 0.76})
    rows.append({**base, "learner_variant": "monotone_easy", "calibration_difficulty": "monotone_miscalibrated", "calibrator": "linear", "relative_mse_vs_uncalibrated_all_data": 0.60})
    rows.append({**base, "learner_variant": "monotone_easy", "calibration_difficulty": "monotone_miscalibrated", "calibrator": "isotonic", "relative_mse_vs_uncalibrated_all_data": 0.70})
    rows.append({**base, "learner_variant": "monotone_hard", "calibration_difficulty": "monotone_miscalibrated", "calibrator": "linear", "relative_mse_vs_uncalibrated_all_data": 0.95})
    rows.append({**base, "learner_variant": "monotone_hard", "calibration_difficulty": "monotone_miscalibrated", "calibrator": "isotonic", "relative_mse_vs_uncalibrated_all_data": 0.88})
    pd.DataFrame(rows).to_csv(results_dir / "eligible_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)
    out = inspect_paper_draft(results_dir, tmp_path / "figures" / "paper")
    text = out.read_text()
    assert "Validation gate passed: `True`" in text
    assert "Affine mechanism" in text
    assert "Monotone mechanism" in text
    assert "matched variant `monotone_hard`" in text
    assert "fail" in text
