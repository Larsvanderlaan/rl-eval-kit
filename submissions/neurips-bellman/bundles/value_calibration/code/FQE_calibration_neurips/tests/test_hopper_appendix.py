from __future__ import annotations

from dataclasses import replace

import numpy as np

from FQE_calibration_neurips.scripts.run_hopper_calibration_benchmark import (
    _promotion_audit,
    _subset_dataset_by_trajectories,
    _summary,
    _trajectory_folds,
    _trajectory_ids_from_steps,
    apply_stage_defaults,
    clipped_normalized_density_weights,
    fit_behavior_density,
    target_policy_log_prob,
)
from FQE_calibration_neurips.scripts.screen_deep_ope_value_calibration import _audit as _deep_ope_audit
from FQE_calibration_neurips.scripts.screen_deep_ope_value_calibration import _summarize as _deep_ope_summarize
import argparse
from hopper_fqe_benchmark.data import HopperTrajectoryDataset


class _TinyPolicy:
    output_distribution = "tanh_gaussian"

    def _forward(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        mean = np.zeros((obs.shape[0], 2), dtype=np.float32)
        mean[:, 0] = 0.1 * obs[:, 0]
        log_std = -0.5 * np.ones_like(mean, dtype=np.float32)
        return mean, log_std


def _tiny_hopper_dataset() -> HopperTrajectoryDataset:
    observations_raw = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.tanh(np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.1], [0.3, 0.0], [0.4, -0.1], [0.5, -0.2]], dtype=np.float32))
    next_observations_raw = observations_raw + 0.05
    rewards_raw = np.arange(6, dtype=np.float32)
    masks = np.ones(6, dtype=np.float32)
    steps = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    state_mean = observations_raw.mean(axis=0)
    state_std = observations_raw.std(axis=0)
    state_std = np.where(state_std < 1e-5, 1.0, state_std).astype(np.float32)
    observations = ((observations_raw - state_mean) / state_std).astype(np.float32)
    next_observations = ((next_observations_raw - state_mean) / state_std).astype(np.float32)
    return HopperTrajectoryDataset(
        observations_raw=observations_raw,
        actions=actions,
        next_observations_raw=next_observations_raw,
        rewards_raw=rewards_raw,
        masks=masks,
        steps=steps,
        initial_observations_raw=observations_raw[[0, 2, 4]],
        initial_weights=np.ones(3, dtype=np.float32),
        state_mean=state_mean,
        state_std=state_std,
        reward_mean=0.0,
        reward_std=1.0,
        observations=observations,
        next_observations=next_observations,
        rewards=rewards_raw.copy(),
        trajectory_count=3,
    )


def test_hopper_trajectory_folds_and_subset_are_disjoint() -> None:
    dataset = _tiny_hopper_dataset()
    ids = _trajectory_ids_from_steps(dataset.steps)
    assert ids.tolist() == [0, 0, 1, 1, 2, 2]
    folds = _trajectory_folds(3, 3, seed=1)
    assert sorted(np.concatenate(folds).tolist()) == [0, 1, 2]
    assert len({int(i) for fold in folds for i in fold}) == 3
    subset = _subset_dataset_by_trajectories(dataset, [0, 2])
    assert subset.trajectory_count == 2
    assert len(subset) == 4
    assert subset.initial_observations_raw.shape[0] == 2


def test_hopper_density_weights_are_finite_and_normalized() -> None:
    dataset = _tiny_hopper_dataset()
    behavior = fit_behavior_density(dataset)
    target_log = target_policy_log_prob(_TinyPolicy(), dataset.observations_raw, dataset.actions)
    behavior_log = behavior.log_prob(dataset.observations_raw, dataset.actions)
    assert np.all(np.isfinite(target_log))
    assert np.all(np.isfinite(behavior_log))
    weights, diagnostics = clipped_normalized_density_weights(target_log, behavior_log, clip=5.0)
    assert np.all(np.isfinite(weights))
    assert abs(float(np.mean(weights)) - 1.0) < 1e-8
    assert diagnostics["importance_weight_ess_fraction"] > 0


def test_hopper_output_summary_schema() -> None:
    rows = [
        {
            "method": "uncalibrated_all_data_neural_fqe",
            "absolute_error": 2.0,
            "bellman_brier_score": 1.0,
            "bellman_calibration_error": 0.5,
            "importance_weight_ess_fraction": 0.7,
            "diagnostic_only": False,
        },
        {
            "method": "strict_cross_isotonic",
            "absolute_error": 1.0,
            "bellman_brier_score": 0.8,
            "bellman_calibration_error": 0.3,
            "importance_weight_ess_fraction": 0.6,
            "diagnostic_only": False,
        },
    ]
    summary = _summary(rows)
    methods = {row["method"] for row in summary}
    assert {"uncalibrated_all_data_neural_fqe", "strict_cross_isotonic"}.issubset(methods)
    for row in summary:
        assert "mean_absolute_error" in row
        assert "mean_bellman_calibration_error" in row
        assert "relative_absolute_error_vs_raw" in row
        assert "relative_bellman_calibration_error_vs_raw" in row


def test_hopper_promotion_audit_requires_ope_and_calibration_gains() -> None:
    rows = []
    for seed in [0, 1]:
        for idx, truth in enumerate([10.0, 20.0, 30.0]):
            policy_id = f"p{idx}"
            raw_est = truth + 4.0
            rows.append(
                {
                    "dataset_name": "hopper-medium-v0",
                    "method": "uncalibrated_all_data_neural_fqe",
                    "seed": seed,
                    "policy_id": policy_id,
                    "ground_truth_return": truth,
                    "estimated_return": raw_est,
                    "absolute_error": abs(raw_est - truth),
                    "bellman_calibration_error": 10.0,
                    "importance_weight_ess_fraction": 0.5,
                    "diagnostic_only": False,
                }
            )
            rows.append(
                {
                    "dataset_name": "hopper-medium-v0",
                    "method": "strict_cross_linear",
                    "seed": seed,
                    "policy_id": policy_id,
                    "ground_truth_return": truth,
                    "estimated_return": truth + 2.0,
                    "absolute_error": 2.0,
                    "bellman_calibration_error": 7.0,
                    "importance_weight_ess_fraction": 0.5,
                    "diagnostic_only": False,
                }
            )
            rows.append(
                {
                    "dataset_name": "hopper-medium-v0",
                    "method": "strict_cross_isotonic",
                    "seed": seed,
                    "policy_id": policy_id,
                    "ground_truth_return": truth,
                    "estimated_return": truth + 3.8,
                    "absolute_error": 3.8,
                    "bellman_calibration_error": 7.0,
                    "importance_weight_ess_fraction": 0.5,
                    "diagnostic_only": False,
                }
            )
    audit = _promotion_audit(rows)
    labels = {row["method"]: row for row in audit}
    assert labels["strict_cross_linear"]["audit_label"] == "promote_main"
    assert labels["strict_cross_isotonic"]["audit_label"] == "not_promoted"
    assert "ope_error_gate_failed" in labels["strict_cross_isotonic"]["failure_reasons"]


def test_hopper_stage_defaults_are_fixed_and_separated() -> None:
    args = argparse.Namespace(stage="smoke", output_dir="x")
    out = apply_stage_defaults(args)
    assert out.output_dir.endswith("hopper_calibration_smoke")
    assert out.seeds == [0]
    assert out.target_policies == ["hopper-medium_00", "hopper-medium_05", "hopper-medium_10"]
    args = argparse.Namespace(stage="pilot_all_policies", output_dir="x")
    out = apply_stage_defaults(args)
    assert out.target_policies is None
    assert out.output_dir.endswith("hopper_calibration_pilot_all_policies")
    args = argparse.Namespace(stage="expansion", output_dir="x")
    out = apply_stage_defaults(args)
    assert out.skip_missing_datasets is True
    assert "hopper-medium-v0" in out.dataset_names


def test_deep_ope_policy_value_screen_audits_calibrated_rows() -> None:
    rows = [
        {
            "family": "rlunplugged",
            "task": "toy",
            "seed": seed,
            "method": "raw_official_fqe_l2",
            "raw_policy_spearman": 0.8,
            "relative_absolute_error_vs_raw": 1.0,
        }
        for seed in range(3)
    ]
    rows.extend(
        {
            "family": "rlunplugged",
            "task": "toy",
            "seed": seed,
            "method": "linear_policy_value_calibration",
            "raw_policy_spearman": 0.8,
            "relative_absolute_error_vs_raw": 0.5,
        }
        for seed in range(3)
    )
    audit = _deep_ope_audit(_deep_ope_summarize(rows))
    labels = {row["method"]: row["audit_label"] for row in audit}
    assert labels["raw_official_fqe_l2"] == "raw_baseline"
    assert labels["linear_policy_value_calibration"] == "policy_value_calibration_benchmark"
