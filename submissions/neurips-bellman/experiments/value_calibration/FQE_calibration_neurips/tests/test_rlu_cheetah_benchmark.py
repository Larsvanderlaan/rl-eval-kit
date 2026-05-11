from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from FQE_calibration_neurips.scripts.run_rlu_cheetah_benchmark import (
    SyntheticLinearPolicy,
    evaluate_prediction,
    fit_linear_or_rf_fqe,
    fit_neural_fqe,
    fit_rkhs_minimax_fqe,
    make_synthetic_batch,
    run_benchmark,
    split_by_episode,
)
from FQE_calibration_neurips.scripts.run_rlu_cheetah_policy_value_benchmark import run as run_policy_value_benchmark
from FQE_calibration_neurips.scripts.run_rlu_cheetah_fqe_reproduction import (
    Normalizer,
    _build_official_transitions_from_episodes,
    validate_official_rlds_batch,
)


CAL_ROOT = Path(__file__).resolve().parents[1]
HOPPER_BENCHMARK_DIR = (
    CAL_ROOT.parent.parent
    / "hopper_fqe_benchmark"
    / "hopper_fqe_benchmark"
    / "artifacts"
    / "benchmark"
    / "dope"
)


def test_episode_splits_are_disjoint() -> None:
    batch = make_synthetic_batch(n_episodes=10, horizon=5, seed=1)
    train, cal, diag = split_by_episode(batch, seed=7)
    sets = [set(x.episode_ids.tolist()) for x in (train, cal, diag)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert len(train.rewards) > 0
    assert len(cal.rewards) > 0
    assert len(diag.rewards) > 0


def test_continuous_learners_return_finite_diagnostics() -> None:
    batch = make_synthetic_batch(n_episodes=8, horizon=6, seed=2)
    train, _, diag = split_by_episode(batch, seed=3)
    policy = SyntheticLinearPolicy(action_dim=batch.actions.shape[1], seed=4)
    learners = [
        fit_linear_or_rf_fqe(train, policy, "linear_fqe", gamma=0.95, ridge=1e-3, n_iters=3, n_components=0, action_samples=1, seed=5),
        fit_linear_or_rf_fqe(train, policy, "rf_fqe", gamma=0.95, ridge=1e-3, n_iters=3, n_components=12, action_samples=1, seed=6),
        fit_neural_fqe(train, policy, gamma=0.95, n_outer_iters=1, epochs_per_iter=1, action_samples=1, seed=7, hidden_dims=(8,), batch_size=16),
        fit_rkhs_minimax_fqe(train, policy, gamma=0.95, ridge=1e-4, n_components=12, n_iters=3, action_samples=1, seed=8),
    ]
    for learner in learners:
        metrics = evaluate_prediction(learner, None, diag, policy, gamma=0.95, seed=9, action_samples=1)
        assert np.isfinite(metrics["bellman_calibration_error"])
        assert np.isfinite(metrics["bellman_outcome_mse"])


def test_synthetic_rlu_benchmark_writes_rows_and_provenance(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        cache_path=str(tmp_path / "cache.npz"),
        tfds_data_dir=str(tmp_path / "tfds"),
        synthetic=True,
        seed=0,
        seeds=[0],
        task="cheetah_run",
        stage="smoke",
        benchmark_dir=str(HOPPER_BENCHMARK_DIR),
        policy_root=None,
        policy_cache_dir=None,
        max_cache_episodes=None,
        max_transitions_per_split=None,
        policy_indices=None,
        include_time_feature=False,
        episode_horizon=20,
        gamma=0.95,
        learners=["linear_fqe"],
        fqe_iters=3,
        linear_solver="iterated",
        rf_components=12,
        rkhs_critic_anchors=12,
        rkhs_critic_bandwidth_scale=1.0,
        neural_iters=1,
        neural_epochs=1,
        neural_num_updates=None,
        neural_target_tau=1.0,
        neural_scaled_outputs=True,
        action_samples=1,
        device="cpu",
    )
    outputs = run_benchmark(args)
    raw = pd.read_csv(outputs["raw"])
    assert {"none", "linear", "isotonic"}.issubset(set(raw["calibrator"]))
    calibrated = raw[raw["calibrator"].ne("none")]
    assert calibrated["calibration_data_provenance"].str.contains("calibration").all()
    assert raw["evaluation_data_provenance"].str.contains("diagnostic").all()
    assert outputs["summary"].exists()
    assert outputs["audit"].exists()


def test_policy_value_benchmark_promotes_heldout_calibration(tmp_path: Path) -> None:
    args = argparse.Namespace(
        benchmark_dir=str(HOPPER_BENCHMARK_DIR),
        output_dir=str(tmp_path / "policy_value"),
        task="cheetah_run",
        policy_indices=list(range(8)),
        splits=["early_to_late"],
    )
    outputs = run_policy_value_benchmark(args)
    summary = pd.read_csv(outputs["summary"])
    audit = pd.read_csv(outputs["audit"])
    assert {"none", "linear", "isotonic"}.issubset(set(summary["method"]))
    promoted = audit[audit["audit_label"].eq("promote_main")]
    assert not promoted.empty
    assert promoted["relative_absolute_ope_error"].lt(0.90).all()


def _fake_rlds_step(t: int, *, reward: float, is_first: bool = False, is_last: bool = False) -> dict[str, object]:
    obs = {
        "position": np.full(8, float(t), dtype=np.float32),
        "velocity": np.full(9, float(t) + 0.5, dtype=np.float32),
    }
    return {
        "action": np.full(6, float(t) / 10.0, dtype=np.float32),
        "discount": np.float32(1.0),
        "is_first": bool(is_first),
        "is_last": bool(is_last),
        "is_terminal": False,
        "observation": obs,
        "reward": np.float32(reward),
    }


def test_official_rlds_transition_builder_uses_next_reward_and_terminal_mask() -> None:
    episodes = [
        {
            "steps": [
                _fake_rlds_step(0, reward=0.1, is_first=True),
                _fake_rlds_step(1, reward=1.0),
                _fake_rlds_step(2, reward=2.0),
                _fake_rlds_step(3, reward=0.0, is_last=True),
            ]
        }
    ]
    batch = _build_official_transitions_from_episodes(episodes, task="cheetah_run", max_episodes=None)
    assert batch.states.shape == (3, 17)
    assert batch.actions.shape == (3, 6)
    np.testing.assert_allclose(batch.rewards, np.array([1.0, 2.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(batch.discounts, np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert batch.next_is_last.tolist() == [False, False, True]
    diagnostics = validate_official_rlds_batch(batch, max_return_gap=0.2)
    assert diagnostics["terminal_mask_fraction"] > 0.0
    assert diagnostics["max_abs_return_sum_gap"] <= 0.2


def test_fqe_reproduction_value_scales_are_distinct_and_finite() -> None:
    normalizer = Normalizer.fit(
        states=np.array([[0.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        rewards=np.array([1.0, 3.0], dtype=np.float32),
        discounts=np.array([1.0, 1.0], dtype=np.float32),
        normalize_rewards=True,
    )
    internal_score = np.array([0.5], dtype=np.float32)
    policy_eval_score = normalizer.unnorm_policy_eval_score(internal_score)
    discounted_return = normalizer.unnorm_scaled_return(internal_score, gamma=0.9)
    assert np.isfinite(policy_eval_score).all()
    assert np.isfinite(discounted_return).all()
    np.testing.assert_allclose(discounted_return, policy_eval_score / 0.1)
