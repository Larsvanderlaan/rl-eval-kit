from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from occupancy_ratio.google_dualdice import (
    GoogleDualDICEConfig,
    fit_google_dualdice_occupancy_ratio,
    preflight_google_dualdice,
)
import occupancy_ratio.google_dualdice as google_dualdice


def test_google_dualdice_config_validation() -> None:
    with pytest.raises(ValueError, match="num_updates"):
        GoogleDualDICEConfig(num_updates=0)
    with pytest.raises(ValueError, match="batch_size"):
        GoogleDualDICEConfig(batch_size=0)


def test_google_dualdice_preflight_missing_source(tmp_path) -> None:
    preflight = preflight_google_dualdice(tmp_path / "missing-google-research")
    assert preflight.available is False
    assert "DualDICE source" in preflight.reason


def test_google_dualdice_wrapper_matches_public_ratio_api(monkeypatch, tmp_path) -> None:
    google_root = tmp_path / "google-research"
    policy_eval = google_root / "policy_eval"
    policy_eval.mkdir(parents=True)
    (policy_eval / "dual_dice.py").write_text("# fake", encoding="utf-8")
    updates = []

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float64)

        def numpy(self):
            return self.value

    class FakeRandom:
        @staticmethod
        def set_seed(seed):
            return None

    class FakeThreading:
        @staticmethod
        def set_intra_op_parallelism_threads(num_threads):
            return None

        @staticmethod
        def set_inter_op_parallelism_threads(num_threads):
            return None

    class FakeConfig:
        threading = FakeThreading()

    class FakeTF:
        float32 = np.float32
        random = FakeRandom()
        config = FakeConfig()

        @staticmethod
        def convert_to_tensor(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def gather(value, idx):
            return np.asarray(value)[idx]

    class FakeDualDICE:
        def __init__(self, state_dim, action_dim, weight_decay):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.weight_decay = weight_decay

        def update(self, *args):
            updates.append(args)
            return FakeTensor(0.25 + 0.01 * len(updates))

        def zeta(self, states, actions):
            n = np.asarray(states).shape[0]
            return FakeTensor(np.linspace(-0.5, 2.0, n))

    fake_policy_eval = types.ModuleType("policy_eval")
    fake_dual_dice = types.ModuleType("policy_eval.dual_dice")
    fake_dual_dice.DualDICE = FakeDualDICE
    monkeypatch.setitem(sys.modules, "policy_eval", fake_policy_eval)
    monkeypatch.setitem(sys.modules, "policy_eval.dual_dice", fake_dual_dice)
    monkeypatch.setattr(google_dualdice, "_load_tensorflow_for_google_dualdice", lambda: FakeTF)

    n = 6
    states = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    actions = np.linspace(-1.0, 1.0, n).reshape(n, 1)
    model = fit_google_dualdice_occupancy_ratio(
        states=states,
        actions=actions,
        next_states=states + 0.1,
        target_actions=actions * 0.5,
        target_next_actions=actions * 0.25,
        gamma=0.9,
        initial_states=states[:3],
        initial_actions=actions[:3],
        terminals=np.zeros(n),
        sample_weight=np.ones(n),
        initial_weights=np.ones(3),
        config=GoogleDualDICEConfig(
            google_research_path=google_root,
            num_updates=3,
            batch_size=4,
            seed=7,
            prediction_max=1.5,
        ),
    )

    assert len(updates) == 3
    assert model.diagnostics["backend"] == "google_dualdice"
    assert model.diagnostics["num_updates"] == 3.0
    assert model.diagnostics["weight_max"] <= 1.5
    weights = model.predict_state_action_ratio(states, actions)
    assert np.all(weights >= 0.0)
    assert np.max(weights) <= 1.5
    assert np.allclose(model.predict_action_ratio(states, actions), np.ones(n))
    rows = model.predict_for_target_actions(states, actions * 0.5, observed_actions=actions)
    assert rows["target_state_action_ratio"].shape == (n,)
    assert rows["observed_state_action_ratio"].shape == (n,)
