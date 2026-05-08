from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.diagnostics import estimator_diagnostics_optional


Array = np.ndarray


@dataclass(frozen=True)
class GoogleDualDICEPreflight:
    available: bool
    reason: str
    repo_path: Path


def preflight_google_dualdice(repo_path: str | Path) -> GoogleDualDICEPreflight:
    """Check whether the official Google Research DualDICE adapter can run."""
    path = Path(repo_path)
    policy_eval_dir = path / "policy_eval"
    if not (policy_eval_dir / "dual_dice.py").exists():
        return GoogleDualDICEPreflight(
            available=False,
            reason=(
                f"Missing {policy_eval_dir / 'dual_dice.py'}. Clone "
                "https://github.com/google-research/google-research and pass --external-repo-path."
            ),
            repo_path=path,
        )
    try:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        import tensorflow  # noqa: F401
        import tensorflow_addons  # noqa: F401
        from policy_eval.dual_dice import DualDICE  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return GoogleDualDICEPreflight(
            available=False,
            reason=f"Google DualDICE import failed: {type(exc).__name__}: {exc}",
            repo_path=path,
        )
    return GoogleDualDICEPreflight(available=True, reason="", repo_path=path)


def estimate_google_dualdice_neural(
    dataset: BenchmarkDataset,
    *,
    preflight: GoogleDualDICEPreflight,
    num_updates: int,
    batch_size: int,
    diagnostic_features: Array,
    value_diagnostics: dict[str, float],
) -> dict[str, Any]:
    """Run the official Google Research neural DualDICE implementation."""
    if not preflight.available:
        return dict(
            estimator="google_dualdice_neural",
            status="skipped",
            weights=None,
            raw_weights=None,
            runtime_sec=0.0,
            diagnostics={},
            skip_reason=preflight.reason,
        )

    start = time.perf_counter()
    if str(preflight.repo_path) not in sys.path:
        sys.path.insert(0, str(preflight.repo_path))
    import tensorflow as tf  # noqa: PLC0415
    from policy_eval.dual_dice import DualDICE  # noqa: PLC0415

    np.random.seed(int(dataset.seed))
    tf.random.set_seed(int(dataset.seed))
    model = DualDICE(dataset.state_dim, dataset.action_dim, weight_decay=1e-5)
    rng = np.random.default_rng(dataset.seed + 44_001)
    actual_batch_size = min(int(batch_size), dataset.n)

    states = tf.convert_to_tensor(dataset.states, dtype=tf.float32)
    actions = tf.convert_to_tensor(dataset.actions, dtype=tf.float32)
    next_states = tf.convert_to_tensor(dataset.next_states, dtype=tf.float32)
    next_actions = tf.convert_to_tensor(dataset.next_target_actions, dtype=tf.float32)
    masks = tf.convert_to_tensor(dataset.masks, dtype=tf.float32)
    weights = tf.ones(dataset.n, dtype=tf.float32)
    initial_states = tf.convert_to_tensor(dataset.initial_states, dtype=tf.float32)
    initial_actions = tf.convert_to_tensor(dataset.initial_actions, dtype=tf.float32)
    initial_weights = tf.convert_to_tensor(dataset.initial_weights, dtype=tf.float32)

    losses = []
    for step in range(int(num_updates)):
        idx = rng.integers(0, dataset.n, size=actual_batch_size)
        loss = model.update(
            initial_states,
            initial_actions,
            initial_weights,
            tf.gather(states, idx),
            tf.gather(actions, idx),
            tf.gather(next_states, idx),
            tf.gather(next_actions, idx),
            tf.gather(masks, idx),
            tf.gather(weights, idx),
            float(dataset.gamma),
        )
        if step == 0 or step == int(num_updates) - 1 or step % 250 == 0:
            losses.append(float(loss.numpy()))

    raw = model.zeta(states, actions).numpy().astype(np.float64)
    clipped = np.maximum(raw, 0.0)
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=clipped,
        raw_ratio=raw,
        reference_weights=dataset.reference_weights,
        feature_matrix=diagnostic_features,
    )
    diagnostics.update(value_diagnostics)
    diagnostics["google_final_loss"] = float(losses[-1]) if losses else np.nan
    diagnostics["google_num_updates"] = float(num_updates)
    diagnostics["google_batch_size"] = float(actual_batch_size)
    return dict(
        estimator="google_dualdice_neural",
        status="ok",
        weights=clipped,
        raw_weights=raw,
        runtime_sec=time.perf_counter() - start,
        diagnostics=diagnostics,
        skip_reason="",
    )


def preflight_google_gridwalk(repo_path: str | Path) -> GoogleDualDICEPreflight:
    """Check whether Google Research's tabular DualDICE GridWalk benchmark can run."""
    path = Path(repo_path)
    if not (path / "dual_dice" / "run.py").exists():
        return GoogleDualDICEPreflight(
            available=False,
            reason=f"Missing Google DualDICE GridWalk source at {path / 'dual_dice'}.",
            repo_path=path,
        )
    try:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        import dual_dice.algos.dual_dice  # noqa: F401
        import dual_dice.gridworld.environments  # noqa: F401
        import dual_dice.gridworld.policies  # noqa: F401
        import dual_dice.transition_data  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return GoogleDualDICEPreflight(
            available=False,
            reason=f"Google GridWalk DualDICE import failed: {type(exc).__name__}: {exc}",
            repo_path=path,
        )
    return GoogleDualDICEPreflight(available=True, reason="", repo_path=path)
