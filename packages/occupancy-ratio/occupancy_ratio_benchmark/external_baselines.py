from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.diagnostics import estimator_diagnostics_optional


Array = np.ndarray


@dataclass(frozen=True)
class GoogleDualDICEPreflight:
    available: bool
    reason: str
    repo_path: Path


@dataclass(frozen=True)
class GoogleDICERLPreflight:
    available: bool
    reason: str
    repo_path: Path


DICE_RL_DUALDICE_RECOVERY_FLAGS: dict[str, float | bool] = {
    "primal_regularizer": 0.0,
    "dual_regularizer": 1.0,
    "zero_reward": True,
    "norm_regularizer": 0.0,
    "zeta_pos": False,
}
DICE_RL_BEST_REGULARIZED_FLAGS: dict[str, float | bool] = {
    "primal_regularizer": 0.0,
    "dual_regularizer": 1.0,
    "zero_reward": False,
    "norm_regularizer": 1.0,
    "zeta_pos": True,
}


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


def preflight_google_dice_rl(repo_path: str | Path) -> GoogleDICERLPreflight:
    """Check whether the Google Research DICE-RL adapter can run."""
    path = Path(repo_path)
    package_dir = _dice_rl_package_dir(path)
    if package_dir is None:
        return GoogleDICERLPreflight(
            available=False,
            reason=(
                f"Missing DICE-RL source under {path}. Clone "
                "https://github.com/google-research/dice_rl to an importable "
                "directory such as /tmp/dice_rl and pass --dice-rl-repo-path."
            ),
            repo_path=path,
        )
    if package_dir.name != "dice_rl":
        return GoogleDICERLPreflight(
            available=False,
            reason=(
                f"DICE-RL package directory must be named 'dice_rl' for its absolute imports; got {package_dir}."
            ),
            repo_path=path,
        )
    try:
        _ensure_dice_rl_importable(path)
        import tensorflow  # noqa: F401
        import tf_agents  # noqa: F401
        from dice_rl.estimators.neural_dice import NeuralDice  # noqa: F401
        from dice_rl.networks.value_network import ValueNetwork  # noqa: F401
        from dice_rl.data import dataset as dice_dataset  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return GoogleDICERLPreflight(
            available=False,
            reason=f"Google DICE-RL import failed: {type(exc).__name__}: {exc}",
            repo_path=path,
        )
    return GoogleDICERLPreflight(available=True, reason="", repo_path=path)


def _dice_rl_package_dir(repo_path: Path) -> Path | None:
    if (repo_path / "__init__.py").exists() and (repo_path / "estimators" / "neural_dice.py").exists():
        return repo_path
    nested = repo_path / "dice_rl"
    if (nested / "__init__.py").exists() and (nested / "estimators" / "neural_dice.py").exists():
        return nested
    return None


def _ensure_dice_rl_importable(repo_path: str | Path) -> None:
    package_dir = _dice_rl_package_dir(Path(repo_path))
    if package_dir is None:
        raise FileNotFoundError(f"Missing DICE-RL package under {repo_path}.")
    if package_dir.name != "dice_rl":
        raise ImportError(f"DICE-RL package directory must be named 'dice_rl', got {package_dir}.")
    import_root = str(package_dir.parent)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    existing = sys.modules.get("dice_rl")
    existing_file = Path(str(getattr(existing, "__file__", ""))).resolve() if existing is not None else None
    if existing_file is not None and existing_file != (package_dir / "__init__.py").resolve():
        for name in tuple(sys.modules):
            if name == "dice_rl" or name.startswith("dice_rl."):
                del sys.modules[name]
    importlib.import_module("dice_rl")


def estimate_google_dualdice_neural(
    dataset: BenchmarkDataset,
    *,
    preflight: GoogleDualDICEPreflight,
    num_updates: int,
    batch_size: int,
    weight_decay: float = 1e-5,
    nu_learning_rate: float = 1e-4,
    zeta_learning_rate: float = 1e-3,
    hidden_dims: Sequence[int] = (256, 256),
    estimator_name: str = "google_dualdice_neural",
    diagnostic_features: Array,
    value_diagnostics: dict[str, float],
) -> dict[str, Any]:
    """Run the official Google Research neural DualDICE implementation."""
    if not preflight.available:
        return dict(
            estimator=estimator_name,
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
    hidden_dims_tuple = tuple(int(width) for width in hidden_dims)
    use_official_default = (
        hidden_dims_tuple == (256, 256)
        and abs(float(nu_learning_rate) - 1e-4) <= 1e-15
        and abs(float(zeta_learning_rate) - 1e-3) <= 1e-15
    )
    if use_official_default:
        model = DualDICE(dataset.state_dim, dataset.action_dim, weight_decay=float(weight_decay))
    else:
        model = _make_tunable_dualdice(
            tf,
            dataset.state_dim,
            dataset.action_dim,
            hidden_dims=hidden_dims_tuple,
            weight_decay=float(weight_decay),
            nu_learning_rate=float(nu_learning_rate),
            zeta_learning_rate=float(zeta_learning_rate),
        )
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
    diagnostics["google_weight_decay"] = float(weight_decay)
    diagnostics["google_nu_learning_rate"] = float(nu_learning_rate)
    diagnostics["google_zeta_learning_rate"] = float(zeta_learning_rate)
    diagnostics["google_hidden_dims"] = "x".join(str(width) for width in hidden_dims_tuple)
    diagnostics["google_official_default_architecture"] = float(use_official_default)
    return dict(
        estimator=estimator_name,
        status="ok",
        weights=clipped,
        raw_weights=raw,
        runtime_sec=time.perf_counter() - start,
        diagnostics=diagnostics,
        skip_reason="",
    )


def estimate_google_dice_rl_neural(
    dataset: BenchmarkDataset,
    *,
    preflight: GoogleDICERLPreflight,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: Sequence[int],
    flags: dict[str, float | bool],
    estimator_name: str,
    diagnostic_features: Array,
    value_diagnostics: dict[str, float],
) -> dict[str, Any]:
    """Run Google Research DICE-RL's neural DICE objective on a benchmark dataset."""
    if not preflight.available:
        return dict(
            estimator=estimator_name,
            status="skipped",
            weights=None,
            raw_weights=None,
            runtime_sec=0.0,
            diagnostics={},
            skip_reason=preflight.reason,
        )

    start = time.perf_counter()
    _ensure_dice_rl_importable(preflight.repo_path)
    import tensorflow as tf  # noqa: PLC0415
    from tf_agents.specs import tensor_spec  # noqa: PLC0415
    from tf_agents.trajectories import policy_step  # noqa: PLC0415
    from dice_rl.data import dataset as dice_dataset  # noqa: PLC0415
    from dice_rl.estimators.neural_dice import NeuralDice  # noqa: PLC0415
    from dice_rl.networks.value_network import ValueNetwork  # noqa: PLC0415

    np.random.seed(int(dataset.seed))
    tf.random.set_seed(int(dataset.seed))
    hidden_dims_tuple = tuple(int(width) for width in hidden_dims)
    state_dim = int(dataset.state_dim)
    action_dim = int(dataset.action_dim)

    step_spec = dice_dataset.EnvStep(
        step_type=tensor_spec.TensorSpec((), tf.int32),
        step_num=tensor_spec.TensorSpec((), tf.int32),
        observation=tensor_spec.TensorSpec((state_dim,), tf.float32),
        action=tensor_spec.TensorSpec((action_dim,), tf.float32),
        reward=tensor_spec.TensorSpec((), tf.float32),
        discount=tensor_spec.TensorSpec((), tf.float32),
        policy_info={},
        env_info={},
        other_info={},
    )
    input_spec = (step_spec.observation, step_spec.action)
    activation_fn = tf.nn.relu
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=int(dataset.seed))
    last_kernel_initializer = tf.keras.initializers.GlorotUniform(seed=int(dataset.seed + 1_003))
    zeta_activation = tf.math.square if bool(flags["zeta_pos"]) else tf.identity
    nu_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims_tuple,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=last_kernel_initializer,
    )
    zeta_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims_tuple,
        activation_fn=activation_fn,
        output_activation_fn=zeta_activation,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(dataset.seed + 2_003)),
        last_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(dataset.seed + 3_003)),
    )
    optimizer_kwargs = {"learning_rate": float(learning_rate), "clipvalue": 1.0}
    estimator = NeuralDice(
        step_spec,
        nu_network,
        zeta_network,
        tf.keras.optimizers.Adam(**optimizer_kwargs),
        tf.keras.optimizers.Adam(**optimizer_kwargs),
        tf.keras.optimizers.Adam(**optimizer_kwargs),
        float(dataset.gamma),
        zero_reward=bool(flags["zero_reward"]),
        f_exponent=2.0,
        primal_regularizer=float(flags["primal_regularizer"]),
        dual_regularizer=float(flags["dual_regularizer"]),
        norm_regularizer=float(flags["norm_regularizer"]),
    )

    states_np = _as_float32_2d(dataset.states)
    actions_np = _as_float32_2d(dataset.actions)
    next_states_np = _as_float32_2d(dataset.next_states)
    next_target_actions_np = _as_float32_2d(dataset.next_target_actions)
    initial_states_np = _as_float32_2d(dataset.initial_states)
    initial_actions_np = _as_float32_2d(dataset.initial_actions)
    target_actions_np = _as_float32_2d(dataset.target_actions)
    rewards_np = np.asarray(dataset.rewards, dtype=np.float32).reshape(-1)
    masks_np = np.asarray(dataset.masks, dtype=np.float32).reshape(-1)

    target_policy = _NearestActionPolicy(
        tf=tf,
        policy_step=policy_step,
        states=np.vstack([states_np, next_states_np, initial_states_np]),
        actions=np.vstack([target_actions_np, next_target_actions_np, initial_actions_np]),
    )

    states = tf.convert_to_tensor(states_np, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions_np, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states_np, dtype=tf.float32)
    next_target_actions = tf.convert_to_tensor(next_target_actions_np, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
    masks = tf.convert_to_tensor(masks_np, dtype=tf.float32)
    initial_states = tf.convert_to_tensor(initial_states_np, dtype=tf.float32)
    initial_actions = tf.convert_to_tensor(initial_actions_np, dtype=tf.float32)

    rng = np.random.default_rng(dataset.seed + 77_003)
    actual_batch_size = min(int(batch_size), dataset.n)
    initial_batch_size = min(actual_batch_size, int(initial_states_np.shape[0]))
    losses: list[tuple[float, float, float]] = []
    for step in range(int(num_steps)):
        idx = rng.integers(0, dataset.n, size=actual_batch_size)
        init_idx = rng.integers(0, int(initial_states_np.shape[0]), size=initial_batch_size)
        experience = _dice_rl_experience_batch(
            dice_dataset,
            tf,
            states=tf.gather(states, idx),
            actions=tf.gather(actions, idx),
            next_states=tf.gather(next_states, idx),
            next_actions=tf.gather(next_target_actions, idx),
            rewards=tf.gather(rewards, idx),
            masks=tf.gather(masks, idx),
        )
        initial_step = _dice_rl_initial_batch(
            dice_dataset,
            tf,
            states=tf.gather(initial_states, init_idx),
            actions=tf.gather(initial_actions, init_idx),
        )
        loss_tuple = estimator.train_step(initial_step, experience, target_policy)
        if step == 0 or step == int(num_steps) - 1 or step % 250 == 0:
            losses.append(tuple(float(value.numpy()) for value in loss_tuple))

    raw = zeta_network((states, actions))[0].numpy().astype(np.float64)
    clipped = np.maximum(raw, 0.0)
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=clipped,
        raw_ratio=raw,
        reference_weights=dataset.reference_weights,
        feature_matrix=diagnostic_features,
    )
    diagnostics.update(value_diagnostics)
    if losses:
        diagnostics["dice_rl_final_nu_loss"] = float(losses[-1][0])
        diagnostics["dice_rl_final_zeta_loss"] = float(losses[-1][1])
        diagnostics["dice_rl_final_lam_loss"] = float(losses[-1][2])
    diagnostics["dice_rl_num_steps"] = float(num_steps)
    diagnostics["dice_rl_batch_size"] = float(actual_batch_size)
    diagnostics["dice_rl_learning_rate"] = float(learning_rate)
    diagnostics["dice_rl_hidden_dims"] = "x".join(str(width) for width in hidden_dims_tuple)
    for key, value in flags.items():
        diagnostics[f"dice_rl_{key}"] = float(value)
    diagnostics["dice_rl_exact_dualdice_recovery"] = float(_dice_rl_flags_match(flags, DICE_RL_DUALDICE_RECOVERY_FLAGS))
    diagnostics["dice_rl_best_regularized_form"] = float(_dice_rl_flags_match(flags, DICE_RL_BEST_REGULARIZED_FLAGS))
    return dict(
        estimator=estimator_name,
        status="ok",
        weights=clipped,
        raw_weights=raw,
        runtime_sec=time.perf_counter() - start,
        diagnostics=diagnostics,
        skip_reason="",
    )


def _dice_rl_flags_match(flags: dict[str, float | bool], target: dict[str, float | bool]) -> bool:
    return all(flags.get(key) == value for key, value in target.items())


def _as_float32_2d(values: Array) -> Array:
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(arr.shape[0], -1)


class _NearestActionPolicy:
    def __init__(self, *, tf, policy_step, states: Array, actions: Array) -> None:
        self._tf = tf
        self._policy_step = policy_step
        self._states = tf.convert_to_tensor(_as_float32_2d(states), dtype=tf.float32)
        self._actions = tf.convert_to_tensor(_as_float32_2d(actions), dtype=tf.float32)

    def action(self, time_step):  # noqa: ANN001
        obs = self._tf.reshape(self._tf.cast(time_step.observation, self._tf.float32), [self._tf.shape(time_step.observation)[0], -1])
        squared_dist = self._tf.reduce_sum(self._tf.square(obs[:, None, :] - self._states[None, :, :]), axis=-1)
        idx = self._tf.argmin(squared_dist, axis=1, output_type=self._tf.int32)
        return self._policy_step.PolicyStep(action=self._tf.gather(self._actions, idx), state=(), info=())


def _dice_rl_experience_batch(
    dice_dataset,
    tf,
    *,
    states,
    actions,
    next_states,
    next_actions,
    rewards,
    masks,
):
    batch_size = tf.shape(states)[0]
    step_type = tf.zeros((batch_size, 2), dtype=tf.int32)
    step_num = tf.zeros((batch_size, 2), dtype=tf.int32)
    return dice_dataset.EnvStep(
        step_type=step_type,
        step_num=step_num,
        observation=tf.stack([states, next_states], axis=1),
        action=tf.stack([actions, next_actions], axis=1),
        reward=tf.stack([rewards, tf.zeros_like(rewards)], axis=1),
        discount=tf.stack([tf.ones_like(masks), masks], axis=1),
        policy_info={},
        env_info={},
        other_info={},
    )


def _dice_rl_initial_batch(dice_dataset, tf, *, states, actions):
    batch_size = tf.shape(states)[0]
    return dice_dataset.EnvStep(
        step_type=tf.zeros((batch_size,), dtype=tf.int32),
        step_num=tf.zeros((batch_size,), dtype=tf.int32),
        observation=states,
        action=actions,
        reward=tf.zeros((batch_size,), dtype=tf.float32),
        discount=tf.ones((batch_size,), dtype=tf.float32),
        policy_info={},
        env_info={},
        other_info={},
    )


def _make_tunable_dualdice(
    tf,
    state_dim: int,
    action_dim: int,
    *,
    hidden_dims: Sequence[int],
    weight_decay: float,
    nu_learning_rate: float,
    zeta_learning_rate: float,
):
    from tensorflow_addons import optimizers as tfa_optimizers  # noqa: PLC0415

    class _CriticNet(tf.keras.Model):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = [tf.keras.layers.Dense(int(width), activation="relu") for width in hidden_dims]
            self.out = tf.keras.layers.Dense(1)

        def call(self, states, actions):  # noqa: ANN001
            x = tf.concat([states, actions], axis=-1)
            for layer in self.hidden:
                x = layer(x)
            return tf.squeeze(self.out(x), axis=1)

    class _TunableDualDICE:
        def __init__(self) -> None:
            self.nu = _CriticNet()
            self.zeta = _CriticNet()
            self.nu_optimizer = tfa_optimizers.AdamW(
                learning_rate=float(nu_learning_rate),
                beta_1=0.0,
                beta_2=0.99,
                weight_decay=float(weight_decay),
            )
            self.zeta_optimizer = tfa_optimizers.AdamW(
                learning_rate=float(zeta_learning_rate),
                beta_1=0.0,
                beta_2=0.99,
                weight_decay=float(weight_decay),
            )

        def update(
            self,
            initial_states,
            initial_actions,
            initial_weights,
            states,
            actions,
            next_states,
            next_actions,
            masks,
            weights,
            discount,
        ):
            with tf.GradientTape(persistent=True) as tape:
                nu = self.nu(states, actions)
                nu_next = self.nu(next_states, next_actions)
                nu_0 = self.nu(initial_states, initial_actions)
                zeta = self.zeta(states, actions)
                nu_loss = (
                    tf.reduce_sum(weights * ((nu - discount * masks * nu_next) * zeta - tf.square(zeta) / 2.0))
                    / tf.reduce_sum(weights)
                    - tf.reduce_sum(initial_weights * (1.0 - discount) * nu_0) / tf.reduce_sum(initial_weights)
                )
                zeta_loss = -nu_loss
            self.nu_optimizer.apply_gradients(zip(tape.gradient(nu_loss, self.nu.trainable_variables), self.nu.trainable_variables))
            self.zeta_optimizer.apply_gradients(zip(tape.gradient(zeta_loss, self.zeta.trainable_variables), self.zeta.trainable_variables))
            del tape
            return nu_loss

    _ = state_dim, action_dim
    return _TunableDualDICE()


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
