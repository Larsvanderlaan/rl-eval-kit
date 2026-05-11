from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np

from causal_ope_benchmark.config import DomainScenario
from causal_ope_benchmark.exceptions import MissingOptionalDependency
from causal_ope_benchmark.policies import CLINIC_ACTIONS, STREAM_ACTIONS, get_fixed_policy
from causal_ope_benchmark.simulators import (
    _clinic_action_availability,
    _clinic_action_dose,
    _clinic_initial,
    _clinic_step,
    _stream_action_availability,
    _stream_action_dose,
    _streamretain_initial,
    _streamretain_step,
)
from causal_ope_benchmark.types import Array


try:  # Prefer Gymnasium, but keep a classic-Gym fallback for EpiCare.
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _GYM_API = "gymnasium"
except ModuleNotFoundError:  # pragma: no cover - depends on optional deps.
    try:
        import gym as _gym  # type: ignore[no-redef]
        from gym import spaces as _spaces  # type: ignore[no-redef]

        _GYM_API = "gym"
    except ModuleNotFoundError:  # pragma: no cover - exercised when neither is installed.
        _gym = None
        _spaces = None
        _GYM_API = "minimal"


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> Array:
        return np.zeros(self.shape, dtype=self.dtype)


class _FallbackDiscrete:
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def sample(self) -> int:
        return 0


class _FallbackSpaces:
    Box = _FallbackBox
    Discrete = _FallbackDiscrete


spaces = _spaces if _spaces is not None else _FallbackSpaces()
_GymBase = _gym.Env if _gym is not None else object


class _BaseNativeEnv(_GymBase):
    family: str
    action_names: tuple[str, ...]

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario: DomainScenario | None = None,
        target_policy: str = "moderate",
        seed: int = 0,
        horizon: int = 24,
    ) -> None:
        self.scenario = scenario or DomainScenario(name="gym_default")
        self.target_policy = str(target_policy)
        self.step_per_episode = int(horizon)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._state: Array | None = None
        self._latent: Array | None = None
        self._time = 0
        self._terminated = False
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.metadata = {
            "render_modes": [],
            "family": self.family,
            "gym_api": _GYM_API,
            "target_policy": self.target_policy,
            "step_per_episode": self.step_per_episode,
        }

    @property
    def state_dim(self) -> int:
        raise NotImplementedError

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        return [self._seed]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.seed(seed)
        self._time = 0
        self._terminated = False
        self._state, self._latent = self._initial_state()
        info = self._info()
        return self._state.astype(np.float32), info

    def step(self, action: int):
        if self._state is None or self._latent is None:
            self.reset()
        if self._terminated:
            raise RuntimeError("step called after episode termination; call reset first.")
        action_idx = int(action)
        availability = self._availability(self._state)
        if action_idx < 0 or action_idx >= availability.shape[0]:
            raise ValueError(f"Action index {action_idx} is outside the action space.")
        if availability[action_idx] <= 0.0:
            raise ValueError(f"Action '{self.action_names[action_idx]}' is unavailable for the current state.")
        dose = self._dose(self._state, action_idx)
        next_state, reward, terminal, censored, components = self._transition(self._state, action_idx, dose)
        self._state = next_state
        self._time += 1
        truncated = bool(censored >= 0.5 or self._time >= self.step_per_episode)
        terminated = bool(terminal >= 0.5)
        self._terminated = bool(terminated or truncated)
        info = self._info()
        info.update(
            {
                "action_dose": float(dose),
                "censoring": float(censored),
                "domain_terminal": float(terminal),
                "outcome_components": components,
            }
        )
        return self._state.astype(np.float32), float(reward), terminated, truncated, info

    def _initial_state(self) -> tuple[Array, Array]:
        raise NotImplementedError

    def _availability(self, state: Array) -> Array:
        raise NotImplementedError

    def _dose(self, state: Array, action: int) -> float:
        raise NotImplementedError

    def _transition(self, state: Array, action: int, dose: float) -> tuple[Array, float, float, float, dict[str, float]]:
        raise NotImplementedError

    def _target_probabilities(self, state: Array | None = None) -> Array:
        current_state = self._state if state is None else state
        if current_state is None:
            current_state, _ = self._initial_state()
        availability = self._availability(current_state)
        policy = get_fixed_policy(self.family, self.target_policy)  # type: ignore[arg-type]
        return policy.probabilities(
            current_state.reshape(1, -1),
            availability.reshape(1, -1),
            np.asarray([self._time], dtype=np.int64),
        )[0]

    def _info(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {}
        availability = self._availability(state)
        return {
            "time": int(self._time),
            "action_available": availability.copy(),
            "target_action_probabilities": self._target_probabilities(state),
            "action_names": self.action_names,
        }


class StreamRetainEnv(_BaseNativeEnv):
    """Gymnasium-style environment for the StreamRetain simulator."""

    family = "streamretain"
    action_names = STREAM_ACTIONS

    @property
    def state_dim(self) -> int:
        return 8

    def _initial_state(self) -> tuple[Array, Array]:
        states, latent = _streamretain_initial(1, self._rng)
        return states[0].copy(), latent[0].copy()

    def _availability(self, state: Array) -> Array:
        return _stream_action_availability(state, self.scenario)

    def _dose(self, state: Array, action: int) -> float:
        return _stream_action_dose(state, action, self.scenario, self._rng)

    def _transition(self, state: Array, action: int, dose: float) -> tuple[Array, float, float, float, dict[str, float]]:
        assert self._latent is not None
        return _streamretain_step(state, action, dose, self._latent, self._time, self.scenario, self._rng)


class ClinicDTREnv(_BaseNativeEnv):
    """Gymnasium-style environment for the ClinicDTR simulator."""

    family = "clinic_dtr"
    action_names = CLINIC_ACTIONS

    @property
    def state_dim(self) -> int:
        return 10

    def _initial_state(self) -> tuple[Array, Array]:
        states, latent = _clinic_initial(1, self._rng)
        return states[0].copy(), latent[0].copy()

    def _availability(self, state: Array) -> Array:
        return _clinic_action_availability(state, self.scenario)

    def _dose(self, state: Array, action: int) -> float:
        return _clinic_action_dose(state, action, self.scenario, self._rng)

    def _transition(self, state: Array, action: int, dose: float) -> tuple[Array, float, float, float, dict[str, float]]:
        assert self._latent is not None
        return _clinic_step(state, action, dose, self._latent, self._time, self.scenario, self._rng)


class FixedPolicyWrapper:
    """Small fixed-policy wrapper exposing action probabilities and sampling."""

    def __init__(self, family: str, name: str, *, n_actions: int | None = None, rng: np.random.Generator | None = None) -> None:
        self.family = family
        self.name = name
        self.policy = get_fixed_policy(family, name)  # type: ignore[arg-type]
        self.rng = rng or np.random.default_rng(0)
        if n_actions is not None:
            self.n_actions = int(n_actions)
        elif family == "streamretain":
            self.n_actions = len(STREAM_ACTIONS)
        elif family == "clinic_dtr":
            self.n_actions = len(CLINIC_ACTIONS)
        else:
            raise ValueError("n_actions is required for generic external policy wrappers.")

    def action_probabilities(self, observation: Array, info: dict[str, Any] | None = None, *, time: int = 0) -> Array:
        state = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        if info is not None and "action_available" in info:
            availability = np.asarray(info["action_available"], dtype=np.float64).reshape(1, -1)
        else:
            if self.family == "streamretain":
                availability = _stream_action_availability(state[0]).reshape(1, -1)
            elif self.family == "clinic_dtr":
                availability = _clinic_action_availability(state[0]).reshape(1, -1)
            else:
                availability = np.ones((1, self.n_actions), dtype=np.float64)
        return self.policy.probabilities(state, availability, np.asarray([time], dtype=np.int64))[0]

    def sample_action(self, observation: Array, info: dict[str, Any] | None = None, *, time: int = 0) -> int:
        probs = self.action_probabilities(observation, info, time=time)
        return int(self.rng.choice(probs.shape[0], p=probs))

    def calc_action_choice_probability(self, x: Array) -> Array:
        states = np.asarray(x, dtype=np.float64)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        probs = []
        for row in states:
            probs.append(self.action_probabilities(row))
        return np.vstack(probs).astype(np.float64)

    def calc_pscore_given_action(self, x: Array, action: Array) -> Array:
        probs = self.calc_action_choice_probability(x)
        action_idx = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_idx.shape[0] != probs.shape[0]:
            raise ValueError("action must have one row per state.")
        return np.clip(probs[np.arange(probs.shape[0]), action_idx], 1e-12, np.inf)

    def sample_action_and_output_pscore(self, x: Array) -> tuple[Array, Array]:
        probs = self.calc_action_choice_probability(x)
        actions = np.asarray([self.rng.choice(probs.shape[1], p=row) for row in probs], dtype=np.int64)
        return actions, self.calc_pscore_given_action(x, actions)

    def predict(self, x: Array) -> Array:
        return np.argmax(self.calc_action_choice_probability(x), axis=1).astype(np.int64)

    def predict_online(self, observation: Array) -> int:
        return self.sample_action(observation)


def make_epicare_gym_env(*, seed: int = 0, env_id: str = "EpiCare-v0"):
    """Create the external EpiCare Gym environment lazily."""

    try:
        importlib.import_module("epicare")
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependency(
            "epicare",
            "EpiCare is not installed. Install it from the EpiCare project before requesting family='epicare'.",
        ) from exc
    try:
        gym_mod = importlib.import_module("gym")
    except ModuleNotFoundError:
        try:
            gym_mod = importlib.import_module("gymnasium")
        except ModuleNotFoundError as exc:
            raise MissingOptionalDependency("gym", "EpiCare integration requires gym or gymnasium.") from exc
    env = gym_mod.make(env_id)
    _seed_external_env(env, seed)
    return env


def make_gym_env(
    family: str,
    scenario: DomainScenario | None = None,
    target_policy: str = "moderate",
    seed: int = 0,
    config: Any | None = None,
):
    """Create a Gym/Gymnasium-compatible environment for supported families."""

    horizon = int(getattr(config, "trajectory_horizon", 24) if config is not None else 24)
    scenario = scenario or DomainScenario(name="gym_default")
    if family == "streamretain":
        return StreamRetainEnv(scenario=scenario, target_policy=target_policy, seed=seed, horizon=horizon)
    if family == "clinic_dtr":
        return ClinicDTREnv(scenario=scenario, target_policy=target_policy, seed=seed, horizon=horizon)
    if family == "epicare":
        return make_epicare_gym_env(seed=seed)
    if family == "streamlift":
        raise ValueError("StreamLift is a short-panel forecasting benchmark, not a native Gym environment.")
    raise ValueError(f"Unknown Gym-compatible family '{family}'.")


def _seed_external_env(env: Any, seed: int) -> None:
    try:
        env.reset(seed=int(seed))
        return
    except TypeError:
        pass
    except Exception:
        return
    if hasattr(env, "seed"):
        env.seed(int(seed))
    try:
        env.reset()
    except Exception:
        return


def unconstrained_scenario(scenario: DomainScenario) -> DomainScenario:
    """Return a copy of a scenario with state-dependent masks disabled."""

    return replace(scenario, action_constraints=False)
