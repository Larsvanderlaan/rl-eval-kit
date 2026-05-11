from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from urllib.request import urlopen

import numpy as np


POLICY_BASE_URL = "https://storage.googleapis.com/gresearch/deep-ope/d4rl/hopper"


@dataclass(frozen=True)
class BenchmarkPolicySpec:
    policy_id: str
    task_name: str
    pickle_filename: str
    label: str

    @property
    def pickle_url(self) -> str:
        return f"{POLICY_BASE_URL}/{self.pickle_filename}"


HOPPER_MEDIUM_POLICY_SPECS: tuple[BenchmarkPolicySpec, ...] = (
    BenchmarkPolicySpec("hopper-medium_00", "hopper-medium", "hopper_online_0.pkl", "Hopper medium 00"),
    BenchmarkPolicySpec("hopper-medium_01", "hopper-medium", "hopper_online_10.pkl", "Hopper medium 01"),
    BenchmarkPolicySpec("hopper-medium_02", "hopper-medium", "hopper_online_1.pkl", "Hopper medium 02"),
    BenchmarkPolicySpec("hopper-medium_03", "hopper-medium", "hopper_online_2.pkl", "Hopper medium 03"),
    BenchmarkPolicySpec("hopper-medium_04", "hopper-medium", "hopper_online_3.pkl", "Hopper medium 04"),
    BenchmarkPolicySpec("hopper-medium_05", "hopper-medium", "hopper_online_4.pkl", "Hopper medium 05"),
    BenchmarkPolicySpec("hopper-medium_06", "hopper-medium", "hopper_online_5.pkl", "Hopper medium 06"),
    BenchmarkPolicySpec("hopper-medium_07", "hopper-medium", "hopper_online_6.pkl", "Hopper medium 07"),
    BenchmarkPolicySpec("hopper-medium_08", "hopper-medium", "hopper_online_7.pkl", "Hopper medium 08"),
    BenchmarkPolicySpec("hopper-medium_09", "hopper-medium", "hopper_online_8.pkl", "Hopper medium 09"),
    BenchmarkPolicySpec("hopper-medium_10", "hopper-medium", "hopper_online_9.pkl", "Hopper medium 10"),
)

POLICY_SPECS: dict[str, BenchmarkPolicySpec] = {spec.policy_id: spec for spec in HOPPER_MEDIUM_POLICY_SPECS}


def _download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        with destination.open("wb") as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
    return destination


class HopperPicklePolicy:
    """Numpy reimplementation of the official D4RL Hopper policy forward pass."""

    def __init__(self, weights: dict[str, np.ndarray], policy_id: str) -> None:
        self.policy_id = policy_id
        self.fc0_w = np.asarray(weights["fc0/weight"], dtype=np.float32)
        self.fc0_b = np.asarray(weights["fc0/bias"], dtype=np.float32)
        self.fc1_w = np.asarray(weights["fc1/weight"], dtype=np.float32)
        self.fc1_b = np.asarray(weights["fc1/bias"], dtype=np.float32)
        self.fclast_w = np.asarray(weights["last_fc/weight"], dtype=np.float32)
        self.fclast_b = np.asarray(weights["last_fc/bias"], dtype=np.float32)
        self.fclast_w_logstd = np.asarray(weights["last_fc_log_std/weight"], dtype=np.float32)
        self.fclast_b_logstd = np.asarray(weights["last_fc_log_std/bias"], dtype=np.float32)
        nonlinearity = str(weights["nonlinearity"])
        output_distribution = str(weights["output_distribution"])
        self.activation = np.tanh if nonlinearity == "tanh" else lambda x: np.maximum(x, 0.0)
        self.output_distribution = output_distribution
        self.action_dim = int(self.fclast_b.shape[0])

    def _forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(observations, dtype=np.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[None, :]
        x = obs @ self.fc0_w.T + self.fc0_b
        x = self.activation(x)
        x = x @ self.fc1_w.T + self.fc1_b
        x = self.activation(x)
        mean = x @ self.fclast_w.T + self.fclast_b
        log_std = x @ self.fclast_w_logstd.T + self.fclast_b_logstd
        if squeeze:
            mean = mean[0]
            log_std = log_std[0]
        return mean.astype(np.float32), log_std.astype(np.float32)

    def sample_actions(
        self,
        observations: np.ndarray,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> np.ndarray:
        mean, log_std = self._forward(observations)
        # Match the official TF-Agents D4RL actor: exp(clip(log_std, -5, 2)).
        log_std = np.clip(log_std, -5.0, 2.0)
        if deterministic:
            action = mean
        else:
            noise = rng.standard_normal(size=np.shape(mean)).astype(np.float32)
            action = mean + np.exp(log_std) * noise
        if self.output_distribution == "tanh_gaussian":
            action = np.tanh(action)
        return np.asarray(action, dtype=np.float32)

    def mean_actions(self, observations: np.ndarray) -> np.ndarray:
        return self.sample_actions(observations, rng=np.random.default_rng(0), deterministic=True)


def ensure_policy(policy_id: str, artifact_dir: str | Path) -> Path:
    if policy_id not in POLICY_SPECS:
        raise KeyError(f"Unknown policy_id '{policy_id}'. Available: {sorted(POLICY_SPECS)}")
    spec = POLICY_SPECS[policy_id]
    artifact_dir = Path(artifact_dir) / "pkl_policies"
    path = artifact_dir / spec.pickle_filename
    if not path.exists():
        _download_file(spec.pickle_url, path)
    return path


def load_policy(policy_id: str, artifact_dir: str | Path) -> HopperPicklePolicy:
    path = ensure_policy(policy_id, artifact_dir)
    with path.open("rb") as handle:
        weights = pickle.load(handle)
    return HopperPicklePolicy(weights=weights, policy_id=policy_id)
