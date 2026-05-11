from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import TransitionBatch
from ..policies import SoftmaxPolicy
from .random_feature_fqe import RandomFeatureFQEConfig, fit_random_feature_fqe


@dataclass
class EnsembleFQEConfig:
    gamma: float = 0.95
    n_members: int = 5
    n_components: int = 96
    bandwidth: float = 0.7
    ridge: float = 1e-3
    n_iters: int = 30
    feature_type: str = "rbf"


class EnsembleFQEModel:
    def __init__(self, members: list[object]):
        self.members = members
        member_iters = [
            float(getattr(member, "diagnostics", {}).get("actual_bellman_iterations", float("nan")))
            for member in members
        ]
        self.diagnostics = {
            "actual_bellman_iterations": float(np.nanmean(member_iters)) if member_iters else float("nan"),
            "ensemble_members": float(len(members)),
        }

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        preds = np.column_stack([member.predict_q(states, actions) for member in self.members])
        return np.mean(preds, axis=1)

    def value(self, states: np.ndarray, policy: SoftmaxPolicy) -> np.ndarray:
        vals = np.column_stack([member.value(states, policy) for member in self.members])
        return np.mean(vals, axis=1)


def fit_ensemble_fqe(
    batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    config: EnsembleFQEConfig,
    seed: int,
) -> EnsembleFQEModel:
    rng = np.random.default_rng(seed)
    members = []
    for member_id in range(int(config.n_members)):
        idx = rng.choice(np.arange(len(batch)), size=len(batch), replace=True)
        member_cfg = RandomFeatureFQEConfig(
            gamma=config.gamma,
            n_components=config.n_components,
            bandwidth=config.bandwidth,
            ridge=config.ridge,
            n_iters=config.n_iters,
            feature_type=config.feature_type,
        )
        members.append(fit_random_feature_fqe(batch.subset(idx), n_actions, policy, member_cfg, seed + 7919 * member_id))
    return EnsembleFQEModel(members)
